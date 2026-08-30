#!/usr/bin/env node
/**
 * context-medic — find what compaction dropped, when you ask.
 *
 * The protocol is the whole contract: read one Event as JSON per line on
 * stdin, write one per line on stdout. `id` and `timestamp` are filled in for
 * you, so a frame is just `{ kind, payload }`.
 *
 * Two things this deliberately does not do:
 *
 *   It does not run on its own. Reviewing costs a model call, and deciding
 *   what re-enters the context is not a decision a plugin should make behind
 *   your back. It waits for `agent.context.audit.requested`, which Studio
 *   publishes when you press Review.
 *
 *   It does not hold an API key. It asks the host to run the completion with
 *   the session's own provider, so it uses the model you configured, shares
 *   the retry and rate-limit wrappers, and its spend lands on the event log
 *   where the token budget counts it.
 */

import { createInterface } from "node:readline";
import { randomUUID } from "node:crypto";

/** Newest assembled context: our picture of what the model could see. */
let assembled = null;
/** Messages folded away by compaction, oldest first, since the session began. */
let dropped = [];
/** How many messages the summary covered before the last compaction. */
let covered = 0;
/** The summary currently in force. */
let summary = null;

/** In-flight completions, keyed by request id. */
const waiting = new Map();

const log = (...a) => console.error("[context-medic]", ...a);
const send = (kind, payload) =>
  process.stdout.write(JSON.stringify({ kind, payload }) + "\n");

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });

for await (const line of rl) {
  if (!line.trim()) continue;

  let event;
  try {
    event = JSON.parse(line);
  } catch {
    continue;
  }

  switch (event.kind) {
    case "agent.context.assembled":
      assembled = event.payload;
      break;

    case "agent.context.summarized":
      remember(event.payload);
      break;

    case "agent.context.audit.requested":
      review(event.payload?.request_id).catch((e) => {
        log("review failed:", e.message);
        send("agent.context.audit.result", {
          request_id: event.payload?.request_id,
          items: [],
          error: e.message,
        });
      });
      break;

    case "llm.response": {
      const pending = waiting.get(event.payload?.request_id);
      if (pending) {
        waiting.delete(event.payload.request_id);
        pending(event.payload);
      }
      break;
    }
  }
}

/**
 * Record what a compaction pass folded away.
 *
 * Compaction is incremental: each pass folds the messages between the previous
 * `summarized_count` and the new one and leaves the rest verbatim. Taking the
 * whole verbatim tail would collect messages the model can still see, and
 * later offer to "restore" text that is already in the context.
 *
 * Cheap and local — no model call happens here, which is what makes it safe to
 * do on every compaction.
 */
function remember(payload) {
  summary = payload;
  const to = payload?.summarized_count ?? 0;
  if (!assembled || to <= covered) {
    covered = Math.max(covered, to);
    return;
  }

  const conversation = (assembled.manifest ?? [])
    // verbatim — real conversation the model could see a moment ago
    // cleared  — tool output the clearing pass blanked to reclaim budget; the
    //            full text is still in the event log, and the summary editor
    //            has no way to bring it back
    .filter((m) => m.source === "verbatim" || m.source === "cleared")
    .sort((a, b) => a.index - b.index);

  dropped = dropped.concat(conversation.slice(0, to - covered));
  covered = to;
  log(`${dropped.length} messages now behind the summary`);
}

/** Answer a review request with a list of candidates for a person to choose from. */
async function review(requestId) {
  if (!summary || dropped.length === 0) {
    send("agent.context.audit.result", { request_id: requestId, items: [] });
    return;
  }

  const original = dropped
    .map((m) => `--- ${m.role} (${m.source})\n${m.text ?? ""}`)
    .join("\n\n");

  const answer = await complete({
    messages: [
      {
        role: "user",
        content: `Below is part of a conversation, followed by the summary that replaced it.

List concrete facts present in the ORIGINAL that the SUMMARY omits or blurs.
Only these categories, and only what is genuinely missing:

- exact file paths, identifiers, and signatures
- decisions, together with the reason for them
- constraints or preferences the user stated
- approaches that were tried and failed, and why
- verbatim error messages
- versions, commands, and configuration values
- work that was started and not finished

Each item must stand alone: someone reading it without the original should
understand it. Return an empty list if nothing important is missing.

<original>
${original}
</original>

<summary>
${summary.summary}
</summary>`,
      },
    ],
    schema_name: "dropped_details",
    schema: {
      type: "object",
      properties: {
        items: {
          type: "array",
          items: {
            type: "object",
            properties: {
              id: { type: "string", description: "short slug, unique in this list" },
              fact: { type: "string", description: "the missing detail, stated in full" },
              why: { type: "string", description: "one short clause on why it matters" },
            },
            required: ["id", "fact"],
          },
        },
      },
      required: ["items"],
    },
  });

  const items = Array.isArray(answer?.items) ? answer.items : [];
  send("agent.context.audit.result", { request_id: requestId, items });
  log(`offered ${items.length} candidates`);
}

/** Ask the host to run a completion with the session's own provider. */
function complete({ messages, schema, schema_name }) {
  const requestId = randomUUID();

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      waiting.delete(requestId);
      reject(new Error("the host did not answer within 120s"));
    }, 120_000);

    waiting.set(requestId, (payload) => {
      clearTimeout(timer);
      if (payload.error) reject(new Error(payload.error));
      else resolve(payload.structured ?? payload.content);
    });

    send("llm.request", { request_id: requestId, messages, schema, schema_name });
  });
}
