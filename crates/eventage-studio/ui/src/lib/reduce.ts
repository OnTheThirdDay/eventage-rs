/**
 * Derive everything the app shows from the event stream.
 *
 * This is the same idea the harness uses on its own event log: keep one
 * append-only record and re-derive every view from it, rather than mutating a
 * view and hoping it stays consistent. The transcript and the trace panel are
 * two projections of the identical input, so they cannot drift apart, and a
 * reconnect that replays events lands on exactly the state it left.
 *
 * A pure function of the event list, so it is testable without a browser.
 */

import type {
  AssistantItem,
  ChatItem,
  ChatState,
  FileDiff,
  PermissionItem,
  PlanEntry,
  SourceLocation,
  StudioEvent,
  ToolItem,
  Workstream,
} from "./types";

const K = {
  userMessage: "user.message",
  assistantDelta: "assistant.delta",
  assistantMessage: "assistant.message",
  toolProposed: "tool.call.proposed",
  toolProgress: "tool.call.progress",
  toolResult: "tool.result",
  permissionRequest: "permission.request",
  permissionDecision: "permission.decision",
  cycleStart: "agent.cycle.start",
  cycleEnd: "agent.cycle.end",
  stuck: "agent.stuck",
  summarized: "agent.context.summarized",
  budgetWarning: "budget.warning",
  budgetExhausted: "budget.exhausted",
  rollback: "system.rollback",
  pruned: "system.pruned",
  epitaph: "system.epitaph",
  plugins: "system.plugins",
  recovered: "system.recovered",
  systemMessage: "system.message",
  elicitation: "mcp.elicitation.request",
  turnEnded: "studio.turn.ended",
  turnFailed: "studio.turn.failed",
  turnInterrupted: "studio.turn.interrupted",
  modeChanged: "studio.mode.changed",
  rewound: "studio.rewound",
  backendLost: "studio.backend.lost",
  fsRefused: "studio.fs.refused",
  goalSet: "cowork.goal.set",
  planProposed: "cowork.plan.proposed",
  wsStarted: "cowork.workstream.started",
  wsFinished: "cowork.workstream.finished",
  wsSealed: "cowork.workstream.sealed",
  adopted: "cowork.adopted",
  adoptionBlocked: "cowork.adoption.blocked",
  notTracked: "cowork.not_tracked",
  treeRestored: "session.working_tree_restored",
  treeUnchanged: "session.working_tree_unchanged",
  plan: "studio.plan",
} as const;

const str = (v: unknown): string => (typeof v === "string" ? v : "");
const num = (v: unknown): number => (typeof v === "number" ? v : 0);

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : undefined;
}

/** Tool arguments arrive as a JSON string from the model, or as an object. */
function parseArgs(raw: unknown): unknown {
  if (typeof raw !== "string") return raw ?? null;
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

/** The one-line title for a tool card: the tool plus its most telling argument. */
export function describeCall(name: string, args: unknown): string {
  const a = asRecord(args);
  if (!a) return name;
  const detail =
    str(a.path) ||
    str(a.file_path) ||
    str(a.pattern) ||
    str(a.command) ||
    str(a.query) ||
    str(a.uri);
  return detail ? `${name} · ${detail.slice(0, 140)}` : name;
}

function extractDiff(result: unknown): FileDiff | undefined {
  const r = asRecord(result);
  const d = asRecord(r?._diff);
  if (!d) return undefined;
  return {
    path: str(d.path),
    old_text: typeof d.old_text === "string" ? d.old_text : null,
    new_text: str(d.new_text),
  };
}

function extractLocations(result: unknown): SourceLocation[] {
  const r = asRecord(result);
  const raw = r?._locations;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    const o = asRecord(item);
    if (!o || typeof o.path !== "string") return [];
    return [
      {
        path: o.path,
        ...(typeof o.line === "number" ? { line: o.line } : {}),
      },
    ];
  });
}

function extractPlan(value: unknown): PlanEntry[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const entries = value.flatMap((item) => {
    const o = asRecord(item);
    if (!o || typeof o.content !== "string") return [];
    const status = str(o.status);
    return [
      {
        content: o.content,
        status:
          status === "in_progress" || status === "completed"
            ? (status as PlanEntry["status"])
            : "pending",
        ...(typeof o.priority === "string" ? { priority: o.priority } : {}),
      },
    ];
  });
  return entries.length ? entries : undefined;
}

/** The text of a `user.message`, whichever payload shape it uses. */
function userText(payload: Record<string, unknown>): string {
  if (typeof payload.text === "string") return payload.text;
  if (typeof payload.content === "string") return payload.content;
  const parts = payload.parts;
  if (Array.isArray(parts)) {
    return parts
      .map((p) => {
        const o = asRecord(p);
        return o && o.type === "text" ? str(o.text) : "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

/** Inline images in a `user.message`, as data URLs ready for an `<img>`. */
function userImages(payload: Record<string, unknown>): string[] {
  const parts = payload.parts;
  if (!Array.isArray(parts)) return [];
  return parts.flatMap((p) => {
    const o = asRecord(p);
    if (!o || o.type !== "image") return [];
    const source = asRecord(o.source);
    if (!source) return [];
    if (source.kind === "url" && typeof source.url === "string") {
      return [source.url];
    }
    if (source.kind === "base64") {
      return [`data:${str(source.media_type)};base64,${str(source.data)}`];
    }
    return [];
  });
}

function notice(
  event: StudioEvent,
  level: "info" | "warn" | "error",
  title: string,
  detail?: string,
): ChatItem {
  return {
    type: "notice",
    key: `n-${event.seq}`,
    eventId: event.id,
    ts: event.ts,
    level,
    title,
    ...(detail ? { detail } : {}),
  };
}

/**
 * Which events a rewind sealed off.
 *
 * This needs its own pass because a rollback is recorded *after* the events
 * it rejects, and the main pass must already know to skip them — otherwise a
 * rewound turn would still be counted in the statistics and its tool calls
 * would still be tallied, while its bubbles disappeared from the transcript.
 */
function rejectedBy(events: StudioEvent[]): Set<string> {
  const rejected = new Set<string>();
  for (const event of events) {
    if (event.kind !== K.rollback) continue;
    const ids = event.payload?.rejected_event_ids;
    if (Array.isArray(ids)) for (const id of ids) rejected.add(str(id));
  }
  return rejected;
}

export function reduce(events: StudioEvent[]): ChatState {
  const items: ChatItem[] = [];
  const toolsByCallId = new Map<string, ToolItem>();
  const permissionsByRequestId = new Map<string, PermissionItem>();
  const rolledBack = rejectedBy(events);

  let plan: PlanEntry[] = [];
  // Keyed by workstream id, with the planned order held separately.
  //
  // A `Map` orders by insertion, so replacing a plan's placeholder with the
  // real stream when it starts moved it to the end — and streams start in
  // whatever order the concurrency limiter admits them, so the panel
  // reshuffled itself while the user was reading it. `order` keeps the list
  // in the order the goal was split, which is the only stable one there is.
  const workstreams = new Map<string, Workstream>();
  const order: string[] = [];
  const placeKey = (key: string) => {
    if (!order.includes(key)) order.push(key);
  };
  const rekey = (from: string, to: string) => {
    const at = order.indexOf(from);
    if (at === -1) placeKey(to);
    else order[at] = to;
  };
  let untracked: string[] = [];
  let running = false;
  let openAssistant: AssistantItem | null = null;
  let cycleStartedAt: number | null = null;

  const stats = {
    inputTokens: 0,
    outputTokens: 0,
    cachedTokens: 0,
    toolCalls: 0,
    turns: 0,
  } as ChatState["stats"];

  /** A real conversational boundary: nothing may join this bubble again. */
  const closeAssistant = () => {
    if (openAssistant) {
      openAssistant.streaming = false;
      openAssistant = null;
    }
  };

  /**
   * Stop the caret without orphaning the bubble.
   *
   * Studio's own turn markers say its turn *task* finished, which is not the
   * same as the model having finished speaking — they are produced by a
   * different path than the agent's events and can arrive before the
   * `assistant.message` they are supposed to follow. Closing the bubble on
   * one meant that message had nothing to attach to and started a second
   * identical bubble. Settling leaves it attachable.
   */
  const settleAssistant = () => {
    if (openAssistant) openAssistant.streaming = false;
  };

  // Folding the same event in twice is not a hypothetical: this is fed from
  // a network stream with reconnects, history refetches and gap repair, and a
  // repeated `assistant.message` renders as a second identical bubble — the
  // first copy having already closed the streaming one. The reducer cannot
  // stop a duplicate arriving, but it is the last place that can stop one
  // being *shown*, so it holds the invariant itself rather than trusting the
  // transport to.
  const folded = new Set<string>();

  for (const event of events) {
    if (folded.has(event.id)) continue;
    folded.add(event.id);
    // A rewound event stays in the trace but must not reach any view: it is
    // no longer part of the conversation the model will be shown either.
    if (rolledBack.has(event.id)) continue;
    const p = event.payload ?? {};

    switch (event.kind) {
      case K.userMessage: {
        closeAssistant();
        items.push({
          type: "user",
          key: `u-${event.seq}`,
          eventId: event.id,
          seq: event.seq,
          ts: event.ts,
          text: userText(p),
          images: userImages(p),
        });
        break;
      }

      case K.assistantDelta: {
        if (!openAssistant) {
          openAssistant = {
            type: "assistant",
            key: `a-${event.seq}`,
            eventId: event.id,
            seq: event.seq,
            ts: event.ts,
            text: "",
            thinking: "",
            streaming: true,
          };
          items.push(openAssistant);
        }
        if (typeof p.content === "string") openAssistant.text += p.content;
        if (typeof p.reasoning_content === "string") {
          openAssistant.thinking += p.reasoning_content;
        }
        break;
      }

      case K.assistantMessage: {
        const text = str(p.content);
        const thinking = str(p.reasoning_content);
        const meta = event.meta ?? {};
        const tokens = {
          input: num(meta.llm_input_tokens),
          output: num(meta.llm_output_tokens),
          cached: num(meta.llm_cached_input_tokens),
        };
        stats.inputTokens += tokens.input;
        stats.outputTokens += tokens.output;
        stats.cachedTokens += tokens.cached;

        if (openAssistant) {
          // The final message is authoritative: streaming can drop a chunk,
          // and a non-streaming provider sends only this.
          if (text) openAssistant.text = text;
          if (thinking) openAssistant.thinking = thinking;
          openAssistant.streaming = false;
          openAssistant.tokens = tokens;
          // The final message is where the assembly for this step ended, so
          // that is the point to look up its context from.
          openAssistant.seq = event.seq;
          if (!openAssistant.text && !openAssistant.thinking) {
            // A tool-call-only turn has no prose; drop the empty bubble.
            const at = items.indexOf(openAssistant);
            if (at >= 0) items.splice(at, 1);
          }
          openAssistant = null;
        } else if (text || thinking) {
          items.push({
            type: "assistant",
            key: `a-${event.seq}`,
            eventId: event.id,
            seq: event.seq,
            ts: event.ts,
            text,
            thinking,
            streaming: false,
            tokens,
          });
        }

        if (str(p.finalized_due_to) === "max_steps") {
          items.push(
            notice(
              event,
              "warn",
              "Stopped at the step limit",
              "The agent ran out of steps for this turn and wrapped up. Ask it to continue if the work is unfinished.",
            ),
          );
        }
        break;
      }

      case K.toolProposed: {
        closeAssistant();
        const callId = str(p.tool_call_id);
        const name = str(p.name) || "tool";
        const args = parseArgs(p.arguments);
        const item: ToolItem = {
          type: "tool",
          key: `t-${event.seq}`,
          eventId: event.id,
          ts: event.ts,
          callId,
          name,
          title: str(p.title) || describeCall(name, args),
          args,
          status: "running",
          locations: [],
        };
        toolsByCallId.set(callId, item);
        items.push(item);
        stats.toolCalls += 1;
        break;
      }

      case K.toolProgress: {
        const item = toolsByCallId.get(str(p.tool_call_id));
        if (item && str(p.status) === "failed") item.status = "failed";
        break;
      }

      case K.toolResult: {
        const item = toolsByCallId.get(str(p.tool_call_id));
        const error = str(p.error);
        const diff = extractDiff(p.result);
        const locations = extractLocations(p.result);
        const planned = extractPlan(asRecord(p.result)?._plan);
        if (planned) plan = planned;

        if (item) {
          item.status = error ? "failed" : "done";
          item.result = p.result;
          if (error) item.error = error;
          if (diff) item.diff = diff;
          item.locations = locations;
          const started = Date.parse(item.ts);
          const ended = Date.parse(event.ts);
          if (!Number.isNaN(started) && !Number.isNaN(ended)) {
            item.durationMs = ended - started;
          }
        }
        break;
      }

      case K.permissionRequest: {
        const requestId = str(p.request_id);
        const item: PermissionItem = {
          type: "permission",
          key: `p-${event.seq}`,
          eventId: event.id,
          ts: event.ts,
          requestId,
          tool: str(p.tool) || "tool",
          args: parseArgs(p.arguments),
          status: "pending",
        };
        permissionsByRequestId.set(requestId, item);
        items.push(item);
        break;
      }

      case K.permissionDecision: {
        const item = permissionsByRequestId.get(str(p.request_id));
        if (item) {
          item.status = p.approve === true ? "approved" : "denied";
          const reason = str(p.reason);
          if (reason) item.reason = reason;
          const auto = str(p.auto);
          if (auto) item.auto = auto;
        }
        break;
      }

      case K.cycleStart: {
        running = true;
        cycleStartedAt = Date.parse(event.ts);
        break;
      }

      case K.cycleEnd: {
        closeAssistant();
        running = false;
        stats.turns += 1;
        const elapsed = num(event.meta?.elapsed_ms);
        if (elapsed) stats.lastTurnMs = elapsed;
        else if (cycleStartedAt) {
          const ended = Date.parse(event.ts);
          if (!Number.isNaN(ended)) stats.lastTurnMs = ended - cycleStartedAt;
        }
        cycleStartedAt = null;
        break;
      }

      case K.stuck: {
        items.push(
          notice(
            event,
            "warn",
            "Loop detected",
            str(p.hint) ||
              "The agent repeated the same action; it was nudged to try something else.",
          ),
        );
        break;
      }

      case K.summarized: {
        items.push(
          notice(
            event,
            "info",
            "Earlier conversation compacted",
            `${num(p.summarized_events)} events folded into a summary; ${num(
              p.retained_events,
            )} kept verbatim.`,
          ),
        );
        break;
      }

      case K.budgetWarning: {
        items.push(notice(event, "warn", "Approaching the token budget"));
        break;
      }

      case K.budgetExhausted: {
        items.push(
          notice(
            event,
            "error",
            "Token budget exhausted",
            "The session hit its configured token limit and stopped.",
          ),
        );
        break;
      }


      case K.recovered: {
        items.push(
          notice(
            event,
            "info",
            "Interrupted tool calls resolved",
            "Calls left open by a restart were closed out so the history stayed valid.",
          ),
        );
        break;
      }

      case K.systemMessage: {
        const text = str(p.content);
        if (text) items.push(notice(event, "info", text));
        break;
      }

      case K.elicitation: {
        items.push(
          notice(
            event,
            "warn",
            "An MCP server is asking for input",
            str(p.message),
          ),
        );
        break;
      }

      case K.turnInterrupted: {
        settleAssistant();
        running = false;
        items.push(notice(event, "warn", "Stopped by you"));
        break;
      }

      case K.turnEnded: {
        settleAssistant();
        running = false;
        break;
      }

      case K.turnFailed: {
        settleAssistant();
        running = false;
        items.push(notice(event, "error", "Turn failed", str(p.error)));
        break;
      }

      case K.backendLost: {
        running = false;
        items.push(
          notice(event, "error", "The agent process exited", str(p.reason)),
        );
        break;
      }

      case K.fsRefused: {
        items.push(
          notice(
            event,
            "warn",
            `Blocked: the agent tried to ${str(p.action)} outside the workspace`,
            `${str(p.path)} — ${str(p.reason)}`,
          ),
        );
        break;
      }

      case K.treeRestored: {
        const paths = Array.isArray(p.paths) ? p.paths.length : 0;
        items.push(
          notice(
            event,
            "info",
            `Working tree restored (${paths} file${paths === 1 ? "" : "s"})`,
            `Recover what the rewound turns wrote with: git checkout ${str(p.undo)} -- .`,
          ),
        );
        break;
      }

      case K.treeUnchanged: {
        const paths = Array.isArray(p.paths) ? (p.paths as unknown[]) : [];
        items.push(
          notice(
            event,
            "warn",
            "The files on disk were not rewound",
            `${str(p.detail)}${paths.length ? `: ${paths.join(", ")}` : ""}`,
          ),
        );
        break;
      }

      case K.planProposed: {
        const parts = Array.isArray(p.workstreams) ? p.workstreams : [];
        for (const part of parts) {
          const rec = asRecord(part);
          if (!rec) continue;
          const title = str(rec.title);
          // Ids are assigned when a stream starts, so before that the title
          // is the only handle there is.
          const key = `planned:${title}`;
          placeKey(key);
          workstreams.set(key, {
            id: "",
            title,
            brief: str(rec.brief),
            status: "running",
            changes: [],
          });
        }
        items.push(
          notice(
            event,
            "info",
            parts.length === 1
              ? "Working on this in one piece"
              : `Split into ${parts.length} workstreams`,
            parts.map((w) => str(asRecord(w)?.title)).join(" · "),
          ),
        );
        break;
      }

      case K.wsStarted: {
        const id = str(p.id);
        const title = str(p.title);
        // Replace the placeholder the plan created, keeping its position.
        const planned = workstreams.get(`planned:${title}`);
        if (planned) workstreams.delete(`planned:${title}`);
        rekey(`planned:${title}`, id);
        workstreams.set(id, {
          id,
          title,
          brief: str(p.brief) || planned?.brief || "",
          status: "running",
          changes: [],
        });
        break;
      }

      case K.wsFinished: {
        const id = str(p.id);
        const existing = workstreams.get(id);
        const changes = (Array.isArray(p.changes) ? p.changes : []).flatMap((c) => {
          const rec = asRecord(c);
          return rec ? [{ path: str(rec.path), status: str(rec.status) }] : [];
        });
        workstreams.set(id, {
          id,
          title: str(p.title) || existing?.title || id,
          brief: existing?.brief ?? "",
          status: "finished",
          changes,
          report: str(p.report) || undefined,
        });
        break;
      }

      case K.wsSealed: {
        const id = str(p.id);
        const existing = workstreams.get(id);
        if (existing) {
          workstreams.set(id, {
            ...existing,
            status: "sealed",
            epitaph: str(p.epitaph) || undefined,
          });
        }
        break;
      }

      case K.adopted: {
        const changed = Array.isArray(p.changes) ? p.changes.length : 0;
        items.push(
          notice(
            event,
            "info",
            `Kept "${str(p.title)}"`,
            `${changed} file${changed === 1 ? "" : "s"} written to your folder${
              p.overrode === true ? ", overriding your own changes" : ""
            }.`,
          ),
        );
        break;
      }

      case K.adoptionBlocked: {
        const conflicts = Array.isArray(p.conflicts) ? p.conflicts : [];
        items.push(
          notice(
            event,
            "warn",
            `"${str(p.title)}" was not applied — you changed the same files`,
            `${conflicts
              .map((c) => str(asRecord(c)?.path))
              .join(", ")} — your folder is untouched. Keep your versions, or adopt again overriding them.`,
          ),
        );
        break;
      }

      case K.notTracked: {
        untracked = (Array.isArray(p.repositories) ? p.repositories : []).map(str);
        if (untracked.length) {
          items.push(
            notice(
              event,
              "warn",
              "Some of this folder is not tracked",
              `${untracked.join(", ")} — these have their own history, so cowork will not snapshot or restore them.`,
            ),
          );
        }
        break;
      }

      case K.plugins: {
        // A plugin changes the system prompt and the tool list without
        // appearing anywhere else. Saying which loaded is the difference
        // between the user seeing that and inferring it from behaviour.
        const loaded = Array.isArray(p.plugins) ? p.plugins : [];
        if (loaded.length) {
          const names = loaded.map((x) => str(asRecord(x)?.name)).filter(Boolean);
          items.push(
            notice(
              event,
              "info",
              `${names.length} plugin${names.length === 1 ? "" : "s"} active`,
              names.join(", "),
            ),
          );
        }
        break;
      }

      case K.epitaph: {
        // The branch's events are gone; this sentence is what is left of
        // them, and it is still steering the agent.
        items.push(
          notice(
            event,
            "info",
            "An earlier attempt was summarised and its detail dropped",
            str(p.epitaph),
          ),
        );
        break;
      }

      case K.modeChanged: {
        items.push(
          notice(event, "info", `Mode set to ${str(p.label) || str(p.mode)}`),
        );
        break;
      }

      case K.rewound: {
        items.push(
          notice(
            event,
            "info",
            `Rewound ${num(p.turns)} turn${num(p.turns) === 1 ? "" : "s"}`,
            "The undone trajectory is kept as a rejected branch — visible in the trace, not sent to the model.",
          ),
        );
        break;
      }

      case K.plan: {
        const planned = extractPlan(p.entries);
        if (planned) plan = planned;
        break;
      }

      default:
        break;
    }
  }

  // Deliberately no final `closeAssistant()`: a bubble still open at the end
  // of the stream is one the model is still writing, and that is exactly when
  // the transcript should show it as live.

  return {
    items,
    plan,
    running,
    stats,
    rolledBack,
    pendingPermissions: items.filter(
      (i): i is PermissionItem => i.type === "permission" && i.status === "pending",
    ),
    // Rolled-back attempts are injected into every request from here on.
    // Derived rather than announced: the reducer already knows which events
    // a rollback sealed, and publishing a second event per turn to say so
    // would be noise in the one place it should not be.
    sealedAttempts: events.filter((e) => e.kind === K.rollback).length,
    workstreams: order.flatMap((key) => {
      const stream = workstreams.get(key);
      return stream ? [stream] : [];
    }),
    untracked,
  };
}
