/**
 * What was actually sent to the model, and where each part came from.
 *
 * The transcript shows the conversation and the trace shows the events, but
 * neither answers the question that matters when an agent starts behaving
 * oddly on a long session: *what was in the request?* Half of it may be a
 * summary of a conversation the model can no longer see, and until now the
 * only evidence of that was a number in a log line.
 *
 * Two things here. A breakdown of each request — how much was the system
 * prefix, how much was verbatim history, how much was compacted away — read
 * from the `agent.context.assembled` record. And the summary itself, in full,
 * editable: if compaction dropped something that mattered, the fix is to
 * replace it rather than to restart the session.
 */

import { useMemo, useState } from "react";
import type { StudioEvent } from "../lib/types";
import { CopyButton, formatTime, formatTokens } from "./primitives";

interface Assembly {
  seq: number;
  ts: string;
  messages: number;
  totalTokens: number;
  systemTokens: number;
  verbatim: number;
  summarized: number;
  summaryTokens: number;
  compacted: boolean;
  budget: number;
  manifest: ManifestEntry[];
}

/** One message in the request, as the harness recorded it. */
interface ManifestEntry {
  index: number;
  role: string;
  tokens: number;
  source: "system" | "summary" | "verbatim" | "cleared" | "other";
  /** What the message said, up to the recording cap. */
  text: string;
  /** Its true length, when the record had to cut it. */
  truncated_from?: number | null;
}

/** One thing a reviewer says the summary lost. */
interface AuditItem {
  id: string;
  fact: string;
  why?: string;
}

interface Audit {
  seq: number;
  requestId: string;
  items: AuditItem[];
  error?: string;
}

interface Summary {
  seq: number;
  ts: string;
  text: string;
  covers: number;
  manual: boolean;
}

const num = (v: unknown) => (typeof v === "number" ? v : 0);
const str = (v: unknown) => (typeof v === "string" ? v : "");

function readAssemblies(events: StudioEvent[]): Assembly[] {
  return events
    .filter((e) => e.kind === "agent.context.assembled")
    .map((e) => ({
      seq: e.seq,
      ts: e.ts,
      messages: num(e.payload?.messages),
      totalTokens: num(e.payload?.total_tokens),
      systemTokens: num(e.payload?.system_tokens),
      verbatim: num(e.payload?.verbatim_messages),
      summarized: num(e.payload?.summarized_messages),
      summaryTokens: num(e.payload?.summary_tokens),
      compacted: e.payload?.compacted === true,
      budget: num(e.payload?.budget),
      manifest: Array.isArray(e.payload?.manifest)
        ? (e.payload.manifest as ManifestEntry[])
        : [],
    }));
}

/** The newest review request, and the answer to it if one has arrived. */
function readAudit(events: StudioEvent[]): {
  asked: number | null;
  result: Audit | null;
} {
  let asked: number | null = null;
  let askedId = "";
  let result: Audit | null = null;

  for (const e of events) {
    if (e.kind === "agent.context.audit.requested") {
      asked = e.seq;
      askedId = str(e.payload?.request_id);
      result = null; // a new question invalidates the previous answer
    }
    if (e.kind === "agent.context.audit.result") {
      const requestId = str(e.payload?.request_id);
      if (requestId && requestId !== askedId) continue;
      const raw = Array.isArray(e.payload?.items) ? e.payload.items : [];
      result = {
        seq: e.seq,
        requestId,
        error: str(e.payload?.error) || undefined,
        items: raw
          .map((item: unknown, i: number) => {
            const o = (item ?? {}) as Record<string, unknown>;
            return {
              id: str(o.id) || `item-${i}`,
              fact: str(o.fact),
              why: str(o.why) || undefined,
            };
          })
          .filter((item: AuditItem) => item.fact.length > 0),
      };
    }
  }
  return { asked, result };
}

function readSummaries(events: StudioEvent[]): Summary[] {
  return events
    .filter((e) => e.kind === "agent.context.summarized")
    .map((e) => ({
      seq: e.seq,
      ts: e.ts,
      text: str(e.payload?.summary),
      covers: num(e.payload?.summarized_count),
      manual: str(e.payload?.source) === "manual_override",
    }));
}

export function Context({
  events,
  position,
  focus,
  onClearFocus,
  canEdit,
  onOverride,
  onReview,
}: {
  events: StudioEvent[];
  position: number;
  /** A message this pane is pinned to, which overrides the playhead. */
  focus: number | null;
  onClearFocus: () => void;
  canEdit: boolean;
  onOverride: (summary: string, covers: number) => void;
  /** Ask an installed reviewer what compaction dropped. */
  onReview: () => void;
}) {
  const assemblies = useMemo(() => readAssemblies(events), [events]);
  const summaries = useMemo(() => readSummaries(events), [events]);
  const [draft, setDraft] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<number>>(() => new Set());
  const audit = useMemo(() => readAudit(events), [events]);
  const [chosen, setChosen] = useState<Set<string>>(() => new Set());
  // Cleared once the answer arrives, so a second review does not inherit the
  // ticks from the first.
  const answeredAt = audit.result?.seq ?? null;
  const [seenAnswer, setSeenAnswer] = useState<number | null>(null);
  if (answeredAt !== seenAnswer) {
    setSeenAnswer(answeredAt);
    if (chosen.size) setChosen(new Set());
  }

  // Normally the request at or before the playhead, so scrubbing the timeline
  // shows what the model was working from at that moment. When a message is
  // pinned, that message wins and the playhead is ignored.
  const at = focus ?? position;
  const current =
    [...assemblies].reverse().find((a) => a.seq <= at) ??
    assemblies[assemblies.length - 1];
  const active = summaries[summaries.length - 1];
  // A request with no answer yet. The reviewer is a plugin, so "no answer" may
  // also mean none is installed — said plainly rather than spun forever.
  const answered = audit.result !== null;
  const reviewing = audit.asked !== null && !answered;

  if (!assemblies.length && !summaries.length) {
    return (
      <div className="trace-blank">
        Nothing assembled yet. Once the agent makes a request, this shows what
        went into it — and once the context is compacted, the summary it was
        replaced with.
      </div>
    );
  }

  return (
    <div className="context-pane">
      {current && (
        <>
          {focus !== null && (
            <div className="ctx-pinned">
              <span>
                Pinned to the message at event <b>{focus}</b>
              </span>
              <span className="spacer" />
              <button className="btn sm ghost" onClick={onClearFocus}>
                Follow the playhead
              </button>
            </div>
          )}

          <div className="ctx-head">
            <span>
              Request at event {current.seq}
              {current.seq !== at && (
                <span className="muted"> · nearest at or before {at}</span>
              )}
            </span>
            <span className="muted">{formatTime(current.ts)}</span>
            <span className="spacer" />
            <span className="muted">
              {formatTokens(current.totalTokens)} of{" "}
              {formatTokens(current.budget)} budget
            </span>
          </div>

          {/* Where the tokens went. Compacted history is drawn as its own
              band because that is the part the model can no longer see. */}
          <div className="ctx-bar" role="img" aria-label="context composition">
            <div
              className="seg system"
              style={{ flex: Math.max(current.systemTokens, 1) }}
              title={`system prefix — ${current.systemTokens} tokens`}
            />
            {current.summaryTokens > 0 && (
              <div
                className="seg summary"
                style={{ flex: current.summaryTokens }}
                title={`summary — ${current.summaryTokens} tokens`}
              />
            )}
            <div
              className="seg verbatim"
              style={{
                flex: Math.max(
                  current.totalTokens - current.systemTokens - current.summaryTokens,
                  1,
                ),
              }}
              title="verbatim conversation"
            />
          </div>

          <div className="ctx-legend">
            <span>
              <i className="swatch system" /> system prefix{" "}
              {formatTokens(current.systemTokens)}
            </span>
            {current.summaryTokens > 0 && (
              <span>
                <i className="swatch summary" /> summary{" "}
                {formatTokens(current.summaryTokens)} ({current.summarized} messages
                folded)
              </span>
            )}
            <span>
              <i className="swatch verbatim" /> verbatim {current.verbatim} messages
            </span>
          </div>

          {!current.compacted && (
            <p className="ctx-note">
              Nothing has been compacted. Everything the model saw is the real
              conversation.
            </p>
          )}

          {current.manifest.length > 0 ? (
            <>
              <div className="ctx-head" style={{ marginTop: 16 }}>
                <span>The {current.manifest.length} messages sent</span>
                <span className="spacer" />
                <CopyButton
                  text={() =>
                    current.manifest
                      .map(
                        (m) =>
                          `--- ${m.index} ${m.role} (${m.source}, ${m.tokens} tok)\n${m.text}`,
                      )
                      .join("\n\n")
                  }
                  label="⧉"
                  className="btn sm ghost icon"
                  title="Copy this list"
                />
              </div>
              <ol className="ctx-messages">
                {current.manifest.map((m) => {
                  const open = expanded.has(m.index);
                  return (
                    <li key={m.index} className={`msg-row ${m.source} ${open ? "open" : ""}`}>
                      <button
                        className="msg-line"
                        onClick={() =>
                          setExpanded((current) => {
                            const next = new Set(current);
                            if (next.has(m.index)) next.delete(m.index);
                            else next.add(m.index);
                            return next;
                          })
                        }
                      >
                        <span className="idx">{m.index}</span>
                        <span className={`role ${m.role}`}>{m.role}</span>
                        <span className={`origin ${m.source}`}>{m.source}</span>
                        <span className="size">{formatTokens(m.tokens)}</span>
                        <span className="preview">
                          {open ? "" : m.text.split("\n")[0] || "(no text)"}
                        </span>
                        <span className="chev">{open ? "▾" : "▸"}</span>
                      </button>
                      {open && (
                        <div className="msg-full">
                          <pre>{m.text || "(no text)"}</pre>
                          <div className="msg-foot">
                            {m.truncated_from ? (
                              <span className="muted">
                                showing the first {m.text.length.toLocaleString()} of{" "}
                                {m.truncated_from.toLocaleString()} characters — the
                                whole thing is in the event log
                              </span>
                            ) : (
                              <span className="muted">complete</span>
                            )}
                            <span className="spacer" />
                            <CopyButton
                              text={m.text}
                              label="⧉ Copy"
                              className="btn sm ghost"
                            />
                          </div>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ol>
              <p className="ctx-note">
                <b>verbatim</b> is the real conversation. <b>summary</b> stands
                in for messages the model can no longer see.{" "}
                <b>cleared</b> is a tool result whose body was dropped to
                reclaim budget — the full text is still in the event log.
              </p>
            </>
          ) : (
            <p className="ctx-note">
              This request predates per-message recording, so only the totals
              are known.
            </p>
          )}
        </>
      )}

      {active && (
        <div className="ctx-summary">
          <div className="ctx-head">
            <span>
              Summary in effect{" "}
              {active.manual && <span className="badge">edited by you</span>}
            </span>
            <span className="muted">covers {active.covers} messages</span>
            <span className="spacer" />
            <CopyButton
              text={active.text}
              label="⧉"
              className="btn sm ghost icon"
              title="Copy the summary"
            />
            {canEdit && draft === null && (
              <button
                className="btn sm"
                onClick={onReview}
                disabled={reviewing}
                title="Ask an installed reviewer which details this summary lost. Costs one model call."
              >
                {reviewing ? "Reviewing…" : "Review"}
              </button>
            )}
            {canEdit && draft === null && (
              <button className="btn sm" onClick={() => setDraft(active.text)}>
                Edit
              </button>
            )}
          </div>

          {draft === null ? (
            <pre className="ctx-text">{active.text}</pre>
          ) : (
            <>
              <textarea
                className="ctx-edit"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                spellCheck={false}
              />
              <div className="ctx-actions">
                <span className="ctx-note">
                  Replacing this does not erase the old one — it is superseded,
                  and both stay in the trace.
                </span>
                <span className="spacer" />
                <button className="btn sm" onClick={() => setDraft(null)}>
                  Cancel
                </button>
                <button
                  className="btn sm primary"
                  disabled={!draft.trim()}
                  onClick={() => {
                    onOverride(draft, active.covers);
                    setDraft(null);
                  }}
                >
                  Replace
                </button>
              </div>
            </>
          )}

          {reviewing && (
            <p className="ctx-note">
              Asking the reviewer what this summary left out. If nothing answers,
              no reviewer plugin is installed for this session.
            </p>
          )}

          {audit.result?.error && (
            <p className="ctx-note error">
              The reviewer could not finish: {audit.result.error}
            </p>
          )}

          {answered && audit.result && !audit.result.error && (
            audit.result.items.length === 0 ? (
              <p className="ctx-note">
                The reviewer found nothing important missing from this summary.
              </p>
            ) : (
              <div className="ctx-review">
                <div className="ctx-head">
                  <span>
                    {audit.result.items.length} details this summary left out
                  </span>
                  <span className="spacer" />
                  <span className="muted">
                    Tick what should go back into the context
                  </span>
                </div>

                <ul className="ctx-items">
                  {audit.result.items.map((item) => (
                    <li key={item.id}>
                      <label>
                        <input
                          type="checkbox"
                          checked={chosen.has(item.id)}
                          onChange={() =>
                            setChosen((current) => {
                              const next = new Set(current);
                              if (next.has(item.id)) next.delete(item.id);
                              else next.add(item.id);
                              return next;
                            })
                          }
                        />
                        <span className="fact">{item.fact}</span>
                        {item.why && <span className="why">{item.why}</span>}
                      </label>
                    </li>
                  ))}
                </ul>

                <div className="ctx-actions">
                  <button
                    className="btn sm ghost"
                    onClick={() =>
                      setChosen(
                        chosen.size === audit.result!.items.length
                          ? new Set()
                          : new Set(audit.result!.items.map((i) => i.id)),
                      )
                    }
                  >
                    {chosen.size === audit.result.items.length
                      ? "Select none"
                      : "Select all"}
                  </button>
                  <span className="spacer" />
                  <button
                    className="btn sm primary"
                    disabled={chosen.size === 0}
                    onClick={() => {
                      const picked = audit
                        .result!.items.filter((i) => chosen.has(i.id))
                        .map((i) => `- ${i.fact}`)
                        .join("\n");
                      onOverride(
                        `${active.text}\n\nDetails restored from the compacted history:\n${picked}`,
                        active.covers,
                      );
                      setChosen(new Set());
                    }}
                  >
                    Add {chosen.size || ""} to context
                  </button>
                </div>
              </div>
            )
          )}

          {summaries.length > 1 && (
            <p className="ctx-note">
              {summaries.length - 1} earlier{" "}
              {summaries.length === 2 ? "version" : "versions"} in the trace.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
