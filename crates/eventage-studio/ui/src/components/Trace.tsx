/**
 * The trace: every event, and three ways to read it.
 *
 * The chat view is a rendering of what happened; this is the record it was
 * rendered from. Three views because three different questions get asked of a
 * session — *when and how long* (timeline), *who asked whom* (flow), and
 * *exactly what was in that payload* (events) — and one list answers only the
 * last of them.
 *
 * The playhead is shared with the transcript, so scrubbing here replays the
 * conversation rather than merely filtering a list.
 */

import { useMemo, useState } from "react";
import type { SessionStats, StudioEvent } from "../lib/types";
import { buildFlow, buildTimeline } from "../lib/timeline";
import type { Checkpoint } from "../lib/timeline";
import { Context } from "./Context";
import { Flow } from "./Flow";
import { Timeline } from "./Timeline";
import { Transport } from "./Transport";
import {
  CopyButton,
  Json,
  formatDuration,
  formatTime,
  formatTokens,
} from "./primitives";

type Tab = "timeline" | "flow" | "context" | "events";
type Group = "llm" | "tools" | "permissions" | "lifecycle" | "system";

const GROUPS: { id: Group; label: string }[] = [
  { id: "llm", label: "model" },
  { id: "tools", label: "tools" },
  { id: "permissions", label: "permissions" },
  { id: "lifecycle", label: "lifecycle" },
  { id: "system", label: "system" },
];

function groupOf(kind: string): Group {
  if (kind.startsWith("assistant.") || kind.startsWith("agent.context.")) return "llm";
  if (kind.startsWith("tool.")) return "tools";
  if (kind.startsWith("permission.") || kind.startsWith("mcp.elicitation"))
    return "permissions";
  if (
    kind.startsWith("agent.cycle.") ||
    kind === "user.message" ||
    kind.startsWith("studio.turn.")
  )
    return "lifecycle";
  return "system";
}

function colourOf(kind: string): string {
  if (kind.startsWith("assistant.")) return "k-assistant";
  if (kind.startsWith("tool.")) return "k-tool";
  if (kind === "user.message") return "k-user";
  if (
    kind.startsWith("budget.") ||
    kind === "studio.turn.failed" ||
    kind === "studio.backend.lost" ||
    kind === "agent.stuck"
  )
    return "k-error";
  return "k-system";
}

const str = (v: unknown) => (typeof v === "string" ? v : "");
const num = (v: unknown) => (typeof v === "number" ? v : 0);

function summarise(event: StudioEvent): string {
  const p = event.payload ?? {};
  switch (event.kind) {
    case "assistant.delta":
      return str(p.content) || str(p.reasoning_content);
    case "assistant.message": {
      const calls = Array.isArray(p.tool_calls) ? p.tool_calls.length : 0;
      const text = str(p.content);
      return calls ? `${calls} tool call${calls === 1 ? "" : "s"} ${text}` : text;
    }
    case "tool.call.proposed":
      return `${str(p.name)} ${str(p.arguments)}`;
    case "tool.result":
      return str(p.error)
        ? `error: ${str(p.error)}`
        : JSON.stringify(p.result ?? "").slice(0, 200);
    case "permission.request":
      return `${str(p.tool)} awaiting approval`;
    case "permission.decision":
      return `${p.approve === true ? "allowed" : "denied"} ${str(p.reason)}`;
    case "user.message":
      return str(p.text) || JSON.stringify(p.parts ?? "").slice(0, 200);
    case "agent.cycle.end":
      return event.meta?.elapsed_ms ? formatDuration(num(event.meta.elapsed_ms)) : "";
    case "agent.context.assembled":
      return `${num(p.total_tokens)} tok — ${num(p.verbatim_messages)} verbatim` +
        (p.compacted === true ? `, ${num(p.summarized_messages)} compacted` : "");
    case "agent.context.summarized":
      return `${num(p.summarized_count)} messages folded${
        str(p.source) === "manual_override" ? " (edited by you)" : ""
      }`;
    case "system.rollback": {
      const ids = Array.isArray(p.rejected_event_ids) ? p.rejected_event_ids.length : 0;
      return `${ids} event${ids === 1 ? "" : "s"} sealed into a rejected branch`;
    }
    default: {
      const text = JSON.stringify(p);
      return text === "{}" ? "" : text.slice(0, 200);
    }
  }
}

export type { Tab };

export function Trace({
  events,
  stats,
  rolledBack,
  fullTrace,
  position,
  contextFocus,
  onClearContextFocus,
  live,
  expanded,
  onSeek,
  onGoLive,
  onToggleExpand,
  onClose,
  onRewindTo,
  onOverrideSummary,
  tab,
  onTab,
}: {
  events: StudioEvent[];
  stats: SessionStats;
  rolledBack: Set<string>;
  fullTrace: boolean;
  position: number;
  /// A message the context pane is pinned to, overriding the playhead.
  contextFocus: number | null;
  onClearContextFocus: () => void;
  live: boolean;
  expanded: boolean;
  onSeek: (seq: number) => void;
  onGoLive: () => void;
  onToggleExpand: () => void;
  onClose: () => void;
  onRewindTo: (checkpoint: Checkpoint) => void;
  onOverrideSummary: (summary: string, covers: number) => void;
  tab: Tab;
  onTab: (tab: Tab) => void;
}) {
  const [active, setActive] = useState<Set<Group>>(
    () => new Set(GROUPS.map((g) => g.id)),
  );
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const model = useMemo(() => buildTimeline(events), [events]);
  const flow = useMemo(() => buildFlow(events, model), [events, model]);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return events.filter((event) => {
      if (!active.has(groupOf(event.kind))) return false;
      if (!needle) return true;
      return (
        event.kind.toLowerCase().includes(needle) ||
        JSON.stringify(event.payload ?? {}).toLowerCase().includes(needle)
      );
    });
  }, [events, active, query]);

  const chosen = selected === null ? null : events.find((e) => e.seq === selected);

  const toggle = (group: Group) =>
    setActive((current) => {
      const next = new Set(current);
      if (next.has(group)) next.delete(group);
      else next.add(group);
      return next;
    });

  const exportJsonl = () => {
    const body = events.map((e) => JSON.stringify(e)).join("\n");
    const url = URL.createObjectURL(new Blob([body], { type: "application/x-ndjson" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `eventage-trace-${Date.now()}.jsonl`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const seek = (seq: number) => {
    onSeek(Math.max(0, Math.min(seq, model.lastSeq)));
  };

  return (
    <aside className={`trace ${expanded ? "expanded" : ""}`}>
      <div className="trace-head">
        <div className="tabs">
          {(["timeline", "flow", "context", "events"] as Tab[]).map((id) => (
            <button
              key={id}
              className={`tab ${tab === id ? "on" : ""}`}
              onClick={() => onTab(id)}
            >
              {id[0]!.toUpperCase() + id.slice(1)}
            </button>
          ))}
        </div>
        <span className="spacer" />
        <span className="trace-stat" title="prompt tokens in / completion out / read from cache">
          {formatTokens(stats.inputTokens)}↑ {formatTokens(stats.outputTokens)}↓
          {stats.cachedTokens > 0 && ` ${formatTokens(stats.cachedTokens)}⚡`}
        </span>
        <button className="btn sm ghost icon" onClick={exportJsonl} title="Export as JSONL">
          ⤓
        </button>
        <button
          className="btn sm ghost icon"
          onClick={onToggleExpand}
          title={expanded ? "Collapse" : "Expand to full window"}
        >
          {expanded ? "⤡" : "⤢"}
        </button>
        <button className="btn sm ghost icon" onClick={onClose} title="Hide the trace">
          ✕
        </button>
      </div>

      <Transport
        position={position}
        lastSeq={model.lastSeq}
        playing={playing}
        speed={speed}
        live={live}
        onSeek={seek}
        onPlayPause={() => setPlaying((v) => !v)}
        onSpeed={setSpeed}
        onGoLive={() => {
          setPlaying(false);
          onGoLive();
        }}
      />

      {!events.length ? (
        <div className="trace-blank">
          {fullTrace
            ? "Nothing yet — the trace fills as the agent works."
            : "In ACP mode the trace shows protocol traffic; hook decisions and token accounting stay inside the agent's own process."}
        </div>
      ) : (
        <>
          {tab === "timeline" && (
            <div className="trace-body">
              <Timeline
                model={model}
                position={position}
                onSeek={seek}
                onSelect={setSelected}
                height={expanded ? 420 : 240}
              />
              {model.checkpoints.length > 0 && (
                <div className="checkpoint-strip">
                  <span className="strip-label">Rewind to</span>
                  {model.checkpoints.map((cp) => (
                    <button
                      key={cp.eventId}
                      className={`chip ${cp.used ? "" : "on"}`}
                      disabled={cp.used}
                      onClick={() => onRewindTo(cp)}
                      title={
                        cp.used
                          ? "already rewound past this point"
                          : `discard everything after turn ${cp.turn - 1}`
                      }
                    >
                      ⚑ turn {cp.turn}
                    </button>
                  ))}
                </div>
              )}
              <TimelineLegend model={model} />
            </div>
          )}

          {tab === "flow" && (
            <div className="trace-body">
              <Flow
                messages={flow}
                model={model}
                position={position}
                selected={selected}
                onSelect={(seq) => {
                  setSelected(seq);
                  seek(seq);
                }}
              />
            </div>
          )}

          {tab === "context" && (
            <div className="trace-body">
              <Context
                events={events}
                position={position}
                focus={contextFocus}
                onClearFocus={onClearContextFocus}
                canEdit={fullTrace}
                onOverride={onOverrideSummary}
              />
            </div>
          )}

          {tab === "events" && (
            <div className="trace-body events">
              <div className="trace-filters">
                {GROUPS.map((group) => (
                  <button
                    key={group.id}
                    className={`chip ${active.has(group.id) ? "on" : ""}`}
                    onClick={() => toggle(group.id)}
                  >
                    {group.label}
                  </button>
                ))}
              </div>
              <div className="trace-search">
                <input
                  value={query}
                  placeholder={`Search ${events.length} events…`}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>
              <div className="trace-list">
                {rows.length === 0 && (
                  <div className="trace-blank">No events match this filter.</div>
                )}
                {rows.map((event) => (
                  <button
                    key={event.seq}
                    className={`trace-row ${selected === event.seq ? "selected" : ""} ${
                      rolledBack.has(event.id) ? "rolled-back" : ""
                    } ${event.seq > position ? "future" : ""}`}
                    onClick={() => {
                      setSelected(event.seq);
                      seek(event.seq);
                    }}
                  >
                    <span className="seq">{event.seq}</span>
                    <span className="time">{formatTime(event.ts)}</span>
                    <span className={`kind ${colourOf(event.kind)}`}>{event.kind}</span>
                    <span className="summary">{summarise(event)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {chosen && (
        <div className="trace-detail">
          <div className="trace-detail-head">
            <span className={`kind ${colourOf(chosen.kind)}`}>{chosen.kind}</span>
            <span className="muted">#{chosen.seq}</span>
            <span className="muted">{formatTime(chosen.ts)}</span>
            <span className="spacer" />
            <CopyButton
              text={() => JSON.stringify(chosen, null, 2)}
              label="⧉"
              className="btn sm ghost icon"
              title="Copy this event as JSON"
            />
            <button className="btn sm ghost icon" onClick={() => setSelected(null)}>
              ✕
            </button>
          </div>
          <div className="trace-detail-body">
            <Json
              value={{
                id: chosen.id,
                ...(chosen.parent ? { parent: chosen.parent } : {}),
                payload: chosen.payload,
                ...(chosen.meta && Object.keys(chosen.meta).length
                  ? { meta: chosen.meta }
                  : {}),
              }}
            />
          </div>
        </div>
      )}
    </aside>
  );
}

function TimelineLegend({ model }: { model: ReturnType<typeof buildTimeline> }) {
  const failed = model.spans.filter((s) => s.status === "failed").length;
  const slowest = model.spans.reduce(
    (worst, s) => (s.endMs - s.startMs > worst.ms ? { name: s.name, ms: s.endMs - s.startMs } : worst),
    { name: "", ms: 0 },
  );
  return (
    <div className="legend">
      <span className="leg">
        <i className="swatch accent" /> tool call
      </span>
      <span className="leg">
        <i className="swatch danger" /> failed
      </span>
      <span className="leg">
        <i className="swatch warn" /> checkpoint
      </span>
      <span className="spacer" />
      {model.turns.length > 0 && (
        <span className="leg muted">
          {model.turns.length} turn{model.turns.length === 1 ? "" : "s"}
        </span>
      )}
      <span className="leg muted">
        {model.spans.length} call{model.spans.length === 1 ? "" : "s"}
        {failed > 0 && `, ${failed} failed`}
      </span>
      {slowest.ms > 0 && (
        <span className="leg muted">
          slowest {slowest.name} {formatDuration(slowest.ms)}
        </span>
      )}
    </div>
  );
}
