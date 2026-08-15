/**
 * Derive a visual model of a session from its events.
 *
 * The event log already records who did what and when, so the timeline is not
 * a separate thing to maintain — it is another projection, like the
 * transcript. Everything here is pure so the geometry can be tested without a
 * canvas.
 *
 * Two ideas carry most of the value:
 *
 * - A tool call is a *span*, not a point. `tool.call.proposed` and its
 *   `tool.result` bracket a duration, and drawing that duration is what makes
 *   concurrency visible — the ReAct strategy runs several tools at once, and a
 *   list of events cannot show that.
 * - A checkpoint is a *rewind anchor*. Marking them on the same axis is what
 *   turns "rewind 1 turn" into "go back to this point", which is the thing a
 *   person actually wants.
 */

import type { StudioEvent } from "./types";

export type LaneKind = "user" | "agent" | "tool" | "system";

export interface Lane {
  id: string;
  label: string;
  kind: LaneKind;
}

/** A tool call, from proposal to result. */
export interface Span {
  callId: string;
  lane: string;
  name: string;
  startMs: number;
  endMs: number;
  startSeq: number;
  endSeq: number;
  status: "running" | "done" | "failed";
  rolledBack: boolean;
}

/** A point event. */
export interface Mark {
  seq: number;
  lane: string;
  atMs: number;
  kind: string;
  label: string;
  rolledBack: boolean;
}

/** A rewind anchor. */
export interface Checkpoint {
  seq: number;
  eventId: string;
  atMs: number;
  /** Turn number this checkpoint opens, 1-based. */
  turn: number;
  /** True once a rewind has discarded everything after it. */
  used: boolean;
}

/** One agent cycle. */
export interface Turn {
  index: number;
  startSeq: number;
  endSeq: number;
  startMs: number;
  endMs: number;
  complete: boolean;
  rolledBack: boolean;
}

/** A trajectory a rewind discarded. */
export interface RejectedBranch {
  atSeq: number;
  atMs: number;
  eventIds: string[];
  count: number;
}

export interface TimelineModel {
  lanes: Lane[];
  marks: Mark[];
  spans: Span[];
  checkpoints: Checkpoint[];
  turns: Turn[];
  rejected: RejectedBranch[];
  startMs: number;
  endMs: number;
  /** Sequence number of the last event, i.e. the end of the scrub range. */
  lastSeq: number;
}

const USER_LANE = "user";
const AGENT_LANE = "agent";
const SYSTEM_LANE = "system";

const str = (v: unknown): string => (typeof v === "string" ? v : "");

const time = (event: StudioEvent): number => {
  const t = Date.parse(event.ts);
  return Number.isNaN(t) ? 0 : t;
};

/** A short, human label for a point event. */
function labelOf(event: StudioEvent): string {
  const p = event.payload ?? {};
  switch (event.kind) {
    case "user.message":
      return "prompt";
    case "assistant.message":
      return str(p.content) ? "reply" : "tool calls";
    case "permission.request":
      return `asks: ${str(p.tool)}`;
    case "permission.decision":
      return p.approve === true ? "allowed" : "denied";
    case "agent.cycle.start":
      return "turn start";
    case "agent.cycle.end":
      return "turn end";
    case "system.checkpoint":
      return "checkpoint";
    case "agent.stuck":
      return "loop detected";
    case "agent.context.summarized":
      return "compacted";
    case "budget.exhausted":
      return "budget spent";
    case "system.rollback":
      return "rewound";
    default:
      return event.kind.split(".").pop() ?? event.kind;
  }
}

/** Events the timeline shows as points. Deltas are far too many to draw. */
const POINT_KINDS = new Set([
  "user.message",
  "assistant.message",
  "permission.request",
  "permission.decision",
  "agent.cycle.start",
  "agent.cycle.end",
  "system.checkpoint",
  "agent.stuck",
  "agent.context.summarized",
  "budget.warning",
  "budget.exhausted",
  "system.rollback",
  "system.recovered",
]);

function laneFor(event: StudioEvent): string {
  switch (event.kind) {
    case "user.message":
    case "permission.decision":
      return USER_LANE;
    case "assistant.message":
    case "permission.request":
    case "agent.cycle.start":
    case "agent.cycle.end":
    case "agent.stuck":
      return AGENT_LANE;
    default:
      return SYSTEM_LANE;
  }
}

export function buildTimeline(events: StudioEvent[]): TimelineModel {
  const rolledBack = new Set<string>();
  const rejected: RejectedBranch[] = [];
  for (const event of events) {
    if (event.kind !== "system.rollback") continue;
    const ids = event.payload?.rejected_event_ids;
    const list = Array.isArray(ids) ? ids.map(str) : [];
    for (const id of list) rolledBack.add(id);
    rejected.push({
      atSeq: event.seq,
      atMs: time(event),
      eventIds: list,
      count: list.length,
    });
  }

  const marks: Mark[] = [];
  const spans: Span[] = [];
  const checkpoints: Checkpoint[] = [];
  const turns: Turn[] = [];
  const toolLanes = new Map<string, Lane>();
  const openCalls = new Map<string, Span>();

  let openTurn: Turn | null = null;
  let turnIndex = 0;

  for (const event of events) {
    const at = time(event);
    const gone = rolledBack.has(event.id);

    if (event.kind === "tool.call.proposed") {
      const name = str(event.payload?.name) || "tool";
      const callId = str(event.payload?.tool_call_id) || `c${event.seq}`;
      const lane = `tool:${name}`;
      if (!toolLanes.has(lane)) {
        toolLanes.set(lane, { id: lane, label: name, kind: "tool" });
      }
      const span: Span = {
        callId,
        lane,
        name,
        startMs: at,
        endMs: at,
        startSeq: event.seq,
        endSeq: event.seq,
        status: "running",
        rolledBack: gone,
      };
      openCalls.set(callId, span);
      spans.push(span);
      continue;
    }

    if (event.kind === "tool.result") {
      const callId = str(event.payload?.tool_call_id);
      const span = openCalls.get(callId);
      if (span) {
        span.endMs = at;
        span.endSeq = event.seq;
        span.status = event.payload?.error ? "failed" : "done";
        openCalls.delete(callId);
      }
      continue;
    }

    if (event.kind === "agent.cycle.start") {
      turnIndex += 1;
      openTurn = {
        index: turnIndex,
        startSeq: event.seq,
        endSeq: event.seq,
        startMs: at,
        endMs: at,
        complete: false,
        rolledBack: gone,
      };
      turns.push(openTurn);
    }
    if (event.kind === "agent.cycle.end" && openTurn) {
      openTurn.endSeq = event.seq;
      openTurn.endMs = at;
      openTurn.complete = true;
      openTurn = null;
    }

    if (event.kind === "system.checkpoint") {
      checkpoints.push({
        seq: event.seq,
        eventId: event.id,
        atMs: at,
        turn: turnIndex + 1,
        used: gone,
      });
    }

    if (POINT_KINDS.has(event.kind)) {
      marks.push({
        seq: event.seq,
        lane: laneFor(event),
        atMs: at,
        kind: event.kind,
        label: labelOf(event),
        rolledBack: gone,
      });
    }
  }

  const lanes: Lane[] = [
    { id: USER_LANE, label: "You", kind: "user" },
    { id: AGENT_LANE, label: "Agent", kind: "agent" },
    ...[...toolLanes.values()].sort((a, b) => a.label.localeCompare(b.label)),
    { id: SYSTEM_LANE, label: "System", kind: "system" },
  ];

  const times = events.map(time).filter((t) => t > 0);
  const startMs = times.length ? Math.min(...times) : 0;
  // A session with one instant would divide by zero when scaling; give it a
  // second of width so a single event still lands somewhere sensible.
  const rawEnd = times.length ? Math.max(...times) : 0;
  const endMs = rawEnd > startMs ? rawEnd : startMs + 1000;

  return {
    lanes,
    marks,
    spans,
    checkpoints,
    turns,
    rejected,
    startMs,
    endMs,
    lastSeq: events.length ? events[events.length - 1]!.seq : 0,
  };
}

// ── Flow view ─────────────────────────────────────────────────────────────────

export interface FlowMessage {
  seq: number;
  atMs: number;
  from: string;
  to: string;
  label: string;
  detail: string;
  status: "ok" | "failed" | "pending";
  rolledBack: boolean;
}

/**
 * Interactions between participants, as a sequence diagram would draw them.
 *
 * A timeline answers "when and how long"; this answers "who asked whom, and
 * what came back" — the shape of the agent's reasoning rather than its
 * duration.
 */
export function buildFlow(events: StudioEvent[], model: TimelineModel): FlowMessage[] {
  const rolled = new Set<string>();
  for (const branch of model.rejected) for (const id of branch.eventIds) rolled.add(id);

  const laneOfCall = new Map<string, string>();
  const messages: FlowMessage[] = [];

  for (const event of events) {
    const p = event.payload ?? {};
    const at = time(event);
    const gone = rolled.has(event.id);

    switch (event.kind) {
      case "user.message":
        messages.push({
          seq: event.seq,
          atMs: at,
          from: USER_LANE,
          to: AGENT_LANE,
          label: "prompt",
          detail: str(p.text) || flatten(p.parts),
          status: "ok",
          rolledBack: gone,
        });
        break;

      case "assistant.message": {
        const text = str(p.content);
        if (!text) break; // a tool-call-only message is drawn by its calls
        messages.push({
          seq: event.seq,
          atMs: at,
          from: AGENT_LANE,
          to: USER_LANE,
          label: "reply",
          detail: text,
          status: "ok",
          rolledBack: gone,
        });
        break;
      }

      case "tool.call.proposed": {
        const name = str(p.name) || "tool";
        const lane = `tool:${name}`;
        laneOfCall.set(str(p.tool_call_id), lane);
        messages.push({
          seq: event.seq,
          atMs: at,
          from: AGENT_LANE,
          to: lane,
          label: name,
          detail: str(p.arguments),
          status: "pending",
          rolledBack: gone,
        });
        break;
      }

      case "tool.result": {
        const lane = laneOfCall.get(str(p.tool_call_id));
        if (!lane) break;
        const failed = Boolean(p.error);
        messages.push({
          seq: event.seq,
          atMs: at,
          from: lane,
          to: AGENT_LANE,
          label: failed ? "error" : "result",
          detail: failed ? str(p.error) : compact(p.result),
          status: failed ? "failed" : "ok",
          rolledBack: gone,
        });
        break;
      }

      case "permission.request":
        messages.push({
          seq: event.seq,
          atMs: at,
          from: AGENT_LANE,
          to: USER_LANE,
          label: "may I?",
          detail: str(p.tool),
          status: "pending",
          rolledBack: gone,
        });
        break;

      case "permission.decision":
        messages.push({
          seq: event.seq,
          atMs: at,
          from: USER_LANE,
          to: AGENT_LANE,
          label: p.approve === true ? "allowed" : "denied",
          detail: str(p.reason),
          status: p.approve === true ? "ok" : "failed",
          rolledBack: gone,
        });
        break;

      default:
        break;
    }
  }
  return messages;
}

function flatten(parts: unknown): string {
  if (!Array.isArray(parts)) return "";
  return parts
    .map((p) =>
      p && typeof p === "object" && "text" in p ? str((p as { text: unknown }).text) : "",
    )
    .filter(Boolean)
    .join(" ");
}

function compact(value: unknown): string {
  if (value === undefined || value === null) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 160 ? `${text.slice(0, 160)}…` : text;
}
