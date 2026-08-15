import { describe, expect, it } from "vitest";
import { buildFlow, buildTimeline } from "./timeline";
import type { StudioEvent } from "./types";

let seq = 0;
const at = (offsetMs: number) =>
  new Date(Date.parse("2026-08-15T10:00:00.000Z") + offsetMs).toISOString();

const ev = (
  kind: string,
  payload: Record<string, unknown> = {},
  offsetMs = 0,
  id?: string,
): StudioEvent => ({
  seq: ++seq,
  id: id ?? `e${seq}`,
  ts: at(offsetMs),
  kind,
  payload,
});

const reset = () => {
  seq = 0;
};

describe("lanes", () => {
  it("gives every tool its own lane, with people and system pinned", () => {
    reset();
    const model = buildTimeline([
      ev("user.message", { text: "go" }),
      ev("tool.call.proposed", { tool_call_id: "1", name: "read_file" }),
      ev("tool.call.proposed", { tool_call_id: "2", name: "bash" }),
      ev("tool.call.proposed", { tool_call_id: "3", name: "read_file" }),
    ]);
    expect(model.lanes.map((l) => l.id)).toEqual([
      "user",
      "agent",
      "tool:bash",
      "tool:read_file",
      "system",
    ]);
  });
});

describe("tool calls as spans", () => {
  it("brackets a call between its proposal and its result", () => {
    reset();
    const model = buildTimeline([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }, 0),
      ev("tool.result", { tool_call_id: "c1", result: {} }, 2500),
    ]);
    expect(model.spans).toHaveLength(1);
    expect(model.spans[0]!.endMs - model.spans[0]!.startMs).toBe(2500);
    expect(model.spans[0]!.status).toBe("done");
  });

  it("keeps an unfinished call open rather than dropping it", () => {
    reset();
    const model = buildTimeline([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }),
    ]);
    expect(model.spans[0]!.status).toBe("running");
  });

  it("marks a failure so it can be drawn differently", () => {
    reset();
    const model = buildTimeline([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }, 0),
      ev("tool.result", { tool_call_id: "c1", error: "exit 1" }, 100),
    ]);
    expect(model.spans[0]!.status).toBe("failed");
  });

  it("keeps overlapping calls separate, which is what shows concurrency", () => {
    reset();
    const model = buildTimeline([
      ev("tool.call.proposed", { tool_call_id: "a", name: "grep" }, 0),
      ev("tool.call.proposed", { tool_call_id: "b", name: "glob" }, 10),
      ev("tool.result", { tool_call_id: "b", result: {} }, 400),
      ev("tool.result", { tool_call_id: "a", result: {} }, 900),
    ]);
    const [first, second] = model.spans;
    expect(model.spans).toHaveLength(2);
    // They overlap in time: b starts before a ends.
    expect(second!.startMs).toBeLessThan(first!.endMs);
  });
});

describe("turns and checkpoints", () => {
  it("pairs cycle start and end into a turn", () => {
    reset();
    const model = buildTimeline([
      ev("agent.cycle.start", {}, 0),
      ev("agent.cycle.end", {}, 5000),
    ]);
    expect(model.turns).toHaveLength(1);
    expect(model.turns[0]!.complete).toBe(true);
    expect(model.turns[0]!.endMs - model.turns[0]!.startMs).toBe(5000);
  });

  it("leaves a running turn open", () => {
    reset();
    const model = buildTimeline([ev("agent.cycle.start")]);
    expect(model.turns[0]!.complete).toBe(false);
  });

  it("numbers each checkpoint with the turn it opens", () => {
    reset();
    const model = buildTimeline([
      ev("system.checkpoint", {}, 0),
      ev("agent.cycle.start", {}, 10),
      ev("agent.cycle.end", {}, 20),
      ev("system.checkpoint", {}, 30),
      ev("agent.cycle.start", {}, 40),
    ]);
    expect(model.checkpoints.map((c) => c.turn)).toEqual([1, 2]);
  });
});

describe("rewound history", () => {
  it("marks discarded events without removing them from the picture", () => {
    reset();
    const model = buildTimeline([
      ev("system.checkpoint", {}, 0, "cp"),
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }, 10, "t1"),
      ev("tool.result", { tool_call_id: "c1", result: {} }, 20, "t2"),
      ev("system.rollback", { rejected_event_ids: ["cp", "t1", "t2"] }, 30),
    ]);
    // A rejected branch is evidence: still drawn, but visibly set aside.
    expect(model.spans[0]!.rolledBack).toBe(true);
    expect(model.checkpoints[0]!.used).toBe(true);
    expect(model.rejected[0]!.count).toBe(3);
  });
});

describe("edge cases", () => {
  it("does not divide by zero when everything happened at once", () => {
    reset();
    const model = buildTimeline([ev("user.message", { text: "hi" }, 0)]);
    expect(model.endMs).toBeGreaterThan(model.startMs);
  });

  it("handles an empty session", () => {
    const model = buildTimeline([]);
    expect(model.lanes.length).toBeGreaterThan(0);
    expect(model.lastSeq).toBe(0);
    expect(model.endMs).toBeGreaterThanOrEqual(model.startMs);
  });

  it("ignores streaming deltas, which would swamp the picture", () => {
    reset();
    const model = buildTimeline([
      ev("assistant.delta", { content: "a" }),
      ev("assistant.delta", { content: "b" }),
      ev("assistant.message", { content: "ab" }),
    ]);
    expect(model.marks).toHaveLength(1);
  });
});

describe("the flow view", () => {
  it("routes each interaction between the right participants", () => {
    reset();
    const events = [
      ev("user.message", { text: "fix it" }),
      ev("tool.call.proposed", { tool_call_id: "c1", name: "read_file" }),
      ev("tool.result", { tool_call_id: "c1", result: { ok: 1 } }),
      ev("permission.request", { request_id: "r", tool: "bash" }),
      ev("permission.decision", { request_id: "r", approve: true }),
      ev("assistant.message", { content: "done" }),
    ];
    const flow = buildFlow(events, buildTimeline(events));
    expect(flow.map((m) => [m.from, m.to, m.label])).toEqual([
      ["user", "agent", "prompt"],
      ["agent", "tool:read_file", "read_file"],
      ["tool:read_file", "agent", "result"],
      ["agent", "user", "may I?"],
      ["user", "agent", "allowed"],
      ["agent", "user", "reply"],
    ]);
  });

  it("does not draw a message for a tool-call-only turn", () => {
    reset();
    const events = [ev("assistant.message", { content: null, tool_calls: [{}] })];
    expect(buildFlow(events, buildTimeline(events))).toHaveLength(0);
  });

  it("marks a rejected result so it can be dimmed", () => {
    reset();
    const events = [
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }, 0, "p"),
      ev("tool.result", { tool_call_id: "c1", error: "boom" }, 10, "r"),
      ev("system.rollback", { rejected_event_ids: ["p", "r"] }, 20),
    ];
    const flow = buildFlow(events, buildTimeline(events));
    expect(flow.every((m) => m.rolledBack)).toBe(true);
    expect(flow[1]!.status).toBe("failed");
  });
});
