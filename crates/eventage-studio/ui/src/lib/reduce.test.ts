import { describe, expect, it } from "vitest";
import { reduce } from "./reduce";
import type { AssistantItem, StudioEvent, ToolItem } from "./types";

let seq = 0;
const ev = (
  kind: string,
  payload: Record<string, unknown> = {},
  extra: Partial<StudioEvent> = {},
): StudioEvent => ({
  seq: ++seq,
  id: extra.id ?? `e${seq}`,
  ts: extra.ts ?? new Date(1_700_000_000_000 + seq * 1000).toISOString(),
  kind,
  payload,
  ...(extra.meta ? { meta: extra.meta } : {}),
});

const only = <T extends { type: string }>(items: T[], type: string) =>
  items.filter((i) => i.type === type);

describe("a turn marker that overtakes the reply it follows", () => {
  it("keeps one bubble when studio.turn.ended arrives before assistant.message", () => {
    // Captured from a live turn. Studio's turn task and the bridge draining
    // the bus are two producers writing to one feed, and the task won:
    //
    //   21 assistant.delta
    //   22 studio.turn.ended     <- the turn task
    //   23 assistant.message     <- the bus, one hop behind
    //   24 agent.cycle.end
    //
    // Closing the bubble at 22 left 23 with nothing to attach to, so it
    // started a second one with the same full text.
    const chat = reduce([
      ev("user.message", { text: "hi" }),
      ev("assistant.delta", { content: "Hi! How can " }),
      ev("assistant.delta", { content: "I help you today?" }),
      ev("studio.turn.ended", { reason: "end_turn" }),
      ev("assistant.message", { content: "Hi! How can I help you today?" }),
      ev("agent.cycle.end", {}),
    ]);

    const replies = chat.items.filter((i) => i.type === "assistant");
    expect(replies).toHaveLength(1);
    expect((replies[0] as { text: string }).text).toBe(
      "Hi! How can I help you today?",
    );
    expect((replies[0] as { streaming: boolean }).streaming).toBe(false);
    expect(chat.running).toBe(false);
  });

  it("still stops the caret when nothing follows the marker", () => {
    // The other half: a turn that ends without a final message must not
    // leave a bubble blinking forever.
    const chat = reduce([
      ev("user.message", { text: "hi" }),
      ev("assistant.delta", { content: "partway th" }),
      ev("studio.turn.failed", { error: "connection reset" }),
    ]);
    const replies = chat.items.filter((i) => i.type === "assistant");
    expect(replies).toHaveLength(1);
    expect((replies[0] as { streaming: boolean }).streaming).toBe(false);
  });
});

describe("streaming assistant messages", () => {
  it("joins deltas into one bubble and closes it on the final message", () => {
    const state = reduce([
      ev("assistant.delta", { content: "Hel" }),
      ev("assistant.delta", { content: "lo" }),
      ev("assistant.message", { content: "Hello" }),
    ]);
    const bubbles = only(state.items, "assistant") as AssistantItem[];
    expect(bubbles).toHaveLength(1);
    expect(bubbles[0]!.text).toBe("Hello");
    expect(bubbles[0]!.streaming).toBe(false);
  });

  it("keeps thinking separate from the answer", () => {
    const state = reduce([
      ev("assistant.delta", { reasoning_content: "let me check" }),
      ev("assistant.delta", { content: "The answer" }),
    ]);
    const bubble = (only(state.items, "assistant") as AssistantItem[])[0]!;
    expect(bubble.thinking).toBe("let me check");
    expect(bubble.text).toBe("The answer");
    expect(bubble.streaming).toBe(true);
  });

  it("prefers the final message when streaming dropped a chunk", () => {
    // The provider streamed a truncated body but sent the whole thing at the
    // end; the transcript must show the complete answer.
    const state = reduce([
      ev("assistant.delta", { content: "partial" }),
      ev("assistant.message", { content: "the complete answer" }),
    ]);
    const bubble = (only(state.items, "assistant") as AssistantItem[])[0]!;
    expect(bubble.text).toBe("the complete answer");
  });

  it("renders a non-streaming provider's single message", () => {
    const state = reduce([ev("assistant.message", { content: "all at once" })]);
    expect((only(state.items, "assistant") as AssistantItem[])[0]!.text).toBe(
      "all at once",
    );
  });

  it("drops the empty bubble of a tool-call-only turn", () => {
    const state = reduce([
      ev("assistant.message", {
        content: null,
        tool_calls: [{ id: "c1" }],
      }),
    ]);
    expect(only(state.items, "assistant")).toHaveLength(0);
  });

  it("starts a new bubble after a tool call rather than appending", () => {
    const state = reduce([
      ev("assistant.delta", { content: "first" }),
      ev("tool.call.proposed", { tool_call_id: "c1", name: "read_file" }),
      ev("tool.result", { tool_call_id: "c1", result: {} }),
      ev("assistant.delta", { content: "second" }),
    ]);
    const bubbles = only(state.items, "assistant") as AssistantItem[];
    expect(bubbles.map((b) => b.text)).toEqual(["first", "second"]);
  });
});

describe("tool calls", () => {
  it("pairs a result with its call and times it", () => {
    const state = reduce([
      ev(
        "tool.call.proposed",
        { tool_call_id: "c1", name: "read_file", arguments: '{"path":"a.rs"}' },
        { ts: "2026-08-15T00:00:00.000Z" },
      ),
      ev(
        "tool.result",
        { tool_call_id: "c1", result: { ok: true } },
        { ts: "2026-08-15T00:00:02.500Z" },
      ),
    ]);
    const tool = (only(state.items, "tool") as ToolItem[])[0]!;
    expect(tool.status).toBe("done");
    expect(tool.durationMs).toBe(2500);
    expect(tool.args).toEqual({ path: "a.rs" });
    expect(tool.title).toContain("a.rs");
  });

  it("marks a failure and surfaces the error", () => {
    const state = reduce([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }),
      ev("tool.result", { tool_call_id: "c1", error: "exit 1" }),
    ]);
    const tool = (only(state.items, "tool") as ToolItem[])[0]!;
    expect(tool.status).toBe("failed");
    expect(tool.error).toBe("exit 1");
  });

  it("leaves an unanswered call showing as running", () => {
    const state = reduce([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }),
    ]);
    expect((only(state.items, "tool") as ToolItem[])[0]!.status).toBe("running");
  });

  it("lifts a diff out of the result so the card can render it", () => {
    const state = reduce([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "edit_file" }),
      ev("tool.result", {
        tool_call_id: "c1",
        result: {
          _diff: { path: "src/a.rs", old_text: "a", new_text: "b" },
          _locations: [{ path: "src/a.rs", line: 3 }],
        },
      }),
    ]);
    const tool = (only(state.items, "tool") as ToolItem[])[0]!;
    expect(tool.diff).toEqual({
      path: "src/a.rs",
      old_text: "a",
      new_text: "b",
    });
    expect(tool.locations).toEqual([{ path: "src/a.rs", line: 3 }]);
  });

  it("treats a new file (no old text) as such rather than as empty", () => {
    const state = reduce([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "write_file" }),
      ev("tool.result", {
        tool_call_id: "c1",
        result: { _diff: { path: "new.rs", new_text: "hi" } },
      }),
    ]);
    expect((only(state.items, "tool") as ToolItem[])[0]!.diff!.old_text).toBe(
      null,
    );
  });

  it("survives malformed tool arguments instead of throwing", () => {
    const state = reduce([
      ev("tool.call.proposed", {
        tool_call_id: "c1",
        name: "bash",
        arguments: "{not json",
      }),
    ]);
    expect((only(state.items, "tool") as ToolItem[])[0]!.args).toBe("{not json");
  });
});

describe("permissions", () => {
  it("shows a request as pending until it is answered", () => {
    const state = reduce([
      ev("permission.request", { request_id: "r1", tool: "bash" }),
    ]);
    expect(state.pendingPermissions).toHaveLength(1);
  });

  it("resolves the matching request and clears the prompt", () => {
    const state = reduce([
      ev("permission.request", { request_id: "r1", tool: "bash" }),
      ev("permission.decision", { request_id: "r1", approve: true }),
    ]);
    expect(state.pendingPermissions).toHaveLength(0);
    expect(only(state.items, "permission")[0]).toMatchObject({
      status: "approved",
    });
  });

  it("records a rejection with its reason", () => {
    const state = reduce([
      ev("permission.request", { request_id: "r1", tool: "bash" }),
      ev("permission.decision", {
        request_id: "r1",
        approve: false,
        reason: "too risky",
      }),
    ]);
    expect(only(state.items, "permission")[0]).toMatchObject({
      status: "denied",
      reason: "too risky",
    });
  });

  it("does not resolve a different request", () => {
    const state = reduce([
      ev("permission.request", { request_id: "r1", tool: "bash" }),
      ev("permission.decision", { request_id: "other", approve: true }),
    ]);
    expect(state.pendingPermissions).toHaveLength(1);
  });
});

describe("turn state", () => {
  it("is running between cycle start and end", () => {
    expect(reduce([ev("agent.cycle.start")]).running).toBe(true);
    expect(
      reduce([ev("agent.cycle.start"), ev("agent.cycle.end")]).running,
    ).toBe(false);
  });

  it("stops running when the turn is interrupted", () => {
    const state = reduce([
      ev("agent.cycle.start"),
      ev("studio.turn.interrupted"),
    ]);
    expect(state.running).toBe(false);
    expect(only(state.items, "notice")[0]).toMatchObject({ level: "warn" });
  });

  it("stops running and reports the error when a turn fails", () => {
    const state = reduce([
      ev("agent.cycle.start"),
      ev("studio.turn.failed", { error: "provider timed out" }),
    ]);
    expect(state.running).toBe(false);
    expect(only(state.items, "notice")[0]).toMatchObject({
      level: "error",
      detail: "provider timed out",
    });
  });

  it("takes turn duration from the harness's own measurement", () => {
    const state = reduce([
      ev("agent.cycle.start"),
      ev("agent.cycle.end", {}, { meta: { elapsed_ms: 4321 } }),
    ]);
    expect(state.stats.lastTurnMs).toBe(4321);
  });
});

describe("token accounting", () => {
  it("totals usage across messages", () => {
    const state = reduce([
      ev(
        "assistant.message",
        { content: "a" },
        {
          meta: {
            llm_input_tokens: 100,
            llm_output_tokens: 20,
            llm_cached_input_tokens: 80,
          },
        },
      ),
      ev(
        "assistant.message",
        { content: "b" },
        { meta: { llm_input_tokens: 150, llm_output_tokens: 30 } },
      ),
    ]);
    expect(state.stats).toMatchObject({
      inputTokens: 250,
      outputTokens: 50,
      cachedTokens: 80,
    });
  });
});

describe("rewind", () => {
  it("removes rolled-back events from every view, not just the transcript", () => {
    const state = reduce([
      ev("user.message", { text: "first" }, { id: "u1" }),
      ev("agent.cycle.end", {}, { id: "end1" }),
      ev("user.message", { text: "second" }, { id: "u2" }),
      ev("tool.call.proposed", { tool_call_id: "c9", name: "bash" }, { id: "t2" }),
      ev("agent.cycle.end", {}, { id: "end2" }),
      ev("system.rollback", {
        rejected_event_ids: ["u2", "t2", "end2"],
      }),
    ]);

    expect(only(state.items, "user")).toHaveLength(1);
    expect(only(state.items, "tool")).toHaveLength(0);
    // The counters must forget the turn too, not merely hide its bubbles.
    expect(state.stats.turns).toBe(1);
    expect(state.stats.toolCalls).toBe(0);
    expect(state.rolledBack.has("u2")).toBe(true);
  });
});

describe("plans", () => {
  it("reads a plan from a local tool result", () => {
    const state = reduce([
      ev("tool.call.proposed", { tool_call_id: "c1", name: "plan" }),
      ev("tool.result", {
        tool_call_id: "c1",
        result: {
          _plan: [
            { content: "read the code", status: "completed" },
            { content: "fix it", status: "in_progress" },
          ],
        },
      }),
    ]);
    expect(state.plan).toHaveLength(2);
    expect(state.plan[1]!.status).toBe("in_progress");
  });

  it("reads the same plan from an ACP update", () => {
    const state = reduce([
      ev("studio.plan", { entries: [{ content: "step", status: "pending" }] }),
    ]);
    expect(state.plan).toEqual([{ content: "step", status: "pending" }]);
  });

  it("replaces the plan rather than accumulating revisions", () => {
    const state = reduce([
      ev("studio.plan", { entries: [{ content: "old", status: "pending" }] }),
      ev("studio.plan", { entries: [{ content: "new", status: "pending" }] }),
    ]);
    expect(state.plan).toHaveLength(1);
    expect(state.plan[0]!.content).toBe("new");
  });
});

describe("user messages", () => {
  it("reads text out of multimodal parts", () => {
    const state = reduce([
      ev("user.message", {
        parts: [
          { type: "text", text: "what is this?" },
          {
            type: "image",
            source: { kind: "base64", media_type: "image/png", data: "AAA" },
          },
        ],
      }),
    ]);
    expect(only(state.items, "user")[0]).toMatchObject({
      text: "what is this?",
      images: ["data:image/png;base64,AAA"],
    });
  });

  it("reads a plain text payload too", () => {
    const state = reduce([ev("user.message", { text: "hello" })]);
    expect(only(state.items, "user")[0]).toMatchObject({ text: "hello" });
  });
});

describe("harness notices", () => {
  it("surfaces a loop-detection hint", () => {
    const state = reduce([ev("agent.stuck", { hint: "you repeated that" })]);
    expect(only(state.items, "notice")[0]).toMatchObject({
      level: "warn",
      detail: "you repeated that",
    });
  });

  it("says when the step budget cut a turn short", () => {
    const state = reduce([
      ev("assistant.message", { content: "…", finalized_due_to: "max_steps" }),
    ]);
    expect(only(state.items, "notice")).toHaveLength(1);
  });

  it("reports an exhausted token budget as an error", () => {
    expect(
      only(reduce([ev("budget.exhausted")]).items, "notice")[0],
    ).toMatchObject({ level: "error" });
  });

  it("ignores kinds it does not know rather than breaking the view", () => {
    const state = reduce([
      ev("some.future.kind", { whatever: 1 }),
      ev("user.message", { text: "still here" }),
    ]);
    expect(state.items).toHaveLength(1);
  });

  it("handles an empty stream", () => {
    const state = reduce([]);
    expect(state.items).toEqual([]);
    expect(state.running).toBe(false);
  });
});
