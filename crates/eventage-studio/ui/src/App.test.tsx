/**
 * The app, mounted for real against a stubbed server.
 *
 * The reducer is covered on its own; this checks the layer above it — that
 * the shell renders, that events become the right things on screen, and that
 * the controls call back into the API. It runs in jsdom, so it verifies
 * structure and behaviour rather than pixels.
 *
 * @vitest-environment jsdom
 */

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { AppInfo, SessionInfo, StudioEvent } from "./lib/types";

const APP_INFO: AppInfo = {
  backend: "local",
  backend_detail: "eventage-code, hosted in this process",
  model: "test-model",
  provider: "Test",
  default_cwd: "/repo",
  modes: [
    { id: "ask", label: "Ask every time", description: "Approve each edit" },
    { id: "auto", label: "Auto-accept edits", description: "Edits apply" },
  ],
  version: "0.2.0",
  full_trace: true,
};

const SESSION: SessionInfo = {
  id: "s1",
  cwd: "/repo",
  mode: "ask",
  title: "fix the parser",
  created_at: "2026-08-15T00:00:00Z",
  running: false,
  turns: 1,
};

let seq = 0;
const ev = (kind: string, payload: Record<string, unknown> = {}): StudioEvent => ({
  seq: ++seq,
  id: `e${seq}`,
  ts: "2026-08-15T00:00:00.000Z",
  kind,
  payload,
});

/** Requests the app made, so tests can assert on them. */
let calls: { url: string; method: string; body: unknown }[];
let events: StudioEvent[];

/** `EventSource` does not exist in jsdom; the app only needs it to connect. */
class StubEventSource {
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  static latest: StubEventSource | null = null;
  constructor(public url: string) {
    StubEventSource.latest = this;
    queueMicrotask(() => this.onopen?.());
  }
  close() {}
  /** Push an event as the server would. */
  emit(event: StudioEvent) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
}

beforeEach(() => {
  seq = 0;
  calls = [];
  events = [];
  StubEventSource.latest = null;
  vi.stubGlobal("EventSource", StubEventSource);
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method, body });

    const json = (value: unknown) =>
      new Response(JSON.stringify(value), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    if (url.startsWith("/api/app")) return json(APP_INFO);
    // No `api_key` field: the server never sends one back, and a fixture that
    // did would let a leak pass the test that exists to catch it.
    if (url.startsWith("/api/model"))
      return json({
        provider: "anthropic",
        model: "test-model",
        base_url: "",
        source: "manual",
        claude_settings_available: true,
        has_key: true,
        key_remembered: false,
        providers: [
          { id: "anthropic", label: "Anthropic", endpoint_hint: "hint" },
          { id: "openai-chat", label: "OpenAI-compatible", endpoint_hint: "hint" },
        ],
      });
    if (url.startsWith("/api/sessions?") || url === "/api/sessions") {
      if (method === "POST") return json(SESSION);
      return json({ open: [SESSION], stored: [] });
    }
    if (url.includes("/events")) return json(events);
    // Mirror the real server: a prompt is accepted asynchronously and answers
    // 202 with no body at all. Answering 204 here once hid a client bug that
    // broke every first message.
    if (url.includes("/prompt")) return new Response(null, { status: 202 });
    if (url.includes("/rewind")) return json({ remaining: 0 });
    return new Response(null, { status: 204 });
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const mount = async () => {
  render(<App />);
  await screen.findByText("Eventage Studio");
  await waitFor(() => expect(StubEventSource.latest).not.toBeNull());
};

/**
 * Set an input's value the way a user would.
 *
 * React tracks the last value it wrote, so assigning `.value` directly is
 * invisible to it; going through the native setter is what makes the change
 * event carry.
 */
const type = async (el: HTMLInputElement | HTMLTextAreaElement, value: string) => {
  const prototype =
    el instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(el, value);
  await act(async () => {
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
};

const push = async (...list: StudioEvent[]) => {
  await act(async () => {
    for (const event of list) StubEventSource.latest!.emit(event);
  });
};

describe("the app shell", () => {
  it("renders the panes and connects to a session", async () => {
    await mount();
    expect(screen.getByRole("navigation")).toBeTruthy(); // sessions sidebar
    expect(document.querySelector(".trace")).toBeTruthy();
    expect(screen.getByPlaceholderText(/Ask for a change/)).toBeTruthy();
    // The title bar shows the workspace name; the full path is the tooltip.
    expect(screen.getByTitle("/repo").textContent).toContain("repo");
    expect(StubEventSource.latest!.url).toContain("/api/sessions/s1/stream");
  });

  it("offers something to do when the transcript is empty", async () => {
    await mount();
    expect(screen.getByText(/What should we work on/)).toBeTruthy();
    expect(
      screen.getByText(/Explain how this project is structured/),
    ).toBeTruthy();
  });
});

describe("the transcript", () => {
  it("shows a user message, streamed reply and finished tool call", async () => {
    await mount();
    await push(
      ev("user.message", { text: "fix the parser" }),
      ev("assistant.delta", { content: "Looking at " }),
      ev("assistant.delta", { content: "the code." }),
      ev("tool.call.proposed", {
        tool_call_id: "c1",
        name: "read_file",
        arguments: '{"path":"src/parse.rs"}',
      }),
      ev("tool.result", { tool_call_id: "c1", result: { ok: true } }),
    );

    // Scoped to the transcript: the trace pane shows the same events, which
    // is the whole point of it.
    const stream = document.querySelector(".stream") as HTMLElement;
    expect(within(stream).getByText("fix the parser")).toBeTruthy();
    expect(within(stream).getByText("Looking at the code.")).toBeTruthy();
    expect(within(stream).getByText(/read_file · src\/parse\.rs/)).toBeTruthy();
  });

  it("renders a diff with its added and removed counts", async () => {
    await mount();
    await push(
      ev("tool.call.proposed", { tool_call_id: "c1", name: "edit_file" }),
      ev("tool.result", {
        tool_call_id: "c1",
        result: {
          _diff: {
            path: "src/a.rs",
            old_text: "fn add(a: i32, b: i32) -> i32 { a - b }",
            new_text: "fn add(a: i32, b: i32) -> i32 { a + b }",
          },
        },
      }),
    );

    // The card opens on click; the diff lives inside it.
    const stream = document.querySelector(".stream") as HTMLElement;
    const card = within(stream)
      .getByText(/edit_file/)
      .closest(".tool") as HTMLElement;
    await act(async () => {
      within(card).getByRole("button").click();
    });
    expect(within(card).getByText("src/a.rs")).toBeTruthy();
    expect(within(card).getByText("+1")).toBeTruthy();
    expect(within(card).getByText("−1")).toBeTruthy();
  });

  it("opens a failed tool card without being asked", async () => {
    await mount();
    await push(
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }),
      ev("tool.result", { tool_call_id: "c1", error: "exit code 101" }),
    );
    expect(screen.getByText("exit code 101")).toBeTruthy();
  });

  it("shows a plan with its progress", async () => {
    await mount();
    await push(
      ev("studio.plan", {
        entries: [
          { content: "read the code", status: "completed" },
          { content: "fix the bug", status: "in_progress" },
        ],
      }),
    );
    const stream = document.querySelector(".stream") as HTMLElement;
    expect(within(stream).getByText("Plan · 1/2")).toBeTruthy();
    expect(within(stream).getByText("fix the bug")).toBeTruthy();
  });
});

describe("permission prompts", () => {
  it("asks, then sends the answer back", async () => {
    await mount();
    await push(
      ev("permission.request", {
        request_id: "r1",
        tool: "bash",
        arguments: { command: "rm -rf build" },
      }),
    );
    expect(screen.getByText("Approve this action?")).toBeTruthy();

    await act(async () => {
      screen.getByText("Allow once").click();
    });

    const answer = calls.find((c) => c.url.includes("/permission"));
    expect(answer?.body).toMatchObject({ request_id: "r1", approve: true });
  });

  it("opens model settings from the chip, and never shows the key", async () => {
    // The chip already names the model, so it is where someone goes to change
    // it. The dialog is given `has_key` and never the key itself.
    await mount();
    await act(async () => {
      screen.getByTitle(/click to change/).click();
    });
    // "Model" is both the dialog heading and a field label; ask for the one
    // that means the dialog opened.
    expect(await screen.findByRole("heading", { name: "Model" })).toBeTruthy();
    // The field is empty with a placeholder, because a key it was never given
    // cannot be pre-filled — and leaving it blank must keep the current one.
    const key = screen.getByPlaceholderText(/leave blank to keep it|no key configured/);
    expect((key as HTMLInputElement).value).toBe("");
    expect((key as HTMLInputElement).type).toBe("password");

    // And the file can be chosen as the source instead of typing anything.
    expect(screen.getByText(/~\/\.claude\/settings\.json/)).toBeTruthy();
  });

  it("shows how many rewound attempts are still steering the agent", async () => {
    // A rolled-back attempt leaves the transcript but is described to the
    // model on every subsequent request. Nothing else on screen says so, so
    // without this the constraint is invisible to the person being
    // constrained by it.
    await mount();
    expect(screen.queryByText(/rewound/)).toBeNull();

    await push(ev("user.message", { text: "go" }));
    await push(ev("system.rollback", { to_event_id: "e1" }));
    expect(screen.getByText("1 rewound")).toBeTruthy();

    await push(ev("system.rollback", { to_event_id: "e2" }));
    expect(screen.getByText("2 rewound")).toBeTruthy();
  });

  it("offers a standing approval for the same call", async () => {
    await mount();
    await push(ev("permission.request", { request_id: "r1", tool: "bash" }));
    await act(async () => {
      screen.getByText("Always allow this call").click();
    });
    expect(calls.find((c) => c.url.includes("/permission"))?.body).toMatchObject(
      { approve: true, always: true },
    );
  });

  it("stops asking once the request is answered", async () => {
    await mount();
    await push(
      ev("permission.request", { request_id: "r1", tool: "bash" }),
      ev("permission.decision", { request_id: "r1", approve: false, reason: "no" }),
    );
    expect(screen.queryByText("Allow once")).toBeNull();
    expect(screen.getByText(/Rejected — no/)).toBeTruthy();
  });
});

describe("sending a message", () => {
  it("accepts a 202 with no body without inventing an error", async () => {
    await mount();
    // The server returns 202 Accepted and an empty body; parsing that as JSON
    // throws "Unexpected end of JSON input", which used to surface as a toast
    // on every single message.
    await act(async () => {
      screen
        .getByText(/Explain how this project is structured/)
        .click();
    });
    expect(calls.some((c) => c.url.includes("/prompt"))).toBe(true);
    expect(document.querySelector(".toast")).toBeNull();
  });

  it("still shows the server's reason when a send is rejected", async () => {
    await mount();
    vi.stubGlobal("fetch", async () =>
      new Response(JSON.stringify({ error: "this session is already working" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await act(async () => {
      screen.getByText(/Find and fix any failing tests/).click();
    });
    await waitFor(() =>
      expect(screen.getByText("this session is already working")).toBeTruthy(),
    );
  });
});

describe("setup problems", () => {
  it("says up front when no API key was found", async () => {
    const hint = "No API key found, so Studio is pointed at http://localhost:11434/v1.";
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const json = (value: unknown) =>
        new Response(JSON.stringify(value), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      if (url.startsWith("/api/app"))
        return json({ ...APP_INFO, credentials_hint: hint });
      if (url.startsWith("/api/sessions?") || url === "/api/sessions")
        return method === "POST" ? json(SESSION) : json({ open: [SESSION], stored: [] });
      if (url.includes("/events")) return json([]);
      return new Response(null, { status: 204 });
    });

    render(<App />);
    // A missing key is a setup problem, not a transient one, so it belongs in
    // a banner that stays rather than a toast that fades.
    await waitFor(() => expect(document.querySelector(".setup-banner")).toBeTruthy());
    expect(document.querySelector(".setup-banner")!.textContent).toContain(hint);
  });
});

describe("the composer", () => {
  it("becomes a stop button while a turn runs", async () => {
    await mount();
    await push(ev("agent.cycle.start"));
    const stop = screen.getByText("■ Stop");

    await act(async () => {
      stop.click();
    });
    expect(calls.some((c) => c.url.includes("/interrupt"))).toBe(true);
  });

  it("goes back to sending when the turn ends", async () => {
    await mount();
    await push(ev("agent.cycle.start"), ev("agent.cycle.end"));
    expect(screen.getByText("Send")).toBeTruthy();
  });
});

describe("time travel", () => {
  const session = async () => {
    await mount();
    await push(
      ev("system.checkpoint", {}),
      ev("user.message", { text: "first prompt" }),
      ev("agent.cycle.start", {}),
      ev("tool.call.proposed", { tool_call_id: "c1", name: "read_file" }),
      ev("tool.result", { tool_call_id: "c1", result: {} }),
      ev("assistant.message", { content: "all done" }),
      ev("agent.cycle.end", {}),
    );
  };

  it("rolls the transcript back to the playhead, not just the trace", async () => {
    await session();
    const stream = () => document.querySelector(".stream") as HTMLElement;
    expect(within(stream()).getByText("all done")).toBeTruthy();

    // Scrub back to before the reply arrived.
    const scrub = document.querySelector(".scrub") as HTMLInputElement;
    await type(scrub, "4");

    expect(within(stream()).queryByText("all done")).toBeNull();
    expect(within(stream()).getByText("first prompt")).toBeTruthy();
  });

  it("says it is replaying and offers the way back", async () => {
    await session();
    const scrub = document.querySelector(".scrub") as HTMLInputElement;
    await type(scrub, "3");

    const banner = document.querySelector(".replay-banner") as HTMLElement;
    expect(banner).toBeTruthy();
    expect(banner.textContent).toContain("Replaying history");

    await act(async () => {
      within(banner).getByText("Return to live").click();
    });
    expect(document.querySelector(".replay-banner")).toBeNull();
    expect(
      within(document.querySelector(".stream") as HTMLElement).getByText("all done"),
    ).toBeTruthy();
  });

  it("keeps the whole session in the trace while the transcript is rewound", async () => {
    await session();
    const scrub = document.querySelector(".scrub") as HTMLInputElement;
    await type(scrub, "2");
    const trace = document.querySelector(".trace") as HTMLElement;
    await act(async () => {
      within(trace).getByText("Events").click();
    });
    // Later events are still listed, marked as ahead of the playhead.
    expect(within(trace).getByText("assistant.message")).toBeTruthy();
    expect(trace.querySelectorAll(".trace-row.future").length).toBeGreaterThan(0);
  });

  it("returns to live when a new message is sent", async () => {
    await session();
    const scrub = document.querySelector(".scrub") as HTMLInputElement;
    await type(scrub, "2");
    expect(document.querySelector(".replay-banner")).toBeTruthy();

    const box = screen.getByPlaceholderText(/Ask for a change/) as HTMLTextAreaElement;
    await type(box, "next thing");
    await act(async () => {
      screen.getByText("Send").click();
    });
    expect(document.querySelector(".replay-banner")).toBeNull();
  });
});

describe("rewind", () => {
  const withTurn = async () => {
    await mount();
    await push(
      ev("system.checkpoint", {}),
      ev("user.message", { text: "make the change" }),
      ev("agent.cycle.start", {}),
      ev("tool.call.proposed", {
        tool_call_id: "c1",
        name: "edit_file",
        arguments: JSON.stringify({ path: "src/main.rs" }),
      }),
      ev("tool.result", { tool_call_id: "c1", result: {} }),
      ev("tool.call.proposed", {
        tool_call_id: "c2",
        name: "bash",
        arguments: JSON.stringify({ command: "cargo test" }),
      }),
      ev("tool.result", { tool_call_id: "c2", result: {} }),
      ev("assistant.message", { content: "changed it" }),
      ev("agent.cycle.end", {}),
    );
  };

  it("shows what will be discarded before doing it", async () => {
    await withTurn();
    await act(async () => {
      screen.getByText("↶ Rewind").click();
    });

    const dialog = document.querySelector(".picker.rewind") as HTMLElement;
    expect(dialog).toBeTruthy();
    // The point of the dialog: name the files, not just a count.
    expect(dialog.textContent).toContain("src/main.rs");
    expect(dialog.textContent).toContain("cargo test");
    expect(dialog.textContent).toContain("2 tool calls");
    expect(dialog.textContent).toContain("stay on disk");
  });

  it("rewinds to the chosen checkpoint, not merely one turn", async () => {
    await withTurn();
    await act(async () => {
      screen.getByText("↶ Rewind").click();
    });
    await act(async () => {
      screen.getByText("Rewind").click();
    });

    const call = calls.find((c) => c.url.includes("/rewind"));
    expect(call?.body).toMatchObject({ to: "e1" });
  });

  it("can be cancelled without touching the session", async () => {
    await withTurn();
    await act(async () => {
      screen.getByText("↶ Rewind").click();
    });
    await act(async () => {
      screen.getByText("Cancel").click();
    });
    expect(document.querySelector(".picker.rewind")).toBeNull();
    expect(calls.some((c) => c.url.includes("/rewind"))).toBe(false);
  });
});

describe("copying", () => {
  const clipboard: string[] = [];

  beforeEach(() => {
    clipboard.length = 0;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async (text: string) => void clipboard.push(text) },
    });
  });

  it("copies a reply", async () => {
    await mount();
    await push(ev("assistant.message", { content: "the answer is 42" }));
    await act(async () => {
      screen.getByTitle("Copy this reply").click();
    });
    expect(clipboard).toEqual(["the answer is 42"]);
  });

  it("copies a code block's text, not its highlighting markup", async () => {
    await mount();
    await push(
      ev("assistant.message", {
        content: "Try this:\n\n```rust\nfn main() { println!(\"hi\"); }\n```",
      }),
    );
    const button = document.querySelector(".copy-code") as HTMLButtonElement;
    expect(button).toBeTruthy();
    await act(async () => {
      button.click();
    });
    expect(clipboard[0]).toContain('println!("hi")');
    expect(clipboard[0]).not.toContain("<span");
  });

  it("copies a tool call's arguments as JSON", async () => {
    await mount();
    await push(
      ev("tool.call.proposed", {
        tool_call_id: "c1",
        name: "bash",
        arguments: JSON.stringify({ command: "cargo test" }),
      }),
      ev("tool.result", { tool_call_id: "c1", error: "exit 1" }),
    );
    // The card opens itself because the call failed.
    const card = document.querySelector(".tool") as HTMLElement;
    const copies = within(card).getAllByText("⧉");
    await act(async () => {
      copies[copies.length - 1]!.click();
    });
    expect(clipboard[0]).toContain("cargo test");
  });

  it("confirms that the copy happened", async () => {
    await mount();
    await push(ev("assistant.message", { content: "done" }));
    const button = screen.getByTitle("Copy this reply");
    expect(button.textContent).toContain("Copy");
    await act(async () => {
      button.click();
    });
    // Silent success is indistinguishable from a broken button.
    expect(button.textContent).toContain("Copied");
  });

  it("says so when the clipboard refuses", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async () => {
          throw new Error("denied");
        },
      },
    });
    // jsdom has no execCommand either, so both paths fail — which is the
    // case worth reporting rather than swallowing.
    (document as unknown as { execCommand?: () => boolean }).execCommand =
      undefined;

    await mount();
    await push(ev("assistant.message", { content: "done" }));
    const button = screen.getByTitle("Copy this reply");
    await act(async () => {
      button.click();
    });
    expect(button.textContent).toContain("Failed");
  });
});

const assembly = (text: string) => ({
  messages: 1, total_tokens: 100, system_tokens: 100,
  verbatim_messages: 0, summarized_messages: 0, summary_tokens: 0,
  compacted: false, budget: 160000,
  manifest: [{ index: 0, role: "system", tokens: 100, source: "system", text }],
});

describe("a stream that repeats itself", () => {
  it("shows a reply once when the same event arrives under a new sequence number", async () => {
    // What actually happened. Sequence numbers are assigned per process, so
    // a Studio restart renumbers the same session — and a browser tab whose
    // EventSource reconnects resumes from a number that now points somewhere
    // else entirely. The server obligingly re-sent a turn the tab was
    // already showing. Every copy carries the same event id, which is the
    // only identity that survives a restart.
    await mount();
    const reply = ev("assistant.message", { content: "the one true reply" });
    await push(ev("user.message", { text: "hi" }), reply);

    const stream = document.querySelector(".stream") as HTMLElement;
    expect(stream.textContent).toContain("the one true reply");

    await push({ ...reply, seq: reply.seq + 40 });

    expect(within(stream).getAllByText("the one true reply")).toHaveLength(1);
    expect(document.querySelectorAll(".msg-assistant")).toHaveLength(1);
  });

  it("shows a streamed reply once when its deltas are replayed too", async () => {
    // The same turn re-delivered whole: deltas first, then the final
    // message. Folding the message in twice is what produced two identical
    // bubbles — the first copy closes the streaming one, so the second has
    // nothing to attach to and starts a new bubble.
    await mount();
    const deltas = [
      ev("assistant.delta", { content: "Hey! " }),
      ev("assistant.delta", { content: "How can I help?" }),
    ];
    const final = ev("assistant.message", { content: "Hey! How can I help?" });
    await push(...deltas, final);
    expect(document.querySelectorAll(".msg-assistant")).toHaveLength(1);

    await push(
      ...deltas.map((d, i) => ({ ...d, seq: d.seq + 40 + i })),
      { ...final, seq: final.seq + 50 },
    );
    expect(document.querySelectorAll(".msg-assistant")).toHaveLength(1);
    const bubble = document.querySelector(".msg-assistant")!;
    expect(bubble.textContent).toContain("Hey! How can I help?");
    // And not twice inside the one bubble either.
    expect(bubble.textContent!.match(/How can I help\?/g)).toHaveLength(1);
  });
});

describe("a feed that was rebuilt underneath us", () => {
  it("refetches from scratch when the server announces a new numbering", async () => {
    await mount();
    const before = calls.filter((c) => c.url.includes("/events")).length;

    // The first hello establishes which numbering we are on; nothing to do.
    await push(ev("studio.stream.hello", { generation: "gen-1" }));
    expect(calls.filter((c) => c.url.includes("/events"))).toHaveLength(before);

    // A different one means Studio restarted between our connections.
    await push(ev("studio.stream.hello", { generation: "gen-2" }));
    await waitFor(() =>
      expect(
        calls.filter((c) => c.url.includes("/events")).length,
      ).toBeGreaterThan(before),
    );
  });

  it("does not put the announcement itself in the transcript", async () => {
    await mount();
    await push(ev("studio.stream.hello", { generation: "gen-1" }));
    const stream = document.querySelector(".stream");
    expect(stream?.textContent ?? "").not.toContain("gen-1");
  });
});

describe("branching", () => {
  it("forks the conversation at a message, leaving this session alone", async () => {
    await mount();
    await push(
      ev("user.message", { text: "the first ask" }),
      ev("assistant.message", { content: "a reply" }),
      ev("user.message", { text: "a direction I might regret" }),
    );

    const stream = document.querySelector(".stream") as HTMLElement;
    const first = within(stream).getByText("the first ask").closest(".msg-user")!;
    await act(async () => {
      (within(first as HTMLElement).getByTitle(/Start a new session from here/) as HTMLButtonElement).click();
    });

    const call = calls.find((c) => c.url.includes("/branch"));
    expect(call?.body).toMatchObject({ from_seq: 1 });
  });
});

describe("inspecting a message's context", () => {
  it("does not rewind the transcript when you ask about a message", async () => {
    // The complaint: clicking Context on an earlier reply cut the chat to
    // that event, which also left the turn looking unfinished because the
    // events that close it come after the message itself.
    await mount();
    await push(
      ev("agent.context.assembled", assembly("early prefix")),
      ev("assistant.message", { content: "the earlier reply" }),
      ev("agent.context.assembled", assembly("later prefix")),
      ev("assistant.message", { content: "the latest reply" }),
      ev("agent.cycle.end", {}),
    );

    const stream = document.querySelector(".stream") as HTMLElement;
    const earlier = within(stream).getByText("the earlier reply").closest(".msg-assistant")!;
    await act(async () => {
      (within(earlier as HTMLElement).getByTitle(/what the model was given/) as HTMLButtonElement).click();
    });

    // The later reply is still on screen: the transcript did not move.
    expect(stream.textContent).toContain("the latest reply");
    // And the pane says what it is pinned to, with a way back.
    const trace = document.querySelector(".trace") as HTMLElement;
    expect(trace.textContent).toContain("Pinned to the message");
    expect(within(trace).getByText("Follow the playhead")).toBeTruthy();
  });

  it("opens the context pane at the message you asked about", async () => {
    // The complaint this answers: the pane only ever showed the newest
    // request, so an earlier reply's context was unreachable.
    await mount();
    await push(
      ev("agent.context.assembled", {
        messages: 1, total_tokens: 100, system_tokens: 100,
        verbatim_messages: 0, summarized_messages: 0, summary_tokens: 0,
        compacted: false, budget: 160000,
        manifest: [{ index: 0, role: "system", tokens: 100, source: "system",
                     text: "early prefix" }],
      }),
      ev("assistant.message", { content: "the earlier reply" }),
      ev("agent.context.assembled", {
        messages: 1, total_tokens: 200, system_tokens: 200,
        verbatim_messages: 0, summarized_messages: 0, summary_tokens: 0,
        compacted: false, budget: 160000,
        manifest: [{ index: 0, role: "system", tokens: 200, source: "system",
                     text: "later prefix" }],
      }),
      ev("assistant.message", { content: "the latest reply" }),
    );

    const stream = document.querySelector(".stream") as HTMLElement;
    const earlier = within(stream).getByText("the earlier reply").closest(".msg-assistant")!;
    await act(async () => {
      (within(earlier as HTMLElement).getByTitle(/what the model was given/) as HTMLButtonElement).click();
    });

    const trace = document.querySelector(".trace") as HTMLElement;
    expect(trace.textContent).toContain("early prefix");
    expect(trace.textContent).not.toContain("later prefix");
  });
});

describe("receiving events", () => {
  it("never renders the same message twice", async () => {
    await mount();
    const message = ev("assistant.message", { content: "the one answer" });
    // The same event delivered twice — two live connections, a reconnect
    // that resumed a step early, a retried frame.
    await push(message, message);

    const stream = document.querySelector(".stream") as HTMLElement;
    expect(within(stream).getAllByText("the one answer")).toHaveLength(1);
  });

  it("keeps an event that arrives after a later one", async () => {
    // The old guard compared only against the newest event held, so an
    // out-of-order arrival was dropped for good — leaving a hole that reads
    // as a message stopping partway through.
    await mount();
    const first = ev("user.message", { text: "asked first" });
    const second = ev("user.message", { text: "asked second" });
    await push(second, first);

    const stream = document.querySelector(".stream") as HTMLElement;
    expect(within(stream).getByText("asked first")).toBeTruthy();
    expect(within(stream).getByText("asked second")).toBeTruthy();
  });

  it("puts out-of-order events back in order", async () => {
    await mount();
    const a = ev("assistant.delta", { content: "one " });
    const b = ev("assistant.delta", { content: "two " });
    const c = ev("assistant.delta", { content: "three" });
    await push(c, a, b);

    const stream = document.querySelector(".stream") as HTMLElement;
    expect(stream.textContent).toContain("one two three");
  });

  it("fetches what it missed rather than showing a hole", async () => {
    await mount();
    await push(ev("user.message", { text: "first" }));

    // Pretend seq 2 and 3 were missed: the server has them, the stream
    // jumps to 4.
    const missed = [
      { ...ev("assistant.message", { content: "the missing middle" }) },
    ];
    missed[0]!.seq = 2;
    events = missed;

    const jumped = ev("assistant.message", { content: "the later one" });
    jumped.seq = 4;
    await push(jumped);

    await waitFor(() => {
      const stream = document.querySelector(".stream") as HTMLElement;
      expect(stream.textContent).toContain("the missing middle");
    });
  });
});

describe("the context pane", () => {
  const compacted = () => [
    ev("agent.context.summarized", {
      summary: "The user is refactoring the parser. Their deploy key lives in .secrets/.",
      summarized_count: 30,
    }),
    ev("agent.context.assembled", {
      messages: 5,
      total_tokens: 18000,
      system_tokens: 4000,
      verbatim_messages: 3,
      summarized_messages: 30,
      summary_tokens: 900,
      compacted: true,
      budget: 160000,
      manifest: [
        { index: 0, role: "system", tokens: 4000, source: "system",
          text: "You are eventage-code, a coding agent" },
        { index: 1, role: "system", tokens: 900, source: "summary",
          text: "The user is refactoring the parser" },
        { index: 2, role: "user", tokens: 20, source: "verbatim",
          text: "now fix the tokenizer\nacross several lines\nof detail" },
        { index: 3, role: "assistant", tokens: 40, source: "verbatim",
          text: "read_file({\"path\":\"src/lex.rs\"})" },
        { index: 4, role: "tool", tokens: 60, source: "cleared",
          text: "[cleared by harness: 12000 chars of stale tool output",
          truncated_from: 41000 },
      ],
    }),
  ];

  const openContext = async () => {
    const trace = document.querySelector(".trace") as HTMLElement;
    await act(async () => {
      within(trace).getByText("Context").click();
    });
    return trace;
  };

  it("shows what the last request was made of", async () => {
    await mount();
    await push(...compacted());
    const trace = await openContext();

    expect(trace.textContent).toContain("system prefix");
    expect(trace.textContent).toContain("30 messages folded");
    expect(trace.textContent).toContain("verbatim 3 messages");
  });

  it("opens a message to its full text, not a trimmed line", async () => {
    // A one-line preview answers "which messages"; the question people have
    // is what the message actually said.
    await mount();
    await push(...compacted());
    const trace = await openContext();

    // Collapsed: only the first line shows.
    expect(trace.textContent).toContain("now fix the tokenizer");
    expect(trace.textContent).not.toContain("across several lines");

    const rows = trace.querySelectorAll(".msg-line");
    await act(async () => {
      (rows[2] as HTMLButtonElement).click();
    });
    expect(trace.textContent).toContain("across several lines");
    expect(trace.textContent).toContain("of detail");
  });

  it("says when the record had to cut a very long message", async () => {
    await mount();
    await push(...compacted());
    const trace = await openContext();
    const rows = trace.querySelectorAll(".msg-line");
    await act(async () => {
      (rows[4] as HTMLButtonElement).click();
    });
    expect(trace.textContent).toContain("of 41,000 characters");
    expect(trace.textContent).toContain("event log");
  });

  it("names every message, not just how many there were", async () => {
    // "16 verbatim messages" is a count, not transparency: the question is
    // which sixteen, and whether the thing you said an hour ago is still
    // there or has been folded into a summary.
    await mount();
    await push(...compacted());
    const trace = await openContext();

    const rows = trace.querySelectorAll(".msg-row");
    expect(rows.length).toBe(5);
    expect(trace.textContent).toContain("now fix the tokenizer");
    expect(trace.textContent).toContain('read_file({"path":"src/lex.rs"})');
  });

  it("marks which messages stand in for history the model cannot see", async () => {
    await mount();
    await push(...compacted());
    const trace = await openContext();

    const summaryRow = trace.querySelector(".msg-row.summary");
    expect(summaryRow).toBeTruthy();
    expect(summaryRow!.textContent).toContain("refactoring the parser");

    // And which were emptied to reclaim budget.
    const clearedRow = trace.querySelector(".msg-row.cleared");
    expect(clearedRow).toBeTruthy();
    expect(clearedRow!.textContent).toContain("cleared by harness");
  });

  it("shows each message's size, so a hog is obvious", async () => {
    await mount();
    await push(...compacted());
    const trace = await openContext();
    const sizes = [...trace.querySelectorAll(".msg-row .size")].map(
      (n) => n.textContent,
    );
    expect(sizes).toEqual(["4.0k", "900", "20", "40", "60"]);
  });

  it("says so when a request predates per-message recording", async () => {
    await mount();
    await push(
      ev("agent.context.assembled", {
        messages: 3, total_tokens: 500, system_tokens: 400,
        verbatim_messages: 2, summarized_messages: 0, summary_tokens: 0,
        compacted: false, budget: 160000,
      }),
    );
    const trace = await openContext();
    expect(trace.textContent).toContain("only the totals are known");
  });

  it("shows the summary the model is actually working from", async () => {
    await mount();
    await push(...compacted());
    const trace = await openContext();
    expect(trace.textContent).toContain("deploy key lives in .secrets/");
  });

  it("says plainly when nothing has been compacted", async () => {
    await mount();
    await push(
      ev("agent.context.assembled", {
        messages: 4,
        total_tokens: 5000,
        system_tokens: 4000,
        verbatim_messages: 4,
        summarized_messages: 0,
        summary_tokens: 0,
        compacted: false,
        budget: 160000,
      }),
    );
    const trace = await openContext();
    expect(trace.textContent).toContain("Nothing has been compacted");
  });

  it("lets a dropped detail be put back, and says the original is kept", async () => {
    await mount();
    await push(...compacted());
    const trace = await openContext();

    await act(async () => {
      within(trace).getByText("Edit").click();
    });
    const box = trace.querySelector(".ctx-edit") as HTMLTextAreaElement;
    expect(box.value).toContain("deploy key");
    expect(trace.textContent).toContain("it is superseded");

    await type(box, "Corrected summary: also, never touch .secrets/");
    await act(async () => {
      within(trace).getByText("Replace").click();
    });

    const call = calls.find((c) => c.url.includes("/summary"));
    expect(call?.body).toMatchObject({
      summary: "Corrected summary: also, never touch .secrets/",
      summarized_count: 30,
    });
  });

  it("marks a summary that a person edited", async () => {
    await mount();
    await push(
      ev("agent.context.summarized", {
        summary: "hand written",
        summarized_count: 10,
        source: "manual_override",
      }),
    );
    const trace = await openContext();
    expect(trace.textContent).toContain("edited by you");
  });

  it("explains itself before anything has been assembled", async () => {
    await mount();
    await push(ev("user.message", { text: "hi" }));
    const trace = await openContext();
    expect(trace.textContent).toContain("Nothing assembled yet");
  });
});

describe("the trace panel", () => {
  it("lists every event, including ones the transcript does not show", async () => {
    await mount();
    await push(
      ev("user.message", { text: "hello" }),
      ev("agent.cycle.start"),
      ev("system.checkpoint", { note: "turn start" }),
      ev("assistant.delta", { content: "hi" }),
    );

    const trace = document.querySelector(".trace") as HTMLElement;
    await act(async () => {
      within(trace).getByText("Events").click();
    });
    expect(within(trace).getByText("system.checkpoint")).toBeTruthy();
    expect(within(trace).getByText("assistant.delta")).toBeTruthy();
    // The checkpoint is bookkeeping — it belongs in the trace, not the chat.
    const stream = document.querySelector(".stream") as HTMLElement;
    expect(within(stream).queryByText("system.checkpoint")).toBeNull();
  });

  it("opens the full event on click", async () => {
    await mount();
    await push(ev("tool.result", { tool_call_id: "c1", result: { rows: 42 } }));

    const trace = document.querySelector(".trace") as HTMLElement;
    await act(async () => {
      within(trace).getByText("Events").click();
    });
    await act(async () => {
      within(trace).getByText("tool.result").click();
    });
    const detail = document.querySelector(".trace-detail") as HTMLElement;
    expect(detail).toBeTruthy();
    expect(detail.textContent).toContain("42");
  });

  it("offers the timeline, flow and event views", async () => {
    await mount();
    await push(
      ev("user.message", { text: "go" }),
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }),
      ev("tool.result", { tool_call_id: "c1", result: {} }),
    );
    const trace = document.querySelector(".trace") as HTMLElement;

    // Timeline is the default and draws on a canvas.
    expect(trace.querySelector("canvas")).toBeTruthy();

    await act(async () => {
      within(trace).getByText("Flow").click();
    });
    const flow = trace.querySelector("svg.flow") as SVGElement;
    expect(flow).toBeTruthy();
    expect(flow.textContent).toContain("bash");
  });

  it("settles instead of re-rendering forever", async () => {
    // A canvas that resizes itself inside a scrolling parent is an easy way
    // to build an infinite render loop; this catches one by noticing that the
    // component tree never stops updating.
    await mount();
    await push(
      ev("user.message", { text: "go" }),
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }),
      ev("tool.result", { tool_call_id: "c1", result: {} }),
    );
    const before = document.querySelectorAll("*").length;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 250));
    });
    expect(document.querySelectorAll("*").length).toBe(before);
  });

  it("filters by group", async () => {
    await mount();
    await push(
      ev("assistant.delta", { content: "hi" }),
      ev("tool.call.proposed", { tool_call_id: "c1", name: "bash" }),
    );

    const trace = document.querySelector(".trace") as HTMLElement;
    await act(async () => {
      within(trace).getByText("Events").click();
    });
    expect(within(trace).queryByText("assistant.delta")).toBeTruthy();

    await act(async () => {
      within(trace).getByText("model").click(); // turn the group off
    });
    expect(within(trace).queryByText("assistant.delta")).toBeNull();
    expect(within(trace).queryByText("tool.call.proposed")).toBeTruthy();
  });
});
