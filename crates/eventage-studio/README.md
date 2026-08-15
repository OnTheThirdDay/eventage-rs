# Eventage Studio

A desktop app for the eventage coding agent: the conversation on one side, a
live trace of everything behind it on the other.

The trace is why the app has this shape. An event-sourced harness already
knows exactly what it did and why — which hook denied a call, what the token
accounting was, which turn a rewind sealed off — and a chat bubble throws all
of that away. Studio shows both, derived from the same event stream, so they
cannot disagree.

```sh
# Host the coding agent in this process. Full event log in the trace.
eventage-studio

# Or drive any ACP agent over stdio, the way an editor does.
eventage-studio --acp eventage-code
```

Studio opens a window on a random loopback port and prints the URL. Chrome and
Edge are launched with `--app` so the result reads as an application rather
than a browser tab; without one of those it falls back to the default handler.

## Building

The front-end is a React/TypeScript app compiled into the binary, and it is
not checked in. Build it once, then build the crate:

```sh
crates/eventage-studio/ui/build.sh
cargo build --release -p eventage-studio
```

Skipping the first step still compiles — the binary serves a placeholder page
telling you what to run.

## Credentials

Same as `eventage-code`, read from the environment:

| Variable | Provider |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Messages API |
| `QWEN_API_KEY` | Alibaba Cloud Qwen (compatible mode) |
| `OPENAI_API_KEY` | OpenAI Responses API |
| `OPENAI_BASE_URL` | any OpenAI-compatible gateway |

## What it does

**Conversation.** Streaming replies with the model's thinking kept in a
separate, collapsed block. Tool calls become cards that expand to show
arguments, results, source locations, and a real diff for anything that
touched a file. Plans render as a live checklist. Permission requests appear
inline with allow / always-allow / reject, and "always" is remembered for the
rest of the session.

**Trace.** Three views over the same log, because a session gets asked three
different questions:

- **Timeline** — swimlanes per participant, with each tool call drawn as a
  *span* from proposal to result. That is what makes concurrency visible: the
  ReAct loop runs several tools at once and a list of events cannot show it.
  Checkpoints appear as flags, turns as alternating bands, and a rewound
  trajectory stays on the chart in muted colour.
- **Flow** — a sequence diagram of who asked whom and what came back,
  including permission round-trips.
- **Events** — the raw list, filterable by group, searchable across payloads,
  with the complete JSON of any event one click away.

Token accounting and per-turn timing come from the harness's own metadata
rather than being re-counted. The whole log exports as JSONL.

**Time travel.** The transport bar scrubs the session, and the playhead drives
*both* panes: stepping back rewinds the transcript as well as the trace, so
you replay the conversation rather than filtering a list. A banner says you
are in the past and offers the way back; sending a message returns to live.

**Sessions.** Persisted per workspace and reopenable from the sidebar. Rewind
undoes turns using the DAG rather than editing a message array — pick any
checkpoint from the timeline, and a dialog names exactly what will be
discarded first: how many replies and tool calls, which files were written,
which commands ran. Edits stay on disk; only the conversation is rewound, and
the dialog says so.

## Two backends

| | `--local` (default) | `--acp <command>` |
|---|---|---|
| Agent | hosted in this process | separate process over stdio |
| Trace | the complete event DAG | protocol traffic only |
| Works with | `eventage-code` | any ACP agent |
| Rewind | yes | only if the agent implements it |

The ACP backend normalises `session/update` notifications onto the same event
kinds the local backend emits, so the UI has one reducer rather than two. The
difference shows up as `full_trace: false`, and the trace panel says so
instead of looking broken.

## Access control

The server binds to loopback and requires a token minted at startup. Loopback
alone is not enough: a session's event stream carries prompts, file contents
and command output, all of which would otherwise be readable by any process on
the machine. The token arrives in the URL, is exchanged for an `HttpOnly`
cookie on first load, and is never logged.

## Layout

```
src/
  main.rs        CLI, window launch, graceful shutdown
  server.rs      HTTP + SSE, token gate
  feed.rs        per-session event log with resumable sequence numbers
  backend/
    local.rs     hosts a CodingSession, forwards its bus
    acp.rs       JSON-RPC client, normalises updates onto our event kinds
  protocol.rs    the wire model the UI consumes
  index.rs       small on-disk index so the sidebar need not scan event logs
ui/
  src/lib/reduce.ts    events → transcript, plan, stats (pure, tested)
  src/components/      chat, trace, composer, sidebar, diff
```

## Tests

```sh
cargo test -p eventage-studio      # feed, index, ACP normalisation, HTTP API
cd ui && npm test                  # reducer + the app mounted in jsdom
```

The HTTP tests run against a stand-in backend, so no model, key or network is
needed to check the parts Studio owns.
