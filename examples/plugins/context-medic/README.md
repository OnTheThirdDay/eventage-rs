# context-medic

When a conversation gets long, the agent compacts it: old messages are replaced
by a short summary, and the model can no longer see the originals. Sometimes
that summary loses something — the exact path of a file, the reason an approach
was abandoned, the error text being chased.

Studio already lets you rewrite that summary by hand. This plugin finds the
missing pieces for you and offers them as a list you pick from.

## How you use it

1. Open the **Context** panel in Studio (the right-hand pane) and find
   **Summary in effect**. This is the block of text standing in for the
   conversation the model can no longer see.
2. Press **Review**. One model call goes out.
3. A list appears: each row is one detail the reviewer thinks the summary lost,
   with a short note on why it matters.
4. Tick the ones you want and press **Add to context**. Only those are written
   back.

Nothing happens until you press Review, and nothing enters the context until
you press Add. Watching compaction go by costs nothing — the plugin just
records what was folded away, locally.

The previous summary is not deleted. It stays in the event log, and Studio
shows a line under the summary saying how many earlier versions exist.

## What it can find that hand-editing cannot

Besides messages folded into the summary, the plugin also reads **cleared tool
results** — command and file output that the harness blanked to reclaim budget
before compaction ever ran. Their full text is still in the event log, and
nothing else in Studio brings it back.

## Install

```sh
mkdir -p ~/.eventage/plugins
cp -r examples/plugins/context-medic ~/.eventage/plugins/
export EVENTAGE_TRUST_PROJECT_SETTINGS=1
```

Then open a new session — plugins load when a session starts, so an
already-open one will not see it.

No API key. The plugin asks the host to run its completion with the model you
configured in Studio, which means it uses your model, shares the same retry and
rate-limit behaviour, and its token spend counts against your session budget
instead of being invisible to it.

Two requirements, and the agent says plainly at startup if either is unmet:

- It must live in `~/.eventage/plugins`, not in a repository's
  `.eventage/plugins`. An observer is a process that reads your conversation, so
  a `git clone` cannot start one.
- Asking the host for a completion is a graded capability. The manifest
  declares `llm = true`, and the project must be trusted —
  `EVENTAGE_TRUST_PROJECT_SETTINGS=1`, the same switch that lets a repository's
  `.claude/settings.json` set environment variables.

## Writing your own

Read one event per line on stdin, write one per line on stdout. `id` and
`timestamp` are filled in for you:

```jsonl
{"kind":"agent.context.audit.result","payload":{"request_id":"...","items":[]}}
```

You only see the kinds listed under `watch`, and may only publish the kinds
listed under `emit`. Anything else is dropped and logged. To use the host's
model, emit `llm.request` with `{request_id, messages, schema?, schema_name?}`
and watch for the matching `llm.response`; a `schema` gets you validated
structured output instead of prose.
