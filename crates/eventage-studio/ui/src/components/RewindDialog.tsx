/**
 * Rewind, with the consequences shown before you commit to them.
 *
 * "Undo one turn" is a fine shortcut but a poor explanation: the thing a
 * person wants to know is *what disappears*, and specifically whether files
 * were written. The DAG already holds the answer — every event after the
 * checkpoint is exactly what a rollback would seal off — so the dialog
 * computes the summary from the same log the rewind will act on rather than
 * guessing.
 *
 * Nothing is destroyed either way: the discarded trajectory becomes a
 * rejected branch, which stays visible in the trace.
 */

import { useMemo } from "react";
import type { Checkpoint } from "../lib/timeline";
import type { StudioEvent } from "../lib/types";

interface Consequences {
  events: number;
  toolCalls: number;
  filesWritten: string[];
  commands: string[];
  messages: number;
  tokens: number;
}

const str = (v: unknown) => (typeof v === "string" ? v : "");
const num = (v: unknown) => (typeof v === "number" ? v : 0);

function parseArgs(raw: unknown): Record<string, unknown> {
  if (raw && typeof raw === "object") return raw as Record<string, unknown>;
  if (typeof raw !== "string") return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

/** What a rollback to `checkpoint` would discard. */
export function consequencesOf(
  events: StudioEvent[],
  checkpointSeq: number,
): Consequences {
  const after = events.filter((e) => e.seq >= checkpointSeq);
  const filesWritten = new Set<string>();
  const commands: string[] = [];
  let toolCalls = 0;
  let messages = 0;
  let tokens = 0;

  for (const event of after) {
    if (event.kind === "tool.call.proposed") {
      toolCalls += 1;
      const name = str(event.payload?.name);
      const args = parseArgs(event.payload?.arguments);
      if (/write|edit/.test(name)) {
        const path = str(args.path) || str(args.file_path);
        if (path) filesWritten.add(path);
      }
      if (/bash|shell/.test(name)) {
        const command = str(args.command);
        if (command) commands.push(command);
      }
    }
    if (event.kind === "assistant.message" && str(event.payload?.content)) messages += 1;
    tokens += num(event.meta?.llm_input_tokens) + num(event.meta?.llm_output_tokens);
  }

  return {
    events: after.length,
    toolCalls,
    filesWritten: [...filesWritten],
    commands,
    messages,
    tokens,
  };
}

export function RewindDialog({
  checkpoint,
  events,
  onConfirm,
  onCancel,
}: {
  checkpoint: Checkpoint;
  events: StudioEvent[];
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const what = useMemo(
    () => consequencesOf(events, checkpoint.seq),
    [events, checkpoint.seq],
  );

  return (
    <div
      className="picker-backdrop"
      onClick={(e) => e.target === e.currentTarget && onCancel()}
    >
      <div className="picker rewind">
        <h3>Rewind to the start of turn {checkpoint.turn}</h3>

        <div className="rewind-body">
          <p className="rewind-lead">
            {what.events === 0
              ? "There is nothing after this point to discard."
              : `This takes the conversation back, discarding ${what.events} events.`}
          </p>

          <div className="rewind-stats">
            <div>
              <b>{what.messages}</b> repl{what.messages === 1 ? "y" : "ies"}
            </div>
            <div>
              <b>{what.toolCalls}</b> tool call{what.toolCalls === 1 ? "" : "s"}
            </div>
            <div>
              <b>{what.tokens.toLocaleString()}</b> tokens spent
            </div>
          </div>

          {what.filesWritten.length > 0 && (
            <>
              <p className="label">
                Files the agent changed after this point
              </p>
              <ul className="rewind-list">
                {what.filesWritten.map((path) => (
                  <li key={path}>{path}</li>
                ))}
              </ul>
              <p className="rewind-warning">
                Rewinding forgets the conversation, not the disk. These edits
                stay on disk — undo them with git if you want them gone.
              </p>
            </>
          )}

          {what.commands.length > 0 && (
            <>
              <p className="label">Commands it ran</p>
              <ul className="rewind-list mono">
                {what.commands.slice(0, 6).map((command, i) => (
                  <li key={i}>{command}</li>
                ))}
                {what.commands.length > 6 && (
                  <li className="muted">and {what.commands.length - 6} more</li>
                )}
              </ul>
            </>
          )}

          <p className="rewind-note">
            The discarded trajectory is kept as a rejected branch — it stays in
            the trace, greyed out, and is not sent to the model again.
          </p>
        </div>

        <div className="foot">
          <button className="btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn danger primary-danger"
            onClick={onConfirm}
            disabled={what.events === 0}
          >
            Rewind
          </button>
        </div>
      </div>
    </div>
  );
}
