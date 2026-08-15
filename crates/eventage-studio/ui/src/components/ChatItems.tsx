/** The individual things that appear in a transcript. */

import { useState } from "react";
import type {
  AssistantItem,
  NoticeItem,
  PermissionItem,
  PlanEntry,
  ToolItem,
  UserItem,
} from "../lib/types";
import { DiffView } from "./DiffView";
import {
  CopyButton,
  Json,
  Markdown,
  Spinner,
  formatDuration,
  formatTokens,
} from "./primitives";

/** A glyph per tool family, so a long transcript stays scannable. */
function iconFor(name: string): string {
  if (name.startsWith("lsp_")) return "◈";
  if (/read|glob|list_directory/.test(name)) return "▤";
  if (/grep|search/.test(name)) return "⌕";
  if (/write|edit/.test(name)) return "✎";
  if (/bash|shell|exec/.test(name)) return "▸";
  if (/git|pull_request/.test(name)) return "⑂";
  if (/task|agent/.test(name)) return "❖";
  if (/plan|todo/.test(name)) return "☰";
  if (/fetch|web/.test(name)) return "◍";
  return "◆";
}

export function UserMessage({
  item,
  onBranchFrom,
}: {
  item: UserItem;
  onBranchFrom?: (seq: number) => void;
}) {
  return (
    <div className="msg-user">
      <div className="hover-actions">
        <CopyButton text={item.text} label="⧉" title="Copy this message" />
        {onBranchFrom && item.seq !== undefined && (
          <button
            className="btn sm"
            onClick={() => onBranchFrom(item.seq!)}
            title="Start a new session from here, leaving this one as it is"
          >
            ⑂
          </button>
        )}
      </div>
      {item.text}
      {item.images.map((src, i) => (
        <img key={i} src={src} alt="attachment" />
      ))}
    </div>
  );
}

export function AssistantMessage({
  item,
  onInspectContext,
}: {
  item: AssistantItem;
  onInspectContext?: (seq: number) => void;
}) {
  return (
    <div className="msg-assistant">
      {item.thinking && (
        <details className="thinking">
          <summary>Thinking</summary>
          <div className="body">{item.thinking}</div>
        </details>
      )}
      {item.text ? <Markdown text={item.text} /> : null}
      {item.streaming && <span className="caret" />}
      {!item.streaming && item.text && (
        <div className="msg-actions">
          <CopyButton text={item.text} label="⧉ Copy" title="Copy this reply" />
          {onInspectContext && item.seq !== undefined && (
            <button
              className="btn sm ghost"
              onClick={() => onInspectContext(item.seq!)}
              title="Show what the model was given when it wrote this"
            >
              ⧉ Context
            </button>
          )}
        </div>
      )}
      {item.tokens && item.tokens.output > 0 && (
        <div className="token-line">
          {formatTokens(item.tokens.input)} in · {formatTokens(item.tokens.output)}{" "}
          out
          {item.tokens.cached > 0 &&
            ` · ${formatTokens(item.tokens.cached)} cached`}
        </div>
      )}
    </div>
  );
}

export function ToolCard({ item }: { item: ToolItem }) {
  // Failures open by default: that is the case someone needs to read.
  const [open, setOpen] = useState(item.status === "failed");
  const hasBody =
    item.diff !== undefined ||
    item.error !== undefined ||
    item.locations.length > 0 ||
    item.args !== null ||
    item.result !== undefined;

  return (
    <div className={`tool ${item.status}`}>
      <button
        className="tool-head"
        onClick={() => hasBody && setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="tool-icon">
          {item.status === "running" ? <Spinner /> : iconFor(item.name)}
        </span>
        <span className="tool-title">{item.title}</span>
        {item.status === "failed" && <span className="tool-meta k-error">failed</span>}
        {item.durationMs !== undefined && item.status !== "running" && (
          <span className="tool-meta">{formatDuration(item.durationMs)}</span>
        )}
        {hasBody && <span className="tool-meta">{open ? "▾" : "▸"}</span>}
      </button>

      {open && hasBody && (
        <div className="tool-body">
          {item.error && (
            <>
              <p className="label">
                Error
                <CopyButton text={item.error} label="⧉" className="btn sm ghost inline" />
              </p>
              <div className="tool-error">{item.error}</div>
            </>
          )}
          {item.diff && (
            <>
              <p className="label">Changes</p>
              <DiffView diff={item.diff} />
            </>
          )}
          {item.locations.length > 0 && (
            <>
              <p className="label">Locations</p>
              <div className="locations">
                {item.locations.map((loc, i) => {
                  const ref = `${loc.path}${loc.line !== undefined ? `:${loc.line}` : ""}`;
                  return (
                    <CopyButton
                      key={i}
                      text={ref}
                      label={ref}
                      className="loc"
                      title="Copy this path"
                    />
                  );
                })}
              </div>
            </>
          )}
          {item.args !== null && item.args !== undefined && (
            <>
              <p className="label">
                Arguments
                <CopyButton
                  text={() => JSON.stringify(item.args, null, 2)}
                  label="⧉"
                  className="btn sm ghost inline"
                />
              </p>
              <Json value={item.args} />
            </>
          )}
          {item.result !== undefined && !item.diff && (
            <>
              <p className="label">
                Result
                <CopyButton
                  text={() => JSON.stringify(item.result, null, 2)}
                  label="⧉"
                  className="btn sm ghost inline"
                />
              </p>
              <Json value={item.result} />
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function PermissionCard({
  item,
  onAnswer,
}: {
  item: PermissionItem;
  onAnswer: (approve: boolean, always: boolean) => void;
}) {
  const pending = item.status === "pending";
  return (
    <div className={`permission ${pending ? "" : "resolved"}`}>
      <div className="headline">
        {pending ? "Approve this action?" : "Permission request"}
      </div>
      <div>
        <span className="tool-name">{item.tool}</span>{" "}
        {pending && <span className="verdict">is waiting for your decision.</span>}
      </div>

      {item.args !== null && item.args !== undefined && (
        <div className="args">
          <Json value={item.args} />
        </div>
      )}

      {pending ? (
        <div className="actions">
          <button className="btn primary" onClick={() => onAnswer(true, false)}>
            Allow once
          </button>
          <button className="btn" onClick={() => onAnswer(true, true)}>
            Always allow {item.tool}
          </button>
          <button className="btn danger" onClick={() => onAnswer(false, false)}>
            Reject
          </button>
        </div>
      ) : (
        <div className="verdict">
          {item.status === "approved" ? "Allowed" : "Rejected"}
          {item.auto === "always_allow" && " automatically (you allowed this tool)"}
          {item.reason && ` — ${item.reason}`}
        </div>
      )}
    </div>
  );
}

export function Notice({ item }: { item: NoticeItem }) {
  const glyph =
    item.level === "error" ? "✕" : item.level === "warn" ? "!" : "i";
  return (
    <div className={`notice ${item.level}`}>
      <span aria-hidden>{glyph}</span>
      <div>
        <div className="title">{item.title}</div>
        {item.detail && <div className="detail">{item.detail}</div>}
      </div>
    </div>
  );
}

export function PlanPanel({ entries }: { entries: PlanEntry[] }) {
  if (!entries.length) return null;
  const done = entries.filter((e) => e.status === "completed").length;
  return (
    <div className="plan">
      <h4>
        Plan · {done}/{entries.length}
      </h4>
      <ul>
        {entries.map((entry, i) => (
          <li key={i} className={entry.status}>
            <span aria-hidden>
              {entry.status === "completed"
                ? "✓"
                : entry.status === "in_progress"
                  ? "▸"
                  : "○"}
            </span>
            <span>{entry.content}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
