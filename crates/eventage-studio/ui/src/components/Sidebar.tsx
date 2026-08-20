/** Sessions: what is open now, and what can be reopened. */

import { useState } from "react";
import type { SessionInfo, StoredSession } from "../lib/types";
import { formatWhen } from "./primitives";

export function Sidebar({
  open,
  stored,
  activeId,
  onSelect,
  onResume,
  onNew,
  onPickWorkspace,
  onClose,
  onForget,
}: {
  open: SessionInfo[];
  stored: StoredSession[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onResume: (session: StoredSession) => void;
  onNew: () => void;
  onPickWorkspace: () => void;
  onClose: (id: string) => void;
  onForget: (id: string) => void;
}) {
  /// Which stored session is one click away from being deleted.
  ///
  /// Deleting history cannot be undone, and the button that does it sits
  /// exactly where the harmless close button sits one list up — a row that
  /// was reopened moves between the two lists, so the same spot means two
  /// different things depending on which list it is currently in. Asking
  /// first, inline, is cheaper than a dialog for a row this small.
  const [confirming, setConfirming] = useState<string | null>(null);

  return (
    <nav className="sidebar">
      <div className="sidebar-head">
        <button className="btn primary" onClick={onNew}>
          New session
        </button>
        <button
          className="btn icon"
          onClick={onPickWorkspace}
          title="Open another workspace"
        >
          ⌂
        </button>
      </div>

      <div className="sidebar-scroll">
        {open.length > 0 && <div className="section-label">Open</div>}
        {open.map((session) => (
          <div
            key={session.id}
            className={`session-row ${session.id === activeId ? "active" : ""}`}
          >
            <button
              className="session-open"
              onClick={() => onSelect(session.id)}
            >
              {session.running && <span className="running-dot" />}
              <span style={{ flex: 1, minWidth: 0 }}>
                <span className="title" style={{ display: "block" }}>
                  {session.title}
                </span>
                <span className="sub">{basename(session.cwd)}</span>
              </span>
            </button>
            <button
              className="row-action"
              aria-label={`Close ${session.title}`}
              title={
                session.running
                  ? "Close this session — the running turn is stopped, the history is kept"
                  : "Close this session — its history is kept under Earlier"
              }
              onClick={() => onClose(session.id)}
            >
              ✕
            </button>
          </div>
        ))}

        {stored.length > 0 && <div className="section-label">Earlier</div>}
        {stored.map((session) => (
          <div key={session.id} className="session-row">
            <button className="session-open" onClick={() => onResume(session)}>
              <span style={{ flex: 1, minWidth: 0 }}>
                <span className="title" style={{ display: "block" }}>
                  {session.title}
                </span>
                <span className="sub">
                  {basename(session.cwd)} · {formatWhen(session.updated_at)}
                </span>
              </span>
            </button>
            {confirming === session.id ? (
              <>
                <button
                  className="row-action shown danger"
                  onClick={() => {
                    setConfirming(null);
                    onForget(session.id);
                  }}
                >
                  Delete
                </button>
                <button
                  className="row-action shown"
                  onClick={() => setConfirming(null)}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                className="row-action"
                aria-label={`Delete ${session.title}`}
                title="Delete this session's history"
                onClick={() => setConfirming(session.id)}
              >
                ✕
              </button>
            )}
          </div>
        ))}

        {open.length === 0 && stored.length === 0 && (
          <div
            style={{
              padding: "18px 8px",
              fontSize: 12.5,
              color: "var(--text-faint)",
            }}
          >
            No sessions yet.
          </div>
        )}
      </div>
    </nav>
  );
}

const basename = (path: string) =>
  path.split(/[/\\]/).filter(Boolean).pop() ?? path;
