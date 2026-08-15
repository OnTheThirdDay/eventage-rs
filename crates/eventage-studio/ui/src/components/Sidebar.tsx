/** Sessions: what is open now, and what can be reopened. */

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
  onForget,
}: {
  open: SessionInfo[];
  stored: StoredSession[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onResume: (session: StoredSession) => void;
  onNew: () => void;
  onPickWorkspace: () => void;
  onForget: (id: string) => void;
}) {
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
          <button
            key={session.id}
            className={`session-row ${session.id === activeId ? "active" : ""}`}
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
        ))}

        {stored.length > 0 && <div className="section-label">Earlier</div>}
        {stored.map((session) => (
          <button
            key={session.id}
            className="session-row"
            onClick={() => onResume(session)}
          >
            <span style={{ flex: 1, minWidth: 0 }}>
              <span className="title" style={{ display: "block" }}>
                {session.title}
              </span>
              <span className="sub">
                {basename(session.cwd)} · {formatWhen(session.updated_at)}
              </span>
            </span>
            <span
              className="forget"
              role="button"
              tabIndex={0}
              title="Delete this session's history"
              onClick={(e) => {
                e.stopPropagation();
                onForget(session.id);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.stopPropagation();
                  onForget(session.id);
                }
              }}
            >
              ✕
            </span>
          </button>
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
