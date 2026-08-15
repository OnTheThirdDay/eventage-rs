/**
 * Choose a workspace root.
 *
 * A browser cannot hand back a real filesystem path, so the directory tree is
 * walked through the server instead — which is also the only way the agent
 * could open it.
 */

import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";

export function WorkspacePicker({
  start,
  onPick,
  onCancel,
}: {
  start: string;
  onPick: (path: string) => void;
  onCancel: () => void;
}) {
  const [path, setPath] = useState(start);
  const [parent, setParent] = useState<string | null>(null);
  const [dirs, setDirs] = useState<{ name: string; path: string }[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (target: string) => {
    try {
      const listing = await api.listDir(target);
      setPath(listing.path);
      setParent(listing.parent);
      setDirs(listing.dirs);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load(start);
  }, [start, load]);

  return (
    <div
      className="picker-backdrop"
      onClick={(e) => e.target === e.currentTarget && onCancel()}
    >
      <div className="picker">
        <h3>Open a workspace</h3>
        <div className="path">{path}</div>
        <div className="list">
          {parent && (
            <button onClick={() => void load(parent)}>../</button>
          )}
          {dirs.map((dir) => (
            <button key={dir.path} onClick={() => void load(dir.path)}>
              {dir.name}/
            </button>
          ))}
          {!dirs.length && !parent && (
            <div style={{ padding: 12, color: "var(--text-faint)" }}>
              Nothing to descend into.
            </div>
          )}
          {error && (
            <div style={{ padding: 12, color: "var(--danger)" }}>{error}</div>
          )}
        </div>
        <div className="foot">
          <button className="btn" onClick={onCancel}>
            Cancel
          </button>
          <button className="btn primary" onClick={() => onPick(path)}>
            Open this folder
          </button>
        </div>
      </div>
    </div>
  );
}
