/**
 * The workstream panel: where a fan-out becomes a choice.
 *
 * A cowork turn deliberately does not merge its own results. Several ran
 * because they are not equally good, and each one worked in a private copy of
 * the folder — so at the end there is a comparison to make, and this is where
 * it happens. Until something is kept, the user's folder is untouched.
 *
 * The two actions are not symmetrical, and the panel should not make them look
 * it. Keeping writes files. Abandoning asks for a reason, because the reason
 * is the whole point: it stays in the graph and a later attempt reads it back.
 * That is the difference from rejecting a diff in Cowork or the Codex app,
 * where the reasoning goes with the result.
 */
import { useState } from "react";
import type { Workstream } from "../lib/types";

export function Workstreams({
  workstreams,
  busy,
  onAdopt,
  onSeal,
}: {
  workstreams: Workstream[];
  busy: boolean;
  onAdopt: (id: string) => void;
  onSeal: (id: string, reason: string) => void;
}) {
  // Nothing to compare, nothing to show. The coding and ACP backends never
  // produce these, so the panel simply is not there.
  if (workstreams.length === 0) return null;

  const finished = workstreams.filter((w) => w.status === "finished");
  return (
    <section className="workstreams" aria-label="Workstreams">
      <header className="workstreams-head">
        <h2>Workstreams</h2>
        <span className="hint">
          {finished.length > 0
            ? "Nothing is in your folder yet — keep one to apply it."
            : "Each works in its own copy of the folder."}
        </span>
      </header>
      <ul>
        {workstreams.map((stream, i) => (
          <WorkstreamRow
            // A stream has no id until it starts, so fall back to position:
            // the plan's order is stable and the placeholder is short-lived.
            key={stream.id || `planned-${i}`}
            stream={stream}
            busy={busy}
            onAdopt={onAdopt}
            onSeal={onSeal}
          />
        ))}
      </ul>
    </section>
  );
}

function WorkstreamRow({
  stream,
  busy,
  onAdopt,
  onSeal,
}: {
  stream: Workstream;
  busy: boolean;
  onAdopt: (id: string) => void;
  onSeal: (id: string, reason: string) => void;
}) {
  const [sealing, setSealing] = useState(false);
  const [reason, setReason] = useState("");

  const glyph =
    stream.status === "finished" ? "✓" : stream.status === "sealed" ? "—" : "⟳";

  return (
    <li className={`workstream ${stream.status}`}>
      <div className="workstream-head">
        <span className="glyph" aria-hidden>
          {glyph}
        </span>
        <span className="title">{stream.title}</span>
        {stream.status === "finished" && (
          <span className="count">
            {stream.changes.length} file{stream.changes.length === 1 ? "" : "s"}
          </span>
        )}
      </div>

      {stream.brief && <p className="brief">{stream.brief}</p>}

      {stream.changes.length > 0 && (
        <ul className="changes">
          {stream.changes.map((change) => (
            <li key={change.path}>
              <span className={`status ${change.status}`}>{change.status}</span>
              <code>{change.path}</code>
            </li>
          ))}
        </ul>
      )}

      {stream.report && <p className="report">{stream.report}</p>}

      {stream.epitaph && (
        <p className="epitaph">
          <strong>Abandoned:</strong> {stream.epitaph}
        </p>
      )}

      {stream.status === "finished" && !sealing && (
        <div className="actions">
          <button
            className="btn primary"
            disabled={busy || !stream.id}
            onClick={() => onAdopt(stream.id)}
            title="Write this workstream's changes into your folder"
          >
            Keep this
          </button>
          <button
            className="btn"
            disabled={busy || !stream.id}
            onClick={() => setSealing(true)}
            title="Abandon it, recording why so a later attempt is told"
          >
            Abandon…
          </button>
        </div>
      )}

      {sealing && (
        <form
          className="seal"
          onSubmit={(e) => {
            e.preventDefault();
            if (!reason.trim()) return;
            onSeal(stream.id, reason.trim());
            setSealing(false);
            setReason("");
          }}
        >
          <label htmlFor={`why-${stream.id}`}>Why is this the wrong result?</label>
          <input
            id={`why-${stream.id}`}
            value={reason}
            autoFocus
            placeholder="e.g. rewriting lost the citations, which are the point"
            onChange={(e) => setReason(e.target.value)}
          />
          <div className="actions">
            {/* Required, not optional: an empty epitaph teaches nothing. */}
            <button className="btn primary" type="submit" disabled={!reason.trim()}>
              Abandon
            </button>
            <button className="btn" type="button" onClick={() => setSealing(false)}>
              Cancel
            </button>
          </div>
        </form>
      )}
    </li>
  );
}
