/**
 * Choosing a provider, endpoint and credential without leaving the app.
 *
 * The form is deliberately shaped by two facts about the server. The key is
 * never sent back — the API reports only whether one is set — so the field
 * starts empty with a placeholder saying a key is already configured, and an
 * untouched field means "leave it alone" rather than "clear it". And a change
 * applies to the *next* session, because the provider is built into a running
 * agent; the dialog says so rather than letting someone wonder why the current
 * conversation did not change model mid-sentence.
 */
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { ModelView } from "../lib/types";

export function ModelSettings({
  onClose,
  onSaved,
}: {
  onClose: () => void;
  onSaved: () => void;
}) {
  const [view, setView] = useState<ModelView | null>(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [remember, setRemember] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void api
      .modelSettings()
      .then((v) => {
        if (cancelled) return;
        setView(v);
        setProvider(v.provider);
        setModel(v.model);
        setBaseUrl(v.base_url);
        setRemember(v.key_remembered);
      })
      .catch((e: unknown) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const chosen = view?.providers.find((p) => p.id === provider);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      await api.setModelSettings({
        provider,
        model,
        base_url: baseUrl,
        // Untouched means untouched. The field cannot be pre-filled with a
        // key it was never given, so sending "" would sign the user out every
        // time they renamed a model.
        ...(apiKey === "" ? {} : { api_key: apiKey }),
        remember_key: remember,
      });
      onSaved();
      onClose();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className="picker-backdrop"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="picker model-settings">
        <h3>Model</h3>

        {view === null ? (
          <p className="muted">Loading…</p>
        ) : (
          <form
            className="model-form"
            onSubmit={(e) => {
              e.preventDefault();
              void save();
            }}
          >
            <label>
              <span>Provider</span>
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
              >
                {view.providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Model</span>
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="e.g. claude-sonnet-4-5"
                autoFocus
              />
            </label>

            <label>
              <span>Endpoint</span>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={chosen?.endpoint_hint ?? ""}
              />
              {chosen && <small className="muted">{chosen.endpoint_hint}</small>}
            </label>

            <label>
              <span>API key</span>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={
                  view.has_key
                    ? "a key is set — leave blank to keep it"
                    : "no key configured"
                }
                autoComplete="off"
              />
              <small className="muted">
                {view.has_key
                  ? "Leave blank to keep the current key, or clear it by typing a space then deleting it."
                  : "Not needed for a local server such as Ollama."}
              </small>
            </label>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />
              <span>
                Remember the key on this machine
                <small className="muted">
                  Written to a file only you can read. Without this it is kept
                  for this run only.
                </small>
              </span>
            </label>

            {error && <p className="model-error">{error}</p>}

            <p className="muted model-note">
              Applies to sessions you open from now on. A conversation already
              running keeps the model it started with.
            </p>

            <div className="picker-actions">
              <button className="btn" type="button" onClick={onClose}>
                Cancel
              </button>
              <button className="btn primary" type="submit" disabled={saving}>
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
