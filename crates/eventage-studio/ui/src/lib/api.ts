/** Typed client for the Studio server. */

import type {
  ModelSource,
  ModelView,
  AppInfo,
  PromptBlock,
  SessionInfo,
  StoredSession,
  StudioEvent,
} from "./types";

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError("Studio's server is not responding.");
  }

  // Read the body as text first. Several endpoints answer with no body at
  // all — `POST /prompt` returns 202 Accepted because the turn runs
  // asynchronously and the events, not the response, carry the outcome — and
  // calling `.json()` on an empty body throws a parse error that has nothing
  // to do with what went wrong.
  const text = await response.text();
  const parsed = text ? safeParse(text) : null;

  if (!response.ok) {
    // The server sends `{error}` for anything a person should read.
    const message =
      parsed && typeof parsed === "object" && "error" in parsed
        ? String((parsed as { error: unknown }).error)
        : text || `${response.status} ${response.statusText}`;
    throw new ApiError(message);
  }

  return parsed as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });

export const api = {
  app: () => request<AppInfo>("/app"),

  sessions: () =>
    request<{ open: SessionInfo[]; stored: StoredSession[] }>("/sessions"),

  open: (req: { cwd?: string; mode?: string; resume?: string }) =>
    post<SessionInfo>("/sessions", req),

  close: (id: string) => request<void>(`/sessions/${id}`, { method: "DELETE" }),

  forget: (id: string) => request<void>(`/stored/${id}`, { method: "DELETE" }),

  events: (id: string, after = 0) =>
    request<StudioEvent[]>(`/sessions/${id}/events?after=${after}`),

  prompt: (id: string, blocks: PromptBlock[]) =>
    post<void>(`/sessions/${id}/prompt`, { blocks }),

  interrupt: (id: string) => post<void>(`/sessions/${id}/interrupt`),

  setMode: (id: string, mode: string) =>
    post<void>(`/sessions/${id}/mode`, { mode }),

  /** Undo turns, or return to a specific checkpoint. */
  rewind: (id: string, what: { turns?: number; to?: string }) =>
    post<{ remaining: number }>(`/sessions/${id}/rewind`, what),

  /** Fork this session at an event into a new one. */
  branch: (id: string, from_seq: number) =>
    post<SessionInfo>(`/sessions/${id}/branch`, { from_seq }),

  /** Replace the summary compaction produced. */
  overrideSummary: (id: string, summary: string, summarized_count: number) =>
    post<void>(`/sessions/${id}/summary`, { summary, summarized_count }),

  /** Ask an installed reviewer what compaction dropped. */
  requestContextAudit: (id: string) =>
    post<void>(`/sessions/${id}/context/audit`, {}),

  permission: (
    id: string,
    body: {
      request_id: string;
      approve: boolean;
      reason?: string;
      always?: boolean;
    },
  ) => post<void>(`/sessions/${id}/permission`, body),

  /**
   * Write one workstream's result into the folder.
   *
   * Refuses by default when the folder changed under the workstream, and
   * returns the conflicting paths instead of resolving them by overwriting.
   * `force` is the user having seen them and chosen anyway.
   */
  adopt: (id: string, workstream_id: string, force = false) =>
    post<{
      changed: string[];
      conflicts: { path: string; workstream: string; live: string }[];
    }>(`/sessions/${id}/adopt`, { workstream_id, force }),

  /**
   * Abandon a workstream, recording why.
   *
   * The reason is required by the server, not merely encouraged: an epitaph
   * with nothing in it teaches a later attempt nothing, which is the only
   * thing sealing is for.
   */
  seal: (id: string, workstream_id: string, reason: string) =>
    post<void>(`/sessions/${id}/seal`, { workstream_id, reason }),

  /** Provider, model and endpoint — never the key. */
  modelSettings: () => request<ModelView>("/model"),

  /**
   * Change the model sessions are opened with.
   *
   * Omit `api_key` to keep the configured one: the form is never given the
   * key, so it cannot round-trip it, and sending an empty string would sign
   * the user out every time they renamed a model.
   */
  setModelSettings: (body: {
    source: ModelSource;
    provider: string;
    model: string;
    base_url: string;
    api_key?: string;
    remember_key: boolean;
  }) => post<ModelView>("/model", body),

  listDir: (path: string) =>
    request<{
      path: string;
      parent: string | null;
      dirs: { name: string; path: string }[];
    }>(`/fs/list?path=${encodeURIComponent(path)}`),
};

// ── Live stream ───────────────────────────────────────────────────────────────

/** Stops the stream; `retry` starts it again after it has given up. */
export interface StreamHandle {
  (): void;
  retry: () => void;
}

export type StreamStatus =
  | "connecting"
  | "live"
  | "reconnecting"
  | "lost"
  | "closed";

/**
 * How many times to retry before giving up and asking the user.
 *
 * Retrying forever looks tidy but is not: against a server that has actually
 * stopped it becomes an endless request storm, and it never tells the user
 * that nothing is coming. Ten attempts covers a laptop waking or a restart;
 * past that, something needs a person.
 */
const MAX_ATTEMPTS = 10;

/**
 * Follow a session's events, resuming rather than replaying after a drop.
 *
 * `EventSource` reconnects on its own, but always to the URL it was given —
 * which would re-deliver the whole history every time the machine wakes from
 * sleep. So reconnection is handled here, with the sequence number advanced to
 * whatever arrived last.
 *
 * Returns a function that stops the stream.
 */
export function streamSession(
  sessionId: string,
  from: number,
  onEvent: (event: StudioEvent) => void,
  onStatus: (status: StreamStatus) => void,
  /// Called when the server's numbering is not the one `from` came from —
  /// the feed was rebuilt, so the whole history has to be refetched.
  onRenumbered?: () => void,
): StreamHandle {
  let lastSeq = from;
  let generation: string | null = null;
  let source: EventSource | null = null;
  let retry: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;
  let stopped = false;

  const connect = () => {
    if (stopped) return;
    onStatus(attempt === 0 ? "connecting" : "reconnecting");
    source = new EventSource(`/api/sessions/${sessionId}/stream?after=${lastSeq}`);

    source.onopen = () => {
      attempt = 0;
      onStatus("live");
    };

    source.onmessage = (message) => {
      let event: StudioEvent;
      try {
        event = JSON.parse(message.data) as StudioEvent;
      } catch {
        return;
      }
      // The server emits this when a client falls behind its live buffer.
      // Reconnecting from `lastSeq` refills the gap from history.
      if (event.kind === "studio.stream.lagged") {
        source?.close();
        connect();
        return;
      }
      // The server states which numbering it is serving. A different one
      // means our sequence numbers point somewhere else in it, so resuming
      // from them re-delivers events we already have — under new numbers, so
      // they look new. Start over instead.
      if (event.kind === "studio.stream.hello") {
        const seen = (event.payload as { generation?: string })?.generation;
        if (typeof seen === "string") {
          if (generation !== null && generation !== seen) {
            generation = seen;
            lastSeq = 0;
            onRenumbered?.();
          } else {
            generation = seen;
          }
        }
        return;
      }
      if (event.seq > lastSeq) lastSeq = event.seq;
      onEvent(event);
    };

    source.onerror = () => {
      source?.close();
      source = null;
      if (stopped) return;
      if (attempt >= MAX_ATTEMPTS) {
        onStatus("lost");
        return;
      }
      onStatus("reconnecting");
      // Back off towards a couple of seconds: short enough to feel instant on
      // a laptop waking up, long enough not to hammer a stopped server.
      const delay = Math.min(2000, 100 * 2 ** attempt);
      attempt += 1;
      retry = setTimeout(connect, delay);
    };
  };

  connect();

  const stop = () => {
    stopped = true;
    if (retry) clearTimeout(retry);
    source?.close();
    onStatus("closed");
  };
  // Let the caller ask for another go after we have given up.
  stop.retry = () => {
    attempt = 0;
    stopped = false;
    connect();
  };
  return stop;
}
