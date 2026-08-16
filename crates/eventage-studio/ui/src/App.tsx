import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chat } from "./components/Chat";
import { Workstreams } from "./components/Workstreams";
import { Composer } from "./components/Composer";
import { RewindDialog } from "./components/RewindDialog";
import { Sidebar } from "./components/Sidebar";
import { Trace } from "./components/Trace";
import type { Tab } from "./components/Trace";
import { WorkspacePicker } from "./components/WorkspacePicker";
import { Menu, Resizer, useStored } from "./components/primitives";
import { api, streamSession } from "./lib/api";
import type { StreamHandle, StreamStatus } from "./lib/api";
import { reduce } from "./lib/reduce";
import { buildTimeline } from "./lib/timeline";
import type { Checkpoint } from "./lib/timeline";
import type {
  AppInfo,
  PromptBlock,
  SessionInfo,
  StoredSession,
  StudioEvent,
} from "./lib/types";

type Theme = "system" | "light" | "dark";

const SUGGESTIONS = [
  "Explain how this project is structured and where the entry points are.",
  "Find and fix any failing tests.",
  "Review my uncommitted changes and point out anything risky.",
  "Add a test that covers the edge case I most likely missed.",
];

const basename = (path: string) =>
  path.split(/[/\\]/).filter(Boolean).pop() ?? path;

export default function App() {
  const [app, setApp] = useState<AppInfo | null>(null);
  const [open, setOpen] = useState<SessionInfo[]>([]);
  const [stored, setStored] = useState<StoredSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [events, setEvents] = useState<StudioEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("closed");
  const [toasts, setToasts] = useState<{ id: number; message: string }[]>([]);
  const [picking, setPicking] = useState(false);
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [rewinding, setRewinding] = useState<Checkpoint | null>(null);

  // The playhead. `live` means it follows the newest event; taking hold of
  // the transport pins it, and the transcript follows it too — scrubbing is
  // replay, not a filter.
  const [position, setPosition] = useState(0);
  // A message the context pane is pinned to, independent of the playhead.
  const [contextFocus, setContextFocus] = useState<number | null>(null);
  const [live, setLive] = useState(true);

  const [showTrace, setShowTrace] = useStored("studio.trace", true);
  const [showSidebar, setShowSidebar] = useStored("studio.sidebar", true);
  const [traceWidth, setTraceWidth] = useStored("studio.traceWidth", 520);
  const [sidebarWidth, setSidebarWidth] = useStored("studio.sidebarWidth", 248);
  const [traceExpanded, setTraceExpanded] = useState(false);
  const [traceTab, setTraceTab] = useState<Tab>("timeline");
  const [theme, setTheme] = useStored<Theme>("studio.theme", "system");

  const streamRef = useRef<StreamHandle | null>(null);
  /// Sequence numbers already held, so a position can never be filled twice.
  const seenSeqs = useRef<Set<number>>(new Set());
  /// Event ids already held. Unlike a sequence number, an id means the same
  /// thing across a restart.
  const seenIds = useRef<Set<string>>(new Set());
  const highestSeq = useRef(0);

  /// Adopt a freshly fetched history as the whole truth.
  ///
  /// The three pieces have to move together — a refetch that replaced the
  /// events but left the seen-sets behind would either drop what came next or
  /// admit what it already had.
  const reset = useCallback((history: StudioEvent[]) => {
    seenSeqs.current = new Set(history.map((e) => e.seq));
    seenIds.current = new Set(history.map((e) => e.id));
    highestSeq.current = history.length ? history[history.length - 1]!.seq : 0;
    setEvents(history);
  }, []);
  const toastId = useRef(0);
  const notify = useCallback((message: string) => {
    const id = ++toastId.current;
    setToasts((list) => [...list, { id, message }]);
    setTimeout(() => setToasts((l) => l.filter((t) => t.id !== id)), 7000);
  }, []);

  const guard = useCallback(
    async (work: () => Promise<unknown>) => {
      try {
        await work();
      } catch (e) {
        notify(e instanceof Error ? e.message : String(e));
      }
    },
    [notify],
  );

  useEffect(() => {
    document.documentElement.setAttribute(
      "data-theme",
      theme === "system" ? "" : theme,
    );
  }, [theme]);

  const refreshSessions = useCallback(async () => {
    const listing = await api.sessions();
    setOpen(listing.open);
    setStored(listing.stored);
    return listing;
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const info = await api.app();
        setApp(info);
        setWorkspace(info.default_cwd);
        const listing = await refreshSessions();
        if (listing.open.length > 0) {
          setActiveId(listing.open[0]!.id);
        } else {
          const session = await api.open({ cwd: info.default_cwd });
          await refreshSessions();
          setActiveId(session.id);
        }
      } catch (e) {
        notify(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [refreshSessions, notify]);

  // Follow the active session: history first, then live.
  useEffect(() => {
    if (!activeId) {
      setEvents([]);
      return;
    }
    let cancelled = false;
    let stop: StreamHandle | undefined;
    setLive(true);

    void (async () => {
      try {
        const history = await api.events(activeId, 0);
        if (cancelled) return;
        reset(history);
        const from = highestSeq.current;
        streamRef.current = stop = streamSession(
          activeId,
          from,
          (event) => {
            // Two guards, because they answer different questions.
            //
            // The sequence number says *where* an event belongs, and keying
            // on it is what stopped an out-of-order arrival being dropped —
            // that showed as a message stopping a few lines in.
            //
            // The id says *what* the event is, and that is the one that
            // survives a restart: sequence numbers are assigned per process,
            // so a rebuilt feed numbers the same session differently and a
            // client resuming from an old number can be handed a turn it is
            // already showing. Then it renders twice.
            if (seenIds.current.has(event.id)) return;
            if (seenSeqs.current.has(event.seq)) return;

            // A gap means something was missed rather than merely delayed —
            // a dropped frame, a reconnect that resumed from the wrong
            // point. Left alone it shows as a message that stops partway, so
            // fetch what is missing instead of rendering a hole.
            const expected = seenSeqs.current.size ? highestSeq.current + 1 : event.seq;
            if (event.seq > expected) {
              void guard(async () => {
                const missing = await api.events(activeId, expected - 1);
                // Bookkeeping stays outside the state updater: React may
                // invoke an updater more than once, and a second pass over
                // ref mutations would see its own first pass as "already
                // seen" and return a list with the fillers dropped.
                const fresh = missing.filter((f) => !seenIds.current.has(f.id));
                for (const filler of fresh) {
                  seenIds.current.add(filler.id);
                  seenSeqs.current.add(filler.seq);
                  highestSeq.current = Math.max(highestSeq.current, filler.seq);
                }
                if (fresh.length) {
                  setEvents((current) =>
                    [...current, ...fresh].sort((a, b) => a.seq - b.seq),
                  );
                }
              });
            }

            seenIds.current.add(event.id);
            seenSeqs.current.add(event.seq);
            highestSeq.current = Math.max(highestSeq.current, event.seq);
            setEvents((current) => {
              const next = [...current, event];
              // Cheap in the common case: the new event almost always belongs
              // at the end, and sort leaves an already-ordered array alone.
              next.sort((a, b) => a.seq - b.seq);
              return next;
            });
          },
          setStatus,
          () => {
            // The feed was rebuilt, so nothing we hold is addressable any
            // more. Refetch rather than trying to reconcile two numberings.
            void guard(async () => reset(await api.events(activeId, 0)));
          },
        );
      } catch (e) {
        if (!cancelled) notify(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      stop?.();
    };
  }, [activeId, notify, guard, reset]);

  const lastSeq = events.length ? events[events.length - 1]!.seq : 0;

  // While live, the playhead sits at the tip.
  useEffect(() => {
    if (live) setPosition(lastSeq);
  }, [live, lastSeq]);

  const seek = useCallback((seq: number) => {
    setLive(false);
    setPosition(seq);
    // Scrubbing means "show me this moment", so a pinned message stops
    // overriding the pane.
    setContextFocus(null);
  }, []);

  const goLive = useCallback(() => setLive(true), []);

  // The transcript renders history up to the playhead; the trace always shows
  // everything, dimming what is ahead of it.
  const shown = useMemo(
    () => (live ? events : events.filter((e) => e.seq <= position)),
    [live, events, position],
  );
  const chat = useMemo(() => reduce(shown), [shown]);
  const fullChat = useMemo(() => reduce(events), [events]);
  const model = useMemo(() => buildTimeline(events), [events]);
  const active = open.find((s) => s.id === activeId) ?? null;

  useEffect(() => {
    void refreshSessions();
  }, [fullChat.running, refreshSessions]);

  const send = useCallback(
    (blocks: PromptBlock[]) => {
      if (!activeId) return;
      goLive();
      void guard(() => api.prompt(activeId, blocks));
    },
    [activeId, guard, goLive],
  );

  /// Show what the model was given when it produced this message.
  ///
  /// This deliberately does *not* move the playhead. Reusing it looked
  /// economical, but asking "what did it see here?" rewound the transcript
  /// underneath you — and landing on the message's own event left the turn
  /// looking unfinished, since the events that close it come after.
  const inspectContext = useCallback((seq: number) => {
    setShowTrace(true);
    setTraceTab("context");
    setContextFocus(seq);
  }, []);

  /// Fork the conversation here, leaving this one intact.
  ///
  /// Rewinding edits a session in place; branching starts a sibling from the
  /// same history, which is what you want when the current direction might
  /// still turn out to be the right one.
  const branchFrom = useCallback(
    (seq: number) =>
      guard(async () => {
        if (!activeId) return;
        const branched = await api.branch(activeId, seq);
        await refreshSessions();
        setActiveId(branched.id);
      }),
    [activeId, guard, refreshSessions],
  );

  /** Write one workstream's result into the folder. */
  const adoptWorkstream = useCallback(
    (workstreamId: string) => {
      if (!activeId) return;
      void guard(() => api.adopt(activeId, workstreamId));
    },
    [activeId, guard],
  );

  /** Abandon a workstream, recording why for a later attempt to read. */
  const sealWorkstream = useCallback(
    (workstreamId: string, reason: string) => {
      if (!activeId) return;
      void guard(() => api.seal(activeId, workstreamId, reason));
    },
    [activeId, guard],
  );

  const answerPermission = useCallback(
    (requestId: string, approve: boolean, always: boolean) => {
      if (!activeId) return;
      void guard(() =>
        api.permission(activeId, {
          request_id: requestId,
          approve,
          always,
          ...(approve ? {} : { reason: "the user rejected this action" }),
        }),
      );
    },
    [activeId, guard],
  );

  const newSession = useCallback(
    (cwd?: string) =>
      guard(async () => {
        const session = await api.open({
          ...(cwd ? { cwd } : workspace ? { cwd: workspace } : {}),
        });
        await refreshSessions();
        setActiveId(session.id);
      }),
    [guard, refreshSessions, workspace],
  );

  const resume = useCallback(
    (session: StoredSession) =>
      guard(async () => {
        const opened = await api.open({ cwd: session.cwd, resume: session.id });
        await refreshSessions();
        setActiveId(opened.id);
      }),
    [guard, refreshSessions],
  );

  const forget = useCallback(
    (id: string) =>
      guard(async () => {
        await api.forget(id);
        await refreshSessions();
        if (activeId === id) setActiveId(null);
      }),
    [guard, refreshSessions, activeId],
  );

  const doRewind = useCallback(
    (checkpoint: Checkpoint) =>
      guard(async () => {
        if (!activeId) return;
        await api.rewind(activeId, { to: checkpoint.eventId });
        setRewinding(null);
        goLive();
        reset(await api.events(activeId, 0));
        await refreshSessions();
      }),
    [activeId, guard, refreshSessions, goLive],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      if (e.key === "n") {
        e.preventDefault();
        void newSession();
      } else if (e.key === "b") {
        e.preventDefault();
        setShowSidebar((v) => !v);
      } else if (e.key === "j") {
        e.preventDefault();
        setShowTrace((v) => !v);
      } else if (e.key === "." && activeId) {
        e.preventDefault();
        void guard(() => api.interrupt(activeId));
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [newSession, setShowSidebar, setShowTrace, activeId, guard]);

  if (!app) {
    return (
      <div className="boot">
        <div className="brand">
          <span className="dot" />
          Eventage Studio
        </div>
        <p>Connecting…</p>
      </div>
    );
  }

  const workspaceName = basename(active?.cwd ?? app.default_cwd);

  const emptyState = (
    <div className="empty">
      <h2>{active ? "What should we work on?" : "No session open"}</h2>
      <p>
        {app.credentials_hint
          ? app.credentials_hint
          : app.backend === "local"
            ? `${app.model} · ${workspaceName}`
            : `Connected to ${app.backend_detail}`}
      </p>
      {active && !app.credentials_hint && (
        <div className="suggestions">
          {SUGGESTIONS.map((text) => (
            <button key={text} onClick={() => send([{ type: "text", text }])}>
              {text}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="app">
      <header className="titlebar">
        <button
          className="btn sm ghost icon"
          onClick={() => setShowSidebar((v) => !v)}
          title="Toggle sessions (⌘B)"
        >
          ☰
        </button>
        <div className="brand">
          <span className="dot" />
          Studio
        </div>

        <span className="divider" />

        {active && (
          <span className="workspace-chip" title={active.cwd}>
            {workspaceName}
          </span>
        )}
        <span className="spacer" />

        {active && model.checkpoints.length > 0 && (
          <button
            className="btn sm ghost"
            disabled={fullChat.running}
            onClick={() => {
              const last = model.checkpoints.filter((c) => !c.used).pop();
              if (last) setRewinding(last);
            }}
            title="Undo the last turn"
          >
            ↶ Rewind
          </button>
        )}

        <span className="model-chip" title={`${app.provider} · ${app.model}`}>
          {app.backend === "local" ? app.model : "ACP"}
        </span>

        <Menu
          align="right"
          trigger={({ toggle }) => (
            <button className="btn sm ghost icon" onClick={toggle} title="Appearance">
              ◐
            </button>
          )}
        >
          {(close) =>
            (["system", "light", "dark"] as Theme[]).map((option) => (
              <button
                key={option}
                className={option === theme ? "selected" : ""}
                onClick={() => {
                  setTheme(option);
                  close();
                }}
              >
                {option[0]!.toUpperCase() + option.slice(1)}
              </button>
            ))
          }
        </Menu>

        <button
          className={`btn sm ${showTrace ? "toggled" : ""}`}
          onClick={() => setShowTrace((v) => !v)}
          title="Toggle the trace (⌘J)"
        >
          Trace
        </button>
      </header>

      {app.credentials_hint && (
        <div className="setup-banner">
          <span aria-hidden>!</span>
          <span>{app.credentials_hint}</span>
        </div>
      )}

      <div
        className={`body ${showTrace ? "with-trace" : ""} ${
          showSidebar ? "" : "no-sidebar"
        } ${traceExpanded ? "trace-only" : ""}`}
        style={
          {
            "--trace-w": `${traceWidth}px`,
            "--sidebar-w": `${sidebarWidth}px`,
          } as React.CSSProperties
        }
      >
        {showSidebar && !traceExpanded && (
          <div className="pane">
            <Sidebar
              open={open}
              stored={stored}
              activeId={activeId}
              onSelect={setActiveId}
              onResume={(s) => void resume(s)}
              onNew={() => void newSession()}
              onPickWorkspace={() => setPicking(true)}
              onForget={(id) => void forget(id)}
            />
            <Resizer
              side="right"
              style={{ right: -4 }}
              onResize={(delta) =>
                setSidebarWidth((w) => Math.min(420, Math.max(180, w + delta)))
              }
            />
          </div>
        )}

        {!traceExpanded && (
          <main className="main">
            {!live && (
              <div className="replay-banner">
                <span aria-hidden>⏱</span>
                <span>
                  Replaying history — event <b>{position}</b> of {lastSeq}
                </span>
                <span className="spacer" />
                <button className="btn sm primary" onClick={goLive}>
                  Return to live
                </button>
              </div>
            )}

            <div className="main-inner">
              <Chat
                items={chat.items}
                plan={chat.plan}
                running={chat.running}
                onPermission={answerPermission}
                onInspectContext={inspectContext}
                onBranchFrom={(seq) => void branchFrom(seq)}
                emptyState={emptyState}
              />
              {/* Below the transcript rather than beside it: the comparison
                  is what you do *after* reading what each stream reported. */}
              <Workstreams
                workstreams={chat.workstreams}
                busy={fullChat.running}
                onAdopt={(ws) => void adoptWorkstream(ws)}
                onSeal={(ws, reason) => void sealWorkstream(ws, reason)}
              />
            </div>

            <Composer
              disabled={!activeId}
              running={fullChat.running}
              modes={app.modes}
              mode={active?.mode ?? app.modes[0]?.id ?? "ask"}
              onSend={send}
              onInterrupt={() =>
                activeId && void guard(() => api.interrupt(activeId))
              }
              onModeChange={(mode) =>
                activeId &&
                void guard(async () => {
                  await api.setMode(activeId, mode);
                  await refreshSessions();
                })
              }
            />

            <div
              className={`status-strip ${
                status === "live" ? "" : status === "lost" ? "offline" : "waiting"
              }`}
            >
              <span className="dot" />
              <span>
                {status === "live"
                  ? "connected"
                  : status === "reconnecting"
                    ? "reconnecting…"
                    : status === "connecting"
                      ? "connecting…"
                      : status === "lost"
                        ? "disconnected"
                        : "not streaming"}
              </span>
              {status === "lost" && (
                <button
                  className="btn sm ghost"
                  onClick={() => streamRef.current?.retry()}
                >
                  Retry
                </button>
              )}
              <span className="spacer" />
              <span>
                {fullChat.stats.turns} turn{fullChat.stats.turns === 1 ? "" : "s"} ·{" "}
                {fullChat.stats.toolCalls} tool call
                {fullChat.stats.toolCalls === 1 ? "" : "s"} · {events.length} events
              </span>
            </div>
          </main>
        )}

        {showTrace && (
          <div className="pane">
            {!traceExpanded && (
              <Resizer
                side="left"
                style={{ left: -4 }}
                onResize={(delta) =>
                  setTraceWidth((w) => Math.min(1100, Math.max(360, w + delta)))
                }
              />
            )}
            <Trace
              events={events}
              stats={fullChat.stats}
              rolledBack={fullChat.rolledBack}
              fullTrace={app.full_trace}
              position={position}
              contextFocus={contextFocus}
              onClearContextFocus={() => setContextFocus(null)}
              live={live}
              expanded={traceExpanded}
              onSeek={seek}
              onGoLive={goLive}
              onToggleExpand={() => setTraceExpanded((v) => !v)}
              onClose={() => {
                setTraceExpanded(false);
                setShowTrace(false);
              }}
              tab={traceTab}
              onTab={setTraceTab}
              onRewindTo={setRewinding}
              onOverrideSummary={(summary, covers) =>
                activeId &&
                void guard(async () => {
                  await api.overrideSummary(activeId, summary, covers);
                  reset(await api.events(activeId, 0));
                })
              }
            />
          </div>
        )}
      </div>

      {rewinding && (
        <RewindDialog
          checkpoint={rewinding}
          events={events}
          onCancel={() => setRewinding(null)}
          onConfirm={() => void doRewind(rewinding)}
        />
      )}

      {picking && (
        <WorkspacePicker
          start={workspace ?? app.default_cwd}
          onCancel={() => setPicking(false)}
          onPick={(path) => {
            setPicking(false);
            setWorkspace(path);
            void newSession(path);
          }}
        />
      )}

      {toasts.length > 0 && (
        <div className="toasts">
          {toasts.map((toast) => (
            <div className="toast" key={toast.id}>
              <span style={{ flex: 1 }}>{toast.message}</span>
              <button
                className="close"
                onClick={() =>
                  setToasts((list) => list.filter((t) => t.id !== toast.id))
                }
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
