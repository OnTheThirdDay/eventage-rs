/** The wire model, mirroring `protocol.rs`. */

export interface StudioEvent {
  seq: number;
  id: string;
  ts: string;
  kind: string;
  payload: Record<string, unknown>;
  meta?: Record<string, unknown>;
  parent?: string;
}

export interface ModeInfo {
  id: string;
  label: string;
  description: string;
}

/**
 * The model configuration, as a settings screen sees it.
 *
 * No credential, ever — `has_key` says whether one is configured and that is
 * all the form needs to render correctly.
 */
export interface ModelView {
  provider: string;
  model: string;
  base_url: string;
  has_key: boolean;
  key_remembered: boolean;
  providers: { id: string; label: string; endpoint_hint: string }[];
}

export interface AppInfo {
  backend: "local" | "acp";
  backend_detail: string;
  model: string;
  provider: string;
  default_cwd: string;
  modes: ModeInfo[];
  version: string;
  /** False in ACP mode, where the protocol carries no event log. */
  full_trace: boolean;
  /** Set when the server found no API key, so the app can say so up front. */
  credentials_hint?: string | null;
}

export interface SessionInfo {
  id: string;
  cwd: string;
  mode: string;
  title: string;
  created_at: string;
  running: boolean;
  turns: number;
}

export interface StoredSession {
  id: string;
  cwd: string;
  title: string;
  updated_at: string;
  size_bytes: number;
}

export type PromptBlock =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }
  | { type: "resource_link"; uri: string; name?: string };

// ── Derived view model ────────────────────────────────────────────────────────

export interface FileDiff {
  path: string;
  old_text: string | null;
  new_text: string;
}

export interface SourceLocation {
  path: string;
  line?: number;
}

export type ToolStatus = "running" | "done" | "failed" | "denied";

export interface UserItem {
  type: "user";
  key: string;
  eventId: string;
  /** Position in the stream, so the conversation can be forked here. */
  seq?: number;
  ts: string;
  text: string;
  images: string[];
}

export interface AssistantItem {
  type: "assistant";
  key: string;
  eventId: string;
  /** Position in the stream, so its context can be looked up. */
  seq?: number;
  ts: string;
  text: string;
  thinking: string;
  streaming: boolean;
  tokens?: { input: number; output: number; cached: number };
}

export interface ToolItem {
  type: "tool";
  key: string;
  eventId: string;
  ts: string;
  callId: string;
  name: string;
  title: string;
  args: unknown;
  status: ToolStatus;
  result?: unknown;
  error?: string;
  diff?: FileDiff;
  locations: SourceLocation[];
  durationMs?: number;
}

export interface PermissionItem {
  type: "permission";
  key: string;
  eventId: string;
  ts: string;
  requestId: string;
  tool: string;
  args: unknown;
  status: "pending" | "approved" | "denied";
  reason?: string;
  auto?: string;
}

/** Anything the harness wants the user to know that is not a message. */
export interface NoticeItem {
  type: "notice";
  key: string;
  eventId: string;
  ts: string;
  level: "info" | "warn" | "error";
  title: string;
  detail?: string;
}

export type ChatItem =
  | UserItem
  | AssistantItem
  | ToolItem
  | PermissionItem
  | NoticeItem;

export interface PlanEntry {
  content: string;
  status: "pending" | "in_progress" | "completed";
  priority?: string;
}

export interface SessionStats {
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  toolCalls: number;
  turns: number;
  /** Wall-clock of the most recent completed turn. */
  lastTurnMs?: number;
}

/**
 * One line of work in a cowork session.
 *
 * Derived from the event stream like everything else here, so a browser that
 * joined halfway through a fan-out renders the same picture as one that
 * watched it from the start.
 */
export interface Workstream {
  id: string;
  title: string;
  brief: string;
  status: "running" | "finished" | "sealed";
  /** Files it changed in its own copy, against the session's base. */
  changes: { path: string; status: string }[];
  /** Its own account of what it did. */
  report?: string;
  /** Why it was abandoned, if it was. */
  epitaph?: string;
}

export interface ChatState {
  items: ChatItem[];
  plan: PlanEntry[];
  running: boolean;
  stats: SessionStats;
  /** Ids the log says were rolled back — hidden from the transcript, kept in the trace. */
  rolledBack: Set<string>;
  pendingPermissions: PermissionItem[];
  /**
   * Cowork workstreams, in the order they were planned.
   *
   * Empty for the coding and ACP backends, which have no such thing — the
   * panel simply does not render.
   */
  workstreams: Workstream[];
  /** Paths of the folder cowork is not tracking, and why it said so. */
  untracked: string[];
  /**
   * Rolled-back attempts being fed back into the agent's context.
   *
   * The agent is told about these on every request, which is invisible from
   * the transcript — the warning goes to the model, not the screen.
   */
  sealedAttempts: number;
}
