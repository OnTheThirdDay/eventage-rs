import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { renderMarkdown } from "../lib/markdown";

export function Markdown({ text }: { text: string }) {
  // Code blocks get a copy button from the renderer; the click is handled
  // here by delegation so each block does not need its own React root.
  const onClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    const button = (e.target as HTMLElement).closest(".copy-code");
    if (!button) return;
    const code = button.parentElement?.querySelector("code");
    if (!code) return;
    const ok = await copyText(code.textContent ?? "");
    button.textContent = ok ? "Copied" : "Failed";
    setTimeout(() => {
      button.textContent = "Copy";
    }, 1400);
  };

  return (
    <div
      className="prose"
      onClick={onClick}
      dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }}
    />
  );
}

export function Spinner() {
  return <span className="spinner" aria-label="working" />;
}

/** JSON with syntax colouring, rendered without a dependency. */
export function Json({ value, indent = 2 }: { value: unknown; indent?: number }) {
  let text: string;
  try {
    text = JSON.stringify(value, null, indent) ?? "undefined";
  } catch {
    text = String(value);
  }
  return (
    <pre className="json" dangerouslySetInnerHTML={{ __html: colour(text) }} />
  );
}

const escape = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function colour(json: string): string {
  return escape(json).replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let cls = "number";
      if (match.startsWith('"')) {
        cls = match.endsWith(":") ? "key" : "string";
      } else if (/true|false/.test(match)) {
        cls = "boolean";
      } else if (/null/.test(match)) {
        cls = "null";
      }
      return `<span class="${cls}">${match}</span>`;
    },
  );
}

/** A dropdown that closes on outside click and on Escape. */
export function Menu({
  trigger,
  children,
  align = "left",
  direction = "down",
}: {
  trigger: (props: { open: boolean; toggle: () => void }) => ReactNode;
  children: (close: () => void) => ReactNode;
  align?: "left" | "right";
  direction?: "up" | "down";
}) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onPointer = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="select" ref={box}>
      {trigger({ open, toggle: () => setOpen((v) => !v) })}
      {open && (
        <div className={`menu ${direction} ${align === "right" ? "right" : ""}`}>
          {children(close)}
        </div>
      )}
    </div>
  );
}

/** A draggable divider that reports a new width in pixels. */
export function Resizer({
  onResize,
  side,
  style,
}: {
  onResize: (delta: number) => void;
  side: "left" | "right";
  style?: React.CSSProperties;
}) {
  const [dragging, setDragging] = useState(false);
  const startX = useRef(0);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => {
      const delta = e.clientX - startX.current;
      startX.current = e.clientX;
      onResize(side === "left" ? -delta : delta);
    };
    const onUp = () => setDragging(false);
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
    // Stop the drag selecting text across the whole app.
    document.body.style.userSelect = "none";
    return () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      document.body.style.userSelect = "";
    };
  }, [dragging, onResize, side]);

  return (
    <div
      className={`resizer ${dragging ? "dragging" : ""}`}
      style={style}
      onPointerDown={(e) => {
        startX.current = e.clientX;
        setDragging(true);
      }}
    />
  );
}

/** Grow a textarea to fit its content, up to the CSS max-height. */
export function useAutoSize(value: string) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [value]);
  return ref;
}

/** Persist a value in localStorage, tolerating a blocked or full store. */
export function useStored<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : (JSON.parse(raw) as T);
    } catch {
      return fallback;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* private browsing, quota — not worth breaking the app over */
    }
  }, [key, value]);
  return [value, setValue] as const;
}

export const formatTokens = (n: number): string =>
  n >= 1_000_000
    ? `${(n / 1_000_000).toFixed(1)}M`
    : n >= 1000
      ? `${(n / 1000).toFixed(1)}k`
      : String(n);

export const formatDuration = (ms: number): string =>
  ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;

export const formatTime = (iso: string): string => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleTimeString(undefined, {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
};

export const formatWhen = (iso: string): string => {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return days < 30 ? `${days}d ago` : d.toLocaleDateString();
};

/**
 * Put text on the clipboard, reporting whether it landed.
 *
 * `navigator.clipboard` needs a secure context; Studio is served over
 * loopback, which counts as one. The fallback covers the odd browser that
 * still refuses, so the button is never a no-op that fails silently.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const staging = document.createElement("textarea");
      staging.value = text;
      staging.style.position = "fixed";
      staging.style.opacity = "0";
      document.body.appendChild(staging);
      staging.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(staging);
      return ok;
    } catch {
      return false;
    }
  }
}

/** A copy button that confirms itself for a moment, then goes quiet again. */
export function CopyButton({
  text,
  label = "Copy",
  className = "btn sm ghost",
  title,
}: {
  text: string | (() => string);
  label?: string;
  className?: string;
  title?: string;
}) {
  const [state, setState] = useState<"idle" | "done" | "failed">("idle");

  useEffect(() => {
    if (state === "idle") return;
    const timer = setTimeout(() => setState("idle"), 1400);
    return () => clearTimeout(timer);
  }, [state]);

  return (
    <button
      className={`${className} copy-btn ${state}`}
      title={title ?? "Copy to clipboard"}
      onClick={async (e) => {
        e.stopPropagation();
        const value = typeof text === "function" ? text() : text;
        setState((await copyText(value)) ? "done" : "failed");
      }}
    >
      {state === "done" ? "Copied" : state === "failed" ? "Failed" : label}
    </button>
  );
}
