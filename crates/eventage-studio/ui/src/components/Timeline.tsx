/**
 * The session on a time axis: one lane per participant, tool calls as bars.
 *
 * Drawn on a canvas because a busy session produces thousands of marks and a
 * DOM node each would crawl. Hit-testing is done against the same geometry
 * the renderer uses, so what you click is what you see.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TimelineModel } from "../lib/timeline";

const LANE_H = 26;
const HEADER_H = 18;
const GUTTER = 96;
const PAD_R = 12;

interface Hit {
  kind: "span" | "mark" | "checkpoint";
  seq: number;
  label: string;
  detail: string;
  x: number;
  y: number;
}

function cssVar(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || fallback;
}

export function Timeline({
  model,
  position,
  onSeek,
  onSelect,
  height,
}: {
  model: TimelineModel;
  /** Sequence number of the playhead. */
  position: number;
  onSeek: (seq: number) => void;
  onSelect: (seq: number) => void;
  height: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<Hit | null>(null);
  const dragging = useRef(false);

  const laneIndex = useMemo(() => {
    const index = new Map<string, number>();
    model.lanes.forEach((lane, i) => index.set(lane.id, i));
    return index;
  }, [model.lanes]);

  const span = model.endMs - model.startMs;
  const plotW = Math.max(40, width - GUTTER - PAD_R);
  const xOf = useCallback(
    (ms: number) => GUTTER + ((ms - model.startMs) / span) * plotW,
    [model.startMs, span, plotW],
  );
  const yOf = useCallback(
    (lane: string) => HEADER_H + (laneIndex.get(lane) ?? 0) * LANE_H + LANE_H / 2,
    [laneIndex],
  );

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    // Ignore sub-pixel churn. Redrawing changes the canvas height, which can
    // bring a scrollbar in or out in the scrolling parent, which changes the
    // width by a few pixels — and a naive observer then oscillates forever
    // instead of settling.
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const next = Math.round(entry.contentRect.width);
      setWidth((current) => (Math.abs(current - next) > 2 ? next : current));
    });
    observer.observe(el);
    setWidth(Math.round(el.clientWidth));
    return () => observer.disconnect();
  }, []);

  // ── Draw ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const h = Math.max(height, HEADER_H + model.lanes.length * LANE_H + 8);
    c.width = width * dpr;
    c.height = h * dpr;
    c.style.height = `${h}px`;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, h);

    const colours = {
      border: cssVar("--border", "#e2e5ea"),
      faint: cssVar("--text-faint", "#8a93a0"),
      muted: cssVar("--text-muted", "#626b7a"),
      text: cssVar("--text", "#16181d"),
      accent: cssVar("--accent", "#3b5bdb"),
      warn: cssVar("--warn", "#a16207"),
      danger: cssVar("--danger", "#c0392f"),
      success: cssVar("--success", "#17803d"),
      sunken: cssVar("--bg-sunken", "#f6f7f9"),
    };
    const mono = '11px ui-monospace, "SF Mono", Menlo, monospace';

    // Turn bands, so the eye can group a turn's activity at a glance.
    model.turns.forEach((turn, i) => {
      if (i % 2 === 1) return;
      const x0 = xOf(turn.startMs);
      const x1 = xOf(turn.complete ? turn.endMs : model.endMs);
      ctx.fillStyle = colours.sunken;
      ctx.fillRect(x0, HEADER_H, Math.max(1, x1 - x0), model.lanes.length * LANE_H);
    });

    // Lane rows and labels.
    ctx.font = mono;
    ctx.textBaseline = "middle";
    model.lanes.forEach((lane, i) => {
      const y = HEADER_H + i * LANE_H;
      ctx.strokeStyle = colours.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(width, y + 0.5);
      ctx.stroke();

      ctx.fillStyle = lane.kind === "tool" ? colours.muted : colours.text;
      const label =
        lane.label.length > 13 ? `${lane.label.slice(0, 12)}…` : lane.label;
      ctx.textAlign = "right";
      ctx.fillText(label, GUTTER - 10, y + LANE_H / 2);
    });

    // Time ruler.
    ctx.textAlign = "center";
    ctx.fillStyle = colours.faint;
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const ms = model.startMs + (span * i) / ticks;
      const x = xOf(ms);
      const label =
        span > 120_000
          ? `${((ms - model.startMs) / 1000 / 60).toFixed(1)}m`
          : `${((ms - model.startMs) / 1000).toFixed(1)}s`;
      ctx.fillText(label, x, HEADER_H / 2);
    }

    // Tool spans: a bar per call, which is what makes concurrency visible.
    for (const s of model.spans) {
      const x0 = xOf(s.startMs);
      const x1 = Math.max(x0 + 3, xOf(s.endMs));
      const y = yOf(s.lane);
      ctx.globalAlpha = s.rolledBack ? 0.28 : 1;
      ctx.fillStyle =
        s.status === "failed"
          ? colours.danger
          : s.status === "running"
            ? colours.warn
            : colours.accent;
      const barH = 9;
      roundRect(ctx, x0, y - barH / 2, x1 - x0, barH, 3);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Point events.
    for (const mark of model.marks) {
      const x = xOf(mark.atMs);
      const y = yOf(mark.lane);
      ctx.globalAlpha = mark.rolledBack ? 0.28 : 1;
      ctx.fillStyle =
        mark.kind === "user.message"
          ? colours.success
          : mark.kind.startsWith("permission")
            ? colours.warn
            : mark.kind.startsWith("budget") || mark.kind === "agent.stuck"
              ? colours.danger
              : colours.accent;
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Checkpoint flags — the anchors a rewind can return to.
    for (const cp of model.checkpoints) {
      const x = xOf(cp.atMs);
      ctx.strokeStyle = cp.used ? colours.faint : colours.warn;
      ctx.setLineDash(cp.used ? [3, 3] : []);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, HEADER_H);
      ctx.lineTo(x + 0.5, HEADER_H + model.lanes.length * LANE_H);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = cp.used ? colours.faint : colours.warn;
      ctx.beginPath();
      ctx.moveTo(x, HEADER_H);
      ctx.lineTo(x + 7, HEADER_H + 4);
      ctx.lineTo(x, HEADER_H + 8);
      ctx.closePath();
      ctx.fill();
    }

    // Playhead.
    const posMs = msOfSeq(model, position);
    const px = xOf(posMs);
    ctx.strokeStyle = colours.danger;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
    ctx.fillStyle = colours.danger;
    ctx.beginPath();
    ctx.moveTo(px - 4, 0);
    ctx.lineTo(px + 4, 0);
    ctx.lineTo(px, 6);
    ctx.closePath();
    ctx.fill();
  }, [model, width, height, position, xOf, yOf, span]);

  // ── Interaction ───────────────────────────────────────────────────────────

  const hitTest = useCallback(
    (px: number, py: number): Hit | null => {
      for (const s of model.spans) {
        const x0 = xOf(s.startMs);
        const x1 = Math.max(x0 + 3, xOf(s.endMs));
        const y = yOf(s.lane);
        if (px >= x0 - 2 && px <= x1 + 2 && Math.abs(py - y) <= 6) {
          const ms = s.endMs - s.startMs;
          return {
            kind: "span",
            seq: s.startSeq,
            label: s.name,
            detail: `${s.status} · ${ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`}`,
            x: (x0 + x1) / 2,
            y,
          };
        }
      }
      for (const cp of model.checkpoints) {
        const x = xOf(cp.atMs);
        if (Math.abs(px - x) <= 5 && py <= HEADER_H + 10) {
          return {
            kind: "checkpoint",
            seq: cp.seq,
            label: `Checkpoint · turn ${cp.turn}`,
            detail: cp.used ? "already rewound past" : "click to seek here",
            x,
            y: HEADER_H,
          };
        }
      }
      for (const mark of model.marks) {
        const x = xOf(mark.atMs);
        const y = yOf(mark.lane);
        if (Math.abs(px - x) <= 5 && Math.abs(py - y) <= 6) {
          return { kind: "mark", seq: mark.seq, label: mark.label, detail: mark.kind, x, y };
        }
      }
      return null;
    },
    [model, xOf, yOf],
  );

  const seqAtX = useCallback(
    (px: number): number => {
      const ratio = Math.min(1, Math.max(0, (px - GUTTER) / plotW));
      const ms = model.startMs + ratio * span;
      // Snap to the nearest event so the playhead always lands on something
      // real rather than between two events.
      let best = 0;
      let bestGap = Infinity;
      for (const mark of model.marks) {
        const gap = Math.abs(mark.atMs - ms);
        if (gap < bestGap) {
          bestGap = gap;
          best = mark.seq;
        }
      }
      for (const s of model.spans) {
        const gap = Math.abs(s.startMs - ms);
        if (gap < bestGap) {
          bestGap = gap;
          best = s.startSeq;
        }
      }
      return best || Math.round(ratio * model.lastSeq);
    },
    [model, plotW, span],
  );

  const pointAt = (e: React.PointerEvent) => {
    const rect = canvas.current?.getBoundingClientRect();
    return rect
      ? { x: e.clientX - rect.left, y: e.clientY - rect.top }
      : { x: 0, y: 0 };
  };

  return (
    <div className="timeline" ref={box}>
      <canvas
        ref={canvas}
        style={{ width: "100%", display: "block", cursor: "pointer" }}
        onPointerDown={(e) => {
          const { x, y } = pointAt(e);
          const hit = hitTest(x, y);
          if (hit) {
            onSelect(hit.seq);
            onSeek(hit.seq);
          } else {
            dragging.current = true;
            (e.target as Element).setPointerCapture(e.pointerId);
            onSeek(seqAtX(x));
          }
        }}
        onPointerMove={(e) => {
          const { x, y } = pointAt(e);
          if (dragging.current) {
            onSeek(seqAtX(x));
            return;
          }
          setHover(hitTest(x, y));
        }}
        onPointerUp={(e) => {
          dragging.current = false;
          (e.target as Element).releasePointerCapture?.(e.pointerId);
        }}
        onPointerLeave={() => setHover(null)}
      />
      {hover && (
        <div
          className="timeline-tip"
          style={{ left: Math.min(hover.x + 10, width - 190), top: hover.y + 12 }}
        >
          <strong>{hover.label}</strong>
          <span>{hover.detail}</span>
        </div>
      )}
    </div>
  );
}

/** Where the playhead sits in time, given the event it points at. */
function msOfSeq(model: TimelineModel, seq: number): number {
  let best = model.startMs;
  for (const mark of model.marks) {
    if (mark.seq <= seq) best = Math.max(best, mark.atMs);
  }
  for (const s of model.spans) {
    if (s.startSeq <= seq) best = Math.max(best, s.startMs);
  }
  return seq >= model.lastSeq ? model.endMs : best;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}
