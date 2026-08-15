/**
 * Playback across the session's history.
 *
 * Scrubbing here moves the playhead, and the playhead governs the transcript
 * as well as the trace — so stepping back is genuinely replaying the session,
 * not filtering a list. Live sessions follow the tip until you take hold of
 * the transport, and a clear way back to live is always on screen.
 */

import { useEffect, useRef } from "react";

const SPEEDS = [0.5, 1, 2, 5] as const;

export function Transport({
  position,
  lastSeq,
  playing,
  speed,
  live,
  onSeek,
  onPlayPause,
  onSpeed,
  onGoLive,
}: {
  position: number;
  lastSeq: number;
  playing: boolean;
  speed: number;
  live: boolean;
  onSeek: (seq: number) => void;
  onPlayPause: () => void;
  onSpeed: (speed: number) => void;
  onGoLive: () => void;
}) {
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  // Advance one event at a time. Stepping by event rather than by wall-clock
  // keeps a long think from stalling playback, which is what you want when
  // reading back a session.
  useEffect(() => {
    if (timer.current) clearInterval(timer.current);
    if (!playing) return;
    timer.current = setInterval(() => {
      onSeek(position + 1);
    }, 320 / speed);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, speed, position, onSeek]);

  // Reaching the end stops playback rather than spinning at the tip.
  useEffect(() => {
    if (playing && position >= lastSeq) onPlayPause();
  }, [playing, position, lastSeq, onPlayPause]);

  return (
    <div className="transport">
      <button
        className="btn sm ghost icon"
        onClick={() => onSeek(0)}
        title="Back to the start"
      >
        ⏮
      </button>
      <button
        className="btn sm ghost icon"
        onClick={() => onSeek(position - 1)}
        disabled={position <= 0}
        title="Previous event"
      >
        ◀
      </button>
      <button
        className="btn sm icon"
        onClick={onPlayPause}
        title={playing ? "Pause" : "Play"}
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <button
        className="btn sm ghost icon"
        onClick={() => onSeek(position + 1)}
        disabled={position >= lastSeq}
        title="Next event"
      >
        ▶
      </button>

      <input
        className="scrub"
        type="range"
        min={0}
        max={Math.max(1, lastSeq)}
        value={Math.min(position, lastSeq)}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-label="Scrub through the session"
      />

      <span className="transport-pos">
        {Math.min(position, lastSeq)} / {lastSeq}
      </span>

      <div className="speeds">
        {SPEEDS.map((s) => (
          <button
            key={s}
            className={`chip ${speed === s ? "on" : ""}`}
            onClick={() => onSpeed(s)}
          >
            {s}×
          </button>
        ))}
      </div>

      <button
        className={`btn sm ${live ? "toggled" : "primary"}`}
        onClick={onGoLive}
        title="Follow the newest events"
      >
        {live ? "Live" : "↓ Go live"}
      </button>
    </div>
  );
}
