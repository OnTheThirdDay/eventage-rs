/**
 * The session as a sequence diagram: who asked whom, and what came back.
 *
 * The timeline answers "when, and how long"; this answers "what was the shape
 * of the reasoning". Drawn as SVG rather than canvas because the row count is
 * bounded by interactions rather than by streaming deltas, and SVG gives
 * selectable text and real hit targets for free.
 */

import { useMemo } from "react";
import type { FlowMessage, TimelineModel } from "../lib/timeline";

const ROW_H = 34;
const HEAD_H = 42;
const COL_W = 150;
const PAD_X = 20;

export function Flow({
  messages,
  model,
  position,
  selected,
  onSelect,
}: {
  messages: FlowMessage[];
  model: TimelineModel;
  position: number;
  selected: number | null;
  onSelect: (seq: number) => void;
}) {
  const lanes = useMemo(
    () => model.lanes.filter((l) => l.id !== "system"),
    [model.lanes],
  );
  const columnOf = useMemo(() => {
    const index = new Map<string, number>();
    lanes.forEach((lane, i) => index.set(lane.id, i));
    return index;
  }, [lanes]);

  const width = PAD_X * 2 + Math.max(1, lanes.length) * COL_W;
  const height = HEAD_H + Math.max(1, messages.length) * ROW_H + 20;
  const xOf = (lane: string) =>
    PAD_X + (columnOf.get(lane) ?? 0) * COL_W + COL_W / 2;

  if (!messages.length) {
    return (
      <div className="flow-empty">
        Nothing has happened yet. Interactions appear here as the agent works.
      </div>
    );
  }

  return (
    <div className="flow-scroll">
      <svg width={width} height={height} className="flow" role="img">
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
          </marker>
        </defs>

        {/* Participants and their lifelines. */}
        {lanes.map((lane) => (
          <g key={lane.id} className={`flow-actor ${lane.kind}`}>
            <rect
              x={xOf(lane.id) - COL_W / 2 + 8}
              y={8}
              width={COL_W - 16}
              height={24}
              rx={5}
            />
            <text x={xOf(lane.id)} y={24} textAnchor="middle" className="flow-actor-label">
              {lane.label}
            </text>
            <line
              x1={xOf(lane.id)}
              y1={HEAD_H}
              x2={xOf(lane.id)}
              y2={height - 10}
              className="flow-lifeline"
            />
          </g>
        ))}

        {/* Messages, newest last. Anything past the playhead is dimmed so
            scrubbing reads as replay rather than as a filter. */}
        {messages.map((message, row) => {
          const y = HEAD_H + row * ROW_H + ROW_H / 2;
          const x1 = xOf(message.from);
          const x2 = xOf(message.to);
          const future = message.seq > position;
          const back = x2 < x1;
          const classes = [
            "flow-msg",
            message.status,
            future ? "future" : "",
            message.rolledBack ? "rolled-back" : "",
            selected === message.seq ? "selected" : "",
          ]
            .filter(Boolean)
            .join(" ");

          return (
            <g
              key={message.seq}
              className={classes}
              onClick={() => onSelect(message.seq)}
            >
              <rect
                x={0}
                y={y - ROW_H / 2}
                width={width}
                height={ROW_H}
                className="flow-hit"
              />
              <text x={6} y={y + 4} className="flow-seq">
                {message.seq}
              </text>
              <line
                x1={x1}
                y1={y}
                x2={x2}
                y2={y}
                markerEnd="url(#arrow)"
                className="flow-arrow"
              />
              <text
                x={(x1 + x2) / 2}
                y={y - 6}
                textAnchor="middle"
                className="flow-label"
              >
                {message.label}
              </text>
              {message.detail && (
                <text
                  x={(x1 + x2) / 2}
                  y={y + 13}
                  textAnchor="middle"
                  className="flow-detail"
                >
                  {trim(message.detail, Math.max(18, Math.abs(x2 - x1) / 6))}
                </text>
              )}
              {back && (
                <circle cx={x2 + 5} cy={y} r={2.5} className="flow-return" />
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function trim(text: string, max: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max)}…` : clean;
}
