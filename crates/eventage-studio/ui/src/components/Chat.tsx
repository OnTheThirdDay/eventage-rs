/**
 * The transcript.
 *
 * Auto-scroll only holds while the reader is already at the bottom. Yanking
 * someone back down while they are reading an earlier tool result is the
 * fastest way to make a streaming UI feel hostile, so scrolling up pins the
 * view and offers a way back instead.
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ChatItem, PlanEntry } from "../lib/types";
import {
  AssistantMessage,
  Notice,
  PermissionCard,
  PlanPanel,
  ToolCard,
  UserMessage,
} from "./ChatItems";

/** How close to the bottom still counts as "at the bottom", in pixels. */
const STICK_THRESHOLD = 120;

export function Chat({
  items,
  plan,
  running,
  onPermission,
  onInspectContext,
  onBranchFrom,
  emptyState,
}: {
  items: ChatItem[];
  plan: PlanEntry[];
  running: boolean;
  onPermission: (requestId: string, approve: boolean, always: boolean) => void;
  /** Show the context the model was given for the message at this sequence. */
  onInspectContext: (seq: number) => void;
  /** Fork the conversation at this event into a new session. */
  onBranchFrom: (seq: number) => void;
  emptyState: React.ReactNode;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(true);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      setStuck(distance < STICK_THRESHOLD);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // Runs before paint so the view never visibly jumps.
  useLayoutEffect(() => {
    if (!stuck) return;
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items, plan, running, stuck]);

  const jump = () => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    setStuck(true);
  };

  // A plan can arrive before any prose does — showing "what should we work
  // on?" underneath a live checklist would be plainly wrong.
  if (!items.length && !plan.length) {
    return (
      <div className="transcript" ref={scroller}>
        {emptyState}
      </div>
    );
  }

  return (
    <>
      <div className="transcript" ref={scroller}>
        <div className="stream">
          {items.map((item) => {
            switch (item.type) {
              case "user":
                return (
                  <UserMessage
                    key={item.key}
                    item={item}
                    onBranchFrom={onBranchFrom}
                  />
                );
              case "assistant":
                return (
                  <AssistantMessage
                    key={item.key}
                    item={item}
                    onInspectContext={onInspectContext}
                  />
                );
              case "tool":
                return <ToolCard key={item.key} item={item} />;
              case "permission":
                return (
                  <PermissionCard
                    key={item.key}
                    item={item}
                    onAnswer={(approve, always) =>
                      onPermission(item.requestId, approve, always)
                    }
                  />
                );
              case "notice":
                return <Notice key={item.key} item={item} />;
            }
          })}
          {plan.length > 0 && <PlanPanel entries={plan} />}
        </div>
      </div>
      {!stuck && (
        <button className="btn jump-btn" onClick={jump}>
          ↓ Jump to latest
        </button>
      )}
    </>
  );
}
