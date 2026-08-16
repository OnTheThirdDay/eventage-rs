/**
 * The input box.
 *
 * Enter sends and Shift+Enter breaks the line, which is what every chat app
 * has trained people to expect. While a turn is running the send button
 * becomes stop, because the one thing a user reaches for mid-turn is the
 * brake.
 */

import { useCallback, useRef, useState } from "react";
import type { ModeInfo, PromptBlock } from "../lib/types";
import { Menu, useAutoSize } from "./primitives";

interface Attachment {
  dataUrl: string;
  mimeType: string;
  base64: string;
}

export function Composer({
  disabled,
  running,
  modes,
  mode,
  sealedAttempts,
  onSend,
  onInterrupt,
  onModeChange,
}: {
  disabled: boolean;
  running: boolean;
  modes: ModeInfo[];
  mode: string;
  /** Rolled-back attempts being fed back into every request. */
  sealedAttempts: number;
  onSend: (blocks: PromptBlock[]) => void;
  onInterrupt: () => void;
  onModeChange: (mode: string) => void;
}) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const textarea = useAutoSize(text);

  const current = modes.find((m) => m.id === mode);

  const send = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed && attachments.length === 0) return;
    const blocks: PromptBlock[] = [];
    if (trimmed) blocks.push({ type: "text", text: trimmed });
    for (const a of attachments) {
      blocks.push({ type: "image", data: a.base64, mimeType: a.mimeType });
    }
    onSend(blocks);
    setText("");
    setAttachments([]);
  }, [text, attachments, onSend]);

  const addImage = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result);
      // `data:<mime>;base64,<payload>` — the API wants the payload alone.
      const comma = dataUrl.indexOf(",");
      if (comma < 0) return;
      setAttachments((list) => [
        ...list,
        {
          dataUrl,
          mimeType: file.type || "image/png",
          base64: dataUrl.slice(comma + 1),
        },
      ]);
    };
    reader.readAsDataURL(file);
  }, []);

  return (
    <div className="composer-wrap">
      <div className="composer">
        {attachments.length > 0 && (
          <div className="attachments">
            {attachments.map((a, i) => (
              <div className="attachment" key={i}>
                <img src={a.dataUrl} alt="" />
                <button
                  onClick={() =>
                    setAttachments((list) => list.filter((_, j) => j !== i))
                  }
                  aria-label="Remove attachment"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <textarea
          ref={textarea}
          rows={1}
          value={text}
          placeholder={
            disabled
              ? "Open a session to start"
              : running
                ? "The agent is working — type your next message, or stop it"
                : "Ask for a change, a fix, an explanation…"
          }
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              if (!running) send();
            }
          }}
          onPaste={(e) => {
            for (const item of e.clipboardData.items) {
              if (item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) {
                  e.preventDefault();
                  addImage(file);
                }
              }
            }
          }}
        />

        <div className="composer-bar">
          <Menu
            direction="up"
            trigger={({ toggle }) => (
              <button className="btn sm ghost" onClick={toggle} disabled={disabled}>
                {current?.label ?? mode} ▾
              </button>
            )}
          >
            {(close) =>
              modes.map((m) => (
                <button
                  key={m.id}
                  className={m.id === mode ? "selected" : ""}
                  onClick={() => {
                    onModeChange(m.id);
                    close();
                  }}
                >
                  {m.label}
                  <span className="desc">{m.description}</span>
                </button>
              ))
            }
          </Menu>

          {sealedAttempts > 0 && (
            // Beside the mode selector because it belongs to the same
            // question: what is shaping the next request. The agent is told
            // about these every time, and nothing else on screen says so —
            // the warning goes to the model, not to the transcript.
            <span
              className="sealed-badge"
              title={`${sealedAttempts} rolled-back attempt${
                sealedAttempts === 1 ? "" : "s"
              } are described to the agent on every request, so it does not repeat them.`}
            >
              {sealedAttempts} rewound
            </span>
          )}

          <button
            className="btn sm ghost"
            onClick={() => fileInput.current?.click()}
            disabled={disabled}
            title="Attach an image"
          >
            ＋
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => {
              for (const file of e.target.files ?? []) addImage(file);
              e.target.value = "";
            }}
          />

          <span className="spacer" />
          <span className="composer-hint">
            {running ? "working…" : "Enter to send · Shift+Enter for a new line"}
          </span>

          {running ? (
            <button className="btn danger" onClick={onInterrupt}>
              ■ Stop
            </button>
          ) : (
            <button
              className="btn primary"
              onClick={send}
              disabled={disabled || (!text.trim() && attachments.length === 0)}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
