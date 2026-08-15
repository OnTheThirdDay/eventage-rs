/**
 * A unified diff for a file the agent changed.
 *
 * Tools report the before and after text, not a patch, so the diff is
 * computed here. Long unchanged stretches are collapsed: a review is about
 * what changed, and a thousand identical lines is not a review.
 */

import { useMemo } from "react";
import { highlight, languageOf } from "../lib/markdown";
import { CopyButton } from "./primitives";
import type { FileDiff } from "../lib/types";

type Row =
  | { kind: "same"; oldLine: number; newLine: number; text: string }
  | { kind: "add"; newLine: number; text: string }
  | { kind: "del"; oldLine: number; text: string }
  | { kind: "gap"; count: number };

/** Lines of context kept either side of a change. */
const CONTEXT = 3;

/**
 * Above this many lines the quadratic diff is skipped and the file is shown
 * as a wholesale replacement. It keeps a generated lockfile from locking up
 * the window, which matters more than a precise diff of one.
 */
const MAX_DIFFABLE_LINES = 3000;

/** Longest common subsequence over lines, the classic dynamic program. */
function diffLines(before: string[], after: string[]): Row[] {
  const n = before.length;
  const m = after.length;
  const lcs: number[][] = Array.from({ length: n + 1 }, () =>
    new Array<number>(m + 1).fill(0),
  );
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i]![j]! =
        before[i] === after[j]
          ? lcs[i + 1]![j + 1]! + 1
          : Math.max(lcs[i + 1]![j]!, lcs[i]![j + 1]!);
    }
  }

  const rows: Row[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (before[i] === after[j]) {
      rows.push({ kind: "same", oldLine: i + 1, newLine: j + 1, text: before[i]! });
      i++;
      j++;
    } else if (lcs[i + 1]![j]! >= lcs[i]![j + 1]!) {
      rows.push({ kind: "del", oldLine: i + 1, text: before[i]! });
      i++;
    } else {
      rows.push({ kind: "add", newLine: j + 1, text: after[j]! });
      j++;
    }
  }
  while (i < n) {
    rows.push({ kind: "del", oldLine: i + 1, text: before[i]! });
    i++;
  }
  while (j < m) {
    rows.push({ kind: "add", newLine: j + 1, text: after[j]! });
    j++;
  }
  return rows;
}

/** Replace runs of unchanged lines longer than 2×CONTEXT with a marker. */
function collapse(rows: Row[]): Row[] {
  const changed = rows.map((r) => r.kind === "add" || r.kind === "del");
  const keep = rows.map((_, index) =>
    changed
      .slice(Math.max(0, index - CONTEXT), index + CONTEXT + 1)
      .some(Boolean),
  );

  const out: Row[] = [];
  let hidden = 0;
  rows.forEach((row, index) => {
    if (keep[index]) {
      if (hidden > 0) {
        out.push({ kind: "gap", count: hidden });
        hidden = 0;
      }
      out.push(row);
    } else {
      hidden++;
    }
  });
  if (hidden > 0) out.push({ kind: "gap", count: hidden });
  return out;
}

export function DiffView({ diff }: { diff: FileDiff }) {
  const language = languageOf(diff.path);

  const { rows, added, removed, tooBig } = useMemo(() => {
    const before = diff.old_text === null ? [] : diff.old_text.split("\n");
    const after = diff.new_text.split("\n");

    if (before.length + after.length > MAX_DIFFABLE_LINES) {
      return {
        rows: [] as Row[],
        added: after.length,
        removed: before.length,
        tooBig: true,
      };
    }

    const all = diffLines(before, after);
    return {
      rows: collapse(all),
      added: all.filter((r) => r.kind === "add").length,
      removed: all.filter((r) => r.kind === "del").length,
      tooBig: false,
    };
  }, [diff.old_text, diff.new_text]);

  return (
    <div className="diff">
      <div className="diff-head">
        <span>{diff.path || "(unnamed file)"}</span>
        {added > 0 && <span className="added">+{added}</span>}
        {removed > 0 && <span className="removed">−{removed}</span>}
        {diff.old_text === null && <span>new file</span>}
        <span style={{ flex: 1 }} />
        <CopyButton
          text={diff.path}
          label="⧉ path"
          className="btn sm ghost inline"
          title="Copy the file path"
        />
        <CopyButton
          text={diff.new_text}
          label="⧉ new"
          className="btn sm ghost inline"
          title="Copy the file's new contents"
        />
      </div>
      {tooBig ? (
        <div className="diff-body">
          <div className="diff-row gap">
            <span className="gutter" />
            <span className="line">
              {removed} lines replaced by {added} — too large to diff inline.
            </span>
          </div>
        </div>
      ) : (
        <div className="diff-body">
          {rows.map((row, index) => {
            if (row.kind === "gap") {
              return (
                <div className="diff-row gap" key={`g${index}`}>
                  <span className="gutter">⋯</span>
                  <span className="line">
                    {row.count} unchanged line{row.count === 1 ? "" : "s"}
                  </span>
                </div>
              );
            }
            const sign = row.kind === "add" ? "+" : row.kind === "del" ? "−" : " ";
            const gutter =
              row.kind === "add"
                ? row.newLine
                : row.kind === "del"
                  ? row.oldLine
                  : row.newLine;
            return (
              <div className={`diff-row ${row.kind}`} key={`${row.kind}${index}`}>
                <span className="gutter">{gutter}</span>
                <span
                  className="line"
                  dangerouslySetInnerHTML={{
                    __html: `${sign} ${highlight(row.text, language)}`,
                  }}
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
