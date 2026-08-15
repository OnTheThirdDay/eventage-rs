/**
 * @vitest-environment jsdom
 */
import { describe, expect, it } from "vitest";
import { renderMarkdown } from "./markdown";

/** Visible text, as a reader would see it. */
function visible(html: string): string {
  const box = document.createElement("div");
  box.innerHTML = html;
  return box.textContent ?? "";
}

describe("rendering an assistant message", () => {
  it("keeps a long structured answer whole", () => {
    // Shaped like what the agent actually returns: headings, lists, tables,
    // fenced code and inline generics.
    const source = [
      "## Overview", "", "The parser has three stages.", "",
      "| Stage | File |", "|---|---|", "| lex | `src/lex.rs` |", "",
      "1. First", "2. Second", "",
      "```rust", "fn parse(input: &str) -> Vec<Token> { todo!() }", "```", "",
      "Then `Arc<Mutex<T>>` guards it.", "", "### Caveats", "",
      "- one", "- two", "", "That is the whole picture.",
    ].join("\n");
    const shown = visible(renderMarkdown(source));
    expect(shown).toContain("Overview");
    expect(shown).toContain("fn parse");
    expect(shown).toContain("That is the whole picture.");
  });

  it("does not swallow text after an unknown tag", () => {
    // Models emit pseudo-tags like <features> constantly. Sanitising must
    // remove the tag and keep the words around it, or a message silently
    // stops a few lines in.
    const source = "Before.\n\n<features>\n- one\n- two\n</features>\n\nAfter.";
    const shown = visible(renderMarkdown(source));
    expect(shown).toContain("Before.");
    expect(shown).toContain("one");
    expect(shown).toContain("After.");
  });

  it("survives an unclosed pseudo-tag", () => {
    const source = "Start of the answer.\n\n<thinking>\n\nThe rest of the answer.";
    const shown = visible(renderMarkdown(source));
    expect(shown).toContain("Start of the answer");
    expect(shown).toContain("The rest of the answer");
  });

  it("survives a generic type that looks like a tag", () => {
    // Rust and TypeScript are full of these.
    const source = "Use `Vec<String>` here.\n\nThen call Arc<Mutex<T>> on it.";
    const shown = visible(renderMarkdown(source));
    expect(shown).toContain("Vec<String>");
    expect(shown).toContain("Then call");
  });
});
