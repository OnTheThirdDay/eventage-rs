/**
 * Markdown rendering for assistant messages.
 *
 * Model output is untrusted text that ends up as HTML, so it is sanitised
 * after rendering rather than before: a model can be talked into emitting
 * markup, and a coding agent quotes HTML from the repository as a matter of
 * course.
 */

import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import { Marked } from "marked";

const escapeHtml = (text: string) =>
  text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const marked = new Marked({ gfm: true, breaks: true });

marked.use({
  renderer: {
    code({ text, lang }) {
      const language = lang && hljs.getLanguage(lang) ? lang : null;
      const body = language
        ? hljs.highlight(text, { language, ignoreIllegals: true }).value
        : escapeHtml(text);
      const label = language
        ? `<span class="code-lang">${escapeHtml(language)}</span>`
        : "";
      // The button carries no payload: the click handler reads the text out
      // of the sibling <code>, which avoids escaping a whole code block into
      // an HTML attribute and getting it subtly wrong.
      return `<div class="code-block">${label}<button class="copy-code" type="button" aria-label="Copy code">Copy</button><pre><code class="hljs">${body}</code></pre></div>`;
    },
  },
});

// Links in a desktop app must not navigate the app window away from itself.
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A" && node.hasAttribute("href")) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

export function renderMarkdown(text: string): string {
  const html = marked.parse(text, { async: false });
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    // Highlighting works by wrapping tokens in classed spans.
    ADD_ATTR: ["class", "target", "rel", "type", "aria-label"],
    // `button` is allowed because we add the copy control ourselves, after
    // highlighting; anything the model emits is still stripped of handlers by
    // DOMPurify's attribute filtering.
    FORBID_TAGS: ["style", "form", "input"],
    FORBID_ATTR: ["onclick", "onerror", "onload", "formaction"],
  });
}

/** Highlight a bare code string (used by the diff view). */
export function highlight(code: string, language?: string): string {
  if (language && hljs.getLanguage(language)) {
    return hljs.highlight(code, { language, ignoreIllegals: true }).value;
  }
  return escapeHtml(code);
}

/** Guess a highlight.js language from a file path. */
export function languageOf(path: string): string | undefined {
  const ext = path.split(".").pop()?.toLowerCase();
  if (!ext) return undefined;
  const map: Record<string, string> = {
    rs: "rust",
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    py: "python",
    go: "go",
    java: "java",
    c: "c",
    h: "c",
    cpp: "cpp",
    hpp: "cpp",
    cs: "csharp",
    rb: "ruby",
    php: "php",
    swift: "swift",
    kt: "kotlin",
    sh: "bash",
    bash: "bash",
    zsh: "bash",
    sql: "sql",
    json: "json",
    yaml: "yaml",
    yml: "yaml",
    toml: "ini",
    md: "markdown",
    html: "xml",
    xml: "xml",
    css: "css",
    scss: "scss",
  };
  return map[ext];
}
