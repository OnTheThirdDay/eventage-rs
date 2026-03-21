//! ratatui TUI — a full-screen terminal interface for the coding agent.
//!
//! ## Modes
//!
//! | Mode | Description |
//! |---|---|
//! | `Idle` | Waiting for user input; prompt is shown |
//! | `Working` | Agent is processing; streaming tokens are displayed |
//! | `AwaitingApproval` | A dangerous tool call awaits user approval |
//!
//! ## Features
//!
//! - Scrollable conversation history (mouse wheel + Up/Down/PgUp/PgDn)
//! - Full input cursor movement (Left/Right/Home/End/Delete/Backspace)
//! - Mouse wheel scrolling
//! - Sub-agent task events displayed inline

use std::io;
use std::sync::{atomic::AtomicBool, Arc};

use crossterm::{
    event::{Event as CrosstermEvent, EventStream, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use eventage::{kinds as core_kinds, Event, EventBus};
use futures_util::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
    Frame, Terminal,
};
use serde_json::json;
use tracing::debug;

use crate::kinds::{
    CODING_APPROVAL_DENIED, CODING_APPROVAL_GRANTED, CODING_APPROVAL_REQUESTED,
    CODING_STREAM_CHUNK, CODING_TURN_DIFF, SUBAGENT_TASK_COMPLETE, SUBAGENT_TASK_ERROR,
    SUBAGENT_TASK_LAUNCH,
};

// ── Display constants ─────────────────────────────────────────────────────────

const MAX_LOG_LINES: usize = 2000; // rolling conversation buffer

// ── TUI mode ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Idle,
    Working,
    AwaitingApproval { tool: String, args_preview: String },
}

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    mode: Mode,
    /// Scrollable conversation log (rendered lines).
    log: Vec<Line<'static>>,
    /// Current user input.
    input: String,
    /// Byte offset of the cursor within `input`.
    cursor_pos: usize,
    /// Session info for the status bar.
    model: String,
    session_id: String,
    /// Incremented on each streaming chunk to drive a cursor blink.
    stream_chars: usize,
    /// Lines from the bottom (0 = pinned to latest content).
    scroll_offset: usize,
    /// When true, auto-scroll to bottom on new content.
    auto_scroll: bool,
    /// When true, the next stream chunk must start on a fresh agent line
    /// (set after a TOOL_RESULT so text doesn't concatenate onto the result).
    pending_stream_newline: bool,
    /// True between AGENT_CYCLE_START and AGENT_CYCLE_END.
    /// Stream chunks outside a cycle are dropped (late-arriving bus events).
    in_cycle: bool,
}

impl AppState {
    fn push(&mut self, line: Line<'static>) {
        self.log.push(line);
        if self.log.len() > MAX_LOG_LINES {
            self.log.remove(0);
        }
        if self.auto_scroll {
            self.scroll_offset = 0;
        }
    }

    fn push_plain(&mut self, s: impl Into<String>) {
        self.push(Line::from(s.into()));
    }

    fn append_to_last(&mut self, s: &str) {
        if let Some(last) = self.log.last_mut() {
            if let Some(span) = last.spans.last_mut() {
                let mut new_content = span.content.to_string();
                new_content.push_str(s);
                span.content = new_content.into();
            } else {
                last.spans.push(Span::raw(s.to_string()));
            }
        } else {
            self.push_plain(s.to_string());
        }
        if self.auto_scroll {
            self.scroll_offset = 0;
        }
    }

    fn scroll_up(&mut self, lines: usize) {
        self.scroll_offset += lines;
        self.auto_scroll = false;
    }

    fn scroll_down(&mut self, lines: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
        if self.scroll_offset == 0 {
            self.auto_scroll = true;
        }
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
        self.auto_scroll = true;
    }
}

// ── UTF-8 cursor helpers ──────────────────────────────────────────────────────

fn prev_char_boundary(s: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut p = pos - 1;
    while p > 0 && !s.is_char_boundary(p) {
        p -= 1;
    }
    p
}

fn next_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    let mut p = pos + 1;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

// ── Display-row helpers ───────────────────────────────────────────────────────

/// Number of terminal rows a `Line` occupies when wrapped to `width` columns.
/// This is an approximation for monospace fonts: each char counts as 1 column.
fn line_display_rows(line: &Line, width: u16) -> usize {
    if width == 0 {
        return 1;
    }
    let chars: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
    if chars == 0 {
        return 1;
    }
    chars.div_ceil(width as usize)
}

/// Total wrapped display rows for the whole log.
fn log_display_rows(log: &[Line], width: u16) -> usize {
    log.iter().map(|l| line_display_rows(l, width)).sum()
}

/// Simulate ratatui's greedy word-wrap (`Wrap { trim: false }`) and return
/// the total number of display rows `text` occupies at `width` columns.
///
/// Iterates character-by-character to avoid the off-by-one errors that arise
/// from `str::split(' ')` skipping leading/trailing spaces.
fn word_wrap_rows(text: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return 1;
    }
    let n = chars.len();
    let mut rows = 1usize;
    let mut col = 0usize; // chars placed on the current row
    let mut i = 0usize;

    while i < n {
        if chars[i] == ' ' {
            if col == width {
                rows += 1;
                col = 0;
            }
            col += 1;
            i += 1;
        } else {
            // Measure this word (sequence of non-space chars).
            let mut j = i;
            while j < n && chars[j] != ' ' {
                j += 1;
            }
            let wlen = j - i;
            // Wrap the whole word to the next row if it doesn't fit here
            // (but never wrap when already at column 0 — just hard-wrap within it).
            if col > 0 && col + wlen > width {
                rows += 1;
                col = 0;
            }
            // Place each character; hard-wrap very long words.
            for _ in 0..wlen {
                if col == width {
                    rows += 1;
                    col = 0;
                }
                col += 1;
            }
            i = j;
        }
    }
    rows
}

/// Simulate ratatui's greedy word-wrap (`Wrap { trim: false }`) and return
/// `(display_row, display_col)` for the cursor sitting at character index
/// `cursor_char_abs` within `text`.
///
/// The wrap decision for each word is made **before** checking the cursor so
/// that a cursor at the start of a wrapped word lands on the new row, not the
/// old one.
fn word_wrap_cursor(text: &str, cursor_char_abs: usize, width: usize) -> (u16, u16) {
    if width == 0 {
        return (0, 0);
    }
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut row = 0u16;
    let mut col = 0usize;
    let mut i = 0usize;

    while i < n {
        if chars[i] == ' ' {
            // Hard-wrap FIRST (space may land on the next row with trim: false).
            if col == width {
                row += 1;
                col = 0;
            }
            // Cursor before this space (after any hard-wrap adjustment).
            if i == cursor_char_abs {
                return (row, col as u16);
            }
            col += 1;
            i += 1;
        } else {
            // Measure word without advancing i.
            let mut j = i;
            while j < n && chars[j] != ' ' {
                j += 1;
            }
            let wlen = j - i;

            // Word-wrap decision FIRST so the cursor at a word-start lands on
            // the correct (new) row.
            if col > 0 && col + wlen > width {
                row += 1;
                col = 0;
            }

            // Place each character: hard-wrap check, then cursor check, then advance.
            for _ in 0..wlen {
                if col == width {
                    row += 1;
                    col = 0;
                }
                if i == cursor_char_abs {
                    return (row, col as u16);
                }
                col += 1;
                i += 1;
            }
        }
    }

    // Cursor is at the very end of the string.
    (row, col as u16)
}

#[cfg(test)]
mod wrap_tests {
    use super::*;

    // Prefix used by the input bar.
    const P: &str = " > ";

    #[test]
    fn empty_input_cursor_after_prefix() {
        // " > " — cursor at position 3 (just after the prefix), width=80
        assert_eq!(word_wrap_cursor(P, 3, 80), (0, 3));
        assert_eq!(word_wrap_rows(P, 80), 1);
    }

    #[test]
    fn one_char_cursor_at_end() {
        // " > a" — cursor at 4
        assert_eq!(word_wrap_cursor(" > a", 4, 80), (0, 4));
        assert_eq!(word_wrap_rows(" > a", 80), 1);
    }

    #[test]
    fn word_wraps_to_new_row() {
        // " > hello world" at width=10: "hello" fits row 0, "world" wraps to row 1.
        // Cursor at 9 ('w') should be (1, 0) after the wrap decision.
        assert_eq!(word_wrap_cursor(" > hello world", 9, 10), (1, 0));
        assert_eq!(word_wrap_rows(" > hello world", 10), 2);
    }

    #[test]
    fn cursor_at_very_start() {
        assert_eq!(word_wrap_cursor("hello", 0, 80), (0, 0));
    }

    #[test]
    fn hard_wrap_long_word() {
        // "aaaaaaaaaaa" (11 a's) at width=10: spills onto row 2
        assert_eq!(word_wrap_rows("aaaaaaaaaaa", 10), 2);
        // Cursor at position 10 (start of row 2)
        assert_eq!(word_wrap_cursor("aaaaaaaaaaa", 10, 10), (1, 0));
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

fn render(frame: &mut Frame, state: &AppState) {
    let area = frame.area();
    let prefix = " > ";

    // Compute how many display rows the current input needs so the input bar
    // can grow to fit (capped at 8 rows to leave room for the conversation).
    // Use word_wrap_rows to match ratatui's Wrap { trim: false } behaviour
    // (word-boundary wrapping can use more rows than chars ÷ width).
    let input_display = format!("{}{}", prefix, state.input);
    let input_display_rows = word_wrap_rows(&input_display, area.width.max(1) as usize);
    // +1 for the top border; at least 3 rows, at most 8
    let input_height = ((input_display_rows + 1) as u16).clamp(3, 8);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),           // status bar
            Constraint::Min(0),              // conversation
            Constraint::Length(input_height), // input bar (dynamic)
        ])
        .split(area);

    // ── Status bar ────────────────────────────────────────────────────────────
    let status_text = match &state.mode {
        Mode::Idle => Span::styled(" ● Idle", Style::default().fg(Color::Green)),
        Mode::Working => Span::styled(
            " ● Working…",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Mode::AwaitingApproval { .. } => Span::styled(
            " ⚠ Awaiting approval",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
    };

    let scroll_hint = if state.scroll_offset > 0 {
        Span::styled(
            format!("  [↑{}↓ scroll]", state.scroll_offset),
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::raw("")
    };

    let status_bar = Paragraph::new(Line::from(vec![
        Span::styled(
            format!(
                " coding-agent │ {} │ {} ",
                state.model,
                &state.session_id[..8.min(state.session_id.len())]
            ),
            Style::default().fg(Color::DarkGray),
        ),
        status_text,
        scroll_hint,
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(status_bar, chunks[0]);

    // ── Conversation log with scroll ──────────────────────────────────────────
    let log_area = chunks[1];
    let visible_height = log_area.height as usize;
    // Count display rows after wrapping so the scroll offset is accurate.
    let total_rows = log_display_rows(&state.log, log_area.width);
    let max_scroll = total_rows.saturating_sub(visible_height);
    let effective_scroll = state.scroll_offset.min(max_scroll);
    let scroll_from_top = max_scroll - effective_scroll;

    let log_para = Paragraph::new(state.log.clone())
        .block(Block::default().borders(Borders::NONE))
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false })
        .scroll((scroll_from_top as u16, 0));
    frame.render_widget(log_para, log_area);

    // Scroll indicator at the bottom of the log area
    if effective_scroll > 0 {
        let indicator_text = format!(
            " ▼ {} more below  [↓/PgDn to scroll, G to jump to bottom] ",
            effective_scroll
        );
        let indicator = Paragraph::new(indicator_text)
            .style(Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM));
        let indicator_area = Rect {
            x: log_area.x,
            y: log_area.y + log_area.height.saturating_sub(1),
            width: log_area.width,
            height: 1,
        };
        frame.render_widget(Clear, indicator_area);
        frame.render_widget(indicator, indicator_area);
    }

    // ── Input bar ─────────────────────────────────────────────────────────────
    let input_area = chunks[2];

    let input_bar = Paragraph::new(input_display.clone())
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false });
    frame.render_widget(input_bar, input_area);

    // Position the terminal cursor accounting for word-wrap.
    // `cursor_abs` is the char-count from the start of the "prefix + input" string.
    if !matches!(state.mode, Mode::AwaitingApproval { .. }) {
        let cursor_abs = prefix.len() + state.input[..state.cursor_pos].chars().count();
        let w = input_area.width.max(1) as usize;
        let (cursor_display_row, cursor_display_col) =
            word_wrap_cursor(&input_display, cursor_abs, w);
        // +1 to skip the TOP border row
        frame.set_cursor_position((
            input_area.x + cursor_display_col,
            input_area.y + 1 + cursor_display_row,
        ));
    }

    // ── Approval overlay (rendered on top) ────────────────────────────────────
    if let Mode::AwaitingApproval { tool, args_preview } = &state.mode {
        render_approval_overlay(frame, frame.area(), tool, args_preview);
    }
}

fn render_approval_overlay(frame: &mut Frame, area: Rect, tool: &str, preview: &str) {
    // Fixed 10-row height ensures Y/N is always visible regardless of args length.
    let popup_area = centred_rect_fixed(area.width.min(70), 10, area);
    frame.render_widget(Clear, popup_area);

    // Hard-limit preview to one line of display chars so the Y/N row is never
    // pushed off screen.  Use chars() to avoid splitting a UTF-8 code point.
    let max_chars = (popup_area.width as usize).saturating_sub(12).max(20);
    let preview_display: String = if preview.chars().count() > max_chars {
        let truncated: String = preview.chars().take(max_chars).collect();
        format!("{truncated}…")
    } else {
        preview.to_string()
    };

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Tool:  "),
            Span::styled(
                tool,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Args:  "),
            Span::styled(preview_display, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  [Y]",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" Allow    "),
            Span::styled(
                "[N]",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" Deny"),
        ]),
        Line::from(""),
    ];

    let overlay = Paragraph::new(text)
        .block(
            Block::default()
                .title(" ⚠  Security Gate ")
                .title_alignment(Alignment::Center)
                .borders(Borders::ALL)
                .border_type(BorderType::Double)
                .border_style(Style::default().fg(Color::Red)),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(overlay, popup_area);
}

#[allow(dead_code)]
fn centred_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Centre a popup with fixed pixel dimensions inside `area`.
fn centred_rect_fixed(width: u16, height: u16, area: Rect) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

// ── Bus event handling ────────────────────────────────────────────────────────

fn handle_bus_event(state: &mut AppState, event: Event) {
    match event.kind.as_str() {
        k if k == core_kinds::USER_MESSAGE => {
            let text = event.payload["text"].as_str().unwrap_or("").to_string();
            state.push(Line::from(""));
            state.push(Line::from(vec![
                Span::styled(
                    "  you  ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(text),
            ]));
            state.mode = Mode::Working;
        }

        k if k == core_kinds::AGENT_CYCLE_START => {
            state.in_cycle = true;
            state.push(Line::from(""));
            state.push(Line::from(vec![Span::styled(
                "  agent  ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )]));
        }

        k if k == CODING_STREAM_CHUNK => {
            // Drop chunks that arrive outside an active cycle (late bus events
            // that sneak in after AGENT_CYCLE_END).
            if !state.in_cycle {
                return;
            }
            // Buffer chunks while the approval dialog is visible — they're the
            // LLM's tail text after the tool-call token and should not be
            // appended to the log while the overlay is shown.
            if matches!(state.mode, Mode::AwaitingApproval { .. }) {
                return;
            }

            let chunk = event.payload["content"].as_str().unwrap_or("");

            // After a tool result the next chunk must start on a fresh line so
            // the assistant text doesn't concatenate onto the result summary.
            if state.pending_stream_newline && !chunk.trim().is_empty() {
                state.pending_stream_newline = false;
                state.push(Line::from(""));
                state.push(Line::from(vec![Span::styled(
                    "  agent  ",
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                )]));
            }

            let parts: Vec<&str> = chunk.split('\n').collect();
            for (i, part) in parts.iter().enumerate() {
                if i > 0 {
                    state.push(Line::from(vec![Span::styled(
                        "         ",
                        Style::default(),
                    )]));
                }
                if !part.is_empty() {
                    state.append_to_last(part);
                }
            }
            state.stream_chars += chunk.len();
        }

        k if k == core_kinds::ASSISTANT_MESSAGE => {
            if let Some(content) = event.payload["content"].as_str() {
                if !content.is_empty() && state.stream_chars == 0 {
                    state.push(Line::from(""));
                    state.push(Line::from(vec![
                        Span::styled(
                            "  agent  ",
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(content.to_string()),
                    ]));
                }
            }
        }

        k if k == core_kinds::TOOL_CALL_PROPOSED => {
            let name = event.payload["name"].as_str().unwrap_or("?");
            let args = event
                .payload
                .get("arguments")
                .map(|v| {
                    let s = v.to_string();
                    if s.len() > 80 {
                        format!("{}…", &s[..80])
                    } else {
                        s
                    }
                })
                .unwrap_or_default();
            state.push(Line::from(vec![
                Span::styled(
                    format!("  [→ {name}]  "),
                    Style::default().fg(Color::Magenta),
                ),
                Span::styled(args, Style::default().fg(Color::DarkGray)),
            ]));
        }

        k if k == core_kinds::TOOL_RESULT => {
            let name = event.payload["name"].as_str().unwrap_or("?");
            let success = event.payload["result"]["success"]
                .as_bool()
                .unwrap_or(false);
            let (icon, style) = if success {
                ("  ✓ ", Style::default().fg(Color::Green))
            } else {
                ("  ✗ ", Style::default().fg(Color::Red))
            };
            let summary = if let Some(err) = event.payload["result"]["error"].as_str() {
                err.to_string()
            } else if let Some(b) = event.payload["result"]["bytes_written"].as_u64() {
                format!("{b} bytes written")
            } else if let Some(stdout) = event.payload["result"]["stdout"].as_str() {
                let trimmed = stdout.trim();
                if trimmed.is_empty() {
                    "ok".to_string()
                } else {
                    let s = trimmed.lines().next().unwrap_or("ok");
                    if s.len() > 60 {
                        format!("{}…", &s[..60])
                    } else {
                        s.to_string()
                    }
                }
            } else {
                name.to_string()
            };
            state.push(Line::from(vec![
                Span::styled(icon, style),
                Span::styled(summary, style),
            ]));
            // Next stream chunk belongs to a new assistant turn — separate it.
            state.pending_stream_newline = true;
        }

        k if k == core_kinds::AGENT_CYCLE_END => {
            state.in_cycle = false;
            state.mode = Mode::Idle;
            state.stream_chars = 0;
            state.pending_stream_newline = false;
            state.push(Line::from(""));
        }

        k if k == CODING_APPROVAL_REQUESTED => {
            let tool = event.payload["tool"].as_str().unwrap_or("?").to_string();
            let args_preview = event
                .payload
                .get("args")
                .map(|v| v.to_string())
                .unwrap_or_default();
            // Push a visible pause line so the conversation log doesn't look
            // abruptly truncated — the stream stopped here on purpose.
            state.push(Line::from(vec![Span::styled(
                format!("  ⏸  approval required for {tool}  (Y/N)"),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM),
            )]));
            state.mode = Mode::AwaitingApproval { tool, args_preview };
        }

        k if k == CODING_APPROVAL_GRANTED || k == CODING_APPROVAL_DENIED => {
            let granted = k == CODING_APPROVAL_GRANTED;
            state.push(Line::from(vec![Span::styled(
                if granted { "  ▶  approved" } else { "  ✗  denied" },
                Style::default()
                    .fg(if granted { Color::Green } else { Color::Red })
                    .add_modifier(Modifier::DIM),
            )]));
            if !matches!(state.mode, Mode::Idle) {
                state.mode = Mode::Working;
            }
        }

        k if k == CODING_TURN_DIFF => {
            let changed = event.payload["changed_files"].as_u64().unwrap_or(0);
            let new = event.payload["new_files"].as_u64().unwrap_or(0);
            let deleted = event.payload["deleted_files"].as_u64().unwrap_or(0);
            if changed + new + deleted > 0 {
                state.push(Line::from(vec![Span::styled(
                    format!("  ∆ {} changed, {} new, {} deleted", changed, new, deleted),
                    Style::default().fg(Color::Blue),
                )]));
            }
        }

        k if k == SUBAGENT_TASK_LAUNCH => {
            let job_id = event.payload["job_id"].as_str().unwrap_or("?");
            let subagent_type = event.payload["subagent_type"].as_str().unwrap_or("?");
            let description = event.payload["description"].as_str().unwrap_or("");
            let preview = if description.len() > 60 {
                format!("{}…", &description[..60])
            } else {
                description.to_string()
            };
            let id_short = &job_id[..job_id.len().min(8)];
            state.push(Line::from(vec![
                Span::styled(
                    format!("  [sub-agent: {subagent_type} #{id_short}]  "),
                    Style::default().fg(Color::Magenta),
                ),
                Span::styled(preview, Style::default().fg(Color::DarkGray)),
            ]));
        }

        k if k == SUBAGENT_TASK_COMPLETE => {
            let job_id = event.payload["job_id"].as_str().unwrap_or("?");
            let result = event.payload["result"].as_str().unwrap_or("done");
            let preview = if result.len() > 60 {
                format!("{}…", &result[..60])
            } else {
                result.to_string()
            };
            let id_short = &job_id[..job_id.len().min(8)];
            state.push(Line::from(vec![
                Span::styled(
                    format!("  ✓ sub-agent #{id_short}  "),
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
                Span::styled(preview, Style::default().fg(Color::Green)),
            ]));
        }

        k if k == SUBAGENT_TASK_ERROR => {
            let job_id = event.payload["job_id"].as_str().unwrap_or("?");
            let error = event.payload["error"].as_str().unwrap_or("error");
            let id_short = &job_id[..job_id.len().min(8)];
            state.push(Line::from(vec![
                Span::styled(
                    format!("  ✗ sub-agent #{id_short}  "),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::styled(error.to_string(), Style::default().fg(Color::Red)),
            ]));
        }

        _ => {}
    }
}

// ── Key action ────────────────────────────────────────────────────────────────

enum KeyAction {
    Quit,
    Continue,
}

// ── Keyboard handling ─────────────────────────────────────────────────────────

async fn handle_key(
    state: &mut AppState,
    key: KeyEvent,
    bus: &EventBus,
    cancelled: &Arc<AtomicBool>,
) -> KeyAction {
    // ── Global shortcuts ──────────────────────────────────────────────────────
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') | KeyCode::Char('q') => {
                return KeyAction::Quit;
            }
            KeyCode::Char('x') => {
                cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                return KeyAction::Continue;
            }
            _ => {}
        }
    }

    match &state.mode.clone() {
        // ── Idle / Working: full input editing + scroll ───────────────────────
        // Typing is allowed while the agent is working so messages can be
        // queued ahead of time; Enter publishes immediately and the agent
        // processes it after the current cycle finishes.
        Mode::Idle | Mode::Working => match key.code {
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                state.input.insert(state.cursor_pos, c);
                state.cursor_pos += c.len_utf8();
            }
            KeyCode::Backspace => {
                if state.cursor_pos > 0 {
                    let prev = prev_char_boundary(&state.input, state.cursor_pos);
                    state.input.remove(prev);
                    state.cursor_pos = prev;
                }
            }
            KeyCode::Delete => {
                if state.cursor_pos < state.input.len() {
                    state.input.remove(state.cursor_pos);
                }
            }
            KeyCode::Left => {
                state.cursor_pos = prev_char_boundary(&state.input, state.cursor_pos);
            }
            KeyCode::Right => {
                state.cursor_pos = next_char_boundary(&state.input, state.cursor_pos);
            }
            KeyCode::Home => {
                state.cursor_pos = 0;
            }
            KeyCode::End => {
                state.cursor_pos = state.input.len();
            }
            KeyCode::Up => state.scroll_up(1),
            KeyCode::Down => state.scroll_down(1),
            KeyCode::PageUp => state.scroll_up(20),
            KeyCode::PageDown => state.scroll_down(20),
            KeyCode::Enter => {
                let msg = state.input.trim().to_string();
                if !msg.is_empty() {
                    state.input.clear();
                    state.cursor_pos = 0;
                    state.scroll_to_bottom();
                    let _ = bus
                        .publish(eventage::Event::new(
                            core_kinds::USER_MESSAGE,
                            json!({ "text": msg }),
                        ))
                        .await;
                    state.mode = Mode::Working;
                }
            }
            _ => {}
        },

        // ── AwaitingApproval: y/n + scroll ────────────────────────────────────
        Mode::AwaitingApproval { .. } => match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                debug!("user granted tool approval");
                let _ = bus
                    .publish(Event::new(CODING_APPROVAL_GRANTED, json!({})))
                    .await;
                state.mode = Mode::Working;
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                debug!("user denied tool approval");
                let _ = bus
                    .publish(Event::new(CODING_APPROVAL_DENIED, json!({})))
                    .await;
                state.mode = Mode::Working;
            }
            KeyCode::Up => state.scroll_up(1),
            KeyCode::Down => state.scroll_down(1),
            KeyCode::PageUp => state.scroll_up(20),
            KeyCode::PageDown => state.scroll_down(20),
            _ => {}
        },
    }

    KeyAction::Continue
}

// ── Main TUI loop ─────────────────────────────────────────────────────────────

/// Run the TUI. This function blocks until the user quits (Ctrl+C or Ctrl+Q).
pub async fn run_tui(
    bus: EventBus,
    model: String,
    session_id: String,
    cancelled: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    // ── Terminal setup ────────────────────────────────────────────────────────
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState {
        mode: Mode::Idle,
        log: Vec::new(),
        input: String::new(),
        cursor_pos: 0,
        model,
        session_id,
        stream_chars: 0,
        scroll_offset: 0,
        auto_scroll: true,
        pending_stream_newline: false,
        in_cycle: false,
    };

    state.push(Line::from(vec![
        Span::styled(
            " coding-agent",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  │  Ctrl+C quit  │  Ctrl+X cancel"),
    ]));
    state.push(Line::from(vec![Span::styled(
        " Scroll: ↑↓ PgUp/PgDn  │  Jump to bottom: G",
        Style::default().fg(Color::DarkGray),
    )]));
    state.push(Line::from(""));

    let mut bus_rx = bus.subscribe();
    let mut crossterm_events = EventStream::new();
    // Coalesce rapid bus events (e.g. stream chunks) into at-most-30fps
    // redraws.  When nothing changes the screen is never touched, leaving it
    // stable so the terminal can register a click-and-drag text selection.
    let mut frame_tick = tokio::time::interval(std::time::Duration::from_millis(33));
    frame_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut needs_redraw = true; // paint once on startup

    let result: anyhow::Result<()> = async {
        loop {
            tokio::select! {
                // ── Frame timer: redraw only when state changed ───────────────
                _ = frame_tick.tick() => {
                    if needs_redraw {
                        terminal.draw(|f| render(f, &state))?;
                        needs_redraw = false;
                    }
                }

                // ── Bus events: mark dirty, no immediate redraw ───────────────
                maybe_event = bus_rx.recv() => {
                    match maybe_event {
                        Some(event) => {
                            handle_bus_event(&mut state, event);
                            needs_redraw = true;
                        }
                        None => break,
                    }
                }

                // ── Key / resize events: update then redraw immediately ───────
                maybe_ct = crossterm_events.next() => {
                    match maybe_ct {
                        Some(Ok(CrosstermEvent::Key(key))) => {
                            match handle_key(&mut state, key, &bus, &cancelled).await {
                                KeyAction::Quit => break,
                                KeyAction::Continue => {}
                            }
                            terminal.draw(|f| render(f, &state))?;
                            needs_redraw = false;
                        }
                        Some(Ok(CrosstermEvent::Resize(..))) => {
                            needs_redraw = true;
                        }
                        Some(Ok(_)) => {}
                        Some(Err(_)) | None => break,
                    }
                }
            }
        }
        Ok(())
    }
    .await;

    // ── Terminal teardown ─────────────────────────────────────────────────────
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}
