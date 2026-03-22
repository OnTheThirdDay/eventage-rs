//! ratatui TUI for eventage-claw.
//!
//! Adapted from example-coding-agent/src/tui.rs with claw-specific additions:
//!
//! - Group switcher overlay (`Ctrl+G` / `Tab`)
//! - Custom event displays: schedule fire (⏰), group message (✉), group register (➕)
//! - Status bar showing active group, model, and scheduled task count
//! - `CLAW_STREAM_CHUNK` replaces `CODING_STREAM_CHUNK`

use std::io;
use std::sync::{atomic::AtomicBool, Arc};

use crossterm::{
    event::{Event as CrosstermEvent, EventStream, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use eventage::{kinds as core_kinds, meta_keys, Event, EventBus};
use futures_util::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use serde_json::json;
use tokio::sync::Mutex;
use tracing::debug;

use crate::kinds::{
    CLAW_APPROVAL_DENIED, CLAW_APPROVAL_GRANTED, CLAW_APPROVAL_REQUESTED, CLAW_GROUP_MESSAGE,
    CLAW_GROUP_REGISTER, CLAW_GROUP_SWITCH, CLAW_SCHEDULE_CREATE, CLAW_SCHEDULE_FIRE,
    CLAW_STREAM_CHUNK,
};
use crate::tools::ScheduleState;

// ── Display constants ─────────────────────────────────────────────────────────

const MAX_LOG_LINES: usize = 2000;

// ── TUI mode ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Idle,
    Working,
    AwaitingApproval {
        tool: String,
        args_preview: String,
    },
    /// Group picker overlay (Ctrl+G / Tab).
    GroupSelect,
}

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    mode: Mode,
    log: Vec<Line<'static>>,
    input: String,
    cursor_pos: usize,
    model: String,
    active_group: String,
    groups: Vec<String>,
    schedule_state: ScheduleState,
    stream_chars: usize,
    scroll_offset: usize,
    auto_scroll: bool,
    pending_stream_newline: bool,
    in_cycle: bool,
    /// Selected index in the group picker overlay.
    group_select_idx: usize,
    /// Cumulative session token count (input + output combined).
    session_tokens_total: u64,
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
                let mut content = span.content.to_string();
                content.push_str(s);
                span.content = content.into();
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

    fn scroll_up(&mut self, n: usize) {
        self.scroll_offset += n;
        self.auto_scroll = false;
    }

    fn scroll_down(&mut self, n: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(n);
        if self.scroll_offset == 0 {
            self.auto_scroll = true;
        }
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
        self.auto_scroll = true;
    }

    fn task_count(&self) -> usize {
        self.schedule_state.try_lock().map(|s| s.len()).unwrap_or(0)
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
    let mut col = 0usize;
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
            let mut j = i;
            while j < n && chars[j] != ' ' {
                j += 1;
            }
            let wlen = j - i;
            if col > 0 && col + wlen > width {
                rows += 1;
                col = 0;
            }
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

#[allow(clippy::cast_possible_truncation)]
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
            if col == width {
                row += 1;
                col = 0;
            }
            if i == cursor_char_abs {
                return (row, col as u16);
            }
            col += 1;
            i += 1;
        } else {
            let mut j = i;
            while j < n && chars[j] != ' ' {
                j += 1;
            }
            let wlen = j - i;
            if col > 0 && col + wlen > width {
                row += 1;
                col = 0;
            }
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
    (row, col as u16)
}

// ── Rendering ─────────────────────────────────────────────────────────────────

#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
fn render(frame: &mut Frame, state: &AppState) {
    let area = frame.area();
    let prefix = " > ";

    let input_display = format!("{}{}", prefix, state.input);
    let input_display_rows = word_wrap_rows(&input_display, area.width.max(1) as usize);
    let input_height = ((input_display_rows + 1) as u16).clamp(3, 8);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),            // status bar
            Constraint::Min(0),               // conversation
            Constraint::Length(input_height), // input bar
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
        Mode::GroupSelect => Span::styled(
            " ⊞ Select group",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
    };

    let task_count = state.task_count();
    let task_hint = if task_count > 0 {
        Span::styled(
            format!("  ⏰ {task_count}"),
            Style::default().fg(Color::Yellow),
        )
    } else {
        Span::raw("")
    };

    let scroll_hint = if state.scroll_offset > 0 {
        Span::styled(
            format!("  [↑{}↓ scroll]", state.scroll_offset),
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::raw("")
    };

    let token_hint = if state.session_tokens_total > 0 {
        let display = if state.session_tokens_total >= 1000 {
            format!("  tokens: {:.1}k", state.session_tokens_total as f64 / 1000.0)
        } else {
            format!("  tokens: {}", state.session_tokens_total)
        };
        Span::styled(display, Style::default().fg(Color::DarkGray))
    } else {
        Span::raw("")
    };

    let status_bar = Paragraph::new(Line::from(vec![
        Span::styled(
            format!(
                " claw │ {} │ {} │ {} ",
                state.active_group, state.model, "Ctrl+G: groups",
            ),
            Style::default().fg(Color::DarkGray),
        ),
        status_text,
        task_hint,
        token_hint,
        scroll_hint,
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(status_bar, chunks[0]);

    // ── Conversation log ──────────────────────────────────────────────────────
    let log_area = chunks[1];
    let visible_height = log_area.height as usize;

    // Build the paragraph first so we can use ratatui's own line_count() —
    // this uses the same wrapping algorithm as rendering, giving an exact row
    // count and eliminating scroll-offset drift that caused content to be
    // clipped at the bottom.
    let log_para = Paragraph::new(state.log.clone())
        .block(Block::default().borders(Borders::NONE))
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false });

    let total_rows = log_para.line_count(log_area.width);
    let max_scroll = total_rows.saturating_sub(visible_height);
    let effective_scroll = state.scroll_offset.min(max_scroll);
    let scroll_from_top = max_scroll - effective_scroll;

    let log_para = log_para.scroll((scroll_from_top as u16, 0));
    frame.render_widget(log_para, log_area);

    if effective_scroll > 0 {
        let indicator_text =
            format!(" ▼ {effective_scroll} more below  [↓/PgDn to scroll, G to bottom] ");
        let indicator = Paragraph::new(indicator_text).style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        );
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

    if !matches!(
        state.mode,
        Mode::AwaitingApproval { .. } | Mode::GroupSelect
    ) {
        let cursor_abs = prefix.len() + state.input[..state.cursor_pos].chars().count();
        let w = input_area.width.max(1) as usize;
        let (cursor_row, cursor_col) = word_wrap_cursor(&input_display, cursor_abs, w);
        frame.set_cursor_position((input_area.x + cursor_col, input_area.y + 1 + cursor_row));
    }

    // ── Overlays ──────────────────────────────────────────────────────────────
    if let Mode::AwaitingApproval { tool, args_preview } = &state.mode {
        render_approval_overlay(frame, frame.area(), tool, args_preview);
    }
    if matches!(state.mode, Mode::GroupSelect) {
        render_group_select(frame, frame.area(), &state.groups, state.group_select_idx);
    }
}

fn render_approval_overlay(frame: &mut Frame, area: Rect, tool: &str, preview: &str) {
    let popup_area = centred_rect_fixed(area.width.min(70), 10, area);
    frame.render_widget(Clear, popup_area);

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

fn render_group_select(frame: &mut Frame, area: Rect, groups: &[String], selected: usize) {
    let height = (groups.len() + 4).min(20) as u16;
    let popup_area = centred_rect_fixed(area.width.min(40), height, area);
    frame.render_widget(Clear, popup_area);

    let items: Vec<ListItem> = groups
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let style = if i == selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(format!("  {name}")).style(style)
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(selected));

    let list = List::new(items)
        .block(
            Block::default()
                .title(" ⊞  Switch Group  (↑↓ Enter Esc) ")
                .title_alignment(Alignment::Center)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_stateful_widget(list, popup_area, &mut list_state);
}

fn centred_rect_fixed(width: u16, height: u16, area: Rect) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

// ── Bus event handling ────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn handle_bus_event(state: &mut AppState, event: &Event) {
    match event.kind.as_str() {
        k if k == core_kinds::USER_MESSAGE => {
            let text = event.payload["text"].as_str().unwrap_or("").to_string();
            let source = event.payload["source"].as_str().unwrap_or("");
            // Only display external sources in a distinct way; TUI input is
            // already echoed when the user presses Enter.
            if source == "scheduler" || source == "http" || !source.is_empty() {
                state.push(Line::from(""));
                state.push(Line::from(vec![
                    Span::styled(
                        format!("  [{source}]  "),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::raw(text),
                ]));
            } else {
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
            }
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

        k if k == CLAW_STREAM_CHUNK => {
            if !state.in_cycle {
                return;
            }
            if matches!(state.mode, Mode::AwaitingApproval { .. }) {
                return;
            }

            let chunk = event.payload["content"].as_str().unwrap_or("");
            if state.pending_stream_newline && !chunk.trim().is_empty() {
                state.pending_stream_newline = false;
                state.push(Line::from(""));
                state.push(Line::from(vec![Span::styled(
                    "  agent  ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
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
            // Accumulate token counts from event metadata.
            if let Some(v) = event.metadata.get(meta_keys::LLM_INPUT_TOKENS).and_then(|v| v.as_u64()) {
                state.session_tokens_total += v;
            }
            if let Some(v) = event.metadata.get(meta_keys::LLM_OUTPUT_TOKENS).and_then(|v| v.as_u64()) {
                state.session_tokens_total += v;
            }

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
            state.pending_stream_newline = true;
        }

        k if k == core_kinds::AGENT_CYCLE_END => {
            state.in_cycle = false;
            state.mode = Mode::Idle;
            state.stream_chars = 0;
            state.pending_stream_newline = false;
            state.push(Line::from(""));
        }

        k if k == CLAW_APPROVAL_REQUESTED => {
            let tool = event.payload["tool"].as_str().unwrap_or("?").to_string();
            let args_preview = event
                .payload
                .get("args")
                .map(ToString::to_string)
                .unwrap_or_default();
            state.push(Line::from(vec![Span::styled(
                format!("  ⏸  approval required for {tool}  (Y/N)"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::DIM),
            )]));
            state.mode = Mode::AwaitingApproval { tool, args_preview };
        }

        k if k == CLAW_APPROVAL_GRANTED || k == CLAW_APPROVAL_DENIED => {
            let granted = k == CLAW_APPROVAL_GRANTED;
            state.push(Line::from(vec![Span::styled(
                if granted {
                    "  ▶  approved"
                } else {
                    "  ✗  denied"
                },
                Style::default()
                    .fg(if granted { Color::Green } else { Color::Red })
                    .add_modifier(Modifier::DIM),
            )]));
            if !matches!(state.mode, Mode::Idle) {
                state.mode = Mode::Working;
            }
        }

        // ── Claw-specific events ──────────────────────────────────────────────
        k if k == CLAW_SCHEDULE_FIRE => {
            let name = event.payload["name"].as_str().unwrap_or("?");
            let desc = event.payload["description"].as_str().unwrap_or("");
            let preview = if desc.len() > 60 {
                format!("{}…", &desc[..60])
            } else {
                desc.to_string()
            };
            state.push(Line::from(vec![
                Span::styled("  ⏰ ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format!("[{name}] {preview}"),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
        }

        k if k == CLAW_GROUP_MESSAGE => {
            let src = event.payload["source_group"].as_str().unwrap_or("?");
            let tgt = event.payload["target_group"].as_str().unwrap_or("?");
            let content = event.payload["content"].as_str().unwrap_or("");
            let preview = if content.len() > 50 {
                format!("{}…", &content[..50])
            } else {
                content.to_string()
            };
            state.push(Line::from(vec![
                Span::styled("  ✉ ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("{src} → {tgt}: {preview}"),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }

        k if k == CLAW_SCHEDULE_CREATE => {
            let name = event.payload["name"].as_str().unwrap_or("?");
            let schedule = event.payload["schedule"].as_str().unwrap_or("?");
            state.push(Line::from(vec![Span::styled(
                format!("  📅 scheduled: {name} ({schedule})"),
                Style::default().fg(Color::DarkGray),
            )]));
        }

        k if k == CLAW_GROUP_REGISTER => {
            let name = event.payload["name"].as_str().unwrap_or("?");
            state.push(Line::from(vec![Span::styled(
                format!("  ➕ registered group: {name}"),
                Style::default().fg(Color::Green),
            )]));
            if !state.groups.contains(&name.to_string()) {
                state.groups.push(name.to_string());
            }
        }

        _ => {}
    }
}

// ── Key action ────────────────────────────────────────────────────────────────

enum KeyAction {
    Quit,
    /// Switch to a different group bus.
    SwitchGroup(String),
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
            KeyCode::Char('c' | 'q') => return KeyAction::Quit,
            KeyCode::Char('x') => {
                cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                return KeyAction::Continue;
            }
            KeyCode::Char('g') => {
                if !matches!(state.mode, Mode::GroupSelect) {
                    state.group_select_idx = state
                        .groups
                        .iter()
                        .position(|g| *g == state.active_group)
                        .unwrap_or(0);
                    state.mode = Mode::GroupSelect;
                } else {
                    state.mode = Mode::Idle;
                }
                return KeyAction::Continue;
            }
            _ => {}
        }
    }

    match &state.mode.clone() {
        Mode::GroupSelect => match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                if state.group_select_idx > 0 {
                    state.group_select_idx -= 1;
                }
            }
            KeyCode::Down | KeyCode::Tab => {
                if state.group_select_idx + 1 < state.groups.len() {
                    state.group_select_idx += 1;
                }
            }
            KeyCode::Enter => {
                if let Some(name) = state.groups.get(state.group_select_idx).cloned() {
                    let previous = state.active_group.clone();
                    if name != previous {
                        state.active_group = name.clone();
                        state.push(Line::from(vec![Span::styled(
                            format!("  ⊞ switched to group: {name}"),
                            Style::default().fg(Color::Cyan),
                        )]));
                        // Publish on the OLD bus so workers can observe the switch.
                        let _ = bus
                            .publish(Event::new(
                                CLAW_GROUP_SWITCH,
                                json!({ "group": name, "previous": previous }),
                            ))
                            .await;
                        state.mode = Mode::Idle;
                        return KeyAction::SwitchGroup(name);
                    }
                }
                state.mode = Mode::Idle;
            }
            KeyCode::Esc => {
                state.mode = Mode::Idle;
            }
            _ => {}
        },

        Mode::Idle | Mode::Working => match key.code {
            KeyCode::Tab => {
                // Tab also opens group picker.
                state.group_select_idx = state
                    .groups
                    .iter()
                    .position(|g| *g == state.active_group)
                    .unwrap_or(0);
                state.mode = Mode::GroupSelect;
            }
            KeyCode::Char('G') => state.scroll_to_bottom(),
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
                        .publish(Event::new(core_kinds::USER_MESSAGE, json!({ "text": msg })))
                        .await;
                    state.mode = Mode::Working;
                }
            }
            _ => {}
        },

        Mode::AwaitingApproval { .. } => match key.code {
            KeyCode::Char('y' | 'Y') => {
                debug!("user granted tool approval");
                let _ = bus
                    .publish(Event::new(CLAW_APPROVAL_GRANTED, json!({})))
                    .await;
                state.mode = Mode::Working;
            }
            KeyCode::Char('n' | 'N') => {
                debug!("user denied tool approval");
                let _ = bus
                    .publish(Event::new(CLAW_APPROVAL_DENIED, json!({})))
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

/// Run the TUI. Blocks until the user quits (Ctrl+C or Ctrl+Q).
///
/// `get_bus` is called whenever the user switches groups; it returns the new
/// group's EventBus so the TUI can re-subscribe.
pub async fn run_tui(
    initial_bus: EventBus,
    model: String,
    active_group: Arc<Mutex<String>>,
    groups: Vec<String>,
    schedule_state: ScheduleState,
    get_bus: impl Fn(&str) -> Option<EventBus>,
    cancelled: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let initial_group = active_group.lock().await.clone();

    let mut state = AppState {
        mode: Mode::Idle,
        log: Vec::new(),
        input: String::new(),
        cursor_pos: 0,
        model,
        active_group: initial_group,
        groups,
        schedule_state,
        stream_chars: 0,
        scroll_offset: 0,
        auto_scroll: true,
        pending_stream_newline: false,
        in_cycle: false,
        group_select_idx: 0,
        session_tokens_total: 0,
    };

    state.push(Line::from(vec![
        Span::styled(
            " claw",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  │  Ctrl+C quit  │  Ctrl+X cancel  │  Ctrl+G / Tab: switch group"),
    ]));
    state.push(Line::from(vec![Span::styled(
        " Scroll: ↑↓ PgUp/PgDn  │  Jump to bottom: G",
        Style::default().fg(Color::DarkGray),
    )]));
    state.push(Line::from(""));

    let mut current_bus = initial_bus;
    let mut bus_rx = current_bus.subscribe();
    let mut crossterm_events = EventStream::new();
    let mut frame_tick = tokio::time::interval(std::time::Duration::from_millis(33));
    frame_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut needs_redraw = true;

    let result: anyhow::Result<()> = async {
        loop {
            tokio::select! {
                _ = frame_tick.tick() => {
                    if needs_redraw {
                        terminal.draw(|f| render(f, &state))?;
                        needs_redraw = false;
                    }
                }

                maybe_event = bus_rx.recv() => {
                    match maybe_event {
                        Some(event) => {
                            handle_bus_event(&mut state, &event);
                            needs_redraw = true;
                        }
                        None => break,
                    }
                }

                maybe_ct = crossterm_events.next() => {
                    match maybe_ct {
                        Some(Ok(CrosstermEvent::Key(key))) => {
                            match handle_key(&mut state, key, &current_bus, &cancelled).await {
                                KeyAction::Quit => break,
                                KeyAction::SwitchGroup(name) => {
                                    if let Some(new_bus) = get_bus(&name) {
                                        // Update the shared active_group so other parts of the
                                        // app know which group is active.
                                        *active_group.lock().await = name;
                                        current_bus = new_bus;
                                        bus_rx = current_bus.subscribe();
                                    }
                                }
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
