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
//! ## Layout
//!
//! ```text
//! ┌──────────────────────── example-coding-agent ─────────────────────────┐
//! │ model: gpt-4o │ session: abc-123 │ ● Working...                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  you  Write a Python web scraper for Hacker News                        │
//! │                                                                         │
//! │  [tool: write_file]  src/scraper.py                                     │
//! │  ✓ 2847 bytes written                                                   │
//! │                                                                         │
//! │  agent  I've written a web scraper that fetches the HN frontpage…       │
//! │         ▌  ← streaming cursor                                           │
//! │                                                                         │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ > _                                                                     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! When `AwaitingApproval`:
//! ```text
//! ╔════════════════════ ⚠ Security Gate ═══════════════════════╗
//! ║  Tool: execute_shell                                       ║
//! ║  Command: python scraper.py                                ║
//! ║                                                            ║
//! ║  [Y] Allow    [N] Deny                                     ║
//! ╚════════════════════════════════════════════════════════════╝
//! ```

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
    widgets::{Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use serde_json::json;
use tracing::debug;

use crate::kinds::{
    CODING_APPROVAL_DENIED, CODING_APPROVAL_GRANTED, CODING_APPROVAL_REQUESTED,
    CODING_STREAM_CHUNK, CODING_TURN_DIFF,
};

// ── Display constants ─────────────────────────────────────────────────────────

const MAX_LOG_LINES: usize = 1000; // rolling conversation buffer

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
    /// Session info for the status bar.
    model: String,
    session_id: String,
    /// Incremented on each streaming chunk to drive a cursor blink.
    stream_chars: usize,
}

impl AppState {
    fn push(&mut self, line: Line<'static>) {
        self.log.push(line);
        if self.log.len() > MAX_LOG_LINES {
            self.log.remove(0);
        }
    }

    fn push_plain(&mut self, s: impl Into<String>) {
        self.push(Line::from(s.into()));
    }

    fn append_to_last(&mut self, s: &str) {
        // Append streaming content to the last line (or start a new one).
        if let Some(last) = self.log.last_mut() {
            // Extend the last span's content.
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
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

fn render(frame: &mut Frame, state: &AppState) {
    // ── Outer layout: header | body | input ──────────────────────────────────
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // status bar
            Constraint::Min(0),    // conversation
            Constraint::Length(3), // input bar
        ])
        .split(frame.area());

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
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(status_bar, chunks[0]);

    // ── Conversation log ──────────────────────────────────────────────────────
    let visible_height = chunks[1].height as usize;
    let start = state.log.len().saturating_sub(visible_height);
    let visible: Vec<ListItem> = state.log[start..]
        .iter()
        .map(|line| ListItem::new(line.clone()))
        .collect();

    let conversation = List::new(visible)
        .block(Block::default().borders(Borders::NONE))
        .style(Style::default().fg(Color::White));
    frame.render_widget(conversation, chunks[1]);

    // ── Input bar ─────────────────────────────────────────────────────────────
    let input_display = if state.mode == Mode::Idle {
        format!(" > {}_", state.input)
    } else {
        format!(" > {}", state.input)
    };

    let input_bar = Paragraph::new(input_display)
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .style(Style::default().fg(Color::White));
    frame.render_widget(input_bar, chunks[2]);

    // ── Approval overlay (rendered on top) ────────────────────────────────────
    if let Mode::AwaitingApproval { tool, args_preview } = &state.mode {
        render_approval_overlay(frame, frame.area(), tool, args_preview);
    }
}

fn render_approval_overlay(frame: &mut Frame, area: Rect, tool: &str, preview: &str) {
    // Centre a popup that is ~60% wide and ~40% tall.
    let popup_area = centred_rect(62, 42, area);
    frame.render_widget(Clear, popup_area);

    // Truncate preview if needed.
    let max_preview = (popup_area.width as usize).saturating_sub(6);
    let preview_display = if preview.len() > max_preview {
        format!("{}…", &preview[..max_preview])
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
            state.push(Line::from(""));
            state.push(Line::from(vec![Span::styled(
                "  agent  ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )]));
        }

        k if k == CODING_STREAM_CHUNK => {
            let chunk = event.payload["content"].as_str().unwrap_or("");
            // Split chunk on newlines and handle each line.
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
            // Non-streaming assistant message (fallback mode).
            if let Some(content) = event.payload["content"].as_str() {
                if !content.is_empty() {
                    // Only add if we didn't stream it already.
                    if state.stream_chars == 0 {
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
                    format!("  [tool: {name}]  "),
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
            // Quick summary from result.
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
        }

        k if k == core_kinds::AGENT_CYCLE_END => {
            state.mode = Mode::Idle;
            state.stream_chars = 0;
            state.push(Line::from(""));
        }

        k if k == CODING_APPROVAL_REQUESTED => {
            let tool = event.payload["tool"].as_str().unwrap_or("?").to_string();
            let args_preview = event
                .payload
                .get("args")
                .map(|v| v.to_string())
                .unwrap_or_default();
            state.mode = Mode::AwaitingApproval { tool, args_preview };
        }

        k if k == CODING_APPROVAL_GRANTED || k == CODING_APPROVAL_DENIED => {
            // Approval resolved — back to working mode.
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

        _ => {}
    }
}

// ── Keyboard handling ─────────────────────────────────────────────────────────

async fn handle_key(
    state: &mut AppState,
    key: KeyEvent,
    bus: &EventBus,
    cancelled: &Arc<AtomicBool>,
) -> bool /* returns true to quit */ {
    // ── Global shortcuts ──────────────────────────────────────────────────────
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') | KeyCode::Char('q') => {
                return true; // quit
            }
            KeyCode::Char('x') => {
                // Cancel current LLM stream.
                cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                return false;
            }
            _ => {}
        }
    }

    match &state.mode {
        // ── Idle: accept input ────────────────────────────────────────────────
        Mode::Idle => match key.code {
            KeyCode::Char(c) => {
                state.input.push(c);
            }
            KeyCode::Backspace => {
                state.input.pop();
            }
            KeyCode::Enter => {
                let msg = state.input.trim().to_string();
                if !msg.is_empty() {
                    state.input.clear();
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

        // ── Working: only global shortcuts apply ──────────────────────────────
        Mode::Working => {}

        // ── AwaitingApproval: y/n toggle ──────────────────────────────────────
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
            _ => {}
        },
    }

    false
}

// ── Main TUI loop ─────────────────────────────────────────────────────────────

/// Run the TUI. This function blocks until the user quits (Ctrl+C or Ctrl+Q).
///
/// It owns the terminal for its entire lifetime and restores it on exit.
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
        model,
        session_id,
        stream_chars: 0,
    };

    state.push(Line::from(vec![
        Span::styled(
            " example-coding-agent",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  │  Ctrl+C to quit  │  Ctrl+X to cancel generation"),
    ]));
    state.push(Line::from(""));

    let mut bus_rx = bus.subscribe();
    let mut crossterm_events = EventStream::new();

    let result: anyhow::Result<()> = async {
        loop {
            terminal.draw(|f| render(f, &state))?;

            tokio::select! {
                maybe_event = bus_rx.recv() => {
                    match maybe_event {
                        Some(event) => handle_bus_event(&mut state, event),
                        None => break,
                    }
                }

                maybe_ct = crossterm_events.next() => {
                    match maybe_ct {
                        Some(Ok(CrosstermEvent::Key(key))) => {
                            if handle_key(&mut state, key, &bus, &cancelled).await {
                                break;
                            }
                        }
                        Some(Ok(_)) => {} // mouse, resize, etc.
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
