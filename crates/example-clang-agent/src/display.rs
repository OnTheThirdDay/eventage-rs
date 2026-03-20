/// Terminal display utilities for the C agent REPL.
///
/// Uses ANSI escape codes.  Respects the `NO_COLOR` environment variable.
use eventage::{kinds, Event};
use std::io::Write;

// ── ANSI helpers ──────────────────────────────────────────────────────────────

fn color_enabled() -> bool {
    std::env::var("NO_COLOR").is_err()
}

macro_rules! ansi {
    ($code:literal) => {
        if color_enabled() {
            $code
        } else {
            ""
        }
    };
}

fn bold() -> &'static str {
    ansi!("\x1b[1m")
}
fn dim() -> &'static str {
    ansi!("\x1b[2m")
}
fn green() -> &'static str {
    ansi!("\x1b[32m")
}
fn red() -> &'static str {
    ansi!("\x1b[31m")
}
fn yellow() -> &'static str {
    ansi!("\x1b[33m")
}
fn cyan() -> &'static str {
    ansi!("\x1b[36m")
}
fn reset() -> &'static str {
    ansi!("\x1b[0m")
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Print the welcome banner.
pub fn print_banner(session_id: &str, model: &str, workspace: &str) {
    let bar = "━".repeat(60);
    println!("\n{}{}{}", bold(), bar, reset());
    println!(
        "  {}C Agent{}  ·  session {}{}{} ",
        bold(),
        reset(),
        cyan(),
        session_id,
        reset()
    );
    println!("  model: {}  ·  workspace: {}", model, workspace);
    println!("{}{}{}\n", bold(), bar, reset());
    println!(
        "{}Type your C programming request. Press Ctrl-C to exit.{}\n",
        dim(),
        reset()
    );
}

/// Print the `You: ` prompt.
pub fn print_prompt() {
    print!("\n{}You:{} ", bold(), reset());
    std::io::stdout().flush().ok();
}

/// "thinking…" indicator shown after the user submits a message.
pub fn print_thinking() {
    println!("{}  thinking…{}", dim(), reset());
}

/// Display a single event as it arrives on the bus (real-time streaming output).
///
/// Called by the display task once per event.  Observability-only events
/// (AGENT_CYCLE_START/END) are silently skipped.
pub fn display_event_live(event: &Event) {
    match event.kind.as_str() {
        kinds::TOOL_CALL_PROPOSED => {
            let name = event.payload["name"].as_str().unwrap_or("?");
            let args = event.payload["arguments"].as_str().unwrap_or("{}");
            print_tool_call(name, args);
        }
        kinds::TOOL_RESULT => {
            let name = event.payload["name"].as_str().unwrap_or("?");
            if let Some(result) = event.payload.get("result") {
                print_tool_result_ok(name, result);
            } else if let Some(err) = event.payload.get("error") {
                print_tool_result_err(name, err.as_str().unwrap_or("unknown"));
            }
        }
        kinds::ASSISTANT_MESSAGE => {
            // Show text content only on turns with no tool calls (the final summary).
            // Strip <think>…</think> blocks that qwen3 and similar models emit as
            // internal reasoning — only the text after </think> is shown.
            let raw = event.payload["content"].as_str().unwrap_or("");
            let content = strip_thinking(raw).trim();
            let has_tool_calls = event.payload["tool_calls"]
                .as_array()
                .is_some_and(|arr| !arr.is_empty());
            if !content.is_empty() && !has_tool_calls {
                println!("\n{}Assistant:{} {}", bold(), reset(), content);
            }
        }
        kinds::AGENT_CYCLE_END => {
            println!("\n{}{}{}", dim(), "─".repeat(60), reset());
        }
        _ => {}
    }
}

/// Print the active sandbox mode.
pub fn print_sandbox(name: &str) {
    println!("  sandbox: {}{}{}\n", cyan(), name, reset());
}

/// Print a startup message when resuming a session.
pub fn print_resumed(event_count: usize) {
    println!(
        "{}Resumed session — {} events loaded.{}\n",
        green(),
        event_count,
        reset()
    );
}

/// Print a non-fatal error message.
pub fn print_error(msg: &str) {
    eprintln!("\n{}error:{} {}", red(), reset(), msg);
}

/// Print the farewell message on exit.
pub fn print_farewell(session_id: &str) {
    println!(
        "\n{}Session {} saved. Goodbye.{}",
        dim(),
        session_id,
        reset()
    );
}

/// Strip `<think>…</think>` reasoning blocks emitted by qwen3 and similar models.
/// Returns only the text that follows the closing tag (the actual response).
fn strip_thinking(s: &str) -> &str {
    if let Some(end) = s.find("</think>") {
        s[end + "</think>".len()..].trim_start()
    } else {
        s
    }
}

// ── Internal formatters ───────────────────────────────────────────────────────

fn print_tool_call(name: &str, args_raw: &str) {
    let label = match name {
        "write_file" => {
            let path = json_field_str(args_raw, "path").unwrap_or_default();
            format!("writing   {}{}{}", yellow(), path, reset())
        }
        "read_file" => {
            let path = json_field_str(args_raw, "path").unwrap_or_default();
            format!("reading   {}{}{}", yellow(), path, reset())
        }
        "list_files" => "listing workspace".to_string(),
        "compile" => {
            let src = json_field_str(args_raw, "source").unwrap_or_default();
            let out = json_field_str(args_raw, "output").unwrap_or_default();
            format!(
                "compiling {}{}{} → {}bin/{}{}",
                yellow(),
                src,
                reset(),
                yellow(),
                out,
                reset()
            )
        }
        "execute" => {
            let bin = json_field_str(args_raw, "binary").unwrap_or_default();
            format!("running   {}{}{}", yellow(), bin, reset())
        }
        other => format!("calling   {}", other),
    };
    println!("  {}▶{} {}", cyan(), reset(), label);
}

fn print_tool_result_ok(name: &str, result: &serde_json::Value) {
    match name {
        "write_file" => {
            let bytes = result["bytes_written"].as_u64().unwrap_or(0);
            println!("    {}✓{} {} bytes written", green(), reset(), bytes);
        }
        "read_file" => {
            let bytes = result["size_bytes"].as_u64().unwrap_or(0);
            println!("    {}✓{} {} bytes read", green(), reset(), bytes);
        }
        "list_files" => {
            let count = result["count"].as_u64().unwrap_or(0);
            println!("    {}✓{} {} file(s)", green(), reset(), count);
        }
        "compile" => {
            let success = result["success"].as_bool().unwrap_or(false);
            let errors = result["errors"].as_str().unwrap_or("").trim();
            let warnings = result["warnings"].as_str().unwrap_or("").trim();
            let sandbox = result["sandbox"].as_str().unwrap_or("?");
            if success {
                let out = result["output_path"].as_str().unwrap_or("?");
                println!(
                    "    {}✓{} compiled → {}{}{} {}(via {}){}",
                    green(),
                    reset(),
                    yellow(),
                    out,
                    reset(),
                    dim(),
                    sandbox,
                    reset()
                );
                if !warnings.is_empty() {
                    print_block("warnings", warnings, yellow());
                }
            } else {
                println!(
                    "    {}✗{} compilation failed {}(via {}){}",
                    red(),
                    reset(),
                    dim(),
                    sandbox,
                    reset()
                );
                if !errors.is_empty() {
                    print_block("errors", errors, red());
                }
            }
        }
        "execute" => {
            let exit_code = result["exit_code"].as_i64().unwrap_or(-1);
            let timed_out = result["timed_out"].as_bool().unwrap_or(false);
            let stdout = result["stdout"].as_str().unwrap_or("").trim();
            let stderr = result["stderr"].as_str().unwrap_or("").trim();
            let sandbox = result["sandbox"].as_str().unwrap_or("?");

            if timed_out {
                println!(
                    "    {}✗{} timed out {}(via {}){}",
                    red(),
                    reset(),
                    dim(),
                    sandbox,
                    reset()
                );
            } else if exit_code == 0 {
                println!(
                    "    {}✓{} exit 0 {}(via {}){}",
                    green(),
                    reset(),
                    dim(),
                    sandbox,
                    reset()
                );
            } else {
                println!(
                    "    {}✗{} exit {} {}(via {}){}",
                    red(),
                    reset(),
                    exit_code,
                    dim(),
                    sandbox,
                    reset()
                );
            }
            if !stdout.is_empty() {
                print_block("stdout", stdout, dim());
            }
            if !stderr.is_empty() {
                print_block("stderr", stderr, yellow());
            }
        }
        _ => {
            println!("    {}✓{} done", green(), reset());
        }
    }
}

fn print_tool_result_err(name: &str, err: &str) {
    println!("    {}✗{} {}: {}", red(), reset(), name, err);
}

fn print_block(header: &str, text: &str, color: &str) {
    println!("      {}{}:{}", color, header, reset());
    for line in text.lines().take(40) {
        println!("        {}{}{}", dim(), line, reset());
    }
    let total = text.lines().count();
    if total > 40 {
        println!("        {}… ({} more lines){}", dim(), total - 40, reset());
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract a string field from a raw JSON string.
fn json_field_str(raw: &str, key: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    v.get(key)?.as_str().map(String::from)
}

/// Format a compact summary of an event payload (for /log command).
pub fn summarise_payload(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(m) => {
            let parts: Vec<String> = m
                .iter()
                .take(3)
                .map(|(k, v)| {
                    let val = match v {
                        serde_json::Value::String(s) => {
                            if s.len() > 40 {
                                format!("\"{}…\"", &s[..40])
                            } else {
                                format!("\"{s}\"")
                            }
                        }
                        other => other.to_string(),
                    };
                    format!("{k}: {val}")
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        other => other.to_string(),
    }
}
