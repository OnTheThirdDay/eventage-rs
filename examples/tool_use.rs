//! Tool-use example — AI agent with a weather tool and calculator.
//!
//! Demonstrates:
//! - Implementing the [`Tool`] trait.
//! - Registering tools via `Session::builder().tool()`.
//! - Using `Session::chat()` for the full ReAct loop.
//!
//! # Setup
//! Requires Ollama running locally with the `qwen3:4b` model:
//! ```bash
//! ollama pull qwen3:4b
//! ```
//!
//! # Run
//! ```bash
//! cargo run -p example-tool-use
//! ```

use async_trait::async_trait;
use eventage::kinds;
use eventage::llm::{types::ToolDefinition, OpenAiProvider};
use eventage::{AgentError, Session, Tool};
use serde_json::{json, Value};

// ── Weather tool (simulated) ──────────────────────────────────────────────────

/// Simulates a weather API. In production, replace with a real HTTP call.
struct GetWeather;

#[async_trait]
impl Tool for GetWeather {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "get_weather",
            "Returns the current weather for a city. \
             Call this whenever the user asks about weather conditions.",
            json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city, e.g. 'London' or 'Tokyo'."
                    }
                },
                "required": ["city"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let city = args["city"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'city'".into()))?;

        // Simulate an API response.
        let (temp_c, condition) = match city.to_lowercase().as_str() {
            "london" => (12, "cloudy with light rain"),
            "tokyo" => (22, "sunny"),
            "new york" => (18, "partly cloudy"),
            "sydney" => (25, "clear"),
            _ => (20, "mild"),
        };

        Ok(json!({
            "city": city,
            "temperature_celsius": temp_c,
            "condition": condition,
            "humidity_percent": 60,
            "wind_kmh": 15
        }))
    }
}

// ── Calculator tool ───────────────────────────────────────────────────────────

struct Calculate;

#[async_trait]
impl Tool for Calculate {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "calculate",
            "Evaluates a simple arithmetic expression (addition, subtraction, \
             multiplication, division). Use this for any numeric calculation.",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(98.6 - 32) * 5/9'"
                    }
                },
                "required": ["expression"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let expr = args["expression"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'expression'".into()))?;

        let result = eval(expr).map_err(|e| AgentError::Tool(format!("eval error: {e}")))?;

        Ok(json!({ "result": result, "expression": expr }))
    }
}

/// Minimal recursive-descent evaluator: handles +−×÷ and parentheses.
fn eval(expr: &str) -> Result<f64, String> {
    let tokens = tokenise(expr)?;
    let mut pos = 0;
    let val = parse_expr(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!("unexpected token at position {pos}"));
    }
    Ok(val)
}

#[derive(Debug, Clone)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenise(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' => {
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        num.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Num(
                    num.parse().map_err(|_| format!("bad number: {num}"))?,
                ));
            }
            '+' | '-' | '*' | '/' => {
                tokens.push(Token::Op(c));
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            other => return Err(format!("unexpected character: '{other}'")),
        }
    }
    Ok(tokens)
}

fn parse_expr(t: &[Token], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_term(t, pos)?;
    while *pos < t.len() {
        match t[*pos] {
            Token::Op('+') => {
                *pos += 1;
                left += parse_term(t, pos)?;
            }
            Token::Op('-') => {
                *pos += 1;
                left -= parse_term(t, pos)?;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_term(t: &[Token], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_factor(t, pos)?;
    while *pos < t.len() {
        match t[*pos] {
            Token::Op('*') => {
                *pos += 1;
                left *= parse_factor(t, pos)?;
            }
            Token::Op('/') => {
                *pos += 1;
                let right = parse_factor(t, pos)?;
                if right == 0.0 {
                    return Err("division by zero".into());
                }
                left /= right;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_factor(t: &[Token], pos: &mut usize) -> Result<f64, String> {
    if *pos >= t.len() {
        return Err("unexpected end of expression".into());
    }
    match t[*pos].clone() {
        Token::Num(n) => {
            *pos += 1;
            Ok(n)
        }
        Token::Op('-') => {
            *pos += 1;
            Ok(-parse_factor(t, pos)?)
        }
        Token::LParen => {
            *pos += 1;
            let val = parse_expr(t, pos)?;
            if *pos >= t.len() {
                return Err("missing closing parenthesis".into());
            }
            match t[*pos] {
                Token::RParen => {
                    *pos += 1;
                    Ok(val)
                }
                _ => Err("expected ')'".into()),
            }
        }
        _ => Err(format!("unexpected token: {:?}", t[*pos])),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .with_writer(std::io::stderr)
        .init();

    // Session wraps agent + bus. Tools are registered directly on the builder.
    let mut session = Session::builder()
        .llm(OpenAiProvider::ollama("qwen3:4b"))
        .system_prompt(
            "You are a helpful assistant. \
             Use the get_weather tool for weather questions and \
             the calculate tool for any arithmetic. \
             Always use a tool when one is relevant — do not guess.",
        )
        .tool(GetWeather)
        .tool(Calculate)
        .max_steps(10)
        .build();

    // Two example prompts — run both cycles sequentially.
    let prompts = [
        "What is the weather like in Tokyo right now?",
        "If it is 98.6°F, what is that in Celsius?",
    ];

    for prompt in &prompts {
        println!("User: {prompt}");

        // chat() handles the full ReAct loop: LLM → tool calls → LLM → ...
        let reply = session.chat(prompt).await?;
        println!("Assistant: {}\n", strip_thinking(&reply));
    }

    // Print a concise event summary using the underlying bus.
    let log = session.bus().log().await;
    println!("─── Event log ({} events) ───", log.len());
    for event in &log {
        match event.kind.as_str() {
            kinds::USER_MESSAGE => {
                let text = event.payload["text"].as_str().unwrap_or("");
                println!("[user]        {text}");
            }
            kinds::TOOL_CALL_PROPOSED => {
                let name = event.payload["name"].as_str().unwrap_or("?");
                println!("[tool call]   {name}(...)");
            }
            kinds::TOOL_RESULT => {
                let name = event.payload["name"].as_str().unwrap_or("?");
                let ok = event.payload.get("result").is_some();
                println!(
                    "[tool result] {} → {}",
                    name,
                    if ok { "ok" } else { "error" }
                );
            }
            kinds::ASSISTANT_MESSAGE => {
                let content = event.payload["content"]
                    .as_str()
                    .map(strip_thinking)
                    .unwrap_or("");
                let tool_calls = event.payload["tool_calls"]
                    .as_array()
                    .map_or(0, |a| a.len());
                if tool_calls > 0 {
                    println!("[assistant]   ({tool_calls} tool call(s))");
                } else if !content.is_empty() {
                    let short = if content.len() > 80 {
                        &content[..80]
                    } else {
                        content
                    };
                    println!("[assistant]   {short}…");
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn strip_thinking(s: &str) -> &str {
    if let Some(end) = s.find("</think>") {
        s[end + "</think>".len()..].trim_start()
    } else {
        s
    }
}
