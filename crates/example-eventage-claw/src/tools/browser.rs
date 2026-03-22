//! Browser tool — invokes a headless Chromium subprocess.
//!
//! Falls back gracefully if Chromium is not found on the system.

use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;
use tracing::warn;

pub struct BrowserTool {
    /// Path to the chromium/chrome binary. None = search PATH.
    pub chromium_path: Option<PathBuf>,
    /// Directory to store screenshots.
    pub screenshots_dir: PathBuf,
}

impl BrowserTool {
    pub fn new(screenshots_dir: PathBuf) -> Self {
        Self {
            chromium_path: find_chromium(),
            screenshots_dir,
        }
    }
}

fn find_chromium() -> Option<PathBuf> {
    for name in &[
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
    ] {
        if let Ok(output) = std::process::Command::new("which").arg(name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Some(PathBuf::from(path));
                }
            }
        }
    }
    None
}

fn chromium_bin(path: &Option<PathBuf>) -> Option<String> {
    path.as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .or_else(|| {
            for name in &["chromium", "chromium-browser", "google-chrome"] {
                if std::process::Command::new("which")
                    .arg(name)
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return Some(name.to_string());
                }
            }
            None
        })
}

#[async_trait]
impl Tool for BrowserTool {
    fn definition(&self) -> ToolDefinition {
        let has_chromium = self.chromium_path.is_some();
        let desc = if has_chromium {
            "Control a headless browser. Actions: navigate (get page text), screenshot (save PNG)."
        } else {
            "Control a headless browser (requires chromium). Actions: navigate, screenshot."
        };
        ToolDefinition::function(
            "browser",
            desc,
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "screenshot"],
                        "description": "'navigate' returns page text; 'screenshot' saves a PNG."
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to open."
                    }
                },
                "required": ["action", "url"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let action = args["action"].as_str().unwrap_or("navigate");
        let url = args["url"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'url'".into()))?;

        let bin = match chromium_bin(&self.chromium_path) {
            Some(b) => b,
            None => {
                warn!("BrowserTool: chromium not found, falling back to web_fetch");
                return Err(AgentError::Tool(
                    "chromium not found; install chromium or use web_fetch instead".into(),
                ));
            }
        };

        match action {
            "navigate" => {
                let result = tokio::time::timeout(
                    Duration::from_secs(15),
                    tokio::process::Command::new(&bin)
                        .args(["--headless", "--disable-gpu", "--dump-dom", url])
                        .output(),
                )
                .await;

                match result {
                    Err(_) => Err(AgentError::Tool("browser navigate timed out".into())),
                    Ok(Err(e)) => Err(AgentError::Tool(format!("browser spawn failed: {e}"))),
                    Ok(Ok(output)) => {
                        let html = String::from_utf8_lossy(&output.stdout).to_string();
                        let text = strip_html_tags(&html);
                        let truncated = text.chars().count() > 8000;
                        let content: String = text.chars().take(8000).collect();
                        Ok(json!({
                            "url": url,
                            "content": content,
                            "truncated": truncated,
                            "success": true,
                        }))
                    }
                }
            }

            "screenshot" => {
                let _ = tokio::fs::create_dir_all(&self.screenshots_dir).await;
                let filename = format!(
                    "screenshot_{}.png",
                    chrono::Utc::now().format("%Y%m%d_%H%M%S")
                );
                let path = self.screenshots_dir.join(&filename);

                let result = tokio::time::timeout(
                    Duration::from_secs(15),
                    tokio::process::Command::new(&bin)
                        .args([
                            "--headless",
                            "--disable-gpu",
                            &format!("--screenshot={}", path.display()),
                            "--window-size=1280,800",
                            url,
                        ])
                        .output(),
                )
                .await;

                match result {
                    Err(_) => Err(AgentError::Tool("browser screenshot timed out".into())),
                    Ok(Err(e)) => Err(AgentError::Tool(format!("browser spawn failed: {e}"))),
                    Ok(Ok(output)) => {
                        let saved = path.exists();
                        Ok(json!({
                            "url": url,
                            "screenshot_path": path.display().to_string(),
                            "saved": saved,
                            "stderr": String::from_utf8_lossy(&output.stderr).trim().to_string(),
                            "success": saved,
                        }))
                    }
                }
            }

            _ => Err(AgentError::Tool(format!(
                "unknown browser action: {action}"
            ))),
        }
    }
}

fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    let mut in_script = false;
    let lower = html.to_lowercase();
    let mut i = 0;
    let chars: Vec<char> = html.chars().collect();

    while i < chars.len() {
        if !in_tag && !in_script && i + 7 < lower.len() && &lower[i..i + 7] == "<script" {
            in_script = true;
            in_tag = true;
        } else if in_script && i + 9 < lower.len() && &lower[i..i + 9] == "</script>" {
            in_script = false;
            in_tag = false;
            i += 9;
            continue;
        }

        if chars[i] == '<' {
            in_tag = true;
        } else if chars[i] == '>' {
            in_tag = false;
            if !in_script {
                result.push(' ');
            }
        } else if !in_tag && !in_script {
            result.push(chars[i]);
        }
        i += 1;
    }

    // Collapse whitespace
    result
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}
