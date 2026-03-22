//! Web tools: web_search (DuckDuckGo) and web_fetch.

use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use reqwest::Client;
use serde_json::{json, Value};

fn tool_err(msg: impl Into<String>) -> AgentError {
    AgentError::Tool(msg.into())
}

// ── WebSearchTool ─────────────────────────────────────────────────────────────

/// Search the web using DuckDuckGo's HTML endpoint (no API key required).
pub struct WebSearchTool {
    client: Client,
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                .build()
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "web_search",
            "Search the web for current information. Returns titles, URLs, and snippets.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)."
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'query'"))?;
        let max_results = args["max_results"].as_u64().unwrap_or(5) as usize;

        let url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencoding(query)
        );

        let resp = self
            .client
            .get(&url)
            .header("Accept", "text/html")
            .send()
            .await
            .map_err(|e| tool_err(format!("web_search request failed: {e}")))?;

        let html = resp
            .text()
            .await
            .map_err(|e| tool_err(format!("web_search read failed: {e}")))?;

        let results = parse_ddg_results(&html, max_results);

        if results.is_empty() {
            return Err(tool_err(
                "web_search returned no results — DuckDuckGo may have returned a CAPTCHA or the query has no matches. Try rephrasing.",
            ));
        }

        Ok(json!({
            "query": query,
            "results": results,
            "count": results.len(),
        }))
    }
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                vec![c as u8]
            }
            ' ' => vec![b'+'],
            c => {
                let mut buf = [0u8; 4];
                let encoded = c.encode_utf8(&mut buf);
                encoded
                    .bytes()
                    .flat_map(|b| format!("%{b:02X}").into_bytes())
                    .collect()
            }
        })
        .map(|b| b as char)
        .collect()
}

/// Very light HTML parser for DuckDuckGo search results.
fn parse_ddg_results(html: &str, max: usize) -> Vec<Value> {
    let mut results = vec![];

    // DuckDuckGo HTML results are in <div class="result__body"> blocks
    // Each result has: .result__title (with <a href>), .result__snippet
    // We use simple string scanning rather than a full HTML parser.

    let mut pos = 0;
    while results.len() < max {
        // Find a result title link
        let Some(title_start) = html[pos..].find("class=\"result__a\"") else {
            break;
        };
        let abs_title_start = pos + title_start;

        // Backtrack to find the href — bounds-checked to avoid panics on short HTML.
        let href_end = (abs_title_start + 200).min(html.len());
        let href_region = &html[abs_title_start.saturating_sub(300)..href_end];
        let url = extract_ddg_url(href_region);

        // Extract title text — guard the +20 lookahead.
        let after_attr = abs_title_start + "class=\"result__a\"".len();
        let lookahead_end = (after_attr + 20).min(html.len());
        let title = if let Some(gt) = html[after_attr..lookahead_end].find('>') {
            let content_start = after_attr + gt + 1;
            let Some(end) = html[content_start..].find("</a>") else {
                pos = abs_title_start + 1;
                continue;
            };
            let content_end = (content_start + end).min(html.len());
            strip_html_tags(&html[content_start..content_end])
        } else {
            pos = abs_title_start + 1;
            continue;
        };

        // Find snippet after this point
        let after = abs_title_start + 300;
        let snippet = if after < html.len() {
            if let Some(snip_start) = html[after..].find("result__snippet") {
                let s = after + snip_start + "result__snippet".len();
                if let Some(gt) = html[s..s.min(s + 50)].find('>') {
                    let cs = s + gt + 1;
                    if let Some(end) = html[cs..].find("</") {
                        let raw = &html[cs..cs + end.min(300)];
                        strip_html_tags(raw)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if !title.is_empty() || !url.is_empty() {
            results.push(json!({
                "title": title.trim(),
                "url": url,
                "snippet": snippet.trim(),
            }));
        }

        pos = abs_title_start + 1;
    }

    results
}

fn extract_ddg_url(region: &str) -> String {
    // DuckDuckGo wraps URLs in uddg= query param or href="/l/?kh=-1&uddg=..."
    if let Some(uddg) = region.find("uddg=") {
        let start = uddg + 5;
        let end = region[start..]
            .find(['&', '"', '\''])
            .map(|e| start + e)
            .unwrap_or(region.len());
        let encoded = &region[start..end];
        // URL-decode
        url_decode(encoded)
    } else if let Some(href_pos) = region.find("href=\"") {
        let start = href_pos + 6;
        let end = region[start..]
            .find('"')
            .map(|e| start + e)
            .unwrap_or(region.len());
        region[start..end].to_string()
    } else {
        String::new()
    }
}

fn url_decode(s: &str) -> String {
    let mut result = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    // char::from(u8) widens to Unicode scalar — safe for Latin-1 code points.
                    result.push(char::from(byte));
                    i += 3;
                    continue;
                }
            }
        } else if bytes[i] == b'+' {
            result.push(' ');
            i += 1;
            continue;
        }
        result.push(char::from(bytes[i]));
        i += 1;
    }
    result
}

fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    let mut in_script = false;
    let lower = html.to_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;

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

// ── WebFetchTool ──────────────────────────────────────────────────────────────

/// Fetch a URL and return its text content (HTML stripped).
pub struct WebFetchTool {
    client: Client,
    pub max_bytes: usize,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("Mozilla/5.0 (compatible; claw-bot/0.1)")
                .build()
                .unwrap_or_default(),
            max_bytes: 50_000,
        }
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "web_fetch",
            "Fetch the content of a URL and return it as plain text.",
            json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch."
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default: 5000)."
                    }
                },
                "required": ["url"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let url = args["url"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'url'"))?;
        let max_chars = args["max_chars"].as_u64().unwrap_or(5000) as usize;

        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| tool_err(format!("fetch {url}: {e}")))?;

        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| tool_err(format!("read {url}: {e}")))?;

        let raw = String::from_utf8_lossy(&bytes[..bytes.len().min(self.max_bytes)]).to_string();

        let text = if content_type.contains("html") {
            strip_html_tags(&raw)
        } else {
            raw
        };

        let truncated = text.chars().count() > max_chars;
        let content: String = text.chars().take(max_chars).collect();

        Ok(json!({
            "url": url,
            "status": status,
            "content_type": content_type,
            "content": content,
            "truncated": truncated,
        }))
    }
}
