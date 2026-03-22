//! Web tools: web_search (DuckDuckGo) and web_fetch.

use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

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
                .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
                .timeout(Duration::from_secs(15))
                .connect_timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    /// POST to html.duckduckgo.com — more reliable than GET for avoiding CAPTCHAs.
    async fn search_html(&self, query: &str, max: usize) -> Vec<Value> {
        let body = format!("q={}&kd=-1&kp=-2&kl=us-en", urlencoding(query));
        let Ok(resp) = self
            .client
            .post("https://html.duckduckgo.com/html/")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
            .header("Accept-Language", "en-US,en;q=0.9")
            .body(body)
            .send()
            .await
        else {
            return vec![];
        };
        let Ok(html) = resp.text().await else {
            return vec![];
        };
        parse_ddg_results(&html, max)
    }

    /// POST to lite.duckduckgo.com — simpler HTML, less bot-detection.
    async fn search_lite(&self, query: &str, max: usize) -> Vec<Value> {
        let body = format!("q={}", urlencoding(query));
        let Ok(resp) = self
            .client
            .post("https://lite.duckduckgo.com/lite/")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "text/html")
            .body(body)
            .send()
            .await
        else {
            return vec![];
        };
        let Ok(html) = resp.text().await else {
            return vec![];
        };
        parse_ddg_lite_results(&html, max)
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

        // Try the full HTML endpoint first; fall back to the lite endpoint if it
        // returns nothing (CAPTCHA, structure change, rate-limit, etc.).
        let mut results = self.search_html(query, max_results).await;
        if results.is_empty() {
            results = self.search_lite(query, max_results).await;
        }

        if results.is_empty() {
            return Err(tool_err(
                "web_search returned no results — the search engine may be rate-limiting or the query has no matches. Try rephrasing.",
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

/// Very light HTML parser for DuckDuckGo HTML-endpoint results.
///
/// Searches for `result__a` as a substring so it tolerates extra CSS classes
/// like `class="result__a result__a--overflow"`.
fn parse_ddg_results(html: &str, max: usize) -> Vec<Value> {
    let mut results = vec![];

    let mut pos = 0;
    while results.len() < max {
        // Match any attribute value containing result__a (substring, not exact).
        let Some(title_start) = html[pos..].find("result__a") else {
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

/// Parser for lite.duckduckgo.com results.
///
/// The lite endpoint uses a table layout where each result has a `result-link`
/// anchor (title + URL) followed shortly by a `result-snippet` cell.
fn parse_ddg_lite_results(html: &str, max: usize) -> Vec<Value> {
    let mut results = vec![];
    let mut pos = 0;

    while results.len() < max {
        let Some(rel) = html[pos..].find("result-link") else {
            break;
        };
        let abs = pos + rel;

        // Find the <a href="..."> after the class attribute.
        let region_end = (abs + 600).min(html.len());
        let region = &html[abs..region_end];

        let url = if let Some(href_pos) = region.find("href=\"") {
            let start = href_pos + 6;
            let end = region[start..]
                .find('"')
                .map(|e| start + e)
                .unwrap_or(region.len());
            let raw = &region[start..end];
            if raw.contains("uddg=") || raw.starts_with("/l/") {
                extract_ddg_url(raw)
            } else {
                raw.to_string()
            }
        } else {
            String::new()
        };

        let title = if let Some(gt) = region.find('>') {
            let cs = gt + 1;
            if let Some(end) = region[cs..].find("</a>") {
                strip_html_tags(&region[cs..cs + end])
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        pos = abs + 1;

        // Snippet is in the next result-snippet cell.
        let snippet = if pos < html.len() {
            if let Some(snip_rel) = html[abs..].find("result-snippet") {
                let s = abs + snip_rel + "result-snippet".len();
                let lookahead = (s + 60).min(html.len());
                if let Some(gt) = html[s..lookahead].find('>') {
                    let cs = s + gt + 1;
                    if let Some(end) = html[cs..].find("</") {
                        strip_html_tags(&html[cs..cs + end.min(400)])
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
    let chars: Vec<char> = html.chars().collect();
    // Use char-indexed lowercased chars for substring comparisons so multi-byte
    // Unicode characters never cause a byte-boundary panic.
    let lower_chars: Vec<char> = html.to_lowercase().chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if !in_tag && !in_script && i + 7 <= lower_chars.len()
            && lower_chars[i..i + 7] == ['<', 's', 'c', 'r', 'i', 'p', 't'][..]
        {
            in_script = true;
            in_tag = true;
        } else if in_script && i + 9 <= lower_chars.len()
            && lower_chars[i..i + 9] == ['<', '/', 's', 'c', 'r', 'i', 'p', 't', '>'][..]
        {
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
