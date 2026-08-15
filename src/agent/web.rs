//! Web access: `web_search` and `web_fetch`.
//!
//! Both live here rather than in one application because every agent that
//! reads documentation wants them and two copies would drift.
//!
//! # Reaching the network is a privilege
//!
//! These are the only built-in tools that talk to a host the user did not
//! name, which makes them the obvious route both for pulling something
//! hostile in and for pushing something private out. Two defences are built
//! in rather than left to the caller:
//!
//! - Only `http`/`https`. A `file://` URL would turn a fetch into an
//!   arbitrary read, and `gopher://` and friends into a request smuggler.
//! - No private or loopback addresses. On a developer's machine those are
//!   the cloud metadata endpoint, the container network, and whatever is
//!   listening on localhost — including this agent's own control surface.
//!
//! Register them behind a permission policy as well: the checks below stop
//! an accident, not a determined prompt injection.

use crate::agent::error::AgentError;
use crate::agent::tool::Tool;
use crate::llm::types::ToolDefinition;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

fn tool_err(msg: impl Into<String>) -> AgentError {
    AgentError::Tool(msg.into())
}

/// Reject a URL that a fetch tool has no business following.
///
/// The failure this prevents is not exotic. A page the agent was asked to
/// read says "now fetch http://169.254.169.254/latest/meta-data/", or
/// `file:///etc/passwd`, or `http://localhost:4600/api/...` — and a tool that
/// simply does what the URL says will oblige. The model is not the last line
/// of defence here; this is.
///
/// Returns the reason for refusal, phrased for the model so it can adapt
/// rather than retry.
pub fn refuse_url(url: &str) -> Option<String> {
    let parsed = match reqwest::Url::parse(url) {
        Ok(parsed) => parsed,
        Err(e) => return Some(format!("that is not a valid URL: {e}")),
    };

    if !matches!(parsed.scheme(), "http" | "https") {
        return Some(format!(
            "only http and https are allowed; '{}' is not a web request",
            parsed.scheme()
        ));
    }

    let Some(host) = parsed.host_str() else {
        return Some("that URL has no host".into());
    };

    // Literal addresses are checked directly. Names are not resolved here:
    // doing so would still race the request's own lookup, and the obvious
    // local names are worth refusing on sight.
    let lowered = host.to_ascii_lowercase();
    if lowered == "localhost" || lowered.ends_with(".localhost") || lowered.ends_with(".internal") {
        return Some(format!(
            "'{host}' is a local address, which is not fetchable"
        ));
    }

    if let Ok(ip) = lowered.trim_matches(['[', ']']).parse::<std::net::IpAddr>() {
        let blocked = match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback()
                    || v4.is_private()
                    || v4.is_link_local()
                    || v4.is_broadcast()
                    || v4.is_unspecified()
                    || v4.octets()[0] == 0
            }
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback() || v6.is_unspecified() || (v6.segments()[0] & 0xfe00) == 0xfc00
            }
        };
        if blocked {
            return Some(format!(
                "{ip} is a private or loopback address; web tools reach the public internet only"
            ));
        }
    }
    None
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
        if !in_tag
            && !in_script
            && i + 7 <= lower_chars.len()
            && lower_chars[i..i + 7] == ['<', 's', 'c', 'r', 'i', 'p', 't'][..]
        {
            in_script = true;
            in_tag = true;
        } else if in_script
            && i + 9 <= lower_chars.len()
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
        if let Some(reason) = refuse_url(url) {
            return Err(tool_err(reason));
        }
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

#[cfg(test)]
mod guard_tests {
    use super::refuse_url;

    #[test]
    fn public_urls_are_allowed() {
        for url in [
            "https://docs.rs/serde/latest/serde/",
            "http://example.com/page?q=1",
            "https://8.8.8.8/",
        ] {
            assert!(refuse_url(url).is_none(), "{url} should be fetchable");
        }
    }

    #[test]
    fn non_web_schemes_cannot_be_smuggled_through_a_fetch_tool() {
        // `file://` would turn a web fetch into an arbitrary file read.
        for url in ["file:///etc/passwd", "gopher://x/", "ftp://host/f"] {
            assert!(refuse_url(url).is_some(), "{url} should be refused");
        }
    }

    #[test]
    fn the_cloud_metadata_endpoint_is_refused() {
        // The single most valuable target on a machine that has credentials.
        assert!(refuse_url("http://169.254.169.254/latest/meta-data/").is_some());
    }

    #[test]
    fn loopback_and_private_ranges_are_refused() {
        for url in [
            "http://127.0.0.1:4600/api/sessions",
            "http://localhost:8080/",
            "http://10.0.0.5/",
            "http://192.168.1.1/admin",
            "http://172.16.4.4/",
            "http://[::1]:9222/json",
            "http://0.0.0.0/",
        ] {
            assert!(refuse_url(url).is_some(), "{url} should be refused");
        }
    }

    #[test]
    fn the_refusal_says_why_so_the_model_can_adapt() {
        let reason = refuse_url("file:///etc/passwd").unwrap();
        assert!(reason.contains("http"), "unhelpful refusal: {reason}");
    }

    #[test]
    fn malformed_input_is_refused_not_panicked_on() {
        assert!(refuse_url("").is_some());
        assert!(refuse_url("not a url").is_some());
    }
}
