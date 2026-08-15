//! Looking at images in the workspace.
//!
//! The model layer has carried images since multimodal support landed, and
//! the ACP bridge converts them when a user attaches one — but nothing on the
//! agent's side could ever *fetch* one. So a mockup, a failing-test
//! screenshot or an architecture diagram sitting in the repository was
//! invisible: `read_file` would either refuse it as binary or hand back bytes
//! as text.
//!
//! This closes that. The tool returns the image under the `_image` key, and
//! the context assembler turns it into a real image part in the next message,
//! so the model sees the picture rather than a base64 string.

use super::arg_str;
use crate::workspace::Workspace;
use async_trait::async_trait;
use base64::Engine;
use eventage::agent::error::AgentError;
use eventage::agent::tool::Tool;
use eventage::llm::types::ToolDefinition;
use serde_json::{json, Value};
use std::sync::Arc;

fn tool_err(msg: impl Into<String>) -> AgentError {
    AgentError::Tool(msg.into())
}

/// Beyond this an image is more likely to blow the context budget than to
/// tell the model anything, and providers reject very large payloads anyway.
const MAX_IMAGE_BYTES: u64 = 4 * 1024 * 1024;

pub struct ViewImage {
    pub ws: Arc<Workspace>,
}

/// The media type for a path, or `None` if it is not an image we can send.
///
/// Sniffing the extension rather than the bytes: providers want a declared
/// media type, and a file named `.png` that is secretly a JPEG is a problem
/// the model can report far better than this tool can guess at.
fn media_type_of(path: &str) -> Option<&'static str> {
    let ext = path.rsplit('.').next()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

#[async_trait]
impl Tool for ViewImage {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "view_image",
            "Look at an image file in the workspace — a screenshot, mockup, \
             diagram or chart. Use it whenever the answer depends on what \
             something looks like rather than on what a file says. Supports \
             png, jpeg, gif and webp.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the image, relative to the workspace."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = arg_str(&args, "path")?;
        let abs = self
            .ws
            .resolve(&path)
            .map_err(|e| tool_err(e.to_string()))?;

        let Some(media_type) = media_type_of(&path) else {
            return Err(tool_err(format!(
                "'{path}' is not an image this tool can read (png, jpeg, gif, webp). \
                 Use read_file for text."
            )));
        };

        // Through the workspace handle, so a symlink pointing out of the
        // repository cannot be read as an "image".
        let meta = self
            .ws
            .metadata(&path)
            .await
            .map_err(|e| tool_err(format!("{e:#}")))?;
        if meta.len() > MAX_IMAGE_BYTES {
            return Err(tool_err(format!(
                "'{path}' is {:.1} MB, over the {} MB limit for images",
                meta.len() as f64 / 1_048_576.0,
                MAX_IMAGE_BYTES / 1_048_576
            )));
        }

        let bytes = self
            .ws
            .read(&path)
            .await
            .map_err(|e| tool_err(format!("{e:#}")))?;
        let data = base64::engine::general_purpose::STANDARD.encode(&bytes);

        Ok(json!({
            "path": path,
            "bytes": meta.len(),
            "media_type": media_type,
            // The assembler turns this into an image part in the next
            // message; nothing else needs to understand it.
            "_image": { "media_type": media_type, "data": data },
            "_locations": [{ "path": abs.display().to_string() }],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> (tempfile::TempDir, Arc<Workspace>) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path().to_str().unwrap()).unwrap());
        (dir, ws)
    }

    /// The smallest valid PNG: an 8-byte signature is enough for a tool that
    /// does not decode.
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n....";

    #[tokio::test]
    async fn an_image_comes_back_ready_for_the_model() {
        let (dir, ws) = workspace();
        std::fs::write(dir.path().join("mockup.png"), PNG).unwrap();

        let out = ViewImage { ws }
            .execute(json!({ "path": "mockup.png" }))
            .await
            .unwrap();

        assert_eq!(out["_image"]["media_type"], "image/png");
        assert!(!out["_image"]["data"].as_str().unwrap().is_empty());
        assert_eq!(out["bytes"], PNG.len());
    }

    #[tokio::test]
    async fn a_text_file_is_refused_with_the_tool_to_use_instead() {
        let (dir, ws) = workspace();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

        let err = ViewImage { ws }
            .execute(json!({ "path": "main.rs" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("read_file"), "unhelpful: {err}");
    }

    #[tokio::test]
    async fn an_oversized_image_is_refused_before_it_is_read() {
        let (dir, ws) = workspace();
        let big = vec![0u8; (MAX_IMAGE_BYTES + 1) as usize];
        std::fs::write(dir.path().join("huge.png"), big).unwrap();

        let err = ViewImage { ws }
            .execute(json!({ "path": "huge.png" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("MB"), "{err}");
    }

    #[tokio::test]
    async fn it_cannot_read_outside_the_workspace() {
        let (_dir, ws) = workspace();
        assert!(ViewImage { ws }
            .execute(json!({ "path": "../../../etc/hosts.png" }))
            .await
            .is_err());
    }

    #[test]
    fn media_types_cover_what_providers_accept() {
        for (path, expected) in [
            ("a.png", Some("image/png")),
            ("a.JPG", Some("image/jpeg")),
            ("a.jpeg", Some("image/jpeg")),
            ("a.webp", Some("image/webp")),
            ("a.svg", None),
            ("a.rs", None),
            ("noextension", None),
        ] {
            assert_eq!(media_type_of(path), expected, "{path}");
        }
    }
}
