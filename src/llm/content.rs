//! Multimodal message content.
//!
//! A [`ChatMessage`](super::ChatMessage) carries either plain text or an
//! ordered list of [`ContentPart`]s (text interleaved with images). The parts
//! are provider-neutral; each provider maps them to its own wire format:
//!
//! | Part | Chat Completions | Anthropic | Responses |
//! |---|---|---|---|
//! | [`ContentPart::Text`] | `{"type":"text"}` | `{"type":"text"}` | `{"type":"input_text"}` |
//! | [`ContentPart::Image`] | `{"type":"image_url"}` | `{"type":"image"}` | `{"type":"input_image"}` |
//!
//! Images come from a URL or inline base64 data:
//!
//! ```
//! use eventage::llm::{ChatMessage, ContentPart, ImageSource};
//!
//! let msg = ChatMessage::user_with_parts(vec![
//!     ContentPart::text("What is in this screenshot?"),
//!     ContentPart::image_url("https://example.com/shot.png"),
//! ]);
//! assert!(msg.is_multimodal());
//! ```

use serde::{Deserialize, Serialize};

/// Where an image comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// A remote (or `data:`) URL the provider fetches itself.
    Url { url: String },
    /// Inline base64-encoded bytes with an explicit media type
    /// (e.g. `"image/png"`).
    Base64 { media_type: String, data: String },
}

impl ImageSource {
    /// Render as a `data:` URL (or pass a plain URL through) — the form
    /// OpenAI-style APIs expect.
    pub fn to_data_url(&self) -> String {
        match self {
            ImageSource::Url { url } => url.clone(),
            ImageSource::Base64 { media_type, data } => {
                format!("data:{media_type};base64,{data}")
            }
        }
    }
}

/// One piece of message content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { source: ImageSource },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    /// An image the provider fetches from `url`.
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::Image {
            source: ImageSource::Url { url: url.into() },
        }
    }

    /// An inline base64 image, e.g. `image_base64("image/png", b64)`.
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        ContentPart::Image {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }

    /// The text of this part, if it is a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentPart::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Chat Completions wire form.
    pub fn to_openai_json(&self) -> serde_json::Value {
        match self {
            ContentPart::Text { text } => serde_json::json!({ "type": "text", "text": text }),
            ContentPart::Image { source } => serde_json::json!({
                "type": "image_url",
                "image_url": { "url": source.to_data_url() }
            }),
        }
    }

    /// Anthropic Messages content-block form.
    pub fn to_anthropic_json(&self) -> serde_json::Value {
        match self {
            ContentPart::Text { text } => serde_json::json!({ "type": "text", "text": text }),
            ContentPart::Image { source } => {
                let source_json = match source {
                    ImageSource::Url { url } => {
                        serde_json::json!({ "type": "url", "url": url })
                    }
                    ImageSource::Base64 { media_type, data } => serde_json::json!({
                        "type": "base64",
                        "media_type": media_type,
                        "data": data
                    }),
                };
                serde_json::json!({ "type": "image", "source": source_json })
            }
        }
    }

    /// Responses API input-content form.
    pub fn to_responses_json(&self) -> serde_json::Value {
        match self {
            ContentPart::Text { text } => {
                serde_json::json!({ "type": "input_text", "text": text })
            }
            ContentPart::Image { source } => serde_json::json!({
                "type": "input_image",
                "image_url": source.to_data_url()
            }),
        }
    }
}

/// Concatenate the text parts, ignoring images — used for logging,
/// summarization prompts, and any text-only view of a message.
pub fn parts_to_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| p.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_renders_as_data_url() {
        let part = ContentPart::image_base64("image/png", "AAAA");
        let openai = part.to_openai_json();
        assert_eq!(openai["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn anthropic_distinguishes_url_and_base64() {
        let url = ContentPart::image_url("https://x/y.png").to_anthropic_json();
        assert_eq!(url["source"]["type"], "url");
        assert_eq!(url["source"]["url"], "https://x/y.png");

        let b64 = ContentPart::image_base64("image/jpeg", "ZZZ").to_anthropic_json();
        assert_eq!(b64["source"]["type"], "base64");
        assert_eq!(b64["source"]["media_type"], "image/jpeg");
        assert_eq!(b64["source"]["data"], "ZZZ");
    }

    #[test]
    fn responses_uses_input_prefixed_types() {
        assert_eq!(
            ContentPart::text("hi").to_responses_json()["type"],
            "input_text"
        );
        assert_eq!(
            ContentPart::image_url("u").to_responses_json()["type"],
            "input_image"
        );
    }

    #[test]
    fn parts_round_trip_through_json() {
        let parts = vec![
            ContentPart::text("look:"),
            ContentPart::image_base64("image/png", "QQ"),
        ];
        let json = serde_json::to_string(&parts).unwrap();
        let back: Vec<ContentPart> = serde_json::from_str(&json).unwrap();
        assert_eq!(parts, back);
        assert_eq!(parts_to_text(&parts), "look:");
    }
}
