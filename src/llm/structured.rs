//! Structured output: constrain a completion to a JSON Schema and
//! deserialize it into a Rust type.
//!
//! [`super::LlmProvider::complete_structured`]
//! is object-safe and returns a [`Value`]; the [`StructuredExt::complete_as`]
//! extension adds the typed sugar:
//!
//! ```no_run
//! # use eventage::llm::{ChatMessage, LlmProvider, StructuredExt};
//! # use serde::Deserialize;
//! #[derive(Deserialize)]
//! struct Verdict { approved: bool, reason: String }
//!
//! # async fn example(llm: &dyn LlmProvider) -> Result<(), Box<dyn std::error::Error>> {
//! let verdict: Verdict = llm
//!     .complete_as(
//!         vec![ChatMessage::user("Should we ship? Answer strictly.")],
//!         "verdict",
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "approved": { "type": "boolean" },
//!                 "reason": { "type": "string" }
//!             },
//!             "required": ["approved", "reason"],
//!             "additionalProperties": false
//!         }),
//!     )
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! Providers with native support (Chat Completions `response_format`,
//! Anthropic forced tool use, Responses `text.format`) constrain decoding
//! server-side. Others fall back to a prompted-JSON strategy, so
//! `complete_as` works against every provider — including local models.

use super::error::LlmError;
use super::provider::LlmProvider;
use super::types::ChatMessage;
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Extract a JSON value from model text that may be wrapped in prose or a
/// fenced code block.
pub fn extract_json(text: &str) -> Option<Value> {
    // Direct parse first.
    if let Ok(v) = serde_json::from_str::<Value>(text.trim()) {
        return Some(v);
    }
    // Strip a ```json … ``` fence if present.
    let unfenced = text
        .split_once("```")
        .map(|(_, rest)| rest.trim_start_matches("json").trim_start_matches('\n'))
        .and_then(|rest| rest.split_once("```").map(|(inner, _)| inner))
        .unwrap_or(text);
    if let Ok(v) = serde_json::from_str::<Value>(unfenced.trim()) {
        return Some(v);
    }
    // Last resort: the outermost {...} or [...] span.
    let start = unfenced.find(['{', '['])?;
    let end = unfenced.rfind(['}', ']'])?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&unfenced[start..=end]).ok()
}

/// Build the prompt used by the provider-agnostic fallback path.
pub(crate) fn json_instruction(schema_name: &str, schema: &Value) -> ChatMessage {
    ChatMessage::system(format!(
        "You must reply with a single JSON value named '{schema_name}' that validates \
         against this JSON Schema:\n\n{schema}\n\n\
         Output ONLY the JSON — no prose, no code fences, no explanation.",
    ))
}

/// Typed structured-output sugar over
/// [`complete_structured`](LlmProvider::complete_structured).
///
/// Implemented for every provider (including `dyn LlmProvider`), so it works
/// on `Arc<dyn LlmProvider>` handles.
#[async_trait]
pub trait StructuredExt {
    /// Complete with output constrained to `schema`, deserialized into `T`.
    ///
    /// The response is validated against `schema` before deserialization, so
    /// a provider that ignores the constraint produces a clear error rather
    /// than a confusing serde failure.
    async fn complete_as<T: DeserializeOwned>(
        &self,
        messages: Vec<ChatMessage>,
        schema_name: &str,
        schema: Value,
    ) -> Result<T, LlmError>;
}

#[async_trait]
impl<P: LlmProvider + ?Sized> StructuredExt for P {
    async fn complete_as<T: DeserializeOwned>(
        &self,
        messages: Vec<ChatMessage>,
        schema_name: &str,
        schema: Value,
    ) -> Result<T, LlmError> {
        let value = self
            .complete_structured(messages, schema_name, schema.clone())
            .await?;
        if let Err(violation) = crate::schema::validate_args(&schema, &value) {
            return Err(LlmError::Structured(format!(
                "response did not match schema '{schema_name}': {violation}"
            )));
        }
        serde_json::from_value(value).map_err(LlmError::Serde)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[test]
    fn extracts_bare_fenced_and_embedded_json() {
        assert_eq!(extract_json(r#"{"a":1}"#).unwrap()["a"], 1);
        assert_eq!(extract_json("```json\n{\"a\": 2}\n```").unwrap()["a"], 2);
        assert_eq!(
            extract_json("Sure! Here you go:\n{\"a\": 3}\nHope that helps.").unwrap()["a"],
            3
        );
        assert!(extract_json("no json here").is_none());
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Verdict {
        approved: bool,
    }

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": { "approved": { "type": "boolean" } },
            "required": ["approved"]
        })
    }

    /// A provider with no native structured support — exercises the fallback.
    struct PromptedProvider {
        reply: String,
    }

    #[async_trait]
    impl LlmProvider for PromptedProvider {
        async fn complete(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Vec<super::super::types::ToolDefinition>,
        ) -> Result<super::super::types::LlmResponse, LlmError> {
            // The fallback must have injected schema instructions.
            assert!(messages.iter().any(|m| m
                .content
                .as_deref()
                .unwrap_or("")
                .contains("JSON Schema")));
            Ok(super::super::types::LlmResponse {
                content: Some(self.reply.clone()),
                finish_reason: "stop".into(),
                ..Default::default()
            })
        }
        fn model(&self) -> &str {
            "prompted"
        }
    }

    #[tokio::test]
    async fn fallback_parses_typed_result() {
        let provider = PromptedProvider {
            reply: "```json\n{\"approved\": true}\n```".into(),
        };
        let verdict: Verdict = provider
            .complete_as(vec![ChatMessage::user("ship?")], "verdict", schema())
            .await
            .unwrap();
        assert_eq!(verdict, Verdict { approved: true });
    }

    #[tokio::test]
    async fn schema_violation_is_reported_clearly() {
        let provider = PromptedProvider {
            reply: r#"{"approved": "yes"}"#.into(),
        };
        let err = provider
            .complete_as::<Verdict>(vec![ChatMessage::user("ship?")], "verdict", schema())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("did not match schema"), "{msg}");
        assert!(msg.contains("approved"), "{msg}");
    }

    #[tokio::test]
    async fn unparseable_output_errors() {
        let provider = PromptedProvider {
            reply: "I cannot do that".into(),
        };
        let err = provider
            .complete_as::<Verdict>(vec![ChatMessage::user("ship?")], "verdict", schema())
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::Structured(_)));
    }
}
