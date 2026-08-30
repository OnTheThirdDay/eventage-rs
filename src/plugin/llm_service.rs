//! Lets a plugin use the agent's own model instead of bringing its own key.
//!
//! A plugin is a separate process, so without this it has no way to reach an
//! LLM except by holding credentials of its own. That is worse than it sounds:
//! it would answer with a different model than the operator configured, run
//! its own uncoordinated backoff and pacing rather than sharing
//! [`RetryProvider`](crate::llm::RetryProvider) and
//! [`RateLimitedProvider`](crate::llm::RateLimitedProvider), and — because
//! [`TokenBudgetHook`](crate::agent::TokenBudgetHook) computes spend by summing
//! token metadata off the event log — spend money no budget could see.
//!
//! So the completion is a request and a response on the bus, the same shape
//! MCP elicitation already uses. The plugin emits [`kinds::LLM_REQUEST`]; this
//! service answers with [`kinds::LLM_RESPONSE`], stamped with the usage the
//! provider reported so the budget counts it like any other call.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use eventage::plugin::llm_service::PluginLlmService;
//! # async fn example(
//! #     host: &eventage::ComponentHost,
//! #     llm: Arc<dyn eventage::llm::LlmProvider>,
//! # ) -> Result<(), Box<dyn std::error::Error>> {
//! host.load(Arc::new(PluginLlmService::new(llm))).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Who may ask
//!
//! Only events carrying the `origin_plugin` stamp the observer bridge applies,
//! and only from a plugin granted the `llm` capability. A request without that
//! stamp did not come through a bridge, so it is ignored rather than served —
//! otherwise anything able to publish could spend the operator's tokens.

use crate::agent::worker::{EventWorker, WorkerError};
use crate::bus::EventBus;
use crate::component::{Component, ComponentContext, ComponentError};
use crate::event::{kinds, meta_keys, Event};
use crate::llm::{ChatMessage, LlmProvider, Role};
use crate::plugin::observer::ORIGIN_PLUGIN_KEY;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, warn};

/// Ceiling on how many messages one request may carry.
///
/// A plugin builds its prompt from event payloads, and a runaway loop that
/// appended history every pass would otherwise turn into an expensive request
/// rather than an obvious error.
const MAX_MESSAGES: usize = 64;

/// Answers [`kinds::LLM_REQUEST`] from plugins using the session's provider.
pub struct PluginLlmService {
    llm: Arc<dyn LlmProvider>,
    name: String,
}

impl PluginLlmService {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            llm,
            name: "plugin-llm".to_string(),
        }
    }
}

#[async_trait]
impl Component for PluginLlmService {
    fn name(&self) -> &str {
        &self.name
    }

    async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
        ctx.worker(Responder {
            llm: Arc::clone(&self.llm),
        });
        Ok(())
    }
}

struct Responder {
    llm: Arc<dyn LlmProvider>,
}

impl Responder {
    /// Publish a failure the plugin can act on, rather than leaving it to time
    /// out with nothing to report.
    async fn fail(bus: &EventBus, request_id: &str, plugin: &str, message: String) {
        warn!(plugin, "plugin LLM request failed: {message}");
        let _ = bus
            .publish(Event::new(
                kinds::LLM_RESPONSE,
                json!({ "request_id": request_id, "plugin": plugin, "error": message }),
            ))
            .await;
    }
}

#[async_trait]
impl EventWorker for Responder {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::LLM_REQUEST.to_string()]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        // The stamp is applied by the bridge and cannot be forged from the
        // wire, so its absence means this did not come from a plugin.
        let Some(plugin) = event
            .metadata
            .get(ORIGIN_PLUGIN_KEY)
            .and_then(|v| v.as_str())
        else {
            debug!("ignoring an llm.request with no plugin origin");
            return Ok(());
        };

        let Some(request_id) = event.payload.get("request_id").and_then(|v| v.as_str()) else {
            Self::fail(bus, "", plugin, "request has no `request_id`".into()).await;
            return Ok(());
        };

        let messages = match parse_messages(&event.payload) {
            Ok(m) => m,
            Err(e) => {
                Self::fail(bus, request_id, plugin, e).await;
                return Ok(());
            }
        };

        // A schema means the plugin wants a shape it can rely on rather than
        // prose it has to parse — which is what makes an answer presentable as
        // a list of choices instead of a wall of text.
        let schema = event.payload.get("schema").cloned();
        let schema_name = event
            .payload
            .get("schema_name")
            .and_then(|v| v.as_str())
            .unwrap_or("result");

        let (body, usage) = match schema {
            Some(schema) => match self
                .llm
                .complete_structured(messages, schema_name, schema)
                .await
            {
                Ok(value) => (json!({ "structured": value }), None),
                Err(e) => {
                    Self::fail(bus, request_id, plugin, e.to_string()).await;
                    return Ok(());
                }
            },
            None => match self.llm.complete(messages, vec![]).await {
                Ok(response) => (
                    json!({ "content": response.content }),
                    Some((response.input_tokens, response.output_tokens)),
                ),
                Err(e) => {
                    Self::fail(bus, request_id, plugin, e.to_string()).await;
                    return Ok(());
                }
            },
        };

        let mut response = Event::new(
            kinds::LLM_RESPONSE,
            json!({
                "request_id": request_id,
                "plugin": plugin,
                "content": body.get("content").cloned().unwrap_or(serde_json::Value::Null),
                "structured": body.get("structured").cloned().unwrap_or(serde_json::Value::Null),
            }),
        );

        // Recorded the same way the harness records its own calls, so a
        // plugin's spend counts against the session budget rather than being
        // invisible to it.
        if let Some((input, output)) = usage {
            if let Some(input) = input {
                response = response.with_meta(meta_keys::LLM_INPUT_TOKENS, json!(input));
            }
            if let Some(output) = output {
                response = response.with_meta(meta_keys::LLM_OUTPUT_TOKENS, json!(output));
            }
        }

        bus.publish(response).await.map_err(WorkerError::Bus)
    }
}

fn parse_messages(payload: &serde_json::Value) -> Result<Vec<ChatMessage>, String> {
    let raw = payload
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "request has no `messages` array".to_string())?;

    if raw.is_empty() {
        return Err("request has no messages".into());
    }
    if raw.len() > MAX_MESSAGES {
        return Err(format!(
            "request carries {} messages; the limit is {MAX_MESSAGES}",
            raw.len()
        ));
    }

    raw.iter()
        .map(|m| {
            let content = m
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "a message has no `content` string".to_string())?;
            Ok(match m.get("role").and_then(|v| v.as_str()) {
                Some("system") => ChatMessage::system(content),
                Some("assistant") => ChatMessage {
                    role: Role::Assistant,
                    content: Some(content.to_string()),
                    ..ChatMessage::user("")
                },
                // Anything else is a user turn: a plugin has no business
                // synthesising tool results, and silently accepting one would
                // let it forge a tool exchange inside its own prompt.
                _ => ChatMessage::user(content),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hook::DynamicHookChain;
    use crate::agent::tool::ToolRegistry;
    use crate::component::ComponentHost;
    use crate::llm::mock::MockLlmProvider;
    use std::time::Duration;

    async fn wait_for(bus: &EventBus, kind: &str) -> Option<Event> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(e) = bus.log().await.into_iter().find(|e| e.kind == kind) {
                return Some(e);
            }
            if tokio::time::Instant::now() > deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn service(bus: &EventBus) -> ComponentHost {
        let host = ComponentHost::new(bus.clone(), ToolRegistry::new(), DynamicHookChain::new());
        let llm = Arc::new(MockLlmProvider::with_texts(["an answer"]));
        host.load(Arc::new(PluginLlmService::new(llm)))
            .await
            .unwrap();
        host
    }

    #[tokio::test]
    async fn a_plugin_request_is_answered_with_the_sessions_model() {
        let bus = EventBus::new();
        let _host = service(&bus).await;

        bus.publish(
            Event::new(
                kinds::LLM_REQUEST,
                json!({ "request_id": "r1", "messages": [{ "role": "user", "content": "hi" }] }),
            )
            .with_meta(ORIGIN_PLUGIN_KEY, json!("medic::medic")),
        )
        .await
        .unwrap();

        let reply = wait_for(&bus, kinds::LLM_RESPONSE).await.unwrap();
        assert_eq!(reply.payload["request_id"], "r1");
        assert_eq!(reply.payload["plugin"], "medic::medic");
        assert_eq!(reply.payload["content"], "an answer");
    }

    /// Otherwise anything able to publish could spend the operator's tokens.
    #[tokio::test]
    async fn a_request_without_a_plugin_stamp_is_ignored() {
        let bus = EventBus::new();
        let _host = service(&bus).await;

        bus.publish(Event::new(
            kinds::LLM_REQUEST,
            json!({ "request_id": "r1", "messages": [{ "role": "user", "content": "hi" }] }),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !bus.log()
                .await
                .iter()
                .any(|e| e.kind == kinds::LLM_RESPONSE),
            "an unstamped request must not be served"
        );
    }

    /// A plugin has to be able to tell "it failed" from "it is still thinking".
    #[tokio::test]
    async fn a_malformed_request_gets_an_error_rather_than_silence() {
        let bus = EventBus::new();
        let _host = service(&bus).await;

        bus.publish(
            Event::new(kinds::LLM_REQUEST, json!({ "request_id": "r1" }))
                .with_meta(ORIGIN_PLUGIN_KEY, json!("medic::medic")),
        )
        .await
        .unwrap();

        let reply = wait_for(&bus, kinds::LLM_RESPONSE).await.unwrap();
        assert!(reply.payload["error"]
            .as_str()
            .unwrap()
            .contains("messages"));
    }
}
