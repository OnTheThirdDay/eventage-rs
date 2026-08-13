use super::error::ObsError;
use super::exporter::ObservabilityExporter;
use crate::event::{kinds, meta_keys, Event};
use async_trait::async_trait;
use opentelemetry::{
    global,
    metrics::Counter,
    trace::{Span, SpanKind, TraceContextExt, Tracer},
    Context, KeyValue,
};
use std::collections::HashMap;
use tokio::sync::Mutex;

struct InFlight {
    /// Cycle contexts keyed by trace_id. Storing the `Context` (not the bare
    /// span) lets tool spans start as **children** of their cycle span, so
    /// backends render one connected trace per cycle.
    cycle_contexts: HashMap<String, Context>,
    tool_spans: HashMap<String, global::BoxedSpan>,
}

/// Translates Eventage events into OpenTelemetry spans and metrics.
///
/// - `agent.cycle.start` / `agent.cycle.end` open and close a cycle span.
/// - `tool.call.proposed` / `tool.result` open and close tool spans **as
///   children of the owning cycle span** (linked via the event's `trace_id`).
/// - `assistant.message` token usage is exported as counters:
///   `eventage.llm.input_tokens`, `eventage.llm.output_tokens`, and
///   `eventage.llm.cached_input_tokens`, attributed by `agent.id`.
///
/// Requires a pre-initialized OTel global tracer/meter provider.
pub struct OtelExporter {
    tracer_name: String,
    in_flight: Mutex<InFlight>,
    input_tokens: Counter<u64>,
    output_tokens: Counter<u64>,
    cached_input_tokens: Counter<u64>,
}

impl OtelExporter {
    /// Creates the exporter matching the provided service/tracer name.
    pub fn new(tracer_name: impl Into<String>) -> Self {
        let tracer_name = tracer_name.into();
        let meter = global::meter("eventage");
        Self {
            tracer_name,
            in_flight: Mutex::new(InFlight {
                cycle_contexts: HashMap::new(),
                tool_spans: HashMap::new(),
            }),
            input_tokens: meter.u64_counter("eventage.llm.input_tokens").build(),
            output_tokens: meter.u64_counter("eventage.llm.output_tokens").build(),
            cached_input_tokens: meter
                .u64_counter("eventage.llm.cached_input_tokens")
                .build(),
        }
    }

    fn meta_str<'a>(event: &'a Event, key: &str) -> &'a str {
        event
            .metadata
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    }
}

#[async_trait]
impl ObservabilityExporter for OtelExporter {
    async fn export(&self, event: &Event) -> Result<(), ObsError> {
        let tracer = global::tracer(self.tracer_name.clone());
        let mut state = self.in_flight.lock().await;

        match event.kind.as_str() {
            kinds::AGENT_CYCLE_START => {
                let trace_id = Self::meta_str(event, meta_keys::TRACE_ID).to_string();
                let agent_id = Self::meta_str(event, meta_keys::AGENT_ID).to_string();

                let span = tracer
                    .span_builder("agent.cycle")
                    .with_kind(SpanKind::Internal)
                    .with_attributes(vec![
                        KeyValue::new("agent.id", agent_id),
                        KeyValue::new("eventage.trace_id", trace_id.clone()),
                    ])
                    .start(&tracer);

                state
                    .cycle_contexts
                    .insert(trace_id, Context::current_with_span(span));
            }

            kinds::AGENT_CYCLE_END => {
                let trace_id = Self::meta_str(event, meta_keys::TRACE_ID);
                if let Some(cx) = state.cycle_contexts.remove(trace_id) {
                    let span = cx.span();
                    if let Some(elapsed) = event
                        .metadata
                        .get(meta_keys::ELAPSED_MS)
                        .and_then(|v| v.as_u64())
                    {
                        span.set_attribute(KeyValue::new("elapsed_ms", elapsed as i64));
                    }
                    span.end();
                }
            }

            kinds::ASSISTANT_MESSAGE => {
                let agent = KeyValue::new(
                    "agent.id",
                    Self::meta_str(event, meta_keys::AGENT_ID).to_string(),
                );
                let read = |key: &str| {
                    event
                        .metadata
                        .get(key)
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                };
                let (input, output, cached) = (
                    read(meta_keys::LLM_INPUT_TOKENS),
                    read(meta_keys::LLM_OUTPUT_TOKENS),
                    read(meta_keys::LLM_CACHED_INPUT_TOKENS),
                );
                let attrs = std::slice::from_ref(&agent);
                if input > 0 {
                    self.input_tokens.add(input, attrs);
                }
                if output > 0 {
                    self.output_tokens.add(output, attrs);
                }
                if cached > 0 {
                    self.cached_input_tokens.add(cached, attrs);
                }
            }

            kinds::TOOL_CALL_PROPOSED => {
                let tool_call_id = event
                    .payload
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let name = event
                    .payload
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let trace_id = Self::meta_str(event, meta_keys::TRACE_ID);

                let builder = tracer
                    .span_builder(format!("tool.{}", name))
                    .with_kind(SpanKind::Internal)
                    .with_attributes(vec![KeyValue::new("tool.call_id", tool_call_id.clone())]);

                // Child of the owning cycle span when we have one.
                let span = match state.cycle_contexts.get(trace_id) {
                    Some(parent_cx) => builder.start_with_context(&tracer, parent_cx),
                    None => builder.start(&tracer),
                };

                state.tool_spans.insert(tool_call_id, span);
            }

            kinds::TOOL_RESULT => {
                let tool_call_id = event
                    .payload
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                if let Some(mut span) = state.tool_spans.remove(tool_call_id) {
                    let is_error = event.payload.get("error").is_some();
                    if is_error {
                        span.set_status(opentelemetry::trace::Status::error("tool returned error"));
                    }
                    span.end();
                }
            }

            _ => {}
        }

        Ok(())
    }

    async fn flush(&self) -> Result<(), ObsError> {
        let mut state = self.in_flight.lock().await;
        for (_, cx) in state.cycle_contexts.drain() {
            cx.span().end();
        }
        for (_, mut span) in state.tool_spans.drain() {
            span.end();
        }
        Ok(())
    }
}
