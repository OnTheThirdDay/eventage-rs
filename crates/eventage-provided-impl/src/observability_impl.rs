use async_trait::async_trait;
use eventage_observability::{ObsError, ObservabilityExporter};
use eventage_core::Event;
use std::path::Path;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

/// File-based exporter appending JSON Lines (JSONL) for replay and analysis.
///
/// Flushes on every write for durability. Safe to tail concurrently (`tail -f`).
///
/// # Example
/// ```no_run
/// # use eventage_provided_impl::JsonlExporter;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let exporter = JsonlExporter::new("events.jsonl").await?;
/// # Ok(())
/// # }
/// ```
pub struct JsonlExporter {
    file: Mutex<tokio::fs::File>,
}

impl JsonlExporter {
    /// Opens or creates the JSONL file at `path` for appending.
    pub async fn new(path: impl AsRef<Path>) -> Result<Self, ObsError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

#[async_trait]
impl ObservabilityExporter for JsonlExporter {
    async fn export(&self, event: &Event) -> Result<(), ObsError> {
        let line = serde_json::to_string(event)?;
        let mut f = self.file.lock().await;
        f.write_all(line.as_bytes()).await?;
        f.write_all(b"\n").await?;
        f.flush().await?;
        Ok(())
    }

    async fn flush(&self) -> Result<(), ObsError> {
        let mut f = self.file.lock().await;
        f.flush().await?;
        Ok(())
    }
}

/// OpenTelemetry exporter mapping agent cycles and tool calls to OTel spans.
///
/// Requires the `opentelemetry` feature.
///
/// # Span Mapping
/// - `agent.cycle.start/end` ➔ Root `agent.cycle` span (keyed on `trace_id`).
/// - `tool.call.proposed`/`tool.result` ➔ Child `tool.<name>` span.
#[cfg(feature = "opentelemetry")]
pub mod otel {
    use async_trait::async_trait;
    use eventage_core::{kinds, meta_keys, Event};
    use eventage_observability::{ObsError, ObservabilityExporter};
    use opentelemetry::{
        global,
        trace::{Span, SpanKind, Tracer},
        KeyValue,
    };
    use std::collections::HashMap;
    use tokio::sync::Mutex;

    struct InFlight {
        cycle_spans: HashMap<String, opentelemetry_sdk::trace::Span>,
        tool_spans: HashMap<String, opentelemetry_sdk::trace::Span>,
    }

    /// Translates Eventage events into OpenTelemetry spans.
    ///
    /// Requires a pre-initialized OTel global tracer.
    pub struct OtelExporter {
        tracer_name: String,
        in_flight: Mutex<InFlight>,
    }

    impl OtelExporter {
        /// Creates the exporter matching the provided service/tracer name.
        pub fn new(tracer_name: impl Into<String>) -> Self {
            Self {
                tracer_name: tracer_name.into(),
                in_flight: Mutex::new(InFlight {
                    cycle_spans: HashMap::new(),
                    tool_spans: HashMap::new(),
                }),
            }
        }
    }

    #[async_trait]
    impl ObservabilityExporter for OtelExporter {
        async fn export(&self, event: &Event) -> Result<(), ObsError> {
            let tracer = global::tracer(self.tracer_name.clone());
            let mut state = self.in_flight.lock().await;

            match event.kind.as_str() {
                kinds::AGENT_CYCLE_START => {
                    let trace_id = event
                        .metadata
                        .get(meta_keys::TRACE_ID)
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let agent_id = event
                        .metadata
                        .get(meta_keys::AGENT_ID)
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let span = tracer
                        .span_builder("agent.cycle")
                        .with_kind(SpanKind::Internal)
                        .with_attributes(vec![
                            KeyValue::new("agent.id", agent_id),
                            KeyValue::new("trace.id", trace_id.clone()),
                        ])
                        .start(&tracer);

                    state.cycle_spans.insert(trace_id, span);
                }

                kinds::AGENT_CYCLE_END => {
                    let trace_id = event
                        .metadata
                        .get(meta_keys::TRACE_ID)
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");

                    if let Some(mut span) = state.cycle_spans.remove(trace_id) {
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

                    let span = tracer
                        .span_builder(format!("tool.{}", name))
                        .with_kind(SpanKind::Internal)
                        .with_attributes(vec![KeyValue::new(
                            "tool.call_id",
                            tool_call_id.clone(),
                        )])
                        .start(&tracer);

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
                            span.set_status(opentelemetry::trace::Status::error(
                                "tool returned error",
                            ));
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
            for (_, span) in state.cycle_spans.drain() {
                span.end();
            }
            for (_, span) in state.tool_spans.drain() {
                span.end();
            }
            Ok(())
        }
    }
}
