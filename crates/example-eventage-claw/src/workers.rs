//! Background workers for eventage-claw.
//!
//! - [`SchedulerWorker`] — fires due tasks on system.heartbeat events.
//!
//! - [`RelayWorker`] — routes `claw.group.message` events to target group buses.

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;
use eventage::{
    agent::worker::{EventWorker, WorkerError},
    event::{kinds, Event},
    EventBus,
};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Shared, live map of group name → per-group EventBus.
///
/// Wrapped in `Arc<RwLock<…>>` so dynamically spawned groups can be inserted
/// at runtime and immediately become routable by `RelayWorker` and
/// `SchedulerWorker` without restarting.
pub type GroupBuses = Arc<RwLock<HashMap<String, EventBus>>>;

use crate::kinds::{CLAW_GROUP_MESSAGE, CLAW_GROUP_REPLY, CLAW_SCHEDULE_FIRE};
use crate::tools::schedule::{advance_schedule, ScheduleState};

/// Produce a valid ChatMessage `name` from an arbitrary string.
///
/// Keeps only alphanumeric, underscore, and hyphen characters, then truncates
/// to 64 chars — the safe subset accepted by all OpenAI-compatible providers.
fn sanitize_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .take(64)
        .collect()
}

// ── SchedulerWorker ───────────────────────────────────────────────────────────

/// Subscribes to `system.heartbeat` events (from `HeartbeatScheduler`).
///
/// On each tick, checks `ScheduleState` for due tasks and publishes
/// `CLAW_SCHEDULE_FIRE`, then injects a `user.message` into the target
/// group's bus so the agent processes it.
pub struct SchedulerWorker {
    pub state: ScheduleState,
    pub group_buses: GroupBuses,
}

#[async_trait]
impl EventWorker for SchedulerWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::SYSTEM_HEARTBEAT.to_string()]
    }

    async fn handle(&self, _event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        let now = Utc::now();
        let mut state = self.state.lock().await;

        struct FiringTask {
            id: String,
            name: String,
            description: String,
            target_group: Option<String>,
            reply_group: Option<String>,
        }
        let mut to_fire: Vec<FiringTask> = vec![];
        let mut completed_once_ids: Vec<String> = vec![];

        for task in state.iter_mut() {
            if task.paused || task.next_fire > now {
                continue;
            }

            to_fire.push(FiringTask {
                id: task.id.clone(),
                name: task.name.clone(),
                description: task.description.clone(),
                target_group: task.target_group.clone(),
                reply_group: task.reply_group.clone(),
            });

            // Advance next_fire
            match advance_schedule(&task.schedule_kind, task.next_fire) {
                Some(next) => {
                    task.next_fire = next;
                    task.fired_count += 1;
                }
                None => {
                    // Once task — mark for removal
                    task.fired_count += 1;
                    completed_once_ids.push(task.id.clone());
                }
            }
        }

        // Remove completed once-tasks
        state.retain(|t| !completed_once_ids.contains(&t.id));
        drop(state);

        for task in to_fire {
            let FiringTask { id: task_id, name, description, target_group, reply_group } = task;
            info!(task_id = %task_id, name = %name, "SchedulerWorker: firing task");

            // Publish the fire event on the shared bus (observable in TUI/log)
            bus.publish(Event::new(
                CLAW_SCHEDULE_FIRE,
                json!({
                    "task_id": task_id,
                    "name": name,
                    "description": description,
                    "target_group": target_group,
                    "reply_group": reply_group,
                }),
            ))
            .await
            .map_err(WorkerError::Bus)?;

            let text = format!("[Task: {name}]\n{description}");

            // If a reply_group is set the task was created by a sub-agent.
            // Fire via the relay mechanism (CLAW_GROUP_MESSAGE on shared bus)
            // so DelegationReplyWorker routes the sub-agent's response back to
            // reply_group, which then delivers it to the user.
            if let Some(ref src) = reply_group {
                let _ = bus
                    .publish(Event::new(
                        CLAW_GROUP_MESSAGE,
                        json!({
                            "message_id": Uuid::new_v4().to_string(),
                            "target_group": target_group,
                            "source_group": src,
                            "content": text,
                            "caller_awaits": false,
                        }),
                    ))
                    .await;
                continue;
            }

            // No reply_group: inject user.message directly into the target bus.
            let buses = self.group_buses.read().await;
            let targets: Vec<EventBus> = if let Some(ref g) = target_group {
                buses.get(g).cloned().into_iter().collect()
            } else {
                buses.values().cloned().collect()
            };
            drop(buses);

            for group_bus in targets {
                let _ = group_bus
                    .publish(Event::new(
                        kinds::SYSTEM_MESSAGE,
                        json!({
                            "text": text,
                            "name": "scheduler",
                            "source": "scheduler",
                            "task_id": task_id,
                        }),
                    ))
                    .await;
            }
        }

        Ok(())
    }
}

// ── ChannelOutputWorker ───────────────────────────────────────────────────────

/// Runs on a per-group bus. Routes final agent responses back to an external
/// channel (WhatsApp, Telegram, etc.) via a webhook POST.
///
/// Flow:
///   1. `user.message` with `reply_to` field arrives → store the JID/address
///   2. `assistant.message` with no pending tool calls → POST response to webhook
///
/// The webhook (e.g. the WhatsApp bridge) receives:
/// ```json
/// { "reply_to": "<jid>", "text": "...", "group": "personal" }
/// ```
/// and forwards it back to the originating channel.
pub struct ChannelOutputWorker {
    pub webhook_url: String,
    pub group_name: String,
    /// Stores the reply address (WhatsApp JID, etc.) from the most recent
    /// inbound `user.message`.
    reply_to: Arc<Mutex<Option<String>>>,
    /// Shared HTTP client — cheaper than creating one per request.
    client: reqwest::Client,
}

impl ChannelOutputWorker {
    pub fn new(webhook_url: impl Into<String>, group_name: impl Into<String>) -> Self {
        Self {
            webhook_url: webhook_url.into(),
            group_name: group_name.into(),
            reply_to: Arc::new(Mutex::new(None)),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EventWorker for ChannelOutputWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![
            kinds::USER_MESSAGE.to_string(),
            kinds::ASSISTANT_MESSAGE.to_string(),
        ]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        match event.kind.as_str() {
            k if k == kinds::USER_MESSAGE => {
                // Capture the reply address from inbound channel messages.
                if let Some(reply_to) = event.payload["reply_to"].as_str() {
                    *self.reply_to.lock().await = Some(reply_to.to_string());
                    debug!(reply_to = %reply_to, "ChannelOutputWorker: captured reply_to");
                }
            }

            k if k == kinds::ASSISTANT_MESSAGE => {
                let content = match event.payload["content"].as_str() {
                    Some(c) if !c.trim().is_empty() => c.to_string(),
                    _ => {
                        debug!("ChannelOutputWorker: skipping tool-only assistant message");
                        return Ok(());
                    }
                };

                let mut reply_to = self.reply_to.lock().await;

                // If reply_to is None the worker missed the user.message that
                // carried it (published before this worker subscribed — a startup
                // race when an HTTP message arrives before Tokio schedules the
                // worker task).  Recover by scanning the bus log backwards for
                // the most recent user.message that has an explicit reply_to.
                if reply_to.is_none() {
                    for e in bus.log().await.iter().rev() {
                        if e.kind == kinds::USER_MESSAGE {
                            if let Some(rt) = e.payload["reply_to"].as_str() {
                                if !rt.is_empty() {
                                    debug!(reply_to = %rt, "ChannelOutputWorker: recovered reply_to from bus log");
                                    *reply_to = Some(rt.to_string());
                                    break;
                                }
                            }
                        }
                    }
                }

                let Some(jid) = reply_to.clone() else {
                    debug!(
                        "ChannelOutputWorker: no reply_to context — skipping (TUI/REPL message)"
                    );
                    return Ok(());
                };

                info!(
                    group = %self.group_name,
                    reply_to = %jid,
                    webhook = %self.webhook_url,
                    chars = content.len(),
                    "ChannelOutputWorker: forwarding response to channel"
                );

                let webhook = self.webhook_url.clone();
                let group = self.group_name.clone();
                let client = self.client.clone();

                // Fire-and-forget: don't block the event loop on network I/O.
                tokio::spawn(async move {
                    let body = json!({
                        "reply_to": jid,
                        "text": content,
                        "group": group,
                    });
                    match client.post(&webhook).json(&body).send().await {
                        Ok(r) if r.status().is_success() => {
                            info!("ChannelOutputWorker: ✓ response delivered to channel");
                        }
                        Ok(r) => {
                            warn!(status = %r.status(), "ChannelOutputWorker: webhook returned error");
                        }
                        Err(e) => {
                            warn!("ChannelOutputWorker: webhook POST failed: {e}");
                        }
                    }
                });
            }

            _ => {}
        }

        Ok(())
    }
}

// ── RelayWorker ───────────────────────────────────────────────────────────────

/// Subscribes to `claw.group.message` and `claw.group.reply` events on the
/// shared bus.
///
/// `claw.group.message` — routes to the target group's per-group bus as an
/// `agent.message`, completing the EventBus-as-IPC pattern.
///
/// `claw.group.reply` — when the original caller used `await_reply=false`
/// (async delegation), routes the sub-agent's reply back to the caller's
/// per-group bus as an `agent.message` so the caller is woken up and can
/// deliver the result to the user.  When `await_reply=true` (sync), the
/// calling tool is already blocked on `wait_for` watching the shared bus
/// directly, so no routing is needed.
///
/// Authorization rules:
/// - Main groups may message any registered target group.
/// - Non-main groups may only message themselves (same name as source_group).
pub struct RelayWorker {
    pub group_buses: GroupBuses,
    /// Names of groups with admin / main-group privileges.
    pub main_groups: Vec<String>,
}

#[async_trait]
impl EventWorker for RelayWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![CLAW_GROUP_MESSAGE.to_string(), CLAW_GROUP_REPLY.to_string()]
    }

    async fn handle(&self, event: &Event, _bus: &EventBus) -> Result<(), WorkerError> {
        match event.kind.as_str() {
            k if k == CLAW_GROUP_MESSAGE => self.handle_message(event).await,
            k if k == CLAW_GROUP_REPLY => self.handle_reply(event).await,
            _ => Ok(()),
        }
    }
}

impl RelayWorker {
    async fn handle_message(&self, event: &Event) -> Result<(), WorkerError> {
        let target = match event.payload["target_group"].as_str() {
            Some(t) => t,
            None => return Ok(()),
        };
        let content = event.payload["content"].as_str().unwrap_or("");
        let source = event.payload["source_group"].as_str().unwrap_or("unknown");
        let msg_id = event.payload["message_id"].as_str().unwrap_or("");
        let caller_awaits = event.payload["caller_awaits"].as_bool().unwrap_or(true);

        // Authorization: non-main groups may only relay to themselves.
        let is_main = self.main_groups.iter().any(|g| g == source);
        if !is_main && target != source {
            warn!(
                source = %source,
                target = %target,
                "RelayWorker: unauthorized IPC — non-main group can only message itself"
            );
            return Ok(());
        }

        debug!(target = %target, source = %source, "RelayWorker: routing message");

        let target_bus = self.group_buses.read().await.get(target).cloned();
        if let Some(bus) = target_bus {
            bus.publish(Event::new(
                kinds::AGENT_MESSAGE,
                json!({
                    "text": content,
                    "name": sanitize_name(&format!("agent_{source}")),
                    "source_group": source,
                    "message_id": msg_id,
                    "via": "relay",
                    "caller_awaits": caller_awaits,
                }),
            ))
            .await
            .map_err(WorkerError::Bus)?;
        } else {
            debug!(target = %target, "RelayWorker: target group not found");
        }

        Ok(())
    }

    /// Route a sub-agent reply back to the caller's per-group bus — only when
    /// the caller used `await_reply=false` (async).  Sync callers (`await_reply=true`)
    /// are already blocked on `wait_for` on the shared bus and don't need routing.
    async fn handle_reply(&self, event: &Event) -> Result<(), WorkerError> {
        // Only route async replies; sync callers pick up CLAW_GROUP_REPLY directly.
        let caller_awaited = event.payload["caller_awaited"].as_bool().unwrap_or(true);
        if caller_awaited {
            return Ok(());
        }

        let target = match event.payload["target_group"].as_str() {
            Some(t) => t,
            None => return Ok(()),
        };
        let source = event.payload["source_group"].as_str().unwrap_or("unknown");
        let content = event.payload["content"].as_str().unwrap_or("");
        let in_reply_to = event.payload["in_reply_to"].as_str().unwrap_or("");

        debug!(
            target = %target,
            source = %source,
            "RelayWorker: routing async reply to caller"
        );

        let target_bus = self.group_buses.read().await.get(target).cloned();
        if let Some(bus) = target_bus {
            bus.publish(Event::new(
                kinds::AGENT_MESSAGE,
                json!({
                    "text": content,
                    "name": sanitize_name(&format!("agent_reply_{source}")),
                    "source_group": source,
                    "message_id": in_reply_to,
                    "via": "async_reply",
                }),
            ))
            .await
            .map_err(WorkerError::Bus)?;
        } else {
            debug!(target = %target, "RelayWorker: caller group not found for async reply");
        }

        Ok(())
    }
}

// ── DelegationReplyWorker ─────────────────────────────────────────────────────

/// Runs on each group's **per-group bus**.
///
/// Closes the reply loop for `MessageGroupTool(await_reply=true)`:
/// 1. Detects inbound relay requests (`agent.message` with `via: "relay"`)
/// 2. On `agent.cycle.start`, marks the cycle as relay-triggered if one is pending
/// 3. Captures `assistant.message` content during relay-triggered cycles only
/// 4. On `agent.cycle.end` of a relay-triggered cycle, publishes `CLAW_GROUP_REPLY`
///    on the shared bus, unblocking the calling agent's `await_reply`.
///
/// Tracking relay-to-cycle correlation via `AGENT_CYCLE_START` prevents returning
/// content from a concurrent non-relay cycle (e.g. an inbound WhatsApp message
/// being processed at the same time as a delegation arrives).
pub struct DelegationReplyWorker {
    pub shared_bus: EventBus,
    pub group_name: String,
    state: Arc<Mutex<RelayState>>,
}

struct RelayState {
    /// Relay requests waiting for the next relay-triggered cycle.
    pending: VecDeque<PendingReply>,
    /// Set true when a cycle starts with a pending relay — cleared on cycle end.
    in_relay_cycle: bool,
    /// Content captured from the current relay-triggered cycle's assistant message.
    last_content: Option<String>,
}

struct PendingReply {
    source_group: String,
    message_id: String,
    /// Whether the caller is blocking on `wait_for` (true) or async (false).
    caller_awaits: bool,
}

impl DelegationReplyWorker {
    pub fn new(shared_bus: EventBus, group_name: impl Into<String>) -> Self {
        Self {
            shared_bus,
            group_name: group_name.into(),
            state: Arc::new(Mutex::new(RelayState {
                pending: VecDeque::new(),
                in_relay_cycle: false,
                last_content: None,
            })),
        }
    }
}

#[async_trait]
impl EventWorker for DelegationReplyWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![
            kinds::AGENT_MESSAGE.to_string(),
            kinds::AGENT_CYCLE_START.to_string(),
            kinds::ASSISTANT_MESSAGE.to_string(),
            kinds::AGENT_CYCLE_END.to_string(),
        ]
    }

    async fn handle(&self, event: &Event, _bus: &EventBus) -> Result<(), WorkerError> {
        match event.kind.as_str() {
            k if k == kinds::AGENT_MESSAGE => {
                if event.payload.get("via").and_then(|v| v.as_str()) != Some("relay") {
                    return Ok(());
                }
                let source = event.payload["source_group"].as_str().unwrap_or("").to_string();
                let msg_id = event.payload["message_id"].as_str().unwrap_or("").to_string();
                let caller_awaits = event.payload["caller_awaits"].as_bool().unwrap_or(true);
                if !msg_id.is_empty() {
                    debug!(
                        group = %self.group_name,
                        source = %source,
                        msg_id = %msg_id,
                        "DelegationReplyWorker: queued relay request"
                    );
                    self.state.lock().await.pending.push_back(PendingReply {
                        source_group: source,
                        message_id: msg_id,
                        caller_awaits,
                    });
                }
            }

            k if k == kinds::AGENT_CYCLE_START => {
                // Only mark relay-triggered if a request is already pending — cycles
                // that started before the relay arrived are never mistakenly flagged.
                let mut state = self.state.lock().await;
                if !state.pending.is_empty() {
                    state.in_relay_cycle = true;
                    state.last_content = None;
                }
            }

            k if k == kinds::ASSISTANT_MESSAGE => {
                if let Some(content) = event.payload.get("content").and_then(|v| v.as_str()) {
                    if !content.trim().is_empty() {
                        let mut state = self.state.lock().await;
                        if state.in_relay_cycle {
                            state.last_content = Some(content.to_string());
                        }
                    }
                }
            }

            k if k == kinds::AGENT_CYCLE_END => {
                // Extract reply data under one lock, then publish without holding it.
                let reply_data = {
                    let mut state = self.state.lock().await;
                    if !state.in_relay_cycle {
                        return Ok(());
                    }
                    state.in_relay_cycle = false;
                    state
                        .pending
                        .pop_front()
                        .map(|r| (r, state.last_content.take().unwrap_or_default()))
                };

                let Some((PendingReply { source_group, message_id, caller_awaits }, content)) = reply_data else {
                    return Ok(());
                };

                info!(
                    group = %self.group_name,
                    target = %source_group,
                    msg_id = %message_id,
                    "DelegationReplyWorker: routing reply back to caller"
                );

                self.shared_bus
                    .publish(Event::new(
                        CLAW_GROUP_REPLY,
                        json!({
                            "target_group": source_group,
                            "source_group": self.group_name,
                            "content": content,
                            "in_reply_to": message_id,
                            // Tells RelayWorker whether to push this back as
                            // agent.message (async path) or leave it for wait_for (sync path).
                            "caller_awaited": caller_awaits,
                        }),
                    ))
                    .await
                    .map_err(WorkerError::Bus)?;
            }

            _ => {}
        }

        Ok(())
    }
}
