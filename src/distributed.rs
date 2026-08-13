//! Distributed event bus: share one logical DAG across processes and hosts.
//!
//! A [`BusTransport`] carries events between buses that live in different
//! processes. [`DistributedBus`] wires a local [`EventBus`] to a transport:
//! locally published events are broadcast to peers, and events arriving from
//! peers are republished locally, so every participant converges on the same
//! event stream.
//!
//! The built-in [`TcpTransport`] speaks newline-delimited JSON over TCP —
//! no broker and no extra dependencies. Implement [`BusTransport`] to put
//! events on NATS, Redis Streams, Kafka, or any other fabric; the wiring in
//! [`DistributedBus`] is transport-agnostic.
//!
//! ```no_run
//! # use eventage::{EventBus, distributed::{DistributedBus, TcpTransport}};
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Host A — accept peers.
//! let bus = EventBus::new();
//! let hub = TcpTransport::listen("0.0.0.0:7700").await?;
//! DistributedBus::new(bus.clone(), hub).spawn();
//!
//! // Host B — dial in. Both buses now see each other's events.
//! let bus_b = EventBus::new();
//! let spoke = TcpTransport::connect("host-a:7700").await?;
//! DistributedBus::new(bus_b.clone(), spoke).spawn();
//! # Ok(())
//! # }
//! ```
//!
//! # Semantics
//!
//! - **Loop-free**: forwarded events carry an origin-node marker; a node
//!   never re-broadcasts an event it received (nor one it already saw).
//! - **Ordering** is per-connection, not global. The DAG's `parent_event_id`
//!   links come from each publisher's local view, so concurrent writers on
//!   different hosts interleave rather than serialize. Treat a distributed
//!   bus as an *observation and coordination* fabric — for a single
//!   authoritative branch, keep one writer per logical conversation.
//! - **Delivery is best-effort**: a peer that disconnects misses events
//!   published while it was away. Pair with persistence (`SqliteEventStore`)
//!   plus `EventBus::restore_from` when a node must catch up.

use crate::bus::EventBus;
use crate::event::Event;
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Metadata key naming the node that first published an event.
pub const ORIGIN_NODE_KEY: &str = "origin_node";

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("transport closed")]
    Closed,
}

/// Moves events between buses in different processes.
///
/// Implementations must be safe to use concurrently: [`DistributedBus`]
/// calls [`send`](Self::send) from a publisher task while a receiver task
/// awaits [`recv`](Self::recv).
#[async_trait]
pub trait BusTransport: Send + Sync + 'static {
    /// Broadcast one event to all peers.
    async fn send(&self, event: &Event) -> Result<(), TransportError>;

    /// Await the next event from any peer. `None` when the transport closes.
    async fn recv(&self) -> Option<Event>;
}

// ── DistributedBus ────────────────────────────────────────────────────────────

/// Bridges a local [`EventBus`] onto a [`BusTransport`].
pub struct DistributedBus {
    bus: EventBus,
    transport: Arc<dyn BusTransport>,
    node_id: String,
    /// Kinds to forward; empty forwards everything.
    filter: Vec<String>,
    /// Events already seen (published locally or received), so nothing
    /// echoes around the network.
    seen: Arc<Mutex<HashSet<Uuid>>>,
}

impl DistributedBus {
    pub fn new(bus: EventBus, transport: impl BusTransport) -> Self {
        Self {
            bus,
            transport: Arc::new(transport),
            node_id: Uuid::new_v4().to_string(),
            filter: Vec::new(),
            seen: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Give this node a stable, human-readable id (defaults to a UUID).
    pub fn with_node_id(mut self, id: impl Into<String>) -> Self {
        self.node_id = id.into();
        self
    }

    /// Only forward events whose kind appears in `kinds`.
    ///
    /// Useful to share coordination traffic (`agent.message`) while keeping
    /// bulky local detail (tool results, deltas) on the originating host.
    pub fn filter_kinds(mut self, kinds: Vec<impl Into<String>>) -> Self {
        self.filter = kinds.into_iter().map(|k| k.into()).collect();
        self
    }

    fn should_forward(&self, event: &Event) -> bool {
        self.filter.is_empty() || self.filter.iter().any(|k| k == &event.kind)
    }

    /// `true` if this event is new to this node (and record it).
    fn mark_seen(seen: &Mutex<HashSet<Uuid>>, id: Uuid) -> bool {
        seen.lock().unwrap_or_else(|e| e.into_inner()).insert(id)
    }

    /// Run the bridge until the bus or transport closes.
    pub async fn run(self) {
        let mut rx = self.bus.subscribe();
        let transport = Arc::clone(&self.transport);
        let seen = Arc::clone(&self.seen);
        let bus = self.bus.clone();
        let node_id = self.node_id.clone();

        // Inbound: peer events → local bus.
        let inbound = {
            let transport = Arc::clone(&transport);
            let seen = Arc::clone(&seen);
            tokio::spawn(async move {
                while let Some(event) = transport.recv().await {
                    if !Self::mark_seen(&seen, event.id) {
                        continue; // already known here
                    }
                    debug!(kind = %event.kind, "distributed: inbound event");
                    if let Err(e) = bus.publish(event).await {
                        warn!("distributed: failed to publish inbound event: {e}");
                        break;
                    }
                }
            })
        };

        // Outbound: locally originated events → peers.
        while let Some(event) = rx.recv().await {
            if !self.should_forward(&event) {
                continue;
            }
            // Skip anything that came from the network (already marked seen)
            // or that carries a foreign origin marker.
            let foreign = event
                .metadata
                .get(ORIGIN_NODE_KEY)
                .and_then(|v| v.as_str())
                .is_some_and(|origin| origin != node_id);
            if foreign || !Self::mark_seen(&seen, event.id) {
                continue;
            }
            let tagged = event.with_meta(ORIGIN_NODE_KEY, serde_json::json!(node_id));
            if let Err(e) = transport.send(&tagged).await {
                warn!("distributed: send failed: {e}");
                break;
            }
        }

        inbound.abort();
    }

    /// Run [`run`](Self::run) on a background task.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(self.run())
    }
}

// ── TcpTransport ──────────────────────────────────────────────────────────────

/// Newline-delimited JSON over TCP: a broker-free transport built on tokio.
///
/// [`listen`](Self::listen) accepts many peers and relays every event to all
/// of them; [`connect`](Self::connect) dials a listener. Both directions are
/// symmetric once established.
pub struct TcpTransport {
    /// Outbound events fan out to every connected peer.
    peers: Arc<Mutex<Vec<mpsc::UnboundedSender<String>>>>,
    /// Inbound events from any peer.
    inbound: tokio::sync::Mutex<mpsc::UnboundedReceiver<Event>>,
    inbound_tx: mpsc::UnboundedSender<Event>,
}

impl TcpTransport {
    fn new() -> (Self, mpsc::UnboundedSender<Event>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                peers: Arc::new(Mutex::new(Vec::new())),
                inbound: tokio::sync::Mutex::new(rx),
                inbound_tx: tx.clone(),
            },
            tx,
        )
    }

    /// Bind `addr` and accept peer connections in the background.
    pub async fn listen(addr: impl ToSocketAddrs) -> Result<Self, TransportError> {
        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        let (transport, inbound_tx) = Self::new();
        let peers = Arc::clone(&transport.peers);

        tokio::spawn(async move {
            info!(%local, "distributed bus listening");
            loop {
                match listener.accept().await {
                    Ok((stream, peer)) => {
                        debug!(%peer, "distributed bus: peer connected");
                        Self::attach(stream, Arc::clone(&peers), inbound_tx.clone());
                    }
                    Err(e) => {
                        warn!("distributed bus: accept failed: {e}");
                        break;
                    }
                }
            }
        });

        Ok(transport)
    }

    /// Dial a listening peer.
    pub async fn connect(addr: impl ToSocketAddrs) -> Result<Self, TransportError> {
        let stream = TcpStream::connect(addr).await?;
        let (transport, inbound_tx) = Self::new();
        Self::attach(stream, Arc::clone(&transport.peers), inbound_tx);
        Ok(transport)
    }

    /// Number of currently connected peers.
    pub fn peer_count(&self) -> usize {
        self.peers.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Spawn reader and writer tasks for one peer connection.
    fn attach(
        stream: TcpStream,
        peers: Arc<Mutex<Vec<mpsc::UnboundedSender<String>>>>,
        inbound_tx: mpsc::UnboundedSender<Event>,
    ) {
        let (read_half, mut write_half) = stream.into_split();
        let (out_tx, mut out_rx) = mpsc::unbounded_channel::<String>();
        peers.lock().unwrap_or_else(|e| e.into_inner()).push(out_tx);

        // Writer: drain the queue onto the socket.
        tokio::spawn(async move {
            while let Some(line) = out_rx.recv().await {
                if write_half.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if write_half.write_all(b"\n").await.is_err() {
                    break;
                }
            }
        });

        // Reader: parse frames into events.
        tokio::spawn(async move {
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Event>(&line) {
                    Ok(event) => {
                        if inbound_tx.send(event).is_err() {
                            break;
                        }
                    }
                    Err(e) => warn!("distributed bus: bad frame: {e}"),
                }
            }
            debug!("distributed bus: peer disconnected");
        });
    }
}

#[async_trait]
impl BusTransport for TcpTransport {
    async fn send(&self, event: &Event) -> Result<(), TransportError> {
        let line = serde_json::to_string(event)?;
        let mut peers = self.peers.lock().unwrap_or_else(|e| e.into_inner());
        // Drop peers whose writer task has exited.
        peers.retain(|tx| tx.send(line.clone()).is_ok());
        Ok(())
    }

    async fn recv(&self) -> Option<Event> {
        self.inbound.lock().await.recv().await
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Dropping the sender lets `recv` terminate.
        let _ = &self.inbound_tx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::kinds;
    use serde_json::json;
    use std::time::Duration;

    async fn wait_for_kind(bus: &EventBus, kind: &str) -> Option<Event> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(e) = bus.log().await.into_iter().find(|e| e.kind == kind) {
                return Some(e);
            }
            if tokio::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn events_replicate_between_two_hosts() {
        let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = tcp_listener.local_addr().unwrap();
        drop(tcp_listener);

        let hub = TcpTransport::listen(addr).await.unwrap();
        let bus_a = EventBus::new();
        DistributedBus::new(bus_a.clone(), hub)
            .with_node_id("a")
            .spawn();

        let spoke = TcpTransport::connect(addr).await.unwrap();
        let bus_b = EventBus::new();
        DistributedBus::new(bus_b.clone(), spoke)
            .with_node_id("b")
            .spawn();

        // Give the connection a moment to establish on both sides.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // A → B
        bus_a
            .publish(Event::new(kinds::AGENT_MESSAGE, json!({"text": "from a"})))
            .await
            .unwrap();
        let on_b = wait_for_kind(&bus_b, kinds::AGENT_MESSAGE)
            .await
            .expect("event should replicate a→b");
        assert_eq!(on_b.payload["text"], "from a");
        assert_eq!(on_b.metadata[ORIGIN_NODE_KEY], "a");

        // B → A
        bus_b
            .publish(Event::new(kinds::USER_MESSAGE, json!({"text": "from b"})))
            .await
            .unwrap();
        let on_a = wait_for_kind(&bus_a, kinds::USER_MESSAGE)
            .await
            .expect("event should replicate b→a");
        assert_eq!(on_a.payload["text"], "from b");

        // No echo storm: each side holds exactly one copy of each event.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let count_a = bus_a
            .log()
            .await
            .iter()
            .filter(|e| e.kind == kinds::AGENT_MESSAGE)
            .count();
        assert_eq!(count_a, 1, "event must not echo back to its origin");
    }

    #[tokio::test]
    async fn filter_limits_what_crosses_the_wire() {
        let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = tcp_listener.local_addr().unwrap();
        drop(tcp_listener);

        let hub = TcpTransport::listen(addr).await.unwrap();
        let bus_a = EventBus::new();
        DistributedBus::new(bus_a.clone(), hub)
            .with_node_id("a")
            .filter_kinds(vec![kinds::AGENT_MESSAGE])
            .spawn();

        let spoke = TcpTransport::connect(addr).await.unwrap();
        let bus_b = EventBus::new();
        DistributedBus::new(bus_b.clone(), spoke)
            .with_node_id("b")
            .spawn();
        tokio::time::sleep(Duration::from_millis(100)).await;

        bus_a
            .publish(Event::new(kinds::TOOL_RESULT, json!({"result": "secret"})))
            .await
            .unwrap();
        bus_a
            .publish(Event::new(kinds::AGENT_MESSAGE, json!({"text": "shared"})))
            .await
            .unwrap();

        assert!(
            wait_for_kind(&bus_b, kinds::AGENT_MESSAGE).await.is_some(),
            "allowed kind should replicate"
        );
        assert!(
            !bus_b
                .log()
                .await
                .iter()
                .any(|e| e.kind == kinds::TOOL_RESULT),
            "filtered kind must not cross the wire"
        );
    }
}
