//! A [`BusTransport`] over a child process's
//! stdin and stdout.
//!
//! Newline-delimited JSON, one [`Event`] per line — the same framing
//! [`TcpTransport`](crate::distributed::TcpTransport) uses and the same the MCP
//! stdio client uses, so a plugin author already knows the shape.
//!
//! # Why stdio and not a socket
//!
//! The channel *is* the child's own file descriptors. There is no address to
//! guess, nothing to authenticate, nothing left behind on a crash, and no
//! third process can attach. A loopback port would need a token — the same
//! secret-distribution problem that makes an HTTP surface the wrong answer for
//! something whose premise is "drop a directory in and it works".
//!
//! It also makes teardown total: the transport owns the child, `kill_on_drop`
//! reaps it, so dropping this struct is the whole cleanup.
//!
//! # The frame bound
//!
//! Inbound frames are capped, and unlike [`TcpTransport`](crate::distributed::TcpTransport)
//! that costs nothing
//! here, because the two directions have opposite trust properties:
//!
//! - **Outbound** (host → child) is large and legitimate — a context manifest
//!   runs to hundreds of kilobytes, a base64 image to megabytes — and carries
//!   no risk, because we serialized it and already hold the bytes.
//! - **Inbound** (child → host) is untrusted, and nothing legitimate is large.
//!   It is the `emit` path, whose allow-list contains only summaries, notes
//!   and decisions: kilobytes at the outside.
//!
//! So the cap is not a compromise between safety and capability, it is an
//! accurate statement of the contract. It is applied with `take()` at the
//! read, so the worst case is a deterministic allocation rather than something
//! discovered after the buffer has already grown.
//!
//! A frame that hits the cap leaves the stream desynchronised — there is no
//! way to know where the next one begins — so the reader stops rather than
//! trying to resynchronise. A plugin emitting over-cap frames is broken, and
//! saying so beats guessing.

use crate::distributed::{BusTransport, TransportError};
use crate::event::{Event, EventId};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::Mutex as StdMutex;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};
use uuid::Uuid;

/// Largest inbound frame accepted from a child, in bytes.
///
/// Two orders of magnitude above anything an honest observer sends.
pub const DEFAULT_MAX_FRAME: usize = 1024 * 1024;

/// How many parsed inbound events may queue before the reader task waits.
const INBOUND_CAPACITY: usize = 64;

/// A child process wired to the bus by newline-delimited JSON.
pub struct ChildTransport {
    /// Held so the child lives exactly as long as the transport; spawned with
    /// `kill_on_drop`, so dropping this kills it.
    _child: StdMutex<Child>,
    stdin: Mutex<ChildStdin>,
    inbound: Mutex<mpsc::Receiver<Event>>,
}

impl ChildTransport {
    /// Spawn `program` and wire its stdio to this transport.
    ///
    /// `stderr` is inherited, so a plugin's diagnostics land in the agent's
    /// log rather than being swallowed. `cwd` is set explicitly rather than
    /// inherited: a plugin that resolves a path relative to "the workspace"
    /// should not be at the mercy of where the host happened to be started.
    pub async fn spawn(
        label: &str,
        program: &str,
        args: &[String],
        env: &[(String, String)],
        cwd: &Path,
        max_frame: usize,
    ) -> Result<Self, TransportError> {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd.spawn().map_err(|e| {
            TransportError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to spawn observer '{label}' ({program}): {e}"),
            ))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| TransportError::Io(std::io::Error::other("child stdin unavailable")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| TransportError::Io(std::io::Error::other("child stdout unavailable")))?;

        let (tx, rx) = mpsc::channel(INBOUND_CAPACITY);
        let label = label.to_string();
        tokio::spawn(async move {
            read_frames(stdout, tx, max_frame, &label).await;
        });

        Ok(Self {
            _child: StdMutex::new(child),
            stdin: Mutex::new(stdin),
            inbound: Mutex::new(rx),
        })
    }
}

/// An inbound event as a plugin author actually writes it.
///
/// `Event` requires an id and a timestamp because everything that publishes
/// one from inside the process has them. A plugin does not, and making it mint
/// a UUID and format an RFC 3339 string before it can say anything is friction
/// with no purpose — so both default here, and `{"kind": ..., "payload": ...}`
/// is a complete frame.
///
/// `parent_event_id` is deliberately absent: the bus assigns lineage from the
/// branch tip at publish, and letting a plugin choose its own parent would let
/// it rewrite the DAG's shape.
#[derive(Debug, Deserialize)]
struct WireEvent {
    #[serde(default = "Uuid::new_v4")]
    id: EventId,
    #[serde(default = "Utc::now")]
    timestamp: DateTime<Utc>,
    kind: String,
    #[serde(default)]
    payload: serde_json::Value,
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
}

impl From<WireEvent> for Event {
    fn from(w: WireEvent) -> Self {
        Event {
            id: w.id,
            timestamp: w.timestamp,
            kind: w.kind,
            payload: w.payload,
            parent_event_id: None,
            metadata: w.metadata,
        }
    }
}

/// Read `\n`-delimited JSON events until EOF, an over-cap frame, or the
/// receiver going away.
async fn read_frames(
    stdout: tokio::process::ChildStdout,
    tx: mpsc::Sender<Event>,
    max_frame: usize,
    label: &str,
) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);

        // `take` bounds the read at the source, so an endless line costs one
        // capped allocation rather than the host's memory.
        let read = (&mut reader)
            .take(max_frame as u64)
            .read_until(b'\n', &mut buf)
            .await;

        let n = match read {
            Ok(0) => {
                debug!(observer = label, "observer closed its stdout");
                return;
            }
            Ok(n) => n,
            Err(e) => {
                warn!(observer = label, "observer read failed: {e}");
                return;
            }
        };

        if buf.last() != Some(&b'\n') {
            warn!(
                observer = label,
                bytes = n,
                limit = max_frame,
                "observer sent a frame over the size limit; stopping it — the \
                 stream cannot be resynchronised after a partial frame"
            );
            return;
        }

        let line = String::from_utf8_lossy(&buf);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        match serde_json::from_str::<WireEvent>(line) {
            Ok(wire) => {
                if tx.send(Event::from(wire)).await.is_err() {
                    return; // bridge gone
                }
            }
            // One bad frame is not a reason to kill a plugin: unlike an
            // over-cap frame, the stream is still aligned on the next newline.
            Err(e) => warn!(observer = label, "observer sent a bad frame: {e}"),
        }
    }
}

impl ChildTransport {
    /// Write an already-serialized frame.
    ///
    /// The bridge serializes each event once and hands the same `Arc<str>` to
    /// every observer watching that kind; going back through [`send`] would
    /// re-encode a 300 KB manifest per subscriber and undo the point.
    pub(crate) async fn send_raw(&self, frame: &str) -> Result<(), TransportError> {
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(frame.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;
        Ok(())
    }
}

#[async_trait]
impl BusTransport for ChildTransport {
    async fn send(&self, event: &Event) -> Result<(), TransportError> {
        self.send_raw(&serde_json::to_string(event)?).await
    }

    async fn recv(&self) -> Option<Event> {
        self.inbound.lock().await.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::kinds;
    use serde_json::json;

    fn cwd() -> std::path::PathBuf {
        std::env::current_dir().unwrap()
    }

    /// The round trip a plugin author actually writes: read a line, write a line.
    #[tokio::test]
    async fn an_event_makes_the_round_trip_through_a_child() {
        let transport = ChildTransport::spawn(
            "echoer",
            "sh",
            &[
                "-c".into(),
                "while IFS= read -r line; do printf '%s\\n' \"$line\"; done".into(),
            ],
            &[],
            &cwd(),
            DEFAULT_MAX_FRAME,
        )
        .await
        .unwrap();

        let sent = Event::new(kinds::USER_MESSAGE, json!({ "text": "ping" }));
        transport.send(&sent).await.unwrap();

        let back = tokio::time::timeout(std::time::Duration::from_secs(10), transport.recv())
            .await
            .expect("child should have echoed within the timeout")
            .expect("stream should still be open");

        assert_eq!(back.id, sent.id);
        assert_eq!(back.payload["text"], "ping");
    }

    /// The whole point of the bound: an endless line must not grow the host.
    #[tokio::test]
    async fn an_oversized_frame_stops_the_reader_instead_of_growing() {
        let transport = ChildTransport::spawn(
            "flooder",
            "sh",
            // Never emits a newline.
            &[
                "-c".into(),
                "while :; do printf 'aaaaaaaaaaaaaaaa'; done".into(),
            ],
            &[],
            &cwd(),
            4096,
        )
        .await
        .unwrap();

        let result = tokio::time::timeout(std::time::Duration::from_secs(10), transport.recv())
            .await
            .expect("the reader should give up rather than buffer forever");
        assert!(result.is_none(), "an over-cap frame ends the stream");
    }

    /// A malformed line is recoverable; the next good one still arrives.
    #[tokio::test]
    async fn a_bad_frame_is_skipped_not_fatal() {
        let transport = ChildTransport::spawn(
            "noisy",
            "sh",
            &[
                "-c".into(),
                "printf 'not json\\n'; printf '{\"kind\":\"x.y\",\"payload\":{}}\\n'; sleep 5"
                    .into(),
            ],
            &[],
            &cwd(),
            DEFAULT_MAX_FRAME,
        )
        .await
        .unwrap();

        let got = tokio::time::timeout(std::time::Duration::from_secs(10), transport.recv())
            .await
            .expect("should not time out")
            .expect("the good frame after the bad one should arrive");
        assert_eq!(got.kind, "x.y");
    }

    /// A plugin should not have to mint a UUID to say something.
    #[tokio::test]
    async fn a_frame_may_omit_its_id_and_timestamp() {
        let transport = ChildTransport::spawn(
            "terse",
            "sh",
            &[
                "-c".into(),
                "printf '{\"kind\":\"agent.context.summarized\",\"payload\":{\"summary\":\"s\"}}\\n'; sleep 5".into(),
            ],
            &[],
            &cwd(),
            DEFAULT_MAX_FRAME,
        )
        .await
        .unwrap();

        let got = tokio::time::timeout(std::time::Duration::from_secs(10), transport.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.kind, "agent.context.summarized");
        assert_eq!(got.payload["summary"], "s");
        assert!(!got.id.is_nil(), "an id was minted for it");
    }
}
