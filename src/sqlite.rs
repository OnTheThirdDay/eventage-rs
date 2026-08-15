//! SQLite-backed persistence for Eventage event logs.
//!
//! Provides two complementary types:
//!
//! - **[`SqliteEventStore`]**: Direct store for reading and writing events.
//! - **[`SqliteExporter`]**: An [`ObservabilityExporter`] that streams live events to SQLite.
//!
//! # Startup restore pattern
//!
//! ```no_run
//! use eventage::EventBus;
//! use eventage::sqlite::SqliteEventStore;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let bus = EventBus::new();
//! let store = SqliteEventStore::new("./events.db").await?;
//!
//! // Restore persisted log before starting agents.
//! let saved = store.load_all().await?;
//! bus.restore_from(saved).await;
//!
//! // Now start agents — they see the full history via bus.log().
//! # Ok(())
//! # }
//! ```
//!
//! # Live persistence pattern
//!
//! ```no_run
//! use eventage::EventBus;
//! use eventage::observability::BusObserver;
//! use eventage::sqlite::{SqliteEventStore, SqliteExporter};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let bus = EventBus::new();
//! let exporter = SqliteExporter::new("./events.db").await?;
//!
//! tokio::spawn(
//!     BusObserver::new(bus.clone())
//!         .add_exporter(exporter)
//!         .run()
//! );
//! # Ok(())
//! # }
//! ```

use crate::event::Event;
use crate::observability::{ObsError, ObservabilityExporter};
use async_trait::async_trait;
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::debug;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum SqliteError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

// ── SqliteEventStore ──────────────────────────────────────────────────────────

/// Persistent, ordered event store backed by SQLite.
///
/// SQLite operations are dispatched via [`tokio::task::spawn_blocking`]
/// to avoid blocking the async executor.
///
/// Creates an `events` table on first use.
/// Events are stored as complete JSON payloads in the `data` column,
/// ensuring exact deserialization without schema migrations.
#[derive(Clone)]
pub struct SqliteEventStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteEventStore {
    /// Open (or create) a SQLite database at `path`.
    pub async fn new(path: impl Into<PathBuf>) -> Result<Self, SqliteError> {
        let path = path.into();
        let conn = tokio::task::spawn_blocking(move || -> Result<Connection, rusqlite::Error> {
            let conn = Connection::open(&path)?;
            conn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=NORMAL;
                 CREATE TABLE IF NOT EXISTS events (
                     idx   INTEGER PRIMARY KEY AUTOINCREMENT,
                     id    TEXT    UNIQUE NOT NULL,
                     kind  TEXT    NOT NULL,
                     ts_ms INTEGER NOT NULL,
                     data  TEXT    NOT NULL
                 );",
            )?;
            Ok(conn)
        })
        .await??;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Append a single event. Silently ignores duplicate `id`s (idempotent).
    pub async fn append(&self, event: &Event) -> Result<(), SqliteError> {
        let id = event.id.to_string();
        let kind = event.kind.clone();
        let ts_ms = event.timestamp.timestamp_millis();
        let data = serde_json::to_string(event)?;
        let conn = Arc::clone(&self.conn);
        let kind_log = kind.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();
            conn.execute(
                "INSERT OR IGNORE INTO events (id, kind, ts_ms, data) VALUES (?1, ?2, ?3, ?4)",
                params![id, kind, ts_ms, data],
            )
        })
        .await??;

        debug!(kind = %kind_log, "event appended to sqlite");
        Ok(())
    }

    /// Loads all events in insertion order.
    ///
    /// Useful with [`EventBus::restore_from`](crate::EventBus::restore_from) to rebuild bus state.
    pub async fn load_all(&self) -> Result<Vec<Event>, SqliteError> {
        let conn = Arc::clone(&self.conn);
        let rows = tokio::task::spawn_blocking(move || -> Result<Vec<String>, rusqlite::Error> {
            let conn = conn.blocking_lock();
            let mut stmt = conn.prepare("SELECT data FROM events ORDER BY idx")?;
            let rows: Result<Vec<String>, _> =
                stmt.query_map([], |row| row.get::<_, String>(0))?.collect();
            rows
        })
        .await??;

        rows.into_iter()
            .map(|json| serde_json::from_str(&json).map_err(SqliteError::Serde))
            .collect()
    }

    /// Loads events inserted after `after_idx` (exclusive) using the internal index.
    ///
    /// Useful for incremental checkpointing.
    pub async fn load_since_idx(&self, after_idx: i64) -> Result<Vec<Event>, SqliteError> {
        let conn = Arc::clone(&self.conn);
        let rows = tokio::task::spawn_blocking(move || -> Result<Vec<String>, rusqlite::Error> {
            let conn = conn.blocking_lock();
            let mut stmt = conn.prepare("SELECT data FROM events WHERE idx > ?1 ORDER BY idx")?;
            let rows: Result<Vec<String>, _> = stmt
                .query_map(params![after_idx], |row| row.get::<_, String>(0))?
                .collect();
            rows
        })
        .await??;

        rows.into_iter()
            .map(|json| serde_json::from_str(&json).map_err(SqliteError::Serde))
            .collect()
    }

    /// Returns the highest row index, or `0` if empty.
    ///
    /// Bookmark this position for subsequent [`load_since_idx`](Self::load_since_idx) calls.
    pub async fn current_idx(&self) -> Result<i64, SqliteError> {
        let conn = Arc::clone(&self.conn);
        let idx = tokio::task::spawn_blocking(move || -> Result<i64, rusqlite::Error> {
            let conn = conn.blocking_lock();
            conn.query_row("SELECT COALESCE(MAX(idx), 0) FROM events", [], |row| {
                row.get::<_, i64>(0)
            })
        })
        .await??;
        Ok(idx)
    }
}

// ── SqliteExporter ────────────────────────────────────────────────────────────

/// Streams live events to a SQLite database.
///
/// Pair with [`crate::observability::BusObserver`] for zero-configuration persistent event logging.
pub struct SqliteExporter {
    store: SqliteEventStore,
}

impl SqliteExporter {
    /// Open (or create) the SQLite database at `path` and prepare for streaming.
    pub async fn new(path: impl Into<PathBuf>) -> Result<Self, SqliteError> {
        Ok(Self {
            store: SqliteEventStore::new(path).await?,
        })
    }

    /// Access the underlying store for read operations (e.g., `load_all`).
    pub fn store(&self) -> &SqliteEventStore {
        &self.store
    }
}

#[async_trait]
impl ObservabilityExporter for SqliteExporter {
    async fn export(&self, event: &Event) -> Result<(), ObsError> {
        self.store
            .append(event)
            .await
            .map_err(|e| ObsError::Other(e.to_string()))
    }

    async fn flush(&self) -> Result<(), ObsError> {
        // WAL mode + synchronous=NORMAL means data is already on disk after each write.
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::EventBus;
    use crate::event::{kinds, Event};
    use serde_json::json;
    use tempfile::tempdir;

    #[tokio::test]
    async fn append_and_load_all() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("test.db");
        let store = SqliteEventStore::new(&db).await.unwrap();

        let e1 = Event::new(kinds::USER_MESSAGE, json!({"text": "hello"}));
        let e2 = Event::new(kinds::ASSISTANT_MESSAGE, json!({"content": "hi"}));
        store.append(&e1).await.unwrap();
        store.append(&e2).await.unwrap();

        let all = store.load_all().await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].kind, kinds::USER_MESSAGE);
        assert_eq!(all[1].kind, kinds::ASSISTANT_MESSAGE);
    }

    #[tokio::test]
    async fn duplicate_append_is_idempotent() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("test.db");
        let store = SqliteEventStore::new(&db).await.unwrap();

        let e = Event::new(kinds::USER_MESSAGE, json!({"text": "dupe"}));
        store.append(&e).await.unwrap();
        store.append(&e).await.unwrap(); // same ID — should be ignored

        let all = store.load_all().await.unwrap();
        assert_eq!(all.len(), 1);
    }

    #[tokio::test]
    async fn load_since_idx_returns_new_events() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("test.db");
        let store = SqliteEventStore::new(&db).await.unwrap();

        let e1 = Event::new(kinds::USER_MESSAGE, json!({}));
        let e2 = Event::new(kinds::USER_MESSAGE, json!({}));
        store.append(&e1).await.unwrap();

        let idx = store.current_idx().await.unwrap();
        store.append(&e2).await.unwrap();

        let new_events = store.load_since_idx(idx).await.unwrap();
        assert_eq!(new_events.len(), 1);
        assert_eq!(new_events[0].id, e2.id);
    }

    #[tokio::test]
    async fn restore_from_rebuilds_bus_log() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("test.db");
        let store = SqliteEventStore::new(&db).await.unwrap();

        // Persist some events.
        for i in 0..5u32 {
            store
                .append(&Event::new(kinds::USER_MESSAGE, json!({"i": i})))
                .await
                .unwrap();
        }

        // Restore into a fresh bus.
        let bus = EventBus::new();
        let saved = store.load_all().await.unwrap();
        assert_eq!(saved.len(), 5);
        bus.restore_from(saved).await;

        assert_eq!(bus.log_len().await, 5);
    }
}
