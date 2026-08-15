use super::error::ObsError;
use super::exporter::ObservabilityExporter;
use crate::event::Event;
use async_trait::async_trait;
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
/// # use eventage::observability::JsonlExporter;
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
