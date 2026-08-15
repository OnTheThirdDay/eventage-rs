//! Per-session event feed: ordered history plus live fan-out.
//!
//! A desktop app cannot assume its connection survives — the machine sleeps,
//! the browser throttles a background tab, the stream drops. So the feed
//! numbers every event and keeps the history, and a reconnecting client asks
//! for everything after the last sequence number it saw. That turns a dropped
//! connection into a gap of a few hundred milliseconds instead of a reload.

use crate::protocol::StudioEvent;
use eventage::event::kinds;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// How many events a live subscriber may fall behind before it is dropped and
/// told to resubscribe from its last sequence number.
const LIVE_BUFFER: usize = 4096;

pub struct EventFeed {
    /// Identifies this numbering.
    ///
    /// Sequence numbers are assigned per feed, and a feed is rebuilt whenever
    /// Studio restarts — so the same session can be numbered differently from
    /// one process to the next. A client resuming with `?after=` from an older
    /// numbering is asking for a slice that means something else now, and gets
    /// handed events it is already showing. It compares this first.
    generation: String,
    /// Sequence numbers are 1-based and equal to the index + 1, which is what
    /// makes `since` a slice rather than a search.
    history: Mutex<Vec<Arc<StudioEvent>>>,
    /// Ids already held.
    ///
    /// A bus event that reaches the feed twice would be given two sequence
    /// numbers, and every consumer would treat them as two separate things —
    /// the transcript would show the message twice, and no amount of care on
    /// the client could tell them apart. Refusing the second one here is the
    /// only place the invariant can actually be held.
    seen: Mutex<HashSet<String>>,
    tx: broadcast::Sender<Arc<StudioEvent>>,
    closed: AtomicBool,
}

impl Default for EventFeed {
    fn default() -> Self {
        Self::new()
    }
}

impl EventFeed {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(LIVE_BUFFER);
        Self {
            generation: uuid::Uuid::new_v4().to_string(),
            history: Mutex::new(Vec::new()),
            seen: Mutex::new(HashSet::new()),
            tx,
            closed: AtomicBool::new(false),
        }
    }

    /// Which numbering these sequence numbers belong to.
    pub fn generation(&self) -> &str {
        &self.generation
    }

    /// Append an event, assigning its sequence number.
    ///
    /// The number is assigned and the event broadcast under the same lock, so
    /// a live subscriber can never see sequence 8 before sequence 7.
    pub fn push(&self, mut event: StudioEvent) -> Arc<StudioEvent> {
        let mut history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());

        // Already held: hand back the copy we have rather than filing a
        // second one under a new sequence number.
        if !seen.insert(event.id.clone()) {
            if let Some(existing) = history.iter().find(|e| e.id == event.id) {
                return Arc::clone(existing);
            }
        }

        event.seq = history.len() as u64 + 1;
        let event = Arc::new(event);
        history.push(Arc::clone(&event));
        // Fails only when nobody is listening, which is normal.
        let _ = self.tx.send(Arc::clone(&event));
        event
    }

    /// Every event after `seq` (pass 0 for the whole history).
    pub fn since(&self, seq: u64) -> Vec<Arc<StudioEvent>> {
        let history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        let start = (seq as usize).min(history.len());
        history[start..].to_vec()
    }

    /// Completed turns, not counting any a rewind undid.
    ///
    /// Rolled-back events stay in the feed — the trace panel shows them, and
    /// that is the point of keeping a rejected branch rather than deleting
    /// one. They must not still be counted as turns, or the session would
    /// claim work it has been told to forget.
    pub fn count_turns(&self) -> usize {
        let history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        let mut rejected: HashSet<&str> = HashSet::new();
        for event in history.iter() {
            if event.kind == kinds::SYSTEM_ROLLBACK {
                if let Some(ids) = event
                    .payload
                    .get("rejected_event_ids")
                    .and_then(|v| v.as_array())
                {
                    rejected.extend(ids.iter().filter_map(|id| id.as_str()));
                }
            }
        }
        history
            .iter()
            .filter(|e| e.kind == kinds::AGENT_CYCLE_END)
            .filter(|e| !rejected.contains(e.id.as_str()))
            .count()
    }

    /// The first user message, trimmed — used as the session title.
    pub fn first_user_text(&self) -> Option<String> {
        let history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        history
            .iter()
            .find(|e| e.kind == kinds::USER_MESSAGE)
            .and_then(|e| first_text_of(&e.payload))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<StudioEvent>> {
        self.tx.subscribe()
    }

    /// Mark the feed finished; live streams end cleanly rather than hanging.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

/// Pull display text out of a `user.message`, whichever shape it has.
///
/// Prompts reach the bus as multimodal `parts`, but a plain `text` payload is
/// just as valid and is what the ACP backend produces, so both are read.
fn first_text_of(payload: &serde_json::Value) -> Option<String> {
    let trim = |s: &str| -> Option<String> {
        let text = s.trim();
        (!text.is_empty()).then(|| text.chars().take(80).collect())
    };

    if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
        if let Some(found) = trim(text) {
            return Some(found);
        }
    }
    if let Some(text) = payload.get("content").and_then(|v| v.as_str()) {
        if let Some(found) = trim(text) {
            return Some(found);
        }
    }
    payload
        .get("parts")
        .and_then(|v| v.as_array())?
        .iter()
        .find_map(|part| part.get("text").and_then(|v| v.as_str()).and_then(trim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ev(kind: &str) -> StudioEvent {
        StudioEvent::studio(kind, json!({}))
    }

    #[test]
    fn sequence_numbers_start_at_one_and_never_repeat() {
        let feed = EventFeed::new();
        let a = feed.push(ev("a"));
        let b = feed.push(ev("b"));
        assert_eq!((a.seq, b.seq), (1, 2));
    }

    #[test]
    fn since_resumes_exactly_where_a_client_left_off() {
        let feed = EventFeed::new();
        for i in 0..5 {
            feed.push(ev(&format!("k{i}")));
        }
        // A client that has seen through seq 3 must get 4 and 5, nothing else.
        let resumed = feed.since(3);
        assert_eq!(resumed.len(), 2);
        assert_eq!(resumed[0].seq, 4);
        assert_eq!(resumed[1].seq, 5);

        // A fresh client gets everything; a caught-up client gets nothing.
        assert_eq!(feed.since(0).len(), 5);
        assert!(feed.since(5).is_empty());
    }

    #[test]
    fn the_same_event_cannot_be_filed_twice() {
        // Two sequence numbers for one event is indistinguishable, downstream,
        // from two events — the transcript renders the message twice and no
        // client-side care can tell them apart.
        let feed = EventFeed::new();
        let mut event = StudioEvent::studio(kinds::ASSISTANT_MESSAGE, json!({ "content": "hi" }));
        event.id = "same-id".into();

        let first = feed.push(event.clone());
        let second = feed.push(event);

        assert_eq!(
            first.seq, second.seq,
            "the second push must not get a new seq"
        );
        assert_eq!(feed.since(0).len(), 1, "and must not be stored again");
    }

    #[test]
    fn distinct_events_are_still_distinct() {
        let feed = EventFeed::new();
        feed.push(StudioEvent::studio("a", json!({})));
        feed.push(StudioEvent::studio("b", json!({})));
        assert_eq!(feed.since(0).len(), 2);
    }

    #[test]
    fn a_client_ahead_of_the_feed_is_not_a_panic() {
        // Can happen if the backend restarted while the tab stayed open.
        let feed = EventFeed::new();
        feed.push(ev("only"));
        assert!(feed.since(999).is_empty());
    }

    #[tokio::test]
    async fn live_subscribers_see_events_in_order() {
        let feed = EventFeed::new();
        let mut rx = feed.subscribe();
        feed.push(ev("first"));
        feed.push(ev("second"));

        assert_eq!(rx.recv().await.unwrap().kind, "first");
        assert_eq!(rx.recv().await.unwrap().kind, "second");
    }

    #[test]
    fn turns_do_not_count_work_a_rewind_undid() {
        let feed = EventFeed::new();
        let first = feed.push(ev(kinds::AGENT_CYCLE_END));
        let second = feed.push(ev(kinds::AGENT_CYCLE_END));
        assert_eq!(feed.count_turns(), 2);

        // A rewind seals the second turn into a rejected branch. The events
        // stay in the feed for the trace, but the turn no longer counts.
        feed.push(StudioEvent::studio(
            kinds::SYSTEM_ROLLBACK,
            json!({ "rejected_event_ids": [second.id] }),
        ));
        assert_eq!(feed.count_turns(), 1);
        assert!(feed.since(0).iter().any(|e| e.id == first.id));
        assert!(
            feed.since(0).iter().any(|e| e.id == second.id),
            "rolled-back events must remain visible in the trace"
        );
    }

    #[test]
    fn a_title_is_found_in_a_multimodal_prompt() {
        // Prompts reach the bus as parts, not as a bare `text` field: the
        // title must come from the first text part, not from nothing.
        let feed = EventFeed::new();
        feed.push(StudioEvent::studio(
            kinds::USER_MESSAGE,
            json!({ "parts": [
                { "type": "image", "source": { "kind": "url", "url": "x" } },
                { "type": "text", "text": "  fix the parser  " }
            ] }),
        ));
        assert_eq!(feed.first_user_text().as_deref(), Some("fix the parser"));
    }

    #[test]
    fn an_empty_prompt_has_no_title_rather_than_a_blank_one() {
        let feed = EventFeed::new();
        feed.push(StudioEvent::studio(
            kinds::USER_MESSAGE,
            json!({ "text": "   ", "parts": [] }),
        ));
        assert_eq!(feed.first_user_text(), None);
    }

    #[test]
    fn the_title_comes_from_the_first_user_message() {
        let feed = EventFeed::new();
        feed.push(StudioEvent::studio("agent.cycle.start", json!({})));
        feed.push(StudioEvent::studio(
            kinds::USER_MESSAGE,
            json!({ "text": "  fix the failing test  " }),
        ));
        feed.push(StudioEvent::studio(
            kinds::USER_MESSAGE,
            json!({ "text": "and now the other one" }),
        ));
        assert_eq!(
            feed.first_user_text().as_deref(),
            Some("fix the failing test")
        );
    }
}
