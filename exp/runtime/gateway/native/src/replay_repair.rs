//! The one repair the waterfall performs on a caller's replayed input.
//!
//! OpenAI binds a Responses reasoning item's `encrypted_content` to the
//! organization (Azure: the tenant) that sealed it and refuses every other
//! replay with `invalid_encrypted_content`. The waterfall re-dials the same
//! rung once with those items stripped and remembers the stripped payload
//! for the rest of the request's ladder on that rung; this module owns the
//! strip, the caller-facing disclosure header, the operator line, and the
//! per-worker memory that lets a LATER request of the same conversation skip
//! the refused dial altogether.
//!
//! A stateless caller keeps the foreign items in its history, so without
//! memory every later turn of that conversation paid one refused dial before
//! it was served (production, 2026-09-15: about 74 refused dials a minute,
//! 82% of one tenant's agent turns, every one a fast unbilled 400 that still
//! counted against the provider's request rate). The memory holds only
//! digests of refused payloads, salted by the caller's stable identity as
//! admission names it (organization and identity ids; the continuation
//! store's own key does not exist for a stateless turn, a prefix-derived
//! conversation key would merge parallel sessions of one agent template, and
//! the request's bearer is the front's ephemeral exchanged token in a hosted
//! worker), for a bounded time and count, per worker: a conversation
//! converges after each worker has repaired it once, and a shared store
//! would buy nothing worth its roll hazards.
//!
//! Telemetry: the refused dial and its repaired successor each reserve and
//! settle a separate physical attempt. The repair is additionally disclosed by:
//! the `x-gateway-replay-repair` header on HTTP responses (the WebSocket
//! Responses transport carries no response headers), the data-plane counters
//! `encrypted_reasoning_stripped` (a refusal repaired on this attempt) and
//! `encrypted_reasoning_stripped_proactive` (stripped from memory before the
//! first dial; a same-request re-dial reusing the rung's stripped payload is
//! logged as `remembered` and counted by neither), and one content-free
//! operator line naming the mode, each written only once the dial has
//! actually opened.

use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::errors::Failure;
use crate::metrics::METRICS;
use crate::waterfall::DeploymentWire;

/// Response header disclosing a data-plane repair of the caller's replayed
/// input on the attempt that served.
pub const REPLAY_REPAIR_HEADER: &str = "x-gateway-replay-repair";

/// The one repair the waterfall performs: the replayed reasoning items whose
/// `encrypted_content` the rung refused were stripped and the rung re-dialed.
pub const ENCRYPTED_REASONING_STRIPPED: &str = "encrypted_reasoning_stripped";

/// How long a refused payload stays remembered after its last replay.
const MEMORY_TTL: Duration = Duration::from_secs(30 * 60);

/// Most remembered payload digests per worker (about 48 bytes each).
const MEMORY_CAPACITY: usize = 100_000;

/// The fixed head and verdict of OpenAI's refusal sentence, which quotes the
/// refused payload as its first and last characters around an ellipsis.
const REFUSAL_SENTENCE_HEAD: &str = "The encrypted content ";
const REFUSAL_SENTENCE_VERDICT: &str = " could not be verified";

/// The disclosure header pairs of one served attempt; empty when the input
/// reached the provider exactly as replayed.
pub fn replay_repair_headers(encrypted_reasoning_stripped: bool) -> Vec<(String, String)> {
    if encrypted_reasoning_stripped {
        vec![(
            REPLAY_REPAIR_HEADER.to_string(),
            ENCRYPTED_REASONING_STRIPPED.to_string(),
        )]
    } else {
        Vec::new()
    }
}

/// The per-worker memory of refused encrypted reasoning payloads.
pub static MEMORY: LazyLock<ReplayRepairMemory> = LazyLock::new(ReplayRepairMemory::default);

/// Digests of refused payloads, each with the instant it was last replayed;
/// an entry expires [`MEMORY_TTL`] after that, and the oldest insertions go
/// first once [`MEMORY_CAPACITY`] is reached.
#[derive(Default)]
pub struct ReplayRepairMemory {
    state: Mutex<MemoryState>,
}

/// One remembered digest: when it was last replayed, and the sequence number
/// of the insertion that placed it in the order queue.
#[derive(Clone, Copy)]
struct Remembered {
    last_seen: Instant,
    sequence: u64,
}

#[derive(Default)]
struct MemoryState {
    entries: HashMap<[u8; 32], Remembered>,
    /// Insertion order as `(digest, sequence)`. A digest forgotten on recall
    /// and learned again gets a new sequence, so its stale queue entry is
    /// recognized and skipped by the capacity sweep instead of evicting the
    /// fresh one.
    insertion_order: VecDeque<([u8; 32], u64)>,
    next_sequence: u64,
}

impl ReplayRepairMemory {
    /// Remember these refused payload digests as of now.
    pub fn remember(&self, digests: impl IntoIterator<Item = [u8; 32]>) {
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for digest in digests {
            match state.entries.get_mut(&digest) {
                Some(entry) => entry.last_seen = now,
                None => {
                    let sequence = state.next_sequence;
                    state.next_sequence += 1;
                    state.entries.insert(
                        digest,
                        Remembered {
                            last_seen: now,
                            sequence,
                        },
                    );
                    state.insertion_order.push_back((digest, sequence));
                }
            }
        }
        // Evict oldest insertions first. A queue entry whose sequence the
        // map no longer holds is stale (its digest expired on recall, or was
        // learned again under a newer sequence) and is popped for free; the
        // map alone is what the capacity bounds.
        while state.entries.len() > MEMORY_CAPACITY {
            let Some((oldest, sequence)) = state.insertion_order.pop_front() else {
                break;
            };
            if state
                .entries
                .get(&oldest)
                .is_some_and(|entry| entry.sequence == sequence)
            {
                state.entries.remove(&oldest);
            }
        }
        while state.insertion_order.len() > 2 * MEMORY_CAPACITY {
            let Some((oldest, sequence)) = state.insertion_order.pop_front() else {
                break;
            };
            if state
                .entries
                .get(&oldest)
                .is_some_and(|entry| entry.sequence == sequence)
            {
                state.entries.remove(&oldest);
            }
        }
    }

    /// Whether this digest is remembered and unexpired; a hit refreshes it.
    pub fn recall(&self, digest: &[u8; 32]) -> bool {
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match state.entries.get_mut(digest) {
            Some(entry) if now.duration_since(entry.last_seen) <= MEMORY_TTL => {
                entry.last_seen = now;
                true
            }
            Some(_) => {
                // Expired: drop the entry in constant time and leave its
                // place in the insertion order to the capacity sweep, which
                // recognizes the stale sequence.
                state.entries.remove(digest);
                false
            }
            None => false,
        }
    }

    /// How many digests are held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entries
            .len()
    }

    /// Whether nothing is held.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The memory key of one payload for one caller: a digest, never the payload.
pub fn payload_digest(scope: &str, encrypted_content: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(scope.as_bytes());
    hasher.update([0u8]);
    hasher.update(encrypted_content.as_bytes());
    hasher.finalize().into()
}

/// The `(head, tail)` OpenAI quoted for the refused payload ("The encrypted
/// content rsn_...hA== could not be verified. Reason: ..."): a payload
/// matches when it starts with the head and ends with the tail. A short
/// payload is quoted whole, so head and tail are then the whole token.
pub fn refused_payload_hint(detail: &str) -> Option<(String, String)> {
    let start = detail.find(REFUSAL_SENTENCE_HEAD)? + REFUSAL_SENTENCE_HEAD.len();
    let rest = &detail[start..];
    let end = rest.find(REFUSAL_SENTENCE_VERDICT)?;
    let quoted = rest[..end].trim();
    if quoted.is_empty() {
        return None;
    }
    Some(match quoted.split_once("...") {
        Some((head, tail)) => (head.to_string(), tail.to_string()),
        None => (quoted.to_string(), quoted.to_string()),
    })
}

/// Whether one payload is the one a refusal hint quoted.
fn matches_hint(encrypted_content: &str, (head, tail): &(String, String)) -> bool {
    encrypted_content.len() >= head.len().max(tail.len())
        && encrypted_content.starts_with(head)
        && encrypted_content.ends_with(tail)
}

/// Every replayed `encrypted_content` in the payload's input, in order.
fn encrypted_payloads(payload: &Value) -> Vec<&str> {
    payload
        .get("input")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
                .filter_map(|item| item.get("encrypted_content").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default()
}

/// The Responses payload with every replayed reasoning item that carries
/// `encrypted_content` removed, or `None` when there is nothing to strip.
pub fn without_encrypted_reasoning(payload: &Value) -> Option<Value> {
    without_encrypted_reasoning_where(payload, |_| true)
}

/// The Responses payload with the replayed reasoning items whose
/// `encrypted_content` `strip` selects removed, or `None` when it selects
/// nothing.
///
/// OpenAI binds an encrypted reasoning payload to the organization (Azure:
/// the tenant) that sealed it and refuses every other with
/// `invalid_encrypted_content`, so a conversation whose earlier turn another
/// lane served, or whose history the caller assembled from another account,
/// cannot replay those items here. The item is optional on replay: the model
/// resumes from the visible message, tool-call, and tool-result items, so
/// dropping it trades hidden reasoning continuity for a served turn.
///
/// The assistant items a stripped reasoning item governed (the function and
/// custom tool calls and the assistant message of that same provider turn,
/// up to the next user or tool-result item) lose their provider `id` as
/// well: OpenAI ties a replayed item id to the reasoning item of its turn
/// and refuses the id without it ("was provided without its required
/// 'reasoning' item"), while an id-less item is accepted as caller-authored.
/// Every kept item keeps its position; nothing outside `input` changes.
pub fn without_encrypted_reasoning_where(
    payload: &Value,
    mut strip: impl FnMut(&str) -> bool,
) -> Option<Value> {
    let mut repaired = payload.clone();
    let items = repaired.get_mut("input")?.as_array_mut()?;
    let mut keep: Vec<bool> = Vec::with_capacity(items.len());
    let mut governed = false;
    for item in items.iter_mut() {
        let kind = item.get("type").and_then(Value::as_str);
        let role = item.get("role").and_then(Value::as_str);
        let encrypted = (kind == Some("reasoning"))
            .then(|| item.get("encrypted_content").and_then(Value::as_str))
            .flatten();
        if encrypted.is_some_and(&mut strip) {
            keep.push(false);
            governed = true;
            continue;
        }
        let same_turn_output = matches!(kind, Some("function_call") | Some("custom_tool_call"))
            || (matches!(kind, Some("message") | None) && role == Some("assistant"))
            || kind == Some("reasoning");
        if governed && same_turn_output {
            if let Some(object) = item.as_object_mut() {
                object.remove("id");
            }
        } else {
            governed = false;
        }
        keep.push(true);
    }
    if keep.iter().all(|kept| *kept) {
        return None;
    }
    let mut index = 0;
    items.retain(|_| {
        let kept = keep[index];
        index += 1;
        kept
    });
    Some(repaired)
}

/// The repair state of one physical attempt on one rung.
///
/// `begin` applies what is already known: the payload this request stripped
/// on an earlier dial of the rung (`repaired`), else the payloads the worker
/// remembers as refused for this caller. `repair_after` decides one reactive
/// re-dial when the provider still refuses, `dial_opened` notes a dial the
/// provider answered, and the drop emits the attempt's one record once its
/// fate is decided.
pub(crate) struct AttemptRepair<'a> {
    wire: &'a DeploymentWire,
    scope: Option<&'a str>,
    request_id: &'a str,
    repaired: &'a mut Option<Value>,
    stripped: bool,
    proactive: bool,
    reactive: bool,
    /// Whether the most recent dial of this attempt opened (a 2xx). A repair
    /// clears it until the re-dial opens, so the one record this attempt
    /// emits on drop describes the dial that actually carried the repair.
    last_dial_opened: bool,
    /// Payloads stripped from memory before the first dial, and after a
    /// refusal, so the operator line tells a mixed attempt from a miss.
    proactive_strips: usize,
    reactive_strips: usize,
}

impl<'a> AttemptRepair<'a> {
    /// Prepare the first dial of this attempt.
    pub(crate) fn begin(
        wire: &'a DeploymentWire,
        scope: Option<&'a str>,
        repaired: &'a mut Option<Value>,
        request_id: &'a str,
    ) -> Self {
        let mut repair = Self {
            wire,
            scope,
            request_id,
            repaired,
            stripped: false,
            proactive: false,
            reactive: false,
            last_dial_opened: false,
            proactive_strips: 0,
            reactive_strips: 0,
        };
        if repair.repaired.is_some() {
            repair.stripped = true;
        } else if let (None, Some(scope)) = (&wire.upstream_body, scope) {
            let remembered = |content: &str| MEMORY.recall(&payload_digest(scope, content));
            if let Some(stripped) =
                without_encrypted_reasoning_where(&wire.upstream_payload, remembered)
            {
                repair.proactive_strips = encrypted_payloads(&wire.upstream_payload).len()
                    - encrypted_payloads(&stripped).len();
                *repair.repaired = Some(stripped);
                repair.stripped = true;
                repair.proactive = true;
            }
        }
        repair
    }

    /// The payload to dial now.
    pub(crate) fn payload(&self) -> &Value {
        self.repaired
            .as_ref()
            .unwrap_or(&self.wire.upstream_payload)
    }

    /// The pre-serialized body to dial now (body-signing dialects only, and
    /// never once the payload was repaired).
    pub(crate) fn raw_body(&self) -> Option<&str> {
        if self.repaired.is_some() {
            None
        } else {
            self.wire.upstream_body.as_deref()
        }
    }

    /// Whether the dialed payload had encrypted reasoning stripped.
    pub(crate) fn stripped(&self) -> bool {
        self.stripped
    }

    /// Decide the reactive re-dial after a refused open: `true` when the
    /// provider refused replayed encrypted reasoning, this attempt has not
    /// repaired reactively yet, and the payload still carried something to
    /// strip. The refused payloads (the one OpenAI quoted, else every one
    /// still present) are remembered for later requests of this caller.
    pub(crate) fn repair_after(&mut self, failure: &Failure) -> bool {
        if self.reactive
            || !failure.encrypted_reasoning_rejected
            || self.wire.upstream_body.is_some()
        {
            return false;
        }
        let Some(stripped) = without_encrypted_reasoning(self.payload()) else {
            return false;
        };
        let (present_count, digests) = {
            let present = encrypted_payloads(self.payload());
            let hint = failure
                .provider_detail
                .as_deref()
                .and_then(refused_payload_hint);
            let quoted: Vec<&str> = match &hint {
                Some(hint) => present
                    .iter()
                    .copied()
                    .filter(|content| matches_hint(content, hint))
                    .collect(),
                None => Vec::new(),
            };
            let refused = if quoted.is_empty() { &present } else { &quoted };
            let digests: Vec<[u8; 32]> = match self.scope {
                Some(scope) => refused
                    .iter()
                    .map(|content| payload_digest(scope, content))
                    .collect(),
                None => Vec::new(),
            };
            (present.len(), digests)
        };
        if !digests.is_empty() {
            MEMORY.remember(digests);
        }
        self.reactive_strips = present_count;
        *self.repaired = Some(stripped);
        self.stripped = true;
        self.reactive = true;
        self.last_dial_opened = false;
        true
    }

    /// Keep a repaired successor's disclosure attached to its own physical attempt.
    pub(crate) fn mark_reactive_successor(&mut self) {
        self.reactive = true;
        self.reactive_strips = encrypted_payloads(&self.wire.upstream_payload)
            .len()
            .saturating_sub(encrypted_payloads(self.payload()).len());
    }

    /// Note that the current dial opened (the provider answered 2xx).
    pub(crate) fn dial_opened(&mut self) {
        self.last_dial_opened = true;
    }
}

impl Drop for AttemptRepair<'_> {
    /// Emit ONE record per attempt, once its fate is decided: counted and
    /// logged only when a repaired dial actually opened, so a refused re-dial
    /// is a failure, not a repair, and a mixed attempt (remembered payloads
    /// stripped, then a new one refused and repaired) is one reactive record
    /// carrying both strip counts rather than two records.
    fn drop(&mut self) {
        if !self.stripped || !self.last_dial_opened {
            return;
        }
        // A same-request re-dial that reuses the rung's stripped payload
        // recalled nothing and may repeat under throttle: logged as
        // `remembered`, counted by neither counter.
        let mode = if self.reactive {
            METRICS.record_encrypted_reasoning_stripped();
            "reactive"
        } else if self.proactive {
            METRICS.record_encrypted_reasoning_stripped_proactive();
            "proactive"
        } else {
            "remembered"
        };
        let line = json!({
            "event": "encrypted_reasoning_stripped",
            "mode": mode,
            "proactive_strips": self.proactive_strips,
            "reactive_strips": self.reactive_strips,
            "request_id": self.request_id,
            "provider": self.wire.provider,
            "deployment_id": self.wire.deployment_id,
        });
        eprintln!("exp-gateway-native: {line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripping_encrypted_reasoning_removes_only_those_items_and_keeps_order() {
        let payload = json!({
            "model": "gpt-test",
            "store": false,
            "input": [
                {"role": "user", "content": "plan"},
                // A call from a turn whose reasoning was never replayed keeps its id.
                {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": "ls", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_0", "output": "."},
                {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": "rsn_foreign=="},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "reasoning", "id": "rs_b", "summary": [], "encrypted_content": "rsn_foreign_2=="},
                {"type": "message", "id": "msg_9", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "done"}]},
                // A reasoning item replayed by id alone is not an encrypted payload.
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "reasoning", "summary": [], "encrypted_content": null},
                {"role": "user", "content": "continue"},
                {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "exec", "arguments": "{}"},
            ],
            "include": ["reasoning.encrypted_content"],
        });
        let repaired = without_encrypted_reasoning(&payload).expect("two items to strip");
        let kept: Vec<&Value> = repaired["input"]
            .as_array()
            .expect("input")
            .iter()
            .collect();
        assert_eq!(kept.len(), 10);
        assert!(kept
            .iter()
            .all(|item| item.get("encrypted_content").is_none_or(Value::is_null)));
        assert_eq!(kept[0]["content"], "plan");
        // Governed by no stripped reasoning item: the id survives.
        assert_eq!(kept[1]["id"], "fc_0");
        // The call and the message of the stripped turns lose their ids and
        // keep everything else.
        assert_eq!(kept[3]["type"], "function_call");
        assert!(kept[3].get("id").is_none());
        assert_eq!(kept[3]["call_id"], "call_1");
        assert_eq!(kept[5]["type"], "message");
        assert!(kept[5].get("id").is_none());
        assert_eq!(kept[5]["status"], "completed");
        // The by-id reasoning item of the same turn also loses its id; the
        // null-content one is kept as sent.
        assert!(kept[6].get("id").is_none());
        assert_eq!(kept[7]["encrypted_content"], Value::Null);
        // A later turn with no stripped reasoning keeps its ids.
        assert_eq!(kept[8]["content"], "continue");
        assert_eq!(kept[9]["id"], "fc_2");
        // Everything beside the input is untouched.
        assert_eq!(repaired["include"], payload["include"]);
        assert_eq!(repaired["store"], false);

        // Nothing to strip: no repair, so the provider's verdict surfaces.
        assert!(without_encrypted_reasoning(&repaired).is_none());
        assert!(without_encrypted_reasoning(&json!({"model": "m", "input": "text"})).is_none());
        assert!(without_encrypted_reasoning(&json!({"model": "m", "messages": []})).is_none());
    }

    #[test]
    fn the_refusal_hint_names_the_quoted_payload() {
        let hint = refused_payload_hint(
            "The encrypted content rsn_...hA== could not be verified. Reason: Encrypted content could not be decrypted or parsed. trace_id: 9f3c",
        )
        .expect("hint");
        assert_eq!(hint, ("rsn_".to_string(), "hA==".to_string()));
        assert!(matches_hint("rsn_sealed_elsewhere_hA==", &hint));
        assert!(!matches_hint("gAAA_local_hA==", &hint));
        assert!(!matches_hint("rsn_sealed_elsewhere==", &hint));
        // A short payload is quoted whole.
        let whole =
            refused_payload_hint("The encrypted content abc could not be verified.").expect("hint");
        assert!(matches_hint("abc", &whole));
        assert!(!matches_hint("abcd", &whole));
        assert!(refused_payload_hint("Invalid value for 'input[1].id'.").is_none());
        assert!(refused_payload_hint("The encrypted content  could not be verified").is_none());
    }

    #[test]
    fn selective_stripping_removes_only_the_selected_payloads() {
        let payload = json!({
            "model": "gpt-test",
            "input": [
                {"role": "user", "content": "plan"},
                {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": "rsn_foreign_hA=="},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "reasoning", "id": "rs_b", "summary": [], "encrypted_content": "gAAA_local=="},
                {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "exec", "arguments": "{}"},
            ],
        });
        let repaired =
            without_encrypted_reasoning_where(&payload, |content| content.starts_with("rsn_"))
                .expect("one item to strip");
        let kept = repaired["input"].as_array().expect("input");
        assert_eq!(kept.len(), 5);
        assert!(
            kept[1].get("id").is_none(),
            "the governed call loses its id"
        );
        assert_eq!(kept[3]["encrypted_content"], "gAAA_local==");
        assert_eq!(
            kept[4]["id"], "fc_2",
            "a turn whose reasoning stayed keeps its ids"
        );
        assert!(without_encrypted_reasoning_where(&payload, |_| false).is_none());
    }

    #[test]
    fn the_memory_recalls_within_its_window_and_refreshes_on_a_hit() {
        let memory = ReplayRepairMemory::default();
        let one = payload_digest("key-a", "rsn_one==");
        let same_content_other_caller = payload_digest("key-b", "rsn_one==");
        assert_ne!(one, same_content_other_caller);
        assert!(!memory.recall(&one));
        memory.remember([one]);
        assert!(memory.recall(&one));
        assert!(!memory.recall(&same_content_other_caller));
        assert_eq!(memory.len(), 1);
        // Re-remembering an entry neither duplicates nor reorders it.
        memory.remember([one]);
        assert_eq!(memory.len(), 1);
        // An expired entry is forgotten on recall.
        {
            let mut state = memory.state.lock().expect("lock");
            state.entries.get_mut(&one).expect("entry").last_seen =
                Instant::now() - MEMORY_TTL - Duration::from_secs(1);
        }
        assert!(!memory.recall(&one));
        assert!(memory.is_empty());
        // An expired digest is learned again and bounded like any other.
        memory.remember([one]);
        assert!(memory.recall(&one));
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn the_memory_evicts_its_oldest_insertion_at_capacity() {
        let memory = ReplayRepairMemory::default();
        let digests: Vec<[u8; 32]> = (0..=MEMORY_CAPACITY)
            .map(|index| payload_digest("key", &index.to_string()))
            .collect();
        memory.remember(digests.iter().copied());
        assert_eq!(memory.len(), MEMORY_CAPACITY);
        assert!(
            !memory.recall(&digests[0]),
            "the first insertion went first"
        );
        assert!(memory.recall(&digests[MEMORY_CAPACITY]));
    }

    #[test]
    fn a_relearned_digest_survives_the_sweep_of_its_stale_queue_entry() {
        let memory = ReplayRepairMemory::default();
        let relearned = payload_digest("key", "relearned");
        memory.remember([relearned]);
        // Expire it and forget it on recall: its first queue entry is now stale.
        {
            let mut state = memory.state.lock().expect("lock");
            state.entries.get_mut(&relearned).expect("entry").last_seen =
                Instant::now() - MEMORY_TTL - Duration::from_secs(1);
        }
        assert!(!memory.recall(&relearned));
        // Fill to one under capacity, learn it again (now the NEWEST live
        // insertion, at capacity), then add one more: the sweep pops the
        // stale entry first and must skip it, evicting the oldest live
        // insertion instead of the freshly relearned digest.
        let filler: Vec<[u8; 32]> = (0..MEMORY_CAPACITY - 1)
            .map(|index| payload_digest("key", &format!("filler-{index}")))
            .collect();
        memory.remember(filler.iter().copied());
        memory.remember([relearned]);
        assert_eq!(memory.len(), MEMORY_CAPACITY);
        let newest = payload_digest("key", "newest");
        memory.remember([newest]);
        assert_eq!(memory.len(), MEMORY_CAPACITY);
        assert!(memory.recall(&relearned), "the stale entry evicted nothing");
        assert!(
            !memory.recall(&filler[0]),
            "the oldest live insertion went instead"
        );
        assert!(memory.recall(&newest));
    }

    #[test]
    fn the_repair_header_appears_only_on_a_repaired_attempt() {
        assert!(replay_repair_headers(false).is_empty());
        assert_eq!(
            replay_repair_headers(true),
            vec![(
                "x-gateway-replay-repair".to_string(),
                "encrypted_reasoning_stripped".to_string(),
            )]
        );
    }
}
