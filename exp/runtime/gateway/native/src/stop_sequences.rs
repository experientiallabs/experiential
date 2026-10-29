//! Gateway-emulated stop sequences.
//!
//! The OpenAI Responses API has no `stop` field, so a caller's stop sequences
//! never reach that wire. Instead the admitted route entry carries the exact
//! sequences and this guard applies them to the decoded event stream: visible
//! text is cut at the first match, everything the model says afterwards is
//! discarded, and the stream terminates with [`Event::StoppedAtSequence`] so
//! the public encoders report the matched sequence the way a native provider
//! would (`finish_reason: stop` on Chat, `stop_reason: stop_sequence` on
//! Messages).
//!
//! A sequence may straddle delta boundaries, so the guard withholds the
//! shortest tail of text that is still a proper prefix of some sequence and
//! releases it once the next delta rules the match out. Only visible text is
//! inspected: reasoning, tool arguments, and refusals pass through untouched.
//! Nothing here logs request or completion text.

use crate::events::Event;

/// Which delta variant carried the withheld text, so it re-emits in kind.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TextShape {
    Plain,
    Provider { output_index: u32, item_id: String },
}

/// Stateful stop-sequence filter for one upstream relay.
#[derive(Debug)]
pub struct StopSequenceGuard {
    sequences: Vec<String>,
    /// Visible text not yet released: at most `max_len - 1` bytes, every one
    /// of them the start of some sequence.
    buffer: String,
    shape: Option<TextShape>,
    matched: Option<String>,
}

impl StopSequenceGuard {
    /// Build a guard for the caller's sequences; `None` when there is nothing
    /// to enforce, so the relay skips the filter entirely.
    pub fn new<I, S>(sequences: I) -> Option<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let sequences: Vec<String> = sequences
            .into_iter()
            .map(Into::into)
            .filter(|sequence| !sequence.is_empty())
            .collect();
        if sequences.is_empty() {
            return None;
        }
        Some(Self {
            sequences,
            buffer: String::new(),
            shape: None,
            matched: None,
        })
    }

    /// The sequence that ended the stream, once one has matched.
    #[cfg(test)]
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// Filter one decoded event into zero or more outward events.
    pub fn filter(&mut self, event: Event) -> Vec<Event> {
        if let Some(sequence) = self.matched.clone() {
            return match event {
                // Everything the model said after the match is discarded.
                Event::TextDelta(_) | Event::ProviderTextDelta { .. } => Vec::new(),
                // The model's own ending is irrelevant: the caller's stop
                // sequence ended this turn. A provider failure still fails.
                Event::Completed | Event::Incomplete | Event::PausedTurn => {
                    vec![Event::StoppedAtSequence(sequence)]
                }
                other => vec![other],
            };
        }
        match event {
            Event::TextDelta(text) => self.push_text(TextShape::Plain, &text),
            Event::ProviderTextDelta {
                output_index,
                item_id,
                delta,
            } => self.push_text(
                TextShape::Provider {
                    output_index,
                    item_id,
                },
                &delta,
            ),
            // Item boundaries and terminals close the text they follow: the
            // withheld tail can no longer complete a sequence across them, and
            // a match held open for a longer candidate commits as it stands.
            Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ToolCallStarted { .. }
            | Event::ServerToolUseStarted { .. }
            | Event::HostedToolItemStarted { .. }
            | Event::Completed
            | Event::Incomplete
            | Event::PausedTurn
            | Event::Failed(_) => {
                let mut events = self.finish_text();
                if let Some(sequence) = self.matched.clone() {
                    events.push(match event {
                        Event::Completed | Event::Incomplete | Event::PausedTurn => {
                            Event::StoppedAtSequence(sequence)
                        }
                        other => other,
                    });
                } else {
                    events.push(event);
                }
                events
            }
            other => vec![other],
        }
    }

    fn push_text(&mut self, shape: TextShape, text: &str) -> Vec<Event> {
        let mut events = Vec::new();
        if self.shape.as_ref().is_some_and(|current| *current != shape) {
            events.extend(self.flush());
        }
        self.shape = Some(shape);
        self.buffer.push_str(text);
        if let Some((index, sequence)) = self.earliest_match() {
            if self.longer_match_still_possible(index, &sequence) {
                // "ab" has matched but "abc" may still complete: release the
                // text before the match and hold the rest, so the sequence
                // reported never depends on where the provider split its
                // deltas. A boundary or terminal resolves the hold.
                if index > 0 {
                    let released: String = self.buffer.drain(..index).collect();
                    events.push(self.delta(released));
                }
                return events;
            }
            events.extend(self.commit_match(index, sequence));
            return events;
        }
        let keep = self.pending_prefix_len();
        let release_len = self.buffer.len() - keep;
        if release_len > 0 {
            let released: String = self.buffer.drain(..release_len).collect();
            events.push(self.delta(released));
        }
        events
    }

    /// Whether some longer sequence could still match at `index` once more
    /// text arrives: the buffer tail from `index` is a proper prefix of it.
    fn longer_match_still_possible(&self, index: usize, matched: &str) -> bool {
        let tail = &self.buffer[index..];
        self.sequences.iter().any(|candidate| {
            candidate.len() > matched.len()
                && candidate.len() > tail.len()
                && candidate.starts_with(tail)
        })
    }

    /// Emit the visible text before `index`, record the match, and drop the
    /// rest of the buffer.
    fn commit_match(&mut self, index: usize, sequence: String) -> Vec<Event> {
        let mut events = Vec::new();
        let visible = self.buffer[..index].to_string();
        if !visible.is_empty() {
            events.push(self.delta(visible));
        }
        self.buffer.clear();
        self.matched = Some(sequence);
        events
    }

    /// Close the current text run at an item boundary or terminal: a held
    /// match (kept open only for a longer candidate) commits now, otherwise
    /// the withheld tail is released as text.
    fn finish_text(&mut self) -> Vec<Event> {
        if let Some((index, sequence)) = self.earliest_match() {
            return self.commit_match(index, sequence);
        }
        self.flush()
    }

    /// Earliest sequence occurrence in the buffer: lowest index wins, then
    /// the longest sequence starting there (a longer sequence is the more
    /// specific stop the caller asked for).
    fn earliest_match(&self) -> Option<(usize, String)> {
        let mut best: Option<(usize, &str)> = None;
        for sequence in &self.sequences {
            if let Some(index) = self.buffer.find(sequence.as_str()) {
                let better = match best {
                    None => true,
                    Some((best_index, best_sequence)) => {
                        index < best_index
                            || (index == best_index && sequence.len() > best_sequence.len())
                    }
                };
                if better {
                    best = Some((index, sequence.as_str()));
                }
            }
        }
        best.map(|(index, sequence)| (index, sequence.to_string()))
    }

    /// Length of the longest buffer tail that is a proper prefix of some
    /// sequence, i.e. text that could still complete a match.
    fn pending_prefix_len(&self) -> usize {
        let mut keep = 0usize;
        for sequence in &self.sequences {
            let bound = sequence.len().min(self.buffer.len());
            // Proper prefixes only: a full occurrence was already handled.
            let mut length = if bound == sequence.len() {
                bound - 1
            } else {
                bound
            };
            while length > 0 {
                if sequence.is_char_boundary(length) && self.buffer.ends_with(&sequence[..length]) {
                    keep = keep.max(length);
                    break;
                }
                length -= 1;
            }
        }
        keep
    }

    fn flush(&mut self) -> Vec<Event> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let text = std::mem::take(&mut self.buffer);
        vec![self.delta(text)]
    }

    fn delta(&self, text: String) -> Event {
        match &self.shape {
            Some(TextShape::Provider {
                output_index,
                item_id,
            }) => Event::ProviderTextDelta {
                output_index: *output_index,
                item_id: item_id.clone(),
                delta: text,
            },
            _ => Event::TextDelta(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::simplified_event;

    fn text(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::TextDelta(text) => Some(text.as_str()),
                Event::ProviderTextDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect()
    }

    fn stopped_at(event: Option<&Event>, expected: &str) -> bool {
        matches!(event, Some(Event::StoppedAtSequence(sequence)) if sequence == expected)
    }

    fn run(sequences: &[&str], deltas: &[&str], terminal: Event) -> Vec<Event> {
        let mut guard = StopSequenceGuard::new(sequences.iter().copied()).expect("sequences");
        let mut out = Vec::new();
        for delta in deltas {
            out.extend(guard.filter(Event::TextDelta((*delta).to_string())));
        }
        out.extend(guard.filter(terminal));
        out
    }

    #[test]
    fn empty_or_blank_sequences_build_no_guard() {
        assert!(StopSequenceGuard::new(Vec::<String>::new()).is_none());
        assert!(StopSequenceGuard::new(["", ""]).is_none());
        assert!(StopSequenceGuard::new(["", "END"]).is_some());
    }

    #[test]
    fn a_match_inside_one_delta_cuts_the_text_and_ends_with_the_sequence() {
        let out = run(
            &["</severity>"],
            &["<severity>high</severity>trailing"],
            Event::Completed,
        );
        assert_eq!(text(&out), "<severity>high");
        assert!(stopped_at(out.last(), "</severity>"));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_match_spanning_delta_boundaries_is_withheld_until_it_resolves() {
        let out = run(
            &["</block>"],
            &["allow</bl", "ock>ignored", " more"],
            Event::Incomplete,
        );
        assert_eq!(text(&out), "allow");
        // The provider's own max-tokens ending is superseded by the stop.
        assert!(stopped_at(out.last(), "</block>"));
    }

    #[test]
    fn a_false_prefix_is_released_once_the_next_delta_rules_it_out() {
        let out = run(&["</block>"], &["a</b", "x", "c"], Event::Completed);
        assert_eq!(text(&out), "a</bxc");
        assert!(matches!(out.last(), Some(Event::Completed)));
    }

    #[test]
    fn a_pending_prefix_at_the_terminal_is_flushed_before_it() {
        let out = run(&["DONE"], &["work DO"], Event::Completed);
        let kinds: Vec<_> = out.iter().map(simplified_event).collect();
        assert_eq!(
            kinds,
            vec![
                simplified_event(&Event::TextDelta("work ".to_string())),
                simplified_event(&Event::TextDelta("DO".to_string())),
                simplified_event(&Event::Completed),
            ]
        );
    }

    #[test]
    fn the_earliest_sequence_wins_and_the_longer_one_breaks_ties() {
        let out = run(&["ab", "abc", "zzz"], &["xxabcd"], Event::Completed);
        assert_eq!(text(&out), "xx");
        assert!(stopped_at(out.last(), "abc"));
    }

    #[test]
    fn overlapping_sequences_resolve_the_same_regardless_of_chunking() {
        // "ab" then "c" in separate deltas must still report "abc", exactly as
        // "abc" in one delta does: providers control delta boundaries, the
        // caller's stop semantics must not.
        let split = run(&["ab", "abc"], &["xxab", "cd"], Event::Completed);
        let whole = run(&["ab", "abc"], &["xxabcd"], Event::Completed);
        assert_eq!(text(&split), "xx");
        assert_eq!(text(&whole), "xx");
        assert!(stopped_at(split.last(), "abc"));
        assert!(stopped_at(whole.last(), "abc"));
        // The text before the held match is released immediately, not at the end.
        let mut guard = StopSequenceGuard::new(["ab", "abc"]).expect("sequences");
        let out = guard.filter(Event::TextDelta("xxab".to_string()));
        assert_eq!(text(&out), "xx");
        assert!(
            guard.matched().is_none(),
            "the shorter match is held for the longer one"
        );
        // A different continuation commits the shorter sequence.
        let out = guard.filter(Event::TextDelta("z".to_string()));
        assert!(out.is_empty());
        assert_eq!(guard.matched(), Some("ab"));
    }

    #[test]
    fn a_held_shorter_match_commits_at_the_terminal() {
        let out = run(&["ab", "abc"], &["xxab"], Event::Completed);
        assert_eq!(text(&out), "xx");
        assert!(stopped_at(out.last(), "ab"));
        let out = run(&["ab", "abc"], &["xxab"], Event::Incomplete);
        assert!(stopped_at(out.last(), "ab"));
    }

    #[test]
    fn provider_text_deltas_re_emit_with_their_item_identity() {
        let mut guard = StopSequenceGuard::new(["END"]).expect("sequences");
        let out = guard.filter(Event::ProviderTextDelta {
            output_index: 1,
            item_id: "msg_1".to_string(),
            delta: "hello E".to_string(),
        });
        assert_eq!(out.len(), 1);
        assert!(matches!(
            &out[0],
            Event::ProviderTextDelta { output_index: 1, item_id, delta }
                if item_id == "msg_1" && delta == "hello "
        ));
        let out = guard.filter(Event::ProviderTextDelta {
            output_index: 1,
            item_id: "msg_1".to_string(),
            delta: "NDafter".to_string(),
        });
        assert!(out.is_empty());
        assert_eq!(guard.matched(), Some("END"));
        // Lifecycle and usage events after the match still flow; text does not.
        let out = guard.filter(Event::Usage(crate::events::Usage::default()));
        assert!(matches!(out.as_slice(), [Event::Usage(_)]));
        assert!(guard
            .filter(Event::TextDelta("late".to_string()))
            .is_empty());
        let out = guard.filter(Event::Completed);
        assert_eq!(out.len(), 1);
        assert!(stopped_at(out.first(), "END"));
    }

    #[test]
    fn a_provider_failure_after_the_match_still_fails() {
        let mut guard = StopSequenceGuard::new(["END"]).expect("sequences");
        guard.filter(Event::TextDelta("xEND".to_string()));
        let failure = crate::errors::Failure::new(
            crate::errors::FailureClass::Transport,
            "provider transport failed; retry the request",
        );
        let out = guard.filter(Event::Failed(failure));
        assert!(matches!(out.as_slice(), [Event::Failed(_)]));
    }

    #[test]
    fn multibyte_text_never_splits_a_character() {
        let out = run(&["—end"], &["héllo —e", "nd tail"], Event::Completed);
        assert_eq!(text(&out), "héllo ");
        assert!(stopped_at(out.last(), "—end"));
        let out = run(&["—end"], &["héllo —", "x"], Event::Completed);
        assert_eq!(text(&out), "héllo —x");
    }

    #[test]
    fn non_text_events_pass_through_and_item_boundaries_flush_the_tail() {
        let mut guard = StopSequenceGuard::new(["END"]).expect("sequences");
        let out = guard.filter(Event::TextDelta("keep E".to_string()));
        assert_eq!(text(&out), "keep ");
        let out = guard.filter(Event::ProviderOutputItemCompleted {
            output_index: 0,
            item_id: None,
            kind: crate::events::ProviderOutputItemKind::Message,
            status: None,
            phase: None,
        });
        assert_eq!(text(&out), "E");
        assert!(matches!(
            out.last(),
            Some(Event::ProviderOutputItemCompleted { .. })
        ));
    }
}
