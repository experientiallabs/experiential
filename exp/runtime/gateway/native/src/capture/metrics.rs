//! Winning-attempt timing and usage, borrowed from native accounting observations.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::events::{Event, Usage};
use crate::settlement::Observation;

/// Provider attempt facts, not request-wide totals across failed fallback attempts.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Metrics {
    pub started_at: f64,
    pub first_token_at: Option<f64>,
    pub terminal_at: Option<f64>,
    pub duration_ms: Option<f64>,
    pub usage: Option<Usage>,
    /// False for interrupted, partial or noncredible all-zero finished meters.
    pub usage_complete: bool,
}

fn timestamp(at: SystemTime) -> f64 {
    at.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

impl Metrics {
    pub(crate) fn observed(observation: &Observation) -> Self {
        let observed = observation.snapshot();
        // Match settlement credibility: a served completion cannot have processed
        // neither input nor output. Preserve the raw report, but never certify
        // it as observed zero-cost usage. Failed-terminal zeros remain distinct.
        let finished_zero = !matches!(observed.terminal, Some(Event::Failed(_)))
            && observed.usage.as_ref().is_some_and(|usage| {
                usage.input_tokens == Some(0) && usage.output_tokens == Some(0)
            });
        Self {
            started_at: timestamp(observed.started_at),
            first_token_at: observed.first_token_at.map(timestamp),
            terminal_at: observed.terminal_at.map(timestamp),
            duration_ms: observed
                .duration
                .map(|elapsed| elapsed.as_secs_f64() * 1000.0),
            usage_complete: observed.terminal.is_some()
                && !finished_zero
                && observed.usage.as_ref().is_some_and(Usage::has_token_counts),
            usage: observed.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Event;

    #[test]
    fn finished_zero_meters_remain_raw_evidence_not_credible_complete_usage() {
        for terminal in [
            Event::Completed,
            Event::Incomplete,
            Event::PausedTurn,
            Event::StoppedAtSequence("stop".into()),
        ] {
            let observation = Observation::default();
            observation.record(&Event::Usage(Usage {
                input_tokens: Some(0),
                output_tokens: Some(0),
                ..Usage::default()
            }));
            observation.record(&terminal);
            let metrics = Metrics::observed(&observation);
            assert!(!metrics.usage_complete);
            assert_eq!(metrics.usage.unwrap().input_tokens, Some(0));
        }
        let observation = Observation::default();
        observation.record(&Event::Usage(Usage {
            input_tokens: Some(13),
            output_tokens: Some(0),
            ..Usage::default()
        }));
        observation.record(&Event::Incomplete);
        assert!(Metrics::observed(&observation).usage_complete);
    }

    #[test]
    fn metrics_preserve_unknown_usage_and_provider_timestamps() {
        let observation = Observation::default();
        observation.record_first_token(Some(SystemTime::now()));
        observation.record(&Event::Usage(Usage {
            input_tokens: Some(17),
            ..Usage::default()
        }));
        let partial = Metrics::observed(&observation);
        assert!(!partial.usage_complete);
        assert!(partial.terminal_at.is_none());
        assert_eq!(partial.usage.unwrap().output_tokens, None);
        observation.record(&Event::Usage(Usage {
            output_tokens: Some(5),
            reasoning_tokens: Some(2),
            ..Usage::default()
        }));
        observation.record(&Event::Completed);
        let complete = Metrics::observed(&observation);
        assert!(complete.usage_complete);
        assert!(complete.first_token_at.unwrap() >= complete.started_at);
        assert!(complete.terminal_at.unwrap() >= complete.first_token_at.unwrap());
        assert!(complete.duration_ms.unwrap() >= 0.0);
        assert_eq!(complete.usage.as_ref().unwrap().input_tokens, Some(17));
        assert_eq!(complete.usage.as_ref().unwrap().reasoning_tokens, Some(2));
        assert_eq!(complete.usage.as_ref().unwrap().cached_input_tokens, None);
    }
}
