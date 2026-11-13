//! Bounded content-free aggregate counters, drained by the host's background logger.

use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const MAX_BUCKETS: usize = 256;
const MAX_CATEGORIES: usize = 64;

#[derive(Default, Serialize)]
struct Totals {
    dimensions: Value,
    samples: u64,
    duration_ms: f64,
    cache_hits: u64,
    cache_misses: u64,
    errors: u64,
    classifier_latency_seconds: f64,
    score_buckets: BTreeMap<String, [u64; 20]>,
}

#[derive(Default)]
pub(super) struct Telemetry {
    buckets: BTreeMap<String, Totals>,
    dropped: u64,
    cumulative: Totals,
    cumulative_dropped: u64,
}

impl Telemetry {
    pub fn record(&mut self, moderation: &Value) {
        let Some(records) = moderation["observations"].as_array() else {
            return;
        };
        for record in records {
            self.cumulative_dropped = self
                .cumulative_dropped
                .saturating_add(self.cumulative.record(record));
            let mut dimensions = json!({});
            for name in [
                "policy_id",
                "revision",
                "check_id",
                "capability",
                "stage",
                "would_action",
                "mode",
                "outcome",
            ] {
                dimensions[name] = record[name].clone();
            }
            let key = dimensions.to_string();
            let duration = record["latency_ms"].as_f64().unwrap_or(0.0);
            if key.len() > 2048
                || !duration.is_finite()
                || duration < 0.0
                || (self.buckets.len() >= MAX_BUCKETS && !self.buckets.contains_key(&key))
            {
                self.dropped = self.dropped.saturating_add(1);
                continue;
            }
            let totals = self.buckets.entry(key).or_insert_with(|| Totals {
                dimensions,
                ..Totals::default()
            });
            self.dropped = self.dropped.saturating_add(totals.record(record));
        }
    }

    pub fn drain(&mut self) -> Value {
        json!({"aggregates": std::mem::take(&mut self.buckets).into_values().collect::<Vec<_>>(),
            "dropped": std::mem::take(&mut self.dropped)})
    }

    /// Metrics are cumulative and never consume the background logger's deltas.
    pub fn metrics(&self) -> Value {
        json!({"totals":self.cumulative,"dropped":self.cumulative_dropped})
    }
}

impl Totals {
    fn record(&mut self, record: &Value) -> u64 {
        let duration = record["latency_ms"].as_f64().unwrap_or(0.0);
        if !duration.is_finite() || duration < 0.0 || !(self.duration_ms + duration).is_finite() {
            return 1;
        }
        self.samples = self.samples.saturating_add(1);
        self.duration_ms += duration;
        if record["classifier_kind"] != "decision_model" {
            return 0;
        }
        if record["cached"] == true {
            self.cache_hits = self.cache_hits.saturating_add(1);
        } else {
            self.cache_misses = self.cache_misses.saturating_add(1);
            self.classifier_latency_seconds += duration / 1000.0;
            if !record["scores"].is_array() {
                self.errors = self.errors.saturating_add(1);
            }
        }
        let mut dropped = 0;
        if let Some(scores) = record["scores"].as_array() {
            for score in scores {
                let (Some(category), Some(value)) = (score[0].as_str(), score[1].as_f64()) else {
                    continue;
                };
                if category.len() > 128
                    || !value.is_finite()
                    || !(0.0..=1.0).contains(&value)
                    || (self.score_buckets.len() >= MAX_CATEGORIES
                        && !self.score_buckets.contains_key(category))
                {
                    dropped += 1;
                    continue;
                }
                let counts = self.score_buckets.entry(category.into()).or_default();
                let index = ((value * 20.0).floor() as usize).min(19);
                counts[index] = counts[index].saturating_add(1);
            }
        }
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_are_bounded_content_free_and_drained_once() {
        let mut telemetry = Telemetry::default();
        let mut record = json!({"policy_id":"policy", "mode":"observe", "outcome":"allow", "latency_ms":4.0,
            "classifier_kind":"decision_model", "cached":false, "scores":[["category",0.95]],
            "subject_sha256":"private-subject-identity", "request":"private-input"});
        telemetry.record(&json!({"observations":[record]}));
        record["cached"] = json!(true);
        telemetry.record(&json!({"observations":[record]}));
        record["cached"] = Value::Null;
        record["scores"] = Value::Null;
        telemetry.record(&json!({"observations":[record]}));
        let snapshot = telemetry.drain();
        let metrics = telemetry.metrics();
        let aggregate = &snapshot["aggregates"][0];
        assert_eq!(aggregate["samples"], 3);
        assert_eq!(aggregate["cache_hits"], 1);
        assert_eq!(aggregate["cache_misses"], 2);
        assert_eq!(aggregate["errors"], 1);
        assert_eq!(aggregate["score_buckets"]["category"][19], 2);
        assert_eq!(aggregate["classifier_latency_seconds"], 0.008);
        assert!(!snapshot.to_string().contains("private-"));
        assert_eq!(telemetry.drain(), json!({"aggregates":[],"dropped":0}));
        assert_eq!(telemetry.metrics(), metrics);
        assert_eq!(metrics["totals"]["cache_hits"], 1);
        assert_eq!(metrics["totals"]["score_buckets"]["category"][19], 2);
        for index in 0..MAX_BUCKETS + 1 {
            record["revision"] = json!(index.to_string());
            telemetry.record(&json!({"observations":[record]}));
        }
        let snapshot = telemetry.drain();
        assert_eq!(
            snapshot["aggregates"].as_array().unwrap().len(),
            MAX_BUCKETS
        );
        assert_eq!(snapshot["dropped"], 1);
        assert_eq!(telemetry.metrics()["totals"]["samples"], MAX_BUCKETS + 4);
    }
}
