//! Content-free data-plane metrics: lock-free counters, gauges, and histograms.
//!
//! One process-global registry accumulates request outcomes, escalations,
//! retry activity, and latency distributions for the native engine. Every
//! value is an aggregate count or a duration bucket; no alias, model, key,
//! prompt, or provider payload ever enters the registry, so the snapshot is
//! publishable on the same terms as the health endpoints. The registry is
//! plain atomics with relaxed ordering: recording never takes a lock and a
//! snapshot is a consistent-enough point-in-time read for operations use.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

/// Upper bounds, in milliseconds, of the fixed latency buckets. The final
/// implicit bucket is unbounded.
pub const LATENCY_BUCKET_UPPER_MS: [u64; 15] = [
    1, 2, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000,
];

const BUCKET_COUNT: usize = LATENCY_BUCKET_UPPER_MS.len() + 1;

/// One fixed-bucket latency histogram with a total count and sum.
pub struct Histogram {
    buckets: [AtomicU64; BUCKET_COUNT],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl Histogram {
    /// Create an empty histogram; usable in `static` initializers.
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            buckets: [ZERO; BUCKET_COUNT],
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    /// Record one observed duration.
    pub fn record(&self, elapsed: Duration) {
        let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
        let ms = micros / 1_000;
        let index = LATENCY_BUCKET_UPPER_MS
            .iter()
            .position(|upper| ms <= *upper)
            .unwrap_or(BUCKET_COUNT - 1);
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add(micros, Ordering::Relaxed);
    }

    /// Snapshot this histogram as one JSON object. Bucket upper bounds are in
    /// milliseconds; the unbounded overflow bucket carries `"le_ms": null`.
    fn snapshot(&self) -> Value {
        let buckets: Vec<Value> = (0..BUCKET_COUNT)
            .map(|index| {
                let upper = LATENCY_BUCKET_UPPER_MS.get(index).copied();
                json!({
                    "le_ms": upper,
                    "count": self.buckets[index].load(Ordering::Relaxed),
                })
            })
            .collect();
        json!({
            "count": self.count.load(Ordering::Relaxed),
            "sum_ms": self.sum_micros.load(Ordering::Relaxed) as f64 / 1_000.0,
            "buckets": buckets,
        })
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// The bounded, content-free classification of one escalated admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationKind {
    /// Project-backed aliases use learned selection on the python engine.
    ProjectAlias,
    /// Multi-deployment pools use the python engine's certified waterfall.
    DeploymentPool,
    /// The resolved provider has no native dialect or wire profile.
    ProviderDialect,
    /// A hosted composition's execution policy retained the route.
    HostPolicy,
    /// Any escalation reason outside the known set.
    Other,
}

/// Classify one display-safe escalation reason into its bounded kind, so the
/// registry never retains the reason text itself.
pub fn classify_escalation(reason: &str) -> EscalationKind {
    if reason.contains("project-backed") {
        EscalationKind::ProjectAlias
    } else if reason.contains("multi-deployment") {
        EscalationKind::DeploymentPool
    } else if reason.contains("native wire profile") || reason.contains("native dialect") {
        EscalationKind::ProviderDialect
    } else if reason.contains("host policy") {
        EscalationKind::HostPolicy
    } else {
        EscalationKind::Other
    }
}

/// The process-global data-plane metrics registry.
pub struct Metrics {
    requests_completed: AtomicU64,
    requests_incomplete: AtomicU64,
    requests_failed: AtomicU64,
    requests_cancelled: AtomicU64,
    served_requests: AtomicU64,
    escalated_project_alias: AtomicU64,
    escalated_deployment_pool: AtomicU64,
    escalated_provider_dialect: AtomicU64,
    escalated_host_policy: AtomicU64,
    escalated_other: AtomicU64,
    proxied_requests: AtomicU64,
    fallback_engine_unavailable: AtomicU64,
    open_retries: AtomicU64,
    settlement_retries: AtomicU64,
    settlement_give_ups: AtomicU64,
    active_requests: AtomicU64,
    active_proxies: AtomicU64,
    pub time_to_first_byte_ms: Histogram,
    pub request_duration_ms: Histogram,
    pub permit_wait_ms: Histogram,
    pub bridge_call_ms: Histogram,
}

/// The one registry shared by the serving runtime and the snapshot readers.
pub static METRICS: Metrics = Metrics::new();

impl Metrics {
    /// Create an empty registry; usable in `static` initializers.
    pub const fn new() -> Self {
        Self {
            requests_completed: AtomicU64::new(0),
            requests_incomplete: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_cancelled: AtomicU64::new(0),
            served_requests: AtomicU64::new(0),
            escalated_project_alias: AtomicU64::new(0),
            escalated_deployment_pool: AtomicU64::new(0),
            escalated_provider_dialect: AtomicU64::new(0),
            escalated_host_policy: AtomicU64::new(0),
            escalated_other: AtomicU64::new(0),
            proxied_requests: AtomicU64::new(0),
            fallback_engine_unavailable: AtomicU64::new(0),
            open_retries: AtomicU64::new(0),
            settlement_retries: AtomicU64::new(0),
            settlement_give_ups: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            active_proxies: AtomicU64::new(0),
            time_to_first_byte_ms: Histogram::new(),
            request_duration_ms: Histogram::new(),
            permit_wait_ms: Histogram::new(),
            bridge_call_ms: Histogram::new(),
        }
    }

    /// Count one natively admitted (served) request.
    pub fn record_served(&self) {
        self.served_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the terminal outcome of one natively served request. A failed
    /// outcome whose failure class is a cancellation counts as cancelled.
    pub fn record_outcome(&self, outcome: &str, cancelled: bool) {
        let counter = if cancelled {
            &self.requests_cancelled
        } else {
            match outcome {
                "completed" => &self.requests_completed,
                "incomplete" => &self.requests_incomplete,
                _ => &self.requests_failed,
            }
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one escalated admission by its bounded kind.
    pub fn record_escalation(&self, kind: EscalationKind) {
        let counter = match kind {
            EscalationKind::ProjectAlias => &self.escalated_project_alias,
            EscalationKind::DeploymentPool => &self.escalated_deployment_pool,
            EscalationKind::ProviderDialect => &self.escalated_provider_dialect,
            EscalationKind::HostPolicy => &self.escalated_host_policy,
            EscalationKind::Other => &self.escalated_other,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one request relayed to the embedded python engine.
    pub fn record_proxied(&self) {
        self.proxied_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one exhausted attempt to reach the embedded python engine. This
    /// is also the only signal a dead embedded engine produces.
    pub fn record_fallback_unavailable(&self) {
        self.fallback_engine_unavailable
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Count one same-deployment retry at the upstream open phase.
    pub fn record_open_retry(&self) {
        self.open_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one retried settlement delivery after a failed write.
    pub fn record_settlement_retry(&self) {
        self.settlement_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one settlement whose bounded retries were all exhausted.
    pub fn record_settlement_give_up(&self) {
        self.settlement_give_ups.fetch_add(1, Ordering::Relaxed);
    }

    /// Track one natively served request entering the data plane.
    pub fn enter_request(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Track one natively served request leaving the data plane.
    pub fn exit_request(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Track one proxied request entering the relay.
    pub fn enter_proxy(&self) {
        self.active_proxies.fetch_add(1, Ordering::Relaxed);
    }

    /// Track one proxied request leaving the relay.
    pub fn exit_proxy(&self) {
        self.active_proxies.fetch_sub(1, Ordering::Relaxed);
    }

    /// Snapshot the whole registry as one JSON object.
    pub fn snapshot(&self) -> Value {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        json!({
            "requests": {
                "completed": load(&self.requests_completed),
                "incomplete": load(&self.requests_incomplete),
                "failed": load(&self.requests_failed),
                "cancelled": load(&self.requests_cancelled),
            },
            "served_requests": load(&self.served_requests),
            "escalated_requests": {
                "project_alias": load(&self.escalated_project_alias),
                "deployment_pool": load(&self.escalated_deployment_pool),
                "provider_dialect": load(&self.escalated_provider_dialect),
                "host_policy": load(&self.escalated_host_policy),
                "other": load(&self.escalated_other),
            },
            "proxied_requests": load(&self.proxied_requests),
            "fallback_engine_unavailable": load(&self.fallback_engine_unavailable),
            "open_retries": load(&self.open_retries),
            "settlement_retries": load(&self.settlement_retries),
            "settlement_give_ups": load(&self.settlement_give_ups),
            "active_requests": load(&self.active_requests),
            "active_proxies": load(&self.active_proxies),
            "time_to_first_byte_ms": self.time_to_first_byte_ms.snapshot(),
            "request_duration_ms": self.request_duration_ms.snapshot(),
            "permit_wait_ms": self.permit_wait_ms.snapshot(),
            "bridge_call_ms": self.bridge_call_ms.snapshot(),
        })
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment_into_the_snapshot() {
        let metrics = Metrics::new();
        metrics.record_served();
        metrics.record_served();
        metrics.record_outcome("completed", false);
        metrics.record_outcome("incomplete", false);
        metrics.record_outcome("failed", false);
        metrics.record_outcome("failed", true);
        metrics.record_escalation(EscalationKind::ProjectAlias);
        metrics.record_proxied();
        metrics.record_fallback_unavailable();
        metrics.record_open_retry();
        metrics.record_settlement_retry();
        metrics.record_settlement_give_up();
        metrics.enter_request();
        metrics.enter_proxy();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["served_requests"], 2);
        assert_eq!(snapshot["requests"]["completed"], 1);
        assert_eq!(snapshot["requests"]["incomplete"], 1);
        assert_eq!(snapshot["requests"]["failed"], 1);
        assert_eq!(snapshot["requests"]["cancelled"], 1);
        assert_eq!(snapshot["escalated_requests"]["project_alias"], 1);
        assert_eq!(snapshot["escalated_requests"]["other"], 0);
        assert_eq!(snapshot["proxied_requests"], 1);
        assert_eq!(snapshot["fallback_engine_unavailable"], 1);
        assert_eq!(snapshot["open_retries"], 1);
        assert_eq!(snapshot["settlement_retries"], 1);
        assert_eq!(snapshot["settlement_give_ups"], 1);
        assert_eq!(snapshot["active_requests"], 1);
        assert_eq!(snapshot["active_proxies"], 1);
        metrics.exit_request();
        metrics.exit_proxy();
        let drained = metrics.snapshot();
        assert_eq!(drained["active_requests"], 0);
        assert_eq!(drained["active_proxies"], 0);
    }

    #[test]
    fn histogram_buckets_by_upper_bound_with_an_overflow_bucket() {
        let histogram = Histogram::new();
        histogram.record(Duration::from_micros(500));
        histogram.record(Duration::from_millis(1));
        histogram.record(Duration::from_millis(3));
        histogram.record(Duration::from_secs(120));
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot["count"], 4);
        let buckets = snapshot["buckets"].as_array().expect("bucket array");
        assert_eq!(buckets.len(), LATENCY_BUCKET_UPPER_MS.len() + 1);
        assert_eq!(buckets[0]["le_ms"], 1);
        assert_eq!(buckets[0]["count"], 2);
        assert_eq!(buckets[2]["le_ms"], 5);
        assert_eq!(buckets[2]["count"], 1);
        let overflow = buckets.last().expect("overflow bucket");
        assert!(overflow["le_ms"].is_null());
        assert_eq!(overflow["count"], 1);
        let sum_ms = snapshot["sum_ms"].as_f64().expect("sum");
        assert!((sum_ms - 120_004.5).abs() < 0.001);
    }

    #[test]
    fn escalation_reasons_classify_into_bounded_kinds() {
        assert_eq!(
            classify_escalation("project-backed aliases use learned selection"),
            EscalationKind::ProjectAlias
        );
        assert_eq!(
            classify_escalation("multi-deployment pools use the certified waterfall"),
            EscalationKind::DeploymentPool
        );
        assert_eq!(
            classify_escalation("provider 'x' has no native wire profile"),
            EscalationKind::ProviderDialect
        );
        assert_eq!(
            classify_escalation("provider 'x' has no native dialect implementation"),
            EscalationKind::ProviderDialect
        );
        assert_eq!(
            classify_escalation("host policy requires the python execution engine"),
            EscalationKind::HostPolicy
        );
        assert_eq!(classify_escalation("anything else"), EscalationKind::Other);
    }

    #[test]
    fn snapshot_shape_is_stable() {
        let snapshot = Metrics::new().snapshot();
        let object = snapshot.as_object().expect("object");
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "requests",
                "served_requests",
                "escalated_requests",
                "proxied_requests",
                "fallback_engine_unavailable",
                "open_retries",
                "settlement_retries",
                "settlement_give_ups",
                "active_requests",
                "active_proxies",
                "time_to_first_byte_ms",
                "request_duration_ms",
                "permit_wait_ms",
                "bridge_call_ms",
            ]
        );
    }
}
