//! Decision-model classification of a bounded request window, run in Rust.
//!
//! One SystemOne call asks every configured question at once: `POST <url>`
//! with `{"model", "state", "questions"}`, answered by
//! `{"answers": {"<id>": {"type": "noul", "noul": <probability>}}}`. Question
//! probabilities combine into categories (`max` or `mean`), and a window is
//! flagged when any category reaches its threshold. Definitive answers are
//! cached by window digest; errors never are.
//!
//! The control plane validates the adapter document, builds the window, and
//! composes one [`Classifier`] per document. Every call runs on one dedicated
//! runtime, so pooled connections never outlive the runtime that opened them
//! whether the caller is the data plane or a python thread. The window text
//! leaves this module only as the decision request's state, and nothing here
//! logs it.
//!
//! An observing policy's checks run here end to end: admission names each
//! classifier by handle with the window to classify, [`spawn_observations`]
//! classifies them beside generation, and one control-plane callback records
//! every answer. Nothing waits for them, and no response byte depends on them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::bridge::Bridge;
use crate::encode::compact_json;

/// Histogram buckets per category, each 0.05 wide.
pub(crate) const SCORE_BUCKETS: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Aggregate {
    Max,
    Mean,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CategorySpec {
    id: String,
    aggregate: Aggregate,
    threshold: f64,
    questions: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheSpec {
    max_entries: usize,
    ttl_seconds: f64,
}

/// One validated adapter document as the control plane composes it.
///
/// `questions` holds each question's SystemOne wire object, sent verbatim, in
/// authored order. `configuration_digest` is the control plane's digest of
/// the endpoint, model, categories, and window, and keys the verdict cache.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ClassifierSpec {
    url: String,
    model: String,
    timeout_ms: u64,
    max_response_bytes: usize,
    configuration_digest: String,
    questions: Map<String, Value>,
    categories: Vec<CategorySpec>,
    cache: CacheSpec,
}

/// Why a decision call ended without a definitive answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClassifyError {
    Timeout,
    Transport,
    Status(u16),
    Oversized,
    NotJson,
    NoAnswers,
    MissingQuestion,
    NotNoul,
    OutOfRange,
}

impl ClassifyError {
    /// The fixed, content-free reason the control plane logs and records.
    pub(crate) fn reason(self) -> String {
        match self {
            Self::Timeout => "decision model timed out".into(),
            Self::Transport => "decision model transport failed".into(),
            Self::Status(status) => format!("decision model returned HTTP {status}"),
            Self::Oversized => "classifier response exceeded max_response_bytes".into(),
            Self::NotJson => "decision model answer is not valid JSON".into(),
            Self::NoAnswers => "decision model answer has no answers object".into(),
            Self::MissingQuestion => "decision model answer is missing a question".into(),
            Self::NotNoul => "decision model answer is not a noul probability".into(),
            Self::OutOfRange => "decision model probability is outside [0, 1]".into(),
        }
    }
}

/// One definitive answer for one window, without the window's text.
#[derive(Debug, Clone)]
pub(crate) struct Answer {
    /// Every question's probability, in authored order.
    pub(crate) probabilities: Arc<[f64]>,
    /// Every category's combined score, in authored order.
    pub(crate) scores: Vec<f64>,
    /// Indexes of the categories whose score reached their threshold.
    pub(crate) flagged: Vec<usize>,
    pub(crate) cached: bool,
    pub(crate) latency: Duration,
}

/// A composed decision-model classifier shared by every request that binds it.
pub(crate) struct Classifier {
    spec: ClassifierSpec,
    question_ids: Vec<String>,
    category_questions: Vec<Vec<usize>>,
    bearer: Option<String>,
    client: reqwest::Client,
    cache: Mutex<VerdictCache>,
    counters: Mutex<Counters>,
}

impl Classifier {
    /// Compose one classifier, refusing a spec the control plane could not have validated.
    pub(crate) fn new(spec: ClassifierSpec, bearer: Option<String>) -> Result<Self, String> {
        if !(spec.url.starts_with("https://") || spec.url.starts_with("http://")) {
            return Err("decision model url must be http or https".into());
        }
        if spec.categories.is_empty() || spec.questions.is_empty() {
            return Err("decision model needs at least one category and one question".into());
        }
        if spec.timeout_ms == 0 || spec.max_response_bytes == 0 {
            return Err("decision model timeout and response bound must be positive".into());
        }
        if !(spec.cache.ttl_seconds.is_finite() && spec.cache.ttl_seconds > 0.0) {
            return Err("decision model cache ttl must be positive".into());
        }
        let question_ids: Vec<String> = spec.questions.keys().cloned().collect();
        let mut category_questions = Vec::with_capacity(spec.categories.len());
        for category in &spec.categories {
            if !(category.threshold > 0.0 && category.threshold <= 1.0) {
                return Err("category thresholds must be in (0, 1]".into());
            }
            if category.questions.is_empty() {
                return Err("every category needs a question".into());
            }
            let indexes = category
                .questions
                .iter()
                .map(|id| question_ids.iter().position(|known| known == id))
                .collect::<Option<Vec<usize>>>()
                .ok_or("a category names an unknown question")?;
            category_questions.push(indexes);
        }
        let client = reqwest::Client::builder()
            // The configured endpoint only: never an ambient HTTP(S)_PROXY.
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .use_rustls_tls()
            .build()
            .map_err(|_| "decision model client could not be built")?;
        let counters = Counters::new(spec.categories.len());
        let cache = VerdictCache::new(
            spec.cache.max_entries,
            Duration::from_secs_f64(spec.cache.ttl_seconds),
        );
        Ok(Self {
            spec,
            question_ids,
            category_questions,
            bearer,
            client,
            cache: Mutex::new(cache),
            counters: Mutex::new(counters),
        })
    }

    /// Classify one window before `deadline`, from the cache or one decision call.
    pub(crate) async fn classify(
        self: &Arc<Self>,
        window: &str,
        deadline: Instant,
    ) -> Result<Answer, ClassifyError> {
        let started = Instant::now();
        let key = cache_key(&self.spec.configuration_digest, window);
        let cached = self
            .cache
            .lock()
            .ok()
            .and_then(|mut cache| cache.get(&key, started));
        let (probabilities, from_cache) = match cached {
            Some(probabilities) => (probabilities, true),
            None => {
                let classifier = self.clone();
                let body = self.request_body(window);
                let decided = runtime()
                    .spawn(async move { classifier.decide(body, deadline).await })
                    .await
                    .unwrap_or(Err(ClassifyError::Transport));
                match decided {
                    Ok(probabilities) => {
                        let probabilities: Arc<[f64]> = probabilities.into();
                        if let Ok(mut cache) = self.cache.lock() {
                            cache.put(key, probabilities.clone(), Instant::now());
                        }
                        (probabilities, false)
                    }
                    Err(error) => {
                        if let Ok(mut counters) = self.counters.lock() {
                            counters.record_error(started.elapsed());
                        }
                        return Err(error);
                    }
                }
            }
        };
        let scores = self.scores(&probabilities);
        let flagged = scores
            .iter()
            .enumerate()
            .filter(|(index, score)| **score >= self.spec.categories[*index].threshold)
            .map(|(index, _)| index)
            .collect();
        let answer = Answer {
            probabilities,
            scores,
            flagged,
            cached: from_cache,
            latency: started.elapsed(),
        };
        if let Ok(mut counters) = self.counters.lock() {
            counters.record(&answer);
        }
        Ok(answer)
    }

    /// One answer as content-free JSON: ids map to probabilities and scores in authored order.
    pub(crate) fn answer_json(&self, answer: &Answer) -> Value {
        let probabilities: Map<String, Value> = self
            .question_ids
            .iter()
            .zip(answer.probabilities.iter())
            .map(|(id, value)| (id.clone(), json!(value)))
            .collect();
        let scores: Map<String, Value> = self
            .spec
            .categories
            .iter()
            .zip(&answer.scores)
            .map(|(category, value)| (category.id.clone(), json!(value)))
            .collect();
        let flagged: Vec<&str> = answer
            .flagged
            .iter()
            .map(|index| self.spec.categories[*index].id.as_str())
            .collect();
        json!({
            "probabilities": probabilities,
            "scores": scores,
            "flagged": flagged,
            "cached": answer.cached,
            "latency_ms": answer.latency.as_secs_f64() * 1000.0,
        })
    }

    /// The counters as the python adapter reports them: score buckets, cache use, latency.
    pub(crate) fn telemetry_json(&self) -> Value {
        let Ok(counters) = self.counters.lock() else {
            return json!({});
        };
        let buckets: Map<String, Value> = self
            .spec
            .categories
            .iter()
            .zip(&counters.buckets)
            .map(|(category, counts)| (category.id.clone(), json!(counts)))
            .collect();
        json!({
            "cache_hits": counters.cache_hits,
            "cache_misses": counters.cache_misses,
            "errors": counters.errors,
            "classifier_latency_seconds": counters.latency.as_secs_f64(),
            "score_buckets": buckets,
        })
    }

    fn request_body(&self, window: &str) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "model": self.spec.model,
            "state": window,
            "questions": self.spec.questions,
        }))
        .unwrap_or_default()
    }

    fn scores(&self, probabilities: &[f64]) -> Vec<f64> {
        self.spec
            .categories
            .iter()
            .zip(&self.category_questions)
            .map(|(category, indexes)| {
                let values = indexes.iter().map(|index| probabilities[*index]);
                match category.aggregate {
                    Aggregate::Max => values.fold(0.0, f64::max),
                    Aggregate::Mean => fsum(values) / indexes.len() as f64,
                }
            })
            .collect()
    }

    /// POST one decision request and return every question's probability, in order.
    async fn decide(&self, body: Vec<u8>, deadline: Instant) -> Result<Vec<f64>, ClassifyError> {
        let bound = Duration::from_millis(self.spec.timeout_ms)
            .min(deadline.saturating_duration_since(Instant::now()));
        if bound.is_zero() {
            return Err(ClassifyError::Timeout);
        }
        let mut request = self
            .client
            .post(&self.spec.url)
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .body(body);
        if let Some(bearer) = &self.bearer {
            request = request.header("authorization", format!("Bearer {bearer}"));
        }
        let read = async {
            let mut response = request.send().await.map_err(transport)?;
            if response.status() != reqwest::StatusCode::OK {
                return Err(ClassifyError::Status(response.status().as_u16()));
            }
            let mut raw = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(transport)? {
                if chunk.len() > self.spec.max_response_bytes.saturating_sub(raw.len()) {
                    return Err(ClassifyError::Oversized);
                }
                raw.extend_from_slice(&chunk);
            }
            Ok(raw)
        };
        let raw = tokio::time::timeout(bound, read)
            .await
            .map_err(|_| ClassifyError::Timeout)??;
        answer_probabilities(&raw, &self.question_ids)
    }
}

/// The correctly rounded sum of `values`, as python's `math.fsum` computes it.
///
/// A mean category compares its score to an inclusive threshold, so ten
/// answers of 0.8 must average to exactly 0.8, never 0.7999999999999999.
fn fsum(values: impl Iterator<Item = f64>) -> f64 {
    // Shewchuk's exact partials, then a round-half-even correction.
    let mut partials: Vec<f64> = Vec::new();
    for mut x in values {
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0.0 {
                partials[kept] = low;
                kept += 1;
            }
            x = high;
        }
        partials.truncate(kept);
        partials.push(x);
    }
    let Some(mut high) = partials.pop() else {
        return 0.0;
    };
    let mut low = 0.0;
    while let Some(y) = partials.pop() {
        let x = high;
        high = x + y;
        low = y - (high - x);
        if low != 0.0 {
            break;
        }
    }
    if let Some(&next) = partials.last() {
        if (low < 0.0 && next < 0.0) || (low > 0.0 && next > 0.0) {
            let y = low * 2.0;
            let x = high + y;
            if y == x - high {
                high = x;
            }
        }
    }
    high
}

fn transport(error: reqwest::Error) -> ClassifyError {
    if error.is_timeout() {
        ClassifyError::Timeout
    } else {
        ClassifyError::Transport
    }
}

/// Parse a SystemOne answer body into one probability per asked question.
fn answer_probabilities(raw: &[u8], question_ids: &[String]) -> Result<Vec<f64>, ClassifyError> {
    let parsed = crate::route_decisions::strict_json(raw).map_err(|_| ClassifyError::NotJson)?;
    let answers = parsed
        .get("answers")
        .and_then(Value::as_object)
        .ok_or(ClassifyError::NoAnswers)?;
    question_ids
        .iter()
        .map(|id| {
            let answer = answers
                .get(id)
                .and_then(Value::as_object)
                .ok_or(ClassifyError::MissingQuestion)?;
            if answer.get("type").is_some_and(|kind| kind != "noul") {
                return Err(ClassifyError::NotNoul);
            }
            answer
                .get("noul")
                .filter(|value| value.is_number())
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .ok_or(ClassifyError::OutOfRange)
        })
        .collect()
}

fn cache_key(digest: &str, window: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(digest.as_bytes());
    hasher.update(b"\n");
    hasher.update(window.as_bytes());
    hasher.finalize().into()
}

/// Return the 0.05-wide histogram bucket of a score in `[0, 1]`.
fn score_bucket(score: f64) -> usize {
    ((score * SCORE_BUCKETS as f64).floor() as usize).min(SCORE_BUCKETS - 1)
}

struct Counters {
    cache_hits: u64,
    cache_misses: u64,
    errors: u64,
    latency: Duration,
    buckets: Vec<[u64; SCORE_BUCKETS]>,
}

impl Counters {
    fn new(categories: usize) -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            errors: 0,
            latency: Duration::ZERO,
            buckets: vec![[0; SCORE_BUCKETS]; categories],
        }
    }

    fn record(&mut self, answer: &Answer) {
        if answer.cached {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
            self.latency += answer.latency;
        }
        for (counts, score) in self.buckets.iter_mut().zip(&answer.scores) {
            counts[score_bucket(*score)] += 1;
        }
    }

    fn record_error(&mut self, latency: Duration) {
        self.cache_misses += 1;
        self.errors += 1;
        self.latency += latency;
    }
}

/// One cached answer: when it expires, when it was last used, and its probabilities.
struct CacheEntry {
    expires_at: Instant,
    used: u64,
    probabilities: Arc<[f64]>,
}

/// Least-recently-used map from a window digest to its question probabilities.
struct VerdictCache {
    max_entries: usize,
    ttl: Duration,
    tick: u64,
    entries: HashMap<[u8; 32], CacheEntry>,
}

impl VerdictCache {
    fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            max_entries,
            ttl,
            tick: 0,
            entries: HashMap::new(),
        }
    }

    fn get(&mut self, key: &[u8; 32], now: Instant) -> Option<Arc<[f64]>> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.entries.get_mut(key)?;
        if entry.expires_at <= now {
            self.entries.remove(key);
            return None;
        }
        entry.used = tick;
        Some(entry.probabilities.clone())
    }

    fn put(&mut self, key: [u8; 32], probabilities: Arc<[f64]>, now: Instant) {
        if self.max_entries == 0 {
            return;
        }
        self.tick += 1;
        let entry = CacheEntry {
            expires_at: now + self.ttl,
            used: self.tick,
            probabilities,
        };
        self.entries.insert(key, entry);
        while self.entries.len() > self.max_entries {
            // Bounded at a few thousand entries and run only on a miss.
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| *key);
            match oldest {
                Some(key) => self.entries.remove(&key),
                None => break,
            };
        }
    }
}

/// The one runtime every decision call runs on.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("exp-decision-classifier")
            .enable_all()
            .build()
            .expect("the decision classifier runtime starts")
    })
}

type Registry = Mutex<HashMap<u64, Weak<Classifier>>>;

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Make one classifier reachable by handle until its last owner drops it.
pub(crate) fn register(classifier: &Arc<Classifier>) -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let handle = NEXT.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut entries) = registry().lock() {
        entries.retain(|_, entry| entry.strong_count() > 0);
        entries.insert(handle, Arc::downgrade(classifier));
    }
    handle
}

/// The live classifier behind a handle, or `None` once it was replaced or dropped.
pub(crate) fn lookup(handle: u64) -> Option<Arc<Classifier>> {
    registry().lock().ok()?.get(&handle)?.upgrade()
}

/// One observing check admission handed to the data plane.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Observation {
    /// The control plane's index of the check, echoed in its result.
    pub(crate) index: u32,
    /// The classifier's handle.
    pub(crate) handle: u64,
    /// The bounded window to classify.
    pub(crate) window: String,
}

/// Requests whose observations may be in flight at once in this process.
const MAXIMUM_OBSERVING_REQUESTS: usize = 256;

fn observing() -> &'static Arc<Semaphore> {
    static OBSERVING: OnceLock<Arc<Semaphore>> = OnceLock::new();
    OBSERVING.get_or_init(|| Arc::new(Semaphore::new(MAXIMUM_OBSERVING_REQUESTS)))
}

/// Classify a request's observations beside generation and report them once.
///
/// Nothing awaits the task. Each observation answers, errs with a fixed
/// reason, or is skipped when the process already observes its maximum of
/// requests or the classifier is gone; `guardrail_observation_results`
/// receives `{request_id, results: [{index, answer | error | skipped}]}`.
pub(crate) fn spawn_observations(
    bridge: Arc<Bridge>,
    request_id: &str,
    observations: &[Observation],
    deadline: Instant,
) {
    if observations.is_empty() {
        return;
    }
    let request_id = request_id.to_owned();
    let observations = observations.to_vec();
    let permit = observing().clone().try_acquire_owned().ok();
    tokio::spawn(async move {
        let results = match permit {
            Some(_permit) => {
                futures_util::future::join_all(
                    observations.iter().map(|observation| observe(observation, deadline)),
                )
                .await
            }
            None => observations
                .iter()
                .map(|observation| json!({"index": observation.index, "skipped": "observation capacity is full"}))
                .collect(),
        };
        let argument = compact_json(&json!({"request_id": request_id, "results": results}));
        // A failed callback loses only content-free observation records.
        let _ = bridge.call("guardrail_observation_results", argument).await;
    });
}

async fn observe(observation: &Observation, deadline: Instant) -> Value {
    let Some(classifier) = lookup(observation.handle) else {
        return json!({"index": observation.index, "skipped": "classifier was replaced"});
    };
    match classifier.classify(&observation.window, deadline).await {
        Ok(answer) => {
            json!({"index": observation.index, "answer": classifier.answer_json(&answer)})
        }
        Err(error) => json!({"index": observation.index, "error": error.reason()}),
    }
}

/// Block the calling thread on one classification; for callers outside the data plane.
pub(crate) fn classify_blocking(
    classifier: &Arc<Classifier>,
    window: &str,
    deadline: Instant,
) -> Result<Answer, ClassifyError> {
    runtime().block_on(classifier.classify(window, deadline))
}

#[cfg(test)]
#[path = "classify_tests.rs"]
mod tests;
