//! Decision-model scoring and content-free, bounded per-authority verdict caches.

use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

type Scores = Vec<(String, f64)>;
#[derive(Default)]
struct Cache {
    entries: HashMap<String, (Instant, u64, Scores)>,
    order: BTreeMap<u64, String>,
    serial: u64,
}

fn caches() -> &'static Mutex<HashMap<String, Cache>> {
    static CACHES: OnceLock<Mutex<HashMap<String, Cache>>> = OnceLock::new();
    CACHES.get_or_init(Mutex::default)
}
fn get(adapter: &Value) -> Option<Scores> {
    let mut all = caches().lock().ok()?;
    let cache = all.get_mut(adapter["cache_scope"].as_str()?)?;
    let key = adapter["cache_key"].as_str()?;
    let (expires, serial, scores) = cache.entries.remove(key)?;
    cache.order.remove(&serial);
    if Instant::now() >= expires {
        return None;
    }
    cache.serial += 1;
    cache
        .entries
        .insert(key.into(), (expires, cache.serial, scores.clone()));
    cache.order.insert(cache.serial, key.into());
    Some(scores)
}
fn put(adapter: &Value, scores: &Scores) -> Result<(), ()> {
    let max = adapter["cache"]["max_entries"]
        .as_u64()
        .filter(|n| *n <= 1_000_000)
        .ok_or(())? as usize;
    let ttl = adapter["cache"]["ttl_seconds"]
        .as_f64()
        .filter(|n| n.is_finite() && *n > 0.0 && *n <= 86_400.0)
        .ok_or(())?;
    if max == 0 {
        return Ok(());
    }
    let scope = adapter["cache_scope"].as_str().ok_or(())?;
    let key = adapter["cache_key"].as_str().ok_or(())?;
    let mut all = caches().lock().map_err(|_| ())?;
    // A configuration rotation cannot retain unbounded retired adapter caches.
    if all.len() >= 256 && !all.contains_key(scope) {
        all.clear();
    }
    let cache = all.entry(scope.into()).or_default();
    if let Some((_, old, _)) = cache.entries.remove(key) {
        cache.order.remove(&old);
    }
    cache.serial += 1;
    cache.entries.insert(
        key.into(),
        (
            Instant::now() + Duration::from_secs_f64(ttl),
            cache.serial,
            scores.clone(),
        ),
    );
    cache.order.insert(cache.serial, key.into());
    while cache.entries.len() > max {
        if let Some((_, key)) = cache.order.pop_first() {
            cache.entries.remove(&key);
        }
    }
    Ok(())
}

pub(super) async fn inspect(adapter: &Value, budget: Duration) -> Result<Value, ()> {
    let cached = get(adapter);
    let hit = cached.is_some();
    let scores = match cached {
        Some(scores) => scores,
        None => {
            let timeout = adapter["timeout_ms"]
                .as_u64()
                .filter(|n| (1..=30_000).contains(n))
                .ok_or(())?;
            let response = super::inspection_http::inspect_async(
                adapter,
                adapter["payload"].clone(),
                budget.min(Duration::from_millis(timeout)),
            )
            .await?;
            let scores = score(adapter, &response)?;
            put(adapter, &scores)?;
            scores
        }
    };
    let mut flagged = false;
    for (name, score) in &scores {
        let threshold = adapter["categories"][name]["threshold"]
            .as_f64()
            .ok_or(())?;
        flagged |= *score >= threshold;
    }
    Ok(
        json!({"flagged": flagged, "scores": scores, "cached": hit, "subject_sha256": adapter["subject_sha256"]}),
    )
}

pub(super) fn score(adapter: &Value, response: &Value) -> Result<Scores, ()> {
    let answers = response["answers"].as_object().ok_or(())?;
    let categories = adapter["categories"].as_object().ok_or(())?;
    let mut scores = Vec::with_capacity(categories.len());
    for (name, category) in categories {
        let questions = category["questions"]
            .as_object()
            .filter(|q| !q.is_empty())
            .ok_or(())?;
        let mut values = Vec::with_capacity(questions.len());
        for id in questions.keys() {
            let answer = answers.get(id).filter(|v| v.is_object()).ok_or(())?;
            if answer.get("type").is_some_and(|v| v != "noul") {
                return Err(());
            }
            let p = answer["noul"]
                .as_f64()
                .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
                .ok_or(())?;
            values.push(p);
        }
        let value = match category["aggregate"].as_str() {
            Some("max") => values.iter().copied().fold(0.0, f64::max),
            Some("mean") => precise_sum(&values) / values.len() as f64,
            _ => return Err(()),
        };
        scores.push((name.clone(), value));
    }
    Ok(scores)
}

// Nonnegative finite probabilities permit an error-free partial sum with the
// same final tie correction as math.fsum; ordinary summation shifts thresholds.
fn precise_sum(values: &[f64]) -> f64 {
    let mut partials = Vec::<f64>::new();
    for &value in values {
        let mut x = value;
        let mut n = 0;
        for i in 0..partials.len() {
            let mut y = partials[i];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[n] = lo;
                n += 1;
            }
            x = hi;
        }
        partials.truncate(n);
        partials.push(x);
    }
    let mut hi = partials.pop().unwrap_or(0.0);
    let mut lo = 0.0;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if partials
        .last()
        .is_some_and(|p| (lo < 0.0 && *p < 0.0) || (lo > 0.0 && *p > 0.0))
    {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}
