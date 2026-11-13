//! Bounded native affinity bindings, including the nonrenewable maximum age.

use hashlink::LinkedHashMap;

struct Binding {
    deployment: String,
    expires: f64,
    age_deadline: f64,
}

pub(super) struct Sticky {
    maximum: usize,
    bindings: LinkedHashMap<Vec<u8>, Binding>,
}

impl Sticky {
    pub fn new(maximum: usize) -> Option<Self> {
        (maximum > 0).then(|| Self {
            maximum,
            bindings: LinkedHashMap::new(),
        })
    }
    pub fn bind(&mut self, fingerprint: &[u8], deployment: &str, ttl: f64, now: f64) {
        if !ttl.is_finite() || ttl <= 0.0 {
            return;
        }
        let previous = self.bindings.remove(fingerprint);
        let age_deadline = previous
            .filter(|b| b.deployment == deployment)
            .map_or(now + 4.0 * ttl, |b| b.age_deadline);
        self.bindings.insert(
            fingerprint.to_vec(),
            Binding {
                deployment: deployment.into(),
                expires: (now + ttl).min(age_deadline),
                age_deadline,
            },
        );
        while self.bindings.len() > self.maximum {
            self.bindings.pop_front();
        }
    }
    pub fn bound(&mut self, fingerprint: &[u8], now: f64) -> Option<&str> {
        if self
            .bindings
            .get(fingerprint)
            .is_some_and(|b| b.expires <= now)
        {
            self.clear(fingerprint);
        }
        self.bindings
            .get(fingerprint)
            .map(|b| b.deployment.as_str())
    }
    pub fn clear(&mut self, fingerprint: &[u8]) {
        self.bindings.remove(fingerprint);
    }
    pub fn len(&self) -> usize {
        self.bindings.len()
    }
}
