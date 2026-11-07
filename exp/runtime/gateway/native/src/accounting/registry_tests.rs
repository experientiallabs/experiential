use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::*;
use crate::accounting::ownership::Driver;

#[derive(Default)]
struct Ledger {
    fail: AtomicBool,
    writes: AtomicUsize,
}

impl Driver for Ledger {
    fn call(&self, _: &str, _: &Value) -> Result<Value, PublicError> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        if self.fail.load(Ordering::SeqCst) {
            return Err(PublicError::internal());
        }
        Ok(json!({"terminalized": true}))
    }
}

fn registry() -> Arc<Registry> {
    Arc::new(Registry {
        shared: Arc::new(Shared::new(Policy::default(), None, 10.0, 100).unwrap()),
        requests: Mutex::new(Requests::default()),
        changed: Condvar::new(),
    })
}

fn register(registry: &Arc<Registry>, id: &str, org: &str, ledger: Arc<Ledger>) {
    registry
        .register(Arc::new(Entry {
            owner: Owner::new(
                id.into(),
                org.into(),
                Instant::now() + Duration::from_secs(60),
                ledger,
            ),
            registry: Arc::downgrade(registry),
        }))
        .unwrap();
}

fn settle(registry: &Registry, id: &str) -> Result<String, PublicError> {
    registry.call(
        "settle",
        &json!({"request_id": id, "attempt_id": "attempt",
        "outcome": "completed", "finalize": true})
        .to_string(),
    )
}

#[test]
fn response_hooks_keep_their_owner_after_financial_retirement() {
    let registry = registry();
    let ledger = Arc::new(Ledger::default());
    register(&registry, "request", "tenant", ledger.clone());
    let pin = registry.retain_request("request").unwrap();
    settle(&registry, "request").unwrap();
    assert!(registry.drained());
    assert!(registry.clear(Some("tenant")));
    registry
        .call("remember", "{\"request_id\":\"request\"}")
        .unwrap();
    assert_eq!(ledger.writes.load(Ordering::SeqCst), 2);
    drop(pin);
    assert!(registry.lookup("request").is_none());
    assert!(registry.requests.lock().unwrap().live.is_empty());
}

#[test]
fn recovery_visits_distinct_owners_and_fences_only_the_affected_tenant() {
    let registry = registry();
    let ledger = Arc::new(Ledger::default());
    ledger.fail.store(true, Ordering::SeqCst);
    register(&registry, "request", "tenant", ledger.clone());
    assert!(settle(&registry, "request").is_err());
    assert!(!registry.clear(Some("tenant")));
    assert!(registry.clear(Some("unrelated")));
    assert_eq!(
        registry.recover_one(&HashSet::new()).as_deref(),
        Some("request")
    );
    assert!(registry
        .recover_one(&HashSet::from(["request".into()]))
        .is_none());
    assert_eq!(ledger.writes.load(Ordering::SeqCst), 2);
    ledger.fail.store(false, Ordering::SeqCst);
    registry.recover_one(&HashSet::new()).unwrap();
    assert!(registry.clear(Some("tenant")));
    assert!(registry.drained());
}

#[test]
fn closing_retains_failed_work_and_rejects_new_admissions_until_drained() {
    let registry = registry();
    let ledger = Arc::new(Ledger::default());
    ledger.fail.store(true, Ordering::SeqCst);
    register(&registry, "request", "tenant", ledger.clone());
    registry.begin_close();
    registry.recover_one(&HashSet::new()).unwrap();
    assert!(!registry.drained());
    assert!(!registry.clear(None));
    let owner = registry.lookup("request").unwrap();
    assert!(registry.register(owner.clone()).is_err());
    ledger.fail.store(false, Ordering::SeqCst);
    registry.recover_one(&HashSet::new()).unwrap();
    assert!(registry.drained());
    assert!(owner.owner.finished());
}

#[test]
fn admission_requires_a_registered_native_request_owner() {
    pyo3::Python::initialize();
    let registry = registry();
    let bridge = crate::bridge::Bridge::new(pyo3::Python::attach(|py| py.None()), 1)
        .unwrap()
        .with_native(Some(crate::native_callbacks::NativeTarget::new(
            registry.clone(),
        )));
    assert!(bridge
        .admitted(&json!({"request_id":"missing","native_accounting":true}))
        .is_err());
    assert!(bridge
        .admitted(&json!({"request_id":"postgres","native_accounting":false}))
        .is_ok());
    register(&registry, "owned", "org", Arc::new(Ledger::default()));
    assert!(bridge
        .admitted(&json!({"request_id":"owned","native_accounting":true}))
        .is_ok());
    assert!(bridge
        .admitted(&json!({"request_id":"owned","native_accounting":false}))
        .is_err());
}
