use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::*;
use crate::accounting::ownership::Driver;

#[derive(Default)]
struct Ledger {
    fail: AtomicBool,
    writes: AtomicUsize,
    inputs: Mutex<Vec<Value>>,
}

impl Driver for Ledger {
    fn call(&self, _: &str, input: &Value) -> Result<Value, PublicError> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.inputs.lock().unwrap().push(input.clone());
        if self.fail.load(Ordering::SeqCst) {
            return Err(PublicError::internal());
        }
        Ok(json!({"terminalized": true}))
    }
}

fn registry() -> Arc<Registry> {
    Arc::new(Registry {
        shared: Arc::new(Shared::new(Policy::default(), None, 10.0, 100).unwrap()),
        reports: Arc::default(),
        requests: Mutex::new(Requests::default()),
        changed: Condvar::new(),
    })
}

fn register(registry: &Arc<Registry>, id: &str, org: &str, ledger: Arc<Ledger>) {
    registry
        .register(Arc::new(Entry {
            resources: Arc::new(super::super::resources::Resources::default()),
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
fn a_late_meter_that_fails_after_retirement_is_recovered_even_while_closing() {
    let registry = registry();
    let ledger = Arc::new(Ledger::default());
    register(&registry, "request", "tenant", ledger.clone());
    let pin = registry.retain_request("request").unwrap();
    registry
        .call("abandon", "{\"request_id\":\"request\"}")
        .unwrap();
    assert!(registry.drained());
    ledger.fail.store(true, Ordering::SeqCst);
    assert!(settle(&registry, "request").is_err());
    // The meter's owner is fenced and owned by recovery again, after the data plane lets go.
    drop(pin);
    assert!(!registry.drained());
    assert!(!registry.clear(Some("tenant")));
    registry.begin_close();
    ledger.fail.store(false, Ordering::SeqCst);
    assert_eq!(
        registry.recover_one(&HashSet::new()).as_deref(),
        Some("request")
    );
    assert!(registry.drained());
    let inputs = ledger.inputs.lock().unwrap();
    // The cancellation, the failed meter and the delivered meter: one meter, twice.
    assert_eq!(inputs.len(), 3);
    assert_eq!(inputs[1], inputs[2]);
    assert_eq!(inputs[2]["outcome"], "completed");
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

#[test]
fn failed_admission_cleanup_retains_its_native_owner_and_retry_before_deadline() {
    Python::initialize();
    for fail in [false, true] {
        let registry = registry();
        let ledger = Arc::new(Ledger::default());
        ledger.fail.store(fail, Ordering::SeqCst);
        register(&registry, "admission", "tenant", ledger.clone());
        let accounting = NativeAccounting {
            inner: registry.clone(),
            background_recovery: false,
        };
        assert_eq!(
            Python::attach(|py| accounting.abort_admission(py, "admission", 3)).is_err(),
            fail
        );
        assert_eq!(ledger.writes.load(Ordering::SeqCst), 1);
        assert!(registry.clear(Some("unrelated")));
        if fail {
            assert!(!registry.drained());
            assert!(!registry.clear(Some("tenant")));
            ledger.fail.store(false, Ordering::SeqCst);
            assert_eq!(
                registry.recover_one(&HashSet::new()).as_deref(),
                Some("admission")
            );
            assert_eq!(ledger.writes.load(Ordering::SeqCst), 2);
        }
        assert!(registry.drained());
        assert!(registry.clear(Some("tenant")));
        Python::attach(|py| accounting.abort_admission(py, "admission", 3)).unwrap();
        assert_eq!(
            ledger.writes.load(Ordering::SeqCst),
            if fail { 2 } else { 1 }
        );
        assert!(ledger
            .inputs
            .lock()
            .unwrap()
            .iter()
            .all(|input| input["web_search_requests"] == 3));
    }
}
