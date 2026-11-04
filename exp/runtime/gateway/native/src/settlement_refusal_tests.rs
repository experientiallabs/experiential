//! A failed terminal write answers with the control plane's own public error.

use super::*;
use pyo3::prelude::*;

#[tokio::test]
async fn a_refused_terminal_write_answers_with_the_control_planes_error() {
    Python::initialize();
    let plane = Python::attach(|py| {
        pyo3::types::PyModule::from_code(py, c"class Refused(Exception):\n public_error_json = '{\"status_code\":503,\"code\":\"accounting_refused\",\"message\":\"Request accounting refused this request. Retry it.\",\"error_type\":\"api_error\",\"param\":null,\"retry_after_seconds\":1}'\nclass Plane:\n def __init__(self): self.settles = 0\n def settle(self, argument):\n  self.settles += 1\n  raise Refused()\n def abandon(self, argument): raise RuntimeError('ledger write lost')\n def close_thread_resources(self, argument): return '{}'\n", c"refused_plane.py", c"refused_plane")
            .unwrap().getattr("Plane").unwrap().call0().unwrap().unbind()
    });
    let bridge = Arc::new(Bridge::new(Python::attach(|py| plane.clone_ref(py)), 1).unwrap());
    let mut guard = AttemptGuard::new(
        bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "request".into(),
        Instant::now(),
    );
    assert_eq!(guard.settlement_error().code, "internal_error");
    guard.rebind("attempt".into());
    guard.mark_opened();
    assert!(!guard.settle("completed", None, &[], None, true).await);
    let refused = guard.settlement_error();
    assert_eq!(
        (
            refused.status_code,
            refused.code.as_str(),
            refused.retry_after_seconds
        ),
        (503, "accounting_refused", Some(1))
    );
    drop(guard);
    Python::attach(|py| {
        let settles: usize = plane
            .bind(py)
            .getattr("settles")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(settles, 4);
    });

    // A write that fails without a public error still answers the internal error.
    let mut accepted = AttemptGuard::new(
        bridge,
        Arc::new(AtomicUsize::new(0)),
        "request-without-attempt".into(),
        Instant::now(),
    );
    assert!(
        !accepted
            .abandon(&Failure::new(FailureClass::Cancelled, "cancelled"))
            .await
    );
    assert_eq!(accepted.settlement_error().code, "internal_error");
}
