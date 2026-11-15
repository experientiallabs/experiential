use super::*;
use pyo3::prelude::*;

/// A bounded write fixture that records the terminal and can withhold its acknowledgement.
fn ordering_plane(settle_fails: bool) -> Py<PyAny> {
    Python::initialize();
    Python::attach(|py| {
        let plane = pyo3::types::PyModule::from_code(py, c"import json, threading\nclass Plane:\n def __init__(self):\n  self.calls = []; self.fail = False; self.entered = threading.Event(); self.release = threading.Event(); self.release.set()\n def price(self, argument): raise AssertionError('a quote cannot authorize completion')\n def settle(self, argument):\n  self.calls.append(['settle', json.loads(argument)]); self.entered.set()\n  if not self.release.wait(5): raise RuntimeError('bounded fixture acknowledgement timeout')\n  if self.fail: raise RuntimeError('settlement unavailable')\n  return '{}'\n def close_thread_resources(self, argument): return '{}'\n", c"ordering_plane.py", c"ordering_plane")
            .unwrap().getattr("Plane").unwrap().call0().unwrap();
        plane.setattr("fail", settle_fails).unwrap();
        plane.unbind()
    })
}

/// Return the recorded `[method, argument]` calls of an ordering plane.
fn recorded_calls(plane: &Py<PyAny>) -> Vec<(String, Value)> {
    let calls: String = Python::attach(|py| {
        py.import("json")
            .unwrap()
            .call_method1("dumps", (plane.bind(py).getattr("calls").unwrap(),))
            .unwrap()
            .extract()
            .unwrap()
    });
    crate::lossless_json::from_str::<Vec<(String, Value)>>(&calls).unwrap()
}

/// A guard bound to one dispatched attempt that observed final usage.
fn terminal_guard(plane: &Py<PyAny>) -> AttemptGuard {
    let bridge = Arc::new(Bridge::scripted(Python::attach(|py| plane.clone_ref(py)), 4).unwrap());
    let mut guard = AttemptGuard::new(
        bridge,
        Arc::new(AtomicUsize::new(0)),
        "request".into(),
        Instant::now(),
    );
    guard.rebind("attempt".into());
    guard.mark_opened();
    guard.begin_dial_observation().record(&Event::Usage(Usage {
        input_tokens: Some(21),
        output_tokens: Some(5),
        ..Usage::default()
    }));
    guard
}

#[tokio::test]
async fn a_terminal_response_requires_a_durable_settlement_acknowledgement() {
    let plane = ordering_plane(false);
    let mut guard = terminal_guard(&plane);
    assert!(
        guard
            .settle_before_terminal("completed", None, &[], None)
            .await
    );
    let calls = recorded_calls(&plane);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "settle", "a quote is not a durable write");
    assert_eq!(calls[0].1["usage"]["input_tokens"], 21);
    assert_eq!(calls[0].1["outcome"], "completed");
    assert!(
        !guard.armed,
        "completion must not leave in-process financial debt"
    );
    assert!(guard.settle_cancelled(None, &[]).await);
    drop(guard);
    assert_eq!(
        recorded_calls(&plane).len(),
        1,
        "delivery cancellation cannot replace or repeat a completed settlement"
    );
}

#[tokio::test]
async fn a_failed_durable_write_never_authorizes_terminal_delivery() {
    let plane = ordering_plane(true);
    let mut guard = terminal_guard(&plane);
    assert!(
        !guard
            .settle_before_terminal("incomplete", None, &[], None)
            .await
    );
    let calls = recorded_calls(&plane);
    assert!(calls.len() > 1, "bounded write retries were exercised");
    assert!(calls
        .iter()
        .all(|(method, argument)| method == "settle" && argument == &calls[0].1));
    assert_eq!(calls[0].1["outcome"], "incomplete");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_terminal_waits_while_the_durable_write_is_unacknowledged() {
    let plane = ordering_plane(false);
    Python::attach(|py| {
        plane
            .bind(py)
            .getattr("release")
            .unwrap()
            .call_method0("clear")
            .unwrap();
    });
    let mut guard = terminal_guard(&plane);
    let task = tokio::spawn(async move {
        guard
            .settle_before_terminal("completed", None, &[], None)
            .await
    });
    let mut entered = false;
    for _ in 0..200 {
        entered = Python::attach(|py| {
            plane
                .bind(py)
                .getattr("entered")
                .unwrap()
                .call_method0("is_set")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        });
        if entered {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let completed_before_ack = task.is_finished();
    Python::attach(|py| {
        plane
            .bind(py)
            .getattr("release")
            .unwrap()
            .call_method0("set")
            .unwrap();
    });
    assert!(
        entered,
        "the write never reached its acknowledgement barrier"
    );
    assert!(
        !completed_before_ack,
        "completion escaped before the durable acknowledgement"
    );
    assert!(task.await.unwrap());
    assert_eq!(recorded_calls(&plane).len(), 1);
}

#[tokio::test]
async fn an_explicitly_unacknowledged_abandon_is_retried_and_never_delivered() {
    Python::initialize();
    let plane = Python::attach(|py| {
        pyo3::types::PyModule::from_code(py,
            c"import json\nclass Plane:\n def __init__(self): self.calls = 0\n def abandon(self, argument):\n  self.calls += 1\n  return '{\"terminalized\":false}'\n def close_thread_resources(self, argument): return '{}'\n",
            c"unacknowledged.py", c"unacknowledged")
            .unwrap().getattr("Plane").unwrap().call0().unwrap().unbind()
    });
    let bridge = Arc::new(Bridge::scripted(Python::attach(|py| plane.clone_ref(py)), 1).unwrap());
    let mut guard = AttemptGuard::new(
        bridge,
        Arc::new(AtomicUsize::new(0)),
        "request".into(),
        Instant::now(),
    );
    assert!(
        !guard
            .abandon(&Failure::new(
                FailureClass::Guardrail,
                "Input was rejected."
            ))
            .await
    );
    assert_eq!(guard.settlement_error().code, "accounting_unavailable");
    Python::attach(|py| {
        assert_eq!(
            plane
                .bind(py)
                .getattr("calls")
                .unwrap()
                .extract::<u32>()
                .unwrap(),
            4
        )
    });
}

#[test]
fn acknowledgement_requires_a_typed_write_result() {
    for value in [
        Value::Null,
        json!([]),
        json!(true),
        json!({"terminalized": false}),
        json!({"terminalized": null}),
    ] {
        assert!(acknowledged("settle", &value).is_err());
        assert!(acknowledged("abandon", &value).is_err());
    }
    assert!(acknowledged("abandon", &json!({})).is_err());
    assert!(acknowledged("abandon", &json!({"terminalized": true})).is_ok());
    assert!(acknowledged("settle", &json!({})).is_ok());
}
