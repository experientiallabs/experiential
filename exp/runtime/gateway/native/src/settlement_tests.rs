use super::*;
use pyo3::prelude::*;

#[tokio::test]
async fn parsed_usage_wins_stale_consumer_usage_on_local_failure() {
    Python::initialize();
    let plane = Python::attach(|py| {
        pyo3::types::PyModule::from_code(py, c"import json\nclass Plane:\n def __init__(self): self.writes = []\n def settle(self, argument):\n  self.writes.append(json.loads(argument))\n  return '{}'\n def close_thread_resources(self, argument): return '{}'\n", c"stale_plane.py", c"stale_plane")
            .unwrap().getattr("Plane").unwrap().call0().unwrap().unbind()
    });
    let bridge = Arc::new(Bridge::new(Python::attach(|py| plane.clone_ref(py)), 1).unwrap());
    for (ordinal, fresh) in [Some(7), None].into_iter().enumerate() {
        let mut guard = AttemptGuard::new(
            bridge.clone(),
            Arc::new(AtomicUsize::new(0)),
            format!("request-{ordinal}"),
            Instant::now(),
        );
        guard.rebind(format!("attempt-{ordinal}"));
        guard.mark_opened();
        guard.begin_dial_observation().record(&Event::Usage(Usage {
            input_tokens: Some(13),
            output_tokens: fresh,
            ..Usage::default()
        }));
        let stale = Usage {
            input_tokens: Some(13),
            output_tokens: Some(0),
            ..Usage::default()
        };
        let failure = Failure::new(
            FailureClass::MalformedResponse,
            "provider response exceeded the allowed size",
        );
        assert!(
            guard
                .settle("failed", Some(&stale), &[], Some(&failure), true)
                .await
        );
    }
    let writes: String = Python::attach(|py| {
        py.import("json")
            .unwrap()
            .call_method1("dumps", (plane.bind(py).getattr("writes").unwrap(),))
            .unwrap()
            .extract()
            .unwrap()
    });
    let writes: Value = serde_json::from_str(&writes).unwrap();
    assert_eq!(writes[0]["usage"]["output_tokens"], 7);
    assert!(writes[1]["usage"]["output_tokens"].is_null());
}

#[tokio::test]
async fn observed_terminal_wins_disconnect_once_and_rebind_clears_facts() {
    Python::initialize();
    let plane = Python::attach(|py| {
        pyo3::types::PyModule::from_code(py, c"import json\nclass Plane:\n def __init__(self): self.writes = []\n def settle(self, argument):\n  self.writes.append(json.loads(argument))\n  return '{}'\n def close_thread_resources(self, argument): return '{}'\n", c"cancel_plane.py", c"cancel_plane")
            .unwrap().getattr("Plane").unwrap().call0().unwrap().unbind()
    });
    let bridge = Arc::new(Bridge::new(Python::attach(|py| plane.clone_ref(py)), 1).unwrap());
    let mut guard = AttemptGuard::new(
        bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "request".into(),
        Instant::now(),
    );
    guard.rebind("attempt".into());
    guard.mark_dispatched();
    let observation = guard.begin_dial_observation();
    observation.record(&Event::Usage(Usage {
        input_tokens: Some(19),
        output_tokens: Some(7),
        ..Usage::default()
    }));
    observation.record(&Event::Completed);
    assert!(guard.settle_cancelled(None, &[]).await);
    assert!(guard.settle_cancelled(None, &[]).await);
    drop(guard);
    let writes: String = Python::attach(|py| {
        py.import("json")
            .unwrap()
            .call_method1("dumps", (plane.bind(py).getattr("writes").unwrap(),))
            .unwrap()
            .extract()
            .unwrap()
    });
    let writes: Value = serde_json::from_str(&writes).unwrap();
    assert_eq!(writes.as_array().unwrap().len(), 1);
    assert_eq!(writes[0]["outcome"], "completed");
    assert_eq!(writes[0]["usage"]["output_tokens"], 7);
    assert_eq!(writes[0]["usage_incomplete_due_to_disconnect"], false);
    for (name, dispatched, opened) in [
        ("pre", false, false),
        ("headers", true, false),
        ("open", true, true),
    ] {
        let mut cancelled = AttemptGuard::new(
            bridge.clone(),
            Arc::new(AtomicUsize::new(0)),
            name.into(),
            Instant::now(),
        );
        cancelled.rebind(name.into());
        if dispatched {
            cancelled.mark_dispatched();
        }
        if opened {
            cancelled.mark_opened();
        }
        cancelled
            .begin_dial_observation()
            .record(&Event::Usage(Usage {
                input_tokens: Some(19),
                output_tokens: Some(0),
                ..Usage::default()
            }));
        assert!(cancelled.settle_cancelled(None, &[]).await);
        let latest: String = Python::attach(|py| {
            py.import("json")
                .unwrap()
                .call_method1(
                    "dumps",
                    (plane
                        .bind(py)
                        .getattr("writes")
                        .unwrap()
                        .get_item(-1)
                        .unwrap(),),
                )
                .unwrap()
                .extract()
                .unwrap()
        });
        let latest: Value = serde_json::from_str(&latest).unwrap();
        assert_eq!(latest["usage_incomplete_due_to_disconnect"], dispatched);
        assert_eq!(latest["dispatched"], dispatched);
        assert_eq!(latest["failure"]["failure_class"], "cancelled");
        assert_eq!(latest["usage"]["input_tokens"], 19);
    }
    let mut guard = AttemptGuard::new(
        bridge,
        Arc::new(AtomicUsize::new(0)),
        "next".into(),
        Instant::now(),
    );
    guard.rebind("one".into());
    let old = guard.begin_dial_observation();
    old.record(&Event::TextDelta("first physical attempt".into()));
    old.record(&Event::Completed);
    guard.rebind("two".into());
    assert!(guard.observation.snapshot().terminal.is_none());
    assert!(guard.observation.snapshot().streamed_output.text.is_empty());
    old.record(&Event::TextDelta("late first attempt".into()));
    assert!(guard.observation.snapshot().streamed_output.text.is_empty());
    guard.disarm_finalized("completed");
}

#[tokio::test]
async fn dispatched_cancellation_carries_streamed_output_only_without_a_terminal() {
    Python::initialize();
    let plane = Python::attach(|py| {
        pyo3::types::PyModule::from_code(py, c"import json\nclass Plane:\n def __init__(self): self.writes = []\n def settle(self, argument):\n  self.writes.append(json.loads(argument))\n  return '{}'\n def close_thread_resources(self, argument): return '{}'\n", c"streamed_plane.py", c"streamed_plane")
            .unwrap().getattr("Plane").unwrap().call0().unwrap().unbind()
    });
    let bridge = Arc::new(Bridge::new(Python::attach(|py| plane.clone_ref(py)), 1).unwrap());
    for (name, terminal) in [("cut", false), ("done", true)] {
        let mut guard = AttemptGuard::new(
            bridge.clone(),
            Arc::new(AtomicUsize::new(0)),
            name.into(),
            Instant::now(),
        );
        guard.rebind(name.into());
        guard.mark_dispatched();
        guard.mark_opened();
        let observation = guard.begin_dial_observation();
        observation.record(&Event::ReasoningContentDelta {
            route_sha256: "route".into(),
            delta: "let me think".into(),
        });
        observation.record(&Event::TextDelta("Hello, ".into()));
        observation.record(&Event::TextDelta("world".into()));
        if terminal {
            observation.record(&Event::Usage(Usage {
                input_tokens: Some(19),
                output_tokens: Some(7),
                ..Usage::default()
            }));
            observation.record(&Event::Completed);
        }
        assert!(guard.settle_cancelled(None, &[]).await);
    }
    let writes: String = Python::attach(|py| {
        py.import("json")
            .unwrap()
            .call_method1("dumps", (plane.bind(py).getattr("writes").unwrap(),))
            .unwrap()
            .extract()
            .unwrap()
    });
    let writes: Value = serde_json::from_str(&writes).unwrap();
    assert_eq!(writes[0]["usage_incomplete_due_to_disconnect"], true);
    assert!(writes[0]["usage"].is_null());
    assert_eq!(writes[0]["streamed_output"]["text"], "Hello, world");
    assert_eq!(writes[0]["streamed_output"]["reasoning"], "let me think");
    assert_eq!(writes[0]["streamed_output"]["text_overflow_chars"], 0);
    assert_eq!(writes[0]["streamed_output"]["reasoning_overflow_chars"], 0);
    assert_eq!(writes[0]["streamed_output"]["images"], 0);
    // An observed provider terminal is the final meter: nothing to estimate.
    assert_eq!(writes[1]["outcome"], "completed");
    assert_eq!(writes[1]["usage_incomplete_due_to_disconnect"], false);
    assert!(writes[1].get("streamed_output").is_none());
}

#[test]
fn rfc3339_formats_epoch_seconds_and_millis_in_utc() {
    assert_eq!(
        system_time_to_rfc3339(UNIX_EPOCH),
        "1970-01-01T00:00:00.000+00:00"
    );
    // A fixed instant with sub-second precision (1_700_000_000.500s).
    let at = UNIX_EPOCH + Duration::from_millis(1_700_000_000_500);
    assert_eq!(system_time_to_rfc3339(at), "2023-11-14T22:13:20.500+00:00");
    // A leap day exercises the civil-from-days month/day recovery.
    let leap = UNIX_EPOCH + Duration::from_secs(1_582_934_400);
    assert_eq!(
        system_time_to_rfc3339(leap),
        "2020-02-29T00:00:00.000+00:00"
    );
}

#[test]
fn settle_argument_retains_cache_write_counts_and_unknowns() {
    for count in [None, Some(0), Some(6108)] {
        let usage = Usage {
            input_tokens: Some(6119),
            cache_creation_input_tokens: count,
            ..Usage::default()
        };
        let argument = settle_argument(
            "req",
            "att",
            "completed",
            Some(&usage),
            &[],
            None,
            true,
            true,
            None,
            None,
            None,
            0,
            0,
        );
        let parsed: Value = serde_json::from_str(&argument).expect("valid json");
        assert_eq!(parsed["usage"]["cache_creation_input_tokens"], json!(count));
    }
}

#[test]
fn settle_argument_bills_web_searches_only_on_the_finalizing_settlement() {
    let settle = |finalize: bool, requests: u32| -> Value {
        let argument = settle_argument(
            "req",
            "att",
            "completed",
            None,
            &[],
            None,
            finalize,
            true,
            None,
            None,
            None,
            requests,
            0,
        );
        serde_json::from_str(&argument).expect("valid json")
    };
    assert_eq!(settle(true, 1)["web_search_requests"], json!(1));
    // A failed rung's non-finalizing settlement never bills the search a
    // second time, and an unsearched request omits the key entirely.
    assert!(settle(false, 1).get("web_search_requests").is_none());
    assert!(settle(true, 0).get("web_search_requests").is_none());
}

#[test]
fn settle_argument_bills_tool_search_rounds_only_on_the_finalizing_settlement() {
    let settle = |finalize: bool, rounds: u32| -> Value {
        let argument = settle_argument(
            "req",
            "att",
            "completed",
            None,
            &[],
            None,
            finalize,
            true,
            None,
            None,
            None,
            0,
            rounds,
        );
        serde_json::from_str(&argument).expect("valid json")
    };
    assert_eq!(settle(true, 2)["tool_search_requests"], json!(2));
    // The search-call turns settle non-finalizing and never bill the
    // rounds; a request that ran none omits the key entirely.
    assert!(settle(false, 2).get("tool_search_requests").is_none());
    assert!(settle(true, 0).get("tool_search_requests").is_none());
    assert!(settle(true, 0).get("web_search_requests").is_none());
}

#[test]
fn settle_argument_preserves_upstream_provider_and_cache_ttl_usage_together() {
    let usage = Usage {
        input_tokens: Some(1_000),
        output_tokens: Some(10),
        cached_input_tokens: Some(100),
        cache_creation_input_tokens: Some(600),
        cache_creation_1h_input_tokens: Some(200),
        reasoning_tokens: None,
    };
    let named = settle_argument(
        "req",
        "att",
        "completed",
        Some(&usage),
        &[],
        None,
        true,
        true,
        None,
        None,
        Some("Azure"),
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&named).expect("valid json");
    assert_eq!(
        parsed["upstream_provider"],
        Value::String("Azure".to_string())
    );
    assert_eq!(parsed["usage"]["input_tokens"], 1_000);
    assert_eq!(parsed["usage"]["cache_creation_input_tokens"], 600);
    assert_eq!(parsed["usage"]["cache_creation_1h_input_tokens"], 200);
    let unnamed = settle_argument(
        "req",
        "att",
        "completed",
        None,
        &[],
        None,
        true,
        true,
        None,
        None,
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&unnamed).expect("valid json");
    assert_eq!(parsed["upstream_provider"], Value::Null);
}

#[test]
fn settle_argument_carries_first_token_at_only_when_observed() {
    let observed = UNIX_EPOCH + Duration::from_millis(1_700_000_000_500);
    let with_token = settle_argument(
        "req",
        "att",
        "completed",
        None,
        &[],
        None,
        true,
        true,
        Some(observed),
        None,
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&with_token).expect("valid json");
    assert_eq!(
        parsed["first_token_at"],
        Value::String("2023-11-14T22:13:20.500+00:00".to_string())
    );
    // A non-streaming attempt observes no first token: the field is null,
    // matching the control plane's backward-compatible parse.
    let without = settle_argument(
        "req",
        "att",
        "completed",
        None,
        &[],
        None,
        true,
        true,
        None,
        None,
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&without).expect("valid json");
    assert_eq!(parsed["first_token_at"], Value::Null);
}

#[test]
fn settle_argument_carries_the_sanitized_provider_detail_on_a_failed_attempt() {
    let failure = Failure::new(FailureClass::InvalidRequest, "provider rejected")
        .with_provider_detail(Some(
            "max_tokens must be greater than thinking budget.".into(),
        ));
    let argument = settle_argument(
        "req",
        "att",
        "failed",
        None,
        &[],
        Some(&failure),
        true,
        true,
        None,
        None,
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&argument).expect("valid json");
    assert_eq!(parsed["failure"]["failure_class"], "invalid_request");
    assert_eq!(
        parsed["failure"]["provider_detail"],
        "max_tokens must be greater than thinking budget."
    );
    assert_eq!(parsed["failure"]["customer_owned"], false);
    let owned = crate::stream_errors::customer_credential_failure(
        crate::upstream::transport_failure(Some(401)),
        "openai",
    );
    let owned_argument = settle_argument(
        "req",
        "att",
        "failed",
        None,
        &[],
        Some(&owned),
        true,
        true,
        None,
        None,
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&owned_argument).expect("valid json");
    assert_eq!(
        parsed["failure"]["failure_class"],
        "provider_authentication"
    );
    assert_eq!(parsed["failure"]["customer_owned"], true);
    // A failure with no provider explanation carries an explicit null, which
    // the control plane parses back to None.
    let bare = Failure::new(FailureClass::ProviderInternal, "provider failed");
    let bare_argument = settle_argument(
        "req",
        "att",
        "failed",
        None,
        &[],
        Some(&bare),
        true,
        true,
        None,
        None,
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&bare_argument).expect("valid json");
    assert_eq!(parsed["failure"]["provider_detail"], Value::Null);
}

#[test]
fn settle_argument_carries_rate_limit_facts_when_harvested() {
    // A throttled open: the failure carries the harvested headers and the
    // integer Retry-After, both settled for the control plane.
    let mut headers = serde_json::Map::new();
    headers.insert("retry-after".to_string(), Value::String("3600".to_string()));
    headers.insert(
        "x-ratelimit-remaining-requests".to_string(),
        Value::String("0".to_string()),
    );
    let throttled = crate::upstream::transport_failure(Some(429))
        .with_rate_limit_facts(Some(headers.clone()), Some(3_600));
    let argument = settle_argument(
        "req",
        "att",
        "failed",
        None,
        &[],
        Some(&throttled),
        true,
        false,
        None,
        throttled.rate_limit_headers.as_deref(),
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&argument).expect("valid json");
    assert_eq!(parsed["failure"]["retry_after_seconds"], 3_600);
    assert_eq!(parsed["rate_limit_headers"]["retry-after"], "3600");
    assert_eq!(
        parsed["rate_limit_headers"]["x-ratelimit-remaining-requests"],
        "0"
    );
    // A successful attempt settles the opened response's headers; absent
    // headers settle an explicit null the control plane treats as absent.
    let success = settle_argument(
        "req",
        "att",
        "completed",
        None,
        &[],
        None,
        true,
        true,
        None,
        Some(&headers),
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&success).expect("valid json");
    assert_eq!(parsed["rate_limit_headers"]["retry-after"], "3600");
    let bare = settle_argument(
        "req",
        "att",
        "completed",
        None,
        &[],
        None,
        true,
        true,
        None,
        None,
        None,
        0,
        0,
    );
    let parsed: Value = serde_json::from_str(&bare).expect("valid json");
    assert_eq!(parsed["rate_limit_headers"], Value::Null);
}

#[test]
fn retry_after_fills_only_throttled_failures_and_never_overwrites() {
    let throttled =
        crate::upstream::transport_failure(Some(429)).with_rate_limit_facts(None, Some(30));
    assert_eq!(throttled.retry_after_seconds, Some(30));
    let already =
        Failure::new(FailureClass::Throttled, "throttled").with_rate_limit_facts(None, Some(30));
    let kept = Failure {
        retry_after_seconds: Some(7),
        ..already
    }
    .with_rate_limit_facts(None, Some(30));
    assert_eq!(kept.retry_after_seconds, Some(7));
    let internal =
        crate::upstream::transport_failure(Some(500)).with_rate_limit_facts(None, Some(30));
    assert_eq!(internal.retry_after_seconds, None);
}
