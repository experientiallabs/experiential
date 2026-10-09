//! Collector retention of gateway-requested token probabilities.
use super::super::tests::{collector, config, drain, request, response};
use super::super::*;
use crate::capture::record::Response;
use serde_json::json;

/// Room for probabilities: a record ceiling well above request plus response.
fn logprobs_config() -> Configuration {
    let mut config = config();
    config.capture_logprobs = true;
    config.delivery.maximum_record_bytes = 65536;
    config.delivery.maximum_bytes = 4 * 65536;
    config
}

fn probability(token: &str) -> crate::logprobs::ChoiceLogprobsDelta {
    crate::logprobs::parse(
        Some(&json!({"content":[{"token":token,"logprob":-1.0,"bytes":null,"top_logprobs":[]}]})),
        0,
    )
    .unwrap()
    .unwrap()
}

#[test]
fn injected_probabilities_are_retained_and_marked_without_touching_the_response() {
    let (collector, receiver) = collector(logprobs_config());
    assert!(collector.begin(request("probabilities")));
    assert!(collector.wants_logprobs("probabilities"));
    assert!(!collector.wants_logprobs("unknown"));
    assert!(collector.wants_logprobs("probabilities"));
    collector.logprobs_injected("probabilities");
    collector.logprobs(
        "probabilities",
        vec![probability("a"), probability("b")],
        false,
    );
    collector.settle("probabilities", true, true);
    assert!(collector.finish("probabilities", Some(response()), Some("d".into())));
    let records = drain(&collector, receiver);
    let captured = records[0].provider_logprobs.as_ref().unwrap();
    assert!(captured.logprobs_injected);
    assert!(!captured.truncated);
    let tokens: Vec<&str> = captured.content.iter().map(|r| r.token.as_str()).collect();
    assert_eq!(tokens, ["a", "b"]);
    let encoded = serde_json::to_value(&records[0]).unwrap();
    assert_eq!(
        encoded["provider_logprobs"]["logprobs_injected"],
        json!(true)
    );
    assert_eq!(
        encoded["response"],
        serde_json::to_value(response()).unwrap()
    );
}

#[test]
fn records_without_injection_serialize_exactly_as_before() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("plain")));
    assert!(!collector.wants_logprobs("plain"));
    // Without the injected mark, stray probabilities are never retained.
    collector.logprobs("plain", vec![probability("a")], false);
    collector.settle("plain", true, true);
    assert!(collector.finish("plain", Some(response()), Some("d".into())));
    let records = drain(&collector, receiver);
    let encoded = serde_json::to_value(&records[0]).unwrap();
    assert!(encoded.get("provider_logprobs").is_none());
}

#[test]
fn probability_overflow_truncates_instead_of_dropping_the_exchange() {
    let (collector, receiver) = collector(logprobs_config());
    assert!(collector.begin(request("overflow")));
    assert!(collector.wants_logprobs("overflow"));
    collector.logprobs_injected("overflow");
    let many: Vec<_> = (0..400).map(|i| probability(&format!("t{i}"))).collect();
    collector.logprobs("overflow", many, false);
    collector.settle("overflow", true, true);
    assert!(collector.finish("overflow", Some(response()), Some("d".into())));
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert!(records[0].response.is_some());
    let captured = records[0].provider_logprobs.as_ref().unwrap();
    assert!(captured.truncated);
    assert!(captured.content.len() < 400);
}

#[test]
fn denied_response_retention_drops_captured_probabilities() {
    let (collector, receiver) = collector(logprobs_config());
    assert!(collector.begin(request("denied-probabilities")));
    assert!(collector.wants_logprobs("denied-probabilities"));
    collector.logprobs_injected("denied-probabilities");
    collector.logprobs("denied-probabilities", vec![probability("a")], false);
    collector.settle("denied-probabilities", true, false);
    let records = drain(&collector, receiver);
    assert!(records[0].response.is_none());
    assert!(records[0].provider_logprobs.is_none());
}

#[test]
fn probabilities_never_outgrow_the_destination_record_ceiling() {
    // The default test ceiling (8 KiB) leaves no room beside a full response.
    let (collector, receiver) = collector(Configuration {
        capture_logprobs: true,
        ..config()
    });
    assert!(collector.begin(request("ceiling")));
    assert!(collector.wants_logprobs("ceiling"));
    collector.logprobs_injected("ceiling");
    collector.logprobs("ceiling", vec![probability("a")], false);
    collector.settle("ceiling", true, true);
    assert!(collector.finish("ceiling", Some(response()), Some("d".into())));
    let records = drain(&collector, receiver);
    let captured = records[0].provider_logprobs.as_ref().unwrap();
    assert!(captured.content.is_empty());
    assert!(captured.truncated);
}

#[test]
fn injection_is_declined_when_its_marker_cannot_be_reserved() {
    let mut config = logprobs_config();
    let (probe, _) = collector(config.clone());
    assert!(probe.begin(request("size")));
    config.maximum_pending_bytes = probe
        .retained_bytes(&probe.pending.lock().unwrap())
        .max(config.maximum_response_bytes);
    config.maximum_request_bytes = config
        .maximum_request_bytes
        .min(config.maximum_pending_bytes);
    let (collector, _) = collector(config);
    assert!(collector.begin(request("full")));
    let room = collector.retained_bytes(&collector.pending.lock().unwrap())
        + crate::capture::logprobs::Captured::default().heap_bytes()
        <= collector.config.maximum_pending_bytes;
    assert_eq!(collector.wants_logprobs("full"), room);
    let reserved = collector.pending.lock().unwrap().entries["full"]
        .record
        .provider_logprobs
        .is_some();
    assert_eq!(reserved, room);
}

#[test]
fn a_reserved_but_unused_sidecar_is_never_serialized() {
    let (collector, receiver) = collector(logprobs_config());
    assert!(collector.begin(request("reserved")));
    assert!(collector.wants_logprobs("reserved"));
    collector.logprobs("reserved", vec![probability("a")], false);
    collector.settle("reserved", true, true);
    assert!(collector.finish("reserved", Some(response()), Some("d".into())));
    let records = drain(&collector, receiver);
    let encoded = serde_json::to_value(&records[0]).unwrap();
    assert!(encoded.get("provider_logprobs").is_none());
}

#[test]
fn probabilities_stay_only_beside_a_complete_captured_response() {
    let (collector, receiver) = collector(logprobs_config());
    assert!(collector.begin(request("prefix")));
    assert!(collector.wants_logprobs("prefix"));
    collector.logprobs_injected("prefix");
    collector.logprobs("prefix", vec![probability("a")], false);
    collector.settle("prefix", true, true);
    let truncated = Response::Sse {
        status: 200,
        frames: vec![json!({"choices":[]})],
        truncated: true,
        client_disconnected: false,
        source_json: None,
    };
    assert!(collector.finish("prefix", Some(truncated), Some("d".into())));
    let records = drain(&collector, receiver);
    assert!(records[0].response.is_some());
    assert!(records[0].provider_logprobs.is_none());
}

#[test]
fn a_stream_the_client_left_early_drops_its_probabilities() {
    let (collector, receiver) = collector(logprobs_config());
    assert!(collector.begin(request("left")));
    assert!(collector.wants_logprobs("left"));
    collector.logprobs_injected("left");
    collector.logprobs("left", vec![probability("a")], false);
    collector.settle("left", true, true);
    let left = Response::Sse {
        status: 200,
        frames: vec![json!({"choices":[]})],
        truncated: false,
        client_disconnected: true,
        source_json: None,
    };
    assert!(collector.finish("left", Some(left), Some("d".into())));
    let records = drain(&collector, receiver);
    assert!(records[0].provider_logprobs.is_none());
}

#[test]
fn a_failed_exchange_drops_its_probabilities() {
    for failed in [
        Response::Json {
            status: 400,
            body: json!({"error": {"message": "bad"}}),
            source_json: None,
        },
        Response::Json {
            status: 200,
            body: json!({"error": {"message": "late"}}),
            source_json: None,
        },
        Response::Sse {
            status: 200,
            frames: vec![json!({"choices":[]}), json!({"error": {"message": "late"}})],
            truncated: false,
            client_disconnected: false,
            source_json: None,
        },
    ] {
        let (collector, receiver) = collector(logprobs_config());
        assert!(collector.begin(request("failed")));
        assert!(collector.wants_logprobs("failed"));
        collector.logprobs_injected("failed");
        collector.logprobs("failed", vec![probability("a")], false);
        collector.settle("failed", true, true);
        assert!(collector.finish("failed", Some(failed), Some("d".into())));
        let records = drain(&collector, receiver);
        assert!(records[0].provider_logprobs.is_none());
    }
}
