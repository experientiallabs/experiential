use super::*;
use crate::capture::record::{Protocol, Scope};
use serde_json::json;
use std::sync::{mpsc, Arc};

struct MemorySink(mpsc::Sender<Record>);

impl Sink for MemorySink {
    fn write(&mut self, encoded: &str) -> Result<(), ()> {
        self.0
            .send(serde_json::from_str(encoded).map_err(|_| ())?)
            .map_err(|_| ())
    }
}

fn config() -> Configuration {
    Configuration {
        delivery: Limits {
            maximum_records: 8,
            maximum_bytes: 65536,
            maximum_record_bytes: 8192,
        },
        maximum_pending_records: 4,
        maximum_pending_bytes: 65536,
        maximum_request_bytes: 4096,
        maximum_response_bytes: 4096,
        ttl_seconds: 30,
        settlement_required: true,
    }
}

fn request(id: &str) -> Request {
    Request {
        request_id: id.to_owned(),
        scope: Scope {
            organization_id: "org".into(),
            identity_id: "identity".into(),
            application_id: "alias".into(),
        },
        protocol: Protocol::ChatCompletions,
        model_id: Some("model".into()),
        context: json!({"schema_version":1,"request":{"messages":[{"role":"user","content":"task"}],"tools":[{"name":"search"}]}}),
    }
}

fn response() -> Response {
    Response::Json {
        status: 200,
        body: json!({"id":"completion","choices":[]}),
    }
}

fn collector(config: Configuration) -> (Arc<Collector>, mpsc::Receiver<Record>) {
    let (sender, receiver) = mpsc::channel();
    (
        Arc::new(Collector::new(config, MemorySink(sender)).unwrap()),
        receiver,
    )
}

fn drain(collector: &Collector, receiver: mpsc::Receiver<Record>) -> Vec<Record> {
    assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
    receiver.try_iter().collect()
}

#[test]
fn routing_provenance_is_optional_until_selected_and_then_immutable() {
    let (collector, receiver) = collector(config());
    let mut input = request("request");
    input.model_id = None;
    assert!(collector.begin(input.clone()));
    collector.select_model("request", "selected");
    collector.select_model("request", "replacement");
    collector.settle("request", true, false);
    input.request_id = "rejected".into();
    assert!(collector.begin(input));
    collector.settle("rejected", true, false);
    let records = drain(&collector, receiver);
    assert_eq!(records[0].request.model_id.as_deref(), Some("selected"));
    assert_eq!(records[1].request.model_id, None);
}

#[test]
fn idle_maintenance_expires_pending_content_without_another_request() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector
        .pending
        .lock()
        .unwrap()
        .entries
        .get_mut("request")
        .unwrap()
        .expires = Instant::now();
    let until = Instant::now() + Duration::from_secs(3);
    while !collector.pending.lock().unwrap().entries.is_empty() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(collector.pending.lock().unwrap().entries.is_empty());
    assert_eq!(collector.counts()[5], 1);
    assert!(drain(&collector, receiver).is_empty());
}

#[test]
fn response_before_settlement_is_held_and_byok_never_writes_any_content() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    assert!(collector.attach("request"));
    collector.finish("request", Some(response()), Some("deployment".into()));
    assert!(receiver.try_recv().is_err());
    collector.settle("request", false, false);
    assert!(drain(&collector, receiver).is_empty());
}

#[test]
fn settlement_before_response_captures_prompt_then_one_response_update() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector.settle("request", true, true);
    assert!(collector.attach("request"));
    assert!(!collector.attach("request"));
    collector.finish("request", Some(response()), Some("deployment".into()));
    collector.finish("request", Some(response()), None);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 2);
    assert!(records[0].response.is_none());
    assert!(records[1].response.is_some());
    assert_eq!(records[1].request.scope.identity_id, "identity");
    assert_eq!(records[1].deployment_id.as_deref(), Some("deployment"));
}

#[test]
fn response_before_eligible_settlement_emits_one_complete_record() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert!(records[0].response.is_some());
}

#[test]
fn failed_host_requests_keep_only_the_permitted_prompt() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, false);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert!(records[0].response.is_none());
}

#[test]
fn unscoped_unknown_duplicate_and_expired_content_is_not_persisted() {
    let (collector, receiver) = collector(config());
    let mut invalid = request("invalid");
    invalid.scope.identity_id.clear();
    assert!(!collector.begin(invalid));
    assert!(collector.begin(request("request")));
    assert!(!collector.begin(request("request")));
    collector
        .pending
        .lock()
        .unwrap()
        .entries
        .get_mut("request")
        .unwrap()
        .expires = Instant::now();
    collector.settle("request", true, true);
    collector.finish("unknown", Some(response()), None);
    assert!(drain(&collector, receiver).is_empty());
    assert_eq!(collector.counts()[5], 3);
}

#[test]
fn pending_count_and_bytes_are_bounded_without_evicting_other_live_requests() {
    let mut configuration = config();
    configuration.maximum_pending_records = 1;
    let (collector, receiver) = collector(configuration);
    assert!(collector.begin(request("first")));
    assert!(!collector.begin(request("overflow")));
    collector.settle("first", true, false);
    assert!(collector.begin(request("next")));
    collector.settle("next", false, false);
    assert_eq!(drain(&collector, receiver).len(), 1);
}

#[test]
fn local_capture_does_not_require_hosted_settlement() {
    let mut configuration = config();
    configuration.settlement_required = false;
    let (collector, receiver) = collector(configuration);
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    assert_eq!(drain(&collector, receiver).len(), 1);
}
