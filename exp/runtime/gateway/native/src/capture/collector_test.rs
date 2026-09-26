use super::*;
use crate::capture::record::{Protocol, Response, Scope};
use serde_json::json;
use std::sync::{mpsc, Arc};

struct MemorySink(mpsc::Sender<Record>);

impl Sink for MemorySink {
    type Prepared = String;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        maximum_record_bytes
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        record.encode(maximum_bytes).ok_or(())
    }

    fn write(&mut self, record: &Self::Prepared) -> Result<(), ()> {
        self.0
            .send(serde_json::from_str(record).map_err(|_| ())?)
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
        relay_metadata: false,
        truncate_request: false,
        asynchronous_delivery: true,
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
        context: Arc::new(
            json!({"schema_version":1,"request":{"messages":[{"role":"user","content":"task"}],"tools":[{"name":"search"}]}}),
        ),
    }
}

fn response() -> Response {
    Response::Json {
        status: 200,
        body: json!({"id":"completion","choices":[]}),
        source_json: None,
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

fn wait_for_handoffs(collector: &Collector) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while collector.handoff_bytes.load(Ordering::Acquire) != 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), 0);
}

#[test]
fn winning_metrics_and_gemini_parts_survive_storage_without_public_reasoning() {
    use crate::events::{Event, Usage};
    use crate::settlement::Observation;
    let (collector, receiver) = collector(config());
    let observation = Observation::default();
    assert!(collector.begin(request("measured")));
    collector.observe_attempt("measured", observation.clone());
    let part = Arc::new(json!({"thought":true,"text":"summary\0雪","thoughtSignature":"opaque=="}));
    collector.gemini_thought_part("measured", part.clone());
    {
        let pending = collector.pending.lock().unwrap();
        assert!(Arc::ptr_eq(
            &part,
            &pending.entries["measured"].record.gemini_thought_parts[0]
        ));
    }
    observation.record_first_token(Some(SystemTime::now()));
    observation.record(&Event::Usage(Usage {
        input_tokens: Some(9),
        output_tokens: Some(6),
        reasoning_tokens: Some(4),
        ..Usage::default()
    }));
    observation.record(&Event::Completed);
    collector.settle("measured", true, true);
    assert!(collector.finish("measured", Some(response()), Some("gemini".into())));
    let records = drain(&collector, receiver);
    let record = &records[0];
    assert!(record.provider_reasoning.is_none());
    let restored: Vec<serde_json::Value> =
        serde_json::from_str(record.gemini_thought_parts_source_json.as_ref().unwrap()).unwrap();
    assert_eq!(restored[0], *part);
    let metrics = record.metrics.as_ref().unwrap();
    assert!(metrics.usage_complete);
    assert_eq!(metrics.usage.as_ref().unwrap().reasoning_tokens, Some(4));
    assert!(metrics.duration_ms.unwrap() >= 0.0);
}

#[test]
fn denied_response_never_persists_gemini_parts_in_either_settlement_order() {
    for before in [true, false] {
        let (collector, receiver) = collector(config());
        assert!(collector.begin(request("denied")));
        let observation = crate::settlement::Observation::default();
        observation.record(&crate::events::Event::Usage(crate::events::Usage {
            input_tokens: Some(10),
            output_tokens: Some(3),
            ..crate::events::Usage::default()
        }));
        observation.record(&crate::events::Event::Completed);
        collector.observe_attempt("denied", observation);
        collector.gemini_thought_part("denied", Arc::new(json!({"thoughtSignature":"private"})));
        if before {
            collector.settle("denied", true, false);
        }
        collector.finish("denied", Some(response()), None);
        if !before {
            collector.settle("denied", true, false);
        }
        let records = drain(&collector, receiver);
        assert_eq!(records.len(), 1);
        assert!(records[0].response.is_none());
        assert!(records[0].gemini_thought_parts.is_empty());
        assert!(records[0].metrics.is_none());
    }
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
fn hosted_checkpoint_is_queued_before_terminal_and_shares_effective_input() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("checkpoint")));
    collector.reasoning("checkpoint", "not yet eligible output");
    assert!(collector.checkpoint("checkpoint"));
    let prompt = receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("queued checkpoint did not reach the destination");
    assert_eq!(prompt.request.request_id, "checkpoint");
    assert!(prompt.response.is_none());
    assert!(prompt.provider_reasoning.is_none());
    assert!(prompt.metrics.is_none());
    {
        let pending = collector.pending.lock().unwrap();
        // The test sink encodes and decodes, but the collector still owns its input.
        assert_eq!(
            prompt.request.context,
            pending.entries["checkpoint"].record.request.context
        );
    }
    collector.settle("checkpoint", true, true);
    assert!(collector.finish("checkpoint", Some(response()), None));
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert!(records[0].response.is_some());
    assert_eq!(records[0].request.context, prompt.request.context);
    assert_eq!(records[0].captured_at, prompt.captured_at);
}

#[test]
fn unregistered_and_local_requests_do_not_pretend_to_checkpoint() {
    let (hosted, receiver) = collector(config());
    assert!(hosted.checkpoint("capture-off"));
    assert!(drain(&hosted, receiver).is_empty());
    let mut configuration = config();
    configuration.settlement_required = false;
    let (local, receiver) = collector(configuration);
    assert!(local.begin(request("local")));
    assert!(local.checkpoint("local"));
    assert!(receiver.try_recv().is_err());
    assert!(local.finish("local", Some(response()), None));
    assert_eq!(drain(&local, receiver).len(), 1);
}

#[test]
fn collector_forwards_destination_cleanup_failure_without_losing_write_success() {
    struct CleanupFailure;
    impl Sink for CleanupFailure {
        type Prepared = ();
        fn preparation_bytes(_: usize) -> usize {
            0
        }
        fn prepare(&self, _: &Record, _: usize) -> Result<(), ()> {
            Ok(())
        }
        fn write(&mut self, _: &()) -> Result<(), ()> {
            Ok(())
        }
        fn take_maintenance_failures(&mut self) -> u64 {
            1
        }
    }
    let collector = Collector::new(config(), CleanupFailure).unwrap();
    assert!(collector.begin(request("saved")));
    collector.settle("saved", true, false);
    assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(collector.counts(), [0, 0, 1, 0, 0, 0]);
    assert_eq!(collector.maintenance_failures(), 1);
}

#[test]
fn selected_model_uses_cached_request_size_including_json_escapes() {
    let mut configuration = config();
    configuration.maximum_request_bytes = 1024;
    let (collector, receiver) = collector(configuration);
    let mut input = request("request");
    input.model_id = None;
    assert!(collector.begin(input));
    collector.select_model("request", "snow-雪-\"quoted\"");
    {
        let pending = collector.pending.lock().unwrap();
        let entry = &pending.entries["request"];
        assert_eq!(
            entry.request_bytes,
            serde_json::to_string(&entry.record.request).unwrap().len()
        );
        assert_eq!(
            entry.bytes,
            entry.record.heap_bytes() + "request".len() + 512
        );
    }
    collector.settle("request", true, false);
    let mut input = request("overflow");
    input.model_id = None;
    assert!(collector.begin(input));
    collector.select_model("overflow", &"\"".repeat(512));
    collector.settle("overflow", true, false);
    assert_eq!(drain(&collector, receiver).len(), 1);
    assert_eq!(collector.counts()[5], 1);
}

#[test]
fn structured_response_size_is_enforced_at_collector_boundary() {
    let mut configuration = config();
    let exact = response().json_bytes();
    configuration.maximum_response_bytes = exact;
    let (collector, receiver) = collector(configuration);
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, true);
    assert_eq!(drain(&collector, receiver).len(), 1);

    let mut configuration = config();
    configuration.maximum_response_bytes = exact - 1;
    let (collector, receiver) = self::collector(configuration);
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, true);
    assert!(drain(&collector, receiver).is_empty());
    assert_eq!(collector.counts()[5], 1);
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
    assert!(collector.attach("request").is_some());
    collector.finish("request", Some(response()), Some("deployment".into()));
    assert!(receiver.try_recv().is_err());
    collector.settle("request", false, false);
    assert!(drain(&collector, receiver).is_empty());
}

#[test]
fn settlement_before_response_emits_one_complete_record_without_a_stale_prompt_update() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector.settle("request", true, true);
    assert!(receiver.try_recv().is_err());
    assert!(collector.attach("request").is_some());
    assert!(collector.attach("request").is_none());
    collector.finish("request", Some(response()), Some("deployment".into()));
    collector.finish("request", Some(response()), None);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert!(records[0].response.is_some());
    assert_eq!(records[0].request.scope.identity_id, "identity");
    assert_eq!(records[0].deployment_id.as_deref(), Some("deployment"));
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
fn provider_reasoning_is_lossless_bounded_and_requires_response_eligibility() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("kept")));
    collector.reasoning("kept", "first\0");
    collector.reasoning("kept", "second雪");
    collector.finish("kept", Some(response()), None);
    collector.settle("kept", true, true);
    assert!(collector.begin(request("discarded")));
    collector.reasoning("discarded", "must not persist");
    collector.settle("discarded", true, false);
    assert!(collector.begin(request("overflow")));
    collector.reasoning("overflow", &"x".repeat(4097));
    collector.finish("overflow", Some(response()), None);
    collector.settle("overflow", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 2);
    let restored: String =
        serde_json::from_str(records[0].provider_reasoning_source_json.as_ref().unwrap()).unwrap();
    assert_eq!(restored, "first\0second雪");
    assert!(records[1].provider_reasoning.is_none());
}

#[test]
fn raw_tool_arguments_survive_projection_but_never_response_denial() {
    let (collector, receiver) = collector(config());
    let mut call = crate::events::CompletedToolCall {
        call_id: "call-1".into(),
        name: "lookup".into(),
        namespace: None,
        caller: None,
        provider_item_id: None,
        provider_status: None,
        raw_arguments: "{  \"x\" : \"雪\"  }".into(),
        custom: false,
    };
    assert!(collector.begin(request("kept")));
    collector.tool_call("kept", &call);
    call.call_id = "call-2".into();
    call.raw_arguments = "freeform\0text".into();
    call.custom = true;
    collector.tool_call("kept", &call);
    collector.finish("kept", Some(response()), None);
    collector.settle("kept", true, true);
    assert!(collector.begin(request("denied")));
    collector.tool_call("denied", &call);
    collector.settle("denied", true, false);
    assert!(collector.begin(request("overflow")));
    call.raw_arguments = "x".repeat(4096);
    collector.tool_call("overflow", &call);
    collector.settle("overflow", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 2);
    let calls: serde_json::Value =
        serde_json::from_str(records[0].provider_tool_calls_json.as_ref().unwrap()).unwrap();
    assert_eq!(calls[0]["raw_arguments"], "{  \"x\" : \"雪\"  }");
    assert_eq!(calls[1]["raw_arguments"], "freeform\0text");
    assert!(records[1].provider_tool_calls_json.is_none());
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
    wait_for_handoffs(&collector);
    assert!(collector.begin(request("next")));
    collector.settle("next", false, false);
    assert_eq!(drain(&collector, receiver).len(), 1);
}

#[test]
fn hosted_oversized_prompt_keeps_a_marked_copy_instead_of_rejecting_inference() {
    let mut configuration = config();
    configuration.truncate_request = true;
    configuration.maximum_request_bytes = 8 * 1024 * 1024;
    configuration.maximum_pending_bytes = 16 * 1024 * 1024;
    configuration.delivery.maximum_record_bytes = 8 * 1024 * 1024;
    configuration.delivery.maximum_bytes = 32 * 1024 * 1024;
    let (collector, receiver) = collector(configuration);
    let mut input = request("large");
    input.context = Arc::new(json!({"schema_version":1,"request": {
        "messages":[{"role":"system","content":"system prompt"},
            {"role":"tool","tool_call_id":"call-1","content":"雪".repeat(22 * 1024 * 1024)}],
        "tools":[{"name":"search","parameters":{"type":"object"}}]
    }}));
    assert!(collector.begin(input));
    collector.settle("large", true, false);
    let records = drain(&collector, receiver);
    let context = &records[0].request.context;
    assert_eq!(
        context["request"]["messages"][0]["content"],
        "system prompt"
    );
    assert_eq!(context["request"]["messages"][1]["tool_call_id"], "call-1");
    let content = context["request"]["messages"][1]["content"]
        .as_str()
        .unwrap();
    assert!(content.starts_with("雪") && content.contains("[truncated for capture:"));
    assert!(content.len() > 4_109_744 && content.len() < 4_110_000);
    assert_eq!(context["capture_limits"]["messages_truncated_strings"], 1);
    assert_eq!(collector.counts()[4..], [0, 0]);
}

#[test]
fn hosted_structural_envelope_overflow_does_not_discard_the_prompt() {
    let mut configuration = config();
    configuration.truncate_request = true;
    let (collector, receiver) = collector(configuration);
    let mut input = request("structural");
    input.context = Arc::new(json!({"schema_version":1,"request": {
        "messages":[{"role":"user","content":"keep this task"}],
        "tools":[{"name":"lookup","parameters":{"enum":(0..10000).collect::<Vec<_>>()}}]
    }}));
    assert!(collector.begin(input));
    collector.settle("structural", true, false);
    let records = drain(&collector, receiver);
    assert_eq!(
        records[0].request.context["request"]["messages"][0]["content"],
        "keep this task"
    );
    assert_eq!(
        records[0].request.context["capture_limits"]["request_envelope_dropped"],
        true
    );
    assert_eq!(collector.counts()[4..], [0, 0]);
}

#[test]
fn hosted_structural_messages_honor_the_configured_request_limit() {
    for maximum in [1024, 4096, 8192, 65536] {
        let mut configuration = config();
        configuration.truncate_request = true;
        configuration.maximum_request_bytes = maximum;
        let (collector, receiver) = collector(configuration);
        let mut input = request("many-messages");
        // No individual string exceeds the 4096-byte truncation threshold.
        // Also cover messages below the cap but above it with the envelope.
        let content = "x".repeat(100);
        let messages = vec![json!({"role":"user","content":content}); maximum / 128];
        input.context = Arc::new(json!({"schema_version":1,"request": {
            "messages":messages,"tools":[{"name":"search"}]
        }}));
        assert!(input.json_bytes() > maximum);
        assert!(collector.begin(input));
        collector.settle("many-messages", true, false);
        let records = drain(&collector, receiver);
        assert_eq!(records.len(), 1);
        assert!(records[0].request.json_bytes() <= maximum);
        let limits = &records[0].request.context["capture_limits"];
        assert_eq!(limits["messages_truncated"], true);
        assert_eq!(limits["messages_dropped"], maximum / 128);
        assert_eq!(collector.counts()[4..], [0, 0]);
    }
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

#[test]
fn shutdown_timeout_preserves_pending_history_and_accepts_late_settlement() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("waiting")));
    assert!(collector.finish("waiting", Some(response()), None));
    let bytes = collector.pending.lock().unwrap().bytes;
    assert!(!collector.close_until(Instant::now()));
    {
        let pending = collector.pending.lock().unwrap();
        assert!(pending.entries.contains_key("waiting"));
        assert_eq!(pending.bytes, bytes);
    }
    assert_eq!(collector.counts()[5], 0);
    assert!(!collector.begin(request("late")));
    collector.settle("waiting", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].request.request_id, "waiting");
    assert!(records[0].response.is_some());
    assert_eq!(collector.counts(), [0, 0, 1, 0, 0, 1]);
}

#[test]
fn shutdown_waits_for_an_accepted_response_instead_of_closing_its_destination() {
    let mut configuration = config();
    configuration.settlement_required = false;
    let (collector, receiver) = collector(configuration);
    assert!(collector.begin(request("active")));
    let closing = collector.clone();
    let thread =
        std::thread::spawn(move || closing.close_until(Instant::now() + Duration::from_secs(2)));
    let until = Instant::now() + Duration::from_secs(1);
    while !collector.pending.lock().unwrap().closed && Instant::now() < until {
        std::thread::yield_now();
    }
    assert!(collector.pending.lock().unwrap().closed);
    assert!(collector.finish("active", Some(response()), None));
    assert!(thread.join().unwrap());
    assert_eq!(receiver.try_iter().count(), 1);
    assert_eq!(collector.counts(), [0, 0, 1, 0, 0, 0]);
}

#[test]
fn shutdown_keeps_delivery_open_during_the_pending_map_handoff() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("handoff")));
    // Pause exactly where finish/settle releases the map lock before emit().
    let entry = {
        let mut pending = collector.pending.lock().unwrap();
        let entry = pending.entries.remove("handoff").unwrap();
        pending.bytes -= entry.bytes;
        entry
    };
    assert!(!collector.close_until(Instant::now()));
    assert!(collector.emit(entry.record, None, Some(entry._admission)));
    assert_eq!(drain(&collector, receiver).len(), 1);
    assert_eq!(collector.counts(), [0, 0, 1, 0, 0, 0]);
}

struct PausedSink {
    entered: mpsc::Sender<()>,
    released: Arc<AtomicBool>,
    records: MemorySink,
}

impl Sink for PausedSink {
    type Prepared = String;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        MemorySink::preparation_bytes(maximum_record_bytes)
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<String, ()> {
        self.records.prepare(record, maximum_bytes)
    }

    fn write(&mut self, record: &String) -> Result<(), ()> {
        if !self.released.load(Ordering::Acquire) {
            let _ = self.entered.send(());
            return Err(());
        }
        self.records.write(record)
    }
}

#[test]
fn queued_prompt_does_not_wait_for_storage_or_release_the_terminal_response_owner() {
    let (entered, started) = mpsc::channel();
    let (records, receiver) = mpsc::channel();
    let released = Arc::new(AtomicBool::new(false));
    let collector = Arc::new(
        Collector::new(
            config(),
            PausedSink {
                entered,
                released: released.clone(),
                records: MemorySink(records),
            },
        )
        .unwrap(),
    );
    assert!(collector.begin(request("slow-checkpoint")));
    assert!(collector.checkpoint("slow-checkpoint"));
    let waiting = started.recv_timeout(Duration::from_secs(2)).is_ok();
    let retained = {
        let pending = collector.pending.lock().unwrap();
        pending.entries.contains_key("slow-checkpoint")
    };
    released.store(true, Ordering::Release);
    assert!(waiting && retained);
    collector.settle("slow-checkpoint", true, true);
    assert!(collector.finish("slow-checkpoint", Some(response()), None));
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 2);
    assert!(records[0].response.is_none());
    assert!(records[1].response.is_some());
    assert_eq!(collector.counts()[4..], [0, 0]);
}

#[test]
fn durable_acknowledgement_is_preserved_unless_host_explicitly_opts_out() {
    for asynchronous in [false, true] {
        for mode in ["local", "settlement", "checkpoint"] {
            let local = mode == "local";
            let mut configuration = config();
            configuration.asynchronous_delivery = asynchronous;
            configuration.settlement_required = !local;
            let (entered, started) = mpsc::channel();
            let (records, receiver) = mpsc::channel();
            let (finished, returned) = mpsc::channel();
            let released = Arc::new(AtomicBool::new(false));
            let collector = Arc::new(
                Collector::new(
                    configuration,
                    PausedSink {
                        entered,
                        released: released.clone(),
                        records: MemorySink(records),
                    },
                )
                .unwrap(),
            );
            assert!(collector.begin(request("acknowledgement")));
            let writer = collector.clone();
            let producer = std::thread::spawn(move || {
                if local {
                    assert!(writer.finish("acknowledgement", Some(response()), None));
                } else if mode == "checkpoint" {
                    assert!(writer.checkpoint("acknowledgement"));
                } else {
                    writer.settle("acknowledgement", true, false);
                }
                finished.send(()).unwrap();
            });
            let reached = started.recv_timeout(Duration::from_secs(2)).is_ok();
            let early = returned.recv_timeout(Duration::from_millis(50)).is_ok();
            let pending = collector.counts()[0];
            released.store(true, Ordering::Release);
            producer.join().unwrap();
            if mode == "checkpoint" {
                collector.settle("acknowledgement", false, false);
            }
            let records = drain(&collector, receiver);
            assert!(reached);
            assert_eq!(early, asynchronous);
            assert_eq!(pending, 1);
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].response.is_some(), local);
            assert_eq!(collector.counts()[4..], [0, 0]);
        }
    }
}

#[test]
fn stalled_handoffs_remain_inside_admission_count_and_byte_limits() {
    for bytes_bound in [false, true] {
        for mode in ["prompt", "response_first", "settle_first", "local"] {
            let mut configuration = config();
            configuration.maximum_pending_records = if bytes_bound { 32 } else { 1 };
            configuration.maximum_pending_bytes = 8192;
            configuration.maximum_request_bytes = 6144;
            configuration.maximum_response_bytes = 1024;
            configuration.settlement_required = mode != "local";
            let (entered, started) = mpsc::channel();
            let (records, receiver) = mpsc::channel();
            let released = Arc::new(AtomicBool::new(false));
            let collector = Arc::new(
                Collector::new(
                    configuration,
                    PausedSink {
                        entered,
                        released: released.clone(),
                        records: MemorySink(records),
                    },
                )
                .unwrap(),
            );
            let mut input = request("first");
            input.context = Arc::new(json!({"schema_version":1,"request": {
                "messages":[{"role":"user","content":"x".repeat(2000)}]
            }}));
            assert!(collector.begin(input.clone()));
            if mode == "response_first" {
                assert!(collector.finish("first", Some(response()), None));
            } else if mode == "settle_first" {
                collector.settle("first", true, true);
            }
            let writer = collector.clone();
            let thread = std::thread::spawn(move || match mode {
                "prompt" => writer.settle("first", true, false),
                "response_first" => writer.settle("first", true, true),
                _ => {
                    assert!(writer.finish("first", Some(response()), None));
                }
            });
            let reached_destination = started.recv_timeout(Duration::from_secs(2)).is_ok();
            input.request_id = "overflow".into();
            let admitted_while_stalled = collector.begin(input.clone());
            if admitted_while_stalled {
                collector.settle("overflow", false, false);
            }
            // Always recover and join before asserting the regression: failures
            // must not strand a native retry worker or leak a test thread.
            released.store(true, Ordering::Release);
            thread.join().unwrap();
            wait_for_handoffs(&collector);
            assert!(reached_destination, "mode={mode} bytes={bytes_bound}");
            assert!(!admitted_while_stalled, "mode={mode} bytes={bytes_bound}");
            assert_eq!(collector.handoff_bytes.load(Ordering::Acquire), 0);
            assert_eq!(collector.admissions.load(Ordering::Acquire), 0);
            assert!(collector.begin(input));
            collector.settle("overflow", false, false);
            assert_eq!(drain(&collector, receiver).len(), 1);
            assert_eq!(collector.counts()[4], 0);
        }
    }
}

#[path = "checkpoint_test.rs"]
mod checkpoints;
