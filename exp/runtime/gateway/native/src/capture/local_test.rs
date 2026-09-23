//! Real SQLite retries preserve one prepared payload and all conversation context.
use super::*;
use crate::capture::delivery::{Delivery, Limits};
use rusqlite::Connection;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct ObservedSink {
    sink: SqliteSink,
    prepared: Arc<AtomicUsize>,
    attempts: Arc<AtomicUsize>,
    lost_ack: bool,
    payload_address: Option<usize>,
}

impl Sink for ObservedSink {
    type Prepared = Option<local_store::Pending>;
    fn preparation_bytes(maximum: usize) -> usize {
        SqliteSink::preparation_bytes(maximum)
    }
    fn prepare(&self, record: &Record, maximum: usize) -> Result<Self::Prepared, ()> {
        self.prepared.fetch_add(1, Ordering::Relaxed);
        self.sink.prepare(record, maximum)
    }
    fn write(&mut self, prepared: &Self::Prepared) -> Result<(), ()> {
        self.attempts.fetch_add(1, Ordering::Relaxed);
        let address = prepared.as_ref().unwrap().payload.as_ptr() as usize;
        assert_eq!(*self.payload_address.get_or_insert(address), address);
        self.sink.write(prepared)?;
        if std::mem::take(&mut self.lost_ack) {
            return Err(());
        }
        Ok(())
    }
    fn maintain(&mut self) -> Result<(), ()> {
        self.sink.maintain()
    }
    fn take_maintenance_failures(&mut self) -> u64 {
        self.sink.take_maintenance_failures()
    }
}

#[test]
fn locked_sqlite_and_lost_ack_retry_one_payload_without_losing_conversation() {
    let path = std::env::temp_dir().join(format!(
        "capture-retry-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let policy = Policy {
        scope: Scope {
            user_id: "user".into(),
            application_id: "app".into(),
        },
        enabled: true,
        maximum_experiences: 10,
        maximum_storage_bytes: 1_000_000,
        maximum_experience_bytes: 16_384,
        retention_seconds: 3600,
    };
    let sink = SqliteSink::open(CaptureConfiguration {
        database_path: path.to_string_lossy().into(),
        bindings: vec![Binding {
            alias: "model".into(),
            policy,
        }],
        queue_capacity: 1,
    })
    .unwrap()
    .unwrap();
    let prepared = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(AtomicUsize::new(0));
    let limits = Limits {
        maximum_records: 1,
        maximum_bytes: 1_000_000,
        maximum_record_bytes: 16_384,
    };
    let delivery = Delivery::new(
        limits.clone(),
        ObservedSink {
            sink,
            prepared: prepared.clone(),
            attempts: attempts.clone(),
            lost_ack: true,
            payload_address: None,
        },
    )
    .unwrap();
    let reader = Connection::open(&path).unwrap();
    reader.execute_batch("BEGIN IMMEDIATE").unwrap();
    let record: Record = serde_json::from_value(json!({
        "schema_version":super::super::record::SCHEMA_VERSION,"request":{"request_id":"request",
        "scope":{"organization_id":"org","identity_id":"user","application_id":"app"},
        "protocol":"responses","model_id":"model","context":{"schema_version":1,
        "session_id":"harness-episode", "request":{"messages":[
            {"role":"system","content":"system 雪"},
            {"role":"developer","content":"developer instructions"},
            {"role":"user","content":"first prompt"},
            {"role":"assistant","tool_calls":[{"id":"call","function":{"name":"lookup","arguments":"{  }"}}]},
            {"role":"tool","tool_call_id":"call","content":" environment response "},
            {"role":"user","content":"second prompt"}
        ],"tools":[{"name":"lookup","parameters":{"type":"object"}}]}}},
        "response":{"kind":"json","status":200,"body":{
            "id":"response","status":"completed","output":[{"text":"answer"}]},"source_json":null},
        "provider_reasoning":"thought 雪", "provider_reasoning_source_json":null,
        "provider_tool_calls_json":null,"metrics":null,"gemini_thought_parts":[],
        "gemini_thought_parts_source_json":null,"deployment_id":"deployment",
        "captured_at":local_store::now() as f64
    })).unwrap();
    let context = record.request.context.clone();
    let response = serde_json::to_value(&record.response).unwrap();
    assert!(delivery.submit(record));
    let deadline = Instant::now() + Duration::from_secs(5);
    while delivery.maintenance_failures() == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(delivery.counts()[3] > 0);
    assert!(delivery.maintenance_failures() > 0);
    assert!(!delivery.close_until(Instant::now()));
    assert_eq!(delivery.counts()[0], 1);
    assert!(delivery.counts()[1] <= limits.maximum_bytes as u64);
    assert_eq!(delivery.counts()[4], 0);
    reader.execute_batch("ROLLBACK").unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(5)));
    assert_eq!(prepared.load(Ordering::Relaxed), 1);
    assert!(attempts.load(Ordering::Relaxed) >= 3);
    let counts = delivery.counts();
    assert_eq!(counts[3] as usize, attempts.load(Ordering::Relaxed) - 1);
    assert_eq!([counts[0], counts[1], counts[2], counts[4]], [0, 0, 1, 0]);
    let (count, payload): (i64, String) = reader
        .query_row(
            "SELECT count(*), payload FROM gateway_captures",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(count, 1);
    let saved: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(saved["request"]["exp_context"], *context);
    assert_eq!(saved["request"]["exp_capture_output"]["response"], response);
    assert_eq!(
        saved["request"]["exp_capture_output"]["provider_reasoning"],
        "thought 雪"
    );
    assert_eq!(saved["episode_id"], "harness-episode");
    drop(reader);
    drop(delivery);
    std::fs::remove_file(path).unwrap();
}
