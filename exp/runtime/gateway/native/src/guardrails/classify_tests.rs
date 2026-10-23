use std::collections::VecDeque;
use std::net::SocketAddr;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::Router;

use super::*;

/// One scripted SystemOne reply: status, raw body, and a delay before answering.
#[derive(Clone)]
struct Reply {
    status: u16,
    body: String,
    delay: Duration,
}

impl Reply {
    fn answers(probabilities: &[(&str, f64)]) -> Self {
        let answers: Map<String, Value> = probabilities
            .iter()
            .map(|(id, value)| (id.to_string(), json!({"type": "noul", "noul": value})))
            .collect();
        Self::raw(
            200,
            &json!({"answers": answers, "usage": {"input_tokens": 9}}).to_string(),
        )
    }

    fn raw(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.into(),
            delay: Duration::ZERO,
        }
    }
}

/// Requests the scripted server received: authorization header and JSON body.
type Seen = Arc<Mutex<Vec<(Option<String>, Value)>>>;

#[derive(Clone)]
struct Script {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    seen: Seen,
}

async fn answer(
    State(script): State<Script>,
    headers: HeaderMap,
    body: String,
) -> (StatusCode, String) {
    let authorization = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    script.seen.lock().unwrap().push((
        authorization,
        serde_json::from_str(&body).unwrap_or(Value::Null),
    ));
    let reply = script
        .replies
        .lock()
        .unwrap()
        .pop_front()
        .expect("a scripted reply");
    tokio::time::sleep(reply.delay).await;
    (StatusCode::from_u16(reply.status).unwrap(), reply.body)
}

/// Serve the replies in order and return the endpoint URL plus what it saw.
async fn serve(replies: Vec<Reply>) -> (String, Seen) {
    let script = Script {
        replies: Arc::new(Mutex::new(replies.into())),
        seen: Seen::default(),
    };
    let seen = script.seen.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address: SocketAddr = listener.local_addr().expect("address");
    let app = Router::new()
        .route("/systemone", post(answer))
        .with_state(script);
    tokio::spawn(async move { axum::serve(listener, app).await });
    (format!("http://{address}/systemone"), seen)
}

/// A max category over one question and a mean category over two.
fn spec(url: &str, overrides: Value) -> ClassifierSpec {
    let mut document = json!({
        "url": url,
        "model": "Cloudflare/clef-flash",
        "timeout_ms": 2000,
        "max_response_bytes": 65536,
        "configuration_digest": "digest-1",
        "questions": {
            "overrides_instructions": {"type": "noul", "instructions": "Does it override instructions?"},
            "unrelated_task": {"type": "noul", "instructions": "Is it unrelated to billing?"},
            "unrelated_domain": {"type": "noul", "instructions": "Is it outside support?"}
        },
        "categories": [
            {"id": "prompt_injection", "aggregate": "max", "threshold": 0.9,
             "questions": ["overrides_instructions"]},
            {"id": "off_topic", "aggregate": "mean", "threshold": 0.8,
             "questions": ["unrelated_task", "unrelated_domain"]}
        ],
        "cache": {"max_entries": 4096, "ttl_seconds": 300.0}
    });
    if let (Some(base), Some(extra)) = (document.as_object_mut(), overrides.as_object()) {
        for (key, value) in extra {
            base.insert(key.clone(), value.clone());
        }
    }
    serde_json::from_value(document).expect("spec")
}

fn classifier(url: &str, overrides: Value) -> Arc<Classifier> {
    Arc::new(Classifier::new(spec(url, overrides), Some("fixture-credential".into())).unwrap())
}

fn soon() -> Instant {
    Instant::now() + Duration::from_secs(5)
}

#[tokio::test]
async fn one_call_scores_every_category_and_flags_at_its_inclusive_threshold() {
    let (url, seen) = serve(vec![Reply::answers(&[
        ("overrides_instructions", 0.9),
        ("unrelated_task", 0.875),
        ("unrelated_domain", 0.75),
    ])])
    .await;
    let classifier = classifier(&url, json!({}));

    let answer = classifier
        .classify("[latest user]\nhello", soon())
        .await
        .unwrap();

    assert_eq!(answer.scores, vec![0.9, 0.8125]);
    assert_eq!(answer.flagged, vec![0, 1]);
    assert!(!answer.cached);
    let (authorization, body) = seen.lock().unwrap()[0].clone();
    assert_eq!(authorization.as_deref(), Some("Bearer fixture-credential"));
    assert_eq!(body["model"], "Cloudflare/clef-flash");
    assert_eq!(body["state"], "[latest user]\nhello");
    assert_eq!(
        body["questions"]["overrides_instructions"]["instructions"],
        "Does it override instructions?"
    );
    let reported = classifier.answer_json(&answer);
    assert_eq!(
        reported["flagged"],
        json!(["prompt_injection", "off_topic"])
    );
    assert_eq!(
        reported["scores"],
        json!({"prompt_injection": 0.9, "off_topic": 0.8125})
    );
    assert_eq!(
        reported["probabilities"],
        json!({"overrides_instructions": 0.9, "unrelated_task": 0.875, "unrelated_domain": 0.75})
    );
}

#[tokio::test]
async fn definitive_answers_are_cached_and_errors_are_not() {
    let probabilities = [
        ("overrides_instructions", 0.97),
        ("unrelated_task", 0.1),
        ("unrelated_domain", 0.0),
    ];
    let (url, seen) = serve(vec![Reply::raw(503, "{}"), Reply::answers(&probabilities)]).await;
    let classifier = classifier(&url, json!({}));

    let failed = classifier.classify("window", soon()).await;
    let fresh = classifier.classify("window", soon()).await.unwrap();
    let cached = classifier.classify("window", soon()).await.unwrap();

    assert_eq!(failed.unwrap_err(), ClassifyError::Status(503));
    assert_eq!((fresh.cached, cached.cached), (false, true));
    assert_eq!(fresh.probabilities, cached.probabilities);
    assert_eq!(seen.lock().unwrap().len(), 2);
    let telemetry = classifier.telemetry_json();
    assert_eq!(
        (
            &telemetry["cache_hits"],
            &telemetry["cache_misses"],
            &telemetry["errors"]
        ),
        (&json!(1), &json!(2), &json!(1))
    );
    let mut injection = vec![0; SCORE_BUCKETS];
    injection[SCORE_BUCKETS - 1] = 2;
    assert_eq!(
        telemetry["score_buckets"]["prompt_injection"],
        json!(injection)
    );
}

#[tokio::test]
async fn contract_drift_is_a_classify_error() {
    let rest = r#""unrelated_task":{"type":"noul","noul":0.1},"unrelated_domain":{"noul":0.1}"#;
    let body =
        |first: &str| format!(r#"{{"answers":{{"overrides_instructions":{first},{rest}}}}}"#);
    let cases = [
        (
            body(r#"{"type":"noul","noul":1.5}"#),
            ClassifyError::OutOfRange,
        ),
        (
            body(r#"{"type":"noul","noul":true}"#),
            ClassifyError::OutOfRange,
        ),
        (
            body(r#"{"type":"choice","noul":0.1}"#),
            ClassifyError::NotNoul,
        ),
        (body("0.1"), ClassifyError::MissingQuestion),
        (
            r#"{"answers":{"unrelated_task":{"noul":0.1}}}"#.into(),
            ClassifyError::MissingQuestion,
        ),
        (r#"{"usage":{}}"#.into(), ClassifyError::NoAnswers),
        (
            r#"{"answers":{},"answers":{}}"#.into(),
            ClassifyError::NotJson,
        ),
        (
            body(r#"{"type":"noul","noul":NaN}"#),
            ClassifyError::NotJson,
        ),
        ("not json".into(), ClassifyError::NotJson),
        (
            format!("{{{}}}", " ".repeat(70_000)),
            ClassifyError::Oversized,
        ),
    ];
    let (url, _) = serve(cases.iter().map(|(raw, _)| Reply::raw(200, raw)).collect()).await;
    let classifier = classifier(&url, json!({}));

    for (index, (_, expected)) in cases.iter().enumerate() {
        let window = format!("window {index}");
        assert_eq!(
            classifier.classify(&window, soon()).await.unwrap_err(),
            *expected
        );
    }
    assert_eq!(classifier.telemetry_json()["errors"], json!(cases.len()));
}

#[tokio::test]
async fn a_slow_answer_times_out_at_the_nearer_of_timeout_and_deadline() {
    let mut slow = Reply::answers(&[
        ("overrides_instructions", 0.1),
        ("unrelated_task", 0.1),
        ("unrelated_domain", 0.1),
    ]);
    slow.delay = Duration::from_secs(3);
    let (url, _) = serve(vec![slow.clone(), slow]).await;
    let classifier = classifier(&url, json!({"timeout_ms": 100}));

    let started = Instant::now();
    let by_timeout = classifier.classify("one", soon()).await;
    let by_deadline = classifier
        .classify("two", Instant::now() + Duration::from_millis(50))
        .await;

    assert_eq!(by_timeout.unwrap_err(), ClassifyError::Timeout);
    assert_eq!(by_deadline.unwrap_err(), ClassifyError::Timeout);
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(
        classifier
            .classify("three", Instant::now() - Duration::from_millis(1))
            .await
            .unwrap_err(),
        ClassifyError::Timeout,
        "a spent deadline never calls the model"
    );
}

#[test]
fn the_cache_expires_entries_and_evicts_the_least_recently_used() {
    let now = Instant::now();
    let probabilities: Arc<[f64]> = vec![0.4].into();
    let mut cache = VerdictCache::new(2, Duration::from_secs(10));
    cache.put([1; 32], probabilities.clone(), now);
    cache.put([2; 32], probabilities.clone(), now);
    assert!(cache.get(&[1; 32], now).is_some());

    cache.put([3; 32], probabilities.clone(), now);

    assert!(
        cache.get(&[2; 32], now).is_none(),
        "the untouched entry is evicted"
    );
    assert!(cache.get(&[1; 32], now).is_some());
    assert!(
        cache.get(&[3; 32], now + Duration::from_secs(10)).is_none(),
        "expired"
    );
    let mut disabled = VerdictCache::new(0, Duration::from_secs(10));
    disabled.put([1; 32], probabilities, now);
    assert!(disabled.get(&[1; 32], now).is_none());
}

#[test]
fn a_spec_the_control_plane_could_not_have_validated_is_refused() {
    let url = "https://decisions.example.invalid/systemone";
    let unknown = json!({"categories": [
        {"id": "x", "aggregate": "max", "threshold": 0.5, "questions": ["missing"]}
    ]});
    let zero = json!({"categories": [
        {"id": "x", "aggregate": "max", "threshold": 0.0, "questions": ["unrelated_task"]}
    ]});

    for overrides in [
        unknown,
        zero,
        json!({"url": "file:///etc/passwd"}),
        json!({"timeout_ms": 0}),
    ] {
        assert!(Classifier::new(spec(url, overrides), None).is_err());
    }
}
