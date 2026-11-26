//! OpenRouter's own per-request charge (`usage.cost`) reaches the settle
//! callback as `usage.provider_reported_cost_nano_usd` when a non-streaming
//! caller's answer drains an OpenRouter rung, and no other rung reports one.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::ladder_tests::{block_on, finish, spawn_rung, wire, Answer, Harness, TEXT_FRAME};
use super::*;

/// OpenRouter's finish chunk.
const STOP_FRAME: &str = "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}";

/// OpenRouter's trailing usage chunk, shape captured from openrouter.ai: a
/// $0.0016 non-BYOK generation.
const OPENROUTER_USAGE_FRAME: &str = concat!(
    "{\"choices\":[],\"provider\":\"Fireworks\",\"usage\":{",
    "\"prompt_tokens\":14,\"completion_tokens\":2,\"total_tokens\":16,",
    "\"cost\":0.0016,\"is_byok\":false,",
    "\"prompt_tokens_details\":{\"cached_tokens\":0},",
    "\"cost_details\":{\"upstream_inference_cost\":null},",
    "\"completion_tokens_details\":{\"reasoning_tokens\":0}}}"
);

/// A rung of `provider` on the OpenAI-compatible (Chat Completions) wire.
fn provider_wire(url: &str, provider: &str) -> DeploymentWire {
    DeploymentWire {
        provider: provider.to_string(),
        ..wire("a", url, 0)
    }
}

/// Drain one buffered (non-streaming) answer from a rung of `provider` and
/// return the usage its one attempt settled.
async fn settled_usage(provider: &str) -> Value {
    let harness = Harness::new();
    let rung = spawn_rung(vec![Answer::Stream(&[
        TEXT_FRAME,
        STOP_FRAME,
        OPENROUTER_USAGE_FRAME,
    ])])
    .await;
    let route = [provider_wire(&rung.url, provider)];
    let mut guard = AttemptGuard::new(
        harness.bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "request-cost".to_string(),
        Instant::now(),
    );
    let context = WaterfallContext {
        pre_first_byte_deadline: None,
        bridge: &harness.bridge,
        http: &harness.http,
        request_id: "request-cost",
        raw_key: "key",
        caller_scope: None,
        route: &route,
        policy: RoutePolicy {
            maximum_total_attempts: 8,
            maximum_same_deployment_attempts: 2,
            refusal_failover: false,
            throttle_redial: None,
            physical_route_cap: None,
            backoff: None,
        },
        deadline: Instant::now() + Duration::from_secs(30),
        time_to_first_byte: Duration::from_secs(5),
        time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
        time_to_first_token: Duration::from_secs(120),
        approximate_input_tokens: 10.0,
        chat_logprobs: false,
        capture_logprobs: false,
        output_less_retention: None,
        output_token_cap: None,
        tool_search: None,
        output_guardrails: None,
        buffered: &[true],
    };
    let won = acquire_attempt(&context, &mut guard).await;
    assert!(matches!(won, Won::Committed(_)), "the rung serves");
    finish(guard, won).await;
    let story = harness.story().await;
    let settles = story["settles"].as_array().expect("settles");
    assert_eq!(settles.len(), 1, "{settles:?}");
    assert_eq!(settles[0]["outcome"], "completed");
    settles[0]["usage"].clone()
}

#[test]
fn a_non_streaming_openrouter_answer_settles_its_reported_cost() {
    block_on(async {
        let usage = settled_usage("openrouter").await;
        assert_eq!(usage["input_tokens"], 14);
        assert_eq!(usage["provider_reported_cost_nano_usd"], 1_600_000);
    });
}

#[test]
fn a_non_openrouter_rung_settles_no_reported_cost() {
    block_on(async {
        let usage = settled_usage("openai").await;
        assert_eq!(usage["input_tokens"], 14);
        assert!(usage.get("provider_reported_cost_nano_usd").is_none());
    });
}
