//! A provider's own content filter refusing a request before anything reached
//! the caller fails over once, to a rung of another provider; the model's own
//! refusal, a second filter refusal, a rung of the same provider, and a live
//! stream that already sent the caller a delta keep their answer.

use std::time::{Duration, Instant};

use serde_json::json;

use super::ladder_tests::{block_on, finish, spawn_rung, wire, Answer, Harness, TEXT_FRAME};
use super::*;
use crate::relay::collect_committed;

/// Azure AI Foundry's answer when its content safety layer blocks a
/// DeepSeek-V4.1-Flash output (live 2026-11-17): HTTP 400 carrying a
/// `chat.completion` whose first choice finished `content_filter`.
pub(super) const AZURE_FILTERED_BODY: &str = concat!(
    "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"model\":\"DeepSeek-V4.1-Flash\",",
    "\"choices\":[{\"index\":0,\"finish_reason\":\"content_filter\",",
    "\"message\":{\"role\":\"assistant\",\"content\":\"\"},",
    "\"content_filter_results\":{\"error\":{\"code\":\"content_filter\",",
    "\"message\":\"Response content blocked by label 'MultiSeverity_SexualScore'.\"}}}]}"
);

/// A stream's closing chunk when the provider's filter cut it.
pub(super) const FILTER_FINISH_FRAME: &str =
    "{\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}";

/// A rung of `provider`.
fn provider_wire(deployment_id: &str, url: &str, provider: &str) -> DeploymentWire {
    DeploymentWire {
        native_tool_translation: Default::default(),
        provider: provider.to_string(),
        ..wire(deployment_id, url, 0)
    }
}

#[test]
fn an_azure_content_filter_refusal_is_served_by_another_providers_rung() {
    block_on(async {
        let harness = Harness::new();
        let azure = spawn_rung(vec![Answer::Rejected(AZURE_FILTERED_BODY)]).await;
        let sibling = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            provider_wire("azure", &azure.url, "azure_openai"),
            provider_wire("novita", &sibling.url, "openai"),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the unfiltered sibling serves the refused request");
        };
        assert_eq!(committed.depth, 1);
        drop(committed);
        assert_eq!(azure.accepted.lock().expect("lock").len(), 1);
        assert_eq!(sibling.accepted.lock().expect("lock").len(), 1);
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 2);
        let failure = &starts[1]["failure"];
        assert_eq!(failure["failure_class"], "refusal");
        assert_eq!(failure["refusal_reason"], "content_policy");
        assert_eq!(failure["provider_content_filter"], true);
        assert_eq!(failure["failover_eligible"], true);
        assert_eq!(failure["retryable_same_deployment"], false);
        // The refused dial keeps its own attempt row; the ladder stays open.
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[0]["finalize"], false);
        assert_eq!(settles[0]["failure"]["failure_class"], "refusal");
        assert_eq!(settles[0]["failure"]["provider_content_filter"], true);
        assert_eq!(settles[1]["outcome"], "completed");
    });
}

#[test]
fn a_content_filter_refusal_never_moves_to_a_rung_of_the_same_provider() {
    block_on(async {
        let harness = Harness::new();
        let azure = spawn_rung(vec![Answer::Rejected(AZURE_FILTERED_BODY)]).await;
        let other_azure = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            provider_wire("azure-east", &azure.url, "azure_openai"),
            provider_wire("azure-west", &other_azure.url, "azure_openai"),
        ];
        let (won, _guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Failed(error) = won else {
            panic!("the same provider's filter would refuse again: the refusal is the answer");
        };
        assert_eq!(error.status_code, 400);
        assert_eq!(error.code, "refusal");
        assert!(other_azure.accepted.lock().expect("lock").is_empty());
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 1);
        assert_eq!(settles[0]["finalize"], true);
    });
}

#[test]
fn a_second_content_filter_refusal_is_the_requests_answer() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null, null, null]}))
            .await;
        let azure = spawn_rung(vec![Answer::Rejected(AZURE_FILTERED_BODY)]).await;
        let filtered = spawn_rung(vec![Answer::Rejected(AZURE_FILTERED_BODY)]).await;
        let third = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            provider_wire("azure", &azure.url, "azure_openai"),
            provider_wire("bedrock", &filtered.url, "bedrock"),
            provider_wire("novita", &third.url, "openai"),
        ];
        let (won, _guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Failed(error) = won else {
            panic!("the failover is spent after one content-filter refusal");
        };
        assert_eq!(error.code, "refusal");
        assert_eq!(filtered.accepted.lock().expect("lock").len(), 1);
        assert!(third.accepted.lock().expect("lock").is_empty());
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 2);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[1]["finalize"], true);
        assert_eq!(settles[1]["failure"]["refusal_reason"], "content_policy");
    });
}

#[test]
fn the_models_own_refusal_never_fails_over() {
    block_on(async {
        let harness = Harness::new();
        // A refusal delta is the model declining in its own words, even when
        // the provider closes the turn with a filter finish.
        let declined = spawn_rung(vec![Answer::Stream(&[
            "{\"choices\":[{\"delta\":{\"refusal\":\"I can't help with that.\"}}]}",
            FILTER_FINISH_FRAME,
        ])])
        .await;
        let sibling = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            provider_wire("azure", &declined.url, "azure_openai"),
            provider_wire("novita", &sibling.url, "openai"),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let won = finish(guard, won).await;
        if let Won::Committed(committed) = &won {
            assert_eq!(committed.depth, 0, "the model's refusal is the answer");
        }
        assert!(sibling.accepted.lock().expect("lock").is_empty());
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
    });
}

#[test]
fn a_live_stream_cut_by_the_filter_after_a_delta_keeps_its_terminal_error() {
    block_on(async {
        let harness = Harness::new();
        let azure = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME, FILTER_FINISH_FRAME])]).await;
        let sibling = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            provider_wire("azure", &azure.url, "azure_openai"),
            provider_wire("novita", &sibling.url, "openai"),
        ];
        let (won, mut guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Committed(mut committed) = won else {
            panic!("the first delta commits the live stream");
        };
        assert_eq!(committed.depth, 0);
        // The surface drains the live relay and meets the refusal itself; it
        // ends the stream with the protocol's error event, as before.
        let events = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(10),
            Duration::from_secs(10),
            Instant::now(),
        )
        .await
        .expect("the stream reaches its terminal");
        let Some(Event::Failed(failure)) = events.last() else {
            panic!("the filter's refusal is the stream's terminal");
        };
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        guard.settle("failed", None, &[], Some(failure), true).await;
        assert!(sibling.accepted.lock().expect("lock").is_empty());
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
    });
}

#[test]
fn a_rung_that_opted_into_content_policy_takes_a_streamed_filter_finish() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null, ["refusal:content_policy"]]}))
            .await;
        let azure = spawn_rung(vec![Answer::Stream(&[FILTER_FINISH_FRAME])]).await;
        let opted = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        // Same provider, so the one-shot gate does not apply: the rung's own
        // opt-in is what takes the refusal, and the filter marker waits
        // uncommitted so it can.
        let route = [
            provider_wire("azure-east", &azure.url, "azure_openai"),
            DeploymentWire {
                native_tool_translation: Default::default(),
                failover_only_on: Some(vec!["refusal:content_policy".to_string()]),
                ..provider_wire("azure-west", &opted.url, "azure_openai")
            },
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the opted-in rung serves the filtered request");
        };
        assert_eq!(committed.depth, 1);
        drop(committed);
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts[1]["failure"]["provider_content_filter"], true);
        // Not promoted: the rung's rule, not the gate, advanced it.
        assert_eq!(starts[1]["failure"]["failover_eligible"], false);
    });
}

#[test]
fn a_responses_stream_the_filter_cut_before_output_fails_over() {
    block_on(async {
        let harness = Harness::new();
        let azure = spawn_rung(vec![Answer::ResponsesFailed(concat!(
            "{\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",",
            "\"incomplete_details\":{\"reason\":\"content_filter\"},",
            "\"usage\":{\"input_tokens\":12,\"output_tokens\":0,\"total_tokens\":12}}}"
        ))])
        .await;
        let sibling = spawn_rung(vec![Answer::ResponsesStream(&[
            super::ladder_tests::RESPONSES_TEXT_FRAME,
        ])])
        .await;
        let mut first = super::ladder_tests::responses_wire("azure", &azure.url, &[]);
        first.provider = "azure_openai".to_string();
        let second = super::ladder_tests::responses_wire("novita", &sibling.url, &[]);
        let route = [first, second];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("another provider serves the filtered Responses turn");
        };
        assert_eq!(committed.depth, 1);
        drop(committed);
        let story = harness.story().await;
        assert_eq!(
            story["starts"][1]["failure"]["provider_content_filter"],
            true
        );
        assert_eq!(
            story["settles"][0]["failure"]["provider_content_filter"],
            true
        );
    });
}

#[test]
fn a_typed_content_filter_code_under_403_fails_over_as_a_filter() {
    block_on(async {
        let harness = Harness::new();
        let azure = spawn_rung(vec![Answer::Forbidden(concat!(
            "{\"error\":{\"code\":\"content_filter\",",
            "\"message\":\"The response was filtered due to the prompt triggering ",
            "the content management policy.\"}}"
        ))])
        .await;
        let sibling = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            provider_wire("azure", &azure.url, "azure_openai"),
            provider_wire("novita", &sibling.url, "openai"),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the 403 filter verdict fails over like any provider filter");
        };
        assert_eq!(committed.depth, 1);
        drop(committed);
        let story = harness.story().await;
        let failure = &story["starts"][1]["failure"];
        assert_eq!(failure["failure_class"], "refusal");
        assert_eq!(failure["provider_content_filter"], true);
    });
}

#[test]
fn a_rule_rungs_own_opt_in_does_not_spend_the_one_shot_gate() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null, ["refusal:content_policy"], null]}))
            .await;
        let azure = spawn_rung(vec![Answer::Rejected(AZURE_FILTERED_BODY)]).await;
        let opted = spawn_rung(vec![Answer::Rejected(AZURE_FILTERED_BODY)]).await;
        let sibling = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            provider_wire("azure-east", &azure.url, "azure_openai"),
            DeploymentWire {
                native_tool_translation: Default::default(),
                failover_only_on: Some(vec!["refusal:content_policy".to_string()]),
                ..provider_wire("azure-west", &opted.url, "azure_openai")
            },
            provider_wire("novita", &sibling.url, "openai"),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the other provider still takes the refusal after the rule rung's");
        };
        assert_eq!(committed.depth, 2);
        drop(committed);
        assert_eq!(opted.accepted.lock().expect("lock").len(), 1);
        assert_eq!(sibling.accepted.lock().expect("lock").len(), 1);
    });
}
