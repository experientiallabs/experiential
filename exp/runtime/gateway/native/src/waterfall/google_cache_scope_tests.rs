//! Verified Vertex project aliases bind the response namespace, never the create destination.

use super::tests::{
    answer, claim, execute_test, expiry, http, ready, response, server, wire, Host,
};
use super::*;

#[test]
fn trusted_prefix_preserves_origin_location_and_numeric_project_identity() {
    let url = "https://us-central1-aiplatform.googleapis.com/v1/projects/fruit-project/locations/us-central1/publishers/google/models/gemini-test:streamGenerateContent";
    let endpoint = cache_endpoint(url, EndpointPolicy::Official).unwrap();
    let expected_url = endpoint.url.clone();
    let prefix = "projects/123456789/locations/us-central1/cachedContents/";
    let bound = bind_resource_prefix(endpoint, prefix).unwrap();
    assert_eq!(bound.url, expected_url);
    assert!(valid_resource(&bound, &format!("{prefix}resource")));
    assert!(!valid_resource(
        &bound,
        "projects/987654321/locations/us-central1/cachedContents/resource"
    ));
    for invalid in [
        "cachedContents/",
        "projects/fruit-project/locations/us-central1/cachedContents/",
        "projects/0123/locations/us-central1/cachedContents/",
        "projects/0/locations/us-central1/cachedContents/",
        "projects/123456789/locations/global/cachedContents/",
        "projects/123456789/locations/us-central1/cachedContents/nested/",
        "projects/123456789/locations/us-central1/cachedContents/?query",
    ] {
        assert!(bind_resource_prefix(
            cache_endpoint(url, EndpointPolicy::Official).unwrap(),
            invalid
        )
        .is_err());
    }
    let numeric = url.replace("fruit-project", "123456789");
    assert!(bind_resource_prefix(
        cache_endpoint(&numeric, EndpointPolicy::Official).unwrap(),
        prefix
    )
    .is_ok());
    assert!(bind_resource_prefix(
        cache_endpoint(&numeric, EndpointPolicy::Official).unwrap(),
        &prefix.replace("123456789", "987654321")
    )
    .is_err());
    let studio = cache_endpoint(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent",
        EndpointPolicy::Official,
    )
    .unwrap();
    assert!(bind_resource_prefix(studio, prefix).is_err());
}

#[tokio::test]
async fn ready_callback_cannot_use_a_resource_outside_its_frozen_prefix() {
    let original = wire("https://aiplatform.googleapis.com/v1/projects/fruit-project/locations/global/publishers/google/models/gemini-test:streamGenerateContent");
    let prefix = "projects/123456789/locations/global/cachedContents/";
    let name = format!("{prefix}existing");
    let mut reply = ready(&name);
    reply["resource_prefix"] = json!(prefix);
    let host = Host::new(vec![reply.clone()]);
    let overlaid = execute_test(&original, &host, EndpointPolicy::Official)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(overlaid.url, original.url);
    assert_eq!(overlaid.upstream_payload["cachedContent"], name);
    reply["resource_prefix"] = json!(prefix.replace("123456789", "987654321"));
    assert!(
        execute_test(&original, &Host::new(vec![reply]), EndpointPolicy::Official)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn project_id_create_records_only_the_host_verified_numeric_resource() {
    for response_project in ["123456789", "987654321"] {
        let expires = expiry();
        let prefix = "projects/123456789/locations/global/cachedContents/";
        let expected_name = format!("{prefix}test-cache");
        let mut body = response(&expires);
        body["name"] = json!(format!(
            "projects/{response_project}/locations/global/cachedContents/test-cache"
        ));
        let (base, task) = server(vec![answer(200, &body.to_string())]).await;
        let path = "/v1/projects/fruit-project/locations/global";
        let original = wire(&format!(
            "{base}{path}/publishers/google/models/gemini-test:streamGenerateContent"
        ));
        let mut create = claim(&format!("{base}{path}/cachedContents"), &expires);
        create["resource_prefix"] = json!(prefix);
        let success = response_project == "123456789";
        let completion = if success {
            ready(&expected_name)
        } else {
            json!({"state":"unavailable"})
        };
        let host = Host::new(vec![create, completion]);
        let result = execute_test(&original, &host, EndpointPolicy::Loopback)
            .await
            .unwrap();
        assert_eq!(result.is_some(), success);
        let calls = host.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[1].1["outcome"],
            if success { "ready" } else { "unknown" }
        );
        if success {
            assert_eq!(calls[1].1["name"], expected_name);
        } else {
            assert!(calls[1].1.get("name").is_none());
        }
        let requests = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with(&format!("POST {path}/cachedContents ")));
    }
}

#[test]
fn jurisdictional_host_matches_exact_location() {
    for location in ["us", "eu"] {
        let url = format!("https://aiplatform.{location}.rep.googleapis.com/v1/projects/fruit-project/locations/{location}/publishers/google/models/gemini-test:streamGenerateContent?alt=sse");
        assert!(cache_endpoint(&url, EndpointPolicy::Official).is_some());
        assert!(cache_endpoint(
            &url.replace(&format!("/locations/{location}/"), "/locations/global/"),
            EndpointPolicy::Official
        )
        .is_none());
        assert!(cache_endpoint(
            &url.replace(".googleapis.com", ".googleapis.com.evil.test"),
            EndpointPolicy::Official
        )
        .is_none());
    }
}

#[tokio::test]
async fn automatic_preflight_counts_only_prefix_and_skips_below_minimum() {
    for tokens in [4095, 6000] {
        let expires = expiry();
        let prefix = "projects/123456789/locations/us/cachedContents/";
        let name = format!("{prefix}test-cache");
        let response =
            json!({"name":name,"expireTime":expires.1,"usageMetadata":{"totalTokenCount":tokens}});
        let mut replies = vec![answer(200, &json!({"totalTokens": tokens}).to_string())];
        if tokens >= 4096 {
            replies.push(answer(200, &response.to_string()));
        }
        let (base, task) = server(replies).await;
        let path = "/v1/projects/fruit-project/locations/us";
        let mut original = wire(&format!(
            "{base}{path}/publishers/google/models/gemini-test:streamGenerateContent?alt=sse"
        ));
        original.automatic_cache = true;
        let mut create = claim(&format!("{base}{path}/cachedContents"), &expires);
        create["resource_prefix"] = json!(prefix);
        create["measure_tokens"] = json!(true);
        create["minimum_tokens"] = json!(4096);
        create["payload"]["model"] =
            json!("projects/fruit-project/locations/us/publishers/google/models/gemini-test");
        let mut accepted = ready(&name);
        accepted["resource_prefix"] = json!(prefix);
        let host = Host::new(vec![
            create,
            if tokens >= 4096 {
                accepted
            } else {
                json!({"state":"unavailable"})
            },
        ]);
        let result = execute_test(&original, &host, EndpointPolicy::Loopback)
            .await
            .unwrap();
        let requests = task.await.unwrap();
        assert!(requests[0].starts_with(&format!(
            "POST {path}/publishers/google/models/gemini-test:countTokens "
        )));
        assert!(!requests[0].contains("expireTime"));
        assert!(!requests[0].contains("generation-only"));
        assert!(!requests[0].contains("prefix and suffix"));
        if tokens < 4096 {
            assert!(result.is_none());
            assert_eq!(host.calls()[1].1["outcome"], "not_created");
            assert_eq!(requests.len(), 1);
        } else {
            let overlaid = result.unwrap();
            assert_eq!(overlaid.upstream_payload["cachedContent"], name);
            // This attempt wrote the cache its generation reads: usage reports
            // Google's created token count as writes read back in-call.
            assert_eq!(overlaid.automatic_cache_written_tokens, Some(tokens));
            assert!(requests[1].starts_with(&format!("POST {path}/cachedContents ")));
        }
    }
}

#[tokio::test]
async fn short_deadline_skips_preflight_and_acknowledges_no_create() {
    let expires = expiry();
    let (base, task) = server(vec![]).await;
    let path = "/v1/projects/fruit-project/locations/us";
    let original = wire(&format!(
        "{base}{path}/publishers/google/models/gemini-test:streamGenerateContent"
    ));
    let mut create = claim(&format!("{base}{path}/cachedContents"), &expires);
    create["resource_prefix"] = json!("projects/123456789/locations/us/cachedContents/");
    create["payload"]["model"] =
        json!("projects/fruit-project/locations/us/publishers/google/models/gemini-test");
    create["measure_tokens"] = json!(true);
    create["minimum_tokens"] = json!(4096);
    let host = Host::new(vec![create, json!({"state":"unavailable"})]);
    let result = execute(
        &http(),
        &original,
        "request",
        Instant::now() + Duration::from_secs(2),
        false,
        EndpointPolicy::Loopback,
        |method, argument| host.call(method, argument),
    )
    .await
    .unwrap();
    assert!(result.is_none());
    assert_eq!(host.calls()[1].1["outcome"], "not_created");
    assert!(task.await.unwrap().is_empty());
}

#[tokio::test]
async fn automatic_cache_failure_falls_back_to_the_plain_generation() {
    // A malformed host reply is an accounting failure: explicit (client-marked)
    // caching fails the attempt, automatic caching must dispatch the plain wire.
    let host = super::tests::Host::new(vec![json!({"state": "bogus"})]);
    let explicit = super::tests::wire("https://generativelanguage.googleapis.com/v1beta/models/gemini:streamGenerateContent?alt=sse");
    let failed = super::tests::execute_test(&explicit, &host, EndpointPolicy::Official).await;
    assert!(failed.is_err());
    assert!(fail_open(&explicit, failed).is_err());

    let mut automatic = explicit.clone();
    automatic.automatic_cache = true;
    let host = super::tests::Host::new(vec![json!({"state": "bogus"})]);
    let failed = super::tests::execute_test(&automatic, &host, EndpointPolicy::Official).await;
    assert!(failed.is_err());
    assert!(matches!(fail_open(&automatic, failed), Ok(None)));
    // Success and deliberate unavailability pass through untouched.
    assert!(matches!(fail_open(&automatic, Ok(None)), Ok(None)));
}

#[test]
fn only_request_shaped_cache_rejections_redial_plain() {
    for class in [
        FailureClass::InvalidRequest,
        FailureClass::ProviderNotFound,
        FailureClass::UnsupportedCapability,
    ] {
        assert!(plain_redial_after(&Failure::new(
            class,
            "cachedContent rejected"
        )));
    }
    for class in [
        FailureClass::Throttled,
        FailureClass::Timeout,
        FailureClass::ProviderInternal,
    ] {
        assert!(!plain_redial_after(&Failure::new(
            class,
            "upstream trouble"
        )));
    }
}

#[test]
fn refused_overlay_settles_without_opening_the_deployment_circuit() {
    // An evicted gateway-chosen resource answers 404; settling that as a missing
    // model would open the rung's circuit for every other request.
    let refused = overlay_refusal(Failure::new(
        FailureClass::ProviderNotFound,
        "cachedContent not found",
    ));
    assert_eq!(refused.failure_class, FailureClass::InvalidRequest);
    assert_eq!(refused.safe_message, "cachedContent not found");
    for class in [
        FailureClass::InvalidRequest,
        FailureClass::UnsupportedCapability,
    ] {
        assert_eq!(
            overlay_refusal(Failure::new(class, "refused")).failure_class,
            class
        );
    }
}
