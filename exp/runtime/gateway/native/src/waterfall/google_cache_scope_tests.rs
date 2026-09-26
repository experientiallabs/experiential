//! Verified Vertex project aliases bind the response namespace, never the create destination.

use super::tests::{answer, claim, execute_test, expiry, ready, response, server, wire, Host};
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
