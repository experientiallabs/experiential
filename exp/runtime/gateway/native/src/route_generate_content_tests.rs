//! Unit tests for the Gemini surface's credential and path parsing.
use super::*;

#[test]
fn keys_come_from_header_query_or_bearer_in_that_order() {
    let mut headers = HeaderMap::new();
    let uri: Uri = "/v1beta/models/m:generateContent?key=xpl_q&alt=sse"
        .parse()
        .unwrap();
    assert_eq!(generate_content_api_key(&headers, &uri).unwrap(), "xpl_q");
    headers.insert("x-goog-api-key", HeaderValue::from_static(" xpl_h "));
    assert_eq!(generate_content_api_key(&headers, &uri).unwrap(), "xpl_h");
    let bare: Uri = "/v1beta/models".parse().unwrap();
    let mut bearer = HeaderMap::new();
    bearer.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer xpl_b"),
    );
    assert_eq!(generate_content_api_key(&bearer, &bare).unwrap(), "xpl_b");
    let missing = generate_content_api_key(&HeaderMap::new(), &bare).unwrap_err();
    assert_eq!(missing.status_code, 401);
    assert!(asks_for_sse(&uri));
    assert!(!asks_for_sse(&bare));
}

#[test]
fn model_actions_split_on_the_last_colon() {
    assert_eq!(
        split_action("gemini-4-argon:free:streamGenerateContent"),
        (
            "gemini-4-argon:free".to_string(),
            Some("streamGenerateContent".to_string())
        )
    );
    assert_eq!(
        split_action("gemini-4-argon"),
        ("gemini-4-argon".to_string(), None)
    );
    assert_eq!(percent_decode("xpl_%5Fa%2Fb"), "xpl__a/b");
}

#[test]
fn a_streamed_refusal_settles_as_the_typed_refusal_ahead_of_empty_completion() {
    // A content-free refusal closed with a plain stop also satisfies the
    // empty-completion predicate; the refusal wins (Cursor, #1362).
    assert_eq!(
        stream_outcome(&Event::Completed, true, &[], true),
        StreamOutcome::Refusal
    );
    // A refusal the guardrail released only with the terminal flush, unseen
    // by the encoder at settle time, is still the refusal.
    let released = vec![Event::RefusalDelta("no".into()), Event::Completed];
    assert_eq!(
        stream_outcome(&Event::Completed, false, &released, true),
        StreamOutcome::Refusal
    );
    // A provider-typed failing terminal keeps its own class.
    let failed = Event::Failed(Failure::new(FailureClass::ProviderInternal, "x"));
    assert_eq!(
        stream_outcome(&failed, true, &[], false),
        StreamOutcome::Terminal
    );
    // Without refusal output the empty-completion rule still applies, then
    // the plain terminal.
    assert_eq!(
        stream_outcome(&Event::Completed, false, &[], true),
        StreamOutcome::EmptyCompletion
    );
    assert_eq!(
        stream_outcome(&Event::Completed, false, &[], false),
        StreamOutcome::Terminal
    );
}
