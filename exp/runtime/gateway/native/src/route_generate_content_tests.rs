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
