//! Unit tests for the video content relay's pure helpers.

use super::*;

#[test]
fn content_fetch_reads_url_headers_and_type() {
    let fetch = content_fetch(
        r#"{"url":"https://example.com/v.mp4","headers":{"x-goog-api-key":"k"},"content_type":"video/mp4"}"#,
    )
    .expect("fetch");
    assert_eq!(fetch.url, "https://example.com/v.mp4");
    assert_eq!(
        fetch.headers,
        vec![("x-goog-api-key".to_string(), "k".to_string())]
    );
    assert_eq!(fetch.content_type, "video/mp4");
}

#[test]
fn content_fetch_defaults_type_and_drops_non_string_headers() {
    let fetch =
        content_fetch(r#"{"url":"https://example.com/v","headers":{"a":1}}"#).expect("fetch");
    assert!(fetch.headers.is_empty());
    assert_eq!(fetch.content_type, "video/mp4");
    assert!(content_fetch(r#"{"headers":{}}"#).is_none());
    assert!(content_fetch("not json").is_none());
}

#[test]
fn asset_errors_carry_public_codes() {
    assert_eq!(asset_unavailable().status_code, 404);
    assert_eq!(asset_failed().status_code, 502);
    assert_eq!(invalid("bad", Some("size")).param.as_deref(), Some("size"));
}
