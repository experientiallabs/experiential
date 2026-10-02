//! Shared content capture, independent of destination and hosted tenancy policy.

mod bounds;
pub(crate) mod budget;
pub(crate) mod collector;
pub(crate) mod delivery;
mod local;
mod local_payload;
mod local_store;
mod messages;
pub(crate) mod metrics;
mod projection;
pub(crate) mod python;
pub(crate) mod reasoning;
pub(crate) mod record;
mod relay;
pub(crate) mod response;

/// Optional caller correlation, never authentication, routing or idempotency authority.
/// Ignore malformed or repeated headers rather than reject otherwise valid inference.
pub(crate) fn session_id(headers: &axum::http::HeaderMap) -> Option<&str> {
    let mut values = headers.get_all("x-session-id").iter();
    let value = values.next()?.to_str().ok()?;
    (values.next().is_none()
        && !value.is_empty()
        && value.len() <= 512
        && value.bytes().all(|byte| byte.is_ascii_graphic()))
    .then_some(value)
}

#[cfg(test)]
mod tests {
    use super::session_id;
    use axum::http::{HeaderMap, HeaderValue};

    #[test]
    fn correlation_is_optional_bounded_and_unambiguous() {
        let mut headers = HeaderMap::new();
        assert_eq!(session_id(&headers), None);
        headers.insert("x-session-id", HeaderValue::from_static("session-123"));
        assert_eq!(session_id(&headers), Some("session-123"));
        headers.append("x-session-id", HeaderValue::from_static("session-456"));
        assert_eq!(session_id(&headers), None);
        for value in ["", " ", "has space", "\t", &"x".repeat(513), "雪"] {
            headers.insert(
                "x-session-id",
                HeaderValue::from_bytes(value.as_bytes()).unwrap(),
            );
            assert_eq!(session_id(&headers), None);
        }
        headers.insert(
            "x-session-id",
            HeaderValue::from_str(&"x".repeat(512)).unwrap(),
        );
        assert!(session_id(&headers).is_some());
    }
}
