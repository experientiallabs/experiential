//! Opaque transport context for an embedder's request policy.
//!
//! This header is NOT authenticated by the engine. An embedder using it for
//! authority must strip public copies and set it at its trusted ingress.
//! It is never added to provider payloads, replay identities, or ledger rows.

use axum::extract::Request;
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;

tokio::task_local! {
    pub(crate) static REQUEST_CONTEXT: Option<String>;
}

/// Treat absent, duplicate, oversized, or malformed context as unavailable.
pub(crate) fn from_headers(headers: &HeaderMap) -> Option<String> {
    let mut values = headers.get_all("x-exp-request-context").iter();
    let first = values.next()?;
    if values.next().is_some() || first.as_bytes().len() > 1024 {
        return None;
    }
    first
        .to_str()
        .ok()
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Bind context to this HTTP task, including callbacks before replay lookup.
pub(crate) async fn scope_request(request: Request, next: Next) -> Response {
    let context = from_headers(request.headers());
    REQUEST_CONTEXT.scope(context, next.run(request)).await
}

/// Capture a context before a callback crosses to a Python worker thread.
pub(crate) fn current() -> Option<String> {
    REQUEST_CONTEXT.try_with(Clone::clone).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguous_context_is_unavailable() {
        let mut headers = HeaderMap::new();
        assert_eq!(from_headers(&headers), None);
        headers.insert("x-exp-request-context", "US".parse().unwrap());
        assert_eq!(from_headers(&headers), Some("US".into()));
        headers.append("x-exp-request-context", "BY".parse().unwrap());
        assert_eq!(from_headers(&headers), None);
        headers.insert("x-exp-request-context", "x".repeat(1025).parse().unwrap());
        assert_eq!(from_headers(&headers), None);
    }

    #[tokio::test]
    async fn concurrent_requests_and_missing_context_are_isolated() {
        let (left, right) = tokio::join!(
            REQUEST_CONTEXT.scope(Some("US".into()), async {
                tokio::task::yield_now().await;
                current()
            }),
            REQUEST_CONTEXT.scope(Some("BY".into()), async {
                tokio::task::yield_now().await;
                current()
            })
        );
        assert_eq!(left, Some("US".into()));
        assert_eq!(right, Some("BY".into()));
        assert_eq!(current(), None);
    }
}
