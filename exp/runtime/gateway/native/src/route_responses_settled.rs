//! Responses formatting for an attempt already settled by the waterfall.

use axum::http::StatusCode;
use axum::response::Response;

use crate::admission::{served_headers, Admission};
use crate::encode::compact_json;
use crate::events::Event;
use crate::relay::collection_public_error;
use crate::replay::{CachedResponse, OwnerLease};
use crate::respond::{
    cached_response, complete_visible_refusal, error_response, json_response, sse_body_response,
};
use crate::tool_search::{completed_responses_body_for, encode_responses_sse};
use crate::waterfall::SettledAttempt;

/// Answer one Responses attempt that the waterfall already settled: a
/// terminal without semantics or an exhausted ladder flushing inspected refusals.
pub(super) async fn settled_responses_response(
    admission: &Admission,
    settled: SettledAttempt,
    created_at: i64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let served = settled.served();
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    let failed = refusal_completed.is_none() && matches!(events.last(), Some(Event::Failed(_)));
    if failed && !admission.stream {
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        if let Some(Event::Failed(failure)) = events.last() {
            return error_response(&collection_public_error(&failure.clone().boundary()));
        }
    }
    let headers = served_headers(admission, client_request_id.as_deref(), served.clone());
    if admission.stream {
        let body = match encode_responses_sse(admission, served.depth, created_at, &events, None) {
            Ok(body) => body,
            Err(error) => return error_response(&error),
        };
        if failed {
            // A failed flush is not a replayable success.
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return sse_body_response(&headers, body);
        }
        if let Some(mut owner) = lease.take() {
            let mut sorted = headers.clone();
            sorted.sort();
            let cached = CachedResponse {
                status_code: 200,
                media_type: "text/event-stream; charset=utf-8".to_string(),
                headers: sorted,
                body: body.clone(),
            };
            return match owner.complete(cached.clone()).await {
                Ok(()) => cached_response(&cached),
                Err(error) => error_response(&error),
            };
        }
        return sse_body_response(&headers, body);
    }
    let aggregated =
        match completed_responses_body_for(admission, served.depth, created_at, &events, None) {
            Ok(aggregated) => aggregated,
            Err(error) => return error_response(&error),
        };
    if let Some(failure) = &aggregated.failure {
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&failure.clone().boundary().public_error());
    }
    if let Some(mut owner) = lease.take() {
        let mut sorted = headers.clone();
        sorted.sort();
        let cached = CachedResponse {
            status_code: 200,
            media_type: "application/json".to_string(),
            headers: sorted,
            body: compact_json(&aggregated.body).into_bytes(),
        };
        return match owner.complete(cached.clone()).await {
            Ok(()) => cached_response(&cached),
            Err(error) => error_response(&error),
        };
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}
