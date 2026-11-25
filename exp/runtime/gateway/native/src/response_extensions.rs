//! Host-supplied additive members on a finished completion.
//!
//! A host that settles requests through its own ledger often reports what it
//! settled on the response itself: the billed amount on `usage` (OpenRouter's
//! `usage.cost` shape), the rung that served at the top level, and a disclosure
//! header. The host returns these facts in the reply to the `price` or
//! finalizing `settle` call, which the data plane already awaits before it
//! encodes a stream's terminal frames or assembles a buffered body, so the
//! members are stamped natively with no second pass over the response bytes.
//!
//! The reply shape is:
//!
//! ```json
//! {"response_extensions": {
//!     "usage": {"cost": 0.0016, "is_byok": false},
//!     "body": {"provider": "example"},
//!     "headers": [["x-example-notice", "value"]]
//! }}
//! ```
//!
//! Every member is ADDITIVE: a key the engine already emits is never replaced,
//! so the OpenAI and Anthropic contracts the engine renders stay exactly as
//! they are and an unmodified SDK parses the result. Headers apply only to a
//! buffered answer, because a live stream's headers leave before settlement.
//! A missing, malformed, or oversized extension is ignored: settlement has
//! already landed, so the answer is still correct without the extras.
//!
//! The members ride the usage-bearing terminal payload only: the buffered
//! Chat, Responses, and Messages bodies, the Chat stream's final usage chunk,
//! the Messages `message_delta` event, the Responses terminal
//! `response.completed` / `response.incomplete` envelope, and the Images API
//! body (which gains a cost-only `usage` object when its lane reports none).
//! Retention for keyed replay happens after the stamp, so a replay reproduces
//! the same bytes.

use serde_json::{Map, Value};

/// Upper bound on the serialized extension a host may attach to one answer.
const MAXIMUM_EXTENSION_BYTES: usize = 16 * 1024;

/// Upper bound on the headers a host may attach to one buffered answer.
const MAXIMUM_EXTENSION_HEADERS: usize = 8;

/// Header names the engine owns on every answer; a host never sets them.
const RESERVED_HEADERS: &[&str] = &[
    "connection",
    "content-encoding",
    "content-length",
    "content-type",
    "retry-after",
    "transfer-encoding",
    "x-request-id",
];

/// Header name prefix the engine owns for its own routing facts.
const RESERVED_HEADER_PREFIX: &str = "x-gateway-";

/// The validated additive members one settlement reply carried.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ResponseExtensions {
    usage: Map<String, Value>,
    body: Map<String, Value>,
    headers: Vec<(String, String)>,
}

impl ResponseExtensions {
    /// Read the extension from one `price` or `settle` reply, if it carries one.
    ///
    /// Returns `None` for an empty reply, an absent or empty extension, and
    /// any extension that fails validation (logged once, content-free).
    pub(crate) fn from_reply(reply: &str) -> Option<Self> {
        if !reply.contains("response_extensions") {
            return None;
        }
        if reply.len() > MAXIMUM_EXTENSION_BYTES {
            log_rejected("oversized");
            return None;
        }
        let parsed: Value = serde_json::from_str(reply).ok()?;
        let extension = parsed.get("response_extensions")?;
        match Self::from_value(extension) {
            Ok(extensions) if !extensions.is_empty() => Some(extensions),
            Ok(_) => None,
            Err(reason) => {
                log_rejected(reason);
                None
            }
        }
    }

    /// Validate one decoded `response_extensions` object.
    fn from_value(value: &Value) -> Result<Self, &'static str> {
        let object = value.as_object().ok_or("not_an_object")?;
        let usage = optional_object(object, "usage")?;
        let body = optional_object(object, "body")?;
        let headers = match object.get("headers") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(entries)) => headers_from(entries)?,
            Some(_) => return Err("headers_not_a_list"),
        };
        Ok(Self {
            usage,
            body,
            headers,
        })
    }

    /// Whether the extension carries nothing to stamp.
    fn is_empty(&self) -> bool {
        self.usage.is_empty() && self.body.is_empty() && self.headers.is_empty()
    }

    /// Stamp one buffered answer body: `usage` members onto its top-level
    /// `usage` object and `body` members beside it. A body without a usage
    /// object (an unaccounted answer) receives only the body members.
    pub(crate) fn extend_body(&self, body: &mut Value) {
        let Some(holder) = body.as_object_mut() else {
            return;
        };
        if let Some(usage) = holder.get_mut("usage").and_then(Value::as_object_mut) {
            merge_absent(usage, &self.usage);
        }
        merge_absent(holder, &self.body);
    }

    /// Stamp one media answer (the Images API). A per-image priced answer
    /// carries no `usage` object at all, so the host's usage members become
    /// its `usage` object rather than being dropped; token counts are never
    /// invented. A `usage` the engine emitted (even `null`) is kept.
    pub(crate) fn extend_media_body(&self, body: &mut Value) {
        if let Some(holder) = body.as_object_mut() {
            if !holder.contains_key("usage") && !self.usage.is_empty() {
                holder.insert("usage".to_string(), Value::Object(self.usage.clone()));
            }
        }
        self.extend_body(body);
    }

    /// Append the host's headers to one buffered answer's header list,
    /// skipping any name the answer already carries.
    pub(crate) fn extend_headers(&self, headers: &mut Vec<(String, String)>) {
        for (name, value) in &self.headers {
            if !headers
                .iter()
                .any(|(present, _)| present.eq_ignore_ascii_case(name))
            {
                headers.push((name.clone(), value.clone()));
            }
        }
    }

    /// Stamp the usage-bearing terminal frame among one batch of SSE frames.
    ///
    /// Each frame is `[event: NAME\n]data: JSON\n\n` with exactly one data
    /// line, as every native encoder emits it. Only the first frame whose
    /// payload carries a usage object in a recognized terminal position is
    /// re-encoded; every other frame is left byte-identical.
    pub(crate) fn extend_frames(&self, frames: &mut [String]) {
        if self.usage.is_empty() && self.body.is_empty() {
            return;
        }
        for frame in frames.iter_mut() {
            if !frame.contains("\"usage\"") {
                continue;
            }
            if let Some(rewritten) = self.extend_frame(frame) {
                *frame = rewritten;
                return;
            }
        }
    }

    /// Stamp a buffered SSE body (a whole stream assembled at once).
    pub(crate) fn extend_sse_body(&self, body: &mut Vec<u8>) {
        let Ok(text) = std::str::from_utf8(body) else {
            return;
        };
        let mut frames: Vec<String> = text
            .split_inclusive("\n\n")
            .map(ToString::to_string)
            .collect();
        self.extend_frames(&mut frames);
        *body = frames.concat().into_bytes();
    }

    /// Re-encode one frame when its payload is a terminal usage carrier.
    fn extend_frame(&self, frame: &str) -> Option<String> {
        let data_start = if frame.starts_with("data: ") {
            0
        } else {
            frame.find("\ndata: ")? + 1
        };
        let json_start = data_start + "data: ".len();
        let json_end = frame[json_start..].find('\n')? + json_start;
        let mut payload: Value = serde_json::from_str(&frame[json_start..json_end]).ok()?;
        if !self.extend_terminal_payload(&mut payload) {
            return None;
        }
        let encoded = serde_json::to_string(&payload).ok()?;
        Some(format!(
            "{}{}{}",
            &frame[..json_start],
            encoded,
            &frame[json_end..]
        ))
    }

    /// Stamp one decoded terminal payload; returns whether it was one.
    fn extend_terminal_payload(&self, payload: &mut Value) -> bool {
        let kind = payload
            .get("type")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        match kind.as_deref() {
            // Responses: the envelope nests under `response`.
            Some("response.completed" | "response.incomplete") => {
                match payload.get_mut("response") {
                    Some(envelope) if has_usage_object(envelope) => {
                        self.extend_body(envelope);
                        true
                    }
                    _ => false,
                }
            }
            // Messages: the final usage rides `message_delta` beside `delta`.
            Some("message_delta") if has_usage_object(payload) => {
                self.extend_body(payload);
                true
            }
            Some(_) => false,
            // Chat: the final chunk carries `usage` at the top level.
            None if payload.get("object").and_then(Value::as_str)
                == Some("chat.completion.chunk")
                && has_usage_object(payload) =>
            {
                self.extend_body(payload);
                true
            }
            None => false,
        }
    }
}

/// Whether a payload carries a `usage` object (not `null`).
fn has_usage_object(payload: &Value) -> bool {
    payload.get("usage").is_some_and(Value::is_object)
}

/// Insert every member of `extra` that `target` does not already carry.
fn merge_absent(target: &mut Map<String, Value>, extra: &Map<String, Value>) {
    for (key, value) in extra {
        if !target.contains_key(key) {
            target.insert(key.clone(), value.clone());
        }
    }
}

/// Read one optional object member of the extension.
fn optional_object(
    object: &Map<String, Value>,
    key: &'static str,
) -> Result<Map<String, Value>, &'static str> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(members)) => Ok(members.clone()),
        Some(_) => Err("member_not_an_object"),
    }
}

/// Validate the host's header pairs: bounded, well-formed, never engine-owned.
fn headers_from(entries: &[Value]) -> Result<Vec<(String, String)>, &'static str> {
    if entries.len() > MAXIMUM_EXTENSION_HEADERS {
        return Err("too_many_headers");
    }
    let mut headers = Vec::with_capacity(entries.len());
    for entry in entries {
        let pair = entry.as_array().filter(|pair| pair.len() == 2);
        let (Some(name), Some(value)) = (
            pair.and_then(|pair| pair[0].as_str()),
            pair.and_then(|pair| pair[1].as_str()),
        ) else {
            return Err("header_not_a_pair");
        };
        let name = name.to_ascii_lowercase();
        let token = !name.is_empty()
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-');
        if !token {
            return Err("header_name_invalid");
        }
        if RESERVED_HEADERS.contains(&name.as_str()) || name.starts_with(RESERVED_HEADER_PREFIX) {
            return Err("header_name_reserved");
        }
        if !value.bytes().all(|byte| (0x20..0x7f).contains(&byte)) {
            return Err("header_value_invalid");
        }
        headers.push((name, value.to_string()));
    }
    Ok(headers)
}

/// One content-free line when a host's extension is refused.
fn log_rejected(reason: &'static str) {
    let line = serde_json::json!({"event": "response_extensions_rejected", "reason": reason});
    eprintln!("exp-gateway-native: {line}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn extensions() -> ResponseExtensions {
        ResponseExtensions::from_reply(
            &json!({"response_extensions": {
                "usage": {"cost": 0.0016, "is_byok": false},
                "body": {"provider": "example"},
                "headers": [["X-Example-Notice", "settled"]],
            }})
            .to_string(),
        )
        .expect("valid extension")
    }

    #[test]
    fn an_empty_or_absent_reply_carries_nothing() {
        assert_eq!(ResponseExtensions::from_reply("{}"), None);
        assert_eq!(ResponseExtensions::from_reply(""), None);
        assert_eq!(
            ResponseExtensions::from_reply(r#"{"response_extensions":{}}"#),
            None
        );
        assert_eq!(
            ResponseExtensions::from_reply(r#"{"response_extensions":null}"#),
            None
        );
    }

    #[test]
    fn malformed_and_engine_owned_extensions_are_refused() {
        for reply in [
            json!({"response_extensions": "cost"}),
            json!({"response_extensions": {"usage": [1]}}),
            json!({"response_extensions": {"headers": [["x-request-id", "forged"]]}}),
            json!({"response_extensions": {"headers": [["x-gateway-provider", "forged"]]}}),
            json!({"response_extensions": {"headers": [["content-length", "1"]]}}),
            json!({"response_extensions": {"headers": [["bad name", "v"]]}}),
            json!({"response_extensions": {"headers": [["x-ok", "line\r\nbreak"]]}}),
            json!({"response_extensions": {"headers": [["x-ok"]]}}),
        ] {
            assert_eq!(
                ResponseExtensions::from_reply(&reply.to_string()),
                None,
                "{reply}"
            );
        }
        let oversized = json!({"response_extensions": {"body": {"pad": "x".repeat(20_000)}}});
        assert_eq!(ResponseExtensions::from_reply(&oversized.to_string()), None);
    }

    #[test]
    fn a_buffered_body_gains_members_without_losing_engine_fields() {
        let mut body = json!({
            "id": "chatcmpl_1", "object": "chat.completion",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        });
        extensions().extend_body(&mut body);
        assert_eq!(
            body,
            json!({
                "id": "chatcmpl_1", "object": "chat.completion",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
                          "cost": 0.0016, "is_byok": false},
                "provider": "example",
            })
        );
    }

    #[test]
    fn an_engine_member_is_never_replaced() {
        let mut body = json!({"id": "x", "usage": {"cost": 9}, "provider": "engine"});
        extensions().extend_body(&mut body);
        assert_eq!(body["usage"]["cost"], json!(9));
        assert_eq!(body["provider"], json!("engine"));
        assert_eq!(body["usage"]["is_byok"], json!(false));
    }

    #[test]
    fn a_media_answer_without_usage_gains_a_cost_only_usage_object() {
        let mut body = json!({"created": 1, "data": [{"b64_json": "AAAA"}]});
        extensions().extend_media_body(&mut body);
        assert_eq!(body["usage"], json!({"cost": 0.0016, "is_byok": false}));
        assert_eq!(body["provider"], json!("example"));
        let mut accounted = json!({"data": [], "usage": {"input_tokens": 4}});
        extensions().extend_media_body(&mut accounted);
        assert_eq!(accounted["usage"]["input_tokens"], json!(4));
        assert_eq!(accounted["usage"]["cost"], json!(0.0016));
    }

    #[test]
    fn headers_are_appended_once_and_never_duplicate_a_present_name() {
        let mut headers = vec![("x-example-notice".to_string(), "engine".to_string())];
        extensions().extend_headers(&mut headers);
        assert_eq!(headers.len(), 1);
        let mut headers = vec![("x-request-id".to_string(), "r".to_string())];
        extensions().extend_headers(&mut headers);
        assert_eq!(
            headers.last(),
            Some(&("x-example-notice".to_string(), "settled".to_string()))
        );
    }

    #[test]
    fn the_chat_usage_chunk_is_stamped_and_other_frames_stay_byte_identical() {
        let delta = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"usage\"}}],\"usage\":null}\n\n";
        let usage = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"total_tokens\":5}}\n\n";
        let mut frames = vec![
            delta.to_string(),
            usage.to_string(),
            "data: [DONE]\n\n".to_string(),
        ];
        extensions().extend_frames(&mut frames);
        assert_eq!(frames[0], delta);
        assert_eq!(
            frames[1],
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"total_tokens\":5,\"cost\":0.0016,\"is_byok\":false},\"provider\":\"example\"}\n\n"
        );
        assert_eq!(frames[2], "data: [DONE]\n\n");
    }

    #[test]
    fn the_messages_delta_is_stamped_but_message_start_is_not() {
        let start = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":3}}}\n\n";
        let delta = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n";
        let mut frames = vec![start.to_string(), delta.to_string()];
        extensions().extend_frames(&mut frames);
        assert_eq!(frames[0], start);
        assert_eq!(
            frames[1],
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2,\"cost\":0.0016,\"is_byok\":false},\"provider\":\"example\"}\n\n"
        );
    }

    #[test]
    fn the_responses_terminal_envelope_is_stamped_inside_response() {
        let completed = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"total_tokens\":5}}}\n\n";
        let mut frames = vec![completed.to_string()];
        extensions().extend_frames(&mut frames);
        let payload: Value =
            serde_json::from_str(frames[0].split("data: ").nth(1).unwrap().trim()).unwrap();
        assert_eq!(payload["response"]["usage"]["cost"], json!(0.0016));
        assert_eq!(payload["response"]["provider"], json!("example"));
        assert!(payload.get("provider").is_none());
    }

    #[test]
    fn an_unrecognized_usage_frame_is_left_alone() {
        let frame = "event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"usage\":{}}}\n\n";
        let mut frames = vec![frame.to_string()];
        extensions().extend_frames(&mut frames);
        assert_eq!(frames[0], frame);
    }

    #[test]
    fn a_buffered_sse_body_is_stamped_in_place() {
        let mut body = b"data: {\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"total_tokens\":5}}\n\ndata: [DONE]\n\n".to_vec();
        extensions().extend_sse_body(&mut body);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("\"cost\":0.0016"));
        assert!(text.ends_with("data: [DONE]\n\n"));
    }
}
