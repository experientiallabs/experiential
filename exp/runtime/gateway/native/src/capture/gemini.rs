//! Reassemble observed `streamGenerateContent` frames into one
//! `GenerateContentResponse`: parts concatenate in order (adjacent text of
//! the same kind merges), the last `usageMetadata` and the first
//! `finishReason` win, and the lifecycle is complete when exactly one
//! candidate finished.
use serde_json::{json, Map, Value};

pub(super) fn assemble(frames: &[Value]) -> (Value, bool) {
    let mut parts: Vec<Value> = Vec::new();
    let mut finish: Option<Value> = None;
    let mut usage: Option<Value> = None;
    let mut envelope = Map::new();
    let mut valid = !frames.is_empty();
    for frame in frames {
        let Some(object) = frame.as_object() else {
            valid = false;
            continue;
        };
        for key in [
            "responseId",
            "modelVersion",
            "x-experiential-ignored-parameters",
        ] {
            if let Some(value) = object.get(key) {
                envelope
                    .entry(key.to_string())
                    .or_insert_with(|| value.clone());
            }
        }
        if let Some(reported) = object.get("usageMetadata").filter(|v| v.is_object()) {
            usage = Some(reported.clone());
        }
        let Some(candidates) = object.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        // Content after the finish is a lifecycle fault the gateway's own
        // encoder never produces.
        valid &= finish.is_none() && candidates.len() == 1;
        let candidate = &candidates[0];
        if let Some(incoming) = candidate["content"]["parts"].as_array() {
            for part in incoming {
                merge_part(&mut parts, part);
            }
        }
        if let Some(reason) = candidate.get("finishReason").filter(|v| v.is_string()) {
            finish = Some(reason.clone());
        }
    }
    let mut body = json!({
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": finish.clone().unwrap_or(Value::Null),
            "index": 0,
        }],
    });
    if let Some(usage) = usage {
        body["usageMetadata"] = usage;
    }
    if let Some(target) = body.as_object_mut() {
        target.extend(envelope);
    }
    (body, valid && finish.is_some())
}

fn merge_part(parts: &mut Vec<Value>, part: &Value) {
    let text = part.get("text").and_then(Value::as_str);
    let thought = part.get("thought") == Some(&Value::Bool(true));
    if let (Some(text), Some(last)) = (text, parts.last_mut()) {
        let mergeable = last.get("thoughtSignature").is_none()
            && part.get("thoughtSignature").is_none()
            && (last.get("thought") == Some(&Value::Bool(true))) == thought;
        if mergeable {
            if let Some(Value::String(existing)) = last.get_mut("text") {
                existing.push_str(text);
                return;
            }
        }
    }
    parts.push(part.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_merge_into_one_response_and_need_a_finish() {
        let frames = vec![
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]},"index":0}],"responseId":"r"}),
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"totalTokenCount":3}}),
        ];
        let (body, complete) = assemble(&frames);
        assert!(complete);
        assert_eq!(
            body["candidates"][0]["content"]["parts"],
            json!([{"text": "hello"}])
        );
        assert_eq!(body["candidates"][0]["finishReason"], "STOP");
        assert_eq!(body["usageMetadata"]["totalTokenCount"], 3);
        assert_eq!(body["responseId"], "r");
        let (_, incomplete) = assemble(&frames[..1]);
        assert!(!incomplete);
    }
}
