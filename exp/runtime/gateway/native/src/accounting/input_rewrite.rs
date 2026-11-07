//! Preserve the authenticated prefix when an input modifier changes messages.

use serde_json::{json, Value};

use super::{
    inspection,
    search_subject::{folded, items, present, Subject},
};

fn authority(message: &Value) -> bool {
    present(&message["provider_reasoning"])
        || message["tool_is_error"] == true
        || [
            "provider_item_id",
            "provider_output_index",
            "provider_status",
            "provider_phase",
            "provider_tool_name",
            "provider_tool_namespace",
            "provider_tool_caller",
        ]
        .iter()
        .any(|k| !message[k].is_null())
        || items(&message["tool_calls"]).iter().any(|call| {
            [
                "raw_arguments",
                "provider_item_id",
                "provider_output_index",
                "provider_status",
                "provider_namespace",
                "provider_caller",
            ]
            .iter()
            .any(|k| !call[k].is_null())
        })
}

fn carrier(message: &Value) -> bool {
    items(&message["provider_reasoning"]).iter().any(|b| {
        matches!(
            b["kind"].as_str(),
            Some("reasoning_content" | "sealed_reasoning_content")
        )
    })
}

pub(super) fn hosted(subject: &Subject, replacement: &Value) -> Result<Vec<Value>, ()> {
    let mut replacement = replacement
        .as_array()
        .ok_or(())?
        .iter()
        .map(super::input_schema::message)
        .collect::<Result<Vec<_>, _>>()?;
    let original = items(&subject.request["messages"]);
    if let Some(bound) = original.iter().rposition(carrier) {
        if replacement
            .iter()
            .all(|m| matches!(m["role"].as_str(), Some("system" | "developer" | "user")))
        {
            return Ok(replacement);
        }
        if replacement.len() <= bound {
            return Err(());
        }
        for i in 0..=bound {
            if original[i]["role"] != replacement[i]["role"] {
                return Err(());
            }
            if matches!(original[i]["role"].as_str(), Some("assistant" | "tool")) {
                if !super::equality::equal(
                    &subject.project_message(&original[i]),
                    &subject.project_message(&replacement[i]),
                ) {
                    return Err(());
                }
                replacement[i] = original[i].clone();
            }
        }
    }
    Ok(replacement)
}

pub(super) fn restore(subject: &Subject, mut replacement: Vec<Value>) -> Result<Vec<Value>, ()> {
    let original = items(&subject.request["messages"]);
    let Some(bound) = original.iter().rposition(authority) else {
        return if replacement.iter().any(authority) {
            Err(())
        } else {
            Ok(replacement)
        };
    };
    let carriers: Vec<_> = original
        .iter()
        .enumerate()
        .filter(|(_, m)| carrier(m))
        .map(|(i, _)| i)
        .collect();
    if !carriers.is_empty() {
        let next: Vec<_> = replacement
            .iter()
            .enumerate()
            .filter(|(_, m)| carrier(m))
            .map(|(i, _)| i)
            .collect();
        if next.is_empty() {
            return if replacement
                .iter()
                .all(|m| matches!(m["role"].as_str(), Some("system" | "developer" | "user")))
            {
                Ok(replacement)
            } else {
                Err(())
            };
        }
        if carriers != next
            || carriers
                .iter()
                .any(|i| !super::equality::equal(&original[*i], &replacement[*i]))
        {
            return Err(());
        }
    }
    if replacement.len() <= bound
        || (0..=bound).any(|i| {
            !super::equality::equal(
                &subject.project_message(&original[i]),
                &subject.project_message(&replacement[i]),
            )
        })
        || replacement[bound + 1..].iter().any(authority)
    {
        return Err(());
    }
    replacement[..=bound].clone_from_slice(&original[..=bound]);
    Ok(replacement)
}

pub(super) fn regex(adapter: &Value) -> Result<Value, ()> {
    let detector = inspection::cached_detector(&adapter["specification"]).ok_or(())?;
    let mut flagged = false;
    let mut immutable = false;
    let context = adapter["immutable_context"].as_str().ok_or(())?;
    if !context.is_empty() {
        immutable = detector.matches(context).map_err(|_| ())?;
        flagged |= immutable;
    }
    let mut messages = Vec::new();
    for original in items(&adapter["redaction_messages"]) {
        let mut message = original.clone();
        if let Some(text) = detector.redact(&folded(original)).map_err(|_| ())? {
            flagged = true;
            if present(&original["provider_text_blocks"]) || original["tool_is_error"] == true {
                immutable = true;
            } else {
                message["content"] = json!(text);
            }
        }
        for call in items(&original["tool_calls"]) {
            let raw = match call["raw_arguments"].as_str() {
                Some(raw) => raw.to_owned(),
                None => super::json_text::spaced(&call["arguments"]).map_err(|_| ())?,
            };
            let found = detector.matches(&raw).map_err(|_| ())?;
            flagged |= found;
            immutable |= found;
        }
        messages.push(message);
    }
    Ok(
        json!({"flagged": flagged, "replacement_messages": if flagged && !immutable { json!(messages) } else { Value::Null }}),
    )
}

pub(super) fn verdict(mut value: Value, action: &str) -> Result<Value, ()> {
    if value.as_object().is_none_or(|v| {
        v.keys().any(|k| {
            !matches!(
                k.as_str(),
                "flagged" | "replacement_text" | "replacement_messages"
            )
        })
    }) {
        return Err(());
    }
    let flagged = super::input_schema::boolean(&value["flagged"])?;
    let messages = !value["replacement_messages"].is_null();
    if !value["replacement_text"].is_null()
        || (messages && (!flagged || action != "modify"))
        || (flagged && action == "modify" && !messages)
    {
        return Err(());
    }
    value["flagged"] = json!(flagged);
    Ok(value)
}
