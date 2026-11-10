//! Caller-facing wording for the gateway's own upstream failure sentences.
//!
//! A failure's `safe_message` is the durable, operator-facing sentence the
//! ledger, alert rules and dashboards key on ("provider service failed",
//! "all exact-model deployments are unavailable"), so it never changes here.
//! What a caller reads is derived at the public boundary instead: each known
//! sentence maps to one that says what happened in the caller's terms (the
//! model's provider, never a rung, lane, deployment or alias) and what to do
//! next. Any sentence not listed passes through unchanged, so a specific
//! message (a parameter refusal, a capability refusal) keeps its own words.
//!
//! Mirrored by `exp.runtime.openai_protocol.caller_copy`; the taxonomy parity
//! test drives both through the same table.

use std::borrow::Cow;

use crate::errors::FailureClass;

/// The caller sentence for `safe_message` of `class`, or the message itself.
pub fn caller_copy(class: FailureClass, safe_message: &str) -> Cow<'_, str> {
    match rewrite(class, safe_message) {
        Some(text) => Cow::Owned(text),
        None => Cow::Borrowed(safe_message),
    }
}

/// The remainder of `message` after `prefix`, without its leading separator.
fn after<'a>(message: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = message.strip_prefix(prefix)?;
    if rest.is_empty() {
        return Some("");
    }
    // A prefix only matches a whole leading clause, never a longer word.
    let trimmed = rest.strip_prefix(';').or_else(|| rest.strip_prefix(':'))?;
    Some(trimmed.trim())
}

/// The whole seconds named by a "retry in Ns" clause, when present.
fn retry_seconds(clause: &str) -> Option<u64> {
    let digits = clause.strip_prefix("retry in ")?.strip_suffix('s')?;
    // Plain ASCII digits only: `u64::parse` alone also accepts a leading `+`,
    // which the Python mirror's pattern refuses.
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn rewrite(class: FailureClass, message: &str) -> Option<String> {
    let text = match class {
        FailureClass::ProviderInternal => {
            if after(message, "provider service failed").is_some() {
                "The model's provider had a server error and no other provider for this \
                 model could take the request. Retry after a short delay; if it keeps \
                 failing, choose another model."
            } else if after(message, "all exact-model deployments are unavailable").is_some() {
                "No provider for this model can take the request right now. Retry after a \
                 short delay; if it keeps failing, choose another model."
            } else if after(message, "provider returned an unexpected status").is_some() {
                "The model's provider returned an unexpected response. Retry the request; \
                 if it keeps failing, choose another model."
            } else if after(message, "provider reported a transient conflict").is_some() {
                "The model's provider reported a temporary conflict. Retry the request."
            } else {
                return None;
            }
        }
        FailureClass::Transport if after(message, "provider transport failed").is_some() => {
            "The gateway could not reach the model's provider. Retry after a short delay; \
             if it keeps failing, choose another model."
        }
        FailureClass::MalformedResponse
            if after(message, "provider returned a malformed response").is_some() =>
        {
            "The model's provider sent an incomplete or invalid response, so the gateway \
             discarded it. Retry the request; if it keeps happening, choose another model."
        }
        FailureClass::Timeout => {
            if after(message, "provider did not send response headers in time").is_some() {
                "The model's provider did not start responding in time and no other \
                 provider for this model was available. Retry after a short delay; if it \
                 keeps happening, choose another model."
            } else if after(message, "provider did not send the first token in time").is_some() {
                "The model's provider accepted the request but sent no output in time. \
                 Retry the request; a very long prompt takes longer to start, so a \
                 shorter prompt can help."
            } else if after(message, "provider stopped making progress").is_some() {
                "The model's provider stopped sending output partway through the response. \
                 Retry the request."
            } else if after(message, "provider request timed out").is_some() {
                "The model's provider timed out before answering. Retry the request."
            } else {
                return None;
            }
        }
        FailureClass::Throttled => {
            if let Some(rest) = after(
                message,
                "all exact-model deployments are inside a provider throttle window",
            ) {
                return Some(match retry_seconds(rest) {
                    Some(seconds) => format!(
                        "Every provider serving this model is rate limiting requests right \
                         now. Retry in {seconds} seconds."
                    ),
                    None => "Every provider serving this model is rate limiting requests \
                             right now. Retry after a short delay (the Retry-After header \
                             gives the wait when the response carries one)."
                        .to_string(),
                });
            } else if after(message, "provider throttled the request").is_some() {
                "The model's provider is rate limiting requests right now. Retry after a \
                 short delay (the Retry-After header gives the wait when the response \
                 carries one)."
            } else {
                return None;
            }
        }
        FailureClass::ProviderNotFound
            if after(message, "provider deployment was not found").is_some() =>
        {
            "The model's provider no longer serves this model at its configured address. \
             Retry later or choose another model; if it persists, ask the gateway operator \
             to update the model's route."
        }
        FailureClass::Refusal => {
            let rest = after(message, "provider refused the request")?;
            // Only a ":" clause names a refusal category; a ";" clause is
            // remediation, not a reason.
            let named = message.starts_with("provider refused the request:");
            return Some(if rest.is_empty() || !named {
                "The model's provider declined to answer this request. Rephrase the request \
                 or choose another model."
                    .to_string()
            } else {
                format!(
                    "The model's provider declined this request ({rest}). Rephrase the \
                     request or choose another model."
                )
            });
        }
        FailureClass::InvalidRequest
            if message
                == "provider rejected the request; verify the request fields against \
                    the model alias capabilities" =>
        {
            "The model's provider rejected the request without giving a reason. Check the \
             request's parameters against what this model supports (GET /v1/models/{model}), \
             or choose another model."
        }
        FailureClass::Internal
            if after(
                message,
                "the native engine cannot serve the authorized route",
            )
            .is_some() =>
        {
            "The gateway cannot serve this model right now because of a problem on its side. \
             Retry later or choose another model; if it persists, contact the gateway \
             operator."
        }
        _ => return None,
    };
    Some(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vague_sentence_becomes_what_happened_and_what_to_do() {
        let old = "all exact-model deployments are unavailable";
        let text = caller_copy(FailureClass::ProviderInternal, old);
        assert_ne!(text, old);
        assert!(text.starts_with("No provider for this model"));
        for word in [
            "rung",
            "lane",
            "deployment",
            "alias",
            "exact-model",
            "native",
        ] {
            assert!(!text.contains(word), "{word} in {text}");
        }
    }

    #[test]
    fn the_throttle_window_keeps_its_wait() {
        let text = caller_copy(
            FailureClass::Throttled,
            "all exact-model deployments are inside a provider throttle window; retry in 27s",
        );
        assert_eq!(
            text,
            "Every provider serving this model is rate limiting requests right now. \
             Retry in 27 seconds."
        );
    }

    #[test]
    fn a_signed_throttle_wait_is_not_a_wait() {
        let text = caller_copy(
            FailureClass::Throttled,
            "all exact-model deployments are inside a provider throttle window; retry in +27s",
        );
        assert!(text.ends_with("when the response carries one)."));
    }

    #[test]
    fn a_refusal_keeps_its_category() {
        let text = caller_copy(
            FailureClass::Refusal,
            "provider refused the request: content policy",
        );
        assert_eq!(
            text,
            "The model's provider declined this request (content policy). Rephrase the \
             request or choose another model."
        );
    }

    #[test]
    fn a_refusal_remediation_clause_is_not_a_category() {
        let text = caller_copy(
            FailureClass::Refusal,
            "provider refused the request; revise the request content and retry",
        );
        assert_eq!(
            text,
            "The model's provider declined to answer this request. Rephrase the request \
             or choose another model."
        );
    }

    #[test]
    fn a_specific_or_unknown_sentence_passes_through() {
        for (class, message) in [
            (
                FailureClass::InvalidRequest,
                "The parameter 'n' is not supported.",
            ),
            (
                FailureClass::ProviderInternal,
                "provider service failedness",
            ),
            (FailureClass::Timeout, "provider service failed"),
        ] {
            assert_eq!(caller_copy(class, message), message);
        }
    }
}
