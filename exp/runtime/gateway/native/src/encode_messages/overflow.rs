//! The Anthropic wording of a context-window overflow on the Messages surface.
//!
//! Claude Code compacts a conversation on its own only when the API answers
//! the way Anthropic does: a 400 `invalid_request_error` whose message starts
//! "prompt is too long: N tokens > M maximum" (and, when the prompt fits but
//! the output budget does not, "input length and `max_tokens` exceed context
//! limit: N + R > M"). A provider's overflow relayed as "provider rejected the
//! request: This model's maximum context length is ..." never triggered it,
//! so a session that crossed the window stayed stuck (deepseek-v4.1-flash
//! uncensored, 2026-11-18: a 1,049,847-token prompt against a 1,048,576
//! window). Every context overflow carries the public code
//! `context_length_exceeded` (OpenAI's own), whichever side refused it; this
//! module renders that code in Anthropic's words, keeping the gateway's own
//! sentence after it.

use crate::errors::PublicError;

/// The public code every context-window overflow carries.
pub const CONTEXT_LENGTH_EXCEEDED_CODE: &str = "context_length_exceeded";

/// Phrases that precede the window a sentence states.
const LIMIT_PREFIXES: &[&str] = &[
    "maximum context length is ",
    "maximum context length of ",
    "context window on this model route is ",
    "maximum context window of ",
];

/// Phrases that precede the prompt's own token count.
const PROMPT_PREFIXES: &[&str] = &[
    "prompt contains ",
    "prompt is at least ",
    "messages resulted in ",
    "prompt has ",
];

/// Phrases that precede the output budget the request reserves.
const RESERVE_PREFIXES: &[&str] = &["the request reserves "];

/// The integer right after the first of `prefixes` found in `message`
/// (digits and thousands separators only).
fn number_after(message: &str, prefixes: &[&str]) -> Option<u64> {
    prefixes.iter().find_map(|prefix| {
        let start = message.find(prefix)? + prefix.len();
        let digits: String = message[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ',')
            .filter(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    })
}

/// The integer right before `suffix` in `message` (vLLM's "you requested N
/// tokens (X in the messages, Y in the completion)").
fn number_before(message: &str, suffix: &str) -> Option<u64> {
    let end = message.find(suffix)?;
    let digits: String = message[..end]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == ',')
        .filter(char::is_ascii_digit)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse().ok()
}

/// The Anthropic sentence for one overflow, when `error` is one; `None` for
/// every other error. Counts the message states are restated in Anthropic's
/// shape; without them the message still opens with "prompt is too long".
pub fn anthropic_overflow_message(error: &PublicError) -> Option<String> {
    if error.status_code != 400 || error.code != CONTEXT_LENGTH_EXCEEDED_CODE {
        return None;
    }
    let message = error.message.as_str();
    let lowered = message.to_ascii_lowercase();
    if lowered.starts_with("prompt is too long") || lowered.starts_with("input length and") {
        return Some(message.to_string());
    }
    let limit = number_after(&lowered, LIMIT_PREFIXES);
    let prompt = number_after(&lowered, PROMPT_PREFIXES)
        .or_else(|| number_before(&lowered, " in the messages"));
    let reserve = number_after(&lowered, RESERVE_PREFIXES)
        .or_else(|| number_before(&lowered, " in the completion"));
    let head = match (prompt, limit, reserve) {
        (Some(prompt), Some(limit), Some(reserve)) if prompt <= limit => format!(
            "input length and `max_tokens` exceed context limit: {prompt} + {reserve} > {limit}, decrease input length or `max_tokens` and try again"
        ),
        (Some(prompt), Some(limit), _) => {
            format!("prompt is too long: {prompt} tokens > {limit} maximum")
        }
        _ => "prompt is too long".to_string(),
    };
    Some(format!("{head}. {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overflow(message: &str) -> PublicError {
        PublicError::new(
            400,
            CONTEXT_LENGTH_EXCEEDED_CODE,
            message,
            "invalid_request_error",
        )
    }

    #[test]
    fn a_provider_overflow_reads_as_anthropics_prompt_is_too_long() {
        // The vLLM sentence production relayed on 2026-11-18.
        let error = overflow(
            "provider rejected the request: This model's maximum context length is 1048576 tokens. However, your prompt contains 1190669 input tokens, which exceeds this limit even without any output tokens. Please reduce the length of the input prompt.",
        );
        let message = anthropic_overflow_message(&error).unwrap();
        assert!(message.starts_with("prompt is too long: 1190669 tokens > 1048576 maximum. "));
        assert!(message.ends_with("Please reduce the length of the input prompt."));
        // OpenAI's own wording.
        let openai = overflow(
            "provider rejected the request: This model's maximum context length is 128,000 tokens. However, your messages resulted in 130,512 tokens.",
        );
        assert!(anthropic_overflow_message(&openai)
            .unwrap()
            .starts_with("prompt is too long: 130512 tokens > 128000 maximum."));
    }

    #[test]
    fn the_gateways_own_estimate_reads_the_same_way() {
        let prompt_only = overflow(
            "The prompt is at least 1,049,847 tokens (4,199,388 bytes of text), but the largest context window on this model route is 1,048,576 tokens. Shorten the conversation or choose a model with a larger context window.",
        );
        assert!(anthropic_overflow_message(&prompt_only)
            .unwrap()
            .starts_with(
                "prompt is too long: 1049847 tokens > 1048576 maximum. The prompt is at least"
            ));
        let with_output = overflow(
            "The prompt is at least 1,000,000 tokens (4,000,000 bytes of text) and the request reserves 64,000 output tokens, but the largest context window on this model route is 1,048,576 tokens. Lower max_tokens to at most 48,576, shorten the conversation, or choose a model with a larger context window.",
        );
        assert!(anthropic_overflow_message(&with_output).unwrap().starts_with(
            "input length and `max_tokens` exceed context limit: 1000000 + 64000 > 1048576, decrease input length or `max_tokens` and try again. "
        ));
        // vLLM's combined wording names both parts.
        let vllm = overflow(
            "provider rejected the request: This model's maximum context length is 32768 tokens. However, you requested 33000 tokens (30000 in the messages, 3000 in the completion).",
        );
        assert!(anthropic_overflow_message(&vllm).unwrap().starts_with(
            "input length and `max_tokens` exceed context limit: 30000 + 3000 > 32768"
        ));
    }

    #[test]
    fn an_overflow_without_counts_still_opens_with_the_phrase_and_nothing_else_changes() {
        let bare = overflow("provider rejected the request: context length exceeded");
        assert_eq!(
            anthropic_overflow_message(&bare).unwrap(),
            "prompt is too long. provider rejected the request: context length exceeded"
        );
        // The Anthropic envelope carries it, the param pointer folded after.
        let mut with_param = overflow(
            "This model's maximum context length is 10 tokens. However, your prompt contains 12 input tokens.",
        );
        with_param.param = Some("messages".into());
        let body = crate::encode_messages::anthropic_error_body(&with_param);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let text = body["error"]["message"].as_str().unwrap();
        assert!(text.starts_with("prompt is too long: 12 tokens > 10 maximum. "));
        assert!(text.ends_with(" (param: messages)"));
        let anthropic = overflow("prompt is too long: 210000 tokens > 200000 maximum");
        assert_eq!(
            anthropic_overflow_message(&anthropic).unwrap(),
            "prompt is too long: 210000 tokens > 200000 maximum"
        );
        let other = PublicError::new(
            400,
            "invalid_request",
            "maximum context length is 5",
            "invalid_request_error",
        );
        assert!(anthropic_overflow_message(&other).is_none());
    }

    #[test]
    fn a_context_window_overflow_carries_openais_code() {
        let refused = crate::context_window::mark(
            crate::errors::Failure::new(
                crate::errors::FailureClass::InvalidRequest,
                "provider rejected the request",
            )
            .with_provider_detail(Some("maximum context length is 8 tokens".into())),
            Some(8),
        );
        assert_eq!(refused.public_error().code, "context_length_exceeded");
        assert_eq!(refused.public_error().status_code, 400);
        let plain = crate::errors::Failure::new(
            crate::errors::FailureClass::InvalidRequest,
            "provider rejected the request",
        );
        assert_eq!(plain.public_error().code, "invalid_request");
    }
}
