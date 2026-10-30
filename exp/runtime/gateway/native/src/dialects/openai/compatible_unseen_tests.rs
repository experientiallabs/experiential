//! A natural `stop` whose usage reports answer tokens the caller never saw.

use super::*;

// A natural `stop` whose usage reports answer tokens that never reached the
// caller as content or a tool call is the upstream dropping its own output
// (deepseek-v4.1-flash on Novita, 2026-10-27: reasoning streamed, then the
// post-reasoning tokens vanished). It settles Incomplete, not a silent empty
// Completed.
#[test]
fn stop_reporting_unseen_answer_tokens_settles_incomplete() {
    let reasoning = SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}]})
            .to_string(),
    };
    let finish = SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).to_string(),
    };
    let usage = SseEvent {
        event: None,
        // 434 completion tokens, 167 reasoning: 267 answer tokens, none surfaced.
        data: json!({"choices": [], "usage": {"prompt_tokens": 47950, "completion_tokens": 434, "reasoning_tokens": 167}})
            .to_string(),
    };
    let done = SseEvent {
        event: None,
        data: "[DONE]".into(),
    };
    let mut n = Normalizer::new(Dialect::OpenAiCompatible);
    let mut ev = Vec::new();
    for f in [reasoning, finish, usage, done] {
        ev.extend(n.feed(&f).unwrap());
    }
    assert!(ev.iter().any(|e| matches!(e, Event::Incomplete)), "{ev:?}");
    assert!(!ev.iter().any(|e| matches!(e, Event::Completed)), "{ev:?}");
}

// A real answer still completes: reasoning then content, same usage shape.
#[test]
fn stop_with_surfaced_answer_still_completes() {
    let reasoning = SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]})
            .to_string(),
    };
    let content = SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {"content": "Here is the answer."}}]})
            .to_string(),
    };
    let finish = SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).to_string(),
    };
    let usage = SseEvent {
        event: None,
        data: json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 434, "reasoning_tokens": 167}})
            .to_string(),
    };
    let done = SseEvent {
        event: None,
        data: "[DONE]".into(),
    };
    let mut n = Normalizer::new(Dialect::OpenAiCompatible);
    let mut ev = Vec::new();
    for f in [reasoning, content, finish, usage, done] {
        ev.extend(n.feed(&f).unwrap());
    }
    assert!(ev.iter().any(|e| matches!(e, Event::Completed)), "{ev:?}");
    assert!(!ev.iter().any(|e| matches!(e, Event::Incomplete)), "{ev:?}");
}

// A reasoning-only natural stop (no answer tokens reported) stays Completed.
#[test]
fn reasoning_only_stop_still_completes() {
    let reasoning = SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {"reasoning_content": "done"}}]})
            .to_string(),
    };
    let finish = SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).to_string(),
    };
    let usage = SseEvent {
        event: None,
        data: json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 167, "reasoning_tokens": 167}})
            .to_string(),
    };
    let done = SseEvent {
        event: None,
        data: "[DONE]".into(),
    };
    let mut n = Normalizer::new(Dialect::OpenAiCompatible);
    let mut ev = Vec::new();
    for f in [reasoning, finish, usage, done] {
        ev.extend(n.feed(&f).unwrap());
    }
    assert!(ev.iter().any(|e| matches!(e, Event::Completed)), "{ev:?}");
    assert!(!ev.iter().any(|e| matches!(e, Event::Incomplete)), "{ev:?}");
}
