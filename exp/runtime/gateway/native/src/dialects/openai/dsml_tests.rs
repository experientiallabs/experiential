//! The DSML guard: ordinary `<` flows promptly, complete DSML (canonical or
//! the malformed spaced spelling) becomes tool calls, and markup that cannot
//! be parsed fails the stream without ever reaching the caller as text.

use serde_json::json;

use super::*;
use crate::dialects::{drain_stream_fixture, Dialect, Normalizer};
use crate::errors::FailureClass;
use crate::events::Event;
use crate::sse::SseEvent;

const CANONICAL: &str = concat!(
    "<｜DSML｜function_calls>\n",
    "<｜DSML｜invoke name=\"shell\">\n",
    "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la < in.txt</｜DSML｜parameter>\n",
    "<｜DSML｜parameter name=\"timeoutMs\" string=\"false\">1260000</｜DSML｜parameter>\n",
    "</｜DSML｜invoke>\n",
    "</｜DSML｜function_calls>",
);

/// The malformed spelling seen in production: a space after the marker and
/// `calls` for `function_calls`, here complete.
const SPACED: &str = concat!(
    "<<｜DSML｜ calls>\n",
    "<｜DSML｜ invoke name=\"shell\">\n",
    "<｜DSML｜ parameter name=\"command\" string=\"true\">ls</｜DSML｜ parameter>\n",
    "<｜DSML｜ parameter name=\"timeoutMs\" string=\"false\">1260000</｜DSML｜ parameter>\n",
    "</｜DSML｜ invoke>\n",
    "</｜DSML｜ calls>",
);

/// The customer's exact tail (request-b8cba0bca30d42e3af1e1145a781a694): no
/// opening, a stray `<`, and a closing parameter tag first.
const CUSTOMER_TAIL: &str = concat!(
    "\n</｜DSML｜ parameter>\n",
    "<｜DSML｜ parameter name=\"timeoutMs\" string=\"false\">1260000</｜DSML｜ parameter>\n",
    "</｜DSML｜ invoke>\n",
    "</｜DSML｜ calls>",
);

/// Push every piece, collecting released text and calls, then finish.
fn run(pieces: &[&str]) -> Result<(String, Vec<DsmlCall>), &'static str> {
    let mut guard = DsmlGuard::default();
    let mut text = String::new();
    let mut calls = Vec::new();
    for piece in pieces {
        let released = guard.push(piece)?;
        text.push_str(&released.text);
        calls.extend(released.calls);
        text.push_str(&released.trailing_text);
    }
    let released = guard.finish()?;
    text.push_str(&released.text);
    calls.extend(released.calls);
    Ok((text, calls))
}

fn shell_call(command: &str) -> DsmlCall {
    let mut arguments = Map::new();
    arguments.insert("command".into(), json!(command));
    arguments.insert("timeoutMs".into(), json!(1_260_000));
    DsmlCall {
        name: "shell".into(),
        arguments,
    }
}

#[test]
fn ordinary_angle_brackets_pass_through_promptly() {
    let mut guard = DsmlGuard::default();
    // Code and HTML are released inside the same delta.
    assert_eq!(guard.push("if a < b {").unwrap().text, "if a < b {");
    assert_eq!(guard.push("<div><p>").unwrap().text, "<div><p>");
    assert_eq!(guard.push("x <= y << 2").unwrap().text, "x <= y << 2");
    assert!(!guard.holding());
    // A `<` at a delta boundary waits for exactly the next character.
    assert_eq!(guard.push("vec<").unwrap().text, "vec");
    assert!(guard.holding());
    assert_eq!(guard.push("u8>").unwrap().text, "<u8>");
    // A full-width bar that is not the marker is ordinary text too.
    assert_eq!(guard.push("<｜note｜>").unwrap().text, "<｜note｜>");
    // A trailing `<` at the end of the stream is released, not dropped.
    assert_eq!(guard.push("a <").unwrap().text, "a ");
    assert_eq!(guard.finish().unwrap().text, "<");
}

#[test]
fn canonical_and_spaced_dsml_become_tool_calls() {
    let (text, calls) = run(&["Running it now.\n\n", CANONICAL]).unwrap();
    assert_eq!(text, "Running it now.\n\n");
    assert_eq!(calls, vec![shell_call("ls -la < in.txt")]);
    let (text, calls) = run(&["Again.", SPACED]).unwrap();
    assert_eq!(text, "Again.");
    assert_eq!(calls, vec![shell_call("ls")]);
}

#[test]
fn a_marker_split_across_deltas_is_still_held_and_parsed() {
    let mut pieces: Vec<String> = vec!["Checking <".into()];
    // One character per delta through the whole block.
    pieces.extend(CANONICAL.trim_start_matches('<').chars().map(String::from));
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let (text, calls) = run(&refs).unwrap();
    assert_eq!(text, "Checking ");
    assert_eq!(calls, vec![shell_call("ls -la < in.txt")]);
}

#[test]
fn text_after_a_closed_block_is_ordinary_content_again() {
    let mut guard = DsmlGuard::default();
    let released = guard.push(&format!("{CANONICAL}\nDone.")).unwrap();
    assert_eq!(released.calls.len(), 1);
    assert_eq!(released.trailing_text, "\nDone.");
    assert!(!guard.holding());
}

#[test]
fn the_customers_tail_only_fragment_is_a_malformed_tool_call() {
    let result = run(&[
        "Waiting twenty more to the post-apply boot.\n\n<",
        CUSTOMER_TAIL,
    ]);
    assert_eq!(result, Err(MALFORMED_TOOL_CALL));
    // The doubled `<` the capture also showed.
    let result = run(&["boot.\n\n<<", CUSTOMER_TAIL]);
    assert_eq!(result, Err(MALFORMED_TOOL_CALL));
}

#[test]
fn incomplete_or_mistyped_blocks_never_parse() {
    // No closing outer tag before the stream ended, and no invoke closing.
    let open = "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"shell\">\n<｜DSML｜parameter";
    assert_eq!(run(&[open]), Err(MALFORMED_TOOL_CALL));
    // An invoke with no name.
    let nameless =
        "<｜DSML｜function_calls><｜DSML｜invoke></｜DSML｜invoke></｜DSML｜function_calls>";
    assert_eq!(run(&[nameless]), Err(MALFORMED_TOOL_CALL));
    // A `string="false"` value that is not JSON.
    let mistyped = concat!(
        "<｜DSML｜function_calls><｜DSML｜invoke name=\"f\">",
        "<｜DSML｜parameter name=\"n\" string=\"false\">not json</｜DSML｜parameter>",
        "</｜DSML｜invoke></｜DSML｜function_calls>",
    );
    assert_eq!(run(&[mistyped]), Err(MALFORMED_TOOL_CALL));
}

fn content_frame(text: &str) -> SseEvent {
    SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {"content": text}}]}).to_string(),
    }
}

fn finish_frame(reason: &str) -> SseEvent {
    SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}).to_string(),
    }
}

fn done() -> SseEvent {
    SseEvent {
        event: None,
        data: "[DONE]".into(),
    }
}

fn relayed_text(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::TextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn the_normalizer_turns_streamed_dsml_into_a_completed_tool_call() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let mut events = Vec::new();
    for piece in [
        "On it.\n\n<",
        "｜DSML｜ calls>",
        SPACED.split_once("calls>").unwrap().1,
    ] {
        events.extend(normalizer.feed(&content_frame(piece)).unwrap());
    }
    events.extend(normalizer.feed(&finish_frame("stop")).unwrap());
    events.extend(normalizer.feed(&done()).unwrap());
    assert_eq!(relayed_text(&events), "On it.\n\n");
    assert!(events.iter().any(|event| matches!(
        event,
        Event::ToolCallStarted { name, .. } if name == "shell"
    )));
    let arguments: String = events
        .iter()
        .filter_map(|event| match event {
            Event::ToolArgumentsDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
        json!({"command": "ls", "timeoutMs": 1_260_000})
    );
    assert!(events.iter().any(Event::is_terminal));
}

#[test]
fn the_normalizer_fails_the_customers_fragment_without_relaying_markup() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let mut events = Vec::new();
    let mut failure = None;
    for piece in [
        "Waiting twenty more to the post-apply boot.\n\n<",
        CUSTOMER_TAIL,
    ] {
        match normalizer.feed(&content_frame(piece)) {
            Ok(more) => events.extend(more),
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    let failure = failure.expect("the closed malformed block fails the stream");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    assert_eq!(failure.safe_message, MALFORMED_TOOL_CALL);
    assert!(failure.failover_eligible);
    let relayed = relayed_text(&events);
    assert_eq!(relayed, "Waiting twenty more to the post-apply boot.\n\n");
    assert!(!relayed.contains('｜') && !relayed.contains('<'));
}

#[test]
fn ordinary_content_is_relayed_unchanged_across_frames() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let first = normalizer.feed(&content_frame("fn f() { a <")).unwrap();
    assert_eq!(relayed_text(&first), "fn f() { a ");
    let second = normalizer.feed(&content_frame(" b }")).unwrap();
    assert_eq!(relayed_text(&second), "< b }");
    let mut events = normalizer.feed(&content_frame("<")).unwrap();
    events.extend(normalizer.feed(&finish_frame("stop")).unwrap());
    events.extend(normalizer.feed(&done()).unwrap());
    // The held `<` at the end is released before the terminal.
    assert_eq!(relayed_text(&events), "<");
    assert!(matches!(events.last(), Some(Event::Completed)));
}

#[test]
fn a_whole_body_drain_never_carries_markup_either() {
    // Non-streaming callers are answered from the same normalized events
    // (the native plane always streams upstream and aggregates), so the
    // drain of a full provider body is the non-streaming shape.
    let body = |content: &str| {
        format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"choices": [{"index": 0, "delta": {"content": content}}]}),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        )
        .into_bytes()
    };
    let (events, failure) = drain_stream_fixture(
        Dialect::OpenAiCompatible,
        &[body(&format!("Sure.{CANONICAL}"))],
    );
    assert!(failure.is_none());
    let text: String = events
        .iter()
        .filter(|event| event["kind"] == "text_delta")
        .filter_map(|event| event["text"].as_str())
        .collect();
    assert_eq!(text, "Sure.");
    let (events, failure) = drain_stream_fixture(
        Dialect::OpenAiCompatible,
        &[body(&format!("boot.\n\n<{CUSTOMER_TAIL}"))],
    );
    let failure = failure.expect("the malformed fragment fails the drain");
    assert_eq!(failure.safe_message, MALFORMED_TOOL_CALL);
    assert!(events
        .iter()
        .filter_map(|event| event["text"].as_str())
        .all(|text| !text.contains('｜')));
}
