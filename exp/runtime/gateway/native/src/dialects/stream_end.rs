//! A provider stream that closes cleanly WITHOUT a terminal frame after it
//! already produced output. Before any output the relay reports
//! `ended_without_terminal` (failover-eligible, nothing to preserve); after
//! output the served tokens are real and the close is the provider's cut:
//! Gemini legitimately ends turns this way and completes, an OpenAI-compatible
//! relay that declared its finish and only dropped `[DONE]` settles by that
//! finish, and every other wire settles `Incomplete` with any mid-fragment
//! call dropped (gpt-5.6-luna on OpenAI's Responses wire, 33 of 34
//! post-commit streams in 14 days, 2026-09-14). Child of `dialects` so the
//! parent stays under the hand-authored line budget.

use super::{finish_open_tools_relay, Dialect, Normalizer, OUTPUT_OVERFLOW_MESSAGE};
use crate::errors::{Failure, FailureClass};
use crate::events::{Event, ProviderOutputItemStatus};

impl Normalizer {
    /// Recover a Gemini abnormal end without discarding served output or meters.
    /// A declared finish remains authoritative after a trailer failure; before
    /// finish, served content is incomplete and pre-content breaks can retry.
    /// Other dialects and deliberate content-validation limits remain errors.
    pub fn recover_abnormal_end(&mut self, failure: Failure) -> Result<Vec<Event>, Failure> {
        if self.metadata_drain_started().is_some() {
            return Ok(self.finish_metadata_drain());
        }
        if failure.safe_message == OUTPUT_OVERFLOW_MESSAGE
            || (failure.failure_class == FailureClass::MalformedResponse
                && !failure.retryable_same_deployment
                && !failure.failover_eligible)
            || self.terminal
            || self.dialect != Dialect::GeminiGenerateContent
        {
            return Err(failure);
        }
        if !self.emitted_output {
            return Err(Failure::new(
                FailureClass::Transport,
                "provider transport failed; retry the request",
            )
            .with_retry(true, true)
            .with_provider_detail(failure.provider_detail));
        }
        let mut events = Vec::new();
        if let Some(usage) = self.usage.take() {
            events.push(Event::Usage(usage));
        }
        events.push(Event::Incomplete);
        self.terminal = true;
        Ok(events)
    }

    /// Synthesize the terminal events for a stream that closed cleanly
    /// without an explicit terminal frame, or nothing when a terminal already
    /// ended the stream or nothing was served (the caller then fails it
    /// closed as terminal-less). Errors only when finishing the open tool
    /// calls fails: a syntactically invalid streamed argument object stays
    /// malformed.
    pub fn on_stream_end(&mut self) -> Result<Vec<Event>, Failure> {
        if self.terminal {
            return Ok(Vec::new());
        }
        if self.metadata_drain_started().is_some() {
            return Ok(self.finish_metadata_drain());
        }
        // An OpenAI-compatible finish reason already seen is a complete
        // ending whether or not output followed it (a content_filter finish
        // with nothing served is the declared refusal, Azure Foundry DeepSeek
        // 2026-09-15); it settles exactly as `[DONE]` would.
        if self.dialect == Dialect::OpenAiCompatible && self.finish_reason.is_some() {
            self.log_stream_end("declared_finish");
            let events = self.openai_compatible_stream_end()?;
            return self.end_with(events);
        }
        if !self.emitted_output {
            return Ok(Vec::new());
        }
        let events = match self.dialect {
            // Gemini ends some streams right after its last content frame
            // without a `finishReason` frame: a complete answer.
            Dialect::GeminiGenerateContent => {
                let mut events = Vec::new();
                if let Some(usage) = self.usage.take() {
                    events.push(Event::Usage(usage));
                }
                events.push(Event::Completed);
                events
            }
            _ => {
                self.log_stream_end("incomplete");
                let mut events = Vec::new();
                if self.dialect == Dialect::OpenAiResponses {
                    events.extend(
                        self.openai_close_unfinished_items(ProviderOutputItemStatus::Incomplete),
                    );
                }
                // A stopped block plus an explicit normal final reason is
                // sufficient evidence for a zero-argument call, even if the
                // final trailer is missing. The whole turn remains incomplete.
                let normal_stop = !self.refusal_seen
                    && matches!(
                        self.stop_reason.as_deref(),
                        Some("end_turn" | "stop_sequence" | "tool_use" | "pause_turn")
                    );
                for (index, tool) in &mut self.tools {
                    if !tool.custom && tool.raw_arguments.is_empty() {
                        let stopped = match self.dialect {
                            Dialect::AnthropicMessages => {
                                self.anthropic_stopped_tools.contains(index)
                            }
                            Dialect::BedrockConverseStream => {
                                self.bedrock_empty_stopped_tools.contains(index)
                            }
                            _ => false,
                        };
                        if !(normal_stop && stopped) {
                            tool.completed = true;
                        }
                    }
                }
                let (tool_events, _dropped) =
                    finish_open_tools_relay(&mut self.tools, "stream_end")?;
                events.extend(tool_events);
                if let Some(usage) = self.usage.take() {
                    events.push(Event::Usage(usage));
                }
                events.push(Event::Incomplete);
                events
            }
        };
        self.end_with(events)
    }

    fn end_with(&mut self, events: Vec<Event>) -> Result<Vec<Event>, Failure> {
        if events.iter().any(Event::is_terminal) {
            self.terminal = true;
        }
        Ok(events)
    }

    fn log_stream_end(&self, verdict: &str) {
        let line = serde_json::json!({
            "event": "stream_ended_without_terminal_after_output",
            "dialect": dialect_name(self.dialect),
            "open_tools": self.tools.values().filter(|tool| !tool.completed).count(),
            "verdict": verdict,
        });
        eprintln!("exp-gateway-native: {line}");
    }
}

fn dialect_name(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::OpenAiResponses => "openai_responses",
        Dialect::AnthropicMessages => "anthropic_messages",
        Dialect::OpenAiCompatible => "openai_compatible",
        Dialect::GeminiGenerateContent => "gemini_generate_content",
        Dialect::BedrockConverseStream => "bedrock_converse_stream",
        Dialect::TypesafeSystemone => "typesafe_systemone",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ToolAccumulator;

    #[test]
    fn eof_after_a_normal_reason_keeps_stopped_empty_tools_but_stays_incomplete() {
        for dialect in [Dialect::AnthropicMessages, Dialect::BedrockConverseStream] {
            let mut normalizer = Normalizer::new(dialect);
            let frames = match dialect {
                Dialect::AnthropicMessages => vec![
                    (
                        None,
                        serde_json::json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"lookup","input":{}}}),
                    ),
                    (
                        None,
                        serde_json::json!({"type":"content_block_stop","index":0}),
                    ),
                    (
                        None,
                        serde_json::json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":0}}),
                    ),
                ],
                _ => vec![
                    (
                        Some("contentBlockStart"),
                        serde_json::json!({"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_1","name":"lookup"}}}),
                    ),
                    (
                        Some("contentBlockStop"),
                        serde_json::json!({"contentBlockIndex":0}),
                    ),
                    (
                        Some("messageStop"),
                        serde_json::json!({"stopReason":"tool_use"}),
                    ),
                ],
            };
            for (event, payload) in frames {
                normalizer
                    .feed(&crate::sse::SseEvent {
                        event: event.map(str::to_string),
                        data: payload.to_string(),
                    })
                    .unwrap();
            }
            let events = normalizer.on_stream_end().unwrap();
            assert!(events.iter().any(|event| matches!(event, Event::ToolCallCompleted { call, .. } if call.raw_arguments == "{}")), "{dialect:?}: {events:?}");
            assert!(matches!(events.last(), Some(Event::Incomplete)));
        }
    }

    #[test]
    fn eof_without_a_final_reason_never_seeds_missing_tool_arguments() {
        for dialect in [
            Dialect::AnthropicMessages,
            Dialect::BedrockConverseStream,
            Dialect::OpenAiCompatible,
            Dialect::OpenAiResponses,
        ] {
            for arguments in ["", "{}", "{\"city\":\"Paris\"}"] {
                let mut normalizer = Normalizer::new(dialect);
                // Seed the state reached after a tool start and its deltas;
                // every dialect shares this EOF path, not a provider stop.
                normalizer.emitted_output = true;
                let mut tool = ToolAccumulator::new("call_1".into(), "lookup".into());
                tool.raw_arguments = arguments.into();
                normalizer.tools.insert(0, tool);
                let events = normalizer.on_stream_end().unwrap();
                assert!(matches!(events.last(), Some(Event::Incomplete)));
                assert!(!events
                    .iter()
                    .any(|event| matches!(event, Event::ToolArgumentsDelta { .. })));
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| matches!(event, Event::ToolCallCompleted { .. }))
                        .count(),
                    usize::from(!arguments.is_empty())
                );
            }
        }
    }
}
