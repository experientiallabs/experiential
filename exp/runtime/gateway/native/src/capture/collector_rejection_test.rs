//! A blocked input permanently closes every provider-output channel.

use std::sync::{atomic::Ordering, Arc};

use serde_json::json;

use super::{collector, config, drain, request, response};
use crate::capture::relay::{Metadata, Relay};
use crate::events::CompletedToolCall;

fn call() -> CompletedToolCall {
    CompletedToolCall {
        call_id: "withheld-call".into(),
        name: "withheld-tool".into(),
        namespace: None,
        caller: None,
        provider_item_id: None,
        provider_status: None,
        raw_arguments: "withheld-arguments".into(),
        custom: true,
        thought_signature: None,
    }
}

#[test]
fn rejected_input_stays_input_only_despite_late_output_and_host_response_permission() {
    for attached_before_denial in [false, true] {
        for settled_before_denial in [false, true] {
            let mut settings = config();
            settings.capture_logprobs = true;
            settings.relay_metadata = true;
            let (collector, receiver) = collector(settings);
            assert!(collector.begin(request("rejected")));
            assert!(collector.claim_relay("rejected"));
            let handle = attached_before_denial.then(|| collector.attach("rejected").unwrap());
            assert!(collector.wants_logprobs("rejected"));
            collector.logprobs_injected("rejected");
            collector.reasoning("rejected", "withheld-before-denial");
            collector.tool_call("rejected", &call());
            if settled_before_denial {
                collector.settle("rejected", true, true);
            }
            collector.finish_unexposed("rejected");
            if let Some(handle) = handle {
                assert!(handle.load(Ordering::Acquire));
            }
            assert!(collector.attach("rejected").is_none());
            assert!(!collector.claim_relay("rejected"));
            assert!(!collector.wants_logprobs("rejected"));
            collector.logprobs_injected("rejected");
            let probability = crate::logprobs::parse(
                Some(&json!({"content":[{
                    "token":"withheld-token","logprob":-1.0,"bytes":null,"top_logprobs":[]
                }]})),
                0,
            )
            .unwrap()
            .unwrap();
            collector.logprobs("rejected", vec![probability], false);
            collector.reasoning("rejected", "withheld-late-reasoning");
            collector.tool_call("rejected", &call());
            collector.gemini_thought_part("rejected", Arc::new(json!({"text":"withheld-thought"})));
            assert!(!collector.finish("rejected", Some(response()), Some("deployment".into())));
            assert!(!collector.finish_relay(
                "rejected",
                Relay {
                    metadata: Metadata {
                        wire_request: None,
                        headers: vec![],
                        timing: json!({}),
                        relay_completed: true,
                        client_disconnected: false,
                    },
                    body: b"withheld-relay".to_vec(),
                }
            ));
            if !settled_before_denial {
                assert!(
                    receiver.try_recv().is_err(),
                    "host retention is still required"
                );
                collector.settle("rejected", true, true);
            }
            let records = drain(&collector, receiver);
            assert_eq!(records.len(), 1);
            let record = &records[0];
            assert!(record.response.is_none());
            assert!(record.provider_reasoning.is_none());
            assert!(record.provider_tool_calls_json.is_none());
            assert!(record.provider_logprobs.is_none());
            assert!(record.metrics.is_none());
            assert!(record.gemini_thought_parts.is_empty());
            assert!(!record.encode(8192).unwrap().contains("withheld"));
        }
    }
}
