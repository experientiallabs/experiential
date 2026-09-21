//! Envelope-spelling tests for the OpenAI-family attribution readers: the
//! live-captured bodies (2026-09-15) whose 400s settled with no detail until
//! `crate::error_envelope` read them.

use super::*;

/// Exact bodies captured live on 2026-09-15 from the lanes whose client
/// errors settled with no detail: xAI (`{code, error: "<sentence>"}`),
/// Novita (`{code, reason, message, metadata}`), and an Azure Foundry
/// DeepSeek deployment (vLLM's flat object followed by a help line).
const XAI_BODY: &str = r#"{"code":"invalid-argument","error":"Incorrect API key provided. You can obtain an API key from https://console.x.ai."}"#;
const NOVITA_BODY: &str = r#"{"code":400,"reason":"INVALID_PARAMETER","message":"tools is not supported by this model","metadata":{}}"#;
const FOUNDRY_VLLM_BODY: &str = "{\"object\":\"error\",\"message\":\"Tool 'g' not found in tools list.\",\"type\":\"BadRequestError\",\"param\":null,\"code\":400}\nPlease check this guide to understand why this error code might have been returned \nhttps://docs.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints#http-status-codes\n";

#[test]
fn every_openai_family_envelope_spelling_yields_its_sentence_and_token() {
    assert_eq!(
        rejected_detail(Dialect::OpenAiCompatible, XAI_BODY, &[]).as_deref(),
        // The console URL is infrastructure and is masked; the sentence survives.
        Some("Incorrect API key provided. You can obtain an API key from [redacted].")
    );
    assert_eq!(
        rejected_code(Dialect::OpenAiCompatible, XAI_BODY).as_deref(),
        Some("invalid-argument")
    );
    assert_eq!(
        rejected_detail(Dialect::OpenAiCompatible, NOVITA_BODY, &[]).as_deref(),
        Some("tools is not supported by this model")
    );
    assert_eq!(
        rejected_code(Dialect::OpenAiCompatible, NOVITA_BODY).as_deref(),
        Some("INVALID_PARAMETER")
    );
    assert_eq!(
        rejected_detail(Dialect::OpenAiCompatible, FOUNDRY_VLLM_BODY, &[]).as_deref(),
        Some("Tool 'g' not found in tools list.")
    );
    // A bare numeric status is a token the caller never sees as detail.
    assert!(generic_error_code(
        &rejected_code(Dialect::OpenAiCompatible, FOUNDRY_VLLM_BODY).expect("token")
    ));
    // The Responses dialect shares the family reader.
    assert_eq!(
        rejected_detail(Dialect::OpenAiResponses, NOVITA_BODY, &[]).as_deref(),
        Some("tools is not supported by this model")
    );
}

#[test]
fn fastapi_detail_yields_its_sentence_path_and_token() {
    let body = r#"{"detail": [{"loc": ["body", "messages", 0, "content"],
        "msg": "an image part is required", "type": "value_error"}]}"#;
    assert_eq!(
        rejected_detail(Dialect::OpenAiCompatible, body, &[]).as_deref(),
        Some("an image part is required")
    );
    assert_eq!(
        rejected_parameter(Dialect::OpenAiCompatible, body).as_deref(),
        Some("messages[0].content")
    );
    assert_eq!(
        rejected_code(Dialect::OpenAiCompatible, body).as_deref(),
        Some("value_error")
    );
    let plain = r#"{"detail": "An image input is required for this model."}"#;
    assert_eq!(
        rejected_detail(Dialect::OpenAiCompatible, plain, &[]).as_deref(),
        Some("An image input is required for this model.")
    );
}

#[test]
fn trailing_text_after_the_json_document_no_longer_drops_the_body() {
    // The same document without the help line reads identically, and a
    // body that is not JSON at all still yields nothing.
    let clean = FOUNDRY_VLLM_BODY.split('\n').next().expect("document");
    assert_eq!(
        rejected_detail(Dialect::OpenAiCompatible, clean, &[]),
        rejected_detail(Dialect::OpenAiCompatible, FOUNDRY_VLLM_BODY, &[])
    );
    assert_eq!(
        rejected_detail(Dialect::OpenAiCompatible, "<html>", &[]),
        None
    );
    assert_eq!(
        rejected_code(Dialect::OpenAiCompatible, "upstream said no"),
        None
    );
    // A flat envelope reporting the model-not-found code takes the lane policy.
    let flat_missing = r#"{"object":"error","message":"The model does not exist.","type":"NotFoundError","code":"model_not_found"}"#;
    assert!(rejected_model_not_found(
        Dialect::OpenAiCompatible,
        flat_missing
    ));
    assert!(!rejected_model_not_found(
        Dialect::OpenAiCompatible,
        NOVITA_BODY
    ));
}

#[test]
fn other_dialects_keep_reading_only_their_documented_message_field() {
    // A flat `message` is Bedrock's shape, not Anthropic's or Gemini's.
    let flat = r#"{"message": "The provided model does not support tool use."}"#;
    assert_eq!(rejected_detail(Dialect::AnthropicMessages, flat, &[]), None);
    assert_eq!(
        rejected_detail(Dialect::GeminiGenerateContent, flat, &[]),
        None
    );
    assert_eq!(
        rejected_detail(Dialect::BedrockConverseStream, flat, &[]).as_deref(),
        Some("The provided model does not support tool use.")
    );
    assert_eq!(
        rejected_detail(Dialect::AnthropicMessages, XAI_BODY, &[]),
        None
    );
}

#[test]
fn novita_reason_tokens_classify_without_being_relayed_as_detail() {
    // The documented 400 reason says nothing beyond "rejected", so it is
    // classified but never relayed on its own; the flat MODEL_NOT_FOUND
    // reason is the catalog's fault (lane policy), case-insensitively.
    assert!(generic_error_code("INVALID_REQUEST_BODY"));
    let missing =
        r#"{"code":404,"reason":"MODEL_NOT_FOUND","message":"Model not found","metadata":{}}"#;
    assert!(rejected_model_not_found(Dialect::OpenAiCompatible, missing));
    assert!(!rejected_model_not_found(
        Dialect::AnthropicMessages,
        missing
    ));
    assert!(!rejected_model_not_found(
        Dialect::OpenAiCompatible,
        NOVITA_BODY
    ));
}
