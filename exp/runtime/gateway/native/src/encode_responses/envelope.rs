//! Request-reflecting fields embedded in every public Responses envelope.

use serde::Deserialize;
use serde_json::{json, Value};

fn default_true() -> bool {
    true
}

fn default_tool_choice() -> Value {
    Value::String("auto".to_string())
}

fn default_tools() -> Value {
    Value::Array(Vec::new())
}

fn default_reasoning() -> Value {
    json!({"effort": Value::Null, "summary": Value::Null})
}

/// Request-reflecting envelope fields built from the canonical request.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesEnvelope {
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub temperature: Value,
    #[serde(default)]
    pub top_p: Value,
    #[serde(default = "default_reasoning")]
    pub reasoning: Value,
    #[serde(default)]
    pub ignored_parameters: Vec<String>,
    #[serde(default = "default_tool_choice")]
    pub tool_choice: Value,
    #[serde(default = "default_tools")]
    pub tools: Value,
    #[serde(default)]
    pub max_output_tokens: Value,
    #[serde(default)]
    pub previous_response_id: Value,
    #[serde(default)]
    pub include_encrypted_reasoning: bool,
}

impl Default for ResponsesEnvelope {
    fn default() -> Self {
        Self {
            metadata: Value::Null,
            parallel_tool_calls: true,
            temperature: Value::Null,
            top_p: Value::Null,
            reasoning: default_reasoning(),
            ignored_parameters: Vec::new(),
            tool_choice: default_tool_choice(),
            tools: default_tools(),
            max_output_tokens: Value::Null,
            previous_response_id: Value::Null,
            include_encrypted_reasoning: false,
        }
    }
}
