//! The bounded category of a provider refusal.

use serde::{Deserialize, Serialize};

/// The bounded category of a provider refusal, derived from the provider's
/// own code and sentence (`stream_errors::refusal_reason`) and shared with
/// the python `GatewayRefusalReason`.
///
/// The caller sees WHICH policy declined the content as a fixed phrase,
/// never the provider's prose: the vocabulary is closed so a client can
/// branch on it, and the raw token keeps riding `provider_detail` into the
/// ledger only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalReason {
    /// A cyber-safety verdict (OpenAI `cyber_policy`).
    CyberPolicy,
    /// A biological, chemical, radiological, or nuclear weapons verdict.
    Cbrn,
    /// A general content-policy or safety-system verdict.
    ContentPolicy,
    /// The output would recite copyrighted material (Gemini `RECITATION`).
    Recitation,
    /// The provider's data-inspection layer blocked the content (Gemini
    /// `SPII`, Qwen `data_inspection_failed`).
    DataInspection,
    /// The provider refused without naming a reason.
    Unspecified,
}

impl RefusalReason {
    /// The wire name shared with the python enum and the public error field.
    pub fn as_str(&self) -> &'static str {
        match self {
            RefusalReason::CyberPolicy => "cyber_policy",
            RefusalReason::Cbrn => "cbrn",
            RefusalReason::ContentPolicy => "content_policy",
            RefusalReason::Recitation => "recitation",
            RefusalReason::DataInspection => "data_inspection",
            RefusalReason::Unspecified => "unspecified",
        }
    }

    /// The fixed caller-facing phrase, or none for an unnamed refusal.
    pub fn phrase(&self) -> Option<&'static str> {
        match self {
            RefusalReason::CyberPolicy => Some("cybersecurity policy"),
            RefusalReason::Cbrn => Some("biological/chemical safety policy"),
            RefusalReason::ContentPolicy => Some("content policy"),
            RefusalReason::Recitation => Some("recitation of copyrighted material"),
            RefusalReason::DataInspection => Some("data inspection"),
            RefusalReason::Unspecified => None,
        }
    }
}
