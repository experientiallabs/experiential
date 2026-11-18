//! Which bound ended a dial waiting for its provider's response headers.

use serde::{Deserialize, Serialize};

/// Which bound ended a dial waiting for its provider's response headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeaderTimeout {
    /// The rung's own first-byte allowance expired: the lane did not answer.
    Allowance,
    /// The request's shared budget (or deadline) expired first.
    Budget,
}

impl HeaderTimeout {
    /// The boundary token, shared with python `GatewayHeaderTimeout`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allowance => "allowance",
            Self::Budget => "budget",
        }
    }
}
