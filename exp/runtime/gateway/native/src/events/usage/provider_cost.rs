//! OpenRouter's own per-request charge on its usage object.
//!
//! OpenRouter reports what it charged the calling account for one generation
//! as `usage.cost`, a USD amount ("the total amount charged to your
//! account", https://openrouter.ai/docs/use-cases/usage-accounting), beside
//! `is_byok`, on both its Chat Completions and Responses usage. The gateway
//! reads it from the final usage report of the upstream stream (a
//! non-streaming caller drains that same stream) and settles it as
//! `provider_reported_cost_nano_usd`.
//!
//! Only `cost` is read. `cost_details.upstream_inference_cost` is what the
//! upstream provider charged, which OpenRouter publishes for BYOK requests;
//! it is not what OpenRouter charged the account, and the usage object carries
//! no fee field that would turn it into that amount. A usage without `cost`
//! therefore reports no charge rather than a reconstructed one. A BYOK answer
//! (`is_byok: true`) also reports none: the caller's own upstream key paid for
//! the generation, so `cost` is not its price. Only an absent `is_byok` or an
//! explicit `false` lets the cost through.
//!
//! The charge is metadata beside the token meter: an absent, malformed or
//! negative `cost` (or an `is_byok` that is neither absent nor `false`)
//! reports no charge and never fails the stream, so the host prices that
//! attempt from its token counts.

use serde_json::{Map, Number, Value};

use super::MAXIMUM_LEDGER_COUNT;

/// The catalog provider id of OpenRouter rungs, whose usage reports `cost`.
pub const OPENROUTER_PROVIDER: &str = "openrouter";

/// Nano-USD per USD.
const NANO_PER_USD: u64 = 1_000_000_000;

/// Decimal places in one nano-USD.
const NANO_DIGITS: usize = 9;

/// Read OpenRouter's `usage.cost` as whole nano-USD rounded up, or `None` when
/// the answer is BYOK or reports no usable cost.
pub(super) fn openrouter_cost_nano_usd(usage: &Map<String, Value>) -> Option<u64> {
    match usage.get("is_byok") {
        None | Some(Value::Bool(false)) => {}
        Some(_) => return None,
    }
    match usage.get("cost") {
        Some(Value::Number(cost)) => usd_to_nano_usd_ceil(cost),
        _ => None,
    }
}

/// Convert a non-negative USD JSON number to whole nano-USD, rounding up.
///
/// A float is converted from its shortest round-trip decimal text (the text
/// OpenRouter sent), never by multiplying the binary value, so `0.000123`
/// becomes exactly 123,000 rather than 123,001 from representation error.
/// OpenRouter serializes `cost` from an IEEE double, whose shortest text this
/// reproduces; a lexeme with more significant digits than a double holds is
/// rounded to the nearest double first (the parsed JSON keeps no lexeme).
/// Negative, non-finite and beyond-ledger amounts are `None`.
fn usd_to_nano_usd_ceil(cost: &Number) -> Option<u64> {
    if let Some(whole) = cost.as_u64() {
        return whole
            .checked_mul(NANO_PER_USD)
            .filter(|nano| *nano <= MAXIMUM_LEDGER_COUNT);
    }
    let value = cost.as_f64()?;
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    if value == 0.0 {
        return Some(0);
    }
    // Rust's float Display is the shortest round-trip text and never uses
    // exponent notation, so it is plain `digits[.digits]`.
    let text = value.to_string();
    let (whole, fraction) = text.split_once('.').unwrap_or((&text, ""));
    let whole: u64 = whole.parse().ok()?;
    let (kept, rest) = fraction.split_at(fraction.len().min(NANO_DIGITS));
    let kept: u64 = format!("{kept:0<NANO_DIGITS$}").parse().ok()?;
    let round_up = u64::from(rest.bytes().any(|digit| digit != b'0'));
    whole
        .checked_mul(NANO_PER_USD)?
        .checked_add(kept)?
        .checked_add(round_up)
        .filter(|nano| *nano <= MAXIMUM_LEDGER_COUNT)
}

#[cfg(test)]
#[path = "provider_cost_tests.rs"]
mod tests;
