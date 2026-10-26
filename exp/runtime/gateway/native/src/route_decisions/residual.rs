//! The score envelope of hundredth bins whose rounding residual was folded
//! into one bin.
//!
//! TypeSafe publishes bins rounded to hundredths with the normalization
//! residual folded into one highest bin, while its score keeps the unrounded
//! expectation; `score_matches_distribution` admits a score inside this
//! envelope beside the independent rounding envelope.

use super::{hundredths, rounding_bounds};

/// Extreme expectations, in half-hundredths, of the latent unit distributions
/// behind hundredth bins published by rounding every bin and folding the
/// normalization residual into one bin that rounded highest; `None` unless every
/// bin is hundredth-valued, the published total is exactly one, and some bin can
/// have been the folded one.
///
/// The bound is exact (an independent enumeration of the convention reaches
/// both extremes; see `score_tests`). In half-hundredth units, with published
/// hundredths `p_i`, latent `q_i` and a folded bin `a`:
///
/// - every other bin rounded to its published value, so `q_i` lies in its
///   closed rounding interval `[2p_i - 1, 2p_i + 1]` clipped to `[0, 200]`;
/// - the folded bin holds the rest, `q_a = 200 - S` with `S = sum of q_i, i != a`,
///   so the published total is one for every such `q`;
/// - the folded bin rounded to some hundredth `r_a >= M = max of p_i, i != a`,
///   which exists exactly when `q_a + 1 >= 2M`: `S <= 201 - 2M`, and `S <= 200`
///   keeps `q_a` a probability.
///
/// That is a box with one budget on `S`, and the weighted index
/// `200a + sum of (i - a) q_i` is linear, so its extremes are a fractional
/// knapsack: start every other bin at its lower bound and spend the budget on
/// the bins whose weight favors the extreme, largest weight first. Every
/// coordinate is an integer, so the extremes are integers too.
pub(super) fn residual_envelope(probabilities: &[f64]) -> Option<(i64, i64)> {
    let published = probabilities
        .iter()
        .map(|value| hundredths(*value))
        .collect::<Option<Vec<i64>>>()?;
    if published.iter().sum::<i64>() != 100 {
        return None;
    }
    let bounds = rounding_bounds(probabilities)?;
    let mut envelope: Option<(i64, i64)> = None;
    for folded in 0..bounds.len() {
        let others: Vec<(i64, i64, i64)> = bounds
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != folded)
            .map(|(index, (low, high))| (index as i64 - folded as i64, *low, *high))
            .collect();
        let highest = published
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != folded)
            .map(|(_, bin)| *bin)
            .max()
            .unwrap_or(0);
        let budget = 200.min(201 - 2 * highest);
        let lower_sum: i64 = others.iter().map(|(_, low, _)| low).sum();
        if lower_sum > budget {
            continue;
        }
        let base = 200 * folded as i64
            + others
                .iter()
                .map(|(weight, low, _)| weight * low)
                .sum::<i64>();
        let extreme = |sign: i64| {
            let mut favored: Vec<_> = others
                .iter()
                .filter(|(weight, _, _)| sign * weight > 0)
                .collect();
            favored.sort_by_key(|(weight, _, _)| -(sign * weight));
            let (mut value, mut remaining) = (base, budget - lower_sum);
            for (weight, low, high) in favored {
                let added = remaining.min(high - low);
                value += weight * added;
                remaining -= added;
            }
            value
        };
        let (minimum, maximum) = (extreme(-1), extreme(1));
        envelope = Some(envelope.map_or((minimum, maximum), |(low, high)| {
            (low.min(minimum), high.max(maximum))
        }));
    }
    envelope
}
