//! Score-consistency envelope and diagnosable rejection regressions.

use super::tests::score_payload;
use super::*;

#[test]
fn a_finer_score_is_judged_against_the_latent_expectation_envelope() {
    // Thirds published as 0.33 each beside the exact latent expectation of
    // (0.333, 0.333, 0.334): the recomputed 0.99 differs by 0.011, yet one
    // unit distribution rounding to every bin explains the score exactly.
    let (admitted, provider) = score_payload(&[0.33, 0.33, 0.33], 1.001);
    public_decisions(provider, &admitted).expect("latent expectation of rounded bins");
    let thirds = [0.33, 0.33, 0.33];
    // Latent expectations of bins rounding to 0.33 span [0.995, 1.005]; a
    // half-thousandth rounding allowance closes the envelope.
    assert!(score_matches_distribution(0.995, &thirds));
    assert!(score_matches_distribution(1.005, &thirds));
    assert!(!score_matches_distribution(1.006, &thirds));
    assert!(!score_matches_distribution(0.994, &thirds));
    // Precision finer than thousandths keeps the exact check.
    assert!(!score_matches_distribution(1.0001, &thirds));
    // Finer bins get no envelope: only the exact recomputed expectation passes.
    assert!(score_matches_distribution(1.001, &[0.333, 0.333, 0.334]));
    assert!(!score_matches_distribution(1.002, &[0.333, 0.333, 0.334]));
}

#[test]
fn a_tenths_score_is_judged_at_the_precision_it_was_reported() {
    // Production 2026-10-17 (jev-latest via typesafe, 16 refusals): bins
    // whose latent expectations span [3.85, 3.89] beside a score the
    // provider rounded to ONE decimal. 3.9 stands for [3.85, 3.95].
    let bins = [0.0, 0.01, 0.01, 0.08, 0.9];
    let (admitted, provider) = score_payload(&bins, 3.9);
    public_decisions(provider, &admitted).expect("a tenths-rounded score of 3.87");
    assert!(score_matches_distribution(3.9, &bins));
    assert!(score_matches_distribution(3.8, &bins));
    // A genuine mismatch at tenths is still refused: 3.7 covers at most
    // 3.75, below every expectation the bins allow.
    assert!(!score_matches_distribution(3.7, &bins));
    let (admitted, provider) = score_payload(&bins, 3.7);
    assert!(public_decisions(provider, &admitted)
        .expect_err("3.7 is not 3.87 at any rounding")
        .safe_message
        .starts_with("decision score does not match its probability distribution ("));
    // A whole number reads at tenths, never as a half-unit allowance: 4
    // covers [3.95, 4.05], above the envelope.
    assert!(!score_matches_distribution(4.0, &bins));
    // Hundredths keep their own half-hundredth allowance.
    assert!(score_matches_distribution(3.89, &bins));
    assert!(!score_matches_distribution(3.91, &bins));
}

#[test]
fn a_rejected_score_names_its_numbers_in_the_ledger_detail() {
    let (admitted, provider) = score_payload(&[0.0, 0.87, 0.13, 0.0, 0.0], 1.5);
    let failure = public_decisions(provider, &admitted).expect_err("inconsistent score");
    assert!(failure
        .safe_message
        .starts_with("decision score does not match its probability distribution ("));
    for fact in [
        "score=1.5",
        "expected=1.13",
        "probabilities=[0,0.87,0.13,0,0]",
        "rounding_envelope=[1.12,1.15]",
        "residual_envelope=[1.12,1.16]",
        "tolerance=0.000001",
    ] {
        assert!(
            failure.safe_message.contains(fact),
            "{fact} in {}",
            failure.safe_message
        );
    }
    let boundary = failure.boundary();
    assert_eq!(
        boundary.safe_message,
        "provider returned a malformed response; retry the request"
    );
    assert!(boundary.provider_detail.unwrap().contains("score=1.5"));
}

#[test]
fn a_score_beside_a_folded_rounding_residual_is_admitted() {
    // Production 2026-10-23 (jev-latest via typesafe, 82 refusals in three
    // hours, numbers only): TypeSafe rounds every bin to hundredths, folds the
    // residual into the highest bin so the total reads exactly one, and keeps
    // the unrounded expectation as the score. Four levels miss the independent
    // envelope by exactly 0.03, five levels by 0.04.
    for (probabilities, score) in [
        (vec![0.49, 0.37, 0.11, 0.03], 0.71),
        (vec![0.62, 0.26, 0.08, 0.04], 0.57),
        (vec![0.73, 0.18, 0.06, 0.03], 0.36),
        (vec![0.47, 0.36, 0.13, 0.04], 0.71),
        (vec![0.44, 0.3, 0.16, 0.1], 0.95),
        (vec![0.6900000000000001, 0.21, 0.06, 0.04], 0.48),
        (vec![0.6799999999999999, 0.2, 0.08, 0.04], 0.45),
        (vec![0.3, 0.5, 0.15, 0.03, 0.02], 0.93),
        (vec![0.02, 0.01, 0.05, 0.52, 0.4], 3.23),
    ] {
        let independent = score_envelope(&probabilities).expect("unit hundredth bins");
        assert!(
            !score_meets_envelope(score, independent.0, independent.1),
            "{probabilities:?} {score} is outside the independent envelope"
        );
        let (admitted, provider) = score_payload(&probabilities, score);
        let (public, _) = public_decisions(provider.clone(), &admitted)
            .expect("a folded rounding residual explains the score");
        assert_eq!(public["answers"], provider["answers"]);
    }
    // One hundredth further is refused: the folded bin sits at most half a
    // hundredth per other bin from its latent value.
    for (probabilities, score) in [
        (vec![0.49, 0.37, 0.11, 0.03], 0.72),
        (vec![0.49, 0.37, 0.11, 0.03], 0.64),
        (vec![0.3, 0.5, 0.15, 0.03, 0.02], 0.92),
        (vec![0.02, 0.01, 0.05, 0.52, 0.4], 3.22),
    ] {
        let (admitted, provider) = score_payload(&probabilities, score);
        assert!(public_decisions(provider, &admitted)
            .expect_err("beyond the folded residual")
            .safe_message
            .contains("residual_envelope=["));
    }
    // A middle highest bin folds less: [0.3, 0.6, 0.1] spans only [0.79, 0.81]
    // under folding, so 0.82 is still refused.
    assert!(!score_matches_distribution(0.82, &[0.3, 0.6, 0.1]));
    // Bins that do not total exactly one were not folded.
    assert_eq!(residual_envelope(&[0.33, 0.33, 0.33]), None);
    assert_eq!(residual_envelope(&[0.333, 0.333, 0.334]), None);
}

#[test]
fn folded_residual_limits_admit_exactly_at_the_bound_in_both_directions() {
    // (bins, exact envelope in half-hundredths, lowest and highest admitted
    // score, the next score below and above that is refused).
    for (probabilities, envelope, admitted, refused) in [
        // Production four-level shape: independent [0.66, 0.70].
        (
            vec![0.49, 0.37, 0.11, 0.03],
            (130, 142),
            [0.65, 0.71],
            [0.649, 0.711],
        ),
        // Production five-level shape, middle bin folded.
        (
            vec![0.3, 0.5, 0.15, 0.03, 0.02],
            (187, 201),
            [0.935, 1.005],
            [0.934, 1.006],
        ),
        // Near tie: either 0.42 may have been folded, but the folded bin must
        // still have rounded highest, so its latent value stays at or above
        // 0.415. Without that coupling the top read 0.755.
        (
            vec![0.42, 0.42, 0.16],
            (145, 150),
            [0.725, 0.75],
            [0.724, 0.755],
        ),
        // A unit bin may have absorbed four bins just under half a hundredth
        // each: latent (0.9804, 0.0049, 0.0049, 0.0049, 0.0049) publishes
        // (1, 0, 0, 0, 0) beside a score of 0.049, reported as 0.05.
        (
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            (0, 10),
            [0.0, 0.05],
            [-0.001, 0.051],
        ),
    ] {
        assert_eq!(residual_envelope(&probabilities), Some(envelope));
        for score in admitted {
            let (admission, provider) = score_payload(&probabilities, score);
            public_decisions(provider, &admission)
                .unwrap_or_else(|_| panic!("{probabilities:?} admits {score}"));
        }
        for score in refused {
            let (admission, provider) = score_payload(&probabilities, score);
            assert!(
                public_decisions(provider, &admission).is_err(),
                "{probabilities:?} refuses {score}"
            );
        }
    }
}

/// Every latent expectation that publishes exactly `probabilities` under the
/// folded-residual convention, by enumeration: latent bins on a quarter-hundredth
/// grid (twice as fine as the half-hundredth rounding ties) within half a
/// hundredth per bin, plus a quarter-hundredth, of their published value (the
/// folded bin is the only one that can drift, by at most half a hundredth per
/// other bin), the last bin taking the rest; each bin
/// rounded to hundredths with ties either way; the residual folded into each bin
/// that rounded highest. Returns the extremes in half-hundredths, or `None` when
/// nothing publishes these bins. Shares nothing with the envelope's knapsack.
fn enumerate_folded(probabilities: &[f64]) -> Option<(i64, i64)> {
    fn walk(
        index: usize,
        latent: &mut Vec<i64>,
        published: &[i64],
        extremes: &mut Option<(i64, i64)>,
    ) {
        if index + 1 < published.len() {
            let (center, span) = (published[index] * 4, 2 * published.len() as i64 + 1);
            for units in (center - span).max(0)..=(center + span).min(400) {
                latent[index] = units;
                walk(index + 1, latent, published, extremes);
            }
            return;
        }
        let last = 400 - latent[..index].iter().sum::<i64>();
        if !(0..=400).contains(&last) {
            return;
        }
        latent[index] = last;
        let mut rounded = [0_i64; 10];
        round_each(
            0,
            &mut rounded[..published.len()],
            latent,
            published,
            extremes,
        );
    }
    fn round_each(
        index: usize,
        rounded: &mut [i64],
        latent: &[i64],
        published: &[i64],
        extremes: &mut Option<(i64, i64)>,
    ) {
        if index == latent.len() {
            let highest = *rounded.iter().max().unwrap();
            let residual = 100 - rounded.iter().sum::<i64>();
            let publishes = |folded: usize| {
                rounded.iter().enumerate().all(|(i, value)| {
                    published[i] == value + if i == folded { residual } else { 0 }
                })
            };
            if (0..rounded.len()).any(|folded| rounded[folded] == highest && publishes(folded)) {
                // Quarter-hundredths to half-hundredths is exact at the
                // extremes; they sit on the half-hundredth grid.
                let value: i64 = latent.iter().enumerate().map(|(i, q)| i as i64 * q).sum();
                *extremes = Some(extremes.map_or((value, value), |(low, high)| {
                    (low.min(value), high.max(value))
                }));
            }
            return;
        }
        let (floor, remainder) = (latent[index] / 4, latent[index] % 4);
        let (first, last) = match remainder {
            2 => (floor, floor + 1),
            3 => (floor + 1, floor + 1),
            _ => (floor, floor),
        };
        for choice in first..=last {
            rounded[index] = choice;
            round_each(index + 1, rounded, latent, published, extremes);
        }
    }
    let published: Vec<i64> = probabilities
        .iter()
        .map(|p| (p * 100.0).round() as i64)
        .collect();
    let mut extremes = None;
    walk(0, &mut vec![0; published.len()], &published, &mut extremes);
    extremes.map(|(low, high)| {
        assert!(
            low % 2 == 0 && high % 2 == 0,
            "extremes on the half-hundredth grid"
        );
        (low / 2, high / 2)
    })
}

#[test]
fn the_folded_residual_envelope_matches_an_independent_enumeration() {
    for probabilities in [
        vec![0.49, 0.37, 0.11, 0.03],
        vec![0.3, 0.5, 0.15, 0.03, 0.02],
        vec![0.02, 0.01, 0.05, 0.52, 0.4],
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.87, 0.13, 0.0, 0.0],
        vec![0.33, 0.34, 0.01, 0.32],
        vec![0.42, 0.42, 0.16],
        vec![0.5, 0.5],
        vec![0.3, 0.6, 0.1],
    ] {
        assert_eq!(
            residual_envelope(&probabilities),
            enumerate_folded(&probabilities),
            "{probabilities:?}"
        );
    }
}

#[test]
fn the_folded_residual_envelope_is_exact_on_random_and_near_tied_bins() {
    // Deterministic xorshift so a failure names a reproducible distribution.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut cases = 0;
    let mut published_cases = 0;
    while cases < 1_500 {
        // Five bins cost ~400k latent points each, so they are sampled sparingly.
        let count = if cases % 300 == 0 {
            5
        } else {
            2 + next(3) as usize
        };
        let probabilities: Vec<f64> = match cases % 3 {
            // A latent distribution published through the convention, ties and
            // the folded bin among the highest chosen at random.
            0 | 1 => {
                let mut latent = vec![0_i64; count];
                let mut left = 400_i64;
                for (index, bin) in latent.iter_mut().enumerate() {
                    *bin = if index + 1 == count {
                        left
                    } else {
                        next(left as u64 + 1) as i64
                    };
                    left -= *bin;
                }
                if cases % 3 == 1 && count > 2 {
                    // Near tie: two bins within a quarter-hundredth.
                    let (first, second) = (latent[0], latent[1]);
                    let spread = (first + second) / 2;
                    latent[0] = spread;
                    latent[1] = first + second - spread;
                }
                let mut rounded: Vec<i64> = latent
                    .iter()
                    .map(|units| match units % 4 {
                        2 => units / 4 + next(2) as i64,
                        3 => units / 4 + 1,
                        _ => units / 4,
                    })
                    .collect();
                let highest = *rounded.iter().max().unwrap();
                let candidates: Vec<usize> =
                    (0..count).filter(|i| rounded[*i] == highest).collect();
                let folded = candidates[next(candidates.len() as u64) as usize];
                rounded[folded] += 100 - rounded.iter().sum::<i64>();
                rounded.iter().map(|bin| *bin as f64 / 100.0).collect()
            }
            // Arbitrary unit hundredth bins, most of which no latent
            // distribution publishes.
            _ => {
                let mut left = 100_i64;
                (0..count)
                    .map(|index| {
                        let bin = if index + 1 == count {
                            left
                        } else {
                            next(left as u64 + 1) as i64
                        };
                        left -= bin;
                        bin as f64 / 100.0
                    })
                    .collect()
            }
        };
        let expected = enumerate_folded(&probabilities);
        published_cases += usize::from(expected.is_some());
        assert_eq!(
            residual_envelope(&probabilities),
            expected,
            "{probabilities:?}"
        );
        cases += 1;
    }
    assert!(
        published_cases >= 1_000,
        "{published_cases} reachable cases"
    );
}
