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
