"""Tests for the Bernoulli mixture fit."""

from __future__ import annotations

import numpy as np
import pytest

from exp.simulation.mining.clustering.mixture import bernoulli_mixture_fit, log_sum_exp


def _two_profiles(rows: int = 60) -> np.ndarray:
    """Return traces drawn from two clearly different tag profiles."""
    rng = np.random.default_rng(13)
    first = rng.binomial(1, [0.9, 0.9, 0.1, 0.1], size=(rows // 2, 4)).astype(float)
    second = rng.binomial(1, [0.1, 0.1, 0.9, 0.9], size=(rows // 2, 4)).astype(float)
    return np.vstack([first, second])


def test_log_sum_exp_survives_values_that_underflow_naively() -> None:
    """Replacing scipy here is only safe if the max-shift form is actually equivalent."""
    values = np.array([[-1000.0, -1001.0, -1002.0], [0.0, -1.0, -2.0]])

    reference = np.log(np.exp(values - values.max(axis=1, keepdims=True)).sum(axis=1)) + values.max(
        axis=1
    )

    assert np.allclose(log_sum_exp(values), reference, atol=1e-12)
    assert np.isfinite(log_sum_exp(values)).all()
    with np.errstate(divide="ignore"):
        # The same row summed without the shift underflows to zero and its log to -inf.
        assert np.isneginf(np.log(np.exp(values[0]).sum()))


def test_two_planted_profiles_are_recovered() -> None:
    """The model has to find structure a reviewer would agree is there.

    The rows are real binomial draws, so a few genuinely look like the other profile and land
    there; demanding a perfect split would be demanding the model ignore its evidence. What must
    hold is that each half has a clear majority group and that the fitted profiles match the ones
    the data was drawn from.
    """
    fit = bernoulli_mixture_fit(_two_profiles(), k=2, seed=1)

    first, second = fit.labels[:30], fit.labels[30:]
    assert np.bincount(first).argmax() != np.bincount(second).argmax()
    assert (first == np.bincount(first).argmax()).mean() > 0.85
    assert (second == np.bincount(second).argmax()).mean() > 0.85
    recovered = sorted(fit.theta.tolist(), key=lambda profile: profile[0])
    assert recovered[0] == pytest.approx([0.1, 0.1, 0.9, 0.9], abs=0.15)
    assert recovered[1] == pytest.approx([0.9, 0.9, 0.1, 0.1], abs=0.15)


def test_bic_chooses_the_planted_group_count_from_the_grid() -> None:
    """Letting BIC pick k is the reason this backend exists for an unlabelled corpus."""
    fit = bernoulli_mixture_fit(_two_profiles(), k=None, k_grid=(2, 3, 4, 5), seed=1)

    assert fit.chosen_k == 2
    assert set(fit.bic) == {2, 3, 4, 5}
    assert fit.bic[2] == min(fit.bic.values())


def test_profiles_stay_strictly_inside_the_unit_interval() -> None:
    """A tag nobody triggers would otherwise take the logarithm of zero."""
    never = np.zeros((10, 3))
    never[:, 0] = 1.0

    fit = bernoulli_mixture_fit(never, k=2, seed=0)

    assert fit.theta.min() > 0.0
    assert fit.theta.max() < 1.0
    assert np.isfinite(fit.log_pi).all()


def test_the_objective_never_decreases() -> None:
    """A monotone objective is the signal that the E and M steps agree with each other."""
    features = _two_profiles(40)
    totals = [
        bernoulli_mixture_fit(features, k=2, max_iter=steps, seed=2).novelty.sum()
        for steps in (1, 2, 5, 20, 100)
    ]

    assert totals == sorted(totals, reverse=True)


def test_soft_answers_are_read_as_uncertainty_rather_than_as_a_separate_class() -> None:
    """A 0.5 is a coin flip, and the model says so through novelty rather than through a label.

    Rows of 0.99 and rows of 0.5 agree about which tag is likelier, so there is no second latent
    profile to find and the mixture correctly explains both with one. The difference between them
    is how much information they carry, and that surfaces as novelty: the uncertain row is less
    well explained by whatever profile it belongs to. A caller wanting these apart in the geometry
    wants the logit transform, not another component.
    """
    confident = np.tile([0.99, 0.01], (10, 1))
    unsure = np.tile([0.5, 0.5], (10, 1))

    fit = bernoulli_mixture_fit(np.vstack([confident, unsure]), k=2, seed=3)

    assert fit.theta[0] == pytest.approx(fit.theta[1], abs=0.05)
    assert fit.novelty[-1] > fit.novelty[0] * 2


def test_the_novelty_pool_is_an_ordinary_nonnegative_group() -> None:
    """A cluster identity is persisted coverage evidence, so a sentinel of -1 is not available."""
    features = _two_profiles()

    fit = bernoulli_mixture_fit(features, k=2, novelty_quantile=0.1, seed=1)

    assert fit.labels.min() >= 0
    assert set(fit.labels.tolist()) == {0, 1, 2}


def test_a_non_probability_matrix_is_refused_with_the_fix() -> None:
    """Normalized or log-odds geometry has no reading as a per-tag probability."""
    with pytest.raises(ValueError, match="every value must be a finite number"):
        bernoulli_mixture_fit(np.array([[-0.7, 0.7]]), k=1)
    with pytest.raises(ValueError, match="every value must be a finite number"):
        bernoulli_mixture_fit(np.array([[1.4, 0.2]]), k=1)


def test_other_unusable_requests_are_refused() -> None:
    """Silently repairing a request would hide a misconfiguration in the spec."""
    with pytest.raises(ValueError, match="non-empty"):
        bernoulli_mixture_fit(np.zeros((0, 2)), k=1)
    with pytest.raises(ValueError, match="prior must be positive"):
        bernoulli_mixture_fit(_two_profiles(), k=2, prior=0.0)
    with pytest.raises(ValueError, match="novelty quantile"):
        bernoulli_mixture_fit(_two_profiles(), k=2, novelty_quantile=1.0)


def test_a_group_count_above_the_trace_count_is_clamped() -> None:
    """A grid written for a big corpus must still fit a small partition."""
    fit = bernoulli_mixture_fit(_two_profiles(6), k=None, k_grid=(2, 50), seed=1)

    assert fit.chosen_k <= 6
