"""A Bernoulli mixture over tag probabilities, fitted by EM.

Each group gets a profile: for every tag, how often cases in that group trigger it. A case's tags
are then treated as independent coin flips given its group. That is a strong assumption and it is
worth naming: two tags asking nearly the same question violate it, and the posteriors come out more
confident than the evidence supports. What the model buys in exchange is the right shape for this
data. k-means on tag probabilities measures distance in a space where "0.5" means a tag half fired;
a mixture reads 0.5 as the model's uncertainty and weighs it as such, and it reports a posterior per
group instead of a hard nearest-centroid verdict.

Soft probabilities enter EM as fractional observations, so what the iteration maximizes is a
cross-entropy rather than a strict likelihood. The quantity still rises monotonically and still
selects a sensible number of groups through BIC, which is what it is used for here.

The number of groups can be left unset, in which case every candidate in the grid is fitted and the
lowest BIC wins. That matters for a corpus nobody has labelled: the alternative is guessing a group
count and then discovering the guess shaped the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from exp.simulation.mining.clustering.kmeans import kmeans_fit

NOVELTY_LABEL_REASON = "above the training novelty quantile"
"""Why a trace lands in the extra group: the mixture explains it worse than its peers."""


@dataclass(frozen=True)
class BernoulliMixtureFit:
    """One completed mixture fit.

    Attributes:
        labels: ``(N,)`` group index per trace, in ``[0, k)``, or ``k`` for a trace the novelty
            quantile set aside. The pool gets its own group rather than a negative sentinel,
            because a cluster identity is persisted coverage evidence and must stay nonnegative.
        theta: ``(k, T)`` per-group probability of each tag.
        log_pi: ``(k,)`` log mixing weights.
        responsibilities: ``(N, k)`` posterior over groups, the soft answer behind the labels.
        novelty: ``(N,)`` mean negative log-likelihood per tag; higher is less well explained.
        bic: BIC per candidate group count that was fitted.
        chosen_k: The group count the fit settled on.
        iterations: EM iterations the chosen fit used.
    """

    labels: np.ndarray
    theta: np.ndarray
    log_pi: np.ndarray
    responsibilities: np.ndarray
    novelty: np.ndarray
    bic: dict[int, float] = field(default_factory=dict)
    chosen_k: int = 0
    iterations: int = 0


def log_sum_exp(values: np.ndarray) -> np.ndarray:
    """Return ``log(sum(exp(values)))`` along the last axis without overflowing.

    Subtracting each row's peak before exponentiating keeps every term in ``(0, 1]``, which is the
    whole trick: the untransformed sum of exponentials of large negative log-likelihoods underflows
    to zero and its logarithm to negative infinity.

    Args:
        values: ``(N, K)`` log-domain values.

    Returns:
        ``(N,)`` log-sum-exp per row.
    """
    peak = values.max(axis=1, keepdims=True)
    return (peak + np.log(np.exp(values - peak).sum(axis=1, keepdims=True)))[:, 0]


def bernoulli_mixture_fit(
    probabilities: np.ndarray,
    *,
    k: int | None = None,
    k_grid: tuple[int, ...] = (5, 10, 25, 50),
    max_iter: int = 300,
    tol: float = 1e-6,
    prior: float = 1.0,
    novelty_quantile: float | None = None,
    seed: int = 42,
) -> BernoulliMixtureFit:
    """Fit a Bernoulli mixture to tag probabilities, choosing the group count by BIC when unset.

    Args:
        probabilities: ``(N, T)`` tag probabilities, every value in ``[0, 1]``.
        k: Number of groups, or None to fit every count in ``k_grid`` and keep the lowest BIC.
        k_grid: Candidate group counts used when ``k`` is None.
        max_iter: EM iterations per fit.
        tol: Stop once the relative change in total log-likelihood falls to this or below.
        prior: Symmetric Beta smoothing on every profile entry. It must be positive, which is what
            keeps a profile strictly inside ``(0, 1)`` so its logarithm stays finite even when a
            tag never fires anywhere in the corpus.
        novelty_quantile: Upper quantile of training novelty whose traces are set aside into one
            extra group, or None to label every trace by its best group.
        seed: Seed for the k-means partition EM starts from.

    Returns:
        The fitted mixture, its labels and the BIC of every count that was tried.

    Raises:
        ValueError: The matrix is empty or carries a value outside ``[0, 1]``, the prior is not
            positive, a requested group count exceeds the number of traces, or the grid is empty.
    """
    features = np.asarray(probabilities, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("a Bernoulli mixture needs a non-empty trace-by-tag probability matrix")
    if not np.isfinite(features).all() or features.min() < 0.0 or features.max() > 1.0:
        raise ValueError(
            "a Bernoulli mixture models per-tag probabilities, so every value must be a finite "
            "number in [0, 1]; pass the untransformed decision-model answers rather than a "
            "normalized or log-odds geometry"
        )
    if prior <= 0.0:
        raise ValueError("the Beta smoothing prior must be positive to keep log profiles finite")
    counts = (k,) if k is not None else tuple(k_grid)
    if not counts:
        raise ValueError("a Bernoulli mixture needs at least one candidate group count")
    usable = tuple(sorted({min(count, len(features)) for count in counts if count >= 1}))
    if not usable:
        raise ValueError("every candidate group count must be at least one")
    scored = {
        count: _fit_one(features, count, max_iter=max_iter, tol=tol, prior=prior, seed=seed)
        for count in usable
    }
    bic = {count: fit.bic[count] for count, fit in scored.items()}
    chosen = min(usable, key=lambda count: (bic[count], count))
    fit = scored[chosen]
    labels = fit.labels
    if novelty_quantile is not None:
        if not 0.0 < novelty_quantile < 1.0:
            raise ValueError("the novelty quantile must be strictly between 0 and 1")
        threshold = float(np.quantile(fit.novelty, 1.0 - novelty_quantile))
        labels = np.where(fit.novelty > threshold, chosen, fit.labels)
    return BernoulliMixtureFit(
        labels=labels,
        theta=fit.theta,
        log_pi=fit.log_pi,
        responsibilities=fit.responsibilities,
        novelty=fit.novelty,
        bic=bic,
        chosen_k=chosen,
        iterations=fit.iterations,
    )


def _fit_one(
    features: np.ndarray, k: int, *, max_iter: int, tol: float, prior: float, seed: int
) -> BernoulliMixtureFit:
    """Run EM for one group count, starting from a single-restart k-means partition."""
    initial = kmeans_fit(features, k, n_init=1, max_iter=max_iter, seed=seed).labels
    responsibilities = np.eye(k)[initial]
    theta = np.zeros((k, features.shape[1]))
    log_pi = np.zeros(k)
    log_likelihood = np.zeros(len(features))
    previous = -np.inf
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        theta, log_pi = _maximize(features, responsibilities, prior)
        responsibilities, log_likelihood = _expect(features, theta, log_pi)
        total = float(log_likelihood.sum())
        if abs(total - previous) <= tol * abs(total):
            break
        previous = total
    theta, log_pi = _maximize(features, responsibilities, prior)
    responsibilities, log_likelihood = _expect(features, theta, log_pi)
    parameters = k * features.shape[1] + k - 1
    bic = float(-2 * log_likelihood.sum() + parameters * np.log(len(features)))
    return BernoulliMixtureFit(
        labels=responsibilities.argmax(axis=1),
        theta=theta,
        log_pi=log_pi,
        responsibilities=responsibilities,
        novelty=-log_likelihood / features.shape[1],
        bic={k: bic},
        chosen_k=k,
        iterations=iterations,
    )


def _maximize(
    features: np.ndarray, responsibilities: np.ndarray, prior: float
) -> tuple[np.ndarray, np.ndarray]:
    """Re-estimate every profile and mixing weight from the current responsibilities."""
    occupancy = responsibilities.sum(axis=0)
    weighted = responsibilities.T @ features
    theta = (weighted + prior) / (occupancy[:, None] + 2 * prior)
    weights = occupancy + prior
    return theta, np.log(weights / weights.sum())


def _expect(
    features: np.ndarray, theta: np.ndarray, log_pi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the posterior over groups and each trace's total log-likelihood."""
    joint = features @ np.log(theta).T + (1 - features) @ np.log1p(-theta).T + log_pi
    log_likelihood = log_sum_exp(joint)
    return np.exp(joint - log_likelihood[:, None]), log_likelihood
