"""Fit an ephemeral multidimensional IRT model to graded test counts.

This module owns only the numeric core for the conditional graded-router study. It has no
filesystem output surface, so fitted abilities, difficulties, and discriminations stay inside the
remote fit process that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

DEFAULT_ABILITY_L2 = 0.01
DEFAULT_DIFFICULTY_L2 = 0.01
DEFAULT_DISCRIMINATION_L2 = 0.05


@dataclass(frozen=True)
class BinomialIrtFit:
    """One ephemeral multidimensional item-response fit."""

    abilities: np.ndarray
    difficulties: np.ndarray
    log_discriminations: np.ndarray
    loss: float
    iterations: int


def _validate_counts(passed: np.ndarray, total: np.ndarray) -> None:
    """Validate a dense task-by-arm matrix of exact binomial counts."""
    if passed.ndim != 2 or total.ndim != 2 or passed.shape != total.shape:
        raise ValueError("passed and total must be matching task-by-arm matrices")
    if not passed.size or not np.isfinite(passed).all() or not np.isfinite(total).all():
        raise ValueError("graded count matrices must be nonempty and finite")
    if np.any(total <= 0.0) or np.any(passed < 0.0) or np.any(passed > total):
        raise ValueError("graded counts require 0 <= passed <= positive total")
    if not np.array_equal(passed, np.rint(passed)) or not np.array_equal(total, np.rint(total)):
        raise ValueError("graded passed and total values must be integer counts")


def _logit(values: np.ndarray) -> np.ndarray:
    """Return a numerically bounded logit."""
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _parameter_shapes(
    task_count: int,
    arm_count: int,
    latent_dimension: int,
) -> tuple[int, int, int]:
    """Return flattened parameter block sizes."""
    if task_count < 1 or arm_count < 2 or latent_dimension < 1:
        raise ValueError("IRT dimensions require tasks, at least two arms, and latent_dimension")
    return (
        arm_count * latent_dimension,
        task_count,
        task_count * latent_dimension,
    )


def _unpack(
    parameters: np.ndarray,
    task_count: int,
    arm_count: int,
    latent_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """View one flat optimizer vector as abilities, difficulties, and log discriminations."""
    ability_size, difficulty_size, discrimination_size = _parameter_shapes(
        task_count,
        arm_count,
        latent_dimension,
    )
    if parameters.shape != (ability_size + difficulty_size + discrimination_size,):
        raise ValueError("IRT parameter vector has the wrong length")
    ability_end = ability_size
    difficulty_end = ability_end + difficulty_size
    abilities = parameters[:ability_end].reshape(arm_count, latent_dimension)
    difficulties = parameters[ability_end:difficulty_end]
    log_discriminations = parameters[difficulty_end:].reshape(task_count, latent_dimension)
    return abilities, difficulties, log_discriminations


def binomial_irt_loss_and_gradient(
    parameters: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
    *,
    ability_l2: float = DEFAULT_ABILITY_L2,
    difficulty_l2: float = DEFAULT_DIFFICULTY_L2,
    discrimination_l2: float = DEFAULT_DISCRIMINATION_L2,
) -> tuple[float, np.ndarray]:
    """Return exact binomial negative log likelihood and its analytic gradient.

    The likelihood is normalized by the number of fail-to-pass assertions, not the number of
    task-arm cells. A 100-test observation therefore contributes 100 times the evidence of an
    otherwise identical one-test observation.
    """
    _validate_counts(passed, total)
    if min(ability_l2, difficulty_l2, discrimination_l2) < 0.0:
        raise ValueError("IRT regularization strengths must be nonnegative")
    task_count, arm_count = passed.shape
    abilities, difficulties, log_discriminations = _unpack(
        parameters,
        task_count,
        arm_count,
        latent_dimension,
    )
    discriminations = np.exp(log_discriminations)
    logits = discriminations @ abilities.T - difficulties[:, None]
    assertion_count = float(np.sum(total))
    loss = float(np.sum(total * np.logaddexp(0.0, logits) - passed * logits) / assertion_count)
    loss += ability_l2 * float(np.mean(abilities**2))
    loss += difficulty_l2 * float(np.mean(difficulties**2))
    loss += discrimination_l2 * float(np.mean(log_discriminations**2))

    error = (total * expit(logits) - passed) / assertion_count
    ability_gradient = error.T @ discriminations
    ability_gradient += 2.0 * ability_l2 * abilities / abilities.size
    difficulty_gradient = -np.sum(error, axis=1)
    difficulty_gradient += 2.0 * difficulty_l2 * difficulties / difficulties.size
    discrimination_gradient = (error @ abilities) * discriminations
    discrimination_gradient += (
        2.0 * discrimination_l2 * log_discriminations / log_discriminations.size
    )
    return loss, np.concatenate(
        [
            ability_gradient.ravel(),
            difficulty_gradient,
            discrimination_gradient.ravel(),
        ]
    )


def _initial_parameters(
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
) -> np.ndarray:
    """Build a deterministic bounded initializer from marginal graded pass rates."""
    task_count, arm_count = passed.shape
    arm_rates = np.sum(passed, axis=0) / np.sum(total, axis=0)
    arm_logits = _logit(arm_rates)
    arm_logits -= np.mean(arm_logits)
    abilities = np.zeros((arm_count, latent_dimension), dtype=np.float64)
    abilities[:, 0] = arm_logits
    if latent_dimension > 1:
        arm_offsets = np.linspace(-0.01, 0.01, arm_count, dtype=np.float64)
        for dimension in range(1, latent_dimension):
            abilities[:, dimension] = arm_offsets * (dimension / latent_dimension)
    task_rates = np.sum(passed, axis=1) / np.sum(total, axis=1)
    difficulties = -_logit(task_rates)
    difficulties -= np.mean(difficulties)
    initial_discrimination = -0.5 * np.log(float(latent_dimension))
    log_discriminations = np.full(
        (task_count, latent_dimension),
        initial_discrimination,
        dtype=np.float64,
    )
    return np.concatenate(
        [abilities.ravel(), difficulties, log_discriminations.ravel()]
    )


def fit_binomial_irt(
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
    *,
    ability_l2: float = DEFAULT_ABILITY_L2,
    difficulty_l2: float = DEFAULT_DIFFICULTY_L2,
    discrimination_l2: float = DEFAULT_DISCRIMINATION_L2,
) -> BinomialIrtFit:
    """Fit the ephemeral count-weighted model with bounded L-BFGS optimization."""
    _validate_counts(passed, total)
    task_count, arm_count = passed.shape
    initial = _initial_parameters(passed, total, latent_dimension)
    ability_size, difficulty_size, discrimination_size = _parameter_shapes(
        task_count,
        arm_count,
        latent_dimension,
    )
    bounds = (
        [(-8.0, 8.0)] * ability_size
        + [(-8.0, 8.0)] * difficulty_size
        + [(-3.0, 3.0)] * discrimination_size
    )
    objective = partial(
        binomial_irt_loss_and_gradient,
        ability_l2=ability_l2,
        difficulty_l2=difficulty_l2,
        discrimination_l2=discrimination_l2,
    )
    result = minimize(
        objective,
        initial,
        args=(passed, total, latent_dimension),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1_000, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"binomial IRT optimization failed: {result.message}")
    abilities, difficulties, log_discriminations = _unpack(
        np.asarray(result.x, dtype=np.float64),
        task_count,
        arm_count,
        latent_dimension,
    )
    return BinomialIrtFit(
        abilities=abilities.copy(),
        difficulties=difficulties.copy(),
        log_discriminations=log_discriminations.copy(),
        loss=float(result.fun),
        iterations=int(result.nit),
    )


def predict_probabilities(fit: BinomialIrtFit) -> np.ndarray:
    """Predict fitted task-by-arm pass probabilities without serializing fit state."""
    if fit.abilities.ndim != 2 or fit.log_discriminations.ndim != 2:
        raise ValueError("IRT abilities and discriminations must be matrices")
    if fit.difficulties.shape != (len(fit.log_discriminations),):
        raise ValueError("IRT difficulty count does not match task discriminations")
    if fit.abilities.shape[1] != fit.log_discriminations.shape[1]:
        raise ValueError("IRT latent dimensions do not match")
    discriminations = np.exp(fit.log_discriminations)
    return expit(discriminations @ fit.abilities.T - fit.difficulties[:, None])
