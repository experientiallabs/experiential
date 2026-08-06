"""Promotion gate for a distilled student, decided on held-out solve rates.

Three measurements on the same held-out task split at `gate.k` attempts each
feed the decision: the teacher running in the harness, the student before
training, and the student after training. The distilled adapter is promoted
iff the trained student reaches the configured fraction of the teacher's solve
rate and (when required) did not regress against its own pre-training
baseline. The verdict and every input number persist as a `DistillGateRecord`
in the run dir, mirroring how `wmo optimize` persists its `GateRecord`.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from wmo.optimize.model.config import GateConfig

# Tie tolerance for solve-rate comparisons. Mirrors the private _TIE_EPS in
# wmo/runtime/harness/create.py: rates computed from identical verdicts must compare
# equal, so a gate must never fail on float noise. Defined locally because
# create.py's constant is module-private.
_TIE_EPS = 1e-9

_SOLVE_RATE_HELP = (
    "solve rates are the fraction of held-out trials the verifier passed, "
    "so they must be finite values in [0, 1]; recompute them from the eval "
    "report instead of passing raw pass counts"
)


class DistillGateRecord(BaseModel):
    """The distillation promotion verdict plus every number that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    reason: str
    """Accept/reject reasoning; states the compared solve rates explicitly."""

    teacher_solve_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    student_before_solve_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    student_after_solve_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    min_teacher_fraction: float = Field(gt=0.0, le=1.0)


def gate_distillation(
    teacher_solve_rate: float,
    student_before_solve_rate: float,
    student_after_solve_rate: float,
    spec: GateConfig,
) -> DistillGateRecord:
    """Decide whether the distilled student is promoted, on held-out solve rates.

    The gate accepts iff the trained student's solve rate is at least
    `spec.min_teacher_fraction` of the teacher's (within a float-noise tie
    epsilon), and, when `spec.require_no_regression` is set, at least the
    student's own pre-training solve rate (same epsilon). All three rates must
    come from the same held-out split at `spec.k` attempts per task so the
    comparison is like for like.

    Args:
        teacher_solve_rate: The teacher-in-harness held-out solve rate in [0, 1].
        student_before_solve_rate: The untrained student's solve rate in [0, 1].
        student_after_solve_rate: The trained student's solve rate in [0, 1].
        spec: The run's gate section (threshold fraction, regression policy,
            and the attempt count `k` the rates were measured at).

    Returns:
        The verdict record; `reason` states the compared numbers.

    Raises:
        ValueError: If a solve rate is not a finite value in [0, 1]; the
            message names the offending rate.
    """
    rates = (
        ("teacher", teacher_solve_rate),
        ("student-before", student_before_solve_rate),
        ("student-after", student_after_solve_rate),
    )
    for label, value in rates:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {label} solve rate {value!r}: {_SOLVE_RATE_HELP}")

    threshold = spec.min_teacher_fraction * teacher_solve_rate
    failures: list[str] = []
    if student_after_solve_rate < threshold - _TIE_EPS:
        failures.append(
            f"after {student_after_solve_rate:.3f} below {spec.min_teacher_fraction:.2f} x "
            f"teacher {teacher_solve_rate:.3f} = {threshold:.3f}"
        )
    if spec.require_no_regression and (
        student_after_solve_rate < student_before_solve_rate - _TIE_EPS
    ):
        failures.append(
            f"after {student_after_solve_rate:.3f} regressed below "
            f"before {student_before_solve_rate:.3f}"
        )

    if failures:
        reason = f"rejected: {'; '.join(failures)} (k={spec.k} attempts)"
    else:
        regression_note = (
            f"; no regression: after {student_after_solve_rate:.3f} >= "
            f"before {student_before_solve_rate:.3f}"
            if spec.require_no_regression
            else "; regression check disabled"
        )
        reason = (
            f"accepted: after {student_after_solve_rate:.3f} >= "
            f"{spec.min_teacher_fraction:.2f} x teacher {teacher_solve_rate:.3f} = "
            f"{threshold:.3f}{regression_note} (k={spec.k} attempts)"
        )
    return DistillGateRecord(
        accepted=not failures,
        reason=reason,
        teacher_solve_rate=teacher_solve_rate,
        student_before_solve_rate=student_before_solve_rate,
        student_after_solve_rate=student_after_solve_rate,
        min_teacher_fraction=spec.min_teacher_fraction,
    )
