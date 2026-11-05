"""Credential-free preparation cost and lifecycle admission shared by planning and execution."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from exp.runtime.models.spend_ceiling import MAXIMUM_UNPRICED_ATTEMPTS

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapApproval, BootstrapPlan
    from exp.optimize.evaluation.variance.bootstrap_resume import BootstrapResumeBinding

BOOTSTRAP_ADMISSION_PROTOCOL = "bootstrap-cost-admission-v1"


def validate_preparation_admission(
    plan: BootstrapPlan,
    binding: BootstrapResumeBinding | None,
    approval: BootstrapApproval | None,
    *,
    now: datetime,
) -> None:
    """Check the full cumulative native quote without granting or dispatching anything.

    Runtime/client readiness and actual execution remain separate. No operational timeout or
    authorization is renewed here; unchanged completed results still replay through their owner.
    """
    if plan.unpriced_attempts > MAXIMUM_UNPRICED_ATTEMPTS:
        raise ValueError("bootstrap full pending inventory exceeds the parent request cap")
    if binding is not None:
        if now >= binding.stop:
            raise ValueError(
                "bootstrap preparation execution deadline expired; register a finite renewal"
            )
        if (
            plan.identity != binding.original_operation
            or plan.known_usd > binding.preparation_usd
            or plan.unpriced_attempts > binding.preparation_unpriced_attempts
            or plan.known_usd > binding.ceiling_usd
            or plan.unpriced_attempts > binding.unpriced_attempts
        ):
            raise ValueError(
                "bootstrap resume inventory exceeds the original preparation authority"
            )
        if approval is not None and (
            approval.plan_sha256 != plan.digest
            or approval.ceiling_usd != binding.preparation_usd
            or approval.unpriced_attempts != binding.preparation_unpriced_attempts
        ):
            raise ValueError("bootstrap resume approval cannot change original authority")
    elif not plan.complete:
        if approval is None or approval.plan_sha256 != plan.digest:
            raise ValueError(
                "bootstrap preparation needs explicit approval of its exact current plan"
            )
        if (
            approval.ceiling_usd < plan.known_usd
            or approval.unpriced_attempts < plan.unpriced_attempts
        ):
            raise ValueError("bootstrap approval does not cover its priced and unpriced inventory")
