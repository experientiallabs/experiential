"""Durable evaluation exclusions for admitted judge calls that return unusable output."""

from __future__ import annotations

import math
from contextlib import nullcontext

from exp.common.judging import Judgment
from exp.common.models import CompletionCostReservation, ModelSnapshot
from exp.common.project import ProjectStore
from exp.optimize.router.automatic.judge import AutomaticRouterJudge, ReservedJudgeClient
from exp.optimize.router.errors import JudgeDispatchExhaustedError
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.transport import ProviderTransportError


class DurableEvaluationJudge:
    """Preserve failed-call spend so malformed model output cannot repeatedly incur charges."""

    def __init__(
        self,
        delegate: AutomaticRouterJudge,
        client: ReservedJudgeClient,
        reservation: CompletionCostReservation,
        *,
        budget: RequestBudget | None = None,
    ) -> None:
        """Bind the canonical judge to the same reservation-enforcing provider client."""
        self._delegate = delegate
        self._client = client
        self._reservation = reservation
        self._budget = budget

    @property
    def model(self) -> ModelSnapshot:
        """Expose the delegate's verified provider identity before any execution."""
        return self._delegate.model

    def judge_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: str,
        rubric_artifact_id: str,
        calibration_artifact_id: str,
    ) -> Judgment:
        """Score a rollout or return a durable, priced exclusion to the shared executor.

        Args:
            store: Immutable evaluation artifact owner.
            rollout_artifact_id: Exact rollout to judge.
            rubric_artifact_id: Frozen scoring axes.
            calibration_artifact_id: Frozen judge provenance.

        Returns:
            Canonical structured judgment when the configured judge succeeds.

        Raises:
            JudgeDispatchExhaustedError: An admitted call failed or returned unusable output.
                Its cost includes all counterbalanced calls, not just the final failure.
            ValueError: A failure before provider admission; no new spend is inferred.
        """
        calls_before = self._client.calls
        economics_before = len(self._client.economics)
        try:
            context = (
                self._budget.scope(f"judge:{rollout_artifact_id}")
                if self._budget
                else nullcontext()
            )
            with context:
                return self._delegate.judge_persisted(
                    store,
                    rollout_artifact_id=rollout_artifact_id,
                    rubric_artifact_id=rubric_artifact_id,
                    calibration_artifact_id=calibration_artifact_id,
                )
        except ProviderParameterError as exc:
            raise ValueError(f"judge request settings are invalid: {exc}") from exc
        except (ValueError, ProviderTransportError, ProviderDeadlineExceeded) as exc:
            dispatched = self._client.calls - calls_before
            if dispatched == 0:
                raise
            economics = self._client.economics[economics_before:]
            missing = dispatched - len(economics)
            costs = [
                item.cost_usd.value
                if item.cost_usd is not None
                else self._reservation.absolute_maximum_call_cost_usd()
                for item in economics
            ]
            if isinstance(exc, JudgeDispatchExhaustedError) and missing > 0:
                costs.append(exc.conservative_cost_usd)
                missing -= 1
            costs.extend([self._reservation.absolute_maximum_call_cost_usd()] * missing)
            raise JudgeDispatchExhaustedError(
                f"judge dispatch did not produce usable scoring evidence ({type(exc).__name__})",
                conservative_cost_usd=math.fsum(costs),
            ) from exc
