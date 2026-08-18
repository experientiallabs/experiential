"""Execution of the approved manual judge contract against plan-bound rollouts."""

from __future__ import annotations

from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.evaluations import EvaluationPlan
from wmo.common.evaluations.evidence import read_evaluation_plan, read_rollout
from wmo.common.judging import Judgment, LMJudge, Rubric
from wmo.common.judging.provenance import read_artifact_json
from wmo.common.models import (
    CompletionCostReservation,
    ModelCapabilities,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    completion_request_cost_usd,
    reconcile_completion_economics,
    verify_completion_reservation,
)
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.rollouts import RolloutArtifact
from wmo.optimize.router.errors import (
    JudgeDispatchExhaustedError,
    JudgeTranscriptAdmissionError,
)
from wmo.optimize.router.judging.contracts import ManualJudgeError, ManualJudgeSetupArtifact
from wmo.optimize.router.judging.protocol import TemplateJudgeClient
from wmo.runtime.models.providers.errors import ProviderRetryableResponseError
from wmo.simulation.engines.text.recording import Utf8UpperBoundTokenCounter


class ReservedJudgeClient:
    """Enforce one frozen request and full-call ceiling around a judge provider."""

    def __init__(
        self,
        client: ModelClient,
        *,
        reservation: CompletionCostReservation,
        model: ModelSnapshot,
        capabilities: ModelCapabilities,
        maximum_attempts: int,
        maximum_provider_calls: int,
    ) -> None:
        """Validate active economics before exposing the provider client.

        Args:
            client: Resolved judge provider client.
            reservation: Exact approved per-request price, retry, and token ceiling.
            model: Active exact judge model identity.
            capabilities: Active explicit catalog declaration.
            maximum_attempts: Active client retry ceiling.
            maximum_provider_calls: Full scalar or counterbalanced provider-call ceiling.

        Raises:
            ValueError: Pricing, model, retries, capacity, or call ceiling is invalid.
        """
        if maximum_provider_calls <= 0:
            raise ValueError("judge provider call ceiling must be positive")
        verify_completion_reservation(
            reservation,
            model=model,
            capabilities=capabilities,
            maximum_attempts=maximum_attempts,
        )
        self._client = client
        self._reservation = reservation
        self._maximum_provider_calls = maximum_provider_calls
        self._calls = 0
        self._counter = Utf8UpperBoundTokenCounter()

    @property
    def calls(self) -> int:
        """Return provider requests made through this reservation boundary."""
        return self._calls

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Preflight one exact request and reconcile bounded response economics.

        Args:
            request: Complete provider-neutral judge request.

        Returns:
            Provider response with known bounded economics.

        Raises:
            JudgeTranscriptAdmissionError: The counted request input exceeds the frozen
                reserved input-token ceiling, so the rendered transcript cannot be admitted.
            JudgeDispatchExhaustedError: The admitted dispatch exhausted its bounded retries
                without usable output; the error carries the conservative billed-spend ceiling.
            ValueError: The request exceeds a bound or provider usage and spend cannot be bounded.
        """
        if self._calls >= self._maximum_provider_calls:
            raise ValueError("judge provider call reservation is exhausted")
        input_tokens = self._counter.count(request)
        output_tokens = request.maximum_output_tokens
        if output_tokens is None:
            raise ValueError("judge request must declare a maximum output-token ceiling")
        if input_tokens > self._reservation.maximum_input_tokens:
            raise JudgeTranscriptAdmissionError(
                "judge request exceeds its reserved input-token ceiling"
            )
        if output_tokens > self._reservation.maximum_output_tokens:
            raise ValueError("judge request exceeds its reserved output-token ceiling")
        self._calls += 1
        try:
            response = self._client.complete(request)
        except ProviderRetryableResponseError as exc:
            raise JudgeDispatchExhaustedError(
                str(exc),
                conservative_cost_usd=completion_request_cost_usd(
                    self._reservation,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            ) from exc
        if response.model != self._reservation.model:
            raise ValueError("judge response model differs from its frozen reservation")
        economics = reconcile_completion_economics(
            self._reservation,
            response.economics,
        )
        return response.model_copy(update={"economics": economics})


class AutomaticRouterJudge:
    """Apply one approved setup to current-plan scalar or same-task pairwise evidence."""

    def __init__(
        self,
        client: ModelClient,
        setup: ManualJudgeSetupArtifact,
        *,
        created_at: datetime,
        code_revision: str,
    ) -> None:
        """Bind the finalized manual setup and provider boundary.

        Args:
            client: Reservation-enforcing provider client.
            setup: Finalized manual judge prompt, mapping, schema, and rubric pointer.
            created_at: Materialization time for provider probes and judgments.
            code_revision: Exact producer revision.
        """
        self._client = client
        self._setup = setup
        self._created_at = created_at
        self._code_revision = code_revision

    def judge_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: str,
        rubric_artifact_id: str,
        calibration_artifact_id: str,
    ) -> Judgment:
        """Judge one verified current-plan rollout under the saved executable contract.

        Args:
            store: Project-local artifact store.
            rollout_artifact_id: Target rollout artifact.
            rubric_artifact_id: Approved rubric artifact.
            calibration_artifact_id: Explicitly approved calibration artifact.

        Returns:
            Unwritten judgment accepted by the shared router composition boundary.

        Raises:
            ManualJudgeError: Plan, setup, rollout, or same-task pairwise evidence is invalid.
        """
        if rubric_artifact_id != self._setup.rubric.artifact_id:
            raise ManualJudgeError("router rubric differs from the finalized judge setup")
        rollout, rollout_input = read_rollout(store.artifacts, rollout_artifact_id)
        reference, reference_input = self._reference_rollout(store, rollout)
        rubric, _rubric_input = read_artifact_json(
            store,
            artifact_id=rubric_artifact_id,
            expected_artifact_type="rubric",
            relative_path="rubric.json",
            model_type=Rubric,
        )
        adapter = TemplateJudgeClient(
            self._client,
            self._setup.prompt_template,
            rollout,
            rubric,
            reference,
            store=store,
            setup_input=_setup_input(store, self._setup),
            rollout_input=rollout_input,
            reference_input=reference_input,
            created_at=self._created_at,
            code_revision=self._code_revision,
        )
        return LMJudge(
            adapter,
            self._setup.prompt_template.prompt,
            code_revision=self._code_revision,
            clock=lambda: self._created_at,
        ).judge_persisted(
            store,
            rollout_artifact_id=rollout_artifact_id,
            rubric_artifact_id=rubric_artifact_id,
            calibration_artifact_id=calibration_artifact_id,
        )

    def _reference_rollout(
        self,
        store: ProjectStore,
        target: RolloutArtifact,
    ) -> tuple[RolloutArtifact | None, ArtifactInput | None]:
        """Select a distinct same-task current-plan output for pairwise feedback.

        Args:
            store: Project-local artifact store.
            target: Verified target rollout.

        Returns:
            Reference rollout and exact input, or two absent values for non-pairwise feedback.

        Raises:
            ManualJudgeError: No distinct current-plan output exists for the same task.
        """
        if self._setup.prompt_template.response_shape != "pairwise":
            return None, None
        plan = self._target_plan(store, target)
        allowed_cells = {cell.cell_id for cell in plan.cells if cell.task_id == target.task_id}
        candidates = []
        for artifact_id in store.artifacts.list_ids():
            try:
                stored = store.artifacts.read(artifact_id)
                if stored.manifest.artifact_type != "rollout":
                    continue
                rollout, rollout_input = read_rollout(store.artifacts, artifact_id)
            except (OSError, ValueError):
                continue
            if (
                rollout.rollout_id != target.rollout_id
                and rollout.task_id == target.task_id
                and rollout.cell_id in allowed_cells
            ):
                candidates.append((rollout.rollout_id, rollout, rollout_input))
        if not candidates:
            raise ManualJudgeError(
                "pairwise router judging needs a distinct completed output for the same task"
            )
        _rollout_id, reference, reference_input = min(candidates, key=lambda item: item[0])
        return reference, reference_input

    def _target_plan(self, store: ProjectStore, target: RolloutArtifact) -> EvaluationPlan:
        """Load the exact frozen evaluation plan the target rollout was simulated under.

        Args:
            store: Project-local artifact store.
            target: Verified target rollout.

        Returns:
            Verified evaluation plan named by the rollout's immutable simulation binding.

        Raises:
            ManualJudgeError: The rollout carries no binding or the persisted plan drifted.
        """
        binding = target.simulation_binding
        if binding is None:
            raise ManualJudgeError(
                "pairwise router judging needs a rollout with a simulation cell binding"
            )
        plan, plan_input = read_evaluation_plan(
            store.artifacts, binding.evaluation_plan_input.artifact_id
        )
        if plan_input != binding.evaluation_plan_input:
            raise ManualJudgeError(
                "persisted evaluation plan differs from the rollout's immutable binding"
            )
        return plan


def _setup_input(store: ProjectStore, setup: ManualJudgeSetupArtifact) -> ArtifactInput:
    """Return the exact finalized setup pointer after identity verification.

    Args:
        store: Project-local artifact store.
        setup: Finalized setup envelope.

    Returns:
        Exact manifest pointer for the setup.

    Raises:
        ManualJudgeError: The setup envelope differs from its immutable artifact.
    """
    stored = store.artifacts.read(setup.setup_id)
    value = artifact_input(stored.manifest)
    if stored.manifest.artifact_type != "manual-judge-setup":
        raise ManualJudgeError("finalized judge setup has the wrong artifact type")
    persisted = ManualJudgeSetupArtifact.model_validate_json(
        store.artifacts.read_bytes(setup.setup_id, "setup.json")
    )
    if persisted != setup:
        raise ManualJudgeError("finalized judge setup differs from its immutable artifact")
    return value
