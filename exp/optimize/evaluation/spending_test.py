"""Pinned served identities and durable pairwise probe coordinates survive budget pauses."""

from pathlib import Path

import pytest

from exp.common.judging import Rubric
from exp.common.judging.provenance import read_artifact_json
from exp.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    Usage,
    completion_cost_reservation,
)
from exp.common.project import artifact_input
from exp.common.rollouts import RolloutArtifact
from exp.optimize.evaluation.spending import BudgetedCompletion
from exp.optimize.router.composition_test import _completion_reservation
from exp.optimize.router.judging.artifacts import write_production_rollout
from exp.optimize.router.judging.protocol import TemplateJudgeClient
from exp.optimize.router.judging.service import (
    commit_manual_judge_setup,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from exp.optimize.router.judging.service_test import (
    _TIME,
    _built_store,
    _catalog,
    _template,
    _wide_axes,
)
from exp.runtime.models.budget import RequestBudget, SpendLimitReached


class _Client:
    """Return a counted successful provider response with deterministic one-attempt usage."""

    def __init__(self, model: ModelSnapshot) -> None:
        """Bind the exact reported provider identity."""
        self.model = model
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return valid pairwise feedback, charging the full requested output for this fixture."""
        self.calls += 1
        return ModelResponse(
            model=self.model,
            output=AssistantAction(
                content='{"dimensions":[{"dimension_id":"task-success","winner":"tie"}]}'
            ),
            economics=OperationEconomics(
                usage=Usage(input_tokens=1, output_tokens=request.maximum_output_tokens or 1),
                provider_attempts=1,
            ),
        )


@pytest.mark.parametrize("drift", [None, "model_id", "provider", "connection_sha256"])
def test_budgeted_completion_accepts_only_the_configured_served_identity(
    tmp_path: Path, drift: str | None
) -> None:
    """An explicit served alias retains all other frozen identity pins and replays unchanged."""
    reservation = _completion_reservation("candidate")
    served = reservation.model.model_copy(update={"model_id": "served-name"})
    if drift is not None:
        served = served.model_copy(update={drift: "c" * 64 if "sha256" in drift else "other"})
    client = _Client(served)
    budget = RequestBudget(tmp_path, identity="served-pin", maximum_cost_usd=100)
    wrapper = BudgetedCompletion(
        client, budget, reservation, role="assistant", served_model_id="served-name"
    )
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hello"),), maximum_output_tokens=1
    )
    if drift is not None:
        with budget.scope("cell"), pytest.raises(ValueError, match="identity"):
            wrapper.complete(request)
        return
    with budget.scope("cell"):
        first = wrapper.complete(request)
    with budget.scope("cell"):
        assert wrapper.complete(request) == first
    assert first.model == served
    assert client.calls == 1


def test_pairwise_budget_resume_preserves_reverse_probe_identity(tmp_path: Path) -> None:
    """A saved forward probe does not shift the reverse request's durable budget coordinate."""
    store = _built_store(tmp_path)
    setup = commit_manual_judge_setup(
        store,
        prepare_manual_judge_setup(
            store,
            _catalog(),
            dimensions=_wide_axes(),
            prompt_template=_template("pairwise"),
            created_at=_TIME,
            code_revision="test-revision",
        ),
        confirmed=True,
    )
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    reference = plan.reference_traces[0]
    assert reference is not None
    pointers = tuple(
        write_production_rollout(store, setup, plan.tasks[0], trace, _TIME, "test-revision")
        for trace in (plan.traces[0], reference)
    )
    rollouts = tuple(
        read_artifact_json(
            store,
            artifact_id=pointer.artifact_id,
            expected_artifact_type="rollout",
            relative_path="rollout.json",
            model_type=RolloutArtifact,
        )[0]
        for pointer in pointers
    )
    rubric, _ = read_artifact_json(
        store,
        artifact_id=setup.rubric.artifact_id,
        expected_artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    reservation = completion_cost_reservation(
        model=setup.judge_model,
        input_usd_per_million_tokens=0,
        output_usd_per_million_tokens=1,
        cached_input_usd_per_million_tokens=0,
        cache_write_usd_per_million_tokens=0,
        maximum_attempts=1,
        maximum_input_tokens=1_000_000,
        maximum_output_tokens=1_000,
    )
    client = _Client(setup.judge_model)
    request = ModelRequest(
        messages=(
            ModelMessage(
                role="system",
                content=setup.prompt_template.prompt.text,
            ),
        )
    )

    def execute(limit: float) -> ModelResponse:
        """Reconstruct both adapters as a fresh process would after increasing its allowance."""
        budget = RequestBudget(tmp_path / "budget", identity="pairwise", maximum_cost_usd=limit)
        adapter = TemplateJudgeClient(
            BudgetedCompletion(client, budget, reservation, role="judge"),
            setup.prompt_template,
            rollouts[0],
            rubric,
            rollouts[1],
            store=store,
            setup_input=artifact_input(store.artifacts.read(setup.setup_id).manifest),
            rollout_input=pointers[0],
            reference_input=pointers[1],
            created_at=_TIME,
            code_revision="test-revision",
            maximum_output_tokens=1_000,
            request_scope=budget.scope,
        )
        return adapter.complete(request)

    with pytest.raises(SpendLimitReached):
        execute(0.001)
    assert client.calls == 1
    result = execute(0.002)
    assert client.calls == 2
    assert execute(0.002) == result
    assert client.calls == 2
