"""Fixed-workflow evaluation: infrastructure failures stop unscored, simulator failures score."""

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    JudgeNotRun,
    ReferenceStep,
    default_judge,
)
from exp.common.models import ToolCall
from exp.common.models.client import RequestAdmissionRefused
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.backpressure import (
    REQUEST_REFUSAL_CODES,
    ProviderAdmissionRefused,
    is_overload,
)
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.models.structured_completion import CompletionNotDispatched
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.evaluation.workflow import (
    DETERMINISTIC_REFUSALS,
    INFRASTRUCTURE_STOPS,
    FixedWorkflow,
    deterministic_refusal,
    evaluate_workflow,
)
from exp.simulation.world_model.durable import DurableWorldModelSession

NOW = datetime(2030, 1, 1, tzinfo=UTC)
ACTIONS = (
    ToolCall(call_id="a", name="read", arguments={"id": 1}),
    ToolCall(call_id="b", name="read", arguments={"id": 2}),
)


class FailingSession:
    """A fresh session whose every step raises one chosen error.

    Attributes:
        error: Raised by every step.
        transitions: Committed transitions; always empty.
        state: Current business state.
        evaluation_context: Evaluation context; always empty.
    """

    def __init__(self, error: BaseException) -> None:
        """Arm the error every step raises."""
        self.error = error
        self.transitions: tuple[JsonObject, ...] = ()
        self.state: JsonObject = {"items": []}
        self.evaluation_context: JsonObject = {}

    def execute(self, action: ToolCall) -> None:
        """Fail the step."""
        raise self.error


def evaluate(project: ProjectStore, error: BaseException) -> DualReport:
    """Evaluate the two-step workflow on a session whose steps fail with ``error``."""

    def unjudged(body: JsonObject, logical_id: str) -> JsonObject:
        """Skip the judge."""
        raise JudgeNotRun("not judged")

    return evaluate_workflow(
        cast(DurableWorldModelSession, FailingSession(error)),
        FixedWorkflow(id="workflow", task="Read two records", actions=ACTIONS),
        references=EvaluationReferences(
            capability="recorded_reference_only",
            steps=tuple(
                ReferenceStep(
                    step=index,
                    action={"name": action.name, "arguments": action.arguments},
                    reply_available=True,
                    reply={"id": index},
                )
                for index, action in enumerate(ACTIONS)
            ),
        ),
        judge=default_judge("fixture"),
        judge_complete=unjudged,
        artifacts=project.artifacts,
        clock=lambda: NOW,
        code_revision="fixture",
        evaluation_id="workflow-report",
    )


@pytest.fixture(name="project")
def _project(tmp_path: Path) -> ProjectStore:
    """An empty initialized project."""
    project = ProjectStore(tmp_path / "project", "workflow")
    project.initialize(ProjectConfig(project_id="workflow"))
    return project


@pytest.mark.parametrize(
    "error",
    [
        ProviderTransportError("connection reset"),
        ProviderDeadlineExceeded("request deadline"),
        TimeoutError("transport deadline reached before physical dispatch"),
        CompletionNotDispatched("refused before dispatch"),
        DispatchOutcomeUnknown("earlier dispatch saved no response"),
        DispatchCapExhaustedError("physical attempt cap reached"),
        RequestAdmissionRefused("dispatch guard refused the slot claim"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_an_infrastructure_failure_stops_without_a_scored_report(
    project: ProjectStore, error: BaseException
) -> None:
    """The run stays incomplete instead of scoring the failed step and its suffix as misses."""
    with pytest.raises(type(error)):
        evaluate(project, error)
    assert not project.artifacts.exists("workflow-report")


def test_a_simulator_failure_scores_the_failed_step_and_blocks_the_rest(
    project: ProjectStore,
) -> None:
    """Behavior of the simulator itself, such as a refused proposal, stays in the denominator."""
    report = evaluate(project, ValueError("proposal refused"))
    rows = report.mechanical["steps"]
    assert isinstance(rows, list)
    assert [row["status"] for row in rows if isinstance(row, dict)] == ["failure", "blocked"]
    assert [turn.fields_matched for turn in report.turns] == [0, 0]
    assert project.artifacts.exists("workflow-report")


def test_only_an_identified_registered_refusal_is_deterministic() -> None:
    """Outages, timeouts, unknown outcomes and unidentified requests are never deterministic."""
    refusal = ProviderTransportError(
        "provider returned HTTP 503",
        status_code=503,
        error_code="invalid_provider_stream",
        request_sha256="a" * 64,
    )
    assert deterministic_refusal(refusal) is refusal
    for error in (
        ProviderTransportError("connection reset"),
        ProviderTransportError("busy", status_code=503, error_code="gateway_overloaded"),
        ProviderTransportError(
            "unidentified", status_code=503, error_code="invalid_provider_stream"
        ),
        ProviderDeadlineExceeded("request deadline"),
        TimeoutError("transport deadline"),
        DispatchOutcomeUnknown("earlier dispatch saved no response"),
    ):
        assert deterministic_refusal(error) is None


def test_deterministic_refusals_never_open_a_provider_breaker() -> None:
    """A retirable per-request 503 is definite, so it is never counted as provider overload."""
    assert DETERMINISTIC_REFUSALS <= REQUEST_REFUSAL_CODES
    assert all(not is_overload(503, code) for code in DETERMINISTIC_REFUSALS)
    assert is_overload(503, "gateway_overloaded") and is_overload(429)


def test_a_host_admission_refusal_is_an_infrastructure_stop() -> None:
    """An open breaker or a slot wait stops the run resumably instead of scoring a miss."""
    refusal = ProviderAdmissionRefused("open", provider="openrouter", reason="circuit_open")
    assert isinstance(refusal, INFRASTRUCTURE_STOPS)
