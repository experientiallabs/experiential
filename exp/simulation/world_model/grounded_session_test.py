"""The grounded transition protocol carries compact state edits and a complete profile."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
    ToolCall,
    load_model_catalog,
)
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.sessions.errors import StepRejectedError
from exp.runtime.sessions.store import SessionStore
from exp.simulation.rules.checker import TransitionVerdict, check_proposed_transition
from exp.simulation.rules.model import Predicate, Rule, RuleBook, Transition
from exp.simulation.world_model.durable import DurableWorldModelSession, SessionPolicy
from exp.simulation.world_model.grounded_session import (
    _GroundedJsonProposal,
    open_grounded_session,
)
from exp.simulation.world_model.runtime import load_grounded_world_model


class _UnusedModel:
    """Reject any provider execution while only opening sessions."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Opening a session must not invoke a model."""
        raise AssertionError("unexpected model call")


OBSERVATION = {"call_id": "c", "reply": {"id": "a"}, "encoding": "json"}


def test_grounded_proposals_carry_state_edits_not_full_state() -> None:
    """Re-writing the whole state each turn dominated output; only edits are accepted."""
    edits = {"kind": "edits", "edits": [{"op": "set", "path": ["orders", "a"], "value": 1}]}
    proposal = _GroundedJsonProposal.model_validate(
        {"observations": [OBSERVATION], "effect": edits}
    )
    assert proposal.next_state({"orders": {"b": 2}}) == {"orders": {"a": 1, "b": 2}}
    with pytest.raises(ValidationError):
        _GroundedJsonProposal.model_validate(
            {"observations": [OBSERVATION], "effect": {"kind": "replace", "state": {}}}
        )


def test_frozen_profile_binds_json_mode_and_served_model_pin(tmp_path: Path) -> None:
    """A calibration qualified under one generation configuration cannot cover another."""
    project, spec = source_project(tmp_path)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path))
    _, capabilities = catalog.snapshot("model")
    runtime = load_grounded_world_model(
        project.artifacts,
        spec.world_model.artifact_id,
        client=_UnusedModel(),
        capabilities=capabilities.model_copy(update={"supports_structured_output": True}),
    )
    policy = SessionPolicy(
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        maximum_operations=1,
        inference_identity="0" * 64,
        code_revision="fixture",
    )

    def unused(body: JsonObject, identity: str) -> JsonObject:
        """Opening a session must not complete a transition."""
        raise AssertionError("unexpected completion")

    def profile(
        name: str, *, json_object_output: bool = False, served_model_id: str | None = None
    ) -> object:
        """Open one session and return the profile its evidence would carry."""
        session = open_grounded_session(
            runtime=runtime,
            task=spec.task,
            store=SessionStore(tmp_path / f"{name}.sqlite"),
            artifacts=project.artifacts,
            session_id=name,
            policy=policy,
            complete=unused,
            lock_path=tmp_path / f"{name}.lock",
            maximum_output_tokens=4096,
            json_object_output=json_object_output,
            served_model_id=served_model_id,
        )
        sources = session.evaluation_context["sources"]
        assert isinstance(sources, dict)
        return sources["frozen_profile_sha256"]

    baseline = profile("baseline")
    assert profile("repeat") == baseline
    assert profile("json", json_object_output=True) != baseline
    assert profile("pinned", served_model_id="served-alias") != baseline


BOOK = RuleBook(
    contracts={},
    rules=(
        Rule(
            rule_id="inspect-sets-one",
            tool="inspect",
            kind="effect",
            statement="Inspecting sets the count to one.",
            predicate=Predicate.model_validate(
                {
                    "op": "eq",
                    "left": {"scope": "after", "path": "/count"},
                    "right": {"value": 1},
                }
            ),
            status="supported",
            support=2,
            contradiction=0,
        ),
    ),
)


def _checked_session(
    tmp_path: Path,
    name: str,
    counts: list[int],
    *,
    maximum_reasks: int,
    checker: Callable[[Transition, bool], TransitionVerdict] | None,
) -> tuple[DurableWorldModelSession, list[JsonObject]]:
    """Open a typed grounded session whose scripted world proposes the given counts in order.

    Args:
        tmp_path: Test directory.
        name: Session and file name.
        counts: Values each successive proposal sets ``count`` to.
        maximum_reasks: Re-asks allowed after a refused proposal.
        checker: Optional rule checker installed on the session's commit path.

    Returns:
        The session and the list that collects every request body sent.
    """
    project, spec = source_project(tmp_path / name)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path))
    _, capabilities = catalog.snapshot("model")
    runtime = load_grounded_world_model(
        project.artifacts,
        spec.world_model.artifact_id,
        client=_UnusedModel(),
        capabilities=capabilities,
    )
    bodies: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return the next scripted typed proposal through the actual commit owner."""
        del identity
        bodies.append(body)
        proposal: JsonObject = {
            "observations": [{"call_id": "call", "reply": "ok", "encoding": "json"}],
            "effect": {
                "kind": "edits",
                "edits": [{"op": "set", "path": ["count"], "value": counts[len(bodies) - 1]}],
            },
        }
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(
                    ToolCall(call_id="proposal", name="submit_transition", arguments=proposal),
                )
            ),
            model=runtime.artifact.model,
            economics=OperationEconomics(),
        ).model_dump(mode="json")

    session = open_grounded_session(
        runtime=runtime,
        task=spec.task,
        store=SessionStore(tmp_path / f"{name}.sqlite"),
        artifacts=project.artifacts,
        session_id=name,
        policy=SessionPolicy(
            deadline=datetime.now(UTC) + timedelta(minutes=5),
            maximum_operations=4,
            maximum_reasks=maximum_reasks,
            inference_identity="0" * 64,
            code_revision="fixture",
        ),
        complete=complete,
        lock_path=tmp_path / f"{name}.lock",
        maximum_output_tokens=4096,
        output_protocol="json-tool",
    )
    # The pinned grounded opener does not forward the hook, so it is installed directly.
    session._transition_checker = checker
    return session, bodies


INSPECT = ToolCall(call_id="call", name="inspect", arguments={"key": "current"})


def _check(transition: Transition, refused: bool) -> TransitionVerdict:
    """Bind the fixture rule book to the session's checker signature."""
    return check_proposed_transition(BOOK, transition, refused=refused)


def test_rule_violating_proposal_is_reasked_and_compliant_one_commits(tmp_path: Path) -> None:
    """A broken effect rule refuses the proposal; the re-ask carries why; only the fix commits."""
    session, bodies = _checked_session(
        tmp_path, "checked", [5, 1], maximum_reasks=1, checker=_check
    )
    session.step((INSPECT,), operation_id="0")
    assert session.state == {"count": 1} and len(bodies) == 2
    messages = bodies[1]["messages"]
    assert isinstance(messages, list) and isinstance(messages[-1], dict)
    assert "inspect-sets-one" in str(messages[-1]["content"])
    (committed,) = session.transitions
    evidence = committed["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["transition_verdict"] == TransitionVerdict(allowed=True).model_dump(mode="json")
    refused = evidence["refused_proposals"]
    assert isinstance(refused, list) and len(refused) == 1


def test_rule_violating_proposal_fails_the_step_when_reasks_run_out(tmp_path: Path) -> None:
    """With no re-ask left the step fails like any refused proposal and nothing commits."""
    session, bodies = _checked_session(tmp_path, "exhausted", [5], maximum_reasks=0, checker=_check)
    with pytest.raises(ValueError, match="breaks environment rules"):
        session.step((INSPECT,), operation_id="0")
    assert session.state == {} and session.transitions == () and len(bodies) == 1
    with pytest.raises(StepRejectedError):
        session.step((INSPECT,), operation_id="0")


def test_without_a_checker_the_same_proposal_commits_unchanged(tmp_path: Path) -> None:
    """The default commit path neither checks rules nor adds verdict evidence."""
    session, bodies = _checked_session(tmp_path, "default", [5], maximum_reasks=0, checker=None)
    session.step((INSPECT,), operation_id="0")
    assert session.state == {"count": 5} and len(bodies) == 1
    (committed,) = session.transitions
    evidence = committed["evidence"]
    assert isinstance(evidence, dict) and "transition_verdict" not in evidence


def test_checked_sessions_refuse_action_batches_before_dispatch(tmp_path: Path) -> None:
    """A rule transition has one action, so a checked batch is refused before any call."""
    session, bodies = _checked_session(tmp_path, "batch", [1], maximum_reasks=0, checker=_check)
    second = INSPECT.model_copy(update={"call_id": "other"})
    with pytest.raises(ValueError, match="exactly one action"):
        session.step((INSPECT, second), operation_id="0")
    assert bodies == [] and session.transitions == ()
