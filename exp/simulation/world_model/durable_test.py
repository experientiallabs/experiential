"""Public reconstruction, guarded commits, information masks and dual evaluation."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.evaluations.reconstruction import EvaluationReferences, ReferenceStep, default_judge
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore, ProjectConfig, ProjectStore
from exp.runtime.sessions.store import SessionStore, StepDeadlineExceededError, StepRejectedError
from exp.simulation.engines.reconstructed import InformationChannels
from exp.simulation.evaluation import FixedWorkflow, evaluate_workflow
from exp.simulation.reconstruction import RawEpisode, RawInputs, initialize, reconstruct
from exp.simulation.world_model.durable import (
    DurableWorldModelSession,
    SessionPolicy,
    open_reconstructed_session,
)


class _ConstructionArgs(TypedDict):
    """The concrete shared arguments used by both construction phases."""

    artifacts: ArtifactStore
    run_id: str
    clock: Callable[[], datetime]
    deadline: datetime
    code_revision: str


NOW = datetime(2030, 1, 1, tzinfo=UTC)


def proposed() -> JsonObject:
    """Declare a synthetic append interface with one preserved unrelated field."""
    return {
        "interface": {
            "tools": [
                {
                    "name": "append",
                    "description": "Append the supplied item",
                    "parameters": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                        "required": ["item"],
                        "additionalProperties": False,
                    },
                    "reply_schema": {"type": "string"},
                }
            ],
            "state_schema": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                    "owner": {"type": "string"},
                },
                "required": ["items", "owner"],
                "additionalProperties": False,
            },
        },
        "claims": [
            {
                "statement": "Synthetic append hypothesis",
                "status": "inferred",
                "scope": "append",
                "origin": "model_prior",
            }
        ],
        "text_rules": "Append the current item; retain prior items and owner.",
    }


def setup(
    tmp_path: Path, *, channels: InformationChannels | None = None
) -> tuple[DurableWorldModelSession, ProjectStore, list[JsonObject], list[datetime]]:
    """Exercise real construction and initialization with deterministic model proposals."""
    project = ProjectStore(tmp_path / "project", "reconstruction")
    project.initialize(ProjectConfig(project_id="reconstruction"))
    time = [NOW]
    kwargs: _ConstructionArgs = {
        "artifacts": project.artifacts,
        "run_id": "construct",
        "clock": lambda: time[0],
        "deadline": NOW + timedelta(minutes=5),
        "code_revision": "offline-integration",
    }
    raw = RawInputs(
        episodes=(RawEpisode(id="observed", records=({"action": "append", "reply": "ok"},)),)
    )
    bundle = reconstruct(raw, complete=lambda *_: proposed(), **kwargs)
    prefix: tuple[JsonObject, ...] = ({"reply": {"items": ["existing"], "owner": "current-user"}},)
    initializer = initialize(
        bundle,
        prefix,
        complete=lambda *_: {
            "state": {"items": ["existing"], "owner": "current-user"},
            "complete_scopes": ["/items", "/owner"],
            "field_sources": {
                "/items": [{"source_id": "public_prefix", "pointer": "/0/reply/items"}],
                "/owner": [{"source_id": "public_prefix", "pointer": "/0/reply/owner"}],
            },
            "field_statuses": {"/items": "observed", "/owner": "observed"},
        },
        **kwargs,
    )
    requests: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return a proposal based on actual current action, never a hidden reference."""
        del identity
        requests.append(body)
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payload = json.loads(str(messages[1]["content"]))
        action = payload["action"]
        return {
            "observations": [{"call_id": action["call_id"], "reply": "ok", "encoding": "json"}],
            "effect": {
                "kind": "edits",
                "edits": [
                    {"op": "append", "path": ["items"], "value": action["arguments"]["item"]}
                ],
            },
            "terminal": False,
        }

    session = open_reconstructed_session(
        bundle=bundle,
        initializer=initializer,
        store=SessionStore(tmp_path / "world.sqlite", clock=lambda: time[0]),
        artifacts=project.artifacts,
        session_id="world",
        policy=SessionPolicy(
            deadline=NOW + timedelta(minutes=2),
            maximum_operations=30,
            inference_identity=sha256_json({"model": "scripted"}),
            code_revision="offline-integration",
        ),
        complete=complete,
        lock_path=tmp_path / "world.lock",
        clock=lambda: time[0],
        channels=channels,
    )
    return session, project, requests, time


def action(index: int = 0) -> ToolCall:
    """Return distinct logical actions even when a value is intentionally repeated."""
    return ToolCall(call_id=f"call-{index}", name="append", arguments={"item": f"item-{index}"})


def test_public_carried_workflow_commits_24_steps_and_required_dual_status(tmp_path: Path) -> None:
    """Drive the public composition with actual SQLite commits and independent fixture checks."""
    session, project, requests, _ = setup(tmp_path)
    actions = tuple(action(i) for i in range(24))
    refs = EvaluationReferences(
        capability="recorded_reference_only",
        description="Synthetic wiring, not native accuracy",
        steps=tuple(
            ReferenceStep(
                step=i,
                action={"name": a.name, "arguments": a.arguments},
                before={
                    "items": ["existing", *(f"item-{j}" for j in range(i))],
                    "owner": "current-user",
                },
                reply_available=True,
                reply="ok",
                state_available=True,
                after={
                    "items": ["existing", *(f"item-{j}" for j in range(i + 1))],
                    "owner": "current-user",
                },
            )
            for i, a in enumerate(actions)
        ),
    )
    judge = default_judge("scripted-judge")

    def assess(body: JsonObject, identity: str) -> JsonObject:
        """Keep scripted judging unassessable rather than claiming scientific calibration."""
        del body, identity
        return {
            "axes": [
                {
                    "dimension_id": d.dimension_id,
                    "status": "unassessable",
                    "score": None,
                    "rationale": "Synthetic test only",
                    "evidence_refs": [],
                }
                for d in judge.definition.dimensions
            ]
        }

    result = evaluate_workflow(
        session,
        FixedWorkflow(id="carried-24", task="Append distinct items", actions=actions),
        references=refs,
        judge=judge,
        judge_complete=assess,
        artifacts=project.artifacts,
        clock=lambda: NOW,
        code_revision="offline-integration",
    )
    assert result.mechanical["workflow_joint_exact"] is True
    assert result.mechanical["horizon_joint_exact"] == {"6": True, "12": True, "24": True}
    assert result.judge_status == "completed"
    assert len(requests) == 24
    assert session.state["owner"] == "current-user"
    assert len(session.transitions) == 24
    assert project.artifacts.read("carried-24-evaluation")


@pytest.mark.parametrize(
    "rules,originals,history",
    [(True, True, True), (False, True, True), (True, False, True), (False, False, False)],
)
def test_masks_change_actual_request(
    tmp_path: Path, rules: bool, originals: bool, history: bool
) -> None:
    """Information removals alter predictor input rather than configuration labels only."""
    session, _, requests, _ = setup(
        tmp_path, channels=InformationChannels(rules=rules, originals=originals, history=history)
    )
    session.execute(action())
    messages = requests[0]["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    payload = json.loads(str(messages[1]["content"]))
    assert ("text_rules" in payload["knowledge"]) is rules
    assert bool(payload["selected_sources"]) is originals
    assert bool(payload["history"]) is history


def test_replay_uses_saved_observation_without_second_dispatch(tmp_path: Path) -> None:
    """A duplicate logical operation returns the original reply and state revision."""
    session, _, requests, _ = setup(tmp_path)
    result = session.execute(action())
    assert session.execute(action()) == result
    assert len(requests) == 1
    assert len(session.transitions) == 1


def test_late_validation_cannot_publish_even_after_provider_return(tmp_path: Path) -> None:
    """The transaction guards the deadline after all proposal parsing and validation."""
    session, _, requests, time = setup(tmp_path)
    original = session._validate

    def delayed_validation(state: JsonObject) -> None:
        """Inject validation work that ends exactly at the registered deadline."""
        original(state)
        time[0] = NOW + timedelta(minutes=2)

    session._validate = delayed_validation
    with pytest.raises(StepDeadlineExceededError):
        session.execute(action())
    assert len(requests) == 1
    assert session.state == {"items": ["existing"], "owner": "current-user"}
    with pytest.raises(StepRejectedError):
        session.execute(action())


def test_unsupported_batch_does_not_dispatch(tmp_path: Path) -> None:
    """Batch semantics are declared, never silently implemented as sequential calls."""
    session, _, requests, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        session.step((action(), action(1)), operation_id="batch")
    assert requests == []
    assert session.transitions == ()


def test_valid_wrong_effects_continue_in_candidate_owned_world(tmp_path: Path) -> None:
    """A semantic error remains measurable; the next action cannot receive native rescue."""
    session, project, requests, time = setup(tmp_path)
    original = session._complete

    def wrong_first(body: JsonObject, identity: str) -> JsonObject:
        """Inject a structurally valid owner error into the first proposal only."""
        proposal = original(body, identity)
        if len(requests) == 1:
            effect = proposal["effect"]
            assert isinstance(effect, dict) and isinstance(effect["edits"], list)
            effect["edits"].append({"op": "set", "path": ["owner"], "value": "wrong-owner"})
        return proposal

    session._complete = wrong_first
    actions = (action(0), action(1))
    refs = EvaluationReferences(
        capability="recorded_reference_only",
        steps=tuple(
            ReferenceStep(
                step=i,
                action={"name": a.name, "arguments": a.arguments},
                before={
                    "items": ["existing", *(f"item-{j}" for j in range(i))],
                    "owner": "current-user",
                },
                reply_available=True,
                reply="ok",
                state_available=True,
                after={
                    "items": ["existing", *(f"item-{j}" for j in range(i + 1))],
                    "owner": "current-user",
                },
            )
            for i, a in enumerate(actions)
        ),
    )
    report = evaluate_workflow(
        session,
        FixedWorkflow(id="wrong-carried", task="Append two values", actions=actions),
        references=refs,
        judge=default_judge("scripted"),
        judge_complete=_unassessable,
        artifacts=project.artifacts,
        clock=lambda: time[0],
        code_revision="fixture",
    )
    assert len(requests) == 2 and report.mechanical["valid_steps"] == 2
    assert report.mechanical["first_semantic_divergence"] == 0
    assert report.mechanical["workflow_joint_exact"] is False
    assert session.state["owner"] == "wrong-owner"
    messages = requests[1]["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    second = json.loads(str(messages[1]["content"]))
    assert second["state"]["owner"] == "wrong-owner"


def test_provider_unknown_is_not_redispatched_after_recovery(tmp_path: Path) -> None:
    """A potentially billed failure consumes its claim and cannot mutate the world."""
    from exp.runtime.sessions.store import StepAmbiguousError

    session, _, _, time = setup(tmp_path)
    calls: list[str] = []

    def failed(_: JsonObject, identity: str) -> JsonObject:
        """Fail after dispatch admission, with deliberately unresolved provider billing."""
        calls.append(identity)
        raise ConnectionError("provider outcome unknown")

    session._complete = failed
    before = session.state
    with pytest.raises(ConnectionError):
        session.execute(action())
    store = SessionStore(tmp_path / "world.sqlite", clock=lambda: time[0])
    assert store.recover().ambiguous == ()
    with pytest.raises(StepAmbiguousError):
        session.execute(action())
    assert len(calls) == 1 and session.state == before


def _unassessable(_: JsonObject, identity: str) -> JsonObject:
    """Return explicit unavailable ratings; synthetic fixtures do not establish calibration."""
    del identity
    return {
        "axes": [
            {
                "dimension_id": axis.dimension_id,
                "status": "unassessable",
                "score": None,
                "rationale": "Synthetic persistence fixture.",
                "evidence_refs": [],
                "critical_flags": [],
            }
            for axis in default_judge("scripted").definition.dimensions
        ]
    }


def test_advertised_proposal_schema_matches_reconstructed_protocol(tmp_path: Path) -> None:
    """The provider must not be offered effects or encodings the parser refuses."""
    from jsonschema import Draft202012Validator

    session, _, _, _ = setup(tmp_path)
    engine = session._engine
    action = ToolCall(call_id="wire", name="append", arguments={"item": "new"})
    prepared = engine.prepare(session.state, (), (action,))
    messages = prepared.body["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    content = messages[1]["content"]
    assert isinstance(content, str)
    payload = json.loads(content)
    validator = Draft202012Validator(payload["output_schema"])
    observations: list[JsonObject] = [{"call_id": "wire", "encoding": "json", "reply": "ok"}]
    proposal: JsonObject = {
        "observations": observations,
        "effect": {"kind": "edits", "edits": []},
        "terminal": False,
    }
    validator.validate(proposal)
    engine.parse(proposal, (action,))
    observations[0]["encoding"] = "text"
    assert not validator.is_valid(proposal)
    with pytest.raises(ValueError):
        engine.parse(proposal, (action,))
    observations[0]["encoding"] = "json"
    proposal["effect"] = {"kind": "replace", "state": {}}
    assert not validator.is_valid(proposal)
    with pytest.raises(ValueError):
        engine.parse(proposal, (action,))


def test_state_reference_returns_exact_current_value_and_rejects_stale_identity(
    tmp_path: Path,
) -> None:
    """Projection preserves identifiers without authorizing stale calls or missing values."""
    from exp.common.core.artifacts import sha256_json
    from exp.simulation.engines.reconstructed import ReconstructedTextEngine
    from exp.simulation.reconstruction.models import ReconstructionBundle

    raw = RawInputs(description="Read a record.")
    data = proposed()
    data["interface"] = {
        "tools": [
            {
                "name": "read",
                "description": "Read.",
                "parameters": {"type": "object"},
                "reply_schema": {"type": "object"},
            }
        ],
        "state_schema": {"type": "object"},
    }
    bundle = ReconstructionBundle.model_validate(
        {
            **data,
            "artifact_id": "projection",
            "sources": raw.model_dump(mode="json"),
            "inputs_sha256": sha256_json(raw.model_dump(mode="json")),
        }
    )
    engine = ReconstructedTextEngine(bundle)
    state: JsonObject = {
        "records": {"1": {"item_id": "5745575001", "state": "NJ"}, "11": {"item_id": "other"}}
    }
    call = ToolCall(call_id="current", name="read", arguments={})
    proposal: JsonObject = {
        "observations": [
            {
                "call_id": "current",
                "encoding": "json",
                "reply_from_state": {
                    "path": ["records", "1"],
                    "wrapper": {"ok": True},
                    "field": "result",
                },
            }
        ],
        "effect": {"kind": "edits", "edits": []},
        "terminal": False,
    }
    result = engine.parse(proposal, (call,), current_state=state)
    assert result.observations[0].reply == {"ok": True, "result": state["records"]["1"]}
    assert result.next_state(state) == state
    with pytest.raises(ValueError, match="identity"):
        engine.parse(proposal, (call.model_copy(update={"call_id": "next"}),), current_state=state)
    with pytest.raises(ValueError, match="resolve"):
        engine.parse(proposal, (call,), current_state={})


def test_prediction_receives_initial_coverage_without_promoting_later_guesses(
    tmp_path: Path,
) -> None:
    """Metadata reaches the predictor and remains explicitly about initialization."""
    session, _, requests, _ = setup(tmp_path)
    session.execute(action(0))
    session.execute(action(1))
    payloads = []
    for body in requests:
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payloads.append(json.loads(str(messages[1]["content"])))
    for payload in payloads:
        assert payload["initial_state_coverage"]["complete_scopes"] == ["/items", "/owner"]
        assert payload["initial_state_coverage"]["authority"] == "initial_public_observations_only"
    assert payloads[1]["state"]["items"] == ["existing", "item-0"]


def test_after_projection_returns_only_model_proposed_edits(tmp_path: Path) -> None:
    """Mutation replies follow candidate changes; no native state is consulted."""
    session, _, requests, _ = setup(tmp_path)
    original = session._complete

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Replace the literal reply by the proposed poststate owner field."""
        proposal = original(body, identity)
        proposal["observations"] = [
            {
                "call_id": "call-0",
                "encoding": "json",
                "reply_from_state": {"path": ["owner"], "view": "after"},
            }
        ]
        effect = proposal["effect"]
        assert isinstance(effect, dict) and isinstance(effect["edits"], list)
        effect["edits"].append({"op": "set", "path": ["owner"], "value": "model-chosen"})
        return proposal

    session._complete = complete
    observation = session.execute(action(0))
    assert json.loads(observation.content) == "model-chosen"
    assert session.state["owner"] == "model-chosen"
    assert session.transitions[-1]["replies"] == ["model-chosen"]
    assert len(requests) == 1


def test_local_preflight_refusal_precedes_provider_claim(tmp_path: Path) -> None:
    """Known non-dispatch is rejected, not made ambiguous or billed as a provider attempt."""
    from exp.common.models.client import RequestCapacityExceeded

    session, _, _, _ = setup(tmp_path)

    class Refused:
        """A completion owner whose exact local request cannot fit."""

        def preflight(self, body: JsonObject) -> None:
            """Establish non-dispatch before entering the external callback."""
            raise RequestCapacityExceeded("does not fit")

        def __call__(self, body: JsonObject, identity: str) -> JsonObject:
            """This callback must never run."""
            raise AssertionError("provider must not be called")

    session._complete = Refused()
    before = session.state
    with pytest.raises(RequestCapacityExceeded):
        session.execute(action())
    assert session.state == before
    assert session._store.open_steps(session.session_id) == ()
    with pytest.raises(StepRejectedError, match="preparation_refused"):
        session.execute(action())


def test_observation_codec_is_explicit_and_rejects_stringified_objects(tmp_path: Path) -> None:
    """A declared codec constrains representation, without converting a wrong proposal."""
    from jsonschema import ValidationError

    from exp.simulation.engines.reconstructed import ReconstructedTextEngine

    session, _, _, _ = setup(tmp_path)
    original = session._engine
    assert isinstance(original, ReconstructedTextEngine)
    engine = ReconstructedTextEngine(original.bundle, observation_schema={"type": "object"})
    call = action()
    proposal: JsonObject = {
        "observations": [{"call_id": call.call_id, "encoding": "json", "reply": '{"ok":false}'}],
        "effect": {"kind": "edits", "edits": []},
    }
    assert engine.identity != original.identity
    with pytest.raises(ValidationError):
        engine.parse(proposal, (call,), current_state=session.state)
