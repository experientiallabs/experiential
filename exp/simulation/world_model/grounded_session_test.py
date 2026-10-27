"""The grounded transition protocol carries compact state edits and a complete profile."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest, ModelResponse, load_model_catalog
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.sessions.store import SessionStore
from exp.simulation.world_model.durable import SessionPolicy
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
