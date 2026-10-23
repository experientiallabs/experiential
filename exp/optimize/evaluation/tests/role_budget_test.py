"""Prediction, investigation and judging carry separate allowances on the real wire."""

import json
from pathlib import Path

import pytest

from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    investigation,
    memory_project,
    predicted,
    trajectory,
)
from exp.runtime.models.providers.transport import JsonHttpResponse

ROLES = {
    "prediction": {"output_tokens": 1500, "reasoning_effort": "low"},
    "investigation": {"output_tokens": 700},
    "judging": {"output_tokens": 3000, "reasoning_effort": "high"},
}


def pin_reasoning(project: ProjectStore) -> None:
    """Declare that the fixture model accepts a pinned reasoning effort."""
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    assert record.capabilities is not None
    caps = record.capabilities.model_copy(
        update={"supports_reasoning": True, "reasoning_effort": "medium"}
    )
    models = {**catalog.models, "model": record.model_copy(update={"capabilities": caps})}
    write_model_catalog(project.model_catalog_path, catalog.model_copy(update={"models": models}))


def length_stop() -> JsonHttpResponse:
    """A provider reply that spent its allowance without producing the tool call."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": "fixture-v1",
            "choices": [
                {"finish_reason": "length", "message": {"role": "assistant", "content": None}}
            ],
        },
    )


def test_roles_get_their_own_allowance_and_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved requests show each role's limit; judge exhaustion stays an explicit failure."""
    project, base = memory_project(tmp_path)
    pin_reasoning(project)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    spec = FidelitySpec.model_validate(
        {**trajectory(base, "traj-a", "alpha", "learn").model_dump(mode="json"), "roles": ROLES}
    )
    wire = RoutedTransport(
        [
            predicted("traj-a-0", "wrong"),
            investigation("alpha"),
            predicted("traj-a-1", "alpha-reply-1"),
            *[length_stop() for _ in range(6)],
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    report = execute_fidelity_evaluation(project, prepared)
    assert report.mechanical["valid_steps"] == 2
    assert report.judge_status == "failed"
    payloads = [request.payload for request in wire.requests]
    by_tool = {
        json.dumps(p.get("tool_choice")): p for p in payloads if isinstance(p.get("tools"), list)
    }
    transition = next(p for p in payloads if "submit_transition" in json.dumps(p.get("tools")))
    assert transition["max_tokens"] == 1500 and transition["reasoning_effort"] == "low"
    research = [p for p in payloads if "Investigate one completed" in json.dumps(p["messages"])]
    assert research and research[0]["max_tokens"] == 700
    assert research[0]["reasoning_effort"] == "medium"
    judge = [p for p in payloads if "Evaluate the workflow" in json.dumps(p["messages"])]
    assert judge and all(p["max_tokens"] == 3000 and p["reasoning_effort"] == "high" for p in judge)
    assert by_tool
    plans = {
        json.loads(project.artifacts.read_bytes(i, "request.json"))["plan"]["output_tokens"]
        for i in project.artifacts.list_ids(artifact_type="structured-call-intent")
    }
    assert plans == {1500, 700, 3000}


def test_reasoning_is_refused_for_models_without_a_pinned_effort(tmp_path: Path) -> None:
    """A role cannot invent a reasoning setting the model's capabilities do not declare."""
    project, base = memory_project(tmp_path)
    spec = FidelitySpec.model_validate(
        {**trajectory(base, "traj-a", "alpha", "learn").model_dump(mode="json"), "roles": ROLES}
    )
    with pytest.raises(ValueError, match="does not accept a configured reasoning effort"):
        prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    too_big = {**ROLES, "judging": {"output_tokens": 999_999}}
    spec = FidelitySpec.model_validate(
        {**trajectory(base, "traj-b", "beta", "learn").model_dump(mode="json"), "roles": too_big}
    )
    pin_reasoning(project)
    with pytest.raises(ValueError, match="exceeds the model's maximum output"):
        prepare_fidelity_evaluation(project, spec, code_revision="fixture")
