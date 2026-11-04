"""Tests for selector and critic validation, variants and agentic isolation from the actor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import ToolChoice
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec, run_actor_critic
from exp.optimize.evaluation.variance.actor_critic_test import (
    _done,
    _edit,
    _estimates,
    _exposure,
    _point,
    _settings,
    _spec,
    _tool_reply,
    _wire,
    _with_usage,
)
from exp.optimize.evaluation.variance.evidence import EstimationPool, EvidencePacket
from exp.optimize.evaluation.variance.evidence_tools import (
    AnalysisTranscript,
    EvidenceBudget,
    EvidenceToolProfile,
)
from exp.optimize.evaluation.variance.evidence_tools_test import price_judge
from exp.optimize.evaluation.variance.role_prompts import assign_prompts
from exp.optimize.evaluation.variance.roles import (
    critic_request,
    judge_request,
    role_protocol_sha256,
    validate_critic,
    validate_judge,
)

_REF = ArtifactInput(artifact_id="checkpoint-a", sha256="a" * 64)
_PACKET = EvidencePacket(
    pool=EstimationPool(branch=1, points={0: _REF, 2: _REF}),
    payload={"group": {"success_rate": 0.5}},
    refs=frozenset({"group", "verifier", "attempt:0", "attempt:1", "point:0", "point:2"}),
)


def _packet_point(step: int, refs: list[str]) -> JsonObject:
    """One estimation point."""
    return {
        "step": step,
        "decision": "d",
        "alternatives": ["a"],
        "deciding_state": [],
        "rationale": "r",
        "evidence_refs": refs,
    }


def test_requests_force_one_typed_tool_and_carry_the_packet() -> None:
    """Both roles get the packet and must answer through their own tool."""
    judge = judge_request(_PACKET, points=2, output_tokens=512)
    critic = critic_request(_PACKET, (0, 2), output_tokens=512)

    assert judge.tool_choice == ToolChoice(name="choose_points")
    assert critic.tool_choice == ToolChoice(name="estimate_points")
    assert '"maximum_points": 2' in str(judge.messages[1].content)
    assert '"chosen_points": [0, 2]' in str(critic.messages[1].content)
    assert len(role_protocol_sha256()) == 64


def test_a_judge_answer_must_stay_inside_the_packet() -> None:
    """Unoffered points, invented locators and too many points are refused."""
    ok: JsonObject = {"points": [_packet_point(2, ["point:2"])], "limitations": ["none"]}
    assert validate_judge(ok, _PACKET, points=1).points[0].step == 2

    with pytest.raises(ValueError, match="not offered"):
        validate_judge(
            {"points": [_packet_point(1, ["group"])], "limitations": ["x"]},
            _PACKET,
            points=2,
        )
    with pytest.raises(ValueError, match="not supplied"):
        validate_judge(
            {"points": [_packet_point(0, ["attempt:7"])], "limitations": ["x"]},
            _PACKET,
            points=2,
        )
    two: JsonObject = {
        "points": [_packet_point(0, ["group"]), _packet_point(2, ["group"])],
        "limitations": ["x"],
    }
    with pytest.raises(ValueError, match="at most 1"):
        validate_judge(two, _PACKET, points=1)


def test_a_critic_must_estimate_exactly_the_chosen_points() -> None:
    """A missing or extra estimate is refused rather than scored."""
    estimate: JsonObject = {
        "step": 0,
        "probability": 0.4,
        "rationale": "r",
        "evidence_refs": ["group"],
    }
    answer: JsonObject = {"estimates": [estimate], "limitations": ["none"]}

    with pytest.raises(ValueError, match="not exactly"):
        validate_critic(answer, _PACKET, (0, 2))
    assert validate_critic(answer, _PACKET, (0,)).estimates[0].step == 0


def test_agentic_selectors_record_their_variant_and_never_reach_the_actor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agentic selector plans, reads, drafts and checks; none of it reaches an actor request."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    digest = price_judge(project)
    job = job.model_copy(update={"catalog_sha256": digest})
    settings = _settings(project, job, digest)
    profile = EvidenceToolProfile(
        mode="agentic",
        budget=EvidenceBudget(model_calls=6, tool_calls=2),
        maximum_result_bytes=256_000,
    )
    spec = _spec(project, job).model_copy(
        update={"judges": (settings,), "critics": (settings,), "selector_profile": profile}
    )
    spec = ActorCriticSpec.model_validate(spec.model_dump())
    marker = "PRIVATE-PLAN-MARKER"
    answer: JsonObject = {"points": [_point(0), _point(1)], "limitations": ["x"]}
    roles = [
        _tool_reply("record_plan", {"hypotheses": [marker], "planned_reads": ["environment"]}),
        _tool_reply("read_world_state", {"point": 1, "path": ""}),
        _tool_reply("submit_draft", answer),
        _tool_reply("record_self_check", {"issues": [], "revise": False}),
        _tool_reply("choose_points", answer),
        _tool_reply("estimate_points", _estimates([0, 1])),
    ]
    roles = [_with_usage(reply, 500, 50) for reply in roles]
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    actor_transport, role_transport = _wire(monkeypatch, group + continuations, roles)

    report = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )

    selector = report.judges[0]
    assert selector.selection is not None and selector.transcript is not None
    transcript = AnalysisTranscript.model_validate_json(
        project.artifacts.read_bytes(selector.transcript.artifact_id, "transcript.json")
    )
    expected = assign_prompts(spec.selector_prompts, seed=spec.seed, draws=1)[0]
    assert transcript.prompt == expected
    assert [row.phase for row in transcript.outputs] == ["plan", "draft", "self_check"]
    assert transcript.outputs[1].host_validation == "valid"
    assert [(t.name, t.status) for t in transcript.tools] == [("read_world_state", "ok")]
    assert f"STANCE ({expected.variant})" in json.dumps(role_transport.requests[0].payload)
    for request in actor_transport.requests:
        sent = json.dumps(request.payload)
        assert marker not in sent and "STANCE" not in sent and "record_plan" not in sent
        assert "world_learning_retrieval" not in sent and expected.stance[:40] not in sent
