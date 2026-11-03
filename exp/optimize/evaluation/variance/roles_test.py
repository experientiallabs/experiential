"""Tests for judge and critic answer validation against their evidence packet."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import ToolChoice
from exp.optimize.evaluation.variance.evidence import EstimationPool, EvidencePacket
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


def _point(step: int, refs: list[str]) -> JsonObject:
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
    ok: JsonObject = {"points": [_point(2, ["point:2"])], "limitations": ["none"]}
    assert validate_judge(ok, _PACKET, points=1).points[0].step == 2

    with pytest.raises(ValueError, match="not offered"):
        validate_judge(
            {"points": [_point(1, ["group"])], "limitations": ["x"]},
            _PACKET,
            points=2,
        )
    with pytest.raises(ValueError, match="not supplied"):
        validate_judge(
            {"points": [_point(0, ["attempt:7"])], "limitations": ["x"]},
            _PACKET,
            points=2,
        )
    two: JsonObject = {"points": [_point(0, ["group"]), _point(2, ["group"])], "limitations": ["x"]}
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
