"""Offline regressions for request-capped rule agents and exact persisted replay."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import AgentProfile, AgentTranscript, run_agent
from exp.optimize.evaluation.variance.evidence_tools import EvidenceBudget
from exp.optimize.evaluation.variance.evidence_tools_test import _ANSWER, _loop_fixture, _reply
from exp.runtime.models.spend_ceiling import (
    OperationSpendCeiling,
    UnpricedDispatchRefused,
    install_spend_ceiling,
    operation_spend_scope,
)


def _run(
    project: ProjectStore, settings: AuthorSettings, profile: AgentProfile
) -> tuple[ArtifactInput, AgentTranscript]:
    """Run a rules role over a tiny opening without any evidence reads."""
    return run_agent(
        project,
        settings,
        profile,
        identity="rules-draw",
        role="research",
        variant="contract",
        instruction="Read and answer.",
        opening={"task": "rules"},
        tools=(),
        runner=lambda call: ("refused", "no evidence", ()),
        answer=_ANSWER,
        check=lambda draft, served: None,
        sources=(),
        code_revision="test",
    )


@pytest.mark.parametrize("request_capped", [False, True])
def test_rules_agent_unknown_prices_need_explicit_request_capped_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_capped: bool
) -> None:
    """Only the explicit mode completes all four phases under the operation allowance."""
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [
            _reply([("record_plan", {"hypotheses": [], "planned_reads": []})]),
            _reply([("submit_draft", {"value": 2})]),
            _reply([("record_self_check", {"issues": [], "revise": False})]),
            _reply([("submit", {"value": 2})]),
        ],
        priced=False,
    )
    profile = AgentProfile(
        budget=EvidenceBudget(
            model_calls=4, tool_calls=1, maximum_usd=None if request_capped else 5.0
        )
    )
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=1.0, unpriced_attempts=4)
    if request_capped:
        with pytest.raises(UnpricedDispatchRefused, match="active finite"):
            _run(project, settings, profile)
        assert transport.requests == []
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        reference, transcript = _run(project, settings, profile)
    if request_capped:
        assert transcript.stop == "answered" and transcript.answer == {"value": 2}
        assert [call.phase for call in transcript.calls] == [
            "plan",
            "investigate",
            "self_check",
            "final",
        ]
        assert transcript.usage.unpriced_calls == 4
        assert transcript.profile.budget.maximum_usd is None
        assert all("request-capped" in call.response.artifact_id for call in transcript.calls)
        assert len(transport.requests) == 4 and ceiling.summary().unpriced_attempts == 4
    else:
        assert transcript.stop == "budget_exhausted"
        assert transcript.exhausted == ("maximum_usd_unverifiable",)
        assert len(transport.requests) == 1 and ceiling.summary().unpriced_attempts == 1
    assert transcript.usage.priced_usd == 0.0
    assert _run(project, settings, profile) == (reference, transcript)
    assert len(transport.requests) == (4 if request_capped else 1)
