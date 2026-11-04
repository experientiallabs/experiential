"""Tests for decision options derived at a screened point and its classified continuations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.decision_measure_test import (
    FORCED,
    choice,
    judged_study,
    reply,
)
from exp.optimize.evaluation.variance.evidence_tools_test import price_judge
from exp.optimize.evaluation.variance.point_options import (
    OPTIONS_TOOL,
    POINT_OPTIONS_TYPE,
    ContinuedPoint,
    PointOptionsSpec,
    check_options,
    derive_point_options,
)
from exp.optimize.evaluation.variance.rewards import (
    PointDecision,
    SelectorObjectiveV2,
    explained_variance,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def _options(*refs: tuple[str, str]) -> JsonObject:
    """A proposal naming one option per (option, cited locator) pair."""
    return {
        "decision": "Whether to refund the order.",
        "options": [
            {"option_id": option, "action": f"Call {option} on the order", "evidence_refs": [ref]}
            for option, ref in refs
        ],
    }


VALID = _options(("refund", "episode:0/event:e3"), ("decline", "episode:1/event:e3"))
"""A proposal citing the first consequential call of two continuations."""


def _proposal(arguments: JsonObject) -> JsonHttpResponse:
    """A forced proposer reply."""
    return reply([(OPTIONS_TOOL, arguments)])


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[JsonHttpResponse]
) -> tuple[ProjectStore, PointOptionsSpec, ContinuedPoint, ScriptedJsonTransport]:
    """A project, forced-profile settings, one continued point and the scripted transport."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    digest = price_judge(project)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    settings = AuthorSettings(
        model_alias="judge",
        model=catalog.snapshot("judge")[0],
        catalog_sha256=digest,
        output_tokens=1024,
        deadline=job.deadline,
    )
    transport = ScriptedJsonTransport(tuple(replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    judged = judged_study(
        project, "point", [("refund", "pass"), ("decline", "fail"), ("refund", "pass")]
    )
    point = ContinuedPoint(run="run", point=2, checkpoint=job.study.checkpoint, judged=judged)
    spec = PointOptionsSpec(proposer=settings, classifier=settings, profile=FORCED)
    return project, spec, point, transport


def test_proposals_are_reconciled_then_every_continuation_is_classified_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three stances propose (one cites an unknown event), a reconcile decides, choices follow."""
    replies = [
        _proposal(VALID),
        _proposal(_options(("refund", "episode:0/event:e9"), ("decline", "episode:1/event:e3"))),
        _proposal(VALID),
        _proposal(VALID),
        choice("refund"),
        choice("decline"),
        choice("refund"),
    ]
    project, spec, point, transport = _fixture(tmp_path, monkeypatch, replies)

    reference, record = derive_point_options(project, spec, point, code_revision="test")

    assert [draft.stance for draft in record.proposals] == list(spec.stances)
    assert "unknown locators" in str(record.proposals[1].failure)
    assert record.reconciled is not None and record.reconciled.stance == "reconcile"
    assert record.options is not None
    assert [option.option_id for option in record.options.options] == ["refund", "decline"]
    assert record.decision == PointDecision(
        point_id="point-2",
        choices=("refund", "decline", "refund"),
        rewards=(1.0, 0.0, 1.0),
    )
    objective = SelectorObjectiveV2(minimum_scored=3)
    value = explained_variance(record.decision, objective) if record.decision else None
    assert value is not None and value > 0
    stored = project.artifacts.read(reference.artifact_id).manifest
    assert stored.artifact_type == POINT_OPTIONS_TYPE
    reconcile = json.dumps(transport.requests[3].payload)
    assert "STANCE (reconcile)" in reconcile and "proposal:action_clustering" in reconcile
    first = json.dumps(transport.requests[0].payload)
    assert "episode:0/event:e3" in first and "STANCE (action_clustering)" in first

    again = derive_point_options(project, spec, point, code_revision="test", joint_score=0.25)
    assert again[0] == reference
    assert again[1].decision is not None and again[1].decision.joint_score == 0.25
    assert len(transport.requests) == len(replies)


def test_one_valid_proposal_stands_and_none_leaves_the_point_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single valid proposal needs no reconcile; with none the point keeps every failure."""
    single = ("action_clustering",)
    replies = [_proposal(VALID), choice("refund"), choice("refund"), choice("decline")]
    project, spec, point, transport = _fixture(tmp_path, monkeypatch, replies)
    spec = spec.model_copy(update={"stances": single})

    with pytest.raises(ValueError, match="stances"):
        spec.model_validate({**spec.model_dump(mode="json"), "stances": ["guess"]})

    _, record = derive_point_options(project, spec, point, code_revision="test")

    assert record.reconciled is None and record.options is not None
    assert record.decision is not None and record.decision.choices == ("refund",) * 2 + ("decline",)

    lone = _options(("refund", "episode:0/event:e3"))
    transport_two = ScriptedJsonTransport((_proposal(lone),))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport_two
    )
    other = point.model_copy(update={"point": 3})
    _, empty = derive_point_options(project, spec, other, code_revision="test")
    assert empty.options is None and empty.decision is None and empty.choices == ()
    assert empty.failure is not None and "no valid option set" in empty.failure
    assert len(transport.requests) == len(replies)


def test_check_options_refuses_uncited_repeated_or_contextless_options() -> None:
    """Options must cite a continuation event that exists and keep distinct identities."""
    locators = frozenset({"point", "episode:0/event:e3", "episode:1/event:e3"})

    assert check_options(VALID, locators)[1] is None
    repeated = _options(("refund", "episode:0/event:e3"), ("refund", "episode:1/event:e3"))
    assert "distinct" in str(check_options(repeated, locators)[1])
    contextless = _options(("refund", "point"), ("decline", "episode:1/event:e3"))
    assert "continuation event" in str(check_options(contextless, locators)[1])
    uncited: JsonObject = {
        "decision": "d",
        "options": [{"option_id": "a", "action": "x", "evidence_refs": []}],
    }
    assert "cites no evidence" in str(check_options(uncited, locators)[1])
