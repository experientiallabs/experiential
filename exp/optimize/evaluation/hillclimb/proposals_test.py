"""The analyst reads train failures and keep bits only, and a round can measure its pick."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.hillclimb import proposals
from exp.optimize.evaluation.hillclimb.design import ClimbSpec, climb_split
from exp.optimize.evaluation.hillclimb.prompts import PromptEdit, load_prompt_version
from exp.optimize.evaluation.hillclimb.proposals import (
    EditProposalSpec,
    preflight_edit_proposals,
    run_edit_proposals,
)
from exp.optimize.evaluation.hillclimb.rounds import ClimbRound, ProposalChoice, run_climb_round
from exp.optimize.evaluation.hillclimb.tests.climb_test import HELPFUL, climb_fixture, deadline
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import RuntimeModelCatalog


class ScriptedAnalyst(ScriptedJsonTransport):
    """Answers one proposal request, citing the supplied failures or an invented one.

    Attributes:
        invent: Cite a failure ref the request did not supply.
    """

    def __init__(self, *, invent: bool = False) -> None:
        """Choose whether to cite an invented failure."""
        super().__init__()
        self.invent = invent

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Propose the helpful edit and a repeat of the reverted one."""
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[-1], dict)
        content = json.loads(str(messages[-1]["content"]))
        refs = content["valid_failure_refs"]
        answer = {
            "root_causes": [
                {
                    "cause": "Deletes or edits more than requested",
                    "explanation": "The failed calls change more than the request names.",
                    "failures": ["failure:9:9"] if self.invent else refs,
                }
            ],
            "edits": [
                {
                    "root_cause": 0,
                    "kind": "append",
                    "text": HELPFUL,
                    "change_note": "Re-read the request before editing.",
                    "expected_effect": "Edits match the requested change.",
                },
                {
                    "root_cause": 0,
                    "kind": "append",
                    "text": "Be concise.",
                    "change_note": "Shorter replies.",
                    "expected_effect": "None on the goal.",
                },
            ],
            "limitations": ["Scripted analyst; not evidence about any real model."],
        }
        return JsonHttpResponse(
            status_code=200,
            body={
                "id": "analyst",
                "model": "judge-v1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "answer",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_proposal",
                                        "arguments": json.dumps(answer),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )


def _after_one_revert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ClimbSpec, EditProposalSpec]:
    """Revert a noise edit in round 1 and register a proposal request after it."""
    project, spec, _ = climb_fixture(tmp_path, monkeypatch, "records")
    noise = PromptEdit(kind="append", text="Be concise.", change_note="Shorter replies.")
    record = run_climb_round(
        project,
        ClimbRound(climb=spec, round=1, edit=noise, deadline=deadline()),
        code_revision="test",
    )
    assert not record.kept
    catalog = load_model_catalog(project.model_catalog_path)
    request = EditProposalSpec(
        climb=spec,
        after_round=1,
        maximum_edits=2,
        analyst_alias="judge",
        analyst_model=RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0],
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        deadline=datetime.now(UTC) + timedelta(minutes=30),
    )
    return project, spec, request


def test_analyst_sees_train_failures_and_keep_bits_and_its_pick_is_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test lineages, scores and gains never reach the analyst; a chosen edit is a round."""
    project, spec, request = _after_one_revert(tmp_path, monkeypatch)
    analyst = ScriptedAnalyst()
    monkeypatch.setattr(proposals, "HttpxJsonTransport", lambda: analyst)
    body = preflight_edit_proposals(project, request)
    saved = run_edit_proposals(project, request, code_revision="test")
    assert len(analyst.requests) == 1
    sent = json.dumps(analyst.requests[0].payload)
    split = climb_split(project, spec)
    for index in split.test:
        assert f"Case {index}." not in sent
    assert any(f"Case {index}." in sent for index in split.train)
    for leaked in ("noise_floor", "pass_rate", "unit_differences", "estimate", "interval"):
        assert leaked not in sent
    messages = body["messages"]
    assert isinstance(messages, list) and isinstance(messages[-1], dict)
    content = json.loads(str(messages[-1]["content"]))
    assert [set(row) for row in content["history"]] == [
        {"round", "kind", "text", "change_note", "kept"}
    ]
    assert content["history"][0]["kept"] is False
    assert saved.measured == (None, 1)
    assert saved.buckets and all(bucket.symptoms for bucket in saved.buckets)
    assert run_edit_proposals(project, request, code_revision="test") == saved
    assert len(analyst.requests) == 1
    reference = artifact_input(project.artifacts.read(request.identity("test")).manifest)
    with pytest.raises(ValueError, match="already measured in round 1"):
        run_climb_round(
            project,
            ClimbRound(
                climb=spec,
                round=2,
                choice=ProposalChoice(proposals=reference, index=1),
                deadline=deadline(),
            ),
            code_revision="test",
        )
    kept = run_climb_round(
        project,
        ClimbRound(
            climb=spec,
            round=2,
            choice=ProposalChoice(proposals=reference, index=0),
            deadline=deadline(),
        ),
        code_revision="test",
    )
    assert kept.kept
    version = load_prompt_version(project.artifacts, kept.candidate)
    assert version.proposal == reference and version.system_prompt == HELPFUL


def test_an_answer_citing_unsupplied_failures_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The analyst cannot invent evidence; nothing is saved."""
    project, _, request = _after_one_revert(tmp_path, monkeypatch)
    monkeypatch.setattr(proposals, "HttpxJsonTransport", lambda: ScriptedAnalyst(invent=True))
    with pytest.raises(ValueError, match="failure:9:9. is not one of"):
        run_edit_proposals(project, request, code_revision="test")
    assert not project.artifacts.exists(request.identity("test"))
