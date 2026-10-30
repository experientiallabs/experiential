"""Author output becomes actual native worlds, with failed prototypes and authority retained."""

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, tzinfo
from pathlib import Path

import pytest
from filelock import Timeout
from pydantic import BaseModel, JsonValue

from exp.common.core.artifacts import ArtifactEnvelope, sha256_bytes
from exp.common.project import ProjectStore
from exp.common.project.manifests import ArtifactManifest
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.scenario_search_test import workbook_search_fixture
from exp.optimize.evaluation.factory.worlds import (
    WorldCandidates,
    WorldHardening,
    WorldProposal,
    WorldProposalSpec,
    preflight_world_proposals,
    propose_worlds,
    world_proposal_request,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.world_construction import ConstructionPlan, load_construction


def world_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, variant: str = "valid"
) -> tuple[ProjectStore, WorldProposalSpec, ScriptedJsonTransport]:
    """Use a scripted provider with real compilation, binding and independent native checks."""
    project, search, _ = workbook_search_fixture(tmp_path, monkeypatch)
    base = load_executable_task(project, search.base)
    assert search.target is not None
    write_catalog(tmp_path)
    # Select a copied trusted implementation so tests never remove or edit repository files.
    template = search.candidates[0].scenario
    runtime = tmp_path / "bound_runtime.py"
    runtime.write_bytes(template.capture.runtime_file.read_bytes())
    template = template.model_copy(
        update={
            "capture": template.capture.model_copy(
                update={
                    "runtime_file": runtime,
                    "runtime_sha256": sha256_bytes(runtime.read_bytes()),
                }
            )
        }
    )
    spec = WorldProposalSpec(
        base=search.base,
        brief=search.brief,
        target=search.target,
        template=template,
        acceptance=search.candidates[0].acceptance,
        asset_requirements=(
            "XLSX sheets have no implicit header. "
            "original.xlsx contains Data (id,key with one header) and Protected. "
            "Keep Protected intact. Public tools inspect input.xlsx/output.xlsx. "
            "Private projection: /final_state/views/main/sha256 hashes the Data sheet, "
            "/files/input.xlsx and /views/protected/sha256 are preservation paths."
        ),
        model_alias=base.job.actor_alias,
        model=base.job.study.actor_model,
        catalog_sha256=base.job.catalog_sha256,
        deadline=base.job.deadline,
    )
    rows = []
    for candidate in search.candidates:
        reference = candidate.scenario.capture.sources[0]
        construction = load_construction(project.artifacts, reference)
        rows.append(
            WorldProposal(
                construction=ConstructionPlan.model_validate(
                    construction.model_dump(exclude={"sources"})
                ),
                rationale="Separate global duplicate removal from adjacency-only deletion.",
                verifier=candidate.verifier,
                controls=candidate.controls,
            ).model_dump(mode="json")
        )
    if variant == "duplicate":
        rows[1]["construction"] = {**rows[0]["construction"], "name": "renamed same data"}
    elif variant == "bad-length":
        rows[0]["construction"]["tables"]["Data"]["columns"]["id"]["values"] = [1, 2]
    elif variant == "forged-sources":
        rows[0]["construction"]["sources"] = [spec.acceptance.model_dump(mode="json")]
    elif variant == "missing-role":
        rows[0]["controls"][-1]["role"] = "solution"
    response = JsonHttpResponse(
        status_code=200,
        body={
            "id": "world-author",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "worlds",
                                "type": "function",
                                "function": {
                                    "name": "propose_worlds",
                                    "arguments": json.dumps({"candidates": rows}),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )
    transport = ScriptedJsonTransport((response,))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    return project, spec, transport


def test_generated_worlds_execute_native_discrimination_and_replay_without_author_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confounded proposal is rejected and a counterexample earns only development status."""
    project, spec, transport = world_fixture(tmp_path, monkeypatch)
    preflight_world_proposals(project, spec)
    request = world_proposal_request(project, spec)
    public = json.loads(request.messages[1].content or "")
    assert "acceptance" not in public and "template" not in public
    assert "configuration" not in public and "verifier" not in public
    assert public["evidence"] == [[["id", "key"], [1, 1], [2, 1], [3, 2], [4, 2]]]
    schema = WorldCandidates.model_json_schema()
    assert "sources" not in schema["$defs"]["ConstructionPlan"]["properties"]
    report = propose_worlds(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert len(report.cells) == 2 and all(cell.candidate for cell in report.cells)
    assert report.search is not None
    assert report.search.cells[0].variant is None
    assert any("targeted_error" in reason for reason in report.search.cells[0].refusals)
    assert report.search.cells[1].variant is not None, report.search.cells[1].refusals
    assert len(report.search.cells[1].witnesses) == 5
    for cell in report.cells:
        assert cell.assets is not None and cell.candidate is not None
        assert load_construction(project.artifacts, cell.assets).sources == (report.response,)
        assert cell.candidate.acceptance == spec.acceptance
        assert cell.candidate.scenario.capture.task == spec.template.capture.task
    project.model_catalog_path.unlink()
    spec.template.capture.runtime_file.unlink()
    assert propose_worlds(project, spec, code_revision="test") == report
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("duplicate", "duplicate materialized world"),
        ("bad-length", "must be 1 or 5"),
        ("missing-role", "every required role"),
    ],
)
def test_invalid_worlds_remain_in_population_without_native_search_or_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, reason: str
) -> None:
    """A failed candidate cannot be silently replaced or reduce the comparison denominator."""
    project, spec, transport = world_fixture(tmp_path, monkeypatch, variant=variant)
    report = propose_worlds(project, spec, code_revision="test")
    assert len(report.cells) == 2 and report.search is None
    assert any(reason in (cell.refusal or "") for cell in report.cells)
    assert not project.artifacts.list_ids(artifact_type="scenario-search-registration")
    assert propose_worlds(project, spec, code_revision="test") == report
    assert len(transport.requests) == 1


def test_author_cannot_supply_provenance_or_replace_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undeclared source authority is rejected after retaining the response, without redraw."""
    project, spec, transport = world_fixture(tmp_path, monkeypatch, variant="forged-sources")
    for _ in range(2):
        with pytest.raises(ValueError, match="sources"):
            propose_worlds(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")
    assert not project.artifacts.list_ids(artifact_type="world-proposal-cell")


def test_publication_crash_reuses_completed_native_cells_after_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence-only settlement does not import a missing runtime or purchase provider work."""
    project, spec, transport = world_fixture(tmp_path, monkeypatch)
    write = project.artifacts.write_json

    def interrupt(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Crash only after all native controls and results have been saved."""
        if artifact_type == "world-proposals":
            raise KeyboardInterrupt("publication interrupted")
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    with monkeypatch.context() as patch:
        patch.setattr(project.artifacts, "write_json", interrupt)
        with pytest.raises(KeyboardInterrupt):
            propose_worlds(project, spec, code_revision="test")
    project.model_catalog_path.unlink()
    spec.template.capture.runtime_file.unlink()

    class ExpiredClock(datetime):
        """Place recovery after the frozen dispatch deadline."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            """Return an aware time after the original call allowance."""
            return spec.deadline + timedelta(days=1)

    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.datetime", ExpiredClock)
    report = propose_worlds(project, spec, code_revision="test")
    assert report.search is not None and report.search.cells[1].variant is not None
    assert len(transport.requests) == 1


@pytest.mark.parametrize("busy", [False, True])
def test_interrupted_binding_is_recorded_and_never_imported_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, busy: bool
) -> None:
    """A durable binding intent with unknown effects cannot silently rerun selected local code."""
    project, spec, transport = world_fixture(tmp_path, monkeypatch)

    def crash(*args: object) -> None:
        """Interrupt after the binding intent and asset publication."""
        if busy:
            raise Timeout("binding.lock")
        raise KeyboardInterrupt("binding interrupted")

    with monkeypatch.context() as patch:
        patch.setattr("exp.optimize.evaluation.factory.worlds.bind_construction", crash)
        with pytest.raises(Timeout if busy else KeyboardInterrupt):
            propose_worlds(project, spec, code_revision="test")
    report = propose_worlds(project, spec, code_revision="test")
    if busy:
        assert report.cells[0].candidate is not None and report.cells[0].refusal is None
        assert report.search is not None
        assert propose_worlds(project, spec, code_revision="test") == report
    else:
        assert report.cells[0].refusal is not None
        assert "unfinished binding intent" in report.cells[0].refusal
        assert report.search is None
    assert report.cells[1].candidate is not None
    assert len(transport.requests) == 1


def test_unknown_author_outcome_refuses_before_catalog_access_or_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost response remains an unknown purchased draw, even after configuration disappears."""
    project, spec, _ = world_fixture(tmp_path, monkeypatch)
    transport = ScriptedJsonTransport(())

    def lose_response() -> JsonHttpResponse:
        """Simulate failure after a provider intent was claimed."""
        raise RuntimeError("lost author response")

    monkeypatch.setattr(transport, "_answer", lose_response)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    with pytest.raises((ValueError, RuntimeError)):
        propose_worlds(project, spec, code_revision="test")
    assert project.artifacts.exists(spec.identity("test") + "-intent")
    count = len(transport.requests)
    project.model_catalog_path.unlink()
    with pytest.raises(ValueError, match="unknown outcome"):
        propose_worlds(project, spec, code_revision="test")
    assert len(transport.requests) == count


def test_citation_and_runtime_drift_refuse_before_author_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No model call is bought for a missing evidence locator or unpinned executable source."""
    project, spec, transport = world_fixture(tmp_path, monkeypatch)
    observation = spec.target.failure.observations[0].model_copy(update={"pointer": "/missing"})
    altered = spec.model_copy(
        update={
            "target": spec.target.model_copy(
                update={
                    "failure": spec.target.failure.model_copy(
                        update={"observations": (observation,)}
                    ),
                }
            )
        }
    )
    with pytest.raises(ValueError):
        propose_worlds(project, altered, code_revision="test")
    spec.template.capture.runtime_file.write_text("raise AssertionError('must not import')")
    with pytest.raises(ValueError, match="runtime source changed"):
        propose_worlds(project, spec, code_revision="test")
    assert not transport.requests


def test_the_generic_author_prompt_takes_layout_facts_from_the_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime-specific layout reaches the author only through the declared requirements."""
    project, spec, _ = world_fixture(tmp_path, monkeypatch)
    request = world_proposal_request(project, spec)
    system, user = request.messages
    assert "xlsx" not in str(system.content).lower()
    assert json.loads(str(user.content))["asset_requirements"] == spec.asset_requirements


def test_hardening_context_must_cite_a_saved_actor_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harder-world request cannot cite arbitrary evidence as its saturated panel."""
    project, spec, transport = world_fixture(tmp_path, monkeypatch)
    hardened = spec.model_copy(
        update={
            "hardening": WorldHardening(
                level=1, assets=spec.template.capture.sources[0], panel=spec.acceptance
            )
        }
    )
    assert hardened.identity("test") != spec.identity("test")
    assert "hardening" not in spec.model_dump(mode="json")
    with pytest.raises(ValueError, match="actor-panel report"):
        world_proposal_request(project, hardened)
    assert not transport.requests
