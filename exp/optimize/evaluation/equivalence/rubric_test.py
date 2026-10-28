"""Trajectory grounding, source admission, review freezes and evaluation separation."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.evaluations.actor_equivalence_test import equivalence_rubric
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.analysis.source_access import declare_source_access
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.equivalence import rubric as owner
from exp.optimize.evaluation.equivalence.review import ReviewModel
from exp.optimize.evaluation.equivalence.rubric import (
    CriterionGrounding,
    RubricCitation,
    RubricDerivationSpec,
    RubricProposal,
    RubricReview,
    RubricTrajectory,
    derive_equivalence_rubric,
    freeze_equivalence_rubric,
    load_equivalence_rubric,
    preflight_rubric_derivation,
    register_rubric_trajectory,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


@pytest.mark.parametrize(
    ("quote", "accepted"),
    [("write(a)\nsave(b)", True), ("write(a)\nremove(b)", False), ("ab", False)],
)
def test_multiline_string_citations_preserve_original_characters(
    quote: str, accepted: bool
) -> None:
    """Quoted Python command text survives JSON escaping without joining unrelated leaves."""
    field = json.dumps({"command": "write(a)\nsave(b)", "separate": ["a", "b"]})
    assert owner._quote_present(quote, field) is accepted


def rubric_source(project: ProjectStore) -> ArtifactInput:
    """Retain observed public recovery and deliberately private probes in a development source."""
    task = TaskCase(
        task_id="rubric-source",
        lineage_group_id="rubric-source-family",
        partition="fit",
        workload_weight=1.0,
        origin="authored",
        instruction="Read the workbook using available tools",
    )
    source = RubricTrajectory(
        task=task,
        trajectory=WorkflowEvidence(
            workflow_id="rubric-source",
            task=task.instruction,
            initial_state={"secret_probe": "PRIVATE-PROBE"},
            steps=(
                WorkflowStep(
                    step=0,
                    action={"name": "install"},
                    status="valid",
                    reply="package refused",
                    before={"private": "PRIVATE-BEFORE"},
                ),
                WorkflowStep(
                    step=1,
                    action={"name": "read"},
                    status="valid",
                    reply="read using existing tools",
                    after={"private": "PRIVATE-AFTER"},
                ),
            ),
            audit_metadata={"label": "PRIVATE-LABEL"},
            final_answer="Read using existing tools",
            stop_reason="completed",
        ),
    )
    return register_rubric_trajectory(project, source, inputs=(), code_revision="test")


def rubric_proposal(source: ArtifactInput) -> RubricProposal:
    """Cite an exact observed refusal for every fixture criterion; no calibration is claimed."""
    rubric = equivalence_rubric()
    return RubricProposal(
        rubric=rubric,
        grounding=tuple(
            CriterionGrounding(
                dimension_id=c.dimension_id,
                citations=(
                    RubricCitation(
                        reference=source.artifact_id + ":step-0:reply", quote="package refused"
                    ),
                ),
            )
            for c in rubric.criteria
        ),
    )


def derivation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, RubricDerivationSpec, RubricProposal]:
    """Configure one admitted source and a pinned proposer without making a provider call."""
    project, base, _ = job_fixture(tmp_path, monkeypatch)
    model = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), environment={}
    ).snapshot("judge")[0]
    reviewer = ReviewModel(alias="judge", model=model, output_tokens=2048)
    source = rubric_source(project)
    declare_source_access(
        project.artifacts,
        source,
        scope="development",
        rationale="Synthetic development fixture",
        code_revision="test",
    )
    spec = RubricDerivationSpec(
        sources=(source,),
        proposer=reviewer,
        catalog_sha256=base.catalog_sha256,
        state_coverage="partial",
        deadline=base.deadline,
    )
    return project, spec, rubric_proposal(source)


def frozen_rubric(project: ProjectStore, reviewer: ReviewModel) -> ArtifactInput:
    """Exercise derive and explicit author review for isolated paired-runtime fixtures."""
    source = rubric_source(project)
    declare_source_access(
        project.artifacts,
        source,
        scope="development",
        rationale="Synthetic development fixture",
        code_revision="test",
    )
    spec = RubricDerivationSpec(
        sources=(source,),
        proposer=reviewer,
        catalog_sha256=sha256_json(
            load_model_catalog(project.model_catalog_path).model_dump(mode="json")
        ),
        state_coverage="partial",
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            owner, "review_call", Mock(return_value=rubric_proposal(source).model_dump(mode="json"))
        )
        draft = derive_equivalence_rubric(project, spec, code_revision="test")
    return freeze_equivalence_rubric(
        project,
        RubricReview(
            draft=draft,
            reviewer="fixture author",
            rationale="Explicit synthetic review, not independent calibration",
        ),
        code_revision="test",
    )


def test_public_chronology_only_and_replay_without_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep private probes out of proposals and replay drafts without credentials."""
    project, spec, proposal = derivation_fixture(tmp_path, monkeypatch)
    observed = Mock(return_value=proposal.model_dump(mode="json"))
    monkeypatch.setattr(owner, "review_call", observed)
    body = preflight_rubric_derivation(project, spec)
    assert "PRIVATE-" not in json.dumps(body)
    assert "package refused" in json.dumps(body) and "existing tools" in json.dumps(body)
    draft = derive_equivalence_rubric(project, spec, code_revision="test")
    assert observed.call_count == 1
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    assert derive_equivalence_rubric(project, spec, code_revision="test") == draft
    assert observed.call_count == 1
    frozen = freeze_equivalence_rubric(
        project,
        RubricReview(
            draft=draft, reviewer="author", rationale="Reviewed limitations and alternatives"
        ),
        code_revision="test",
    )
    assert (
        load_equivalence_rubric(project, frozen, evaluation_lineages=("fresh-family",)).calibration
        == "provisional"
    )
    with pytest.raises(ValueError, match="lineages overlap"):
        load_equivalence_rubric(project, frozen, evaluation_lineages=("rubric-source-family",))


@pytest.mark.parametrize(
    "fault",
    [
        "protected",
        "unknown",
        "invented_quote",
        "foreign_citation",
        "coverage",
        "omitted_criterion",
        "oversized",
    ],
)
def test_derivation_fails_closed_before_usable_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Reject bad source scope, fabricated grounding and incomplete requests."""
    project, spec, proposal = derivation_fixture(tmp_path, monkeypatch)
    if fault == "protected":
        declare_source_access(
            project.artifacts,
            spec.sources[0],
            scope="protected",
            rationale="Reserved evaluation",
            code_revision="test",
        )
    if fault == "unknown":
        source = rubric_source(project)
        task = RubricTrajectory.model_validate_json(
            project.artifacts.read_bytes(source.artifact_id, "trajectory.json")
        )
        source = register_rubric_trajectory(
            project,
            task.model_copy(
                update={"task": task.task.model_copy(update={"task_id": "unclassified"})}
            ),
            inputs=(),
            code_revision="test",
        )
        spec = spec.model_copy(update={"sources": (source,)})
    if fault == "coverage":
        proposal = proposal.model_copy(
            update={"rubric": proposal.rubric.model_copy(update={"state_coverage": "full"})}
        )
    if fault == "omitted_criterion":
        proposal = proposal.model_copy(update={"grounding": proposal.grounding[:-1]})
    if fault in {"invented_quote", "foreign_citation"}:
        citation = RubricCitation(
            reference=spec.sources[0].artifact_id + ":step-0:reply"
            if fault == "invented_quote"
            else "held-out:step-0:reply",
            quote="invented success",
        )
        proposal = proposal.model_copy(
            update={
                "grounding": (
                    proposal.grounding[0].model_copy(update={"citations": (citation,)}),
                    *proposal.grounding[1:],
                )
            }
        )
    if fault == "oversized":
        spec = spec.model_copy(update={"maximum_input_bytes": 1024})
    callback = Mock(return_value=proposal.model_dump(mode="json"))
    monkeypatch.setattr(owner, "review_call", callback)
    with pytest.raises(ValueError):
        derive_equivalence_rubric(project, spec, code_revision="test")
    if fault in {"protected", "unknown", "oversized"}:
        callback.assert_not_called()
    assert not project.artifacts.list_ids(artifact_type="actor-rubric-draft")


def test_reserved_task_is_rejected_even_with_local_fit_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace-backed task cannot bypass frozen build resolution using a forged fit label."""
    project, spec, _ = derivation_fixture(tmp_path, monkeypatch)
    source = RubricTrajectory.model_validate_json(
        project.artifacts.read_bytes(spec.sources[0].artifact_id, "trajectory.json")
    )
    forged = source.task.model_copy(update={"origin": "trace", "source_trace_ids": ("held-out",)})
    with pytest.raises(ValueError, match="frozen build"):
        register_rubric_trajectory(
            project, source.model_copy(update={"task": forged}), inputs=(), code_revision="test"
        )


def test_protected_ancestor_and_reclassification_refuse_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admitting a wrapper cannot conceal protected parents or authorize a stale frozen rubric."""
    project, spec, proposal = derivation_fixture(tmp_path, monkeypatch)
    callback = Mock(return_value=proposal.model_dump(mode="json"))
    monkeypatch.setattr(owner, "review_call", callback)
    draft = derive_equivalence_rubric(project, spec, code_revision="test")
    frozen = freeze_equivalence_rubric(
        project,
        RubricReview(draft=draft, reviewer="author", rationale="Reviewed provisional criteria."),
        code_revision="test",
    )
    parent = spec.sources[0]
    source = RubricTrajectory.model_validate_json(
        project.artifacts.read_bytes(parent.artifact_id, "trajectory.json")
    )
    child = register_rubric_trajectory(project, source, inputs=(parent,), code_revision="test")
    declare_source_access(
        project.artifacts,
        child,
        scope="development",
        rationale="Wrapper only",
        code_revision="test",
    )
    declare_source_access(
        project.artifacts, parent, scope="protected", rationale="Reserved", code_revision="test"
    )
    with pytest.raises(ValueError):
        preflight_rubric_derivation(project, spec.model_copy(update={"sources": (child,)}))
    with pytest.raises(ValueError):
        load_equivalence_rubric(project, frozen, evaluation_lineages=("untouched",))
    assert callback.call_count == 1


def test_rejected_raw_proposal_replays_without_redraw_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known unsupported citation retains its original physical response on every inspection."""
    project, spec, proposal = derivation_fixture(tmp_path, monkeypatch)
    content = proposal.model_dump(mode="json")
    content["grounding"][0]["citations"][0]["quote"] = "invented recovery"
    transport = ScriptedJsonTransport(
        (
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "review",
                    "object": "chat.completion",
                    "created": 0,
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
                                        "id": "proposal",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_proposal",
                                            "arguments": json.dumps(content),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            ),
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: transport
    )
    with pytest.raises(ValueError, match="invented trajectory evidence"):
        derive_equivalence_rubric(project, spec, code_revision="test")
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="invented trajectory evidence"):
        derive_equivalence_rubric(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")
    assert not project.artifacts.list_ids(artifact_type="actor-rubric-draft")


def test_author_corrections_are_bound_to_review_and_revalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reviewed correction changes identity and cannot introduce fabricated evidence."""
    project, spec, proposal = derivation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "review_call", Mock(return_value=proposal.model_dump(mode="json")))
    draft = derive_equivalence_rubric(project, spec, code_revision="test")
    review = RubricReview(draft=draft, reviewer="author", rationale="Reviewed paired meanings.")
    original = freeze_equivalence_rubric(project, review, code_revision="test")
    corrected = proposal.model_copy(
        update={
            "rubric": proposal.rubric.model_copy(
                update={
                    "definition": proposal.rubric.definition.model_copy(
                        update={"syllabus": "Compare recovery behavior, including shared mistakes."}
                    )
                }
            )
        }
    )
    frozen = freeze_equivalence_rubric(
        project, review.model_copy(update={"proposal": corrected}), code_revision="test"
    )
    assert frozen != original
    assert (
        load_equivalence_rubric(project, frozen, evaluation_lineages=("untouched",))
        == corrected.rubric
    )
    bad = corrected.grounding[0].model_copy(
        update={"citations": (RubricCitation(reference="invented", quote="unsupported"),)}
    )
    with pytest.raises(ValueError):
        freeze_equivalence_rubric(
            project,
            review.model_copy(
                update={
                    "proposal": corrected.model_copy(
                        update={"grounding": (bad, *corrected.grounding[1:])}
                    )
                }
            ),
            code_revision="test",
        )


def test_declared_families_exclude_every_lineage_of_a_development_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A policy family groups lineages, so a sibling lineage of the rubric's family is refused."""
    project, spec, proposal = derivation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "review_call", Mock(return_value=proposal.model_dump(mode="json")))
    draft = derive_equivalence_rubric(project, spec, code_revision="test")
    frozen = freeze_equivalence_rubric(
        project,
        RubricReview(draft=draft, reviewer="author", rationale="Reviewed paired meanings."),
        code_revision="test",
    )
    assert load_equivalence_rubric(project, frozen, evaluation_lineages=("sibling",))
    families = {"rubric-source-family": "workbooks", "sibling": "workbooks"}
    with pytest.raises(ValueError, match="lineages overlap"):
        load_equivalence_rubric(
            project, frozen, evaluation_lineages=("sibling",), families=families
        )


def test_a_rubric_never_judges_a_family_its_sources_descend_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source citing another family's trajectory excludes that family too."""
    project, spec, _ = derivation_fixture(tmp_path, monkeypatch)
    parent = spec.sources[0]
    source = RubricTrajectory.model_validate_json(
        project.artifacts.read_bytes(parent.artifact_id, "trajectory.json")
    )
    task = source.task.model_copy(update={"task_id": "child", "lineage_group_id": "child-family"})
    child = register_rubric_trajectory(
        project, source.model_copy(update={"task": task}), inputs=(parent,), code_revision="test"
    )
    access = declare_source_access(
        project.artifacts, child, scope="development", rationale="Child", code_revision="test"
    )
    spec = spec.model_copy(update={"sources": (child,)})
    monkeypatch.setattr(
        owner, "review_call", Mock(return_value=rubric_proposal(child).model_dump(mode="json"))
    )
    draft = derive_equivalence_rubric(project, spec, code_revision="test")
    frozen = freeze_equivalence_rubric(
        project,
        RubricReview(draft=draft, reviewer="author", rationale="Reviewed provisional criteria."),
        code_revision="test",
    )
    assert load_equivalence_rubric(project, frozen, evaluation_lineages=("untouched",))
    with pytest.raises(ValueError, match="lineages overlap"):
        load_equivalence_rubric(project, frozen, evaluation_lineages=("rubric-source-family",))

    # An ancestor whose lineage is unknown cannot prove separation, so it is refused.
    declare_source_access(
        project.artifacts,
        access,
        scope="development",
        rationale="Not a trajectory",
        code_revision="test",
    )
    opaque = register_rubric_trajectory(
        project, source.model_copy(update={"task": task}), inputs=(access,), code_revision="test"
    )
    declare_source_access(
        project.artifacts, opaque, scope="development", rationale="Opaque", code_revision="test"
    )
    with pytest.raises(ValueError, match="ancestors must be registered development trajectories"):
        preflight_rubric_derivation(project, spec.model_copy(update={"sources": (opaque,)}))
