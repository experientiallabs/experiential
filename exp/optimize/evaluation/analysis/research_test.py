"""Falsification research preserves unknowns, repair ownership and source authority."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactEnvelope
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.research import (
    CounterexampleProposal,
    FailureHypothesis,
    FailureResearchSpec,
    ProvisionalRepair,
    ResearchEvidence,
    ResearchFailure,
    ResearchRequirement,
    check_failure_research,
    persist_failure_research,
)
from exp.simulation.evaluation.tasks import FieldRelation, TaskVerifier


def research_fixture(tmp_path: Path) -> tuple[ProjectStore, FailureResearchSpec]:
    """A non-contiguous duplicate witness with separately authored expected retained records."""
    project = ProjectStore(tmp_path, "research")
    source = project.artifacts.write_json(
        artifact_id="originals",
        artifact_type="research-fixture",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={
            "original.json": {
                "task": "Keep the last row for each key, across the entire input.",
                "observed_program": "keep if next key differs",
                "witness": {
                    "final_state": {"rows": ["a1", "b2", "a3"]},
                    "expected": ["b2", "a3"],
                    "stop_reason": "completed",
                },
                "positive": {
                    "final_state": {"rows": ["b2", "a3"]},
                    "expected": ["b2", "a3"],
                    "stop_reason": "completed",
                },
                "negative": {
                    "final_state": {"rows": []},
                    "expected": ["b2", "a3"],
                    "stop_reason": "completed",
                },
                "missing": {"expected": ["b2", "a3"], "stop_reason": "completed"},
            }
        },
    )
    ref = artifact_input(source)
    task = ResearchEvidence(
        source=ref, file="original.json", pointer="/task", quote="Keep the last row"
    )
    requirement = ResearchRequirement(
        id="last",
        statement="All keys, not just adjacent groups",
        evidence=task,
        status="declared",
        verifier=TaskVerifier(
            description="Retain the declared last occurrences",
            outcome_schema={"type": "object"},
            required_paths=("/final_state/rows", "/expected"),
            relations=(FieldRelation(left="/final_state/rows", right="/expected"),),
        ),
    )
    attempts = tuple(
        CounterexampleProposal(
            id=name,
            hypothesis="adjacent",
            requirement="last",
            purpose="Non-contiguous repeated key",
            kind=kind,
            basis="constructed_witness",
            outcome=ResearchEvidence(source=ref, file="original.json", pointer="/" + name),
            construction="Explicit small list witness; original actor program not executed",
        )
        for name, kind in (
            ("witness", "candidate"),
            ("positive", "positive_control"),
            ("negative", "negative_control"),
            ("missing", "missing_evidence_control"),
        )
    )
    return project, FailureResearchSpec(
        failure=ResearchFailure(
            symptom="Program appears to compare only neighboring keys",
            observations=(
                ResearchEvidence(source=ref, file="original.json", pointer="/observed_program"),
            ),
            hypotheses=(
                FailureHypothesis(
                    id="adjacent",
                    explanation="Actor assumes sorted keys",
                    destination="actor",
                    falsifier="Non-contiguous duplicates are handled correctly",
                ),
                FailureHypothesis(
                    id="world",
                    explanation="Simulator implied sortedness",
                    destination="simulator",
                    falsifier="Same behavior on original observation",
                ),
            ),
        ),
        requirements=(requirement,),
        counterexamples=attempts,
        revision=ProvisionalRepair(
            hypothesis="adjacent",
            destination="actor",
            change="Test handling of repeated keys separated by other keys",
            remaining_uncertainty="No causal attribution or original program execution",
        ),
    )


def test_counterexample_and_controls_use_independent_predicate_not_citation_votes(
    tmp_path: Path,
) -> None:
    """A failure is retained with both competing explanations and cannot publish simulator rules."""
    project, spec = research_fixture(tmp_path)
    report = check_failure_research(project.artifacts, spec)
    assert [row.result for row in report.checks] == [
        "violates_declared_requirement",
        "not_falsified",
        "violates_declared_requirement",
        "unassessable",
    ]
    assert report.controls == {"last": "passed"}
    assert report.revision_status == "provisional"
    assert not report.memory_published and report.attribution == "unresolved"
    assert report.spec.revision.destination == "actor"
    assert len(report.spec.failure.hypotheses) == 2
    saved = persist_failure_research(
        project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    assert (
        persist_failure_research(
            project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
        )
        == saved
    )


def test_unstated_write_boundary_cannot_become_a_failed_actor_label(tmp_path: Path) -> None:
    """An output range does not automatically establish a prohibition on all other writes."""
    project, spec = research_fixture(tmp_path)
    requirement = spec.requirements[0].model_copy(update={"status": "unresolved", "verifier": None})
    spec = spec.model_copy(update={"requirements": (requirement,)})
    report = check_failure_research(project.artifacts, spec)
    assert all(row.result == "unassessable" for row in report.checks)
    assert report.revision_status == "blocked"
    assert not report.memory_published


@pytest.mark.parametrize(
    "fault", ["tautology", "missing_control", "missing_candidate", "missing_result"]
)
def test_self_confirming_or_incomplete_checks_block_revision(tmp_path: Path, fault: str) -> None:
    """An always-pass grader fails the negative control; missing evidence never earns support."""
    project, spec = research_fixture(tmp_path)
    if fault == "tautology":
        requirement = spec.requirements[0].model_copy(
            update={
                "verifier": TaskVerifier(
                    description="Always pass", outcome_schema={"type": "object"}
                )
            }
        )
        spec = spec.model_copy(update={"requirements": (requirement,)})
    elif fault == "missing_control":
        spec = spec.model_copy(update={"counterexamples": spec.counterexamples[:-1]})
    elif fault == "missing_candidate":
        spec = spec.model_copy(update={"counterexamples": spec.counterexamples[1:]})
    else:
        proposal = spec.counterexamples[0].model_copy(
            update={"outcome": None, "basis": "unexecuted_proposal"}
        )
        spec = spec.model_copy(update={"counterexamples": (proposal, *spec.counterexamples[1:])})
    assert check_failure_research(project.artifacts, spec).revision_status == "blocked"


@pytest.mark.parametrize(
    "fault", ["quote", "digest", "destination", "duplicate", "invented_requirement"]
)
def test_research_cannot_relabel_or_rewrite_its_sources(tmp_path: Path, fault: str) -> None:
    """Reject invented quotations, altered digests and accidental actor-to-simulator routing."""
    project, spec = research_fixture(tmp_path)
    data = spec.model_dump(mode="json")
    if fault == "quote":
        data["requirements"][0]["evidence"]["quote"] = "Never touch B11"
    elif fault == "digest":
        data["failure"]["observations"][0]["source"]["sha256"] = "f" * 64
    elif fault == "destination":
        data["revision"]["destination"] = "simulator"
    elif fault == "duplicate":
        data["counterexamples"].append(data["counterexamples"][0])
    else:
        data["counterexamples"][0]["requirement"] = "invented"
    with pytest.raises(ValueError):
        check_failure_research(project.artifacts, FailureResearchSpec.model_validate(data))


def test_actual_cli_saves_research_and_replays_without_calls(tmp_path: Path) -> None:
    """Public eval surface reaches evidence checks and immutable research artifacts."""
    project, spec = research_fixture(tmp_path)
    path = tmp_path / "research.json"
    path.write_text(
        ContinuationCommand(action="check-failure-research", research=spec).model_dump_json()
    )
    args = ["eval", "research", "--root", str(tmp_path), "--continuations", str(path)]
    dry = CliRunner().invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="failure-research")
    for _ in range(2):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert '"memory_published": false' in result.output
        assert "saved_research" in result.output
    assert len(project.artifacts.list_ids(artifact_type="failure-research")) == 1
