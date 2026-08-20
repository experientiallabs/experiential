"""Behavioral tests for incremental judge-first calibration review artifacts."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

from exp.common.judging.judgment import Judgment
from exp.common.judging.rubric import JudgeCalibration
from exp.common.models import ModelCapabilities, ModelSnapshot
from exp.common.project import ProjectConfig, ProjectStore
from exp.optimize.router.judging.contracts import (
    HumanJudgeCorrection,
    JudgeCalibrationBudget,
    JudgeProtocolProbeArtifact,
    ManualJudgeAxisDecision,
    ManualJudgeError,
    ManualJudgeSetupArtifact,
)
from exp.optimize.router.judging.labels import calibration_sample_digest
from exp.optimize.router.judging.review import (
    ManualJudgeTraceProposal,
    completed_trace_review_count,
    read_trace_reviews,
)
from exp.optimize.router.judging.service import (
    ManualJudgeCalibrationPlan,
    calibrate_manual_judge,
    calibration_sample,
    commit_manual_judge_setup,
    estimate_manual_judge_budget,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from exp.optimize.router.judging.service_test import (
    _TIME,
    _built_store,
    _catalog,
    _persist_grounded_build,
    _RuntimeCatalog,
    _StructuredJudgeClient,
    _trace,
)
from exp.runtime.models.registry import ResolvedModel, RuntimeModelCatalog
from exp.simulation.build import build_project
from exp.simulation.ingest.otlp import TraceNormalizationResult
from exp.simulation.mining.service import MiningSpec


def test_trace_review_separates_judge_human_final_and_provenance_fields(
    tmp_path: Path,
) -> None:
    """One correction retains raw judge output and every distinct authorship field.

    Args:
        tmp_path: Pytest-owned project storage.
    """
    store = _built_store(tmp_path)
    setup = _commit_setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    client, runtime = _runtime(plan.setup.judge_model)
    budget = _budget(plan)

    def correct(proposal: ManualJudgeTraceProposal) -> tuple[ManualJudgeAxisDecision, ...]:
        """Correct both score and judgment after observing immutable judge evidence.

        Args:
            proposal: Persisted configured-judge result awaiting a decision.

        Returns:
            One complete human correction.
        """
        assert proposal.judgment.judgment_id in store.artifacts.list_ids()
        return (
            ManualJudgeAxisDecision(
                dimension_id="task-success",
                accepted=False,
                correction=HumanJudgeCorrection(
                    corrected_score=0,
                    corrected_judgment="The trace made progress but did not prove completion.",
                ),
            ),
        )

    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        (),
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME,
        code_revision="test-revision",
        reviewer=correct,
    )

    sample_sha256 = calibration_sample_digest(setup, calibration_sample(plan))
    reviews = read_trace_reviews(store, setup, sample_sha256)
    assert len(reviews) == 1
    review = reviews[0]
    assert result.audit.trace_reviews[0].artifact_id == review.review_id
    assert review.trace_id == plan.traces[0].trace_id
    assert review.lineage_id == plan.tasks[0].lineage_group_id
    assert review.rubric_revision == setup.rubric
    assert review.trace_evidence.artifact_id.startswith("production-rollout-")
    assert review.reference_trace_evidence is None
    assert review.judge_model == plan.setup.judge_model
    assert review.pricing.input_usd_per_million_tokens == 1.0
    assert review.pricing.output_usd_per_million_tokens == 2.0
    assert review.pricing.pricing_source.value == "configured"
    assert review.pricing.authorized_call_count == 1
    assert review.pricing.maximum_reserved_cost_usd == pytest.approx(budget.estimated_cost_usd)
    assert review.provenance.proposal_author == "configured_judge"
    assert review.provenance.decision_author == "human"
    assert review.provenance.final_label_authority == "human_acceptance"
    assert review.provenance.historical_source == "recorded_model"
    axis = review.axes[0]
    assert axis.judge_proposal.proposed_score == 1
    assert axis.judge_proposal.proposed_judgment == "Structured evidence supports the verdict."
    assert axis.judge_proposal.cited_trace_evidence == ()
    assert axis.human_correction is not None
    assert axis.human_correction.corrected_score == 0
    assert axis.human_correction.corrected_judgment == (
        "The trace made progress but did not prove completion."
    )
    assert axis.final_accepted_label.score == 0
    assert axis.final_accepted_label.judgment == axis.human_correction.corrected_judgment
    assert axis.final_accepted_label.score_source == "human_correction"
    assert axis.final_accepted_label.judgment_source == "human_correction"
    assert len(review.original_judge_response) == 1
    probe = JudgeProtocolProbeArtifact.model_validate_json(
        store.artifacts.read_bytes(review.original_judge_response[0].artifact_id, "probe.json")
    )
    judgment = Judgment.model_validate_json(
        store.artifacts.read_bytes(review.normalized_judgment.artifact_id, "judgment.json")
    )
    assert probe.response == {
        "dimensions": [
            {
                "dimension_id": "task-success",
                "raw_score": 1,
                "rationale": "Structured evidence supports the verdict.",
            }
        ]
    }
    assert probe.model == review.judge_model
    assert judgment.dimensions[0].rationale == axis.judge_proposal.proposed_judgment
    assert len(client.requests) == 1


def test_interruption_resumes_next_incomplete_review_and_replays_saved_probe(
    tmp_path: Path,
) -> None:
    """Completed reviews and a pre-decision probe survive interruption without duplicate calls.

    Args:
        tmp_path: Pytest-owned project storage.
    """
    store = _built_store(tmp_path)
    setup = _commit_setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    first_client, first_runtime = _runtime(plan.setup.judge_model)
    budget = _budget(plan)
    seen: list[int] = []

    def interrupt(proposal: ManualJudgeTraceProposal) -> tuple[ManualJudgeAxisDecision, ...]:
        """Accept the first trace and interrupt after the second judge response is saved.

        Args:
            proposal: Current immutable judge proposal.

        Returns:
            Acceptance for the first trace only.

        Raises:
            RuntimeError: The second proposal simulates a human interruption.
        """
        seen.append(proposal.position)
        if proposal.position == 2:
            raise RuntimeError("simulated human interruption")
        return (_accept(),)

    with pytest.raises(RuntimeError, match="simulated human interruption"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, first_runtime),
            plan,
            (),
            budget,
            spend_consented=True,
            approve=False,
            accept_insufficient_labels=True,
            created_at=_TIME,
            code_revision="test-revision",
            reviewer=interrupt,
        )
    assert seen == [1, 2]
    assert len(first_client.requests) == 2
    sample_sha256 = calibration_sample_digest(setup, calibration_sample(plan))
    assert completed_trace_review_count(store, setup, sample_sha256) == 1

    retry_client, retry_runtime = _runtime(plan.setup.judge_model)
    resumed: list[int] = []

    def accept_remaining(
        proposal: ManualJudgeTraceProposal,
    ) -> tuple[ManualJudgeAxisDecision, ...]:
        """Accept only incomplete trace proposals in frozen sample order.

        Args:
            proposal: Current resumed immutable judge proposal.

        Returns:
            One explicit acceptance.
        """
        resumed.append(proposal.position)
        return (_accept(),)

    remaining_budget = _budget(plan, completed_review_count=1)
    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, retry_runtime),
        plan,
        (),
        remaining_budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(minutes=10),
        code_revision="test-revision",
        reviewer=accept_remaining,
    )
    assert resumed == [2, 3]
    assert len(retry_client.requests) == 1
    assert result.provider_calls_made == 1
    assert completed_trace_review_count(store, setup, sample_sha256) == 3

    preflight_calls = retry_runtime.preflight_calls
    provider_calls = len(retry_client.requests)
    replay = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, retry_runtime),
        plan,
        (),
        remaining_budget,
        spend_consented=False,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(minutes=20),
        code_revision="test-revision",
        reviewer=accept_remaining,
    )
    assert replay.audit == result.audit
    assert replay.provider_calls_made == 0
    assert retry_runtime.preflight_calls == preflight_calls
    assert len(retry_client.requests) == provider_calls
    assert resumed == [2, 3]


def test_five_default_lineages_are_normally_sufficient_without_risk_acceptance(
    tmp_path: Path,
) -> None:
    """Five completed distinct-lineage reviews produce an approvable normal calibration.

    Args:
        tmp_path: Pytest-owned project storage.
    """
    store = _five_lineage_store(tmp_path)
    _commit_setup(store)
    plan = prepare_manual_judge_calibration(store)
    assert len(plan.traces) == len({task.lineage_group_id for task in plan.tasks}) == 5
    client, runtime = _runtime(plan.setup.judge_model)
    budget = _budget(plan)
    reviewed_positions: list[int] = []

    def accept(proposal: ManualJudgeTraceProposal) -> tuple[ManualJudgeAxisDecision, ...]:
        """Accept each configured-judge proposal independently in sample order.

        Args:
            proposal: Current immutable configured-judge proposal.

        Returns:
            One explicit acceptance.
        """
        reviewed_positions.append(proposal.position)
        return (_accept(),)

    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        (),
        budget,
        spend_consented=True,
        approve=True,
        accept_insufficient_labels=False,
        created_at=_TIME,
        code_revision="test-revision",
        reviewer=accept,
    )

    assert reviewed_positions == [1, 2, 3, 4, 5]
    assert len(client.requests) == 5
    assert result.report.eligible_rollout_count == 5
    assert result.report.eligible_lineage_count == 5
    assert result.report.recommended_label_count == 5
    assert result.report.status == "ready_for_approval"
    assert result.approved_calibration is not None
    approved = JudgeCalibration.model_validate_json(
        store.artifacts.read_bytes(
            result.approved_calibration.artifact_id,
            "calibration.json",
        )
    )
    assert approved.status == "human_calibrated"
    assert approved.recommended_label_count == 5
    assert approved.risk_acceptance is None


def test_completed_replay_verifies_each_trace_review_without_provider_calls(
    tmp_path: Path,
) -> None:
    """A corrupt review blocks audit replay before the configured judge is resolved.

    Args:
        tmp_path: Pytest-owned project storage.
    """
    store = _built_store(tmp_path)
    _commit_setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    client, runtime = _runtime(plan.setup.judge_model)
    budget = _budget(plan)

    def accept(_proposal: ManualJudgeTraceProposal) -> tuple[ManualJudgeAxisDecision, ...]:
        """Accept the immutable proposal without altering judge-authored fields.

        Args:
            _proposal: Current proposal, unused by this single-axis fixture.

        Returns:
            One explicit acceptance.
        """
        return (_accept(),)

    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        (),
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME,
        code_revision="test-revision",
        reviewer=accept,
    )
    review_input = result.audit.trace_reviews[0]
    review_path = store.artifacts.read(review_input.artifact_id).directory / "review.json"
    review_path.write_text("{}", encoding="utf-8")
    preflight_calls = runtime.preflight_calls
    provider_calls = len(client.requests)

    with pytest.raises(ManualJudgeError, match="trace review is unavailable"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, runtime),
            plan,
            (),
            budget,
            spend_consented=False,
            approve=False,
            accept_insufficient_labels=True,
            created_at=_TIME + timedelta(minutes=1),
            code_revision="test-revision",
            reviewer=accept,
        )

    assert runtime.preflight_calls == preflight_calls
    assert len(client.requests) == provider_calls


def _accept() -> ManualJudgeAxisDecision:
    """Return one explicit acceptance for the fixture rubric axis."""
    return ManualJudgeAxisDecision(dimension_id="task-success", accepted=True)


def _commit_setup(store: ProjectStore) -> ManualJudgeSetupArtifact:
    """Persist the default finalized judge setup for one completed build.

    Args:
        store: Selected completed build receiving the finalized setup.

    Returns:
        Immutable finalized manual judge setup.
    """
    setup_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME,
        code_revision="test-revision",
    )
    return commit_manual_judge_setup(store, setup_plan, confirmed=True)


def _runtime(model: ModelSnapshot) -> tuple[_StructuredJudgeClient, _RuntimeCatalog]:
    """Return a deterministic scalar judge client and injected runtime catalog.

    Args:
        model: Exact configured judge snapshot.

    Returns:
        Fake provider client and its runtime resolver.
    """
    client = _StructuredJudgeClient(model, "scalar")
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    return client, runtime


def _budget(
    plan: ManualJudgeCalibrationPlan,
    *,
    completed_review_count: int = 0,
) -> JudgeCalibrationBudget:
    """Return the exact conservative fixture reservation for remaining reviews.

    Args:
        plan: Frozen calibration sample.
        completed_review_count: Reviews that require no more provider reservation.

    Returns:
        Conservative budget for incomplete reviews.
    """
    return estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
        completed_review_count=completed_review_count,
    )


def _five_lineage_store(tmp_path: Path) -> ProjectStore:
    """Create a selected build with five fit lineages and one held-out lineage.

    Args:
        tmp_path: Pytest-owned project root.

    Returns:
        Initialized project store with a selected completed build.
    """
    store = ProjectStore(tmp_path / ".exp", "support")
    store.initialize(ProjectConfig(project_id="support"))
    built = build_project(
        TraceNormalizationResult(traces=tuple(_trace(index) for index in range(100)), issues=()),
        store,
        created_at=_TIME,
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=5, held_out_task_budget=1),
    )
    _persist_grounded_build(store, built, select=True)
    return store
