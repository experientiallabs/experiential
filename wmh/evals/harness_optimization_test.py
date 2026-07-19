"""Tests for the sealed harness-optimization study lifecycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from wmh.agents import default_agent
from wmh.evals.harbor.config import HarborEnvironmentBackend
from wmh.evals.harbor.paired_runner import PairedHarborPanelRoute
from wmh.evals.harness_optimization import (
    BenchmarkProvenance,
    CandidateChangePolicy,
    ConfirmationDecisionRule,
    DiscoverySearchPlan,
    HarnessOptimizationMemberOutcome,
    HarnessOptimizationOutcome,
    HarnessOptimizationProtocol,
    freeze_harness_optimization_candidate,
    open_harness_optimization_confirmation,
    prepare_harness_optimization_study,
    run_harness_optimization_search,
)
from wmh.evals.paired import BoundedMeanBet, PairedPanelPlan
from wmh.evals.partition import (
    BenchmarkPartitionManifest,
    PartitionControlScope,
    PartitionControlStore,
    PartitionTask,
    initialize_partition_genesis,
)
from wmh.harness.create import SearchCheckpoint
from wmh.harness.delta import (
    FailureSignature,
    HarnessDelta,
    SurfaceOp,
    compute_delta_id,
)
from wmh.harness.doc import HarnessDoc, SurfaceKind
from wmh.harness.proposer import ProposalFailure
from wmh.harness.scoring import (
    HarnessScoreReport,
    ScoreCapabilities,
    ScoreRequest,
    ScoreRunHealth,
    TaskScore,
)
from wmh.providers.base import ProviderConfig, ProviderKind


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _roster_digest(manifest: BenchmarkPartitionManifest) -> str:
    payload = json.dumps(
        [task.model_dump(mode="json") for task in manifest.tasks],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _partition(tmp_path: Path) -> tuple[PartitionControlStore, BenchmarkPartitionManifest]:
    control_dir = tmp_path / "partition-control"
    control_dir.mkdir(mode=0o700)
    store = PartitionControlStore(control_dir)
    tasks = tuple(
        PartitionTask(
            task_id=f"task-{index}",
            stratum="shell",
            group_id=f"family-{index}",
            content_digest=_digest(f"task-{index}"),
        )
        for index in range(4)
    )
    genesis = initialize_partition_genesis(
        store,
        scope=PartitionControlScope(
            experiment_id="optimizer-study",
            protocol_id="protocol-v1",
        ),
        tasks=tasks,
        discovery_counts={"shell": 2},
    )
    return store, BenchmarkPartitionManifest.create(
        tasks=tasks,
        discovery_counts={"shell": 2},
        genesis=genesis,
    )


class _Scorer:
    capabilities = ScoreCapabilities(task_subsets=False, attempt_overrides=False)
    default_attempts = 2
    configuration_id = "scorer-config"

    def __init__(self, task_ids: tuple[str, ...]) -> None:
        self.task_ids = task_ids

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        return None

    def before_proposal_batch(self) -> None:
        return None

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        improved = any("wmh-test-improvement" in item.content for item in candidate.code_files())
        score = float(improved)
        per_task = {
            task_id: TaskScore(
                task_id=task_id,
                score=score,
                secondary_score=score,
                passed=improved,
                mechanisms=() if improved else ("agent-control-flow",),
                evidence="synthetic score evidence",
            )
            for task_id in self.task_ids
        }
        identity = hashlib.sha256(
            f"{candidate.execution_hash}:{request.purpose}".encode()
        ).hexdigest()
        return HarnessScoreReport(
            evaluation_id=identity,
            score=score,
            secondary_score=score,
            attempts=self.default_attempts,
            run_health=ScoreRunHealth.VALID,
            per_task=per_task,
        )


class _CodeProposer:
    configuration_id = "proposer-config"
    durable_state_required = False
    score_archive_required = False

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, history, should_cancel
        target = parent.code_files()[0]
        op = SurfaceOp(
            op="replace",
            surface_id=target.id,
            content=target.content + "\n// wmh-test-improvement\n",
            rationale="Exercise a source-code candidate in the lifecycle test.",
        )
        delta = HarnessDelta(
            delta_id=compute_delta_id(parent.doc_hash, [op]),
            parent_doc_hash=parent.doc_hash,
            trigger=trigger,
            preconditions={target.id: target.content_hash},
            ops=[op],
            expected_effect="Every synthetic discovery task changes from failure to success.",
        )
        return [delta.model_copy(deep=True) for _ in range(count)]


class _PromptProposer(_CodeProposer):
    configuration_id = "prompt-proposer-config"

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, history, should_cancel
        target = next(item for item in parent.surfaces if item.kind is SurfaceKind.PROMPT)
        op = SurfaceOp(
            op="replace",
            surface_id=target.id,
            content=target.content + "\nBe careful.",
            rationale="Exercise the source-code policy rejection.",
        )
        delta = HarnessDelta(
            delta_id=compute_delta_id(parent.doc_hash, [op]),
            parent_doc_hash=parent.doc_hash,
            trigger=trigger,
            preconditions={target.id: target.content_hash},
            ops=[op],
            expected_effect="The prompt-only test candidate changes the synthetic score.",
        )
        return [delta.model_copy(deep=True) for _ in range(count)]


def _protocol(
    manifest: BenchmarkPartitionManifest,
    baseline: HarnessDoc,
    *,
    proposer_configuration_id: str = "proposer-config",
    roster_digest: str | None = None,
) -> HarnessOptimizationProtocol:
    panel = tuple(
        PairedPanelPlan(panel_member=member, attempts=15) for member in ("glm", "haiku", "opus")
    )
    routes = tuple(
        PairedHarborPanelRoute(
            panel_member=member,
            provider_config=ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model=f"model-{member}",
                region="us-west-2",
            ),
        )
        for member in ("glm", "haiku", "opus")
    )
    return HarnessOptimizationProtocol.create(
        experiment_id="optimizer-study",
        protocol_id="protocol-v1",
        provenance=BenchmarkProvenance(
            adapter="harbor",
            adapter_version="0.18.0",
            dataset="terminal-benchmark",
            dataset_revision="revision-1",
            roster_digest=roster_digest or _roster_digest(manifest),
        ),
        partition=manifest,
        baseline=baseline,
        search=DiscoverySearchPlan(
            iterations=1,
            proposal_batch_size=1,
            attempts_per_task=2,
            scorer_configuration_id="scorer-config",
            proposer_configuration_id=proposer_configuration_id,
        ),
        candidate_policy=CandidateChangePolicy(minimum_changed_code_surfaces=1),
        confirmation=ConfirmationDecisionRule(
            panel=panel,
            bounded_mean_bets=(BoundedMeanBet(fraction=0.5, weight=1.0),),
            schedule_seed="schedule-seed",
            analysis_seed="analysis-seed",
            randomization_samples=999,
            alpha=0.05,
            minimum_panel_delta=0.03,
            minimum_member_delta=0.03,
            noninferiority_margin=0.0,
        ),
        panel_routes=routes,
        environment_backend=HarborEnvironmentBackend.LOCAL,
        runner_image="ghcr.io/experientiallabs/pi-runner@sha256:" + "1" * 64,
        reward_key="reward",
        turn_timeout_s=300.0,
        max_concurrent_blocks=3,
        retry_policy_digest=_digest("no-retries"),
        budget_policy_digest=_digest("hard-budget"),
    )


def test_search_freeze_and_open_confirmation_without_exposing_heldout_ids(
    tmp_path: Path,
) -> None:
    control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _CodeProposer()
    protocol = _protocol(manifest, baseline)
    prepared = prepare_harness_optimization_study(
        protocol=protocol,
        partition=manifest,
        baseline=baseline,
    )

    public_json = prepared.discovery_contract().model_dump_json()
    assert all(task_id in public_json for task_id in manifest.discovery_task_ids)
    assert all(task_id not in public_json for task_id in manifest.confirmation_task_ids)
    assert manifest.seal_nonce not in public_json

    checkpoints: list[SearchCheckpoint] = []
    result = run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        on_checkpoint=checkpoints.append,
    )
    assert result.best.execution_digest != baseline.execution_digest
    assert checkpoints[-1].completed_iteration == protocol.search.iterations

    frozen = freeze_harness_optimization_candidate(
        control_store,
        prepared=prepared,
        checkpoint=checkpoints[-1],
    )
    assert frozen.candidate == result.best
    assert frozen.freeze_record.confirmation_protocol_digest == protocol.digest
    assert frozen.freeze_record.selection_evidence_digest == frozen.checkpoint_payload_digest
    with pytest.raises(ValueError, match="selection checkpoint"):
        type(frozen).model_validate(
            {
                **frozen.model_dump(mode="json"),
                "checkpoint_payload_digest": _digest("different-checkpoint"),
            }
        )

    opened = open_harness_optimization_confirmation(
        control_store,
        prepared=prepared,
        frozen=frozen,
    )
    assert tuple(task.task_id for task in opened.confirmation.tasks) == tuple(
        sorted(manifest.confirmation_task_ids)
    )
    assert opened.design.panel_members == ("glm", "haiku", "opus")
    assert opened.design.attempts_by_member == {"glm": 15, "haiku": 15, "opus": 15}
    assert opened.confirmation.confirmation_protocol_digest == protocol.digest


def test_freeze_rejects_a_prompt_only_champion_when_code_change_is_required(
    tmp_path: Path,
) -> None:
    control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _PromptProposer()
    protocol = _protocol(
        manifest,
        baseline,
        proposer_configuration_id=proposer.configuration_id,
    )
    prepared = prepare_harness_optimization_study(
        protocol=protocol,
        partition=manifest,
        baseline=baseline,
    )
    checkpoints: list[SearchCheckpoint] = []
    run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        on_checkpoint=checkpoints.append,
    )

    with pytest.raises(ValueError, match="code surface"):
        freeze_harness_optimization_candidate(
            control_store,
            prepared=prepared,
            checkpoint=checkpoints[-1],
        )


def test_search_rejects_runtime_component_drift_before_scoring(tmp_path: Path) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared = prepare_harness_optimization_study(
        protocol=_protocol(manifest, baseline),
        partition=manifest,
        baseline=baseline,
    )
    scorer = _Scorer(tuple(reversed(manifest.discovery_task_ids)))

    with pytest.raises(ValueError, match="task matrix"):
        run_harness_optimization_search(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            on_checkpoint=lambda _checkpoint: None,
        )


def test_protocol_rejects_a_caller_asserted_roster_digest(tmp_path: Path) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    with pytest.raises(ValueError, match="roster_digest"):
        _protocol(
            manifest,
            baseline,
            roster_digest=_digest("caller-asserted-roster"),
        )


def test_compact_outcome_requires_every_predeclared_lane_to_pass() -> None:
    members = tuple(
        HarnessOptimizationMemberOutcome(
            panel_member=member,
            delta=0.04,
            simultaneous_lower_bound=0.001,
            minimum_required_delta=0.03,
            passed=True,
        )
        for member in ("glm", "haiku", "opus")
    )
    outcome = HarnessOptimizationOutcome(
        protocol_digest=_digest("protocol"),
        paired_protocol_digest=_digest("paired-protocol"),
        paired_report_digest=_digest("paired-report"),
        baseline_execution_digest=_digest("baseline"),
        candidate_execution_digest=_digest("candidate"),
        panel_delta=0.04,
        minimum_required_panel_delta=0.03,
        panel_passed=True,
        members=members,
        passed=True,
    )
    assert outcome.passed

    with pytest.raises(ValueError, match="frozen decisions"):
        HarnessOptimizationOutcome.model_validate(
            {**outcome.model_dump(mode="json"), "passed": False}
        )

    panel_failure = HarnessOptimizationOutcome.model_validate(
        {
            **outcome.model_dump(mode="json"),
            "panel_delta": 0.02,
            "panel_passed": False,
            "passed": False,
        }
    )
    assert not panel_failure.passed

    with pytest.raises(ValueError, match="frozen decisions"):
        HarnessOptimizationOutcome.model_validate(
            {**panel_failure.model_dump(mode="json"), "passed": True}
        )
