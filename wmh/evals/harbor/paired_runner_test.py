"""Tests for exact paired Harbor execution and evidence admission."""

from __future__ import annotations

import asyncio
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import pytest
from harbor.models.job.config import DatasetConfig

import wmh.evals.harbor.paired_runner as mod
from wmh.evals.benchmark import (
    BenchmarkCandidateOutcome,
    BenchmarkCandidateStatus,
    BenchmarkCell,
    BenchmarkRunHealth,
    BenchmarkRunResult,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
)
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.results import HarborTrialLocator, LoadedHarborJobResult
from wmh.evals.paired import PairedArm, PairedEvaluationDesign, PairedPanelPlan
from wmh.evals.partition import ConfirmationPartition, PartitionTask
from wmh.harness.doc import HarnessDoc, Surface
from wmh.harness.pi_local import PI_CONTAINER_IMAGE
from wmh.harness.pi_runner import pi_node_baseline
from wmh.providers.base import ProviderConfig, ProviderKind

_TASK_IDS = ("task-a", "task-b")
_TASK_KEYS = {
    "task-a": "sha256:" + "a" * 64,
    "task-b": "sha256:" + "b" * 64,
}
_CONTENT_DIGESTS = {
    "task-a": "sha256:" + "c" * 64,
    "task-b": "sha256:" + "d" * 64,
}
_ENVIRONMENT_DIGESTS = {
    "task-a": "sha256:" + "e" * 64,
    "task-b": "sha256:" + "f" * 64,
}
_CONFIG_DIGEST = "sha256:" + "1" * 64
_RETRY_POLICY_DIGEST = "sha256:" + "5" * 64
_BUDGET_POLICY_DIGEST = "sha256:" + "6" * 64


def _candidate() -> HarnessDoc:
    baseline = pi_node_baseline("candidate")
    surfaces = list(baseline.surfaces)
    prompt_index = next(
        index for index, surface in enumerate(surfaces) if surface.id == "prompt:core"
    )
    prompt = surfaces[prompt_index]
    surfaces[prompt_index] = Surface.model_validate(
        {**prompt.model_dump(), "content": prompt.content + "\nKeep a concise task ledger."}
    )
    return HarnessDoc(name="candidate", surfaces=surfaces)


def _design() -> PairedEvaluationDesign:
    return PairedEvaluationDesign.create(
        task_ids=_TASK_IDS,
        panel=(PairedPanelPlan(panel_member="worker", attempts=2),),
        schedule_seed="paired-schedule-v1",
        analysis_seed="paired-analysis-v1",
        randomization_samples=1_000,
        bootstrap_samples=1_000,
        minimum_panel_delta=0.05,
        minimum_member_delta=0.03,
        noninferiority_margin=0.02,
    )


def _confirmation(candidate: HarnessDoc) -> ConfirmationPartition:
    return ConfirmationPartition(
        partition_version="1",
        partition_manifest_digest="sha256:" + "3" * 64,
        candidate_hash=candidate.execution_digest,
        tasks=tuple(
            PartitionTask(
                task_id=task_id,
                stratum="held-out",
                group_id=task_id,
                content_digest=_CONTENT_DIGESTS[task_id],
            )
            for task_id in _TASK_IDS
        ),
        confirmation_commitment="sha256:" + "4" * 64,
    )


def _provider() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="worker-model",
        region="us-west-2",
    )


def _spec(tmp_path: Path) -> HarborJobSpec:
    return HarborJobSpec(
        job_name="paired-template",
        jobs_dir=tmp_path / "jobs",
        datasets=[
            DatasetConfig(
                path=tmp_path / "dataset",
                task_names=list(_TASK_IDS),
            )
        ],
        n_attempts=1,
        n_concurrent_trials=1,
        agent_n_concurrent=1,
    )


def _qualifications() -> tuple[mod.QualifiedHarborTask, ...]:
    return tuple(
        mod.QualifiedHarborTask(
            task_id=task_id,
            content_digest=_CONTENT_DIGESTS[task_id],
            task_key=_TASK_KEYS[task_id],
            task_environment_digest=_ENVIRONMENT_DIGESTS[task_id],
        )
        for task_id in _TASK_IDS
    )


def _runner(
    tmp_path: Path,
    candidate: HarnessDoc,
    *,
    baseline: HarnessDoc | None = None,
    **updates: object,
) -> mod.PairedHarborRunner:
    baseline = baseline or pi_node_baseline("baseline")
    protocol_values: dict[str, object] = {
        "design": _design(),
        "confirmation": _confirmation(candidate),
        "baseline": baseline,
        "candidate": candidate,
        "job_spec": _spec(tmp_path),
        "panel_routes": (
            mod.PairedHarborPanelRoute(
                panel_member="worker",
                provider_config=_provider(),
                max_concurrent_blocks=2,
            ),
        ),
        "qualified_tasks": _qualifications(),
        "reward_key": "reward",
        "max_concurrent_blocks": 4,
        "retry_policy_digest": _RETRY_POLICY_DIGEST,
        "budget_policy_digest": _BUDGET_POLICY_DIGEST,
    }
    protocol_values.update(updates)
    protocol = mod.PairedHarborProtocol.freeze(**cast("Any", protocol_values))
    return mod.PairedHarborRunner(
        protocol=protocol,
        job_spec=protocol_values["job_spec"],  # type: ignore[arg-type]
        operation_id="offline-test-operation",
        generation_id=1,
    )


def _loaded_result(
    spec: HarborJobSpec,
    provider: ProviderConfig,
    harness: HarnessDoc,
    *,
    reward: float,
) -> LoadedHarborJobResult:
    task_names = spec.datasets[0].task_names
    assert task_names is not None
    task_id = task_names[0]
    cell = BenchmarkCell(
        task_key=_TASK_KEYS[task_id],
        task_name=task_id,
        attempt=1,
        config_digest=_CONFIG_DIGEST,
    )
    trial = BenchmarkTrialResult(
        cell=cell,
        task_identity=task_id,
        task_checksum=_CONTENT_DIGESTS[task_id],
        source="test-dataset",
        task_instruction=f"Solve {task_id}.",
        task_environment_digest=_ENVIRONMENT_DIGESTS[task_id],
        status=BenchmarkTrialStatus.SCORED,
        rewards={"reward": reward},
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.COMPLETED,
        ),
        run_health=BenchmarkRunHealth.VALID,
    )
    job_dir = spec.jobs_dir / spec.job_name
    trial_dir = Path("trial")
    (job_dir / trial_dir).mkdir(parents=True, exist_ok=True)
    receipt = mod.PairedProviderCallReceipt(
        call_index=1,
        provider_request_id="provider-" + spec.job_name,
        provider=provider.kind.value,
        provider_config_digest=mod._canonical_digest(provider.model_dump(mode="json")),
        requested_model=provider.model,
        route_evidence_kind="aws_response_metadata",
        response_model=None,
        system_fingerprint=None,
        wire_config_digest=mod.paired_provider_wire_config_digest(
            provider_config=provider,
            requested_model=provider.model,
            temperature=None,
            max_output_tokens=pi_node_baseline("limits").max_output_tokens(),
        ),
        temperature=None,
        max_output_tokens=pi_node_baseline("limits").max_output_tokens(),
    )
    (job_dir / trial_dir / "wmh-events.jsonl").write_text(
        '{"kind":"assistant_message","payload":{"text":"done"}}\n'
        + json.dumps(
            {"kind": "provider_receipt", "payload": receipt.model_dump(mode="json")},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    identity = mod.harbor_run_expectation(
        candidate=harness,
        spec=spec,
        provider_config=provider,
        runner_image=PI_CONTAINER_IMAGE,
        turn_timeout_s=300.0,
    ).identity
    return LoadedHarborJobResult(
        result=BenchmarkRunResult(
            job_name=spec.job_name,
            identity=identity,
            expected_cells=[cell],
            trials=[trial],
        ),
        job_dir=job_dir,
        locators=(
            HarborTrialLocator(
                cell=cell,
                trial_dir=trial_dir,
                result_path=trial_dir / "result.json",
                artifacts_dir=trial_dir / "artifacts",
            ),
        ),
    )


def _install_fake_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate: HarnessDoc,
    fail_job: int | None = None,
) -> tuple[
    list[tuple[HarborJobSpec, ProviderConfig, HarnessDoc]],
    dict[str, int],
]:
    calls: list[tuple[HarborJobSpec, ProviderConfig, HarnessDoc]] = []
    active: defaultdict[str, int] = defaultdict(int)
    maximum: defaultdict[str, int] = defaultdict(int)

    class FakeEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            *,
            runner_image: str,
            turn_timeout_s: float,
            session: object,
        ) -> None:
            assert runner_image == PI_CONTAINER_IMAGE
            assert turn_timeout_s == 300.0
            assert isinstance(session, mod.HarborEvaluatorSession)
            self._spec = spec
            self._provider = provider_config

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            task_names = self._spec.datasets[0].task_names
            assert task_names is not None
            task_id = task_names[0]
            active[task_id] += 1
            maximum[task_id] = max(maximum[task_id], active[task_id])
            try:
                await asyncio.sleep(0)
                calls.append((self._spec, self._provider, harness))
                if fail_job is not None and len(calls) == fail_job:
                    raise RuntimeError("synthetic infrastructure failure")
                reward = float(harness.execution_digest == candidate.execution_digest)
                return _loaded_result(
                    self._spec,
                    self._provider,
                    harness,
                    reward=reward,
                )
            finally:
                active[task_id] -= 1

    monkeypatch.setattr(mod, "HarborEvaluator", FakeEvaluator)
    return calls, maximum


def test_runs_every_frozen_block_in_order_and_analyzes_exact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, maximum = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(tmp_path, candidate, baseline=baseline)

    report = asyncio.run(runner.run(baseline=baseline, candidate=candidate))

    design = _design()
    assert len(calls) == 2 * len(design.blocks)
    assert len(report.evidence) == len(design.blocks)
    assert report.protocol.design_digest == design.digest
    assert report.protocol.baseline_execution_digest == baseline.execution_digest
    assert report.protocol.candidate_execution_digest == candidate.execution_digest
    assert report.analysis.panel_delta == 1.0
    assert all(item.outcome.baseline_reward == 0.0 for item in report.evidence)
    assert all(item.outcome.candidate_reward == 1.0 for item in report.evidence)
    assert all(item.first.arm is item.block.first_arm for item in report.evidence)
    assert all(item.first.job_name != item.second.job_name for item in report.evidence)
    assert all(value == 1 for value in maximum.values())
    assert report.digest.startswith("sha256:")

    for spec, provider, _harness in calls:
        assert spec.n_attempts == 1
        assert spec.n_concurrent_trials == 1
        assert spec.agent_n_concurrent == 1
        assert len(spec.datasets) == 1
        assert len(spec.datasets[0].task_names or []) == 1
        assert provider.model_dump() == _provider().model_dump()


def test_job_names_are_deterministic_and_bind_each_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    first_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    first = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    first_names = [
        item.job_name
        for evidence in first.evidence
        for item in (evidence.first, evidence.second)
    ]

    second_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    second = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    second_names = [
        item.job_name for evidence in second.evidence for item in (evidence.first, evidence.second)
    ]

    assert first_names == second_names
    assert len(set(first_names)) == len(first_names)
    assert [call[0].job_name for call in first_calls] == [call[0].job_name for call in second_calls]


def test_rejects_candidate_opening_and_compute_envelope_mismatches(tmp_path: Path) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    wrong_opening = _confirmation(candidate).model_copy(
        update={"candidate_hash": "sha256:" + "9" * 64}
    )

    with pytest.raises(ValueError, match="confirmation opening"):
        asyncio.run(
            _runner(tmp_path, candidate, confirmation=wrong_opening).run(
                baseline=baseline,
                candidate=candidate,
            )
        )

    changed_surfaces = [
        (
            Surface.model_validate({**surface.model_dump(), "content": "99"})
            if surface.id == "param:max-turns"
            else surface
        )
        for surface in candidate.surfaces
    ]
    changed_compute = HarnessDoc(name="candidate", surfaces=changed_surfaces)
    with pytest.raises(ValueError, match="compute envelope"):
        asyncio.run(
            _runner(tmp_path, changed_compute, baseline=baseline).run(
                baseline=baseline,
                candidate=changed_compute,
            )
        )


def test_constructor_rejects_qualification_or_route_drift(tmp_path: Path) -> None:
    candidate = _candidate()
    confirmation = _confirmation(candidate)
    with pytest.raises(ValueError, match="opened confirmation tasks"):
        _runner(
            tmp_path,
            candidate,
            confirmation=confirmation.model_copy(
                update={"tasks": tuple(reversed(confirmation.tasks))}
            ),
        )

    qualifications = list(_qualifications())
    qualifications[0] = qualifications[0].model_copy(
        update={"content_digest": "sha256:" + "8" * 64}
    )
    with pytest.raises(ValueError, match="qualification content"):
        _runner(tmp_path, candidate, qualified_tasks=tuple(qualifications))

    duplicate_routes = (
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
        ),
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
        ),
    )
    with pytest.raises(ValueError, match="duplicate routes"):
        _runner(tmp_path, candidate, panel_routes=duplicate_routes)


def test_collects_block_failures_without_returning_partial_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate, fail_job=1)

    with pytest.raises(mod.PairedHarborMatrixError) as captured:
        asyncio.run(
            _runner(tmp_path, candidate, baseline=baseline).run(
                baseline=baseline,
                candidate=candidate,
            )
        )

    assert captured.value.failures
    assert len(calls) >= len(_design().blocks)
    assert "synthetic infrastructure failure" not in str(captured.value)


def test_schedule_has_both_first_arm_directions() -> None:
    counts = Counter(block.first_arm for block in _design().blocks)
    assert counts == {PairedArm.BASELINE: 2, PairedArm.CANDIDATE: 2}


def test_protocol_digest_is_path_independent_and_binds_nonsecret_execution_inputs(
    tmp_path: Path,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    first = _runner(tmp_path / "host-a", candidate, baseline=baseline)._protocol
    second = _runner(tmp_path / "host-b", candidate, baseline=baseline)._protocol

    assert first.job_template == second.job_template
    assert first.digest == second.digest

    changed_route = first.panel_routes[0].model_copy(update={"max_concurrent_blocks": 1})
    changed = mod.PairedHarborProtocol.freeze(
        design=_design(),
        confirmation=_confirmation(candidate),
        baseline=baseline,
        candidate=candidate,
        job_spec=_spec(tmp_path / "host-c"),
        panel_routes=(changed_route,),
        qualified_tasks=_qualifications(),
        reward_key="reward",
        max_concurrent_blocks=4,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=_BUDGET_POLICY_DIGEST,
    )
    assert changed.digest != first.digest

    changed_budget = first.model_copy(
        update={"budget_policy_digest": "sha256:" + "7" * 64}
    )
    assert changed_budget.digest != first.digest


def test_rejects_scored_e2b_even_when_job_shape_is_otherwise_exact(tmp_path: Path) -> None:
    candidate = _candidate()
    e2b = _spec(tmp_path).model_copy(
        update={"environment_backend": HarborEnvironmentBackend.E2B}
    )
    with pytest.raises(ValueError, match="scored paired E2B is unsupported"):
        _runner(tmp_path, candidate, job_spec=e2b)


def test_provider_receipt_contract_distinguishes_bedrock_and_openai_evidence() -> None:
    with pytest.raises(ValueError, match="Bedrock Converse does not return"):
        mod.PairedHarborPanelRoute(
            panel_member="bedrock",
            provider_config=_provider(),
            expected_response_model="fabricated-model",
        )

    azure = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="glm-model-family",
        deployment="glm-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    with pytest.raises(ValueError, match="require an expected response model"):
        mod.PairedHarborPanelRoute(panel_member="azure", provider_config=azure)
    route = mod.PairedHarborPanelRoute(
        panel_member="azure",
        provider_config=azure,
        expected_response_model="glm-served-model",
        expected_system_fingerprint="fp-123",
    )
    assert route.expected_response_model == "glm-served-model"

    with pytest.raises(ValueError, match="OpenAI response receipt requires"):
        mod.PairedProviderCallReceipt(
            call_index=1,
            provider_request_id="response-id",
            provider=ProviderKind.AZURE_OPENAI.value,
            provider_config_digest="sha256:" + "1" * 64,
            requested_model="glm-deployment",
            route_evidence_kind="openai_response",
            response_model=None,
            system_fingerprint=None,
            wire_config_digest="sha256:" + "2" * 64,
            temperature=None,
            max_output_tokens=1_000,
        )


def test_job_and_pair_generation_identities_bind_operation_generation_and_block(
    tmp_path: Path,
) -> None:
    protocol = _runner(tmp_path, _candidate())._protocol
    first, second = protocol.design.blocks[:2]
    pair = mod.paired_harbor_pair_generation_id(
        protocol_digest=protocol.digest,
        operation_id="operation-a",
        generation_id=1,
        block=first,
    )
    assert pair != mod.paired_harbor_pair_generation_id(
        protocol_digest=protocol.digest,
        operation_id="operation-a",
        generation_id=2,
        block=first,
    )
    assert pair != mod.paired_harbor_pair_generation_id(
        protocol_digest=protocol.digest,
        operation_id="operation-a",
        generation_id=1,
        block=second,
    )
    names = {
        mod.paired_harbor_job_name(
            protocol_digest=protocol.digest,
            operation_id=operation,
            generation_id=generation,
            block=first,
            arm=arm,
        )
        for operation in ("operation-a", "operation-b")
        for generation in (1, 2)
        for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
    }
    assert len(names) == 8


def test_partial_existing_pair_is_rejected_before_any_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(tmp_path, candidate, baseline=baseline)
    block = runner._protocol.design.blocks[0]
    one_arm = mod.paired_harbor_job_name(
        protocol_digest=runner._protocol.digest,
        operation_id="offline-test-operation",
        generation_id=1,
        block=block,
        arm=PairedArm.BASELINE,
    )
    (runner._job_spec.jobs_dir / one_arm).mkdir(parents=True)

    with pytest.raises(mod.PartialPairedHarborReuseError, match="only one pre-existing arm"):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))
    assert calls == []


def test_realistic_trace_without_provider_authored_receipt_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()

    class ReceiptlessEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **_kwargs: object,
        ) -> None:
            self._spec = spec
            self._provider = provider_config

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(self._spec, self._provider, harness, reward=1.0)
            trace = loaded.job_dir / loaded.locators[0].trial_dir / "wmh-events.jsonl"
            trace.write_text(
                '{"kind":"assistant_message","payload":{"text":"done"}}\n',
                encoding="utf-8",
            )
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", ReceiptlessEvaluator)
    with pytest.raises(mod.PairedHarborMatrixError) as captured:
        asyncio.run(
            _runner(tmp_path, candidate, baseline=baseline).run(
                baseline=baseline,
                candidate=candidate,
            )
        )
    assert any(
        "lacks provider-authored request receipts" in str(error)
        for _, error in captured.value.failures
    )


def test_report_json_reload_recomputes_every_binding_and_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    report = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    canonical = cast("dict[str, Any]", report.model_dump(mode="json"))
    assert mod.PairedHarborRunReport.model_validate_json(
        json.dumps(canonical)
    ) == report

    mutations = []

    def change_job(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["job_name"] += "-tampered"

    mutations.append(change_job)

    def change_task(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["trial"]["task_checksum"] = "sha256:" + "9" * 64

    mutations.append(change_task)

    def change_cell_config(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["trial"]["cell"]["config_digest"] = (
            "sha256:" + "7" * 64
        )

    mutations.append(change_cell_config)

    def change_environment(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["trial"]["task_environment_digest"] = (
            "sha256:" + "8" * 64
        )

    mutations.append(change_environment)

    def change_route(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["run_identity"]["model_name"] = "other"

    mutations.append(change_route)

    def change_score(payload: dict[str, Any]) -> None:
        current = payload["evidence"][0]["first"]["analysis_score"]
        payload["evidence"][0]["first"]["analysis_score"] = 1.0 - current

    mutations.append(change_score)

    def change_analysis(payload: dict[str, Any]) -> None:
        payload["analysis"]["panel_delta"] = 0.5

    mutations.append(change_analysis)

    def duplicate_request_id(payload: dict[str, Any]) -> None:
        first_id = payload["evidence"][0]["first"]["provider_receipts"][0][
            "provider_request_id"
        ]
        payload["evidence"][0]["second"]["provider_receipts"][0][
            "provider_request_id"
        ] = first_id

    mutations.append(duplicate_request_id)

    def fabricate_bedrock_response_model(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipts"][0][
            "response_model"
        ] = "unfrozen-model"

    mutations.append(fabricate_bedrock_response_model)

    def add_seed(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipts"][0]["seed"] = 7

    mutations.append(add_seed)

    def change_temperature(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipts"][0][
            "temperature"
        ] = 0.7

    mutations.append(change_temperature)

    def change_pair_generation(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["pair_generation_id"] = "sha256:" + "8" * 64

    mutations.append(change_pair_generation)

    def reverse_blocks(payload: dict[str, Any]) -> None:
        payload["evidence"] = list(reversed(payload["evidence"]))

    mutations.append(reverse_blocks)

    for mutate in mutations:
        payload = copy.deepcopy(canonical)
        mutate(payload)
        with pytest.raises(ValueError):
            mod.PairedHarborRunReport.model_validate(payload)


def test_scheduler_is_bounded_route_fair_and_serializes_each_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    design = PairedEvaluationDesign.create(
        task_ids=_TASK_IDS,
        panel=(
            PairedPanelPlan(panel_member="route-a", attempts=2),
            PairedPanelPlan(panel_member="route-b", attempts=2),
        ),
        schedule_seed="fair-schedule",
        analysis_seed="fair-analysis",
        randomization_samples=1_000,
        bootstrap_samples=1_000,
        minimum_panel_delta=0.05,
        minimum_member_delta=0.03,
        noninferiority_margin=0.02,
    )
    providers = {
        "worker-a": ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="worker-a",
            region="us-west-2",
        ),
        "worker-b": ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="worker-b",
            region="us-west-2",
        ),
    }
    routes = (
        mod.PairedHarborPanelRoute(
            panel_member="route-a",
            provider_config=providers["worker-a"],
            max_concurrent_blocks=1,
        ),
        mod.PairedHarborPanelRoute(
            panel_member="route-b",
            provider_config=providers["worker-b"],
            max_concurrent_blocks=1,
        ),
    )
    protocol = mod.PairedHarborProtocol.freeze(
        design=design,
        confirmation=_confirmation(candidate),
        baseline=baseline,
        candidate=candidate,
        job_spec=_spec(tmp_path),
        panel_routes=routes,
        qualified_tasks=_qualifications(),
        reward_key="reward",
        max_concurrent_blocks=2,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=_BUDGET_POLICY_DIGEST,
    )

    active_global = 0
    maximum_global = 0
    active_routes: Counter[str] = Counter()
    maximum_routes: Counter[str] = Counter()
    active_tasks: Counter[str] = Counter()
    maximum_tasks: Counter[str] = Counter()
    starts: list[tuple[str, str]] = []

    class FairEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **_kwargs: object,
        ) -> None:
            self._spec = spec
            self._provider = provider_config

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            nonlocal active_global, maximum_global
            task_names = self._spec.datasets[0].task_names
            assert task_names is not None
            task_id = task_names[0]
            model = self._provider.model
            starts.append((model, task_id))
            active_global += 1
            active_routes[model] += 1
            active_tasks[task_id] += 1
            maximum_global = max(maximum_global, active_global)
            maximum_routes[model] = max(maximum_routes[model], active_routes[model])
            maximum_tasks[task_id] = max(maximum_tasks[task_id], active_tasks[task_id])
            try:
                await asyncio.sleep(0.001)
                reward = float(harness.execution_digest == candidate.execution_digest)
                return _loaded_result(
                    self._spec,
                    self._provider,
                    harness,
                    reward=reward,
                )
            finally:
                active_global -= 1
                active_routes[model] -= 1
                active_tasks[task_id] -= 1

    monkeypatch.setattr(mod, "HarborEvaluator", FairEvaluator)
    report = asyncio.run(
        mod.PairedHarborRunner(
            protocol=protocol,
            job_spec=_spec(tmp_path),
            operation_id="fair-operation",
            generation_id=1,
        ).run(baseline=baseline, candidate=candidate)
    )

    assert len(report.evidence) == len(design.blocks)
    assert maximum_global == 2
    assert maximum_routes == {"worker-a": 1, "worker-b": 1}
    assert all(value == 1 for value in maximum_tasks.values())
    assert {model for model, _task in starts[:2]} == {"worker-a", "worker-b"}
    assert starts[0][1] != starts[1][1]
