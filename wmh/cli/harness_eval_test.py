"""CLI tests for ground-truth Harbor harness evaluation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner, Result

from wmh.cli import app
from wmh.evals.benchmark import (
    BenchmarkCandidateFailureReason,
    BenchmarkCandidateOutcome,
    BenchmarkCandidateStage,
    BenchmarkCandidateStatus,
    BenchmarkCell,
    BenchmarkError,
    BenchmarkFailureKind,
    BenchmarkRunIdentity,
    BenchmarkRunResult,
    BenchmarkTaskEnvironment,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
)
from wmh.evals.harbor import _file_lease
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.results import LoadedHarborJobResult
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_runner import pi_node_baseline
from wmh.harness.pi_runner_backend import LocalPiRunnerSpec, PiRunnerBackendSpec
from wmh.harness.store import HarnessStore
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    TokenPriceCeiling,
    bootstrap_budget_ledger,
)

harness_eval_module = importlib.import_module("wmh.cli.harness_eval")
runner = CliRunner()
_RUN_CONFIG_DIGEST = "sha256:" + "a" * 64
_CELL_CONFIG_DIGEST = "sha256:" + "b" * 64
_RUNNER_CONFIG_DIGEST = "sha256:" + "c" * 64
_RUNNER_ENVIRONMENT_DIGEST = "sha256:" + "d" * 64


def _save_harness(root: Path) -> HarnessDoc:
    return HarnessStore(root).save_version(pi_node_baseline("agent"), alias="champion")


def _loaded_result(tmp_path: Path) -> LoadedHarborJobResult:
    identity = BenchmarkRunIdentity(
        candidate_hash="candidate-hash",
        agent_name="wmh-pi",
        agent_version="0.1.0",
        provider="bedrock",
        model_name="model",
        task_environment=BenchmarkTaskEnvironment.DOCKER,
        runner_config_digest=_RUNNER_CONFIG_DIGEST,
        runner_environment_digest=_RUNNER_ENVIRONMENT_DIGEST,
        run_config_digest=_RUN_CONFIG_DIGEST,
    )
    cells = [
        BenchmarkCell(
            task_key=f"task-{index}",
            task_name=f"task-{index}",
            attempt=1,
            config_digest=_CELL_CONFIG_DIGEST,
        )
        for index in range(4)
    ]
    trials = [
        BenchmarkTrialResult(
            cell=cells[0],
            task_identity="task-0",
            task_checksum="checksum-0",
            source="benchmark",
            status=BenchmarkTrialStatus.SCORED,
            rewards={"custom_score": 0, "partial": 0.5},
            candidate_outcome=BenchmarkCandidateOutcome(
                status=BenchmarkCandidateStatus.FAILED,
                stage=BenchmarkCandidateStage.EXECUTION,
                failure_reason=BenchmarkCandidateFailureReason.RUNTIME_ERROR,
            ),
        ),
        BenchmarkTrialResult(
            cell=cells[1],
            task_identity="task-1",
            task_checksum="checksum-1",
            source="benchmark",
            status=BenchmarkTrialStatus.TASK_TIMEOUT,
            error=BenchmarkError(
                kind=BenchmarkFailureKind.TASK_TIMEOUT,
                type="AgentTimeoutError",
                message="timed out",
            ),
        ),
        BenchmarkTrialResult(
            cell=cells[2],
            task_identity="task-2",
            task_checksum="checksum-2",
            source="benchmark",
            status=BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            error=BenchmarkError(
                kind=BenchmarkFailureKind.PROVIDER,
                type="WmhPiProviderError",
                message="provider failed",
            ),
        ),
        BenchmarkTrialResult(
            cell=cells[3],
            task_identity="task-3",
            task_checksum="checksum-3",
            source="benchmark",
            status=BenchmarkTrialStatus.INCOMPLETE,
        ),
    ]
    result = BenchmarkRunResult(
        job_name="job",
        identity=identity,
        expected_cells=cells,
        trials=trials,
    )
    return LoadedHarborJobResult(result=result, job_dir=tmp_path / "private-job", locators=())


def _all_scored_result(tmp_path: Path) -> LoadedHarborJobResult:
    mixed = _loaded_result(tmp_path)
    trials = [
        BenchmarkTrialResult(
            cell=trial.cell,
            task_identity=trial.task_identity,
            task_checksum=trial.task_checksum,
            source=trial.source,
            status=BenchmarkTrialStatus.SCORED,
            rewards={"custom_score": 0 if index == 0 else 1},
            error=(
                BenchmarkError(
                    kind=BenchmarkFailureKind.TASK_TIMEOUT,
                    type="AgentTimeoutError",
                    message="timed out after verifier-visible work",
                )
                if index == 0
                else None
            ),
        )
        for index, trial in enumerate(mixed.result.trials)
    ]
    result = BenchmarkRunResult(
        job_name=mixed.result.job_name,
        identity=mixed.result.identity,
        expected_cells=mixed.result.expected_cells,
        trials=trials,
    )
    return LoadedHarborJobResult(result=result, job_dir=mixed.job_dir, locators=())


def _all_infrastructure_result(tmp_path: Path) -> LoadedHarborJobResult:
    mixed = _loaded_result(tmp_path)
    trials = [
        BenchmarkTrialResult(
            cell=trial.cell,
            task_identity=trial.task_identity,
            task_checksum=trial.task_checksum,
            source=trial.source,
            status=BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            error=BenchmarkError(
                kind=BenchmarkFailureKind.ENVIRONMENT,
                type="EnvironmentStartTimeoutError",
                message="environment failed",
            ),
        )
        for trial in mixed.result.trials
    ]
    result = BenchmarkRunResult(
        job_name=mixed.result.job_name,
        identity=mixed.result.identity,
        expected_cells=mixed.result.expected_cells,
        trials=trials,
    )
    return LoadedHarborJobResult(result=result, job_dir=mixed.job_dir, locators=())


def _cancelled_result(tmp_path: Path) -> LoadedHarborJobResult:
    mixed = _loaded_result(tmp_path)
    cancelled = BenchmarkTrialResult(
        cell=mixed.result.trials[3].cell,
        task_identity="task-3",
        task_checksum="checksum-3",
        source="benchmark",
        status=BenchmarkTrialStatus.CANCELLED,
        error=BenchmarkError(
            kind=BenchmarkFailureKind.CANCELLED,
            type="CancelledError",
            message="cancelled",
        ),
    )
    result = BenchmarkRunResult(
        job_name=mixed.result.job_name,
        identity=mixed.result.identity,
        expected_cells=mixed.result.expected_cells,
        trials=[*mixed.result.trials[:3], cancelled],
    )
    return LoadedHarborJobResult(result=result, job_dir=mixed.job_dir, locators=())


def _empty_result(tmp_path: Path) -> LoadedHarborJobResult:
    mixed = _loaded_result(tmp_path)
    result = BenchmarkRunResult(
        job_name=mixed.result.job_name,
        identity=mixed.result.identity,
        expected_cells=[],
        trials=[],
    )
    return LoadedHarborJobResult(result=result, job_dir=mixed.job_dir, locators=())


def _patch_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    loaded: LoadedHarborJobResult,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class FakeEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            *,
            runner_spec: PiRunnerBackendSpec,
            turn_timeout_s: float,
            budget_account: BudgetAccount | None = None,
        ) -> None:
            self._call: dict[str, object] = {
                "spec": spec,
                "provider_config": provider_config,
                "runner_spec": runner_spec,
                "turn_timeout_s": turn_timeout_s,
                "budget_account": budget_account,
            }

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            self._call["candidate"] = candidate
            calls.append(self._call)
            return loaded

    monkeypatch.setattr(harness_eval_module, "HarborEvaluator", FakeEvaluator)
    return calls


def _base_args(
    root: Path,
    out: Path,
    *,
    bedrock_region: str = "us-east-1",
) -> list[str]:
    return [
        "harness",
        "eval",
        "agent@v1",
        "--provider",
        "bedrock",
        "--model",
        "model",
        "--bedrock-region",
        bedrock_region,
        "--root",
        str(root),
        "--out",
        str(out),
        "--allow-unbudgeted-development",
    ]


def _write_budget_account(path: Path, provider_config: ProviderConfig) -> BudgetAccount:
    policy = BudgetPolicy(
        study_id="cli-study",
        manifest_digest="sha256:" + "c" * 64,
        hard_limit_nano_usd=1_000_000,
        phase_limits_nano_usd={"qualification": 1_000_000},
        meters={
            "worker": ProviderCostMeter(
                provider_config=provider_config,
                price=TokenPriceCeiling(
                    input_nano_usd_per_token=1,
                    output_nano_usd_per_token=5,
                ),
            )
        },
    )
    ledger_path = (path.parent / "spend.sqlite3").resolve()
    account = BudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=bootstrap_budget_ledger(ledger_path, policy).ledger_identity,
        policy=policy,
        scope=BudgetScope(
            phase="qualification",
            category="worker",
            run_id="cli-run",
        ),
        meter_id="worker",
    )
    path.write_text(account.model_dump_json(indent=2), encoding="utf-8")
    return account


def test_paid_eval_requires_budget_or_explicit_development_bypass(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    args = _base_args(root, out)
    args.remove("--allow-unbudgeted-development")

    result = runner.invoke(
        app,
        [*args, "--dataset-path", str(dataset), "--yes"],
    )

    assert result.exit_code == 2
    assert "paid harness evaluation requires --budget-account" in result.output


def test_budget_account_is_loaded_and_wired_into_the_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    provider_config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="model",
        region="us-east-1",
    )
    account_path = tmp_path / "budget-account.json"
    account = _write_budget_account(account_path, provider_config)
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))
    args = _base_args(root, out)
    args.remove("--allow-unbudgeted-development")

    result = runner.invoke(
        app,
        [
            *args,
            "--dataset-path",
            str(dataset),
            "--budget-account",
            str(account_path),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    [call] = calls
    assert call["budget_account"] == account


def test_local_bedrock_eval_wires_exact_inputs_and_writes_only_canonical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    saved = _save_harness(root)
    dataset = tmp_path / "local-benchmark"
    dataset.mkdir()
    out = tmp_path / "reports" / "result.json"
    out.parent.mkdir()
    out.write_text("old", encoding="utf-8")
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))
    runner_spec_path = tmp_path / "runner.json"
    runner_spec_path.write_text(
        LocalPiRunnerSpec().model_dump_json(),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            *_base_args(root, out, bedrock_region="us-west-2"),
            "--dataset-path",
            str(dataset),
            "--task",
            "task-b",
            "--task",
            "task-a",
            "--task",
            "task-b",
            "--attempts",
            "3",
            "--concurrency",
            "2",
            "--job-name",
            "candidate-eval",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--turn-timeout",
            "42",
            "--runner-spec",
            str(runner_spec_path),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    [call] = calls
    candidate = cast("HarnessDoc", call["candidate"])
    assert candidate.name == saved.name
    assert candidate.version == 1
    spec = cast("HarborJobSpec", call["spec"])
    assert spec.datasets[0].path == dataset.resolve()
    assert spec.datasets[0].task_names == ["task-b", "task-a"]
    assert spec.datasets[0].exclude_task_names is None
    assert spec.n_attempts == 3
    assert spec.n_concurrent_trials == 2
    assert spec.job_name == "candidate-eval"
    assert spec.jobs_dir == (tmp_path / "jobs").resolve()
    assert spec.environment_backend is HarborEnvironmentBackend.LOCAL
    provider = cast("ProviderConfig", call["provider_config"])
    assert provider.kind is ProviderKind.BEDROCK
    assert provider.model == "model"
    assert provider.region == "us-west-2"
    assert provider.endpoint is None
    assert provider.deployment is None
    assert call["turn_timeout_s"] == 42.0
    assert call["runner_spec"] == LocalPiRunnerSpec()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["identity"]["run_config_digest"] == _RUN_CONFIG_DIGEST
    assert {trial["cell"]["config_digest"] for trial in payload["trials"]} == {_CELL_CONFIG_DIGEST}
    assert payload["trials"][0]["rewards"] == {"custom_score": 0}
    assert payload["trials"][0]["error"]["kind"] == "task_timeout"
    assert "job_dir" not in payload
    assert "locators" not in payload
    assert "provider_config" not in payload
    flat = " ".join(result.output.split())
    assert "scored=4" in flat
    assert "task_timeout=1" in flat
    assert "candidate_failed=0" in flat
    assert "candidate_timeout=0" in flat
    assert "candidate_resource_limit=0" in flat
    assert "candidate_runtime_error=0" in flat
    assert "candidate_unclassified=0" in flat
    assert "infra=0" in flat
    assert "incomplete=0" in flat
    assert "resolved task digests=4" in flat


def test_registry_azure_eval_wires_ref_endpoint_deployment_and_e2b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    out = tmp_path / "result.json"
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))
    dataset_ref = "sha256:" + "b" * 64

    result = runner.invoke(
        app,
        [
            "harness",
            "eval",
            "agent@champion",
            "--dataset",
            "org/benchmark",
            "--dataset-ref",
            dataset_ref,
            "--exclude-task",
            "broken-*",
            "--exclude-task",
            "broken-*",
            "--provider",
            "azure",
            "--model",
            "gpt-model",
            "--azure-endpoint",
            "https://example.openai.azure.com",
            "--azure-deployment",
            "deployment-a",
            "--azure-api-version",
            "2026-01-01-preview",
            "--task-backend",
            "e2b",
            "--root",
            str(root),
            "--out",
            str(out),
            "--allow-unbudgeted-development",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    [call] = calls
    spec = cast("HarborJobSpec", call["spec"])
    assert spec.datasets[0].name == "org/benchmark"
    assert spec.datasets[0].ref == dataset_ref
    assert spec.datasets[0].task_names is None
    assert spec.datasets[0].exclude_task_names == ["broken-*"]
    assert spec.environment_backend is HarborEnvironmentBackend.E2B
    provider = cast("ProviderConfig", call["provider_config"])
    assert provider.kind is ProviderKind.AZURE_OPENAI
    assert provider.model == "gpt-model"
    assert provider.endpoint == "https://example.openai.azure.com"
    assert provider.deployment == "deployment-a"
    assert provider.api_version == "2026-01-01-preview"
    assert provider.region is None


def test_cost_confirmation_is_required_without_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    calls = _patch_evaluator(monkeypatch, _loaded_result(tmp_path))

    result = runner.invoke(
        app,
        [*_base_args(root, out), "--dataset-path", str(dataset)],
        input="n\n",
    )

    assert result.exit_code == 1, result.output
    assert calls == []
    flat = " ".join(result.output.split())
    assert "can incur model and environment costs" in flat
    assert "Proceed with paid evaluation?" in flat
    assert not out.exists()


def test_partial_result_is_written_but_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    _patch_evaluator(monkeypatch, _loaded_result(tmp_path))

    result = runner.invoke(
        app,
        [*_base_args(root, out), "--dataset-path", str(dataset), "--yes"],
    )

    assert result.exit_code == 1
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["trials"]) == 4
    flat = " ".join(result.output.split())
    assert "not fully scored" in flat
    assert "scored=1" in flat
    assert "task_timeout=1" in flat
    assert "candidate_failed=1" in flat
    assert "candidate_timeout=0" in flat
    assert "candidate_resource_limit=0" in flat
    assert "candidate_runtime_error=1" in flat
    assert "candidate_unclassified=0" in flat
    assert "infra=1" in flat
    assert "incomplete=1" in flat
    assert "wrote canonical result" in flat


def test_all_infrastructure_result_is_written_but_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    _patch_evaluator(monkeypatch, _all_infrastructure_result(tmp_path))

    result = runner.invoke(
        app,
        [*_base_args(root, out), "--dataset-path", str(dataset), "--yes"],
    )

    assert result.exit_code == 1
    assert out.is_file()
    flat = " ".join(result.output.split())
    assert "not fully scored" in flat
    assert "scored=0" in flat
    assert "infra=4" in flat
    assert "incomplete=0" in flat


def test_cancelled_result_is_terminal_but_requires_a_new_named_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    _patch_evaluator(monkeypatch, _cancelled_result(tmp_path))

    result = runner.invoke(
        app,
        [*_base_args(root, out), "--dataset-path", str(dataset), "--yes"],
    )

    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "cancelled=1" in flat
    assert "incomplete=0" in flat
    assert "use a new --job-name" in flat


def test_empty_result_is_written_but_never_reported_as_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    _patch_evaluator(monkeypatch, _empty_result(tmp_path))

    result = runner.invoke(
        app,
        [*_base_args(root, out), "--dataset-path", str(dataset), "--yes"],
    )

    assert result.exit_code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["expected_cells"] == []
    flat = " ".join(result.output.split())
    assert "not fully scored" in flat
    assert "scored=0" in flat
    assert "resolved task digests=0" in flat


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--dataset-path", "{dataset}", "--dataset", "org/benchmark", "--dataset-ref", "v1"],
        ["--dataset", "org/benchmark"],
        ["--dataset", "org/benchmark", "--dataset-ref", "v1", "--provider", "openai"],
    ],
)
def test_invalid_dataset_or_provider_selection_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    calls = _patch_evaluator(monkeypatch, _loaded_result(tmp_path))
    expanded = [str(dataset) if value == "{dataset}" else value for value in extra]
    args = _base_args(root, out)
    if "--provider" in expanded:
        provider_index = args.index("--provider")
        del args[provider_index : provider_index + 2]

    result = runner.invoke(app, [*args, *expanded, "--yes"])

    assert result.exit_code == 2
    assert calls == []


def test_azure_requires_deployment_and_api_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    calls = _patch_evaluator(monkeypatch, _loaded_result(tmp_path))
    args = _base_args(root, out)
    provider_index = args.index("--provider")
    args[provider_index + 1] = "azure"
    region_index = args.index("--bedrock-region")
    del args[region_index : region_index + 2]

    result = runner.invoke(
        app,
        [
            *args,
            "--dataset-path",
            str(dataset),
            "--azure-endpoint",
            "https://example.openai.azure.com",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "--azure-deployment" in result.output
    assert calls == []


def test_eval_requires_explicit_provider_route_for_canonical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    calls = _patch_evaluator(monkeypatch, _loaded_result(tmp_path))

    bedrock_args = _base_args(root, out)
    region_index = bedrock_args.index("--bedrock-region")
    del bedrock_args[region_index : region_index + 2]
    bedrock_result = runner.invoke(
        app,
        [*bedrock_args, "--dataset-path", str(dataset), "--yes"],
    )

    azure_args = _base_args(root, out)
    provider_index = azure_args.index("--provider")
    azure_args[provider_index + 1] = "azure"
    region_index = azure_args.index("--bedrock-region")
    del azure_args[region_index : region_index + 2]
    azure_result = runner.invoke(
        app,
        [
            *azure_args,
            "--dataset-path",
            str(dataset),
            "--azure-deployment",
            "deployment-a",
            "--azure-api-version",
            "2026-01-01-preview",
            "--yes",
        ],
    )

    assert bedrock_result.exit_code == 2
    assert "--bedrock-region" in bedrock_result.output
    assert azure_result.exit_code == 2
    assert "--azure-endpoint" in azure_result.output
    assert calls == []


def test_help_exposes_no_credential_flags() -> None:
    result: Result = runner.invoke(app, ["harness", "eval", "--help"])

    assert result.exit_code == 0, result.output
    lowered = result.output.lower()
    assert "api-key" not in lowered
    assert "api key" not in lowered
    assert "--azure-endpoint" in result.output
    assert "--bedrock-region" in result.output
    assert "--task" in result.output
    assert "--exclude-task" in result.output
    flat = " ".join(result.output.replace("│", " ").split())
    assert "Pi runner backend spec" in flat
    assert "Local Docker is the default" in flat


@pytest.mark.parametrize(
    "task_args",
    [
        ["--task", "task-a", "--exclude-task", "task-b"],
        ["--task", "   "],
        ["--exclude-task", ""],
    ],
)
def test_task_filters_are_mutually_exclusive_and_nonblank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_args: list[str],
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))

    result = runner.invoke(
        app,
        [
            *_base_args(root, out),
            "--dataset-path",
            str(dataset),
            *task_args,
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert calls == []


def test_output_cannot_overwrite_the_active_harbor_job_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    jobs_dir = tmp_path / "jobs"
    out = jobs_dir / "evaluation" / "result.json"
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))

    result = runner.invoke(
        app,
        [
            *_base_args(root, out),
            "--dataset-path",
            str(dataset),
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            "evaluation",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "output path cannot be inside the active Harbor job" in " ".join(result.output.split())
    assert calls == []


def test_output_cannot_overwrite_a_different_harbor_job_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    jobs_dir = tmp_path / "jobs"
    out = jobs_dir / "different-evaluation" / "result.json"
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))

    result = runner.invoke(
        app,
        [
            *_base_args(root, out),
            "--dataset-path",
            str(dataset),
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            "evaluation",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "another Harbor job" in " ".join(result.output.split())
    assert calls == []


def test_concurrent_writer_for_same_output_is_rejected_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result.json"
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))

    with harness_eval_module._exclusive_output_lease(out.resolve()):
        result = runner.invoke(
            app,
            [*_base_args(root, out), "--dataset-path", str(dataset), "--yes"],
        )

    assert result.exit_code == 1
    assert result.exception is not None
    assert "already publishing" in str(result.exception)
    assert calls == []
    assert not out.exists()


def test_output_lease_rejects_platform_without_posix_locking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_file_lease, "fcntl", None)
    out = tmp_path / "reports" / "result.json"

    with pytest.raises(RuntimeError, match="output leases require POSIX file locking"):
        with harness_eval_module._exclusive_output_lease(out):
            raise AssertionError("unsupported platform must not acquire the output lease")

    assert not out.parent.exists()


def test_output_cannot_replace_the_active_harbor_job_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    jobs_dir = tmp_path / "jobs"
    out = jobs_dir / ".evaluation.wmh-eval.lock"
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))

    result = runner.invoke(
        app,
        [
            *_base_args(root, out),
            "--dataset-path",
            str(dataset),
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            "evaluation",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    flat = " ".join(result.output.split())
    assert "active Harbor job" in flat
    assert "lease" in flat
    assert calls == []


def test_output_symlink_is_rejected_without_overwriting_its_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    target = tmp_path / "protected.json"
    target.write_text("protected", encoding="utf-8")
    out = tmp_path / "result.json"
    out.symlink_to(target)
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))

    result = runner.invoke(
        app,
        [*_base_args(root, out), "--dataset-path", str(dataset), "--yes"],
    )

    assert result.exit_code == 2
    assert "cannot be a symlink" in result.output
    assert calls == []
    assert target.read_text(encoding="utf-8") == "protected"


def test_output_directory_is_rejected_before_paid_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".wmh"
    _save_harness(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "result-directory"
    out.mkdir()
    calls = _patch_evaluator(monkeypatch, _all_scored_result(tmp_path))

    result = runner.invoke(
        app,
        [*_base_args(root, out), "--dataset-path", str(dataset), "--yes"],
    )

    assert result.exit_code == 2
    assert "regular file location" in " ".join(result.output.split())
    assert calls == []
