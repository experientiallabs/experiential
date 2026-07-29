"""Offline regression tests for coding-router scaffold-failure handling."""

from __future__ import annotations

import json
from pathlib import Path

import coding_model_router_matrix as matrix_runner
import coding_model_router_smoke as smoke_runner
import pytest

from wmo.harness.scoring import ScoreCell
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import ModelPool, PoolEntry


def _entry() -> PoolEntry:
    return PoolEntry(
        name="test-arm",
        kind=ProviderKind.OPENAI,
        model="test-model",
        input_per_mtok=1.0,
        output_per_mtok=2.0,
    )


def _artifact(root: Path) -> Path:
    artifact = root / "artifact"
    trace = artifact / "agent" / "wmo-run.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "instruction": "repair the repository",
                "stop_reason": "error",
                "steps": [
                    {
                        "action": {
                            "kind": "message",
                            "content": "(pi runtime)",
                            "name": None,
                            "arguments": {},
                        },
                        "observation": {
                            "content": (
                                "remote materialize failed (rc=255): Host key verification failed."
                            ),
                            "is_error": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _cell(artifact: Path) -> ScoreCell:
    return ScoreCell(
        task_id="task",
        attempt=1,
        reward=0.0,
        passed=False,
        artifact_dir=str(artifact),
        infra_failed=False,
    )


def _unmetered_artifact(root: Path, *, stop_reason: str = "submitted") -> Path:
    artifact = root / "artifact"
    trace = artifact / "agent" / "wmo-run.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "instruction": "repair the repository",
                "stop_reason": stop_reason,
                "steps": [
                    {
                        "action": {
                            "kind": "tool_call",
                            "name": "bash",
                            "arguments": {"command": "true"},
                        },
                        "observation": {"content": "", "is_error": False},
                    }
                ],
                "worker_usage": None,
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_smoke_scaffold_stop_is_ungradeable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(smoke_runner, "_wall_seconds", lambda path: 0.0)

    outcome = smoke_runner._outcome(
        _cell(artifact),
        entry=_entry(),
        logical_attempt=1,
        artifact_dir=artifact,
    )

    assert outcome.reward is None
    assert outcome.completion_status == "scaffold_failure"
    assert outcome.failure_class == "scaffold"
    assert outcome.error == ("remote materialize failed (rc=255): Host key verification failed.")


def test_full_matrix_scaffold_stop_is_ungradeable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(matrix_runner, "_wall_seconds", lambda path: 0.0)

    outcome = matrix_runner._outcome(
        _cell(artifact),
        benchmark="terminal-bench-2",
        entry=_entry(),
        attempt=1,
        artifact_dir=artifact,
    )

    assert outcome.reward is None
    assert outcome.completion_status == "scaffold_failure"
    assert outcome.failure_class == "scaffold"


@pytest.mark.parametrize("runner", [smoke_runner, matrix_runner])
def test_submitted_cell_without_worker_usage_is_ungradeable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: object,
) -> None:
    artifact = _unmetered_artifact(tmp_path)
    monkeypatch.setattr(runner, "_wall_seconds", lambda path: 0.0)
    if runner is smoke_runner:
        outcome = smoke_runner._outcome(
            _cell(artifact),
            entry=_entry(),
            logical_attempt=2,
            artifact_dir=artifact,
        )
    else:
        outcome = matrix_runner._outcome(
            _cell(artifact),
            benchmark="terminal-bench-2",
            entry=_entry(),
            attempt=2,
            artifact_dir=artifact,
        )

    assert outcome.reward is None
    assert outcome.completion_status == "metering_failure"
    assert outcome.failure_class == "metering"
    assert outcome.error == "worker usage is missing a completed provider call"


def test_existing_scaffold_zero_is_normalized_before_resume(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    measured = OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair the repository",
                model=entry.name,
                benchmark="terminal-bench-2",
                reward=0.0,
                completion_status="scored_failure",
                failure_class="scaffold",
                artifact_dir=str(artifact),
            )
        ],
    )

    normalized = smoke_runner._normalize_existing_ungradeable_attempts(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    outcome = normalized.outcomes[0]
    assert outcome.reward is None
    assert outcome.completion_status == "scaffold_failure"
    assert outcome.error == ("remote materialize failed (rc=255): Host key verification failed.")
    assert Path(outcome.artifact_dir).is_dir()
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["completion_status"] == (
        "scaffold_failure"
    )


def test_existing_unmetered_score_is_normalized_and_cost_is_unknown(tmp_path: Path) -> None:
    artifact = _unmetered_artifact(tmp_path)
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    measured = OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair the repository",
                model=entry.name,
                benchmark="terminal-bench-2",
                attempt_number=2,
                reward=1.0,
                success=True,
                completion_status="scored_pass",
                artifact_dir=str(artifact),
            )
        ],
    )

    normalized = smoke_runner._normalize_existing_ungradeable_attempts(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    outcome = normalized.outcomes[0]
    assert outcome.reward is None
    assert outcome.failure_class == "metering"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["model_cost_usd"] is None
    assert ledger["model_cost_accounting_status"] == "missing_provider_usage"


def test_existing_unmetered_scaffold_is_reclassified_as_unknown_cost(tmp_path: Path) -> None:
    artifact = _unmetered_artifact(tmp_path, stop_reason="max_turns")
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    measured = OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair the repository",
                model=entry.name,
                benchmark="terminal-bench-2",
                attempt_number=2,
                reward=None,
                completion_status="scaffold_failure",
                failure_class="scaffold",
                artifact_dir=str(artifact),
            )
        ],
    )

    normalized = smoke_runner._normalize_existing_ungradeable_attempts(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    assert normalized.outcomes[0].failure_class == "metering"
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["model_cost_usd"] is None


def test_archive_keeps_distinct_artifacts_for_the_same_logical_attempt(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    first = _artifact(tmp_path / "first")
    second = _unmetered_artifact(tmp_path / "second")

    first_archive = smoke_runner._archive_infra(
        root,
        first,
        task_id="task",
        arm="arm",
        attempt=2,
    )
    second_archive = smoke_runner._archive_infra(
        root,
        second,
        task_id="task",
        arm="arm",
        attempt=2,
    )

    assert first_archive != second_archive
    assert smoke_runner._artifact_digest(str(first_archive)) != smoke_runner._artifact_digest(
        str(second_archive)
    )


def test_unknown_paid_cost_stops_before_another_smoke_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    pool = ModelPool(models=[entry])
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:prior",
                task="prior",
                model=entry.name,
                benchmark="terminal-bench-2",
                reward=None,
                completion_status="metering_failure",
                failure_class="metering",
            )
        ],
    ).save(matrix_path)
    monkeypatch.setattr(smoke_runner, "ARMS", (entry.name,))

    with pytest.raises(RuntimeError, match="exact spend"):
        smoke_runner._run_cell(
            root,
            template_path=tmp_path / "unused.yaml",
            task_id="next",
            entry=entry,
            matrix_path=matrix_path,
            ledger_path=ledger_path,
            pool=pool,
        )


def test_identical_harbor_resume_does_not_consume_an_attempt(tmp_path: Path) -> None:
    first_artifact = _artifact(tmp_path / "first")
    duplicate_artifact = _artifact(tmp_path / "duplicate")
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    outcomes = [
        ScenarioOutcome(
            scenario_id="terminal-bench-2:task",
            task="repair the repository",
            model=entry.name,
            benchmark="terminal-bench-2",
            attempt_number=attempt,
            episode=attempt - 1,
            reward=None,
            completion_status="scaffold_failure",
            failure_class="scaffold",
            artifact_dir=str(artifact),
        )
        for attempt, artifact in ((1, first_artifact), (2, duplicate_artifact))
    ]
    measured = OutcomeMatrix(pool=[entry], outcomes=outcomes)
    matrix_path.parent.mkdir(parents=True)
    measured.save(matrix_path)
    ledger_path.write_text(
        "".join(
            json.dumps(
                {
                    "event_id": (
                        f"smoke:terminal-bench-2:terminal-bench-2:task:{entry.name}:{attempt}"
                    ),
                    "status": "completed",
                    "model_cost_usd": 0.0,
                }
            )
            + "\n"
            for attempt in (1, 2)
        ),
        encoding="utf-8",
    )

    deduplicated = smoke_runner._drop_duplicate_retry_noops(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    assert [row.attempt_number for row in deduplicated.outcomes] == [1]
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 1
    audit = json.loads((root / "retry-noops.json").read_text(encoding="utf-8"))
    assert audit["rows"][0]["attempt_number"] == 2
