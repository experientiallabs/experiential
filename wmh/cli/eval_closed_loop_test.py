"""CLI tests for `wmh eval --mode closed-loop`: sim-vs-e2b env plumbing, driven via CliRunner.

Scoring is faked at the `evaluate_with_env` / `ClosedLoopEval` seam — these tests pin the
WIRING: the `@e2b` label suffix, backend-appropriate concurrency defaults, when the world model
is optional, and flag validation. No sandbox (or model) is ever touched.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from wmh.cli import app
from wmh.evals.closed_loop import ClosedLoopReport, EnvFactory, TaskOutcome
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.runtime import AgentRuntime, Runtime
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind

eval_cl_module = importlib.import_module("wmh.cli.eval_closed_loop")

runner = CliRunner()

_Progress = Callable[[str, int, GoldVerdict], None] | None


class _Provider:
    """A do-nothing provider: scoring is faked, so no LLM role is ever exercised."""

    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(self, system: str, messages: list[Message], **kw: object) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> object:
        raise NotImplementedError


def _tasks_file(tmp_path: Path) -> str:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "do it", "gold": ["done"]}\n', encoding="utf-8"
    )
    return str(path)


def _report(label: str, k: int) -> ClosedLoopReport:
    outcome = TaskOutcome(task_id="t1", success_rate=1.0, mean_fraction=1.0, passes=k)
    return ClosedLoopReport(
        label=label, success_rate=1.0, mean_fraction=1.0, k=k, per_task={"t1": outcome}
    )


def _invoke(tmp_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "eval",
            _tasks_file(tmp_path),
            "--mode",
            "closed-loop",
            "--root",
            str(tmp_path / ".wmh"),
            *extra,
        ],
    )


def _patch_e2b_seams(monkeypatch: pytest.MonkeyPatch, seen: dict[str, object]) -> None:
    """Fake `evaluate_with_env` (recording its wiring) and the no-world-model provider."""

    def fake_evaluate(
        tasks: list[TaskSpec],
        make_env: EnvFactory,
        runtime: Runtime,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        on_progress: _Progress = None,
    ) -> ClosedLoopReport:
        seen.update({"label": label, "k": k, "concurrency": concurrency, "runtime": runtime})
        return _report(label, k)

    monkeypatch.setattr(eval_cl_module, "evaluate_with_env", fake_evaluate)
    monkeypatch.setattr(
        eval_cl_module, "default_worker_provider", lambda root: (_Provider(), "fallback-model")
    )


def test_eval_env_e2b_runs_without_a_world_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    _patch_e2b_seams(monkeypatch, seen)

    result = _invoke(tmp_path, "--env", "e2b")  # note: no world model exists under this root

    assert result.exit_code == 0, result.output
    assert seen["label"] == "baseline@e2b"  # the label agreement reports read
    assert seen["concurrency"] == 0  # e2b default: every (task, attempt) cell at once
    assert isinstance(seen["runtime"], AgentRuntime)  # the baseline loop stays host-side
    flat = result.output.replace("\n", " ")  # rich wraps lines
    assert "real E2B sandboxes" in flat
    assert "OVERALL" in result.output


def test_eval_env_e2b_concurrency_flag_caps_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    _patch_e2b_seams(monkeypatch, seen)
    result = _invoke(tmp_path, "--env", "e2b", "--eval-concurrency", "3")
    assert result.exit_code == 0, result.output
    assert seen["concurrency"] == 3


def test_eval_env_sim_keeps_world_model_label_and_sequential_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeStore:
        def __init__(self, root: str) -> None:
            self.root = root

        def resolve(self, name: str | None) -> Path:
            return Path("/models/wm-alpha")

    wm = object()
    seen: dict[str, object] = {}

    class _FakeEval:
        def __init__(
            self,
            tasks: list[TaskSpec],
            world_model: object,
            provider: object,
            judge: GoldJudge,
            *,
            label: str,
            k: int,
            concurrency: int,
            runtime: Runtime | None,
            on_progress: _Progress,
        ) -> None:
            seen.update({"world_model": world_model, "label": label, "concurrency": concurrency})
            self._label = label
            self._k = k

        def run(self) -> ClosedLoopReport:
            return _report(self._label, self._k)

    monkeypatch.setattr(eval_cl_module, "WorldModelStore", _FakeStore)
    monkeypatch.setattr(eval_cl_module, "load_world_model", lambda d: (wm, _Provider()))
    monkeypatch.setattr(eval_cl_module, "ClosedLoopEval", _FakeEval)

    result = _invoke(tmp_path)  # --env defaults to sim

    assert result.exit_code == 0, result.output
    assert seen["world_model"] is wm  # sim still runs against the world model
    assert seen["label"] == "baseline@wm-alpha"  # label keeps naming the model, not e2b
    assert seen["concurrency"] == 1  # sim default: the sequential loop, unchanged


def test_eval_env_sim_still_requires_a_world_model(tmp_path: Path) -> None:
    # No model built under the tmp root: sim (the default) must fail as a usage error.
    result = _invoke(tmp_path)
    assert result.exit_code == 2
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_eval_rejects_unknown_env(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "--env", "banana")
    assert result.exit_code == 2  # usage error, not a traceback
    assert "choose sim or e2b" in result.output
