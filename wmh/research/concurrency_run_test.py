"""Tests for the live world/real runner builders (fakes only; no network/subprocess/Docker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmh.research.concurrency_run import build_real_runner, build_world_runner
from wmh.retrieval.leakfree import DemoRetriever


class _FakeProvider:
    """A provider returning a fixed observation, so the world runner does real work with no net."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="fake")

    def complete(
        self, system: str, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 0
    ) -> Completion:
        return Completion(text="ok", usage=TokenUsage(input_tokens=1, output_tokens=1))

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:  # pragma: no cover - unused
        return VerifyResult(ok=True, kind=self.config.kind, model=self.config.model)


def _trace(tid: str, n: int) -> Trace:
    steps = [
        Step(
            action=Action(
                kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": f"echo {i}"}
            ),
            observation=Observation(content="ok", is_error=False),
            state_before=EnvState(),
        )
        for i in range(n)
    ]
    return Trace(trace_id=tid, steps=steps)


def _provider_factory() -> Provider:
    return _FakeProvider()


def test_world_runner_times_and_counts_batch() -> None:
    selected = [(0, _trace("a", 2)), (1, _trace("b", 3))]
    demos = DemoRetriever(None, [], top_k=5)  # zero-shot: no retriever
    runner = build_world_runner(_provider_factory, "prompt", demos, selected)
    batch = runner(2)
    assert batch.total == 2
    assert batch.ok == 2
    assert batch.wall_seconds >= 0.0
    assert batch.fidelity == 1.0  # fake predicts is_error=False, matching every recorded step


def test_world_runner_runs_fixed_batch_at_every_level() -> None:
    # Every level runs the whole fixed batch; only the worker-pool width changes.
    selected = [(i, _trace(f"t{i}", 1)) for i in range(4)]
    demos = DemoRetriever(None, [], top_k=5)
    runner = build_world_runner(_provider_factory, "prompt", demos, selected)
    assert runner(1).total == 4  # W=1 -> all 4 (sequential)
    assert runner(2).total == 4  # W=2 -> all 4
    assert runner(4).total == 4  # W=4 -> all 4


def test_world_runner_meters_tokens_and_cost() -> None:
    # The metered path must populate tokens/cost (the fake bills 1+1 tokens per predict call).
    selected = [(0, _trace("a", 2)), (1, _trace("b", 3))]  # 5 steps -> 5 predict calls
    demos = DemoRetriever(None, [], top_k=5)
    batch = build_world_runner(_provider_factory, "prompt", demos, selected)(2)
    assert batch.tokens == 10  # 5 steps * (1 in + 1 out)
    assert batch.cost_usd >= 0.0  # priced by the pricing table (fake model -> 0.0 is fine)


def test_real_runner_missing_runsh_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run.sh"):
        build_real_runner(tmp_path, [(0, _trace("a", 1))], train_split=0.7)


def test_real_runner_non_executable_runsh_raises(tmp_path: Path) -> None:
    runsh = tmp_path / "run.sh"
    runsh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    runsh.chmod(0o644)  # present but not executable
    with pytest.raises(PermissionError, match="not executable"):
        build_real_runner(tmp_path, [(0, _trace("a", 1))], train_split=0.7)


def test_real_runner_pins_trace_id(tmp_path: Path) -> None:
    # The runner must invoke run.sh with --trace-id <the trace's id>, not a positional index.
    runsh = tmp_path / "run.sh"
    runsh.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > args.txt\nexit 0\n", encoding="utf-8"
    )
    runsh.chmod(0o755)
    build_real_runner(tmp_path, [(3, _trace("zzz-trace-id", 1))], train_split=0.7)(1)
    args = (tmp_path / "args.txt").read_text(encoding="utf-8")
    assert "--trace-id" in args
    assert "zzz-trace-id" in args
    assert "3" not in args.split()  # the pool index is NOT what gets passed


def test_real_runner_shells_runsh(tmp_path: Path) -> None:
    # A stub run.sh that exits 0 quickly; the runner should time it and count it ok.
    runsh = tmp_path / "run.sh"
    runsh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    runsh.chmod(0o755)
    runner = build_real_runner(
        tmp_path, [(0, _trace("a", 1)), (1, _trace("b", 1))], train_split=0.7
    )
    batch = runner(2)
    assert batch.total == 2
    assert batch.ok == 2
    assert batch.wall_seconds >= 0.0


def test_real_runner_nonzero_exit_marks_not_ok(tmp_path: Path) -> None:
    runsh = tmp_path / "run.sh"
    runsh.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    runsh.chmod(0o755)
    runner = build_real_runner(tmp_path, [(0, _trace("a", 1))], train_split=0.7)
    batch = runner(1)
    assert batch.total == 1
    assert batch.ok == 0
