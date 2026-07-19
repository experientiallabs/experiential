"""CLI tests for `wmh harness create`: the harness-backend wiring, driven via CliRunner.

The search itself is faked (`create_harness` is monkeypatched to a recorder) — these tests pin
the WIRING the flags control: which harness backend reaches the search, that the world model is
ALWAYS loaded (it is the environment on every backend), and what the cost-confirmation line
advertises. Flag validation, task loading, and the harness store are real.
"""

from __future__ import annotations

import importlib
import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from wmh.cli import app
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.evals.tasks import TaskSpec
from wmh.harness.create import CreateResult, DeltaArchive
from wmh.harness.doc import HarnessDoc
from wmh.harness.proposer import ProviderDeltaProposer
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind

# The Typer object `harness_app` shadows the submodule name on plain attribute access; go
# through importlib to monkeypatch module globals (same pattern as app_test.py).
harness_app_module = importlib.import_module("wmh.cli.harness_app")
model_roles_module = importlib.import_module("wmh.cli.model_roles")

runner = CliRunner()


class _Provider:
    """A do-nothing provider: the search is faked, so no role is ever exercised."""

    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(self, system: str, messages: list[Message], **kw: object) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> object:
        raise NotImplementedError


class _CreateRecorder:
    """Stands in for `create_harness`: records each call, returns a minimal valid result."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        name: str,
        seed_doc: HarnessDoc,
        tasks: list[TaskSpec],
        world_model: object,
        agent_provider: object,
        proposer: object,
        judge: object,
        **kwargs: object,
    ) -> CreateResult:
        self.calls.append(
            {
                "name": name,
                "world_model": world_model,
                "provider": agent_provider,
                "proposer": proposer,
                **kwargs,
            }
        )
        best = seed_doc.model_copy(update={"name": name})
        return CreateResult(best=best, best_score=1.0, archive=DeltaArchive(seed=seed_doc))


def _tasks_file(tmp_path: Path) -> str:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "do it", "gold": ["done"]}\n', encoding="utf-8"
    )
    return str(path)


def _invoke(tmp_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "harness",
            "create",
            "made",
            "--tasks",
            _tasks_file(tmp_path),
            "--iterations",
            "2",
            "--root",
            str(tmp_path / ".wmh"),
            *extra,
        ],
    )


def _patch_load(
    monkeypatch: pytest.MonkeyPatch, wm: object, provider: _Provider
) -> list[str | None]:
    loads: list[str | None] = []

    def fake_load(model: str | None, root: str) -> tuple[object, _Provider, str]:
        loads.append(model)
        return wm, provider, "wm-alpha"

    monkeypatch.setattr(harness_app_module, "_load_world_model", fake_load)
    return loads


def test_create_e2b_wires_backend_flags_and_still_loads_the_world_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _CreateRecorder()
    wm = object()
    anchored = _Provider()
    monkeypatch.setattr(harness_app_module, "create_harness", recorder)
    loads = _patch_load(monkeypatch, wm, anchored)

    result = _invoke(
        tmp_path,
        "--harness-backend",
        "e2b",
        "--eval-concurrency",
        "4",
        "--e2b-template",
        "tmpl-x",
    )

    assert result.exit_code == 0, result.output
    assert loads == [None]  # the world model is the environment: ALWAYS loaded, even for e2b
    [call] = recorder.calls
    assert call["world_model"] is wm
    assert call["provider"] is anchored
    assert call["harness_backend"] == "e2b"
    assert call["eval_concurrency"] == 4
    assert call["e2b_template"] == "tmpl-x"
    flat = " ".join(result.output.split())  # rich wraps (and pads) lines
    # The cost line keeps the rollout estimate ((iterations+1) * k * tasks = 3 * 3 * 1 = 9)
    # and says where the harness process runs — while the env stays the world model.
    assert "9 rollouts" in flat
    assert "pooled E2B sandboxes" in flat
    assert "world model" in flat and "wm-alpha" in flat
    expected = shlex.join(
        [
            "wmh",
            "eval",
            _tasks_file(tmp_path),
            "--mode",
            "closed-loop",
            "--name",
            "wm-alpha",
            "--root",
            str(tmp_path / ".wmh"),
            "--k",
            "3",
            "--harness",
            "made@1",
            "--harness-backend",
            "e2b",
            "--eval-concurrency",
            "4",
            "--e2b-template",
            "tmpl-x",
        ]
    )
    assert expected in flat
    assert "wmh eval closed-loop" not in flat


def test_create_default_local_loads_the_world_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WMH_E2B_TEMPLATE", raising=False)  # --e2b-template defaults from it
    project_root = tmp_path / "project with spaces"
    project_root.mkdir()
    recorder = _CreateRecorder()
    wm = object()
    monkeypatch.setattr(harness_app_module, "create_harness", recorder)
    loads = _patch_load(monkeypatch, wm, _Provider())

    result = _invoke(project_root)

    assert result.exit_code == 0, result.output
    assert loads == [None]  # default: resolve the only built model
    [call] = recorder.calls
    assert call["world_model"] is wm
    assert call["harness_backend"] == "local"
    assert call["eval_concurrency"] is None  # backend default decided downstream (local -> 1)
    assert call["e2b_template"] is None
    flat = " ".join(result.output.split())
    assert "world model" in flat and "wm-alpha" in flat
    assert "sandbox" not in flat  # no sandbox note on the local path
    expected = shlex.join(
        [
            "wmh",
            "eval",
            _tasks_file(project_root),
            "--mode",
            "closed-loop",
            "--name",
            "wm-alpha",
            "--root",
            str(project_root / ".wmh"),
            "--k",
            "3",
            "--harness",
            "made@1",
        ]
    )
    assert expected in flat
    assert "--harness-backend" not in flat  # and the run-it hint stays plain


def test_create_meta_role_from_settings_drives_the_proposer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[models.meta]` in settings.toml selects the delta proposer's provider."""
    recorder = _CreateRecorder()
    anchored = _Provider()
    monkeypatch.setattr(harness_app_module, "create_harness", recorder)
    _patch_load(monkeypatch, object(), anchored)
    root = tmp_path / ".wmh"
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                meta=ModelRole(
                    provider="azure",
                    model="gpt-5.5",
                    endpoint="https://x.example",
                    deployment="gpt-5-5",
                )
            )
        ),
        root,
    )
    meta_sentinel = _Provider()

    def fake_get_provider(_config: ProviderConfig) -> _Provider:
        return meta_sentinel

    monkeypatch.setattr(model_roles_module, "get_provider", fake_get_provider)

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    assert call["provider"] is anchored  # agent + judge stay on the world model's provider
    assert isinstance(call["proposer"], ProviderDeltaProposer)
    assert call["proposer"].provider is meta_sentinel
    flat = " ".join(result.output.split())
    assert "proposer: gpt-5.5 from settings models.meta" in flat  # the banner names it


def test_create_agent_role_from_settings_drives_the_agent_under_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[models.agent]` selects the agent-under-test provider, distinct from the world model."""
    recorder = _CreateRecorder()
    anchored = _Provider()  # the world model's serve provider
    monkeypatch.setattr(harness_app_module, "create_harness", recorder)
    _patch_load(monkeypatch, object(), anchored)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                agent=ModelRole(
                    provider="openai",
                    model="Qwen/Qwen3.5-9B",
                    endpoint="http://127.0.0.1:8002/v1",
                )
            )
        ),
        tmp_path / ".wmh",
    )
    agent_sentinel = _Provider()

    def fake_get_provider(_config: ProviderConfig) -> _Provider:
        return agent_sentinel

    monkeypatch.setattr(model_roles_module, "get_provider", fake_get_provider)

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    # The agent-under-test is the configured provider; the proposer (meta unset) and judge
    # keep the world model's provider.
    assert call["provider"] is agent_sentinel
    assert isinstance(call["proposer"], ProviderDeltaProposer)
    assert call["proposer"].provider is anchored
    flat = " ".join(result.output.split())
    assert "agent-under-test: Qwen/Qwen3.5-9B from settings models.agent" in flat


def test_create_agent_defaults_to_the_world_model_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `[models.agent]` the agent-under-test stays on the world model's provider."""
    recorder = _CreateRecorder()
    anchored = _Provider()
    monkeypatch.setattr(harness_app_module, "create_harness", recorder)
    _patch_load(monkeypatch, object(), anchored)

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    assert call["provider"] is anchored
    assert "models.agent" not in result.output


def test_create_meta_defaults_to_the_world_model_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `[models.meta]` the proposer stays on the world model's provider."""
    recorder = _CreateRecorder()
    anchored = _Provider()
    monkeypatch.setattr(harness_app_module, "create_harness", recorder)
    _patch_load(monkeypatch, object(), anchored)

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    assert isinstance(call["proposer"], ProviderDeltaProposer)
    assert call["proposer"].provider is anchored
    assert "models.meta" not in result.output


def test_create_rejects_unknown_harness_backend(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "--harness-backend", "banana")
    assert result.exit_code == 2  # usage error, not a traceback
    assert "choose local or e2b" in result.output
