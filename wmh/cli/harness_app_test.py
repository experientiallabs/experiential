"""CLI tests for `wmh optimize`: optimizer-backend wiring, driven via CliRunner.

The search itself is faked (`create_harness` is monkeypatched to a recorder) — these tests pin
the WIRING the flags control: which harness backend reaches the search, that the world model is
ALWAYS loaded (it is the environment on every backend), and what the cost-confirmation line
advertises. Flag validation, task loading, and the harness store are real.
"""

from __future__ import annotations

import importlib
import json
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from wmh.cli import app
from wmh.cli.harness_app import _HarborRunConfig
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.evals.tasks import TaskSpec
from wmh.harness.create import CreateResult, DeltaArchive
from wmh.harness.doc import HarnessDoc
from wmh.harness.population import CandidateProposal, EvaluatedCandidate, candidate_slot_id
from wmh.harness.proposer import ProviderDeltaProposer
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.harness.store import HarnessStore
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
            "optimize",
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
        "--backend",
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


def test_optimize_accepts_world_model_as_second_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _CreateRecorder()
    monkeypatch.setattr(harness_app_module, "create_harness", recorder)
    loads = _patch_load(monkeypatch, object(), _Provider())

    result = runner.invoke(
        app,
        [
            "optimize",
            "made",
            "wm-user",
            "--tasks",
            _tasks_file(tmp_path),
            "--iterations",
            "1",
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert loads == ["wm-user"]


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


def test_optimize_rejects_unknown_backend(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "--backend", "banana")
    assert result.exit_code == 2  # usage error, not a traceback
    assert "choose local or e2b" in result.output


# -- the harbor environment: dispatch, durable state, resume, and publication ------------------


_HARBOR_JOB_YAML = """\
job_name: template
jobs_dir: jobs
environment:
  type: docker
agents:
  - {}
datasets:
  - path: tasks
"""


class _FakeHarborScorer:
    """Passes every cell; the report matrix comes from the recorded run config."""

    def __init__(self, task_ids: tuple[str, ...], attempts: int) -> None:
        self._request = ScoreRequest(task_ids=task_ids, attempts=attempts)

    @property
    def request(self) -> ScoreRequest:
        return self._request

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport:
        del should_cancel
        cells = tuple(
            ScoreCell(task_id=task_id, attempt=attempt, reward=1.0, passed=True)
            for task_id in self._request.task_ids
            for attempt in range(1, self._request.attempts + 1)
        )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=self._request,
            reward_mode="positive-binary",
            cells=cells,
        )


class _FakeHarborProposer:
    """Returns the seed tree with a rewritten SYSTEM.md, one candidate per slot."""

    def propose(
        self,
        population: Sequence[EvaluatedCandidate],
        *,
        slot: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal:
        del should_cancel
        files = {
            item.path: item.content
            for item in population[0].source.files
            if item.path != "SYSTEM.md"
        }
        files["SYSTEM.md"] = f"improved by slot {slot}"
        tree = HarnessSourceTree(
            files=tuple(
                HarnessSourceFile(path=path, content=content) for path, content in files.items()
            )
        )
        candidate_id = candidate_slot_id(slot)
        return CandidateProposal(
            candidate_id=candidate_id, source=tree, candidate=tree.to_doc(candidate_id)
        )


def _harbor_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Write harbor inputs + role settings and fake the scorer/proposer builder seams."""
    (tmp_path / "job.yaml").write_text(_HARBOR_JOB_YAML, encoding="utf-8")
    (tmp_path / "tasks.json").write_text('["t1"]', encoding="utf-8")
    root = tmp_path / ".wmh"
    role = ModelRole(provider="azure", model="gpt-5.5", endpoint="https://x.example")
    save_settings(ProjectSettings(models=ModelsSettings(meta=role, agent=role)), root)
    captured: dict[str, object] = {}

    def fake_scorer(
        config: _HarborRunConfig, *, run_dir: Path, provider_config: ProviderConfig
    ) -> tuple[_FakeHarborScorer, dict[str, str]]:
        del run_dir
        captured["config"] = config
        captured["provider_config"] = provider_config
        salt = str(captured.get("pin_salt", ""))
        pins = {task_id: f"path:/tasks/{task_id}{salt}" for task_id in config.task_ids}
        return _FakeHarborScorer(config.task_ids, config.attempts), pins

    def fake_proposer(
        *, run_dir: Path, meta_config: ProviderConfig, e2b_template: str | None
    ) -> _FakeHarborProposer:
        del run_dir
        captured["meta_config"] = meta_config
        captured["proposer_e2b_template"] = e2b_template
        return _FakeHarborProposer()

    monkeypatch.setattr(harness_app_module, "_build_harbor_scorer", fake_scorer)
    monkeypatch.setattr(harness_app_module, "_build_harbor_proposer", fake_proposer)
    return captured


def _invoke_harbor(tmp_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "optimize",
            "pi",
            "harbor",
            "--harbor-config",
            str(tmp_path / "job.yaml"),
            "--task-ids",
            str(tmp_path / "tasks.json"),
            "--iterations",
            "1",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
            *extra,
        ],
    )


def test_harbor_dispatch_runs_the_population_and_publishes_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _harbor_project(tmp_path, monkeypatch)

    result = _invoke_harbor(tmp_path)

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert isinstance(config, _HarborRunConfig)
    assert config.task_ids == ("t1",)
    assert config.attempts == 1
    assert config.reward_mode == "positive-binary"  # the harbor default, not scoring "raw"
    assert config.backend == "local"
    run_config = json.loads((tmp_path / "run" / "run-config.json").read_text(encoding="utf-8"))
    assert run_config["agent"] == "pi"
    assert run_config["iterations"] == 1
    assert run_config["seed_version"] is None  # the built-in seed carries no store version
    assert run_config["task_pins"] == {"t1": "path:/tasks/t1"}
    assert run_config["worker_model"]["model"] == "gpt-5.5"
    assert run_config["proposer_model"]["provider"] == "azure"
    state = json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))
    assert [entry["candidate_id"] for entry in state["outcomes"]] == [
        "candidate-0000",
        "candidate-0001",
    ]
    # Winner publication: a lineage-less doc lands as pi v1 with the champion alias.
    store = HarnessStore(str(tmp_path / ".wmh"))
    saved = store.load("pi")
    assert saved.version == 1
    assert store.aliases("pi")["champion"] == 1
    assert "optimized" in result.output


def test_harbor_checkpoint_then_resume_completes_and_rejects_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _harbor_project(tmp_path, monkeypatch)

    first = _invoke_harbor(tmp_path, "--max-iterations-this-run", "1")
    assert first.exit_code == 0, first.output
    assert "checkpointed" in first.output
    assert not HarnessStore(str(tmp_path / ".wmh")).exists("pi")

    conflict = _invoke_harbor(tmp_path, "--resume", "--reward-mode", "raw")
    assert conflict.exit_code == 2
    flat = " ".join(conflict.output.split())  # rich wraps (and boxes) the error line
    assert "conflicting" in flat
    assert "--reward-mode" in flat

    resumed = runner.invoke(
        app,
        [
            "optimize",
            "pi",
            "harbor",
            "--resume",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "optimized" in resumed.output
    assert HarnessStore(str(tmp_path / ".wmh")).load("pi").version == 1


def test_harbor_errors_when_a_world_model_is_named_harbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _harbor_project(tmp_path, monkeypatch)
    model_dir = tmp_path / ".wmh" / "models" / "harbor"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("", encoding="utf-8")

    result = _invoke_harbor(tmp_path)

    assert result.exit_code == 2
    assert "rename" in result.output


def test_harbor_requires_a_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _harbor_project(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        [
            "optimize",
            "pi",
            "harbor",
            "--harbor-config",
            str(tmp_path / "job.yaml"),
            "--task-ids",
            str(tmp_path / "tasks.json"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )
    assert result.exit_code == 2
    assert "--run-dir is required" in result.output


def test_world_model_path_rejects_harbor_only_flags(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "made", "--run-dir", str(tmp_path / "run"), "--root", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "apply only to the harbor environment" in result.output


def test_harbor_run_config_preserves_sensitive_env_values_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded job template is the PARSED RAW mapping, never a harbor model_dump.

    Harbor's env serializers redact sensitive-named values that differ from this process's
    environment; a round-trip through model_dump would corrupt the recorded run config and
    every trial. MY_API_KEY is sensitive-named and deliberately absent from os.environ.
    """
    captured = _harbor_project(tmp_path, monkeypatch)
    monkeypatch.delenv("MY_API_KEY", raising=False)
    (tmp_path / "job.yaml").write_text(
        _HARBOR_JOB_YAML + "verifier:\n  env:\n    MY_API_KEY: run-specific-secret\n",
        encoding="utf-8",
    )

    result = _invoke_harbor(tmp_path)

    assert result.exit_code == 0, result.output
    run_config = json.loads((tmp_path / "run" / "run-config.json").read_text(encoding="utf-8"))
    assert run_config["harbor_job_template"]["verifier"]["env"]["MY_API_KEY"] == (
        "run-specific-secret"
    )
    config = captured["config"]
    assert isinstance(config, _HarborRunConfig)
    verifier = config.harbor_job_template["verifier"]
    assert isinstance(verifier, dict)
    assert verifier["env"] == {"MY_API_KEY": "run-specific-secret"}


def test_harbor_publication_is_idempotent_across_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _harbor_project(tmp_path, monkeypatch)
    first = _invoke_harbor(tmp_path)
    assert first.exit_code == 0, first.output
    store = HarnessStore(str(tmp_path / ".wmh"))
    assert store.versions("pi") == [1]

    again = runner.invoke(
        app,
        [
            "optimize",
            "pi",
            "harbor",
            "--resume",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )

    assert again.exit_code == 0, again.output
    assert "already published" in again.output
    assert store.versions("pi") == [1]  # no duplicate version, champion did not move again
    assert store.aliases("pi")["champion"] == 1


def test_harbor_rejects_world_model_only_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _harbor_project(tmp_path, monkeypatch)
    result = _invoke_harbor(tmp_path, "--tasks", "tasks.jsonl", "--k", "3")
    assert result.exit_code == 2
    flat = " ".join(result.output.split())
    assert "--tasks" in flat
    assert "world-model environment" in flat


def test_harbor_seed_ref_resolves_through_the_store_and_resume_pins_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'pi@champion' seeds from the store, and a resume uses the RECORDED seed even after
    this run's own publication (or anything else) moves the champion alias."""
    _harbor_project(tmp_path, monkeypatch)
    store = HarnessStore(str(tmp_path / ".wmh"))
    stored_seed = HarnessDoc.baseline("pi").model_copy(update={"name": "pi"})
    store.save_version(stored_seed, alias="champion")  # v1, champion

    first = runner.invoke(
        app,
        [
            "optimize",
            "pi@champion",
            "harbor",
            "--harbor-config",
            str(tmp_path / "job.yaml"),
            "--task-ids",
            str(tmp_path / "tasks.json"),
            "--iterations",
            "1",
            "--max-iterations-this-run",
            "1",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )
    assert first.exit_code == 0, first.output
    assert "checkpointed" in first.output
    run_config = json.loads((tmp_path / "run" / "run-config.json").read_text(encoding="utf-8"))
    assert run_config["seed_version"] == 1
    recorded = tmp_path / "run" / "candidates" / "candidate-0000" / "source" / "SYSTEM.md"
    assert recorded.read_text(encoding="utf-8") == stored_seed.system_prompt()

    # Champion moves before the resume; the run must keep its recorded seed regardless.
    store.save_version(stored_seed.model_copy(update={"version": 0}), alias="champion")  # v2
    resumed = runner.invoke(
        app,
        [
            "optimize",
            "pi@champion",
            "harbor",
            "--resume",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "optimized" in resumed.output
    assert recorded.read_text(encoding="utf-8") == stored_seed.system_prompt()
    assert store.versions("pi") == [1, 2, 3]  # the winner published as the next version


def test_harbor_resume_rejects_changed_model_roles_and_dataset_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _harbor_project(tmp_path, monkeypatch)
    first = _invoke_harbor(tmp_path, "--max-iterations-this-run", "1")
    assert first.exit_code == 0, first.output

    # A mid-run settings.toml edit silently changing the worker model is rejected.
    root = tmp_path / ".wmh"
    changed = ModelRole(provider="azure", model="gpt-6", endpoint="https://x.example")
    kept = ModelRole(provider="azure", model="gpt-5.5", endpoint="https://x.example")
    save_settings(ProjectSettings(models=ModelsSettings(meta=kept, agent=changed)), root)
    role_conflict = _invoke_harbor(tmp_path, "--resume")
    assert role_conflict.exit_code == 2
    assert "models.agent" in " ".join(role_conflict.output.split())

    # A dataset that re-resolves to different task pins is rejected too.
    save_settings(ProjectSettings(models=ModelsSettings(meta=kept, agent=kept)), root)
    captured["pin_salt"] = "@drifted"
    pin_conflict = _invoke_harbor(tmp_path, "--resume")
    assert pin_conflict.exit_code == 2
    assert "resolved differently" in " ".join(pin_conflict.output.split())


def test_harbor_resume_accepts_a_restated_episode_timeout_for_an_e2b_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag consistency is judged on the EFFECTIVE config: restating --episode-timeout on a
    resumed e2b run must not trip the local-backend guard via this invocation's defaults."""
    _harbor_project(tmp_path, monkeypatch)
    first = _invoke_harbor(
        tmp_path, "--backend", "e2b", "--episode-timeout", "120", "--max-iterations-this-run", "1"
    )
    assert first.exit_code == 0, first.output

    resumed = runner.invoke(
        app,
        [
            "optimize",
            "pi",
            "harbor",
            "--resume",
            "--episode-timeout",
            "120",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "optimized" in resumed.output
