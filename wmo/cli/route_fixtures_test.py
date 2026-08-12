"""Shared fixtures for route command tests."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from filelock import FileLock
from rich.console import Console
from typer.testing import CliRunner, Result

import wmo.simulation as env_module
from wmo.cli import consent as consent_module
from wmo.cli.app import app
from wmo.common.config import HarnessConfig, save_config
from wmo.common.core.artifacts import canonical_json_bytes
from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Session, Step, Trace
from wmo.common.judging.episode import EpisodeScore
from wmo.common.observability import Phase, RunRecord, UsageTotals, load_runs
from wmo.common.project import ProjectConfig, ProjectStore
from wmo.common.providers import pool as pool_module
from wmo.common.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.common.providers.openrouter import OPENROUTER_API_KEY_ENV
from wmo.common.providers.pool import PoolEntry, load_pool
from wmo.common.providers.registry import get_provider as registry_get_provider
from wmo.common.tasks import TaskCase, TaskSet, ToolSchema
from wmo.optimize.model.store import DistillModelCard
from wmo.optimize.routing import evaluate_policy
from wmo.optimize.routing.compression import (
    CompressionConfig,
    Compressor,
    TruncateCompressor,
    register_compressor,
    registered_compressor_ids,
    same_compression,
)
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.policy import POLICY_FILENAME, RoutingPolicy, select_model
from wmo.optimize.routing.sweep_partial import PartialHeader, PlanIdentity
from wmo.runtime.agents.llm import DEFAULT_HISTORY_CHARS
from wmo.simulation.model.world_model import WorldModel

runner = CliRunner()


@pytest.fixture(autouse=True)
def _local_model_uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin `--embedder auto` off its local leg for every test in this module.

    Auto prefers the in-process local model when its weights happen to be in THIS machine's
    Hugging Face cache; the fit tests here assert the hashing and azure legs, and must observe
    the same resolution on a machine with the cache warm as on CI without it.
    """
    monkeypatch.setattr(
        "wmo.optimize.routing.policy.default_model_cached", lambda backend=None: False
    )


def _arm(compression: CompressionConfig | None) -> tuple[str, str, float]:
    """The D-COMPRESS fields an episode measured under `compression` would carry.

    A matrix records the arm its rewards were produced under, and `fit` refuses to stamp a
    policy whose compression config disagrees, so a fixture fitting `--compressor` has to look
    like episodes that actually ran that way.
    """
    if compression is None:
        return "", "", 0.0
    return (
        compression.compressor_id,
        compression.compressor_version,
        compression.aggressiveness,
    )


def _matrix_file(tmp_path: Path, *, compression: CompressionConfig | None = None) -> Path:
    """The uncompressed arm by default; `compression` stamps the rows as that arm instead."""
    pool = [
        PoolEntry(
            name="a", kind=ProviderKind.OPENAI, model="a", input_per_mtok=1.0, output_per_mtok=1.0
        ),
        PoolEntry(
            name="b", kind=ProviderKind.OPENAI, model="b", input_per_mtok=1.0, output_per_mtok=1.0
        ),
    ]
    arm_id, arm_version, arm_aggressiveness = _arm(compression)
    outcomes = []
    tasks = {
        "s1": "SELECT count(*) FROM t",
        "s2": "SELECT name FROM users WHERE id = 4",
        "s3": "write a poem about rivers",
        "s4": "draft a thank-you note",
    }
    for sid, task in tasks.items():
        sql = sid in ("s1", "s2")
        for model in ("a", "b"):
            wins = (model == "a") == sql
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=sid,
                    task=task,
                    model=model,
                    reward=1.0 if wins else 0.0,
                    success=wins,
                    cost_usd=0.001,
                    compressor_id=arm_id,
                    compressor_version=arm_version,
                    aggressiveness=arm_aggressiveness,
                )
            )
    path = tmp_path / "matrix.json"
    OutcomeMatrix(pool=pool, outcomes=outcomes).save(path)
    return path


def _fit_then_report(tmp_path: Path, *extra: str) -> tuple[Result, Path]:
    """Fit a knn policy, then report over the same matrix with `extra` report flags.

    knn because only a dialable policy produces routed detents, and so a curve at all.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    fit = runner.invoke(
        app,
        [
            *("optimize", "route", "fit", str(matrix_file)),
            *("--kind", "knn", "--fallback", "a", "--out", str(policy_file)),
            *("--z", "0.5", "--rag-num", "3", "--min-pairs", "2"),
        ],
    )
    assert fit.exit_code == 0, fit.output
    report_file = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            *("optimize", "route", "report", str(matrix_file), str(policy_file)),
            *("--baseline", "a", "--out", str(report_file)),
            *extra,
        ],
    )
    return result, report_file.parent / "pareto.json"


def _knn_matrix_file(
    tmp_path: Path,
    *,
    flip: bool = False,
    name: str = "knn_matrix.json",
    compression: CompressionConfig | None = None,
) -> Path:
    """Twelve scenarios: enough neighbors per query for a guarded fit to route at all.

    `flip` swaps which model wins each half, so two matrices built here disagree on every cell.
    That is what makes an artifact mix-up observable: a policy fitted on one and served the
    other's evidence routes every request the wrong way.
    """
    pool = [
        PoolEntry(
            name="a", kind=ProviderKind.OPENAI, model="a", input_per_mtok=1.0, output_per_mtok=1.0
        ),
        PoolEntry(
            name="b", kind=ProviderKind.OPENAI, model="b", input_per_mtok=1.0, output_per_mtok=1.0
        ),
    ]
    sql = [
        "SELECT count(*) FROM orders WHERE total > 100",
        "SELECT name FROM users WHERE id = 4",
        "SELECT avg(price) FROM products GROUP BY category",
        "SELECT id FROM events WHERE kind = 'click'",
        "SELECT max(score) FROM matches WHERE season = 2025",
        "SELECT city FROM stores WHERE stock > 0",
    ]
    prose = [
        "write a friendly email to the team about the offsite",
        "write a warm welcome note for new employees",
        "write a short thank-you message for the organizers",
        "write a cheerful newsletter intro about spring",
        "write a gentle reminder about the expense deadline",
        "write a farewell note for a departing teammate",
    ]
    arm_id, arm_version, arm_aggressiveness = _arm(compression)
    outcomes = []
    for group, tasks in (("sql", sql), ("prose", prose)):
        for index, task in enumerate(tasks):
            for model in ("a", "b"):
                wins = ((model == "a") == (group == "sql")) != flip
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=f"{group}:{index}",
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001,
                        compressor_id=arm_id,
                        compressor_version=arm_version,
                        aggressiveness=arm_aggressiveness,
                    )
                )
    path = tmp_path / name
    OutcomeMatrix(pool=pool, outcomes=outcomes).save(path)
    return path


def _fit_knn(matrix_file: Path, policy_file: Path, *, fallback: str = "a") -> Result:
    """Run `route fit --kind knn` with the neighbor budget this twelve-scenario matrix needs."""
    return runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "knn",
            "--fallback",
            fallback,
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )


def _fitted_knn_policy(tmp_path: Path) -> Path:
    policy_file = tmp_path / POLICY_FILENAME
    result = _fit_knn(_knn_matrix_file(tmp_path), policy_file)
    assert result.exit_code == 0, result.output
    return policy_file


def _run_dir(tmp_path: Path, sampler: str = "tinker://fake/sampler/final/0") -> Path:
    """A distillation run dir with just the artifact `route student` reads."""
    run_dir = tmp_path / "distill" / "support"
    run_dir.mkdir(parents=True, exist_ok=True)
    card = DistillModelCard(
        base_model="Qwen/Qwen3-8B",
        lora_rank=32,
        teacher_model="glm-5.2",
        sampler_path=sampler,
        steps_completed=200,
    )
    (run_dir / "model_card.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def _built_model(tmp_path: Path, name: str = "support") -> Path:
    """A world model dir as `WorldModelStore` recognizes one (a dir carrying config.toml)."""
    model_dir = tmp_path / "models" / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("", encoding="utf-8")
    return model_dir


def _add_student(tmp_path: Path, pool_file: Path, *, name: str = "student") -> Result:
    return runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--name",
            name,
            "--pool",
            str(pool_file),
        ],
    )


route_module = importlib.import_module("wmo.cli.route_app")


_HELD_OUT_IDS = ("tr-010", "tr-018", "tr-020", "tr-027")


_HELD_OUT_TOOL = "holdout_only"


def _corpus(count: int = 30) -> list[Trace]:
    """A trace corpus whose split has a real held-out band, one task prompt per trace.

    Emitted in DESCENDING id order on purpose: the sweep sorts the held-out band by trace id
    before applying `--scenarios`, and a corpus already in id order would make that sort
    unobservable (any assertion on which prefix was cut would hold without it).
    """
    traces: list[Trace] = []
    for index in reversed(range(count)):
        trace_id = f"tr-{index:03d}"
        # Held-out traces call a tool no train trace does, so a tools hint derived from the wrong
        # band is visible in the candidate's system prompt.
        tool = _HELD_OUT_TOOL if trace_id in _HELD_OUT_IDS else "ls"
        traces.append(
            Trace(
                trace_id=trace_id,
                steps=[
                    Step(
                        action=Action(
                            kind=ActionKind.TOOL_CALL, name=tool, arguments={"path": "."}
                        ),
                        observation=Observation(content="a.txt"),
                        task=f"task {trace_id}",
                    ),
                    Step(
                        action=Action(kind=ActionKind.MESSAGE, content="done"),
                        observation=Observation(content="ok"),
                        task=f"task {trace_id}",
                    ),
                ],
            )
        )
    return traces


def _write_task_set(root: Path, traces: list[Trace]) -> None:
    """Persist a verified immutable task set for direct routing command fixtures."""
    project = ProjectStore(root, "default")
    project.initialize(ProjectConfig(project_id="default"))
    ordered = tuple(sorted(traces, key=lambda trace: trace.trace_id))
    held_out_ids = set(_HELD_OUT_IDS).intersection(trace.trace_id for trace in ordered)
    if not held_out_ids:
        held_out_ids = {trace.trace_id for trace in ordered[: min(2, len(ordered))]}
    tasks = tuple(
        TaskCase(
            task_id=trace.trace_id,
            lineage_group_id=f"lineage-{trace.trace_id}",
            partition="held_out" if trace.trace_id in held_out_ids else "fit",
            instruction=trace.steps[0].task or f"task {trace.trace_id}",
            tools=(
                ToolSchema(
                    name=trace.steps[0].action.name or "message",
                    description="Fixture task tool.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                ),
            ),
            workload_weight=1.0,
            source_trace_ids=(trace.trace_id,),
        )
        for trace in ordered
    )
    payload = b"\n".join(canonical_json_bytes(task) for task in tasks) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test",
        task_set_id="task-set-routing",
        task_ids=tuple(task.task_id for task in tasks),
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(payload).hexdigest(),
    )
    project.artifacts.write(
        artifact_id=task_set.task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={
            "tasks.jsonl": payload,
            "task-set.json": canonical_json_bytes(task_set),
        },
    )


def _project(tmp_path: Path, *, traces: list[Trace] | None) -> Path:
    """Write a minimal model config and direct immutable task-set fixture root."""
    root = tmp_path / ".wmo"
    model_dir = root / "models" / "support"
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve")],
            serve_provider=ProviderKind.ANTHROPIC,
            train_split=0.8,
        ),
        model_dir,
    )
    if traces is not None:
        _write_task_set(root, traces)
    return root


def _pool_file(tmp_path: Path) -> Path:
    """Two priced candidates: 1/2 and 10/20 USD per Mtok, so cost lines are distinguishable."""
    path = tmp_path / "pool.toml"
    path.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "pricey"\n'
        'kind = "openai"\n'
        'model = "pricey-1"\n'
        "input_per_mtok = 10.0\n"
        "output_per_mtok = 20.0\n",
        encoding="utf-8",
    )
    return path


class _FakeWorldModel:
    """`WorldModel`-shaped stub: in-memory sessions, a canned episode score, no LLM at all."""

    def __init__(
        self,
        reward: float = 0.75,
        judge_fails_on: frozenset[str] = frozenset(),
        session_usd: float = 0.0,
    ) -> None:
        self._reward = reward
        # What this fake charges per session for its OWN serve + judge calls, which is the
        # world-model side of a sweep's bill. Zero by default so existing expectations are
        # unchanged; a test that cares about that side sets it.
        self._session_usd = session_usd
        self._frozen = False
        # Tasks whose judge call raises, for every candidate: a scenario the whole pool loses.
        self._judge_fails_on = judge_fails_on
        self._task_of: dict[str, str | None] = {}
        self.tasks: list[str | None] = []
        self.scored: list[str] = []
        self.opened_frozen: list[bool] = []  # was index enrichment suspended for this episode?

    @contextmanager
    def frozen(self) -> Iterator[_FakeWorldModel]:
        self._frozen = True
        try:
            yield self
        finally:
            self._frozen = False

    def new_session(
        self, task: str | None = None, seed_state: EnvState | None = None, *, enrich: bool = True
    ) -> Session:
        self.tasks.append(task)
        self.opened_frozen.append(self._frozen)
        session = Session(id=f"s{len(self.tasks)}", task=task, enrich=enrich)
        self._task_of[session.id] = task
        return session

    def step(self, session_id: str, action: Action) -> Observation:
        return Observation(content="ok")

    def score_session(self, session_id: str) -> EpisodeScore:
        task = self._task_of.get(session_id)
        if task is not None and task in self._judge_fails_on:
            # WorldModelEnv.close preserves this and `last_score` re-raises it, which is how a
            # throttled judge leaves a cell unscored for every candidate that ran the scenario.
            raise RuntimeError("judge throttled (429)")
        self.scored.append(session_id)
        return EpisodeScore(reward=self._reward, success=True, critique="fine")

    def end_session(self, session_id: str) -> RunRecord:
        return self._usage_record(session_id)

    def session_usage(self, session_id: str) -> RunRecord:
        return self._usage_record(session_id)

    def _usage_record(self, session_id: str) -> RunRecord:
        totals = UsageTotals(
            calls=1, input_tokens=100, output_tokens=20, cost_usd=self._session_usd
        )
        return RunRecord(
            run_id=session_id,
            kind="serve",
            duration_seconds=0.5,
            total=totals,
            by_phase={Phase.SERVE: totals},
        )


class _ScriptedCandidate:
    """A candidate model that calls one tool and then declares itself done.

    `throttled` makes every completion raise instead, the way a rate-limited candidate does: the
    episode errors, `run_episode` records it, and the cell comes back unscored.
    """

    def __init__(
        self, config: ProviderConfig, systems: list[str], *, throttled: bool = False
    ) -> None:
        self.config = config
        self._systems = systems
        self._throttled = throttled
        self._script = ['{"tool": "ls", "arguments": {}}', '{"done": true, "summary": "ok"}']
        self._index = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self._systems.append(system)
        if self._throttled:
            raise RuntimeError("rate limit exceeded (429)")
        text = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return Completion(text=text, usage=TokenUsage(input_tokens=10, output_tokens=5))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


class _Answer:
    """A `rich.prompt.Confirm` stand-in that always answers the same way."""

    def __init__(self, answer: bool) -> None:
        self._answer = answer

    def ask(self, prompt: str, *, default: bool = True) -> bool:
        return self._answer


class _Seams:
    """What the sweep's two stubbed seams recorded, for post-run assertions."""

    def __init__(self, world_model: _FakeWorldModel) -> None:
        self.world_model = world_model
        self.built_providers: list[str] = []  # one entry per candidate provider constructed
        self.systems: list[str] = []  # every system prompt a candidate was called with


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reward: float = 0.75,
    no_scoring: bool = False,
    throttled_models: frozenset[str] = frozenset(),
    throttled_episodes: dict[str, tuple[bool, ...]] | None = None,
    judge_fails_on: frozenset[str] = frozenset(),
    real_kinds: frozenset[ProviderKind] = frozenset(),
    session_usd: float = 0.0,
) -> _Seams:
    """Stub the world model and the pool's provider construction; return the recorder.

    Args:
        monkeypatch: The patcher whose lifetime the stubs live for.
        reward: Reward the fake judge returns for every scored episode.
        no_scoring: Build the env WITHOUT `score_on_close`, which is what the pre-change code path
            would amount to: it exists so a test can show the difference is observable.
        throttled_models: Provider model ids (`PoolEntry.model`) whose completions raise, so that
            candidate's cells come back unscored while the others are scored.
        throttled_episodes: Per-model cycle of throttled flags over each scenario's episodes:
            `{"pricey-1": (False, True, True)}` keeps episode 0 and loses episodes 1 and 2 of EVERY
            scenario. `evaluate_pool` builds one provider per episode in scenario-major order, so
            the cycle position is the episode index within the scenario. This is how a candidate
            comes back with the same scenarios as the others but FEWER scored episodes on them.
        judge_fails_on: Scenario tasks whose episode SCORING raises, for every candidate: the
            whole pool loses those scenarios together.
        real_kinds: Provider kinds to construct for real instead of faking, so a test can exercise
            a backend that refuses its own config or cannot build its lazy client. Construction and
            preparation are both request-free, and no real provider is ever called (the sweep must
            fail before any cell runs).
    """
    seams = _Seams(
        _FakeWorldModel(reward=reward, judge_fails_on=judge_fails_on, session_usd=session_usd)
    )
    episode_cycles = throttled_episodes or {}

    def _load(model_dir: Path) -> tuple[WorldModel, Provider]:
        provider = _ScriptedCandidate(
            ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve"), []
        )
        return cast("WorldModel", seams.world_model), cast("Provider", provider)

    def _get_provider(config: ProviderConfig, api_key: str | None = None) -> Provider:
        if config.kind in real_kinds:
            return registry_get_provider(config, api_key=api_key)
        seams.built_providers.append(config.model)
        cycle = episode_cycles.get(config.model)
        throttled = config.model in throttled_models
        if cycle:
            # The pre-flight builds one provider per candidate before any episode runs, so the
            # sweep's first episode is this model's SECOND construction; from there the count is
            # the episode index (`evaluate_pool` builds one provider per episode).
            episode = seams.built_providers.count(config.model) - 2
            throttled = throttled or (episode >= 0 and cycle[episode % len(cycle)])
        return cast(
            "Provider",
            _ScriptedCandidate(config, seams.systems, throttled=throttled),
        )

    monkeypatch.setattr("wmo.simulation.model.load_world_model", _load)
    monkeypatch.setattr("wmo.common.providers.pool.get_provider", _get_provider)
    if no_scoring:
        real = env_module.WorldModelEnv
        monkeypatch.setattr(
            env_module,
            "WorldModelEnv",
            lambda world_model, *, score_on_close=False: real(world_model),
        )
    return seams


def _sweep(tmp_path: Path, root: Path, *extra: str) -> tuple[Path, Result]:
    """Invoke `wmo optimize route sweep` against the temp project; return (out path, result)."""
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "--root",
            str(root),
            "--pool",
            str(_pool_file(tmp_path)),
            "--out",
            str(out),
            *extra,
        ],
    )
    return out, result


_FRAME_CHARS = frozenset("│┃╭╮╰╯─━┏┓┗┛┡┩┢┪╇╈├┤┬┴┼")


def _flat(text: str) -> str:
    """Text with whitespace and rich's box-drawing frame removed.

    Typer renders a usage error inside a rich panel and the cost estimate inside a rich table,
    both of which wrap (and frame) long values, so a literal substring check against the raw
    output is a coin flip on where the wrap landed. Assert against this instead.
    """
    return "".join(ch for ch in text if not ch.isspace() and ch not in _FRAME_CHARS)


def _says(result_output: str, phrase: str) -> bool:
    """Whether the CLI said `phrase`, ignoring wherever rich wrapped the line."""
    return _flat(phrase) in _flat(result_output)


def _no_traceback(result: Result) -> bool:
    return result.exception is None or isinstance(result.exception, SystemExit)


def _unscored_matrix_file(tmp_path: Path) -> Path:
    """What a sweep writes when every episode errored: rows on disk, not one reward.

    `sweep` still saves this matrix (the cells were paid for and their `error` fields are the
    diagnosis) and exits 1 saying "fitting will fail", so it is a state a user reaches `fit`
    from rather than an invented one.
    """
    matrix = OutcomeMatrix.load(_matrix_file(tmp_path))
    path = tmp_path / "unscored.json"
    OutcomeMatrix(
        pool=matrix.pool,
        outcomes=[o.model_copy(update={"reward": None, "success": False}) for o in matrix.outcomes],
    ).save(path)
    return path


def _report(matrix_file: Path, policy_file: Path, out: Path, *, baseline: str = "a") -> Result:
    """`route report`, always with an explicit --out so nothing lands in the working dir."""
    return runner.invoke(
        app,
        [
            "optimize",
            "route",
            "report",
            str(matrix_file),
            str(policy_file),
            "--baseline",
            baseline,
            "--out",
            str(out),
        ],
    )


def _deepswe_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny publisher-shaped DeepSWE source directory plus its embedding cache."""
    import numpy as np

    from wmo.optimize.routing.deepswe import PROMPT_BOILERPLATE

    source = tmp_path / "deepswe-source"
    trials: list[dict[str, object]] = []
    tasks = ["alpha", "beta", "gamma"]
    (source / "deep-swe-main" / "tasks").mkdir(parents=True)
    for task in tasks:
        task_dir = source / "deep-swe-main" / "tasks" / task
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text(
            f"Fix the {task} bug.{PROMPT_BOILERPLATE}", encoding="utf-8"
        )
        for index in range(2):
            for config, model, effort, f2p, passed, cost in (
                ("mini_swe_agent_claude_opus_5_high", "claude-opus-5", "high", 1.0, True, 2.0),
                ("mini_swe_agent_gpt_5_6_sol_medium", "gpt-5-6-sol", "medium", 0.5, False, 1.0),
            ):
                trials.append(
                    {
                        "config": config,
                        "task_name": task,
                        "trial_name": f"{config}-{task}-{index}",
                        "included_in_score": True,
                        "model": model,
                        "reasoning_effort": effort,
                        "f2p": f2p,
                        "passed": passed,
                        "cost_usd": cost,
                        "outcome": "pass" if passed else "fail",
                        "n_agent_steps": 5,
                        "n_input_tokens": 1000,
                        "n_output_tokens": 100,
                        "n_cache_tokens": 500,
                    }
                )
    (source / "trials.json").write_text(
        json.dumps({"n_trials": len(trials), "rows": trials}), encoding="utf-8"
    )
    (source / "tasks.json").write_text(
        json.dumps(
            {
                "n_tasks": len(tasks),
                "rows": [{"id": task, "repository": f"org/{task}"} for task in tasks],
            }
        ),
        encoding="utf-8",
    )
    (source / "leaderboard-live.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"config": "mini_swe_agent_claude_opus_5_high", "pass_at_1": 1.0},
                    {"config": "mini_swe_agent_gpt_5_6_sol_medium", "pass_at_1": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "embeddings.json"
    rng = np.random.default_rng(11)
    cache.write_text(
        json.dumps({task: rng.normal(size=16).tolist() for task in tasks}), encoding="utf-8"
    )
    return source, cache


route_sweep_module = importlib.import_module("wmo.cli.route_sweep_cmd")

__all__ = (
    "importlib",
    "itertools",
    "json",
    "sys",
    "Counter",
    "Iterator",
    "contextmanager",
    "Path",
    "cast",
    "pytest",
    "FileLock",
    "Console",
    "CliRunner",
    "Result",
    "env_module",
    "consent_module",
    "app",
    "HarnessConfig",
    "save_config",
    "Action",
    "ActionKind",
    "EnvState",
    "Observation",
    "Session",
    "Step",
    "Trace",
    "Phase",
    "RunRecord",
    "UsageTotals",
    "load_runs",
    "pool_module",
    "Completion",
    "Message",
    "Provider",
    "ProviderConfig",
    "ProviderKind",
    "TokenUsage",
    "VerifyResult",
    "OPENROUTER_API_KEY_ENV",
    "PoolEntry",
    "load_pool",
    "registry_get_provider",
    "DistillModelCard",
    "EpisodeScore",
    "evaluate_policy",
    "CompressionConfig",
    "Compressor",
    "TruncateCompressor",
    "register_compressor",
    "registered_compressor_ids",
    "same_compression",
    "OutcomeMatrix",
    "ScenarioOutcome",
    "POLICY_FILENAME",
    "RoutingPolicy",
    "select_model",
    "PartialHeader",
    "PlanIdentity",
    "DEFAULT_HISTORY_CHARS",
    "WorldModel",
    "runner",
    "_local_model_uncached",
    "_arm",
    "_matrix_file",
    "_fit_then_report",
    "_knn_matrix_file",
    "_fit_knn",
    "_fitted_knn_policy",
    "_run_dir",
    "_built_model",
    "_add_student",
    "route_module",
    "_HELD_OUT_IDS",
    "_HELD_OUT_TOOL",
    "_corpus",
    "_write_task_set",
    "_project",
    "_pool_file",
    "_FakeWorldModel",
    "_ScriptedCandidate",
    "_Answer",
    "_Seams",
    "_patch_seams",
    "_sweep",
    "_FRAME_CHARS",
    "_flat",
    "_says",
    "_no_traceback",
    "_unscored_matrix_file",
    "_report",
    "_deepswe_fixture",
    "route_sweep_module",
)
