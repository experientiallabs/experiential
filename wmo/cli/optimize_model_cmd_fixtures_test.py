"""Shared fixtures for staged model optimizer command tests."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

import wmo.simulation as env_module
from wmo.cli import consent as consent_module
from wmo.cli.app import app
from wmo.common.config import HarnessConfig, save_config
from wmo.common.core.types import (
    Action,
    ActionKind,
    EnvState,
    Observation,
    Session,
    Step,
    Trace,
)
from wmo.common.observability import Phase, RunRecord, UsageTotals, load_runs
from wmo.common.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.common.providers.pool import load_pool
from wmo.optimize.reward import EpisodeScore
from wmo.optimize.routing.compression import CompressionConfig
from wmo.optimize.routing.evaluation import scenario_id
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.pipeline import (
    MANIFEST_FILENAME,
    MATRIX_FILENAME,
    REPORT_FILENAME,
    RunManifest,
    Stage,
)
from wmo.optimize.routing.policy import (
    AZURE_EMBEDDER_DEPLOYMENT,
    AZURE_EMBEDDER_ENV,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
)
from wmo.optimize.routing.report import ImprovementReport
from wmo.optimize.routing.sweep import SweepPlan, plan_sweep, resolve_config
from wmo.optimize.routing.sweep_partial import PartialHeader
from wmo.simulation.ingest.otel_writer import write_traces_jsonl
from wmo.simulation.model.world_model import WorldModel
from wmo.simulation.serving.traces_source import TRACES_FILENAME

runner = CliRunner()


@pytest.fixture(autouse=True)
def _local_model_uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin `--embedder auto` off its local leg for every test in this module.

    Auto prefers the in-process local model when its weights happen to be in THIS machine's
    Hugging Face cache; these tests assert the azure and hashing legs, and must observe the
    same resolution on a machine with the cache warm as on CI without it.
    """
    monkeypatch.setattr(
        "wmo.optimize.routing.policy.default_model_cached", lambda backend=None: False
    )


optimize_module = importlib.import_module("wmo.cli.optimize_model_cmd")


route_module = importlib.import_module("wmo.cli.route_app")


_HELD_OUT_IDS = ("tr-010", "tr-018", "tr-020", "tr-027")


_FRAME_CHARS = frozenset("│┃╭╮╰╯─━┏┓┗┛┡┩┢┪╇╈├┤┬┴┼")


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    """Text with color, whitespace, and rich's box-drawing frame removed.

    Rich wraps (and frames, and at a forced terminal colorizes) everything it prints, so a
    literal substring check against the raw output is a coin flip on where the wrap landed and
    whether the highlighter split a number out of its sentence.
    """
    plain = _ANSI.sub("", text)
    return "".join(ch for ch in plain if not ch.isspace() and ch not in _FRAME_CHARS)


def _says(output: str, phrase: str) -> bool:
    return _flat(phrase) in _flat(output)


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render the plan table wide enough that no cell wraps.

    Rich lays a table out line by line, so a wrapped cell interleaves its continuation with the
    NEXT column's text: at 80 columns "SKIP (matrix.json is current: ...)" comes back cut in
    three and spliced through the plan column, and every assertion on a printed sentence becomes
    an assertion about where the wrap landed. Width is presentation, not behavior.
    """
    monkeypatch.setattr(optimize_module, "_console", Console(width=240))


@pytest.fixture(autouse=True)
def _no_azure_embedder_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the pair `--embedder auto` looks for, so these runs never reach a real resource.

    Without this the suite's behavior depends on the developer's shell: a machine with
    `AZURE_OPENAI_*` exported would resolve auto to text-embedding-3-large and fit against a
    billed embedding API, which is exactly the spend these tests promise not to make. The
    env-present branch is tested by setting them deliberately, on the pure resolver.
    """
    for name in AZURE_EMBEDDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _corpus(count: int = 30) -> list[Trace]:
    """A corpus whose deterministic split leaves a real held-out band, one task per trace."""
    return [
        Trace(
            trace_id=f"tr-{index:03d}",
            steps=[
                Step(
                    action=Action(kind=ActionKind.TOOL_CALL, name="ls", arguments={"path": "."}),
                    observation=Observation(content="a.txt"),
                    task=f"task tr-{index:03d}",
                ),
                Step(
                    action=Action(kind=ActionKind.MESSAGE, content="done"),
                    observation=Observation(content="ok"),
                    task=f"task tr-{index:03d}",
                ),
            ],
        )
        for index in reversed(range(count))
    ]


def _project(tmp_path: Path) -> Path:
    """A built-model artifact dir (config + its own corpus); returns the project root."""
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
    write_traces_jsonl(_corpus(), model_dir / TRACES_FILENAME)
    return root


def _pool_file(tmp_path: Path, *, pricey_out: float = 20.0) -> Path:
    """Two priced candidates, 1/2 and 10/`pricey_out` USD per Mtok, so costs are distinguishable."""
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
        f"output_per_mtok = {pricey_out}\n",
        encoding="utf-8",
    )
    return path


class _FakeWorldModel:
    """`WorldModel`-shaped stub: in-memory sessions, a canned episode score, no LLM at all."""

    def __init__(self, rewards: dict[str, float] | None = None, session_usd: float = 0.02) -> None:
        # Per-candidate rewards keyed by the model id the episode's provider was built with, so a
        # sweep can produce a matrix where one candidate is genuinely better than another.
        self._rewards = rewards or {}
        # What the simulator charges per episode for its OWN serve + judge calls: the
        # world-model side of a sweep's bill, which is metered separately from the candidates.
        self._session_usd = session_usd
        self._frozen = False
        self.tasks: list[str | None] = []
        self.current_model = "cheap-1"

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
        return Session(id=f"s{len(self.tasks)}", task=task, enrich=enrich)

    def step(self, session_id: str, action: Action) -> Observation:
        return Observation(content="ok")

    def score_session(self, session_id: str) -> EpisodeScore:
        reward = self._rewards.get(self.current_model, 0.75)
        return EpisodeScore(reward=reward, success=reward >= 0.5, critique="fine")

    def end_session(self, session_id: str) -> RunRecord:
        return self._usage_record(session_id)

    def session_usage(self, session_id: str) -> RunRecord:
        return self._usage_record(session_id)

    def _usage_record(self, session_id: str) -> RunRecord:
        serve = UsageTotals(
            calls=2, input_tokens=400, output_tokens=60, cost_usd=self._session_usd * 0.75
        )
        judge = UsageTotals(
            calls=1, input_tokens=200, output_tokens=30, cost_usd=self._session_usd * 0.25
        )
        return RunRecord(
            run_id=session_id,
            kind="serve",
            duration_seconds=0.5,
            total=serve.merged(judge),
            by_phase={Phase.SERVE: serve, Phase.JUDGE: judge},
        )


class _ScriptedCandidate:
    """A candidate that calls one tool and then declares itself done.

    `throttled` makes every completion raise instead, the way a rate-limited candidate does: the
    episode errors, `run_episode` records it, and the cell comes back unscored.
    """

    def __init__(
        self, config: ProviderConfig, world_model: _FakeWorldModel, *, throttled: bool = False
    ) -> None:
        self.config = config
        self._world_model = world_model
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
        if self._throttled:
            raise RuntimeError("rate limit exceeded (429)")
        # The env scores on close, after this provider's last call, so recording which candidate
        # is live here is what lets the fake judge give different candidates different rewards.
        self._world_model.current_model = self.config.model
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
        self.asked: list[str] = []

    def ask(self, prompt: str, *, default: bool = True) -> bool:
        self.asked.append(prompt)
        return self._answer


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rewards: dict[str, float] | None = None,
    modules: tuple[object, ...] = (),
    session_usd: float = 0.02,
    throttled_models: frozenset[str] = frozenset(),
) -> _FakeWorldModel:
    """Stub the world model and every pool provider; return the fake for post-run assertions.

    `modules` is retained for call-site compatibility; both `optimize model` and `route`
    resolve the world model through `wmo.simulation.model.load_world_model`, so that is the only
    seam.
    """
    _ = modules
    world_model = _FakeWorldModel(rewards=rewards, session_usd=session_usd)

    def _load(model_dir: Path) -> tuple[WorldModel, Provider]:
        provider = _ScriptedCandidate(
            ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve"), world_model
        )
        return cast("WorldModel", world_model), cast("Provider", provider)

    def _get_provider(config: ProviderConfig, api_key: str | None = None) -> Provider:
        return cast(
            "Provider",
            _ScriptedCandidate(config, world_model, throttled=config.model in throttled_models),
        )

    monkeypatch.setattr("wmo.simulation.model.load_world_model", _load)
    monkeypatch.setattr("wmo.common.providers.pool.get_provider", _get_provider)
    return world_model


def _run(tmp_path: Path, root: Path, *extra: str, pool: Path | None = None) -> Result:
    """Invoke `wmo optimize model support` against the temp project."""
    return runner.invoke(
        app,
        [
            "optimize",
            "model",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool or _pool_file(tmp_path)),
            "--scenarios",
            "3",
            "--max-steps",
            "4",
            *extra,
        ],
    )


def _paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """(matrix, policy, report, manifest) for the `support` model under `root`."""
    model_dir = root / "models" / "support"
    run_dir = model_dir / "optimize"
    return (
        run_dir / MATRIX_FILENAME,
        model_dir / POLICY_FILENAME,
        run_dir / REPORT_FILENAME,
        run_dir / MANIFEST_FILENAME,
    )


def _sweep_plan(tmp_path: Path, root: Path) -> SweepPlan:
    """The plan `_run` builds, so a test can stamp a sidecar with the same identity."""
    model_dir = root / "models" / "support"
    return plan_sweep(
        model_dir=model_dir,
        config=resolve_config(model_dir),
        pool=load_pool(_pool_file(tmp_path)),
        out_path=_paths(root)[0],
        traces_file=None,
        scenarios=3,
        episodes=1,
        max_steps=4,
        assume_input_tokens=2000,
        assume_output_tokens=250,
    )


_ARM = ("--compressor", "truncate", "--aggressiveness", "0.3")


"""A local, free, append-stable compressor at a dial that actually removes words."""


_ARM_LINE = "compressor 'truncate' version 1 at aggressiveness 0.3"


def _stage_rows(output: str) -> list[str]:
    """The plan table's stage column, in order: which stages the run advertised."""
    flat = _flat(output)
    return [stage for stage in (item.value for item in Stage) if stage in flat]


optimize_plan_module = importlib.import_module("wmo.cli.optimize_model_plan")

__all__ = (
    "importlib",
    "json",
    "re",
    "Iterator",
    "contextmanager",
    "Path",
    "cast",
    "pytest",
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
    "Completion",
    "Message",
    "Provider",
    "ProviderConfig",
    "ProviderKind",
    "TokenUsage",
    "VerifyResult",
    "load_pool",
    "EpisodeScore",
    "CompressionConfig",
    "scenario_id",
    "OutcomeMatrix",
    "ScenarioOutcome",
    "MANIFEST_FILENAME",
    "MATRIX_FILENAME",
    "REPORT_FILENAME",
    "RunManifest",
    "Stage",
    "AZURE_EMBEDDER_DEPLOYMENT",
    "AZURE_EMBEDDER_ENV",
    "POLICY_FILENAME",
    "EmbedderSpec",
    "RoutingPolicy",
    "ImprovementReport",
    "SweepPlan",
    "plan_sweep",
    "resolve_config",
    "PartialHeader",
    "write_traces_jsonl",
    "WorldModel",
    "TRACES_FILENAME",
    "runner",
    "_local_model_uncached",
    "optimize_module",
    "route_module",
    "_HELD_OUT_IDS",
    "_FRAME_CHARS",
    "_ANSI",
    "_flat",
    "_says",
    "_wide_console",
    "_no_azure_embedder_env",
    "_corpus",
    "_project",
    "_pool_file",
    "_FakeWorldModel",
    "_ScriptedCandidate",
    "_Answer",
    "_patch_seams",
    "_run",
    "_paths",
    "_sweep_plan",
    "_ARM",
    "_ARM_LINE",
    "_stage_rows",
    "optimize_plan_module",
)
