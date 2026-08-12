"""Shared fixtures for split command-module CLI tests."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from wmo.cli import app, pool_registry
from wmo.cli.pool_registry import read_pool_entries
from wmo.common.config import (
    FIDELITY_TIERS,
    FidelityTier,
    HarnessConfig,
    ModelInfo,
    ModelRole,
    WorldModelStore,
    load_config,
    load_settings,
    save_settings,
)
from wmo.common.core.types import Action, ActionKind, Observation, Step, Trace
from wmo.common.observability.pricing import ModelPrice
from wmo.common.providers.base import (
    Completion,
    EmbedderKind,
    Message,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
    verify_via_ping,
)
from wmo.common.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.common.providers.tinker import _MISSING_TINKER_EXTRA
from wmo.simulation.ingest import VendorPull
from wmo.simulation.model.build import DEFAULT_TRAIN_SPLIT, split_traces, split_traces_3way

cli_app_module = importlib.import_module("wmo.cli.app")


runner = CliRunner()


def _flat(text: str) -> str:
    """Collapse rich wrapping (and typer's error-box borders) for substring asserts."""
    return " ".join(text.replace("│", " ").split())


def _framed(output: str) -> str:
    """Collapse a rich-framed message into one line for assertions."""
    return " ".join(output.replace("│", " ").split())


class FakeProvider:
    """Canned world-model JSON for rollouts/steps; a fixed prompt for GEPA reflection."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")
        self.systems: list[str] = []  # system prompt of every complete() call, for assertions

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        """Return deterministic completion payloads for build-flow tests."""
        self.systems.append(system)
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:
            return Completion(
                text=(
                    '{"format": 0.5, "factuality": 0.5, "consistency": 0.5, '
                    '"realism": 0.5, "quality": 0.5, "critique": "be more specific"}'
                )
            )
        return Completion(text='{"output": "user u1 found", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic embedding for each requested text."""
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:
        """Use the shared ping path without reaching a real backend."""
        return verify_via_ping(self)


def _squashed(text: str) -> str:
    """Whitespace-free view of rich output, so a boxed+wrapped message still matches a substring.

    Typer renders usage errors inside a panel that hard-wraps at the terminal width, which splits
    long paths and command hints across lines; dropping whitespace and the box rules puts them
    back together. Callers squash the expected string the same way.
    """
    return "".join(ch for ch in text if not ch.isspace() and ch not in "│┃")


def _traces_file(tmp_path) -> str:  # noqa: ANN001 - pytest fixture path
    span_llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def patched_provider(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture
    """Swap the real provider registry for the fake everywhere the CLI constructs one.

    Each module binds `get_provider` at its own import time (build.py for the build pipeline,
    loader.py for serve/demo/play), so patch every module-level name plus the registry the lazy
    imports read.
    """
    import sys

    import wmo.common.providers as providers_pkg
    import wmo.common.providers.registry as registry
    import wmo.common.providers.waterfall as waterfall_mod

    fake = FakeProvider()
    # `wmo.simulation.model.__init__` rebinds `build` to the function, shadowing the submodule
    # attribute, so reach module objects through sys.modules rather than attribute access.
    monkeypatch.setattr(
        sys.modules["wmo.simulation.model.build"], "get_provider", lambda config: fake
    )
    # loader.py (serve/demo/play) and the CLI construct through the chain-aware seam.
    monkeypatch.setattr(
        sys.modules["wmo.simulation.model.loader"], "provider_or_chain", lambda config, **kw: fake
    )
    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: fake)
    monkeypatch.setattr(providers_pkg, "provider_or_chain", lambda config, **kw: fake)
    # The pre-build verify guard pings via verify_all/verify_embedder, which construct providers
    # through the registry's own get_provider - patch that too so the guard sees the fake, and
    # patch the name waterfall.py bound at import for its no-chain-file passthrough.
    monkeypatch.setattr(registry, "get_provider", lambda config: fake)
    monkeypatch.setattr(waterfall_mod, "get_provider", lambda config: fake)


def _build(root, name: str, tmp_path) -> None:  # noqa: ANN001 - pytest fixture paths
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            name,
            "--file",
            _traces_file(tmp_path),
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
        ],
    )
    assert result.exit_code == 0, result.output


def _flat(output: str) -> str:
    """Rich wraps error panels; flatten box drawing and newlines before matching a message."""
    return " ".join(output.replace("│", " ").split())


def _accept_every_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub both live pings so `providers set` reaches, and gets through, pool registration.

    Two seams, because the command proves two different things: `verify_all` proves the worker
    provider, and `verify_pool_entry` proves each routing candidate over its own route.
    """
    monkeypatch.setattr(
        "wmo.common.providers.verify_all",
        lambda configs: [
            VerifyResult(ok=True, kind=config.kind, model=config.model) for config in configs
        ],
    )
    monkeypatch.setattr(
        pool_registry,
        "verify_pool_entry",
        lambda entry: VerifyResult(ok=True, kind=entry.kind, model=entry.model),
    )


def _seed_openrouter_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the OpenRouter price resolver at a fixture catalog (the suite never fetches)."""
    catalog = PriceCatalog(
        fetched_at=time.time(),
        source="test fixture",
        prices={
            "anthropic/claude-sonnet-4.5": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)
        },
    )
    path = tmp_path / "openrouter-prices.json"
    path.write_text(catalog.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))


def _record_eval_providers(monkeypatch: pytest.MonkeyPatch) -> list[ProviderConfig]:
    """Capture every ProviderConfig `wmo eval` builds, and answer with the fake provider."""
    seen: list[ProviderConfig] = []
    fake = FakeProvider()

    def record(config: ProviderConfig, **_kwargs: object) -> FakeProvider:
        seen.append(config)
        return fake

    monkeypatch.setattr("wmo.common.providers.provider_or_chain", record)
    monkeypatch.setattr("wmo.common.providers.get_provider", record)
    return seen


def _write_worker_role(root: Path, provider: str, model: str) -> None:
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider=provider, model=model)
    save_settings(settings, root)


def _write_broken_model(root: Path, name: str) -> None:
    (root / "models" / name).mkdir(parents=True)
    (root / "models" / name / "config.toml").write_text("this is not toml =", encoding="utf-8")


def _flat(output: str) -> str:
    """CliRunner output with rich's panel borders and line wrapping removed, for substrings."""
    return " ".join(output.replace("│", " ").split())


def _pull_trace(trace_id: str, *, usable: bool) -> Trace:
    """One single-step trace. `usable=False` makes it degenerate (empty observation)."""
    return Trace(
        trace_id=trace_id,
        source="otel-genai:vendor",
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}),
                observation=Observation(content="found u1" if usable else ""),
            )
        ],
    )


def _many_traces_file(tmp_path, count: int) -> str:  # noqa: ANN001 - pytest fixture path
    """`count` copies of the single-trace export, each under its own trace id."""
    base = Path(_traces_file(tmp_path)).read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    for i in range(count):
        for line in base:
            lines.append(json.dumps({**json.loads(line), "traceId": f"{i:032d}"}))
    path = tmp_path / "many.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


_BROKEN_SETTINGS = pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('[models.worker\nprovider = "openai"\n', "is not valid TOML"),
        ('[models]\nworker = "openai"\n', "does not match the current settings schema"),
    ],
    ids=["malformed-toml", "schema-invalid"],
)


def _write_settings(tmp_path: Path, payload: str) -> Path:
    root = tmp_path / ".wmo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.toml").write_text(payload, encoding="utf-8")
    return root


def _record_verify_all(monkeypatch: pytest.MonkeyPatch, pinged: list[ProviderConfig]) -> None:
    """Record exactly which providers `providers verify` decided to ping, and report them ok."""

    def fake_verify_all(configs: list[ProviderConfig]) -> list[VerifyResult]:
        pinged.extend(configs)
        return [VerifyResult(ok=True, kind=c.kind, model=c.model) for c in configs]

    monkeypatch.setattr("wmo.common.providers.verify_all", fake_verify_all)


def _azure_worker_settings(monkeypatch: pytest.MonkeyPatch, deployment: str | None) -> None:
    from wmo.common.config.settings import ModelRole, ModelsSettings, ProjectSettings

    monkeypatch.setattr(
        command_common_module,
        "load_settings_or_abort",
        lambda: ProjectSettings(
            models=ModelsSettings(
                worker=ModelRole(
                    provider="azure",
                    model="gpt-5.4",
                    endpoint="https://azure.example/v1",
                    deployment=deployment,
                )
            )
        ),
    )


def _build_cli_train_split_default() -> float:
    """The `--train-split` default `wmo build` registers, read off the Typer option itself."""
    option = inspect.signature(build_module.build).parameters["train_split"].default
    return cast(float, option.default)


def _eval_cli_train_split_default() -> float:
    """The train split `wmo eval` resolves when the user passes no `--train-split`."""
    # `_eval_options` is the resolver under test: it is where `wmo eval` turns "no flag given"
    # into a concrete split, so asserting on its output is asserting on the real default.
    options = eval_module._eval_options(
        prompt_file=None,
        train_split=None,
        embed_dim=None,
        rag=None,
        sample_turns=None,
        seed=None,
        top_k=None,
    )
    return options.train_split


# The lean root app composes these commands; tests target their owning modules rather than
# retaining aliases in the composition layer.
catalog_module = importlib.import_module("wmo.cli.catalog_cmd")
build_module = importlib.import_module("wmo.cli.build_cmd")
eval_module = importlib.import_module("wmo.cli.eval_cmd")
command_common_module = importlib.import_module("wmo.cli.command_common")

__all__ = (
    "importlib",
    "inspect",
    "json",
    "os",
    "sys",
    "time",
    "Path",
    "SimpleNamespace",
    "cast",
    "pytest",
    "typer",
    "ValidationError",
    "CliRunner",
    "app",
    "pool_registry",
    "read_pool_entries",
    "FIDELITY_TIERS",
    "FidelityTier",
    "HarnessConfig",
    "ModelInfo",
    "ModelRole",
    "WorldModelStore",
    "load_config",
    "load_settings",
    "save_settings",
    "Action",
    "ActionKind",
    "Observation",
    "Step",
    "Trace",
    "ModelPrice",
    "Completion",
    "EmbedderKind",
    "Message",
    "ProviderConfig",
    "ProviderKind",
    "VerifyResult",
    "verify_via_ping",
    "CATALOG_PATH_ENV",
    "PriceCatalog",
    "_MISSING_TINKER_EXTRA",
    "VendorPull",
    "DEFAULT_TRAIN_SPLIT",
    "split_traces",
    "split_traces_3way",
    "cli_app_module",
    "runner",
    "_flat",
    "_framed",
    "FakeProvider",
    "_squashed",
    "_traces_file",
    "patched_provider",
    "_build",
    "_accept_every_provider",
    "_seed_openrouter_catalog",
    "_record_eval_providers",
    "_write_worker_role",
    "_write_broken_model",
    "_pull_trace",
    "_many_traces_file",
    "_BROKEN_SETTINGS",
    "_write_settings",
    "_record_verify_all",
    "_azure_worker_settings",
    "_build_cli_train_split_default",
    "_eval_cli_train_split_default",
    "catalog_module",
    "build_module",
    "eval_module",
    "command_common_module",
)
