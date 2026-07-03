"""Tests for the CLI: command surface + build/list/play driven via CliRunner (fake provider)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from wmh.cli import app
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, verify_via_ping

# `wmh.cli`'s `app` attribute (the Typer object) shadows the `wmh.cli.app` submodule on
# plain `import wmh.cli.app as ...`; go through importlib to monkeypatch module globals.
cli_app_module = importlib.import_module("wmh.cli.app")

runner = CliRunner()


class FakeProvider:
    """Canned world-model JSON for rollouts/steps; a fixed prompt for GEPA reflection."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:
            return Completion(text='{"score": 0.5, "critique": "be more specific"}')
        return Completion(text='{"output": "user u1 found", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        # The pre-build verify guard pings through this; delegate to the shared ping so the fake
        # reports ok without hitting a real backend.
        return verify_via_ping(self)


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

    import wmh.providers as providers_pkg
    import wmh.providers.registry as registry

    fake = FakeProvider()
    # `wmh.engine.__init__` rebinds the name `build` to the function, shadowing the submodule
    # attribute, so reach module objects through sys.modules rather than attribute access.
    for module_name in ("wmh.engine.build", "wmh.engine.loader"):
        monkeypatch.setattr(sys.modules[module_name], "get_provider", lambda config: fake)
    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: fake)
    # The pre-build verify guard pings via verify_all/verify_embedder, which construct providers
    # through the registry's own get_provider — patch that too so the guard sees the fake.
    monkeypatch.setattr(registry, "get_provider", lambda config: fake)


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
            "--gepa-budget",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_exposes_the_small_command_set() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    assert names == {"build", "list", "serve", "demo", "eval", "play"}


@pytest.mark.parametrize("args", [[], ["providers"], ["examples"], ["config"]])
def test_bare_invocation_shows_help(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert "Missing command" not in result.output
    assert "Usage:" in result.output
    assert "--help" in result.output
    # Bare invocation keeps the usage-error exit code (click >=8.2), unlike explicit --help
    # which exits 0 — scripts can still tell "asked for help" from "forgot the command".
    assert result.exit_code == 2


def test_build_rejects_invalid_name_flag_with_friendly_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        ["build", "--name", "tau/bench", "--file", _traces_file(tmp_path), "--no-interactive"],
    )
    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "invalid world model name" in result.output


def test_examples_run_rejects_invalid_name_with_friendly_error() -> None:
    result = runner.invoke(app, ["examples", "run", "tau bench"])
    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "unknown example" in result.output


def test_serve_rejects_invalid_name_with_friendly_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["serve", "--name", "tau bench", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "invalid world model name" in result.output


def test_examples_discovery_skips_unresolvable_names(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # A dir whose name validate_name rejects can never be run, so list (and the "available:"
    # hint in the unknown-example error) must not advertise it.
    for dirname in ("good-example", "tau bench"):
        example = tmp_path / dirname
        example.mkdir()
        (example / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(cli_app_module, "_examples_root", lambda: tmp_path)

    listed = runner.invoke(app, ["examples", "list"])
    assert listed.exit_code == 0, listed.output
    assert "good-example" in listed.output
    assert "tau bench" not in listed.output

    unknown = runner.invoke(app, ["examples", "run", "nope"])
    assert unknown.exit_code == 2
    assert "available: good-example" in unknown.output


def test_main_entry_loads_dotenv_before_dispatch(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # The persistence half of the wizard's credential flow: keys saved to .env must be back in
    # os.environ on the next `wmh` invocation (main), and importing the module must NOT load.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("WMH_TEST_MAIN_VAR=loaded\n", encoding="utf-8")
    monkeypatch.delenv("WMH_TEST_MAIN_VAR", raising=False)
    monkeypatch.setattr(cli_app_module, "app", lambda: None)
    cli_app_module.main()
    assert os.environ["WMH_TEST_MAIN_VAR"] == "loaded"


def test_demo_replays_a_sampled_scenario_open_loop(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "demo-model", tmp_path)
    result = runner.invoke(
        app,
        [
            "demo",
            "--name",
            "demo-model",
            "--root",
            str(root),
            "--traces",
            _traces_file(tmp_path),
            "--seed",
            "0",
            "--steps",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "replaying scenario" in result.output
    assert "predicted" in result.output
    assert "actual" in result.output
    assert "exact matches" in result.output


def test_providers_subcommand_is_registered() -> None:
    group_names = {group.name for group in app.registered_groups}
    assert "providers" in group_names
    assert "examples" in group_names
    assert "config" in group_names


def test_config_telemetry_command_manages_project_settings(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"

    disabled = runner.invoke(app, ["config", "telemetry", "disable", "--root", str(root)])
    assert disabled.exit_code == 0, disabled.output
    assert "telemetry disabled" in disabled.output
    assert "enabled = false" in (root / "settings.toml").read_text(encoding="utf-8")

    status = runner.invoke(app, ["config", "telemetry", "--root", str(root)])
    assert status.exit_code == 0, status.output
    assert "telemetry disabled" in status.output

    enabled = runner.invoke(app, ["config", "telemetry", "enable", "--root", str(root)])
    assert enabled.exit_code == 0, enabled.output
    assert "telemetry enabled" in enabled.output


def test_examples_list_shows_task_folders() -> None:
    result = runner.invoke(app, ["examples", "list"])
    assert result.exit_code == 0, result.output
    assert "tau-bench" in result.output
    assert "swe-bench" in result.output
    assert "terminal-tasks" in result.output


def test_examples_run_invokes_task_launcher(monkeypatch) -> None:  # noqa: ANN001
    seen: dict[str, object] = {}

    def fake_run(command: list[str], *, cwd: object, check: bool) -> subprocess.CompletedProcess:
        seen["command"] = command
        seen["cwd"] = cwd
        seen["check"] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(app, ["examples", "run", "tau-bench", "--", "--trace", "0"])

    assert result.exit_code == 0, result.output
    command = cast(list[str], seen["command"])
    assert command[0].endswith("examples/tau-bench/run.sh")
    assert command[1:] == ["--trace", "0"]
    assert str(seen["cwd"]).endswith("examples/tau-bench")
    assert seen["check"] is False


def test_eval_trace_file_command_still_scores(patched_provider, tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        ["eval", _traces_file(tmp_path), "--judge", "match", "--no-rag"],
    )

    assert result.exit_code == 0, result.output
    assert "OVERALL" in result.output
    assert "fidelity=0.500" in result.output


def test_eval_suite_list_run_and_results(patched_provider, tmp_path) -> None:  # noqa: ANN001
    examples_root = tmp_path / "examples"
    task_dir = examples_root / "tiny-task"
    evals_dir = task_dir / "evals"
    evals_dir.mkdir(parents=True)
    trace_path = task_dir / "traces.otel.jsonl"
    trace_path.write_text(
        Path(_traces_file(tmp_path)).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (evals_dir / "default.toml").write_text(
        "\n".join(
            [
                'description = "Tiny deterministic suite"',
                'files = ["../traces.otel.jsonl"]',
                'judge = "match"',
                "train_split = 0.5",
            ]
        ),
        encoding="utf-8",
    )

    listed = runner.invoke(app, ["eval", "list", "--examples-root", str(examples_root)])
    assert listed.exit_code == 0, listed.output
    assert "tiny-task/default" in listed.output

    results_root = tmp_path / ".wmh" / "evals"
    ran = runner.invoke(
        app,
        [
            "eval",
            "run",
            "tiny-task",
            "--examples-root",
            str(examples_root),
            "--results-root",
            str(results_root),
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert "wrote eval result" in ran.output
    result_files = list(results_root.glob("tiny-task/default/*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["suite"] == "tiny-task/default"
    assert payload["report"]["overall_fidelity"] == 0.5
    assert set(payload["report"]["per_file"]) == {"tiny-task"}

    summarized = runner.invoke(
        app,
        [
            "eval",
            "results",
            "tiny-task",
            "--examples-root",
            str(examples_root),
            "--results-root",
            str(results_root),
        ],
    )
    assert summarized.exit_code == 0, summarized.output
    assert "tiny-task/default" in summarized.output
    assert "0.500" in summarized.output


def test_build_then_list_shows_named_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "tau2-airline", tmp_path)

    # The artifact lands under <root>/models/<name>/.
    assert (root / "models" / "tau2-airline" / "config.toml").exists()

    listed = runner.invoke(app, ["list", "--root", str(root)])
    assert listed.exit_code == 0, listed.output
    assert "tau2-airline" in listed.output


def test_list_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["list", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 0
    assert "no world models" in result.output


def test_play_repl_steps_and_quits(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "default", tmp_path)

    # Feed one tool call then quit; the world model's canned observation should surface.
    result = runner.invoke(
        app,
        ["play", "--root", str(root), "--task", "look up users"],
        input='get_user {"id": "u1"}\n:quit\n',
    )
    assert result.exit_code == 0, result.output
    assert "user u1 found" in result.output


def test_build_interactive_wizard_creates_model(
    patched_provider,  # noqa: ANN001 - pytest fixture
    tmp_path,  # noqa: ANN001 - pytest fixture
    monkeypatch,  # noqa: ANN001 - pytest fixture
) -> None:
    root = tmp_path / ".wmh"
    for var in ("AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(var, "test-cred")  # creds present: no interactive key prompts
    # --interactive forces the wizard even under CliRunner (non-TTY); feed each answer line in
    # prompt order: name, file, provider (select), model (select), region (bedrock only),
    # judge model (select), budget, embedder (select). The offline 'hashing' embedder skips
    # the embed-model prompt; phi dim isn't prompted. Provider/model/embedder pick by index.
    answers = "\n".join(
        [
            "wizard-built",
            _traces_file(tmp_path),
            "3",  # provider: bedrock (order: openai, anthropic, bedrock, azure, ...)
            "1",  # model: us.anthropic.claude-opus-4-8
            "us-east-1",
            "",  # judge model: accept the bedrock default (dated haiku)
            "4",  # gepa budget
            "1",  # embedder: hashing
        ]
    )
    result = runner.invoke(
        app, ["build", "--interactive", "--root", str(root)], input=answers + "\n"
    )
    assert result.exit_code == 0, result.output
    assert (root / "models" / "wizard-built" / "config.toml").exists()


def test_build_non_interactive_without_source_errors(tmp_path) -> None:  # noqa: ANN001
    # No --file/--vendor and --no-interactive: should fail fast rather than hang on input.
    result = runner.invoke(app, ["build", "--no-interactive", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code != 0


def test_build_aborts_when_provider_sdk_missing(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A missing SDK must abort the build before any rollouts, with the `uv sync` extra hint.

    Regression: previously the ModuleNotFoundError was swallowed inside GEPA and the build
    "succeeded" with a useless held-out-0.0 model.
    """
    import sys

    from wmh.providers.base import VerifyResult

    appmod = sys.modules["wmh.cli.app"]
    monkeypatch.setattr(
        appmod,
        "verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=configs[0].kind,
                model=configs[0].model,
                detail="No module named 'boto3'",
            )
        ],
    )
    root = tmp_path / ".wmh"
    result = runner.invoke(
        app, ["build", "--name", "x", "--file", _traces_file(tmp_path), "--root", str(root)]
    )
    assert result.exit_code == 1
    assert "run `uv sync` to install the provider SDKs" in result.output
    # Aborted before building: no artifact written.
    assert not (root / "models" / "x" / "config.toml").exists()


def test_play_unknown_model_errors(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["play", "--name", "nope", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code != 0
    # A clean usage error, not an uncaught FileNotFoundError traceback.
    assert not isinstance(result.exception, FileNotFoundError)
    assert "nope" in result.output


def test_demo_unknown_model_is_clean_error(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["demo", "--name", "ghost", "--root", str(root)])
    assert result.exit_code != 0
    # Resolved through _load_model -> _resolve_name; must surface as a usage error, not a traceback.
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_providers_verify_unknown_model_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["providers", "verify", "--name", "ghost", "--root", str(tmp_path / ".wmh")]
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)


def test_providers_verify_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["providers", "verify", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 0
    assert "no world models built yet" in result.output


def test_providers_verify_reports_built_model_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])
    assert result.exit_code == 0, result.output
    # The bedrock provider configured at build time shows up in the verify report.
    assert "bedrock" in result.output
