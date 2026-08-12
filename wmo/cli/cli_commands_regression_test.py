"""Build, knowledge, and root CLI command tests."""

import importlib

# ruff: noqa: F403, F405
from wmo.cli.cli_fixtures_test import *


def test_cli_exposes_the_small_command_set() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    core = {
        "build",
        "list",
        "eval",
        "download",
        "knowledge",
    }
    assert names == core
    # `optimize` is a GROUP (route and persisted-dataset model SFT; harness search moved out).
    groups = {group.name for group in app.registered_groups}
    assert "optimize" in groups


def test_main_keeps_environment_loading_before_cli_dispatch(monkeypatch) -> None:  # noqa: ANN001
    """Preserve the explicit environment-loading boundary without reading a real env file."""
    app_module = importlib.import_module("wmo.cli.app")
    calls: list[str] = []
    monkeypatch.setattr(app_module, "load_env_file", lambda: calls.append("load_env_file"))
    monkeypatch.setattr(app_module, "_quiet_http_logs", lambda: calls.append("quiet_logs"))
    monkeypatch.setattr(app_module, "app", lambda: calls.append("dispatch"))

    app_module.main()

    assert calls == ["load_env_file", "quiet_logs", "dispatch"]


def test_knowledge_command_prints_path_and_files(tmp_path) -> None:  # noqa: ANN001 - fixture
    from wmo.common.config import save_config
    from wmo.common.config.config import HarnessConfig
    from wmo.simulation.model.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(), root=model_dir)
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "knowledge" in result.output  # the folder path (the real editing surface)
    assert "rules.md" in result.output
    assert "gate: auth required" in result.output


def test_knowledge_command_prints_bracketed_markdown_verbatim(tmp_path) -> None:  # noqa: ANN001
    """Knowledge is hand-edited markdown, so rich must not read its brackets as style tags.

    Unescaped, `[/items]` raised MarkupError (the command died on ordinary content) and both
    `list[str]` and the link text were silently deleted from the rendered output.
    """
    from wmo.common.config import save_config
    from wmo.common.config.config import HarnessConfig
    from wmo.simulation.model.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(), root=model_dir)
    KnowledgeBase(model_dir / "knowledge").write_file(
        "schemas.md",
        "Use the XML close marker [/items] to end a list.\n"
        "reservations: list[str]\n"
        "See [the docs](https://example.com) for details.",
    )

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.exception
    assert "[/items]" in result.output
    assert "list[str]" in result.output
    assert "[the docs](https://example.com)" in result.output


def test_knowledge_command_without_kb_says_how_to_enable(tmp_path) -> None:  # noqa: ANN001
    # The empty state used to print a directory that does not exist and say "drop *.md files in
    # this folder" without naming the flag that seeds one, so this now pins both halves.
    from wmo.common.config import save_config
    from wmo.common.config.config import HarnessConfig

    root = tmp_path / ".wmo"
    save_config(HarnessConfig(), root=root / "models" / "airline")
    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.output
    flat = _squashed(result.output)
    assert _squashed("does not exist yet") in flat  # the printed dir is absent, and it says so
    assert "--knowledge" in flat  # the exact build flag that creates one


def test_knowledge_command_flags_a_kb_the_model_ignores(tmp_path) -> None:  # noqa: ANN001
    """Files under `knowledge/` are inert unless the model was built with `--knowledge`."""
    from wmo.common.config import save_config
    from wmo.common.config.config import HarnessConfig
    from wmo.simulation.model.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(), root=model_dir)  # knowledge=False, the build default
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])

    assert result.exit_code == 0, result.output
    flat = _squashed(result.output)
    assert "inert" in flat
    assert "--knowledge" in flat  # names the flag that activates them
    assert _squashed("gate: auth required") in flat  # the files are still shown


def test_knowledge_command_stays_quiet_when_the_kb_is_live(tmp_path) -> None:  # noqa: ANN001
    from wmo.common.config import save_config
    from wmo.common.config.config import HarnessConfig
    from wmo.simulation.model.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(knowledge=True), root=model_dir)
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "inert" not in result.output


def test_knowledge_resolves_a_shipped_example(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """`wmo knowledge` resolves a shipped example model when it is available."""
    from wmo.common.config import save_config
    from wmo.common.config.config import HarnessConfig
    from wmo.simulation.model.knowledge import KnowledgeBase

    example = tmp_path / "airline-bench"
    model_dir = example / "models" / "airline"
    save_config(HarnessConfig(knowledge=True), root=model_dir)
    (example / "traces.otel.jsonl").write_text("", encoding="utf-8")
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(catalog_module, "_benchmark_roots", lambda: (tmp_path,))
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["knowledge", "--name", "airline"])

    assert result.exit_code == 0, result.output
    assert "gate: auth required" in result.output


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["build", "--help"], "Canonical source format: otlp"),
        (["eval", "--help"], "`[models.agent]` selects a distinct agent provider"),
        (["providers", "set", "--help"], "settings.toml` as `[models.worker]`"),
        (["providers", "verify", "--help"], "the `[models.<role>]` roles in"),
    ],
)
def test_help_keeps_the_bracketed_pointer_it_exists_to_teach(
    argv: list[str], expected: str
) -> None:
    """Typer renders help through rich markup, which swallows an unescaped `[...]` whole.

    Each is the only pointer in that help text to a concrete user-facing setting, so a swallowed
    pair is silent misinformation.
    """
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    rendered = " ".join(result.output.replace("│", " ").split())
    assert expected in rendered


@pytest.mark.parametrize("args", [[], ["providers"], ["config"]])
def test_bare_invocation_shows_help(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert "Missing command" not in result.output
    assert "Usage:" in result.output
    assert "--help" in result.output
    # Bare invocation keeps the usage-error exit code (click >=8.2), unlike explicit --help
    # which exits 0 - scripts can still tell "asked for help" from "forgot the command".
    assert result.exit_code == 2


def test_examples_discovery_skips_unresolvable_names(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # A downloaded dir whose name validate_name rejects can never be named on a command line, so
    # discovery must drop it rather than offer it as a model candidate or an "available:" hint.
    for dirname in ("good-example", "tau bench"):
        example = tmp_path / dirname
        example.mkdir()
        (example / "traces.otel.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(catalog_module, "_benchmark_roots", lambda: (tmp_path,))

    found = [path.name for path in catalog_module._discover_examples()]

    assert found == ["good-example"]
