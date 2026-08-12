"""Small permanent checks for the current repository and CLI shape."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_HAND_AUTHORED_LINES = 999
HAND_AUTHORED_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
ALLOWED_TOP_DIRS = {".claude", ".github", "docs", "web", "wmo"}
ALLOWED_TOP_FILES = {
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "conftest.py",
    "justfile",
    "pyproject.toml",
    "uv.lock",
}


def _tracked_files() -> tuple[str, ...]:
    """Return Git-tracked paths or skip outside a source checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git is required for repository-layout checks")
    if result.returncode != 0:
        pytest.skip("repository-layout checks require a Git checkout")
    return tuple(result.stdout.splitlines())


def _physical_lines(path: Path) -> int:
    """Return physical UTF-8 line count without inventing a final blank line."""
    text = path.read_text(encoding="utf-8")
    return text.count("\n") + int(bool(text) and not text.endswith("\n"))


def test_hand_authored_files_stay_below_one_thousand_lines() -> None:
    """Every tracked hand-authored text file contains at most 999 physical lines."""
    oversized: list[tuple[str, int]] = []
    for relative_path in _tracked_files():
        path = REPO_ROOT / relative_path
        if (
            not path.is_file()
            or path.suffix.lower() not in HAND_AUTHORED_SUFFIXES
            or path.name == "package-lock.json"
        ):
            continue
        lines = _physical_lines(path)
        if lines > MAX_HAND_AUTHORED_LINES:
            oversized.append((relative_path, lines))
    assert not oversized, f"hand-authored files exceed 999 physical lines: {oversized}"


def test_top_level_paths_are_allowlisted() -> None:
    """Tracked root directories and files remain on the documented closed allowlist."""
    tracked = _tracked_files()
    actual_dirs = {path.split("/", 1)[0] for path in tracked if "/" in path}
    actual_files = {path for path in tracked if "/" not in path}
    assert actual_dirs == ALLOWED_TOP_DIRS
    assert actual_files == ALLOWED_TOP_FILES


def test_no_local_state_or_cache_is_tracked() -> None:
    """Generated settings, caches, bytecode, and local environment files stay untracked."""
    offenders = [
        path
        for path in _tracked_files()
        if Path(path).name in {".env", "settings.toml"}
        or "__pycache__" in Path(path).parts
        or path.endswith(".pyc")
    ]
    assert not offenders, f"local state or cache files are tracked: {offenders}"


def test_root_cli_and_subgroups_are_exact() -> None:
    """The public CLI remains build, config, optimize, and run with two exact subgroups."""
    from typer import Context
    from typer.core import TyperGroup
    from typer.main import get_group

    from wmo.cli.app import app

    root = get_group(app)
    root_context = Context(root)
    assert set(root.list_commands(root_context)) == {"build", "config", "optimize", "run"}

    expected_subcommands = {"config": {"telemetry"}, "optimize": {"model", "router"}}
    for name, expected in expected_subcommands.items():
        command = root.get_command(root_context, name)
        assert isinstance(command, TyperGroup)
        context = Context(command, parent=root_context, info_name=name)
        assert set(command.list_commands(context)) == expected
