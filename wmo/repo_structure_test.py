"""Executable repository structure guardrails.

Runs against `git ls-files` so it checks tracked paths rather than incidental local files.
Skipped outside a Git checkout, such as an installed sdist.
"""

from __future__ import annotations

import functools
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_TOP_DIRS = {
    "wmo",
    "docs",
    "assets",
    "web",
    ".claude",
    ".github",
}

ALLOWED_TOP_FILES = {
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "conftest.py",
    "justfile",
    "pyproject.toml",
    "uv.lock",
}

RETIRED_TOP_DIRS = (".agents/", "deploy/", "examples/", "packages/")
_RETIRED_PATTERNS = tuple(
    (retired, re.compile(rf"(?<![\w./-]){re.escape(retired)}")) for retired in RETIRED_TOP_DIRS
)


@functools.lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    """Return every Git-tracked repository path."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; repository-structure rules require a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; repository-structure rules require the repository")
    return tuple(result.stdout.splitlines())


def test_top_level_directories_are_allowlisted() -> None:
    """Every tracked top-level directory is on the closed AGENTS.md allowlist."""
    tracked_dirs = {path.split("/", 1)[0] for path in _tracked_files() if "/" in path}
    unexpected = tracked_dirs - ALLOWED_TOP_DIRS
    assert not unexpected, (
        f"top-level directories {sorted(unexpected)} are not on the allowlist "
        f"{sorted(ALLOWED_TOP_DIRS)}. A new directory requires a human to authorize its exact "
        "name and update AGENTS.md in the same change."
    )


def test_top_level_files_are_allowlisted() -> None:
    """Root files remain an allowlist rather than a collection of ad hoc tooling."""
    tracked_root_files = {path for path in _tracked_files() if "/" not in path}
    unexpected = tracked_root_files - ALLOWED_TOP_FILES
    assert not unexpected, (
        f"top-level files {sorted(unexpected)} are not allowlisted; configuration belongs in "
        "pyproject.toml and repository code belongs under wmo/."
    )


def test_no_local_settings_files_are_tracked() -> None:
    """Local settings.toml files remain generated, ignored artifacts."""
    offenders = [path for path in _tracked_files() if Path(path).name == "settings.toml"]
    assert not offenders, f"local settings files are tracked: {offenders}"


def test_no_bytecode_or_caches_are_tracked() -> None:
    """Bytecode and cache artifacts are never committed."""
    offenders = [
        path for path in _tracked_files() if "__pycache__" in path or path.endswith(".pyc")
    ]
    assert not offenders, f"bytecode or cache files are tracked: {offenders[:5]}"


def test_docs_layout_is_exactly_readme_research_reference() -> None:
    """Documentation stays inside the reviewed layout named by AGENTS.md."""
    allowed = re.compile(
        r"^docs/(README\.md"
        r"|usage\.md"
        r"|research/[^/]+\.md"
        r"|research/figures/[^/]+\.png"
        r"|reference/[^/]+\.md"
        r"|cookbook/[^/]+\.md)$"
    )
    offenders = [
        path for path in _tracked_files() if path.startswith("docs/") and not allowed.match(path)
    ]
    assert not offenders, f"files outside the approved docs layout: {offenders}"


def test_docs_never_point_at_a_retired_directory() -> None:
    """Finished documentation does not direct readers to deleted repository surfaces."""
    offenders: list[tuple[str, str]] = []
    for relative_path in _tracked_files():
        if not (relative_path.startswith("docs/") and relative_path.endswith(".md")):
            continue
        path = REPO_ROOT / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            (relative_path, retired)
            for retired, pattern in _RETIRED_PATTERNS
            if pattern.search(text)
        )
    assert not offenders, f"docs point at retired top-level directories: {offenders}"


def test_docs_readme_indexes_every_doc() -> None:
    """The documentation manifest names every tracked documentation artifact."""
    readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    missing = [
        path
        for path in _tracked_files()
        if path.startswith("docs/")
        and path != "docs/README.md"
        and path.removeprefix("docs/") not in readme
    ]
    assert not missing, f"docs files absent from docs/README.md: {missing}"


def test_no_tracked_file_is_matched_by_ignore_rules() -> None:
    """Tracked files are not hidden by the repository's own ignore patterns."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-i", "-c", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; repository-structure rules require a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; repository-structure rules require the repository")
    assert not result.stdout.splitlines(), "tracked files are matched by ignore rules"


def test_there_is_no_uv_workspace() -> None:
    """The flagship distribution has no retired uv workspace or member sources."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as file_handle:
        root = tomllib.load(file_handle)
    uv_config = root.get("tool", {}).get("uv", {})
    assert "workspace" not in uv_config
    assert "sources" not in uv_config


def test_root_gate_covers_the_whole_package() -> None:
    """Pytest runs the one inline test suite rooted at wmo/."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as file_handle:
        root = tomllib.load(file_handle)
    assert root["tool"]["pytest"]["ini_options"]["testpaths"] == ["wmo"]


def test_no_finder_duplicate_files_are_tracked() -> None:
    """Finder-style numbered copies do not bypass imports and test discovery."""
    duplicates = [path for path in _tracked_files() if re.search(r" \d+\.\w+$", path)]
    assert not duplicates, f"tracked Finder-style duplicate files: {sorted(duplicates)}"
