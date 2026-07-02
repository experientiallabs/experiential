"""Executable form of AGENTS.md rule 5: the repo's top level is an allowlist.

Runs against `git ls-files` so it checks what is TRACKED, not what happens to be on disk.
Skipped outside a git checkout (e.g. an installed sdist).
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# AGENTS.md rule 5: tracked top-level directories must be within this set. The allowlist may
# exceed the current tree (web/ and .github/ are decided-but-not-yet-landed surfaces): it bounds
# what MAY exist, it does not require existence.
ALLOWED_TOP_DIRS = {"wmh", "examples", "docs", "assets", "web", ".agents", ".claude", ".github"}


@functools.lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    """Every git-tracked path in the repo (one `git ls-files`, cached across the tests)."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; repo-layout rules only apply to a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; repo-layout rules only apply to the repository")
    return tuple(result.stdout.splitlines())


def test_top_level_directories_are_allowlisted() -> None:
    """Every tracked top-level directory is on the AGENTS.md rule 5 allowlist."""
    tracked_dirs = {path.split("/", 1)[0] for path in _tracked_files() if "/" in path}
    unexpected = tracked_dirs - ALLOWED_TOP_DIRS
    assert not unexpected, (
        f"top-level directories {sorted(unexpected)} are not in the AGENTS.md rule 5 allowlist "
        f"{sorted(ALLOWED_TOP_DIRS)}; put one-off work in .agents/, dataset tooling in "
        "examples/<task>/, reusable code in wmh/, finished reports in docs/"
    )


def test_no_local_settings_files_are_tracked() -> None:
    """No generated settings.toml (telemetry ids) is ever committed."""
    offenders = [p for p in _tracked_files() if Path(p).name == "settings.toml"]
    assert not offenders, (
        f"local settings files are tracked: {offenders}; these are generated per-root artifacts "
        "(telemetry ids) and must stay gitignored"
    )


def test_no_bytecode_or_caches_are_tracked() -> None:
    """No __pycache__/.pyc artifacts are committed."""
    offenders = [p for p in _tracked_files() if "__pycache__" in p or p.endswith(".pyc")]
    assert not offenders, (
        f"bytecode/cache files are tracked: {offenders[:5]}; git rm --cached them and keep "
        "__pycache__/ in .gitignore"
    )


def test_docs_holds_no_data_files() -> None:
    """docs/ is writeups + rendered figures only; raw results/data live in .agents/docs/."""
    allowed_suffixes = (".md", ".png")
    offenders = [
        p for p in _tracked_files() if p.startswith("docs/") and not p.endswith(allowed_suffixes)
    ]
    assert not offenders, (
        f"non-writeup files under docs/: {offenders}; move raw results/vector sources to "
        ".agents/docs/research/ (AGENTS.md rule 5 keeps docs/ deliberately small)"
    )


def test_no_tracked_file_is_matched_by_ignore_rules() -> None:
    """A tracked file matched by a .gitignore rule is a conflict waiting to bite (re-adds fail)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-i", "-c", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; repo-layout rules only apply to a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; repo-layout rules only apply to the repository")
    offenders = result.stdout.splitlines()
    assert not offenders, (
        f"tracked files matched by ignore rules: {offenders[:5]}; fix the .gitignore pattern "
        "(add a ! negation or narrow the glob) so tracked artifacts stay re-addable"
    )
