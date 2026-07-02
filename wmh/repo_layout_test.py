"""Executable form of AGENTS.md rule 5: the repo's top level is an allowlist.

Runs against `git ls-files` so it checks what is TRACKED, not what happens to be on disk.
Skipped outside a git checkout (e.g. an installed sdist).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# AGENTS.md rule 5: tracked top-level directories are exactly these.
ALLOWED_TOP_DIRS = {"wmh", "examples", "docs", "assets", "web", ".agents", ".claude", ".github"}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout; repo-layout rules only apply to the repository")
    return result.stdout.splitlines()


def test_top_level_directories_are_allowlisted() -> None:
    tracked_dirs = {path.split("/", 1)[0] for path in _tracked_files() if "/" in path}
    unexpected = tracked_dirs - ALLOWED_TOP_DIRS
    assert not unexpected, (
        f"top-level directories {sorted(unexpected)} are not in the AGENTS.md rule 5 allowlist "
        f"{sorted(ALLOWED_TOP_DIRS)}; put one-off work in .agents/, dataset tooling in "
        "examples/<task>/, reusable code in wmh/, finished reports in docs/"
    )


def test_no_local_settings_files_are_tracked() -> None:
    offenders = [p for p in _tracked_files() if Path(p).name == "settings.toml"]
    assert not offenders, (
        f"local settings files are tracked: {offenders}; these are generated per-root artifacts "
        "(telemetry ids) and must stay gitignored"
    )


def test_no_bytecode_or_caches_are_tracked() -> None:
    offenders = [p for p in _tracked_files() if "__pycache__" in p or p.endswith(".pyc")]
    assert not offenders, f"bytecode/cache files are tracked: {offenders[:5]}"


def test_committed_results_only_under_docs_experiment_results() -> None:
    """Generated result JSONs may only be committed as figure-backing summaries in docs/."""
    offenders = [
        p
        for p in _tracked_files()
        if p.startswith("docs/") and p.endswith(".json") and "_results/" not in p
    ]
    assert not offenders, (
        f"result JSONs outside docs/<experiment>_results/: {offenders}; AGENTS.md rule 5 blesses "
        "only small summary JSONs that back a published figure"
    )
