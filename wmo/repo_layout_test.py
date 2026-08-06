"""Executable form of AGENTS.md rule 5: the repo's top level is an allowlist.

Runs against `git ls-files` so it checks what is TRACKED, not what happens to be on disk.
Skipped outside a git checkout (e.g. an installed sdist).
"""

from __future__ import annotations

import functools
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# AGENTS.md rule 5: tracked top-level directories must be within this set, and the set is CLOSED.
# An agent may never add to it. A new entry requires a human to name that exact directory and
# grant permission for the name; the entry then lands in the same change that documents it in
# AGENTS.md rule 5. If work does not fit a surface below, it goes under the closest one or stays
# out of the repo — never into a new sibling.
ALLOWED_TOP_DIRS = {
    "wmo",  # the flagship package: all importable code
    "docs",  # reviewed public documentation (see the docs/ layout tests below)
    "assets",  # media referenced by README/docs
    ".claude",  # checked-in agent skills
    ".github",  # CI workflows
}


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
        f"{sorted(ALLOWED_TOP_DIRS)}. The allowlist is closed and agents may not extend it: put "
        "reusable code in wmo/ (self-contained building blocks in wmo/common/), finished reports "
        "in docs/, and one-off or scratch work OUTSIDE the repo. Adding a new top-level "
        "directory requires a human to grant permission for that exact name."
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


def test_docs_layout_is_exactly_readme_research_reference() -> None:
    """docs/ is the manifest, the CLI map, writeups with their figures, references, and cookbooks.

    Anything else (other top-level pages, stray dirs, figures outside figures/) is clutter that
    rule 5 says gets relocated or deleted.
    """
    allowed = re.compile(
        r"^docs/(README\.md"
        r"|usage\.md"
        r"|research/[^/]+\.md"
        r"|research/figures/[^/]+\.png"
        r"|reference/[^/]+\.md"
        r"|cookbook/[^/]+\.md)$"
    )
    offenders = [p for p in _tracked_files() if p.startswith("docs/") and not allowed.match(p)]
    assert not offenders, (
        f"files outside the docs/ layout: {offenders}; writeups go in docs/research/*.md with "
        "figures in docs/research/figures/, references in docs/reference/*.md, end-to-end walks "
        "in docs/cookbook/*.md, and docs/usage.md is the only other root page (AGENTS.md rule 5)"
    )


# Top-level directories this repo used to have. They are gone; a doc that still points at one is
# sending the reader to a path that does not exist.
RETIRED_TOP_DIRS = (".agents/", "deploy/", "examples/", "packages/", "web/")

#: Each retired directory as a regex anchored at a path-token boundary. A bare substring test
#: would fail the gate on ordinary prose: `web/` matches inside
#: `https://api.search.brave.com/res/v1/web/search`, and `packages/` inside `site-packages/` or
#: any `files.pythonhosted.org/packages/...` wheel URL.
_RETIRED_PATTERNS = tuple(
    (retired, re.compile(rf"(?<![\w./-]){re.escape(retired)}")) for retired in RETIRED_TOP_DIRS
)


def test_docs_never_point_at_a_retired_directory() -> None:
    """docs/ are finished products: every path they quote must still exist.

    Reproduction lives in the report itself (public wmo API or CLI), never behind a path that
    was deleted, and never behind a scratch workspace, which this repo no longer has.
    """
    offenders: list[tuple[str, str]] = []
    for p in _tracked_files():
        if not (p.startswith("docs/") and p.endswith(".md")):
            continue
        path = REPO_ROOT / p
        if not path.is_file():  # tolerate uncommitted deletes/renames mid-edit
            continue
        text = path.read_text(encoding="utf-8")  # once per doc, not once per retired dir
        offenders.extend(
            (p, retired) for retired, pattern in _RETIRED_PATTERNS if pattern.search(text)
        )
    assert not offenders, (
        f"docs pointing at retired top-level directories: {offenders}; those paths no longer "
        "exist. Quote reproduction as public wmo API/CLI in the report itself (AGENTS.md rule 5)"
    )


def test_docs_readme_indexes_every_doc() -> None:
    """docs/README.md's justification table must name every tracked docs/ file (rule 5).

    The manifest is what makes the justification rule enforceable; a doc or figure absent from
    it is either unjustified or the table has drifted.
    """
    readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    missing = [
        p
        for p in _tracked_files()
        if p.startswith("docs/") and p != "docs/README.md" and p.removeprefix("docs/") not in readme
    ]
    assert not missing, (
        f"docs files absent from docs/README.md's justification table: {missing}; every doc "
        "and figure gets a row or gets deleted (AGENTS.md rule 5)"
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


def test_there_is_no_uv_workspace() -> None:
    """One distribution, no members (AGENTS.md § One package).

    The workspace was retired when `packages/` was deleted: `environment-capture` resolves from
    PyPI and `llm-waterfall` was vendored into `wmo/common/vendor/waterfall/`. Reintroducing a
    member means reintroducing a top-level `packages/` directory, which rule 5 forbids outright.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        root = tomllib.load(fh)
    uv_config = root.get("tool", {}).get("uv", {})
    assert "workspace" not in uv_config, (
        "[tool.uv.workspace] is back; this repo publishes one distribution whose importable code "
        "is all of wmo/. Depend on PyPI or vendor under wmo/common/vendor/ "
        "(AGENTS.md § One package)"
    )
    assert "sources" not in uv_config, (
        "[tool.uv.sources] is back; with no workspace every dependency resolves from PyPI "
        "(AGENTS.md § One package)"
    )


def test_root_gate_covers_the_whole_package() -> None:
    """AGENTS.md § One package promises one root gate over the single package."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        root = tomllib.load(fh)
    testpaths = root["tool"]["pytest"]["ini_options"]["testpaths"]
    assert testpaths == ["wmo"], (
        f"testpaths is {testpaths}, not ['wmo']; the root gate covers the one package and every "
        "test is inline beside the module it covers (AGENTS.md § One package)"
    )


ALLOWED_TOP_FILES = {
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",  # not yet present; allowlisted so adding one never fights the gate
    "README.md",
    "conftest.py",
    "justfile",
    "pyproject.toml",
    "uv.lock",
}


def test_top_level_files_are_allowlisted() -> None:
    """Root files are an allowlist too — no Makefile/tox.ini/setup.cfg sprawl (rule 5)."""
    tracked_root_files = {p for p in _tracked_files() if "/" not in p}
    unexpected = tracked_root_files - ALLOWED_TOP_FILES
    assert not unexpected, (
        f"top-level files {sorted(unexpected)} are not allowlisted; config belongs in "
        "pyproject.toml, tasks in the justfile, and everything else under an allowlisted dir"
    )


def test_no_finder_duplicate_files_are_tracked() -> None:
    """macOS Finder copies ("foo 2.py") dodge pytest collection and imports, so they rot
    silently; 24 of them once shipped in a PR before anyone noticed."""
    duplicates = [p for p in _tracked_files() if re.search(r" \d+\.\w+$", p)]
    assert not duplicates, (
        f"tracked Finder-style duplicate files {sorted(duplicates)}; delete the copies "
        "(they are never imported or collected) and keep the originals"
    )
