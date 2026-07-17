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

# AGENTS.md rule 5: tracked top-level directories must be within this set. The allowlist may
# exceed the current tree (web/ and .github/ are decided-but-not-yet-landed surfaces): it bounds
# what MAY exist, it does not require existence.
ALLOWED_TOP_DIRS = {
    "wmh",
    "examples",
    "docs",
    "assets",
    "web",
    ".agents",
    ".claude",
    ".github",
    "packages",  # monorepo workspace members live here (AGENTS.md § Monorepo)
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


def test_docs_layout_is_exactly_readme_research_reference() -> None:
    """docs/ is the manifest, writeups with their rendered figures, and how-to references.

    Anything else (top-level pages, stray dirs, figures outside figures/) is clutter that rule 5
    says gets relocated or deleted.
    """
    allowed = re.compile(
        r"^docs/(README\.md"
        r"|research/[^/]+\.md"
        r"|research/figures/[^/]+\.png"
        r"|reference/[^/]+\.md)$"
    )
    offenders = [p for p in _tracked_files() if p.startswith("docs/") and not allowed.match(p)]
    assert not offenders, (
        f"files outside the docs/ layout: {offenders}; writeups go in docs/research/*.md with "
        "figures in docs/research/figures/, references in docs/reference/*.md (AGENTS.md rule 5)"
    )


def test_docs_never_mention_the_agents_workspace() -> None:
    """docs/ are finished products: the agents' workspace must be invisible from them.

    Not even disclaimed pointers: a reader of docs/ should never learn the workspace exists.
    Reproduction lives in the report itself (public wmh API or CLI), never behind a workspace
    path.
    """
    offenders = [
        p
        for p in _tracked_files()
        if p.startswith("docs/")
        and p.endswith(".md")
        and (REPO_ROOT / p).is_file()  # tolerate uncommitted deletes/renames mid-edit
        and ".agents" in (REPO_ROOT / p).read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"docs mentioning the agents workspace: {offenders}; quote reproduction as public "
        "wmh API/CLI in the report itself and drop every workspace path (AGENTS.md rule 5)"
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


def _workspace_member_dirs() -> list[Path]:
    """Existing member directories from [tool.uv.workspace].members (glob-aware)."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        root = tomllib.load(fh)
    members = root.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    assert members, "the workspace must declare its members (AGENTS.md § Monorepo)"
    dirs: list[Path] = []
    for member in members:
        dirs.extend(p for p in REPO_ROOT.glob(member) if p.is_dir())
    return dirs


def test_workspace_members_are_real_packages() -> None:
    """Every existing [tool.uv.workspace] member dir must carry its own pyproject.toml."""
    for member_dir in _workspace_member_dirs():
        assert (member_dir / "pyproject.toml").is_file(), (
            f"workspace member {member_dir.name!r} has no pyproject.toml; every member is an "
            "independently packaged, publishable unit (AGENTS.md § Monorepo)"
        )


def test_root_gate_covers_every_python_member() -> None:
    """AGENTS.md § Monorepo promises one root gate: testpaths must include every member."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        root = tomllib.load(fh)
    testpaths = set(root["tool"]["pytest"]["ini_options"]["testpaths"])

    def covered(member: Path) -> bool:
        rel = member.relative_to(REPO_ROOT)
        return any(rel == Path(tp) or Path(tp) in rel.parents for tp in testpaths)

    missing = [str(d.relative_to(REPO_ROOT)) for d in _workspace_member_dirs() if not covered(d)]
    assert not missing, (
        f"workspace members {missing} are not in [tool.pytest.ini_options].testpaths; the root "
        "gate must cover every Python member (AGENTS.md § Monorepo)"
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
