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

RETIRED_TOP_DIRS = (".agents/", "deploy/", "examples/", "packages/", "web/")
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


# --- Test pairing (AGENTS.md rule 2) ---

# A module needs no sibling suite when it has no behavior of its own to cover. `__init__.py` is a
# re-export surface (what it promises is asserted from the suite of whichever module the package is
# mostly about), `__main__.py` is the `python -m` shim over an already-tested entry point, and
# `conftest.py` is pytest wiring that every suite exercises by running at all.
UNTESTED_MODULE_NAMES = {"__init__.py", "__main__.py", "conftest.py"}

# The CLOSED set of suites that cover something other than one sibling module. Every entry is a
# repo-wide guardrail whose subject is this repository, so none of them has a module to pair with by
# construction. Everything else lives beside what it covers, including the seam suites that used to
# sit here: a cross-backend contract is a section of each backend's own suite, a live end-to-end
# path a section of the pipeline's, a package's public surface a section of the suite for the module
# the package is about. An agent may never extend this list: a new test file either sits beside the
# module it covers or merges into that module's suite. Adding an entry, and with it the argument for
# why nothing beside a module could hold the test, requires a human to grant that exact path in the
# same change that documents it in AGENTS.md rule 2.
CROSS_CUTTING_TESTS = {
    "wmo/repo_structure_test.py": "the repo's own layout and test pairing: this file",
    "wmo/repo_layout_test.py": "the migration guardrails (file size, inventories)",
    "wmo/repo_docstrings_test.py": "public docstring coverage across the repo",
    "wmo/repo_import_boundaries_test.py": "the import direction between product domains",
}

# The vendored trees, listed rather than matched on the name `vendor`: they are verbatim upstream
# copies, so their own `test/` layout is theirs and not ours, and that exemption must not extend to
# a product package that merely happens to be called `vendor`. A new vendored tree is added here in
# the change that vendors it.
VENDORED_TREES = ("wmo/common/vendor/", "wmo/runtime/harness/vendor/")


def _tracked_python_files() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every tracked `.py` path, split into (production modules, test suites)."""
    paths = [path for path in _tracked_files() if path.endswith(".py")]
    return (
        tuple(path for path in paths if not path.endswith("_test.py")),
        tuple(path for path in paths if path.endswith("_test.py")),
    )


def test_every_module_has_a_sibling_test() -> None:
    """AGENTS.md rule 2: `foo.py` is covered by `foo_test.py` in the same directory."""
    modules, tests = _tracked_python_files()
    suites = set(tests)
    missing = sorted(
        path
        for path in modules
        if Path(path).name not in UNTESTED_MODULE_NAMES
        and path.removesuffix(".py") + "_test.py" not in suites
    )
    assert not missing, (
        f"modules with no sibling test: {missing}; add `<module>_test.py` beside each one "
        f"(AGENTS.md rule 2). Only {sorted(UNTESTED_MODULE_NAMES)} are exempt, and coverage "
        "living in some other file does not count: the pairing is what makes it findable"
    )


def test_every_test_covers_the_module_beside_it() -> None:
    """AGENTS.md rule 2, the other direction: no test file without the module it names.

    A `foo_test.py` whose `foo.py` was renamed or deleted keeps passing while covering nothing,
    which is worse than no test at all: the suite reports green over code that is gone.
    """
    modules, tests = _tracked_python_files()
    known = set(modules)
    orphans = sorted(
        path
        for path in tests
        if path not in CROSS_CUTTING_TESTS and path.removesuffix("_test.py") + ".py" not in known
    )
    assert not orphans, (
        f"test files with no module beside them: {orphans}; rename them to match the module they "
        "cover, fold them into the sibling suite for that module, or (for a genuinely "
        "cross-cutting suite) have a human add the exact path to CROSS_CUTTING_TESTS in "
        "wmo/repo_structure_test.py in the same change that documents it in AGENTS.md rule 2"
    )


def test_the_cross_cutting_allowlist_has_no_stale_entries() -> None:
    """An allowlisted path that no longer exists would silently license a future file's name."""
    _modules, tests = _tracked_python_files()
    stale = sorted(set(CROSS_CUTTING_TESTS) - set(tests))
    assert not stale, (
        f"CROSS_CUTTING_TESTS names paths that are not tracked test files: {stale}; drop the "
        "entries so the exception list keeps describing what actually exists"
    )


def test_there_is_no_tests_directory() -> None:
    """AGENTS.md rule 2: tests are inline, so no directory anywhere collects them.

    Every tracked path counts, not just `.py` files: a fixture, snapshot, or README under
    `tests/` is the beginning of the directory this rule forbids, and the suite that reads it
    follows. Vendored trees are exempt: they are verbatim upstream copies whose own `test/`
    layout is theirs, not ours.
    """
    offenders = sorted(
        path
        for path in _tracked_files()
        if re.search(r"(^|/)tests?/", path) and not path.startswith(VENDORED_TREES)
    )
    assert not offenders, (
        f"tracked files under a tests/ directory: {offenders}; every suite lives beside the "
        "module it covers (AGENTS.md rule 2)"
    )


# --- The package tree (AGENTS.md rule 4) ---

#: The package tree, as one table. Every domain names the subpackages it owns, so a responsibility
#: that quietly moves out of its domain (the `wmo/evals/harbor` and `wmo/optimize/harness` shapes
#: this repo has already had and retired) fails here instead of in a reviewer's memory. A domain may
#: grow a new subpackage without editing this table; only `wmo` itself is a closed set, because a
#: package directly under it is a new product domain.
PACKAGE_TREE = {
    "wmo": {"cli", "common", "optimize", "runtime", "simulation"},
    "wmo/common": {"config", "core", "observability", "providers", "vendor"},
    "wmo/optimize": {"routing", "model", "research"},
    "wmo/runtime": {"agents", "evaluation", "harness", "platform", "runs"},
    "wmo/simulation": {
        "context",
        "evaluation",
        "ingest",
        "model",
        "retrieval",
        "scenarios",
        "serving",
    },
}

#: Modules that must stay at a domain's root: each is the domain's own contract, not a detail of
#: one of its subpackages, so moving it down would hide the seam the domain is entered through.
DOMAIN_ROOT_MODULES = {
    "wmo/runtime": {"environment.py", "episode.py"},
    "wmo/simulation": {"environment.py", "hub.py"},
}


def _subpackages(package: str) -> set[str]:
    """The directory names directly under a package, ignoring bytecode caches."""
    return {
        path.name
        for path in (REPO_ROOT / package).iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }


def test_every_domain_still_owns_the_subpackages_it_is_named_for() -> None:
    """AGENTS.md rule 4: a responsibility never quietly leaves the domain that owns it."""
    missing = {
        package: sorted(expected - _subpackages(package))
        for package, expected in PACKAGE_TREE.items()
        if expected - _subpackages(package)
    }
    assert not missing, (
        f"subpackages missing from the domain that owns them: {missing}; the tree in AGENTS.md "
        "rule 4 changes with the rule 4 text, in the same change"
    )


def test_the_wmo_root_is_the_closed_set_of_product_domains() -> None:
    """A directory directly under `wmo/` is a new product domain, so the set is closed (rule 4)."""
    unexpected = _subpackages("wmo") - PACKAGE_TREE["wmo"]
    assert not unexpected, (
        f"new packages directly under wmo/: {sorted(unexpected)}; nest the work under the domain "
        "that owns its concern (rule 4's four domains plus the CLI), and add a fifth domain only "
        "with the rule 4 text in the same change"
    )


def test_each_domain_keeps_its_own_entry_modules_at_its_root() -> None:
    """AGENTS.md rule 4: a domain's own contract stays at its root, not inside a subpackage."""
    missing = {
        package: sorted(name for name in expected if not (REPO_ROOT / package / name).is_file())
        for package, expected in DOMAIN_ROOT_MODULES.items()
    }
    offenders = {package: names for package, names in missing.items() if names}
    assert not offenders, f"domain entry modules missing from their package root: {offenders}"


def test_no_production_module_sits_in_the_flat_wmo_namespace() -> None:
    """AGENTS.md rule 4: `wmo/` holds the domains, not modules of its own.

    Only the package shims live here: `__init__.py`, the `python -m` entry point, pytest wiring,
    and the repo-wide guardrail suites.
    """
    flat = sorted(
        path.name
        for path in (REPO_ROOT / "wmo").glob("*.py")
        if path.name not in UNTESTED_MODULE_NAMES and not path.name.endswith("_test.py")
    )
    assert not flat, (
        f"production modules returned to the flat wmo namespace: {flat}; move each under the "
        "domain that owns it (AGENTS.md rule 4)"
    )
