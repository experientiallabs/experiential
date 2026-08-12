"""Executable form of AGENTS.md rules 2, 4, and 5: the layout of this repo, in one file.

Test pairing, the package tree and its import direction, and the top level as an allowlist. Most
checks run against `git ls-files`, so they check what is TRACKED rather than what happens to be on
disk, and are skipped outside a git checkout (e.g. an installed sdist).

The per-package `package_layout_test.py` suites used to live one per domain. They are here now:
five files asserted one property (the tree AGENTS.md rule 4 describes is the tree on disk), none of
them paired with a module, and splitting them per package made each boundary BETWEEN two domains
the property of whichever one happened to hold the assertion.
"""

from __future__ import annotations

import ast
import functools
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WMO_DIR = Path(__file__).resolve().parent

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


# AGENTS.md rule 2: a module needs no sibling suite when it has no behavior of its own to cover.
# `__init__.py` is a re-export surface (its package tests it through `api_test.py`), `__main__.py`
# is the `python -m` shim over an already-tested entry point, and `conftest.py` is pytest wiring
# that every suite exercises by running at all.
UNTESTED_MODULE_NAMES = {"__init__.py", "__main__.py", "conftest.py"}

# AGENTS.md rule 2: the CLOSED set of suites that cover something other than one sibling module,
# each mapped to what it does cover. One more is patterned rather than listed (a package's
# `api_test.py` covers its `__init__.py` re-export surface). An agent may never extend this list: a
# new test file either sits beside the module it covers, or it merges into the cross-cutting suite
# that already owns its concern. Adding an entry requires a human to grant it for that exact path.
CROSS_CUTTING_TESTS = {
    "wmo/repo_layout_test.py": "the repo layout itself: this file",
    "wmo/cli/startup_test.py": "the CLI's import graph: `wmo --help` must stay off heavy modules",
    "wmo/common/providers/streaming_test.py": (
        "one streaming contract held across every backend at once"
    ),
    "wmo/common/vendor/waterfall/import_hygiene_test.py": (
        "the vendored package importing and constructing with zero provider SDKs installed"
    ),
    "wmo/simulation/model/integration_test.py": (
        "convert -> build -> load -> step against a real Bedrock model (live, env-gated)"
    ),
    "wmo/simulation/serving/chat_openai_client_test.py": (
        "wire compatibility proven with the real `openai` SDK against a booted server"
    ),
}

#: The one suite name that pairs with a package rather than a module: a package's `api_test.py`
#: covers the `__init__.py` re-export surface, which is the package's public API.
_PACKAGE_SUITES = {"api_test.py": "__init__.py"}


def _tracked_python_files() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every tracked `.py` path, split into (production modules, test suites)."""
    paths = [p for p in _tracked_files() if p.endswith(".py")]
    return (
        tuple(p for p in paths if not p.endswith("_test.py")),
        tuple(p for p in paths if p.endswith("_test.py")),
    )


def test_every_module_has_a_sibling_test() -> None:
    """AGENTS.md rule 2: `foo.py` is covered by `foo_test.py` in the same directory."""
    modules, tests = _tracked_python_files()
    suites = set(tests)
    missing = sorted(
        p
        for p in modules
        if Path(p).name not in UNTESTED_MODULE_NAMES
        and p.removesuffix(".py") + "_test.py" not in suites
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
        p
        for p in tests
        if p not in CROSS_CUTTING_TESTS
        and str(Path(p).parent / _PACKAGE_SUITES.get(Path(p).name, "")) not in known
        and p.removesuffix("_test.py") + ".py" not in known
    )
    assert not orphans, (
        f"test files with no module beside them: {orphans}; rename them to match the module they "
        "cover, fold them into the sibling suite for that module, or (for a genuinely "
        "cross-cutting suite) have a human add the exact path to CROSS_CUTTING_TESTS in "
        "wmo/repo_layout_test.py in the same change that documents it in AGENTS.md rule 2"
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
        p for p in _tracked_files() if re.search(r"(^|/)tests?/", p) and "/vendor/" not in p
    )
    assert not offenders, (
        f"tracked files under a tests/ directory: {offenders}; every suite lives beside the "
        "module it covers (AGENTS.md rule 2)"
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


#: AGENTS.md rule 4: the package tree, as one table. Every domain names the subpackages it owns, so
#: a responsibility that quietly moves out of its domain (the `wmo/evals/harbor` and
#: `wmo/optimize/harness` shapes this repo has already had and retired) fails here instead of in a
#: reviewer's memory. A domain may grow a new subpackage without editing this table; only `wmo`
#: itself is a closed set, because a package directly under it is a new product domain.
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
    actual = _subpackages("wmo")
    unexpected = actual - PACKAGE_TREE["wmo"]
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
    and this gate.
    """
    flat = sorted(
        path.name
        for path in WMO_DIR.glob("*.py")
        if path.name not in UNTESTED_MODULE_NAMES and not path.name.endswith("_test.py")
    )
    assert not flat, (
        f"production modules returned to the flat wmo namespace: {flat}; move each under the "
        "domain that owns it (AGENTS.md rule 4)"
    )


def test_imports_point_inward_from_common_and_runtime() -> None:
    """AGENTS.md rule 4: common is a leaf, and runtime does not know simulation or optimization.

    This is the one rule the tree cannot show: the directories can be perfect while an import
    inverts the dependency, and the cycle only surfaces later as an unimportable package.
    """
    violations = _banned_imports(
        WMO_DIR / "common", {"wmo.cli", "wmo.optimize", "wmo.runtime", "wmo.simulation"}
    ) + _banned_imports(WMO_DIR / "runtime", {"wmo.cli", "wmo.optimize", "wmo.simulation"})
    assert not violations, (
        f"imports point outward: {violations}; shared code belongs in wmo/common only if every "
        "domain may depend on it, and runtime may not reach into simulation or optimization "
        "(AGENTS.md rule 4)"
    )


def _banned_imports(root: Path, banned: set[str]) -> list[str]:
    """Every `<path>:<line> imports <module>` under `root` that names a banned package."""
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "conftest.py" or path.name.endswith("_test.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            violations.extend(
                f"{path.relative_to(WMO_DIR)}:{getattr(node, 'lineno', 0)} imports {module}"
                for module in modules
                if any(module == prefix or module.startswith(f"{prefix}.") for prefix in banned)
            )
    return violations


def test_no_finder_duplicate_files_are_tracked() -> None:
    """macOS Finder copies ("foo 2.py") dodge pytest collection and imports, so they rot
    silently; 24 of them once shipped in a PR before anyone noticed."""
    duplicates = [p for p in _tracked_files() if re.search(r" \d+\.\w+$", p)]
    assert not duplicates, (
        f"tracked Finder-style duplicate files {sorted(duplicates)}; delete the copies "
        "(they are never imported or collected) and keep the originals"
    )
