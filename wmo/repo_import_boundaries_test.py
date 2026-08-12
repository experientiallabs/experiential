"""AST-based production import-boundary and transition-inventory guardrails."""

from __future__ import annotations

import ast
import functools
import importlib.util
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_REVISION = "e7aad17b2f5041769ad8107ab25e77d4e88729ca"
PR_BASE_REVISION = "origin/main"
SPLIT_PUBLIC_MODULES: Final[frozenset[str]] = frozenset(
    {
        "wmo.cli.optimize_model_app",
        "wmo.optimize.routing.scorecard",
    }
)

FORBIDDEN_IMPORTS: Final[dict[str, frozenset[str]]] = {
    "common": frozenset({"runtime", "simulation", "optimize", "cli"}),
    "runtime": frozenset({"simulation", "optimize", "cli"}),
    "simulation": frozenset({"optimize", "cli"}),
    "optimize": frozenset({"simulation", "cli"}),
}

IMPORT_TRANSITION_INVENTORY: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("wmo/optimize/gepa.py", "wmo.simulation.retrieval"),
        ("wmo/optimize/gepa.py", "wmo.simulation.retrieval.leakfree"),
        ("wmo/optimize/routing/evaluation.py", "wmo.simulation.scenarios.spec"),
        ("wmo/optimize/routing/policy.py", "wmo.simulation.retrieval.embedders"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.ingest"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.model"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.model.world_model"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.scenarios.spec"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.serving.traces_source"),
        ("wmo/simulation/model/build.py", "wmo.optimize"),
        ("wmo/simulation/model/replay.py", "wmo.optimize.gepa"),
        ("wmo/simulation/model/world_model.py", "wmo.optimize.gepa"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.compression"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.knn"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.pareto"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.policy"),
        ("wmo/simulation/serving/savings.py", "wmo.optimize.routing.knn"),
        ("wmo/simulation/serving/savings.py", "wmo.optimize.routing.policy"),
        ("wmo/simulation/serving/server.py", "wmo.optimize.routing.pareto"),
        ("wmo/simulation/serving/server.py", "wmo.optimize.routing.policy"),
    }
)
IMPORT_TRANSITION_TOMBSTONES: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("wmo/optimize/research/concurrency_run.py", "wmo.simulation.retrieval.leakfree"),
        ("wmo/optimize/research/gepa_scaling.py", "wmo.simulation.model.replay"),
        ("wmo/optimize/research/gepa_scaling.py", "wmo.simulation.retrieval"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.model.grounding"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.model.replay"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.model.workspace"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.retrieval"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.environment"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.model.world_model"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.scenarios.synthesis"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.scenarios.verification"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.grounding"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.knowledge"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.replay"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.workspace"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.retrieval"),
        ("wmo/simulation/environment.py", "wmo.optimize.reward"),
        ("wmo/simulation/evaluation/open_loop.py", "wmo.optimize.judge"),
        ("wmo/simulation/evaluation/grid.py", "wmo.optimize.judge"),
        ("wmo/simulation/model/autoconfig.py", "wmo.optimize.judge"),
        ("wmo/simulation/model/replay.py", "wmo.optimize.judge"),
        ("wmo/simulation/model/world_model.py", "wmo.optimize.reward"),
        ("wmo/simulation/serving/server.py", "wmo.optimize.reward"),
    }
)


@functools.lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    """Return every tracked path or skip outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; import-boundary guardrails require a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; import-boundary guardrails require the repository")
    return tuple(result.stdout.splitlines())


def _git_output(arguments: list[str]) -> str:
    """Return local Git output or skip outside a checkout."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; import-boundary guardrails require a git checkout")
    if result.returncode != 0:
        pytest.skip("the frozen import baseline is unavailable in this checkout")
    return result.stdout


@functools.cache
def _tracked_files_at_revision(revision: str) -> tuple[str, ...]:
    """Return every tracked path at an immutable Git revision."""
    return tuple(_git_output(["ls-tree", "-r", "--name-only", revision]).splitlines())


def _module_name(relative_path: str) -> str:
    """Return the importable module name for a tracked WMO Python path."""
    parts = list(Path(relative_path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@functools.cache
def _new_production_modules() -> tuple[str, ...]:
    """Return PR-added modules plus the public facades changed by extracted modules.

    `origin/main` is deliberate: this test is a PR regression rather than a comparison to the
    frozen W1 migration baseline, and it also covers staged shepherd fixes before their commit.
    Every listed module runs in its own interpreter below, so an import that succeeds only after
    another module happened to initialize cannot pass.
    """
    paths = _git_output(["diff", "--name-only", "--diff-filter=A", PR_BASE_REVISION, "--", "wmo"])
    modules = {
        _module_name(relative_path)
        for relative_path in paths.splitlines()
        if relative_path.endswith(".py")
        and not relative_path.endswith("_test.py")
        and relative_path != "wmo/conftest.py"
    }
    # The command and scorecard facades predate this PR, but both now delegate to extracted
    # owners. Exercise their direct imports alongside the new modules that form each split.
    modules.update(SPLIT_PUBLIC_MODULES)
    return tuple(sorted(modules))


def _module_package(relative_path: str) -> str:
    """Return the package used to resolve relative imports from one source module."""
    module_name = _module_name(relative_path)
    if relative_path.endswith("/__init__.py"):
        return module_name
    return module_name.rpartition(".")[0]


@pytest.mark.parametrize("module", _new_production_modules())
def test_pr_added_and_split_production_modules_import_in_fresh_interpreters(module: str) -> None:
    """Each PR-added or split production module imports without prior import state."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module({module!r})",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"fresh import failed for {module}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _resolved_import_target(node: ast.ImportFrom, package: str) -> str:
    """Resolve a possibly-relative import without importing its target module."""
    if node.level == 0:
        return node.module or ""
    try:
        return importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    except ImportError:
        return ""


def _dynamic_import_targets(tree: ast.AST) -> Iterable[str]:
    """Yield literal targets supplied to importlib and __import__ calls."""
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            importlib_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "importlib"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            import_module_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "import_module"
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_import_module = (
            isinstance(function, ast.Name) and function.id in import_module_aliases
        ) or (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_aliases
            and function.attr == "import_module"
        )
        is_builtin_import = isinstance(function, ast.Name) and function.id == "__import__"
        argument = node.args[0]
        if (is_import_module or is_builtin_import) and isinstance(argument, ast.Constant):
            if isinstance(argument.value, str):
                yield argument.value


def _import_targets(tree: ast.AST, package: str) -> Iterable[str]:
    """Yield static, TYPE_CHECKING, and literal dynamic import targets from an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = _resolved_import_target(node, package)
            if target:
                yield target
                if node.module is None or node.module == "wmo":
                    yield from (
                        f"{target}.{alias.name}" for alias in node.names if alias.name != "*"
                    )
    yield from _dynamic_import_targets(tree)


def _forbidden_imports_in_source(relative_path: str, source: str) -> frozenset[tuple[str, str]]:
    """Return forbidden package edges from one source string."""
    module_name = _module_name(relative_path)
    owner_parts = module_name.split(".")
    owner = owner_parts[1] if len(owner_parts) > 1 and owner_parts[0] == "wmo" else ""
    tree = ast.parse(source, filename=relative_path)
    violations: set[tuple[str, str]] = set()
    for target in _import_targets(tree, _module_package(relative_path)):
        target_parts = target.split(".")
        if len(target_parts) < 2 or target_parts[0] != "wmo":
            continue
        if target_parts[1] in FORBIDDEN_IMPORTS.get(owner, frozenset()):
            violations.add((relative_path, target))
    return frozenset(violations)


def _forbidden_imports_in_repository(paths: Iterable[str]) -> frozenset[tuple[str, str]]:
    """Return forbidden production-package edges without importing provider SDKs."""
    violations: set[tuple[str, str]] = set()
    for relative_path in paths:
        if not relative_path.endswith(".py") or relative_path.endswith("_test.py"):
            continue
        path = REPO_ROOT / relative_path
        if path.is_file():
            violations.update(
                _forbidden_imports_in_source(relative_path, path.read_text(encoding="utf-8"))
            )
    return frozenset(violations)


@functools.cache
def _frozen_import_inventory() -> frozenset[tuple[str, str]]:
    """Return the exact forbidden import edge set at the frozen W1 revision."""
    violations: set[tuple[str, str]] = set()
    for relative_path in _tracked_files_at_revision(BASELINE_REVISION):
        if not relative_path.endswith(".py") or relative_path.endswith("_test.py"):
            continue
        source = _git_output(["show", f"{BASELINE_REVISION}:{relative_path}"])
        violations.update(_forbidden_imports_in_source(relative_path, source))
    return frozenset(violations)


def test_import_boundaries_match_the_exact_monotonic_transition_inventory() -> None:
    """Forbidden imports remain active or tombstoned entries from the frozen baseline only."""
    actual = _forbidden_imports_in_repository(_tracked_files())
    history = IMPORT_TRANSITION_INVENTORY | IMPORT_TRANSITION_TOMBSTONES
    unexpected = actual - history
    stale = IMPORT_TRANSITION_INVENTORY - actual
    reintroduced = actual & IMPORT_TRANSITION_TOMBSTONES
    assert history == _frozen_import_inventory()
    assert not unexpected, f"new forbidden import edges: {sorted(unexpected)}"
    assert not stale, f"active import entries must be tombstoned: {sorted(stale)}"
    assert not reintroduced, f"retired forbidden imports were reintroduced: {sorted(reintroduced)}"
    assert not IMPORT_TRANSITION_INVENTORY & IMPORT_TRANSITION_TOMBSTONES


@pytest.mark.parametrize(
    ("owner", "dependency"),
    sorted(
        (owner, dependency)
        for owner, dependencies in FORBIDDEN_IMPORTS.items()
        for dependency in dependencies
    ),
)
def test_each_forbidden_import_direction_is_detected_by_ast(owner: str, dependency: str) -> None:
    """Every forbidden dependency direction has a direct AST fixture."""
    relative_path = f"wmo/{owner}/fixture.py"
    violations = _forbidden_imports_in_source(
        relative_path,
        f"from wmo.{dependency} import forbidden_fixture\n",
    )
    assert violations == {(relative_path, f"wmo.{dependency}")}


@pytest.mark.parametrize(
    "dependency", sorted({item for values in FORBIDDEN_IMPORTS.values() for item in values})
)
def test_root_package_reexports_are_checked_by_ast(dependency: str) -> None:
    """A root-package re-export cannot hide a forbidden dependency direction."""
    relative_path = "wmo/common/fixture.py"
    violations = _forbidden_imports_in_source(relative_path, f"from wmo import {dependency}\n")
    assert violations == {(relative_path, f"wmo.{dependency}")}


def test_relative_imports_are_resolved_from_the_containing_package() -> None:
    """A relative import cannot bypass a forbidden dependency by one package level."""
    relative_path = "wmo/simulation/engine/fixture.py"
    violations = _forbidden_imports_in_source(
        relative_path,
        "from ...optimize import forbidden_fixture\n",
    )
    assert violations == {(relative_path, "wmo.optimize")}


def test_type_checking_imports_are_checked_by_ast() -> None:
    """TYPE_CHECKING blocks remain package dependencies and cannot evade the gate."""
    relative_path = "wmo/common/fixture.py"
    violations = _forbidden_imports_in_source(
        relative_path,
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from wmo.runtime import forbidden_fixture\n",
    )
    assert violations == {(relative_path, "wmo.runtime")}


@pytest.mark.parametrize(
    "source",
    (
        'import importlib\nimportlib.import_module("wmo.optimize")\n',
        'from importlib import import_module\nimport_module("wmo.optimize")\n',
        '__import__("wmo.optimize")\n',
    ),
)
def test_literal_dynamic_imports_are_checked_by_ast(source: str) -> None:
    """Literal dynamic imports obey the same approved dependency direction."""
    relative_path = "wmo/simulation/fixture.py"
    violations = _forbidden_imports_in_source(relative_path, source)
    assert violations == {(relative_path, "wmo.optimize")}
