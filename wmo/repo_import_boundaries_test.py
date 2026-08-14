"""Direct AST checks for production package dependency direction."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WMO_DIR = REPO_ROOT / "wmo"
FORBIDDEN_IMPORTS = {
    "common": frozenset({"runtime", "simulation", "optimize", "cli"}),
    "runtime": frozenset({"simulation", "optimize", "cli"}),
    "simulation": frozenset({"optimize", "cli"}),
    "optimize": frozenset({"simulation", "cli"}),
}


def _module_name(path: Path) -> str:
    """Return one production path's importable module name."""
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_package(path: Path) -> str:
    """Return the package used to resolve relative imports in one source file."""
    module = _module_name(path)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _dynamic_targets(tree: ast.AST) -> Iterable[str]:
    """Yield literal dynamic import targets."""
    importlib_aliases: set[str] = set()
    function_aliases: set[str] = set()
    builtin_module_aliases: set[str] = set()
    builtin_function_aliases = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            importlib_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "importlib"
            )
            builtin_module_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "builtins"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            function_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "import_module"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            builtin_function_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "__import__"
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        literal_import = isinstance(function, ast.Name) and function.id in {
            *builtin_function_aliases,
            *function_aliases,
        }
        attribute_import = (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_aliases
            and function.attr == "import_module"
        )
        builtin_attribute_import = (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in builtin_module_aliases
            and function.attr == "__import__"
        )
        argument = node.args[0] if node.args else None
        if argument is None:
            argument = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "name"),
                None,
            )
        if (literal_import or attribute_import or builtin_attribute_import) and isinstance(
            argument, ast.Constant
        ):
            if isinstance(argument.value, str):
                yield argument.value


def _import_targets(tree: ast.AST, package: str) -> Iterable[str]:
    """Yield static and literal dynamic import targets from one AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                try:
                    target = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                except ImportError:
                    continue
            else:
                target = node.module or ""
            if target:
                yield target
                if target == "wmo":
                    yield from (
                        f"wmo.{alias.name}"
                        for alias in node.names
                        if alias.name in FORBIDDEN_IMPORTS
                    )
    for target in _dynamic_targets(tree):
        if target.startswith("."):
            try:
                yield importlib.util.resolve_name(target, package)
            except ImportError:
                continue
        else:
            yield target


def _violations(path: Path, source: str) -> frozenset[tuple[str, str]]:
    """Return forbidden package edges in one source string."""
    owner_parts = _module_name(path).split(".")
    owner = owner_parts[1] if len(owner_parts) > 1 and owner_parts[0] == "wmo" else ""
    violations: set[tuple[str, str]] = set()
    tree = ast.parse(source, filename=str(path))
    for target in _import_targets(tree, _module_package(path)):
        parts = target.split(".")
        if len(parts) >= 2 and parts[0] == "wmo" and parts[1] in FORBIDDEN_IMPORTS.get(owner, ()):
            violations.add((path.relative_to(REPO_ROOT).as_posix(), target))
    return frozenset(violations)


def test_production_imports_follow_dependency_direction() -> None:
    """Current production code has no outward package dependency edge."""
    violations: set[tuple[str, str]] = set()
    for path in WMO_DIR.rglob("*.py"):
        if path.name.endswith("_test.py") or path.name == "conftest.py":
            continue
        violations.update(_violations(path, path.read_text(encoding="utf-8")))
    assert not violations, f"forbidden production imports: {sorted(violations)}"


def test_every_forbidden_direction_is_detected() -> None:
    """The gate flags each forbidden dependency edge rather than a subset of them."""
    for owner, dependencies in FORBIDDEN_IMPORTS.items():
        for dependency in dependencies:
            path = WMO_DIR / owner / "fixture.py"
            assert _violations(path, f"from wmo.{dependency} import forbidden\n") == {
                (f"wmo/{owner}/fixture.py", f"wmo.{dependency}")
            }


def test_relative_and_dynamic_imports_are_detected() -> None:
    """Relative and literal dynamic imports cannot evade dependency direction checks."""
    relative = WMO_DIR / "simulation" / "nested" / "fixture.py"
    dynamic = WMO_DIR / "simulation" / "fixture.py"
    assert _violations(relative, "from ...optimize import forbidden\n") == {
        ("wmo/simulation/nested/fixture.py", "wmo.optimize")
    }
    assert _violations(dynamic, '__import__("wmo.optimize")\n') == {
        ("wmo/simulation/fixture.py", "wmo.optimize")
    }
    assert _violations(dynamic, "from wmo import optimize\n") == {
        ("wmo/simulation/fixture.py", "wmo.optimize")
    }


@pytest.mark.parametrize(
    "source",
    [
        'import importlib\nimportlib.import_module("wmo.optimize")\n',
        'import importlib as loader\nloader.import_module(name="wmo.optimize")\n',
        'from importlib import import_module\nimport_module("wmo.optimize")\n',
        'from importlib import import_module as loader\nloader(name="wmo.optimize")\n',
        'from importlib import import_module\nimport_module("...optimize", package=__package__)\n',
        '__import__(name="wmo.optimize")\n',
        'from builtins import __import__ as loader\nloader("wmo.optimize")\n',
        'import builtins as builtin_loader\nbuiltin_loader.__import__(name="wmo.optimize")\n',
    ],
)
def test_literal_dynamic_import_forms_are_detected(source: str) -> None:
    """Positional, keyword, aliased, relative, and builtin literal imports are checked."""
    path = WMO_DIR / "simulation" / "nested" / "fixture.py"
    assert _violations(path, source) == {("wmo/simulation/nested/fixture.py", "wmo.optimize")}


@pytest.mark.parametrize(
    "source",
    [
        'name = "wmo.optimize"\n__import__(name)\n',
        'def import_module(name):\n    return name\nimport_module("wmo.optimize")\n',
        'loader(name="wmo.optimize")\n',
        'import importlib\nimportlib.import_module("wmo.common")\n',
    ],
)
def test_nonliteral_unrelated_and_inward_dynamic_imports_are_not_flagged(source: str) -> None:
    """The guard avoids guessing variable values, local functions, or permitted inward edges."""
    path = WMO_DIR / "simulation" / "fixture.py"
    assert not _violations(path, source)
