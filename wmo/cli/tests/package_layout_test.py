"""Structural checks for CLI package ownership and dependency direction."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable
from pathlib import Path

CLI_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYTHON_FILES = frozenset(
    {
        "__init__.py",
        "app.py",
        "app_test.py",
    }
)
CLI_PACKAGES = frozenset(
    {
        "build",
        "config",
        "gateway",
        "judge",
        "optimize",
        "providers",
        "run",
        "shared",
    }
)


def _production_modules(package: str) -> Iterable[Path]:
    """Yield production Python modules below one CLI package.

    Args:
        package: CLI package name relative to ``wmo/cli``.

    Yields:
        Python source paths excluding test modules.
    """
    for path in (CLI_ROOT / package).rglob("*.py"):
        if not path.name.endswith("_test.py"):
            yield path


def _cli_imports(path: Path) -> Iterable[str]:
    """Yield absolute CLI imports declared by one Python module.

    Args:
        path: Python source file to parse.

    Yields:
        Imported module names rooted at ``wmo.cli``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(CLI_ROOT).with_suffix("")
    module_parts = ["wmo", "cli", *relative.parts]
    if module_parts[-1] == "__init__":
        module_parts.pop()
    module = ".".join(module_parts)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("wmo.cli"):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                try:
                    target = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""),
                        package,
                    )
                except ImportError:
                    continue
            else:
                target = node.module or ""
            if target.startswith("wmo.cli"):
                yield target
                if target == "wmo.cli":
                    yield from (f"wmo.cli.{alias.name}" for alias in node.names)


def _forbidden_imports(package: str, allowed: frozenset[str]) -> frozenset[tuple[str, str]]:
    """Return CLI imports that point outside one package's allowed dependencies.

    Args:
        package: CLI package whose production modules are inspected.
        allowed: Absolute CLI package prefixes the owner may import.

    Returns:
        Source-relative path and forbidden import pairs.
    """
    violations: set[tuple[str, str]] = set()
    for path in _production_modules(package):
        for target in _cli_imports(path):
            if not any(target == prefix or target.startswith(f"{prefix}.") for prefix in allowed):
                violations.add((path.relative_to(CLI_ROOT).as_posix(), target))
    return frozenset(violations)


def test_cli_root_is_composition_only() -> None:
    """The CLI root contains only composition and its direct regression test."""
    root_python_files = frozenset(path.name for path in CLI_ROOT.glob("*.py"))
    packages = frozenset(
        path.name for path in CLI_ROOT.iterdir() if (path / "__init__.py").is_file()
    )

    assert root_python_files == ROOT_PYTHON_FILES
    assert packages == CLI_PACKAGES


def test_shared_package_does_not_depend_on_cli_domains() -> None:
    """Shared CLI primitives import only other shared CLI primitives."""
    violations = _forbidden_imports("shared", frozenset({"wmo.cli.shared"}))
    assert not violations, f"shared CLI imports command domains: {sorted(violations)}"


def test_providers_package_does_not_depend_on_cli_commands() -> None:
    """Provider setup depends only on provider and shared CLI packages."""
    violations = _forbidden_imports(
        "providers",
        frozenset({"wmo.cli.providers", "wmo.cli.shared"}),
    )
    assert not violations, f"provider CLI imports command domains: {sorted(violations)}"
