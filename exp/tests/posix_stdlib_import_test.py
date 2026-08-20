"""Package-wide guards against Unix-only stdlib imports at module scope.

``fcntl``, ``pty``, ``termios``, and ``tty`` are absent from Windows CPython. Pytest loads
``exp/conftest.py`` before collecting any ``exp/`` test, and ``import exp.cli.app`` pulls in the
picker, so those modules must not be imported until a POSIX helper actually needs them.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "exp"
POSIX_STDLIB = frozenset({"fcntl", "pty", "termios", "tty"})


def _root_module(name: str) -> str:
    """Return the top-level module name of one import target.

    Args:
        name: Dotted module path from an ``import`` or ``from`` statement.

    Returns:
        The first path segment.
    """
    return name.split(".", 1)[0]


def _posix_names_from_import(node: ast.stmt) -> Iterable[str]:
    """Yield Unix-only stdlib names imported by one statement.

    Args:
        node: A statement that may be an import.

    Yields:
        Root module names that belong to ``POSIX_STDLIB``.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = _root_module(alias.name)
            if root in POSIX_STDLIB:
                yield root
        return
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        root = _root_module(node.module)
        if root in POSIX_STDLIB:
            yield root


def _walk_module_body(body: list[ast.stmt]) -> Iterable[str]:
    """Yield Unix-only stdlib names imported outside functions and classes.

    Module-level ``if`` / ``try`` / ``with`` / ``match`` blocks still count: they run at import
    time. Function and class bodies are skipped so POSIX helpers may import locally.

    Args:
        body: Statement list from a module or a module-level compound statement.

    Yields:
        Root module names that belong to ``POSIX_STDLIB``.
    """
    for node in body:
        yield from _posix_names_from_import(node)
        if isinstance(node, ast.If | ast.For | ast.AsyncFor | ast.While):
            yield from _walk_module_body(node.body)
            yield from _walk_module_body(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _walk_module_body(node.body)
            for handler in node.handlers:
                yield from _walk_module_body(handler.body)
            yield from _walk_module_body(node.orelse)
            yield from _walk_module_body(node.finalbody)
        elif isinstance(node, ast.With | ast.AsyncWith):
            yield from _walk_module_body(node.body)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                yield from _walk_module_body(case.body)


def _module_level_posix_imports(path: Path) -> frozenset[str]:
    """Return Unix-only stdlib names imported at module scope in one file.

    Args:
        path: Python source path under ``exp/``.

    Returns:
        The forbidden names present at import time.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(_walk_module_body(tree.body))


def test_exp_sources_do_not_import_posix_stdlib_at_module_scope() -> None:
    """Collection and CLI import must not require Unix-only stdlib modules."""
    violations: list[str] = []
    for path in sorted(EXP_DIR.rglob("*.py")):
        names = _module_level_posix_imports(path)
        if names:
            relative = path.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{relative}: {', '.join(sorted(names))}")
    assert not violations, "module-level POSIX stdlib imports:\n" + "\n".join(violations)


def test_cli_and_conftest_import_when_posix_stdlib_is_missing() -> None:
    """A fresh interpreter still imports the CLI and suite conftest without Unix stdlib."""
    script = """
import sys

for name in ("fcntl", "pty", "termios", "tty"):
    sys.modules[name] = None

import exp.cli.app
import exp.conftest
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)
