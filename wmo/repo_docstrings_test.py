"""Monotonic public-docstring guardrails for the W1 migration."""

from __future__ import annotations

import ast
import functools
import subprocess
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCSTRING_BASELINE_REVISION = "e7aad17b2f5041769ad8107ab25e77d4e88729ca"


@dataclass(frozen=True, order=True)
class DocstringViolation:
    """One public API docstring requirement that the current source does not satisfy."""

    path: str
    symbol: str
    kind: str
    reason: str


# A migration owner moves a fixed baseline violation here. The test requires every fixed baseline
# violation to be recorded, preserves the row as history, and rejects a reintroduced violation.
DOCSTRING_TOMBSTONES: frozenset[DocstringViolation] = frozenset()


@functools.lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    """Return every tracked repository path or skip outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; public-docstring guardrails require a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; public-docstring guardrails require the repository")
    return tuple(result.stdout.splitlines())


def _git_output(arguments: list[str]) -> str:
    """Return Git output for a local read command or skip outside a checkout."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; public-docstring guardrails require a git checkout")
    if result.returncode != 0:
        pytest.skip("the frozen docstring baseline is unavailable in this checkout")
    return result.stdout


@functools.cache
def _tracked_files_at_revision(revision: str) -> tuple[str, ...]:
    """Return all repository paths at one immutable Git revision."""
    return tuple(_git_output(["ls-tree", "-r", "--name-only", revision]).splitlines())


def _is_production_python_path(relative_path: str) -> bool:
    """Return whether a path contains production Python subject to public-docstring checks."""
    return (
        relative_path.startswith("wmo/")
        and relative_path.endswith(".py")
        and not relative_path.endswith("_test.py")
        and Path(relative_path).name != "conftest.py"
    )


def _is_public_name(name: str) -> bool:
    """Return whether a Python name is public under the migration convention."""
    return not name.startswith("_")


def _is_protocol_class(node: ast.ClassDef) -> bool:
    """Return whether a class explicitly extends typing.Protocol."""
    for base in node.bases:
        candidate = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(candidate, ast.Name) and candidate.id == "Protocol":
            return True
        if isinstance(candidate, ast.Attribute) and candidate.attr == "Protocol":
            return True
    return False


def _body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """Return a function body after its optional leading docstring expression."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_trivial_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether one public function is eligible for a one-line docstring."""
    body = _body_without_docstring(node)
    if len(body) != 1:
        return False
    statement = body[0]
    return isinstance(statement, (ast.Pass, ast.Return)) or (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _descendants_without_nested_definitions(node: ast.AST) -> Iterator[ast.AST]:
    """Yield descendants of one API without crossing into nested definitions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _descendants_without_nested_definitions(child)


def _public_argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Return documented argument names after excluding the receiver convention."""
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    return tuple(argument.arg for argument in arguments if argument.arg not in {"self", "cls"})


def _has_google_section(docstring: str, section: str) -> bool:
    """Return whether a docstring contains one exact Google-style section heading."""
    return any(line.strip() == f"{section}:" for line in docstring.splitlines())


def _function_violations(
    relative_path: str,
    qualified_name: str,
    kind: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[DocstringViolation]:
    """Return public-docstring violations for one function or method."""
    docstring = ast.get_docstring(node, clean=False)
    if docstring is None:
        return {DocstringViolation(relative_path, qualified_name, kind, "missing-docstring")}
    if _is_trivial_function(node):
        return set()
    violations: set[DocstringViolation] = set()
    if len(docstring.splitlines()) == 1:
        violations.add(
            DocstringViolation(relative_path, qualified_name, kind, "nontrivial-one-line-docstring")
        )
    if _public_argument_names(node) and not _has_google_section(docstring, "Args"):
        violations.add(
            DocstringViolation(relative_path, qualified_name, kind, "missing-args-section")
        )
    descendants = tuple(_descendants_without_nested_definitions(node))
    if any(
        isinstance(descendant, ast.Return) and descendant.value is not None
        for descendant in descendants
    ):
        if not _has_google_section(docstring, "Returns"):
            violations.add(
                DocstringViolation(relative_path, qualified_name, kind, "missing-returns-section")
            )
    if any(isinstance(descendant, (ast.Yield, ast.YieldFrom)) for descendant in descendants):
        if not _has_google_section(docstring, "Yields"):
            violations.add(
                DocstringViolation(relative_path, qualified_name, kind, "missing-yields-section")
            )
    if any(isinstance(descendant, ast.Raise) for descendant in descendants):
        if not _has_google_section(docstring, "Raises"):
            violations.add(
                DocstringViolation(relative_path, qualified_name, kind, "missing-raises-section")
            )
    return violations


def _docstring_violations_in_source(
    relative_path: str, source: str
) -> frozenset[DocstringViolation]:
    """Return public module, class, protocol, function, and method violations from source."""
    if relative_path.endswith("_test.py"):
        return frozenset()
    tree = ast.parse(source, filename=relative_path)
    violations: set[DocstringViolation] = set()
    if ast.get_docstring(tree, clean=False) is None:
        violations.add(DocstringViolation(relative_path, "<module>", "module", "missing-docstring"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public_name(node.name):
            class_kind = "protocol" if _is_protocol_class(node) else "class"
            if ast.get_docstring(node, clean=False) is None:
                violations.add(
                    DocstringViolation(relative_path, node.name, class_kind, "missing-docstring")
                )
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(
                    member.name
                ):
                    violations.update(
                        _function_violations(
                            relative_path,
                            f"{node.name}.{member.name}",
                            "method",
                            member,
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(
            node.name
        ):
            violations.update(_function_violations(relative_path, node.name, "function", node))
    return frozenset(violations)


def _docstring_violations(paths: Iterable[str]) -> frozenset[DocstringViolation]:
    """Return current public-docstring violations from tracked production Python modules."""
    violations: set[DocstringViolation] = set()
    for relative_path in paths:
        if not _is_production_python_path(relative_path):
            continue
        path = REPO_ROOT / relative_path
        if path.is_file():
            violations.update(
                _docstring_violations_in_source(relative_path, path.read_text(encoding="utf-8"))
            )
    return frozenset(violations)


@functools.cache
def _baseline_docstring_violations() -> frozenset[DocstringViolation]:
    """Return the exact W1 baseline violations used as the active transition inventory."""
    violations: set[DocstringViolation] = set()
    for relative_path in _tracked_files_at_revision(DOCSTRING_BASELINE_REVISION):
        if not _is_production_python_path(relative_path):
            continue
        source = _git_output(["show", f"{DOCSTRING_BASELINE_REVISION}:{relative_path}"])
        violations.update(_docstring_violations_in_source(relative_path, source))
    return frozenset(violations)


def test_ruff_selects_public_google_docstring_rules() -> None:
    """Ruff selects public module, class, function, and method docstring presence checks."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as file_handle:
        config = tomllib.load(file_handle)
    selected = set(config["tool"]["ruff"]["lint"]["extend-select"])
    assert {"D100", "D101", "D102", "D103", "D104"} <= selected
    assert config["tool"]["ruff"]["lint"]["pydocstyle"]["convention"] == "google"


def test_public_docstring_transition_inventory_is_monotonic() -> None:
    """New violations fail and fixed baseline violations become permanent tombstones."""
    baseline = _baseline_docstring_violations()
    current = _docstring_violations(_tracked_files())
    new_violations = current - baseline
    fixed_violations = baseline - current
    missing_tombstones = fixed_violations - DOCSTRING_TOMBSTONES
    stale_tombstones = DOCSTRING_TOMBSTONES - fixed_violations
    reintroduced = current & DOCSTRING_TOMBSTONES
    assert not new_violations, f"new public-docstring violations: {sorted(new_violations)}"
    assert not missing_tombstones, (
        "fixed baseline public-docstring violations must be tombstoned: "
        f"{sorted(missing_tombstones)}"
    )
    assert not stale_tombstones, f"stale public-docstring tombstones: {sorted(stale_tombstones)}"
    assert not reintroduced, f"reintroduced public-docstring tombstones: {sorted(reintroduced)}"


def test_google_docstrings_accept_trivial_and_nontrivial_public_apis() -> None:
    """One-line trivial APIs and full Google sections are both direct passing fixtures."""
    source = '''"""Fixture module."""

from typing import Protocol

class CustomerProtocol(Protocol):
    """Provides a customer extension point."""

    def execute(self, request: str) -> str:
        """Normalize one request.

        Args:
            request: Request text supplied by the customer.

        Returns:
            The normalized request.

        Raises:
            ValueError: If the request is blank.
        """
        if not request:
            raise ValueError("request is required")
        return request.strip()


def identity(value: str) -> str:
    """Return the supplied value."""
    return value


def stream(values: list[str]):
    """Yield normalized values.

    Args:
        values: Values to normalize.

    Yields:
        Normalized values.
    """
    yield from (value.strip() for value in values)
'''
    assert not _docstring_violations_in_source("wmo/fixture.py", source)


def test_google_docstrings_reject_missing_and_nontrivial_one_line_public_apis() -> None:
    """Direct failing fixtures cover public modules, classes, protocols, functions, and methods."""
    source = '''from typing import Protocol


class UndocumentedProtocol(Protocol):
    ...


class CustomerProtocol(Protocol):
    """Provides a customer extension point."""

    def execute(self, request: str) -> str:
        """Execute a request."""
        if not request:
            raise ValueError("request is required")
        return request.strip()


def build(request: str) -> str:
    """Build a result."""
    normalized = request.strip()
    return normalized


def stream(values: list[str]):
    """Stream values."""
    yield from values
'''
    violations = _docstring_violations_in_source("wmo/fixture.py", source)
    reasons = {violation.reason for violation in violations}
    kinds = {violation.kind for violation in violations}
    assert "missing-docstring" in reasons
    assert "nontrivial-one-line-docstring" in reasons
    assert "missing-args-section" in reasons
    assert "missing-returns-section" in reasons
    assert "missing-raises-section" in reasons
    assert "missing-yields-section" in reasons
    assert {"module", "protocol", "method", "function"} <= kinds


def test_private_and_test_helpers_are_not_public_docstring_apis() -> None:
    """Private helpers and test fixtures remain outside the public API contract."""
    source = '''"""Fixture module."""

def _helper() -> None:
    pass
'''
    assert not _docstring_violations_in_source("wmo/fixture.py", source)
    assert not _docstring_violations_in_source("wmo/fixture_test.py", "def test_case(): pass\n")
