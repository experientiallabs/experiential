"""Dependency-surface test keeping production imports on the lean provider contract."""

from __future__ import annotations

import ast
from pathlib import Path

WMO_DIR = Path(__file__).resolve().parent.parent


def test_forbidden_provider_imports_are_absent() -> None:
    """Callers use canonical model contracts and explicit HTTP clients and nothing else."""
    violations = _banned_imports(
        WMO_DIR,
        {
            "anthropic",
            "boto3",
            "botocore",
            "environment_capture",
            "mlx",
            "mlx_lm",
            "opentelemetry",
            "sklearn",
            "transformers",
            "wmo.common.providers",
            "wmo.common.vendor",
        },
    )
    assert not violations, f"forbidden provider imports present: {violations}"


def _banned_imports(root: Path, banned: set[str]) -> list[str]:
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
            for module in modules:
                if any(module == prefix or module.startswith(f"{prefix}.") for prefix in banned):
                    relative = path.relative_to(WMO_DIR)
                    line = getattr(node, "lineno", 0)
                    violations.append(f"{relative}:{line} imports {module}")
    return violations
