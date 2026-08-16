"""Dependency-surface test keeping production imports on the lean provider contract."""

from __future__ import annotations

import ast
from pathlib import Path

WMO_DIR = Path(__file__).resolve().parent.parent


def test_forbidden_provider_imports_are_absent() -> None:
    """Prove production callers stay on the lean provider dependency surface.

    The repository scan rejects direct imports of provider SDKs, retired provider layers, and
    heavyweight model libraries outside the canonical contracts and HTTP clients.
    """
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
        allowed={
            ("runtime/models/providers/bedrock.py", "boto3"),
            ("runtime/models/providers/bedrock.py", "botocore.config"),
        },
    )
    assert not violations, f"forbidden provider imports present: {violations}"


def _banned_imports(
    root: Path,
    banned: set[str],
    *,
    allowed: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Report production imports of banned modules found under a package tree.

    Args:
        root: Package directory searched recursively for Python sources. Test modules and
            `conftest.py` are skipped, so only production imports are reported.
        banned: Module paths that production code may not import. A source matches when it
            imports the exact path or any submodule of it.
        allowed: Optional exact ``(relative path, imported module)`` pairs that remain legal.

    Returns:
        One `path:line imports module` entry per banned import, ordered by file path.
    """
    permitted = allowed or set()
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "conftest.py" or path.name.endswith("_test.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            line = 0
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                line = node.lineno
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
                line = node.lineno
            for module in modules:
                if any(module == prefix or module.startswith(f"{prefix}.") for prefix in banned):
                    relative = path.relative_to(WMO_DIR)
                    if (relative.as_posix(), module) in permitted:
                        continue
                    violations.append(f"{relative}:{line} imports {module}")
    return violations
