"""Architecture tests for shared packages and import direction."""

from __future__ import annotations

import ast
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent
WMO_DIR = COMMON_DIR.parent


def test_common_domains_are_nested() -> None:
    """Shared infrastructure stays under one leaf package."""
    expected_dirs = {
        "config",
        "core",
        "evaluations",
        "judging",
        "models",
        "observability",
        "project",
        "rollouts",
        "routing",
        "tasks",
        "traces",
    }
    missing_dirs = sorted(name for name in expected_dirs if not (COMMON_DIR / name).is_dir())
    assert not missing_dirs, f"common packages missing under wmo/common: {missing_dirs}"

    legacy_dirs = sorted(
        name
        for name in ("config", "core", "providers", "research", "tracking", "utils")
        if (WMO_DIR / name).exists()
    )
    assert not legacy_dirs, f"packages returned to the flat wmo namespace: {legacy_dirs}"
    assert not (WMO_DIR / "telemetry.py").exists(), (
        "observability returned to the flat wmo namespace"
    )


def test_common_and_runtime_imports_point_inward() -> None:
    """Shared code is a leaf, and runtime does not depend on simulation or optimization."""
    common_violations = _banned_imports(
        COMMON_DIR,
        {"wmo.cli", "wmo.optimize", "wmo.runtime", "wmo.simulation"},
    )
    runtime_violations = _banned_imports(
        WMO_DIR / "runtime",
        {"wmo.cli", "wmo.optimize", "wmo.simulation"},
    )
    assert not common_violations, f"common imports product domains: {common_violations}"
    assert not runtime_violations, f"runtime imports outer domains: {runtime_violations}"


def test_retired_provider_imports_do_not_return() -> None:
    """Callers use canonical model contracts and HTTP clients, never the retired provider stack."""
    violations = _banned_imports(
        WMO_DIR,
        {
            "anthropic",
            "boto3",
            "botocore",
            "mlx",
            "mlx_lm",
            "openai",
            "transformers",
            "wmo.common.providers",
            "wmo.common.vendor",
        },
    )
    assert not violations, f"retired provider imports returned: {violations}"


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
