"""Keep the learner independent of gateway serving and simulation orchestration."""

import ast
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path

_EXP = Path(__file__).resolve().parents[3]
_PACKAGES = (_EXP / "common/claas", _EXP / "optimize/claas")
_FORBIDDEN = ("exp.runtime.gateway", "exp.simulation")
_PROCESS_CALLS = {
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "os.system",
    "os.popen",
    "os.spawnl",
    "os.spawnv",
    "os.execv",
    "os.execl",
    "multiprocessing.Process",
}


def _sources() -> Iterator[tuple[Path, ast.Module]]:
    """Inspect production source directly without importing optional model dependencies."""
    for package in _PACKAGES:
        for path in sorted(package.rglob("*.py")):
            if path.name.endswith("_test.py") or "tests" in path.relative_to(package).parts:
                continue
            yield path, ast.parse(path.read_text(), filename=str(path))


def _imports(path: Path, tree: ast.Module) -> Iterator[tuple[str, str]]:
    """Resolve direct imports and aliases, including relative package imports."""
    package = ".".join(("exp", *path.relative_to(_EXP).parts[:-1]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.asname or alias.name.partition(".")[0], alias.name
        elif isinstance(node, ast.ImportFrom):
            source = "." * node.level + (node.module or "")
            module = resolve_name(source, package) if node.level else source
            for alias in node.names:
                yield alias.asname or alias.name, f"{module}.{alias.name}"


def _qualified(node: ast.expr, aliases: dict[str, str]) -> str:
    """Resolve the function called through an imported name or module alias."""
    if isinstance(node, ast.Name):
        target = aliases.get(node.id, node.id)
        # ``import asyncio.subprocess`` still binds the top-level asyncio name.
        return node.id if target.startswith(node.id + ".") else target
    if isinstance(node, ast.Attribute):
        return f"{_qualified(node.value, aliases)}.{node.attr}"
    return ""


def test_learner_has_no_gateway_or_simulation_imports() -> None:
    """Scaffolds call the learner externally; gateway and synthesis are not core dependencies."""
    violations: list[str] = []
    for path, tree in _sources():
        modules = [module for _, module in _imports(path, tree)]
        # Literal plugin selections have the same ownership constraint as static imports.
        modules.extend(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        for module in modules:
            if any(module == prefix or module.startswith(prefix + ".") for prefix in _FORBIDDEN):
                violations.append(f"{path.relative_to(_EXP)} imports {module}")
    assert not violations, "learner crosses its package boundary: " + "; ".join(violations)


def test_owned_model_engines_do_not_spawn_vllm_processes() -> None:
    """Permit only the resident engines, local learner host, and explicit Modal volume sync."""
    violations: list[str] = []
    for path, tree in _sources():
        aliases = dict(_imports(path, tree))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _qualified(node.func, aliases)
            if called not in _PROCESS_CALLS:
                continue
            volume_sync = (
                path.relative_to(_EXP).as_posix() == "optimize/claas/backends/modal/persistence.py"
                and called == "asyncio.create_subprocess_exec"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "/usr/bin/sync"
            )
            local_learner = (
                path.relative_to(_EXP).as_posix() == "optimize/claas/backends/local/hosting.py"
                and called == "subprocess.Popen"
                and node.args
                and isinstance(node.args[0], ast.List)
                and len(node.args[0].elts) == 4
                and ast.unparse(node.args[0].elts[0]) == "sys.executable"
                and [ast.literal_eval(item) for item in node.args[0].elts[1:]]
                == ["-m", "exp.optimize.claas.service.launcher", "--config-stdin"]
            )
            if not volume_sync and not local_learner:
                violations.append(f"{path.relative_to(_EXP)}:{node.lineno} calls {called}")
    assert not violations, "model process ownership bypasses the engine: " + "; ".join(violations)
