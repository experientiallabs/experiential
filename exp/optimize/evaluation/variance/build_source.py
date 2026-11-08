"""The variance build's code identity: its own semantic sources, not the repository revision.

A variance build is keyed on its inputs (registration, project build, decision and reviewer
models, banks) and on the code that decides what it computes. Keying that code by the git
revision re-keyed the build on every commit, so an unrelated change cost a full replay (about
two hours of CPU even with every model call cached). :func:`build_source_identity` instead
digests :data:`VARIANCE_BUILD_PROTOCOL` and the bytes of every module in the build's import
closure inside :data:`SEMANTIC_PACKAGES` (exposure, views, scrubbing, tagging, clustering,
family review, scoping, placement, tool names, retrieval, world models, traces and tasks).
Edits to the transport, gateway, CLI or other stages leave a saved build replayable; the
revision is still recorded on every artifact envelope.

The trade-off: a semantic change in a module outside these packages that alters what the build
computes would replay the old build. Bump :data:`VARIANCE_BUILD_PROTOCOL` with such a change.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

from exp.common.core.artifacts import sha256_bytes, sha256_json

VARIANCE_BUILD_PROTOCOL = "variance-build-v1"
"""Bump when build semantics change outside :data:`SEMANTIC_PACKAGES`."""

BUILD_ENTRY_MODULES = (
    "exp.optimize.evaluation.variance.build",
    "exp.optimize.evaluation.variance.family_scopes",
)
"""Modules whose import closure is the build's code."""

SEMANTIC_PACKAGES = (
    "exp.optimize.evaluation.variance",
    "exp.simulation.mining",
    "exp.simulation.retrieval",
    "exp.simulation.rules",
    "exp.simulation.world_model",
    "exp.common.traces",
    "exp.common.tasks",
)
"""Packages whose modules, when imported by the build, decide what it computes."""

_ROOT = Path(__file__).resolve().parents[4]


def _source(module: str) -> Path | None:
    """Return a module's source file under the repository root, or None."""
    base = _ROOT / module.replace(".", "/")
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def build_sources() -> dict[str, Path]:
    """Return the build's semantic import closure, by module name."""
    seen: dict[str, Path] = {}
    stack = list(BUILD_ENTRY_MODULES)
    while stack:
        module = stack.pop()
        path = None if module in seen else _source(module)
        if path is None:
            continue
        seen[module] = path
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            stack.extend(name for name in names if name.startswith(SEMANTIC_PACKAGES))
    return seen


@functools.cache
def build_source_identity() -> str:
    """Digest the build protocol and its semantic sources; stable across unrelated commits."""
    return sha256_json(
        {
            "protocol": VARIANCE_BUILD_PROTOCOL,
            "sources": {
                module: sha256_bytes(path.read_bytes())
                for module, path in sorted(build_sources().items())
            },
        }
    )
