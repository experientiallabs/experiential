"""Executable repository structure and migration guardrails.

Runs against `git ls-files` so it checks what is TRACKED, not what happens to be on disk.
Skipped outside a git checkout (e.g. an installed sdist).
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import re
import subprocess
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

HAND_AUTHORED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
        ".md",
        ".rst",
    }
)
MAX_HAND_AUTHORED_LINES: Final[int] = 999

# Frozen from origin/main at the W1 baseline. Active entries may only shrink to a tombstone.
OVERSIZED_FILE_BASELINE_REVISION: Final[str] = "e7aad17b2f5041769ad8107ab25e77d4e88729ca"
OVERSIZED_FILE_INVENTORY: Final[dict[str, tuple[int, str]]] = {
    "wmo/cli/app.py": (3716, "active"),
    "wmo/cli/app_test.py": (3487, "active"),
    "wmo/cli/model_app.py": (1316, "active"),
    "wmo/cli/model_app_test.py": (1418, "active"),
    "wmo/cli/optimize_model_app.py": (1720, "active"),
    "wmo/cli/optimize_model_app_test.py": (1692, "active"),
    "wmo/cli/pool_registry_test.py": (1184, "active"),
    "wmo/cli/route_app.py": (1654, "active"),
    "wmo/cli/route_app_test.py": (3616, "active"),
    "wmo/cli/run_cmd.py": (1105, "active"),
    "wmo/cli/run_cmd_test.py": (1209, "active"),
    "wmo/cli/ui.py": (1147, "active"),
    "wmo/common/providers/pool_test.py": (1307, "active"),
    "wmo/common/providers/tinker_test.py": (1251, "active"),
    "wmo/optimize/model/config.py": (1142, "active"),
    "wmo/optimize/model/config_test.py": (1018, "active"),
    "wmo/optimize/model/data_test.py": (1727, "active"),
    "wmo/optimize/model/loop.py": (3656, "active"),
    "wmo/optimize/model/loop_test.py": (3700, "active"),
    "wmo/optimize/model/rollouts_test.py": (1028, "active"),
    "wmo/optimize/model/store.py": (1031, "active"),
    "wmo/optimize/routing/policy.py": (1345, "active"),
    "wmo/optimize/routing/policy_test.py": (1508, "active"),
    "wmo/optimize/routing/scorecard.py": (1101, "active"),
    "wmo/optimize/telemetry/hooks.py": (1111, "active"),
    "wmo/runtime/harness/pi_e2b.py": (1711, "active"),
    "wmo/runtime/harness/pi_e2b_test.py": (2440, "active"),
    "wmo/runtime/harness/vendor/pi-agent/src/harness/agent-harness.ts": (1029, "active"),
    "wmo/runtime/harness/vendor/pi-agent/test/agent-loop.test.ts": (1351, "active"),
    "wmo/simulation/serving/chat.py": (1915, "active"),
    "wmo/simulation/serving/chat_test.py": (3050, "active"),
}
OVERSIZED_FILE_TOMBSTONES: Final[frozenset[str]] = frozenset()

# Generated outputs are exempt by exact path, not by a broad suffix or directory rule. A future
# generated API client must add its exact tracked path here and live in a named `generated/`
# directory. The only generated output in this checkout today is the uv lockfile.
GENERATED_OUTPUTS: Final[frozenset[str]] = frozenset({"uv.lock"})

FORBIDDEN_IMPORTS: Final[dict[str, frozenset[str]]] = {
    "common": frozenset({"runtime", "simulation", "optimize", "cli"}),
    "runtime": frozenset({"simulation", "optimize", "cli"}),
    "simulation": frozenset({"optimize", "cli"}),
    "optimize": frozenset({"simulation", "cli"}),
}

# These are the current-main edges that the target package graph will remove with their owning
# legacy surfaces. The inventory is intentionally exact at the importing-module and imported-
# module level: adding a new forbidden edge, even from a known legacy module, fails immediately.
IMPORT_TRANSITION_INVENTORY: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("wmo/optimize/gepa.py", "wmo.simulation.retrieval"),
        ("wmo/optimize/gepa.py", "wmo.simulation.retrieval.leakfree"),
        ("wmo/simulation/environment.py", "wmo.optimize.reward"),
        ("wmo/simulation/serving/server.py", "wmo.optimize.reward"),
        ("wmo/simulation/serving/server.py", "wmo.optimize.routing.pareto"),
        ("wmo/simulation/serving/server.py", "wmo.optimize.routing.policy"),
        ("wmo/simulation/serving/savings.py", "wmo.optimize.routing.knn"),
        ("wmo/simulation/serving/savings.py", "wmo.optimize.routing.policy"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.compression"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.knn"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.pareto"),
        ("wmo/simulation/serving/chat.py", "wmo.optimize.routing.policy"),
        ("wmo/simulation/model/build.py", "wmo.optimize"),
        ("wmo/simulation/model/replay.py", "wmo.optimize.gepa"),
        ("wmo/simulation/model/replay.py", "wmo.optimize.judge"),
        ("wmo/simulation/model/world_model.py", "wmo.optimize.gepa"),
        ("wmo/simulation/model/world_model.py", "wmo.optimize.reward"),
        ("wmo/simulation/model/autoconfig.py", "wmo.optimize.judge"),
        ("wmo/simulation/evaluation/grid.py", "wmo.optimize.judge"),
        ("wmo/simulation/evaluation/open_loop.py", "wmo.optimize.judge"),
        ("wmo/optimize/research/gepa_scaling.py", "wmo.simulation.model.replay"),
        ("wmo/optimize/research/gepa_scaling.py", "wmo.simulation.retrieval"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.environment"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.model.world_model"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.scenarios.synthesis"),
        ("wmo/optimize/research/scenario_fidelity.py", "wmo.simulation.scenarios.verification"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.grounding"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.knowledge"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.replay"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.model.workspace"),
        ("wmo/optimize/research/trace_scaling.py", "wmo.simulation.retrieval"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.model.grounding"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.model.replay"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.model.workspace"),
        ("wmo/optimize/research/pipeline.py", "wmo.simulation.retrieval"),
        ("wmo/optimize/research/concurrency_run.py", "wmo.simulation.retrieval.leakfree"),
        ("wmo/optimize/routing/evaluation.py", "wmo.simulation.scenarios.spec"),
        ("wmo/optimize/routing/policy.py", "wmo.simulation.retrieval.embedders"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.ingest"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.model"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.scenarios.spec"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.serving.traces_source"),
        ("wmo/optimize/routing/sweep.py", "wmo.simulation.model.world_model"),
    }
)

# Deletion PRs move entries from the active inventory to this append-only tombstone set. Keeping
# the set explicit prevents a removed dependency from being quietly reintroduced later.
IMPORT_TRANSITION_TOMBSTONES: Final[frozenset[tuple[str, str]]] = frozenset()

LEGACY_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "wmo/optimize/gepa.py",
    "wmo/optimize/judge.py",
    "wmo/optimize/judge_quality.py",
    "wmo/optimize/research",
    "wmo/optimize/reward.py",
    "wmo/optimize/telemetry",
    "wmo/runtime/evaluation/harbor",
    "wmo/runtime/harness/vendor/pi-agent",
    "wmo/runtime/platform",
    "wmo/runtime/runs",
    "wmo/simulation/context",
    "wmo/simulation/evaluation",
    "wmo/simulation/serving",
)

# Exact tracked paths under the current legacy roots. This is deliberately explicit rather than
# derived at runtime, so a new path under a legacy root fails immediately.
LEGACY_PATH_INVENTORY: Final[frozenset[str]] = frozenset(
    {
        "wmo/optimize/gepa.py",
        "wmo/optimize/judge.py",
        "wmo/optimize/judge_quality.py",
        "wmo/optimize/research/__init__.py",
        "wmo/optimize/research/ablation.py",
        "wmo/optimize/research/ablation_test.py",
        "wmo/optimize/research/concurrency_plot.py",
        "wmo/optimize/research/concurrency_plot_test.py",
        "wmo/optimize/research/concurrency_run.py",
        "wmo/optimize/research/concurrency_run_test.py",
        "wmo/optimize/research/concurrency_scaling.py",
        "wmo/optimize/research/concurrency_scaling_test.py",
        "wmo/optimize/research/gepa_scaling.py",
        "wmo/optimize/research/gepa_scaling_test.py",
        "wmo/optimize/research/pipeline.py",
        "wmo/optimize/research/pipeline_test.py",
        "wmo/optimize/research/scaling_split.py",
        "wmo/optimize/research/scaling_split_test.py",
        "wmo/optimize/research/scenario_fidelity.py",
        "wmo/optimize/research/scenario_fidelity_test.py",
        "wmo/optimize/research/scenario_recovery.py",
        "wmo/optimize/research/scenario_recovery_test.py",
        "wmo/optimize/research/seed_stability.py",
        "wmo/optimize/research/seed_stability_test.py",
        "wmo/optimize/research/trace_scaling.py",
        "wmo/optimize/research/trace_scaling_test.py",
        "wmo/optimize/reward.py",
        "wmo/optimize/telemetry/__init__.py",
        "wmo/optimize/telemetry/backfill.py",
        "wmo/optimize/telemetry/backfill_test.py",
        "wmo/optimize/telemetry/conftest.py",
        "wmo/optimize/telemetry/hooks.py",
        "wmo/optimize/telemetry/hooks_test.py",
        "wmo/runtime/evaluation/harbor/__init__.py",
        "wmo/runtime/evaluation/harbor/agent.py",
        "wmo/runtime/evaluation/harbor/agent_test.py",
        "wmo/runtime/evaluation/harbor/ctrf.py",
        "wmo/runtime/evaluation/harbor/ctrf_test.py",
        "wmo/runtime/evaluation/harbor/e2b_environment.py",
        "wmo/runtime/evaluation/harbor/e2b_environment_test.py",
        "wmo/runtime/evaluation/harbor/e2b_template_policy.py",
        "wmo/runtime/evaluation/harbor/scorer.py",
        "wmo/runtime/evaluation/harbor/scorer_test.py",
        "wmo/runtime/evaluation/harbor/tasks.py",
        "wmo/runtime/evaluation/harbor/tasks_test.py",
        "wmo/runtime/harness/vendor/pi-agent/CHANGELOG.md",
        "wmo/runtime/harness/vendor/pi-agent/LICENSE",
        "wmo/runtime/harness/vendor/pi-agent/README.md",
        "wmo/runtime/harness/vendor/pi-agent/VENDOR.md",
        "wmo/runtime/harness/vendor/pi-agent/docs/agent-harness.md",
        "wmo/runtime/harness/vendor/pi-agent/docs/durable-harness.md",
        "wmo/runtime/harness/vendor/pi-agent/docs/hooks.md",
        "wmo/runtime/harness/vendor/pi-agent/docs/models.md",
        "wmo/runtime/harness/vendor/pi-agent/docs/observability.md",
        "wmo/runtime/harness/vendor/pi-agent/package.json",
        "wmo/runtime/harness/vendor/pi-agent/src/agent-loop.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/agent.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/agent-harness.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/compaction/branch-summarization.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/compaction/compaction.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/compaction/utils.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/env/nodejs.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/messages.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/prompt-templates.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/session/jsonl-repo.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/session/jsonl-storage.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/session/memory-repo.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/session/memory-storage.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/session/repo-utils.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/session/session.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/session/uuid.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/skills.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/system-prompt.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/types.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/utils/shell-output.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/harness/utils/truncate.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/index.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/node.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/proxy.ts",
        "wmo/runtime/harness/vendor/pi-agent/src/types.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/agent-loop.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/agent.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/e2e.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/agent-harness-stream.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/agent-harness.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/compaction.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/nodejs-env.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/prompt-templates.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/repo.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/resource-formatting.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/session-test-utils.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/session-uuid.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/session.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/skills.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/storage.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/system-prompt.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/harness/truncate.test.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/scratch/simple.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/utils/calculate.ts",
        "wmo/runtime/harness/vendor/pi-agent/test/utils/get-current-time.ts",
        "wmo/runtime/harness/vendor/pi-agent/tsconfig.build.json",
        "wmo/runtime/harness/vendor/pi-agent/vitest.config.ts",
        "wmo/runtime/harness/vendor/pi-agent/vitest.harness.config.ts",
        "wmo/runtime/platform/__init__.py",
        "wmo/runtime/platform/auth.py",
        "wmo/runtime/platform/auth_test.py",
        "wmo/runtime/platform/client.py",
        "wmo/runtime/platform/client_test.py",
        "wmo/runtime/platform/credentials.py",
        "wmo/runtime/platform/credentials_test.py",
        "wmo/runtime/platform/transfer.py",
        "wmo/runtime/platform/transfer_test.py",
        "wmo/runtime/runs/__init__.py",
        "wmo/runtime/runs/client.py",
        "wmo/runtime/runs/client_test.py",
        "wmo/runtime/runs/conftest.py",
        "wmo/runtime/runs/ledger.py",
        "wmo/runtime/runs/ledger_test.py",
        "wmo/runtime/runs/reader.py",
        "wmo/runtime/runs/reader_test.py",
        "wmo/runtime/runs/schema.py",
        "wmo/runtime/runs/schema_test.py",
        "wmo/simulation/context/__init__.py",
        "wmo/simulation/context/api_test.py",
        "wmo/simulation/context/apps.py",
        "wmo/simulation/context/apps_test.py",
        "wmo/simulation/context/brave.py",
        "wmo/simulation/context/brave_test.py",
        "wmo/simulation/context/connector.py",
        "wmo/simulation/context/connector_test.py",
        "wmo/simulation/context/credentials.py",
        "wmo/simulation/context/credentials_test.py",
        "wmo/simulation/context/github.py",
        "wmo/simulation/context/github_test.py",
        "wmo/simulation/context/google.py",
        "wmo/simulation/context/google_test.py",
        "wmo/simulation/context/notion.py",
        "wmo/simulation/context/notion_test.py",
        "wmo/simulation/context/oauth.py",
        "wmo/simulation/context/oauth_test.py",
        "wmo/simulation/context/slack.py",
        "wmo/simulation/context/slack_test.py",
        "wmo/simulation/context/store.py",
        "wmo/simulation/context/store_test.py",
        "wmo/simulation/context/types.py",
        "wmo/simulation/context/types_test.py",
        "wmo/simulation/evaluation/__init__.py",
        "wmo/simulation/evaluation/agreement.py",
        "wmo/simulation/evaluation/agreement_test.py",
        "wmo/simulation/evaluation/base.py",
        "wmo/simulation/evaluation/base_test.py",
        "wmo/simulation/evaluation/closed_loop.py",
        "wmo/simulation/evaluation/closed_loop_test.py",
        "wmo/simulation/evaluation/failover.py",
        "wmo/simulation/evaluation/failover_test.py",
        "wmo/simulation/evaluation/gold.py",
        "wmo/simulation/evaluation/gold_test.py",
        "wmo/simulation/evaluation/grid.py",
        "wmo/simulation/evaluation/grid_plot.py",
        "wmo/simulation/evaluation/grid_plot_test.py",
        "wmo/simulation/evaluation/grid_test.py",
        "wmo/simulation/evaluation/open_loop.py",
        "wmo/simulation/evaluation/open_loop_test.py",
        "wmo/simulation/evaluation/tasks.py",
        "wmo/simulation/evaluation/tasks_test.py",
        "wmo/simulation/serving/__init__.py",
        "wmo/simulation/serving/builds.py",
        "wmo/simulation/serving/builds_test.py",
        "wmo/simulation/serving/chat.py",
        "wmo/simulation/serving/chat_openai_client_test.py",
        "wmo/simulation/serving/chat_test.py",
        "wmo/simulation/serving/endpoint_config.py",
        "wmo/simulation/serving/endpoint_config_test.py",
        "wmo/simulation/serving/query_embeddings.py",
        "wmo/simulation/serving/query_embeddings_test.py",
        "wmo/simulation/serving/savings.py",
        "wmo/simulation/serving/savings_test.py",
        "wmo/simulation/serving/server.py",
        "wmo/simulation/serving/server_test.py",
        "wmo/simulation/serving/traces_source.py",
        "wmo/simulation/serving/traces_source_test.py",
    }
)
LEGACY_PATH_TOMBSTONES: Final[frozenset[str]] = frozenset()

# Exact current root commands. The target CLI is smaller, but deletion owners remove these
# entries with their commands. A command absent from this set fails before it can become an
# accidental supported surface.
ROOT_CLI_COMMAND_INVENTORY: Final[frozenset[str]] = frozenset(
    {
        "build",
        "config",
        "demo",
        "download",
        "eval",
        "ingest",
        "knowledge",
        "list",
        "login",
        "logout",
        "optimize",
        "play",
        "providers",
        "pull",
        "push",
        "research",
        "run",
        "runs",
        "scenarios",
        "serve",
        "status",
    }
)
ROOT_CLI_COMMAND_TOMBSTONES: Final[frozenset[str]] = frozenset()

# AGENTS.md rule 5: tracked top-level directories must be within this set, and the set is CLOSED.
# An agent may never add to it. A new entry requires a human to name that exact directory and
# grant permission for the name; the entry then lands in the same change that documents it in
# AGENTS.md rule 5. If work does not fit a surface below, it goes under the closest one or stays
# out of the repo — never into a new sibling.
ALLOWED_TOP_DIRS = {
    "wmo",  # the flagship package: all importable code
    "docs",  # reviewed public documentation (see the docs/ layout tests below)
    "assets",  # media referenced by README/docs
    ".claude",  # checked-in agent skills
    ".github",  # CI workflows
}


@functools.lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    """Every git-tracked path in the repo (one `git ls-files`, cached across the tests)."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; repo-layout rules only apply to a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; repo-layout rules only apply to the repository")
    return tuple(result.stdout.splitlines())


def _is_hand_authored_path(relative_path: str) -> bool:
    """Return whether a tracked path is subject to the physical-line rule."""
    return (
        Path(relative_path).suffix in HAND_AUTHORED_SUFFIXES
        and relative_path not in GENERATED_OUTPUTS
    )


def _line_count(path: Path) -> int:
    """Count physical lines, including blank and comment lines."""
    return len(path.read_text(encoding="utf-8").splitlines())


def _is_oversized_hand_authored_path(relative_path: str, path: Path) -> bool:
    """Return whether one covered path exceeds the physical-line limit."""
    return _is_hand_authored_path(relative_path) and _line_count(path) > MAX_HAND_AUTHORED_LINES


def _oversized_hand_authored_files(paths: Iterable[str]) -> tuple[tuple[str, int], ...]:
    """Return covered tracked files that reach the 1,000-line boundary."""
    oversized: list[tuple[str, int]] = []
    for relative_path in paths:
        if not _is_hand_authored_path(relative_path):
            continue
        path = REPO_ROOT / relative_path
        if not path.is_file():
            continue
        lines = _line_count(path)
        if lines > MAX_HAND_AUTHORED_LINES:
            oversized.append((relative_path, lines))
    return tuple(sorted(oversized))


def _active_oversized_file_counts() -> dict[str, int]:
    """Return baseline counts for active oversized-file transition entries."""
    return {
        path: baseline
        for path, (baseline, status) in OVERSIZED_FILE_INVENTORY.items()
        if status == "active"
    }


def _oversized_file_inventory_violations(
    oversized: Iterable[tuple[str, int]],
    tombstones: frozenset[str] = OVERSIZED_FILE_TOMBSTONES,
) -> tuple[str, ...]:
    """Return violations against the frozen oversized-file migration inventory."""
    actual = dict(oversized)
    active = _active_oversized_file_counts()
    violations = [
        f"new oversized path: {path}"
        for path in sorted(set(actual) - set(OVERSIZED_FILE_INVENTORY))
    ]
    violations.extend(
        f"active path is missing from the current oversized set and must be tombstoned: {path}"
        for path in sorted(set(active) - set(actual))
    )
    violations.extend(
        f"active path grew beyond its baseline: {path} ({actual[path]} > {baseline})"
        for path, baseline in sorted(active.items())
        if path in actual and actual[path] > baseline
    )
    violations.extend(
        f"tombstoned oversized path was reactivated: {path}"
        for path in sorted(set(actual) & tombstones)
    )
    invalid_entries = {
        path
        for path, (baseline, status) in OVERSIZED_FILE_INVENTORY.items()
        if baseline <= MAX_HAND_AUTHORED_LINES or status not in {"active", "tombstoned"}
    }
    if invalid_entries:
        violations.append(f"invalid oversized-file inventory entries: {sorted(invalid_entries)}")
    inventory_tombstones = {
        path for path, (_, status) in OVERSIZED_FILE_INVENTORY.items() if status == "tombstoned"
    }
    if set(OVERSIZED_FILE_INVENTORY) & tombstones:
        violations.append("oversized-file inventory and tombstones overlap")
    if inventory_tombstones != tombstones:
        violations.append("oversized-file tombstone rows and tombstones disagree")
    return tuple(violations)


def _module_name(relative_path: str) -> str:
    """Return the importable module name for a tracked WMO Python path."""
    parts = list(Path(relative_path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolved_import_target(node: ast.ImportFrom, module_name: str) -> str:
    """Resolve a possibly-relative import without importing the referenced module."""
    if node.level == 0:
        return node.module or ""
    relative = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(relative, module_name)
    except ImportError:
        return ""


def _import_targets(tree: ast.AST, module_name: str) -> Iterable[str]:
    """Yield absolute import targets found in an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = _resolved_import_target(node, module_name)
            if target:
                yield target
                if node.module is None or node.module == "wmo":
                    yield from (
                        f"{target}.{alias.name}" for alias in node.names if alias.name != "*"
                    )


def _forbidden_imports_in_source(relative_path: str, source: str) -> frozenset[tuple[str, str]]:
    """Find forbidden package edges in one source string."""
    module_name = _module_name(relative_path)
    parts = module_name.split(".")
    owner = parts[1] if len(parts) > 1 and parts[0] == "wmo" else ""
    tree = ast.parse(source, filename=relative_path)
    violations: set[tuple[str, str]] = set()
    for target in _import_targets(tree, module_name):
        target_parts = target.split(".")
        if len(target_parts) < 2 or target_parts[0] != "wmo":
            continue
        dependency = target_parts[1]
        if dependency in FORBIDDEN_IMPORTS.get(owner, frozenset()):
            violations.add((relative_path, target))
    return frozenset(violations)


def _forbidden_imports_in_repository(paths: Iterable[str]) -> frozenset[tuple[str, str]]:
    """Find all forbidden production-package edges using AST inspection only."""
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


def _legacy_path_candidates(paths: Iterable[str]) -> frozenset[str]:
    """Return tracked paths under the explicitly inventoried legacy roots."""
    return frozenset(
        path
        for path in paths
        if any(path == prefix or path.startswith(f"{prefix}/") for prefix in LEGACY_PATH_PREFIXES)
    )


def _unknown_legacy_paths(paths: Iterable[str]) -> frozenset[str]:
    """Return legacy-root paths absent from the frozen transition inventory."""
    return _legacy_path_candidates(paths) - LEGACY_PATH_INVENTORY


def _root_cli_commands() -> frozenset[str]:
    """Read the root Typer command map without dispatching a command or loading credentials."""
    from typer.core import TyperGroup
    from typer.main import get_command

    from wmo.cli.app import app

    command = get_command(app)
    if not isinstance(command, TyperGroup):
        raise AssertionError("the root Typer app did not produce a command group")
    return frozenset(command.commands)


def test_top_level_directories_are_allowlisted() -> None:
    """Every tracked top-level directory is on the AGENTS.md rule 5 allowlist."""
    tracked_dirs = {path.split("/", 1)[0] for path in _tracked_files() if "/" in path}
    unexpected = tracked_dirs - ALLOWED_TOP_DIRS
    assert not unexpected, (
        f"top-level directories {sorted(unexpected)} are not in the AGENTS.md rule 5 allowlist "
        f"{sorted(ALLOWED_TOP_DIRS)}. The allowlist is closed and agents may not extend it: put "
        "reusable code in wmo/ (self-contained building blocks in wmo/common/), finished reports "
        "in docs/, and one-off or scratch work OUTSIDE the repo. Adding a new top-level "
        "directory requires a human to grant permission for that exact name."
    )


def test_no_local_settings_files_are_tracked() -> None:
    """No generated settings.toml (telemetry ids) is ever committed."""
    offenders = [p for p in _tracked_files() if Path(p).name == "settings.toml"]
    assert not offenders, (
        f"local settings files are tracked: {offenders}; these are generated per-root artifacts "
        "(telemetry ids) and must stay gitignored"
    )


def test_no_bytecode_or_caches_are_tracked() -> None:
    """No __pycache__/.pyc artifacts are committed."""
    offenders = [p for p in _tracked_files() if "__pycache__" in p or p.endswith(".pyc")]
    assert not offenders, (
        f"bytecode/cache files are tracked: {offenders[:5]}; git rm --cached them and keep "
        "__pycache__/ in .gitignore"
    )


def test_docs_layout_is_exactly_readme_research_reference() -> None:
    """docs/ is the manifest, the CLI map, writeups with their figures, references, and cookbooks.

    Anything else (other top-level pages, stray dirs, figures outside figures/) is clutter that
    rule 5 says gets relocated or deleted.
    """
    allowed = re.compile(
        r"^docs/(README\.md"
        r"|usage\.md"
        r"|research/[^/]+\.md"
        r"|research/figures/[^/]+\.png"
        r"|reference/[^/]+\.md"
        r"|cookbook/[^/]+\.md)$"
    )
    offenders = [p for p in _tracked_files() if p.startswith("docs/") and not allowed.match(p)]
    assert not offenders, (
        f"files outside the docs/ layout: {offenders}; writeups go in docs/research/*.md with "
        "figures in docs/research/figures/, references in docs/reference/*.md, end-to-end walks "
        "in docs/cookbook/*.md, and docs/usage.md is the only other root page (AGENTS.md rule 5)"
    )


# Top-level directories this repo used to have. They are gone; a doc that still points at one is
# sending the reader to a path that does not exist.
RETIRED_TOP_DIRS = (".agents/", "deploy/", "examples/", "packages/", "web/")

#: Each retired directory as a regex anchored at a path-token boundary. A bare substring test
#: would fail the gate on ordinary prose: `web/` matches inside
#: `https://api.search.brave.com/res/v1/web/search`, and `packages/` inside `site-packages/` or
#: any `files.pythonhosted.org/packages/...` wheel URL.
_RETIRED_PATTERNS = tuple(
    (retired, re.compile(rf"(?<![\w./-]){re.escape(retired)}")) for retired in RETIRED_TOP_DIRS
)


def test_docs_never_point_at_a_retired_directory() -> None:
    """docs/ are finished products: every path they quote must still exist.

    Reproduction lives in the report itself (public wmo API or CLI), never behind a path that
    was deleted, and never behind a scratch workspace, which this repo no longer has.
    """
    offenders: list[tuple[str, str]] = []
    for p in _tracked_files():
        if not (p.startswith("docs/") and p.endswith(".md")):
            continue
        path = REPO_ROOT / p
        if not path.is_file():  # tolerate uncommitted deletes/renames mid-edit
            continue
        text = path.read_text(encoding="utf-8")  # once per doc, not once per retired dir
        offenders.extend(
            (p, retired) for retired, pattern in _RETIRED_PATTERNS if pattern.search(text)
        )
    assert not offenders, (
        f"docs pointing at retired top-level directories: {offenders}; those paths no longer "
        "exist. Quote reproduction as public wmo API/CLI in the report itself (AGENTS.md rule 5)"
    )


def test_docs_readme_indexes_every_doc() -> None:
    """docs/README.md's justification table must name every tracked docs/ file (rule 5).

    The manifest is what makes the justification rule enforceable; a doc or figure absent from
    it is either unjustified or the table has drifted.
    """
    readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    missing = [
        p
        for p in _tracked_files()
        if p.startswith("docs/") and p != "docs/README.md" and p.removeprefix("docs/") not in readme
    ]
    assert not missing, (
        f"docs files absent from docs/README.md's justification table: {missing}; every doc "
        "and figure gets a row or gets deleted (AGENTS.md rule 5)"
    )


def test_no_tracked_file_is_matched_by_ignore_rules() -> None:
    """A tracked file matched by a .gitignore rule is a conflict waiting to bite (re-adds fail)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-i", "-c", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; repo-layout rules only apply to a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; repo-layout rules only apply to the repository")
    offenders = result.stdout.splitlines()
    assert not offenders, (
        f"tracked files matched by ignore rules: {offenders[:5]}; fix the .gitignore pattern "
        "(add a ! negation or narrow the glob) so tracked artifacts stay re-addable"
    )


def test_there_is_no_uv_workspace() -> None:
    """One distribution, no members (AGENTS.md § One package).

    The workspace was retired when `packages/` was deleted: `environment-capture` resolves from
    PyPI and `llm-waterfall` was vendored into `wmo/common/vendor/waterfall/`. Reintroducing a
    member means reintroducing a top-level `packages/` directory, which rule 5 forbids outright.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        root = tomllib.load(fh)
    uv_config = root.get("tool", {}).get("uv", {})
    assert "workspace" not in uv_config, (
        "[tool.uv.workspace] is back; this repo publishes one distribution whose importable code "
        "is all of wmo/. Depend on PyPI or vendor under wmo/common/vendor/ "
        "(AGENTS.md § One package)"
    )
    assert "sources" not in uv_config, (
        "[tool.uv.sources] is back; with no workspace every dependency resolves from PyPI "
        "(AGENTS.md § One package)"
    )


def test_root_gate_covers_the_whole_package() -> None:
    """AGENTS.md § One package promises one root gate over the single package."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        root = tomllib.load(fh)
    testpaths = root["tool"]["pytest"]["ini_options"]["testpaths"]
    assert testpaths == ["wmo"], (
        f"testpaths is {testpaths}, not ['wmo']; the root gate covers the one package and every "
        "test is inline beside the module it covers (AGENTS.md § One package)"
    )


ALLOWED_TOP_FILES = {
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",  # not yet present; allowlisted so adding one never fights the gate
    "README.md",
    "conftest.py",
    "justfile",
    "pyproject.toml",
    "uv.lock",
}


def test_top_level_files_are_allowlisted() -> None:
    """Root files are an allowlist too — no Makefile/tox.ini/setup.cfg sprawl (rule 5)."""
    tracked_root_files = {p for p in _tracked_files() if "/" not in p}
    unexpected = tracked_root_files - ALLOWED_TOP_FILES
    assert not unexpected, (
        f"top-level files {sorted(unexpected)} are not allowlisted; config belongs in "
        "pyproject.toml, tasks in the justfile, and everything else under an allowlisted dir"
    )


def test_no_finder_duplicate_files_are_tracked() -> None:
    """macOS Finder copies ("foo 2.py") dodge pytest collection and imports, so they rot
    silently; 24 of them once shipped in a PR before anyone noticed."""
    duplicates = [p for p in _tracked_files() if re.search(r" \d+\.\w+$", p)]
    assert not duplicates, (
        f"tracked Finder-style duplicate files {sorted(duplicates)}; delete the copies "
        "(they are never imported or collected) and keep the originals"
    )


def test_hand_authored_files_stay_below_the_physical_line_limit() -> None:
    """Every new or rewritten covered file stays below the frozen migration boundary."""
    oversized = _oversized_hand_authored_files(_tracked_files())
    assert not _oversized_file_inventory_violations(oversized), (
        "hand-authored files must contain fewer than 1,000 physical lines unless they are an "
        "exact active W1 inventory entry at or below its frozen baseline: "
        f"{_oversized_file_inventory_violations(oversized)}"
    )


def test_oversized_file_inventory_is_frozen_at_the_baseline() -> None:
    """The inventory matches every current oversized path and records the W1 baseline revision."""
    assert OVERSIZED_FILE_BASELINE_REVISION == "e7aad17b2f5041769ad8107ab25e77d4e88729ca"
    assert not _oversized_file_inventory_violations(
        _oversized_hand_authored_files(_tracked_files())
    )


def test_oversized_file_inventory_rejects_new_paths_growth_and_tombstones() -> None:
    """The migration gate rejects new exceptions, growth, and reactivated tombstones."""
    active_path, baseline = next(iter(_active_oversized_file_counts().items()))
    assert "new oversized path: wmo/new_large.py" in _oversized_file_inventory_violations(
        (("wmo/new_large.py", 1000),)
    )
    assert any(
        f"active path grew beyond its baseline: {active_path}" in message
        for message in _oversized_file_inventory_violations(((active_path, baseline + 1),))
    )

    reactivation = _oversized_file_inventory_violations(
        (("wmo/reactivated.py", 1000),), frozenset({"wmo/reactivated.py"})
    )
    assert "tombstoned oversized path was reactivated: wmo/reactivated.py" in reactivation


@pytest.mark.parametrize("suffix", sorted(HAND_AUTHORED_SUFFIXES))
def test_line_limit_rejects_a_1000_line_fixture_for_each_covered_suffix(
    tmp_path: Path, suffix: str
) -> None:
    """A 1,000-line fixture is rejected for every covered hand-authored suffix."""
    fixture = tmp_path / f"fixture{suffix}"
    fixture.write_text("line\n" * (MAX_HAND_AUTHORED_LINES + 1), encoding="utf-8")
    assert _line_count(fixture) > MAX_HAND_AUTHORED_LINES
    assert _is_oversized_hand_authored_path(f"fixture{suffix}", fixture)


def test_only_exactly_named_generated_outputs_are_exempt() -> None:
    """The line-count exemption does not expand to a suffix or arbitrary generated directory."""
    assert not _is_hand_authored_path("uv.lock")
    assert _is_hand_authored_path("wmo/generated/api_client.py")
    assert _is_hand_authored_path("wmo/generated/large_output.py")


def test_generated_output_inventory_has_only_named_generated_paths() -> None:
    """Every exemption is either the lockfile or an exact path under a generated directory."""
    assert all(path == "uv.lock" or "generated" in Path(path).parts for path in GENERATED_OUTPUTS)


def test_import_boundaries_match_the_explicit_transition_inventory() -> None:
    """AST edges must be current inventory entries, with stale entries removed monotonically."""
    actual = _forbidden_imports_in_repository(_tracked_files())
    unexpected = actual - IMPORT_TRANSITION_INVENTORY
    stale = IMPORT_TRANSITION_INVENTORY - actual
    reintroduced = actual & IMPORT_TRANSITION_TOMBSTONES
    assert not unexpected, f"new forbidden import edges: {sorted(unexpected)}"
    assert not stale, (
        f"stale import transition entries must be removed with the old edge: {sorted(stale)}"
    )
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
    """Every forbidden dependency direction has a source fixture that the AST gate rejects."""
    relative_path = f"wmo/{owner}/fixture.py"
    violations = _forbidden_imports_in_source(
        relative_path, f"from wmo.{dependency} import forbidden_fixture\n"
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


def test_legacy_paths_match_the_explicit_transition_inventory() -> None:
    """Legacy paths remain exact until their owning deletion PR removes them."""
    actual = _legacy_path_candidates(_tracked_files())
    unexpected = _unknown_legacy_paths(_tracked_files())
    stale = LEGACY_PATH_INVENTORY - actual
    reintroduced = actual & LEGACY_PATH_TOMBSTONES
    assert not unexpected, (
        f"new legacy paths require their owning deletion PR: {sorted(unexpected)}"
    )
    assert not stale, f"stale legacy inventory entries: {sorted(stale)}"
    assert not reintroduced, f"retired legacy paths were reintroduced: {sorted(reintroduced)}"
    assert not LEGACY_PATH_INVENTORY & LEGACY_PATH_TOMBSTONES


def test_new_paths_under_a_legacy_root_are_rejected_by_the_transition_helper() -> None:
    """A new file under an inventoried legacy root is visible to the transition gate."""
    candidate = "wmo/optimize/research/new_surface.py"
    assert candidate in _unknown_legacy_paths((*_tracked_files(), candidate))


def test_root_cli_commands_match_the_explicit_transition_inventory() -> None:
    """The root CLI cannot grow a command outside its reviewed transition snapshot."""
    actual = _root_cli_commands()
    unexpected = actual - ROOT_CLI_COMMAND_INVENTORY
    missing = ROOT_CLI_COMMAND_INVENTORY - actual
    reintroduced = actual & ROOT_CLI_COMMAND_TOMBSTONES
    assert not unexpected, f"new root CLI commands: {sorted(unexpected)}"
    assert not missing, (
        f"root CLI inventory drifted without its owning deletion PR: {sorted(missing)}"
    )
    assert not reintroduced, f"retired root CLI commands were reintroduced: {sorted(reintroduced)}"
    assert not ROOT_CLI_COMMAND_INVENTORY & ROOT_CLI_COMMAND_TOMBSTONES
