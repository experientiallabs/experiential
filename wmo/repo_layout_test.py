"""Executable repository migration guardrails.

Runs against `git ls-files` so it checks what is TRACKED, not what happens to be on disk.
Skipped outside a git checkout (e.g. an installed sdist).
"""

from __future__ import annotations

import functools
import subprocess
import tomllib
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
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

GUARDRAIL_CONFIG_PATH = REPO_ROOT / "wmo" / "repository_guardrails.toml"
GUARDRAIL_CONFIG_RELATIVE_PATH = GUARDRAIL_CONFIG_PATH.relative_to(REPO_ROOT).as_posix()


@dataclass(frozen=True)
class OversizedFileEntry:
    """One frozen legacy file-size exception and its monotonic state."""

    baseline_lines: int
    status: str


@dataclass(frozen=True)
class GuardrailConfig:
    """Parsed machine-readable inventories used by repository migration checks."""

    oversized_file_baseline_revision: str
    oversized_file_entries: Mapping[str, OversizedFileEntry]
    generated_outputs: frozenset[str]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or fail with an actionable inventory error."""
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be a string-keyed TOML table")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AssertionError(f"{label} must be a string-keyed TOML table")
        result[key] = item
    return result


def _parse_guardrail_config(raw_config: object) -> GuardrailConfig:
    """Parse one version of the single machine-readable guardrail inventory."""
    config = _mapping(raw_config, "guardrail configuration")
    oversized = _mapping(config.get("oversized_file_inventory"), "oversized_file_inventory")
    revision = oversized.get("baseline_revision")
    if not isinstance(revision, str):
        raise AssertionError("oversized_file_inventory.baseline_revision must be a string")
    raw_entries = _mapping(oversized.get("entries"), "oversized_file_inventory.entries")
    entries: dict[str, OversizedFileEntry] = {}
    for relative_path, raw_entry in raw_entries.items():
        entry = _mapping(raw_entry, f"oversized_file_inventory.entries.{relative_path}")
        baseline_lines = entry.get("baseline_lines")
        status = entry.get("status")
        if not isinstance(baseline_lines, int) or baseline_lines <= MAX_HAND_AUTHORED_LINES:
            raise AssertionError(f"{relative_path} must have an oversized baseline_lines value")
        if not isinstance(status, str) or status not in {"active", "tombstoned"}:
            raise AssertionError(f"{relative_path} must be active or tombstoned")
        entries[relative_path] = OversizedFileEntry(baseline_lines, status)
    generated = _mapping(config.get("generated_outputs"), "generated_outputs")
    raw_paths = generated.get("paths")
    if not isinstance(raw_paths, list):
        raise AssertionError("generated_outputs.paths must be a list of exact paths")
    generated_paths: list[str] = []
    for path in raw_paths:
        if not isinstance(path, str):
            raise AssertionError("generated_outputs.paths must be a list of exact paths")
        generated_paths.append(path)
    return GuardrailConfig(revision, entries, frozenset(generated_paths))


@functools.cache
def _guardrail_config() -> GuardrailConfig:
    """Load the current single machine-readable oversized-file inventory."""
    with GUARDRAIL_CONFIG_PATH.open("rb") as file_handle:
        return _parse_guardrail_config(tomllib.load(file_handle))


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

# W4.5 retired the final active paths below the frozen legacy roots. The exact set remains here as
# permanent history, while the active inventory is now empty.
W4_5_LEGACY_PATH_TOMBSTONES: Final[frozenset[str]] = frozenset(
    {
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
    }
)
LEGACY_PATH_INVENTORY: Final[frozenset[str]] = frozenset()
LEGACY_PATH_TOMBSTONES: Final[frozenset[str]] = W4_5_LEGACY_PATH_TOMBSTONES | frozenset(
    {
        "wmo/optimize/gepa.py",
        "wmo/simulation/evaluation/__init__.py",
        "wmo/simulation/evaluation/agreement.py",
        "wmo/simulation/evaluation/agreement_test.py",
        "wmo/simulation/evaluation/base.py",
        "wmo/simulation/evaluation/base_test.py",
        "wmo/simulation/evaluation/closed_loop.py",
        "wmo/simulation/evaluation/closed_loop_test.py",
        "wmo/simulation/evaluation/open_loop.py",
        "wmo/simulation/evaluation/open_loop_test.py",
        "wmo/simulation/evaluation/tasks.py",
        "wmo/simulation/evaluation/tasks_test.py",
        "wmo/simulation/serving/__init__.py",
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
        "wmo/optimize/telemetry/__init__.py",
        "wmo/optimize/telemetry/backfill.py",
        "wmo/optimize/telemetry/backfill_test.py",
        "wmo/optimize/telemetry/conftest.py",
        "wmo/optimize/telemetry/hooks.py",
        "wmo/optimize/telemetry/hooks_test.py",
        "wmo/optimize/judge.py",
        "wmo/optimize/judge_quality.py",
        "wmo/optimize/reward.py",
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
        "wmo/simulation/evaluation/failover.py",
        "wmo/simulation/evaluation/failover_test.py",
        "wmo/simulation/evaluation/grid.py",
        "wmo/simulation/evaluation/grid_plot.py",
        "wmo/simulation/evaluation/grid_plot_test.py",
        "wmo/simulation/evaluation/grid_test.py",
        "wmo/simulation/evaluation/gold.py",
        "wmo/simulation/evaluation/gold_test.py",
        "wmo/simulation/serving/builds.py",
        "wmo/simulation/serving/builds_test.py",
    }
)

# Exact current root commands. The target CLI is smaller, but deletion owners remove these
# entries with their commands. A command absent from this set fails before it can become an
# accidental supported surface.
ROOT_CLI_COMMAND_INVENTORY: Final[frozenset[str]] = frozenset(
    {
        "build",
        "config",
        "optimize",
        "run",
    }
)
ROOT_CLI_COMMAND_TOMBSTONES: Final[frozenset[str]] = frozenset(
    {
        "demo",
        "download",
        "eval",
        "ingest",
        "knowledge",
        "list",
        "login",
        "logout",
        "play",
        "providers",
        "pull",
        "push",
        "research",
        "runs",
        "scenarios",
        "serve",
        "status",
    }
)

# W8.6 is a clean break. Exact standalone files and every descendant of the retired package
# roots remain forbidden so later work cannot reintroduce an old world-model compatibility shim.
W8_RETIRED_PATH_TOMBSTONES: Final[frozenset[str]] = frozenset(
    {
        "wmo/optimize/base.py",
        "wmo/optimize/base_test.py",
        "wmo/optimize/gepa.py",
        "wmo/optimize/gepa_test.py",
        "wmo/simulation/environment.py",
        "wmo/simulation/environment_test.py",
        "wmo/simulation/evaluation",
        "wmo/simulation/model",
        "wmo/simulation/retrieval",
    }
)

# W4.5 is the matching clean break for the old agent and environment runtime. Package roots block
# every descendant, and standalone modules block compatibility files at their former locations.
W4_RETIRED_PATH_TOMBSTONES: Final[frozenset[str]] = frozenset(
    {
        "docs/reference/harness_delta.md",
        "wmo/runtime/__init__.py",
        "wmo/runtime/agents/default.py",
        "wmo/runtime/agents/llm.py",
        "wmo/runtime/agents/llm_test.py",
        "wmo/runtime/environment.py",
        "wmo/runtime/episode.py",
        "wmo/runtime/episode_test.py",
        "wmo/runtime/evaluation",
        "wmo/runtime/harness",
    }
)

# W3D removes the compatibility provider stack after every supported caller moved to canonical
# model contracts, runtime HTTP clients, and the minimal telemetry configuration owner.
W3D_RETIRED_PATH_TOMBSTONES: Final[frozenset[str]] = frozenset(
    {
        "wmo/cli/command_common.py",
        "wmo/cli/model_roles.py",
        "wmo/cli/model_roles_test.py",
        "wmo/common/config/card.py",
        "wmo/common/config/card_test.py",
        "wmo/common/config/config.py",
        "wmo/common/config/config_test.py",
        "wmo/common/config/store.py",
        "wmo/common/config/store_test.py",
        "wmo/common/judging/assertions.py",
        "wmo/common/judging/assertions_test.py",
        "wmo/common/judging/checklist.py",
        "wmo/common/judging/checklist_test.py",
        "wmo/common/judging/episode.py",
        "wmo/common/judging/episode_test.py",
        "wmo/common/judging/fidelity.py",
        "wmo/common/judging/fidelity_test.py",
        "wmo/common/observability/clock.py",
        "wmo/common/observability/metered.py",
        "wmo/common/observability/metered_test.py",
        "wmo/common/observability/pricing.py",
        "wmo/common/observability/pricing_test.py",
        "wmo/common/observability/reporting.py",
        "wmo/common/observability/store.py",
        "wmo/common/observability/store_test.py",
        "wmo/common/observability/tracker.py",
        "wmo/common/observability/tracker_test.py",
        "wmo/common/providers",
        "wmo/common/vendor",
    }
)


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


def _git_output(arguments: list[str]) -> str:
    """Run a local Git read command or skip outside a repository checkout."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; repository guardrails require a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; repository guardrails require the frozen baseline")
    return result.stdout


@functools.cache
def _tracked_files_at_revision(revision: str) -> tuple[str, ...]:
    """Return every tracked path at one immutable Git revision."""
    return tuple(_git_output(["ls-tree", "-r", "--name-only", revision]).splitlines())


@functools.cache
def _frozen_legacy_path_inventory() -> frozenset[str]:
    """Return every legacy-root path that existed at the frozen W1 revision."""
    return _legacy_path_candidates(
        _tracked_files_at_revision(_guardrail_config().oversized_file_baseline_revision)
    )


def _file_text_at_revision(revision: str, relative_path: str) -> str:
    """Return one UTF-8 text file from an immutable Git revision."""
    return _git_output(["show", f"{revision}:{relative_path}"])


@functools.cache
def _frozen_oversized_file_counts() -> Mapping[str, int]:
    """Recompute the exact oversized-file inventory at the frozen W1 revision."""
    revision = _guardrail_config().oversized_file_baseline_revision
    counts: dict[str, int] = {}
    for relative_path in _tracked_files_at_revision(revision):
        if not _is_hand_authored_path(relative_path):
            continue
        count = _physical_line_count(_file_text_at_revision(revision, relative_path))
        if count > MAX_HAND_AUTHORED_LINES:
            counts[relative_path] = count
    return counts


def _changed_paths_since_frozen_baseline() -> frozenset[str]:
    """Return tracked paths whose content differs from the frozen W1 revision."""
    revision = _guardrail_config().oversized_file_baseline_revision
    return frozenset(_git_output(["diff", "--name-only", revision, "--"]).splitlines())


@functools.cache
def _historical_tombstoned_oversized_paths() -> frozenset[str]:
    """Return every oversized path tombstoned by any reachable inventory revision."""
    tombstones: set[str] = set()
    revisions = _git_output(
        ["log", "--format=%H", "--", GUARDRAIL_CONFIG_RELATIVE_PATH]
    ).splitlines()
    for revision in revisions:
        source = _git_output(["show", f"{revision}:{GUARDRAIL_CONFIG_RELATIVE_PATH}"])
        historical_config = _parse_guardrail_config(tomllib.loads(source))
        tombstones.update(
            path
            for path, entry in historical_config.oversized_file_entries.items()
            if entry.status == "tombstoned"
        )
    return frozenset(tombstones)


def _is_hand_authored_path(relative_path: str) -> bool:
    """Return whether a tracked path is subject to the physical-line rule."""
    return (
        Path(relative_path).suffix in HAND_AUTHORED_SUFFIXES
        and relative_path not in _guardrail_config().generated_outputs
    )


def _physical_line_count(text: str) -> int:
    """Count newline-delimited physical lines without adding a line for a final newline."""
    if not text:
        return 0
    return text.count("\n") + int(not text.endswith("\n"))


def _line_count(path: Path) -> int:
    """Count physical lines, including blank and comment lines."""
    return _physical_line_count(path.read_text(encoding="utf-8"))


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


def _oversized_file_inventory_violations(
    oversized: Iterable[tuple[str, int]],
    *,
    entries: Mapping[str, OversizedFileEntry],
    baseline_counts: Mapping[str, int],
    changed_paths: Collection[str],
    historical_tombstones: Collection[str],
) -> tuple[str, ...]:
    """Return violations against the one frozen oversized-file inventory."""
    actual = dict(oversized)
    violations: list[str] = []
    if set(entries) != set(baseline_counts):
        missing_history = sorted(set(baseline_counts) - set(entries))
        added_entries = sorted(set(entries) - set(baseline_counts))
        violations.append(
            "oversized-file inventory history differs from the frozen baseline: "
            f"missing={missing_history}, added={added_entries}"
        )
    for path, entry in sorted(entries.items()):
        frozen_count = baseline_counts.get(path)
        if frozen_count != entry.baseline_lines:
            violations.append(
                f"frozen baseline count changed for {path}: "
                f"{entry.baseline_lines} != {frozen_count}"
            )
        if entry.status == "active":
            if path in historical_tombstones:
                violations.append(f"tombstoned oversized path was reactivated: {path}")
            if path not in actual:
                violations.append(
                    "active oversized path is absent or at most 999 lines and must be tombstoned: "
                    f"{path}"
                )
                continue
            if actual[path] > entry.baseline_lines:
                violations.append(
                    f"active path grew beyond its baseline: {path} "
                    f"({actual[path]} > {entry.baseline_lines})"
                )
            if path in changed_paths:
                violations.append(
                    "rewritten active oversized path must reach 999 lines or fewer and be "
                    f"tombstoned: {path}"
                )
        elif entry.status == "tombstoned":
            if path in actual:
                violations.append(f"tombstoned oversized path was reactivated: {path}")
        else:
            violations.append(f"invalid oversized-file status for {path}: {entry.status}")
    violations.extend(f"new oversized path: {path}" for path in sorted(set(actual) - set(entries)))
    return tuple(violations)


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


def _retired_w8_paths(paths: Iterable[str]) -> frozenset[str]:
    """Return paths that reintroduce an exact file or descendant of a W8.6 tombstone."""
    return frozenset(
        path
        for path in paths
        if any(
            path == tombstone or path.startswith(f"{tombstone}/")
            for tombstone in W8_RETIRED_PATH_TOMBSTONES
        )
    )


def _retired_w4_paths(paths: Iterable[str]) -> frozenset[str]:
    """Return paths that reintroduce an exact file or descendant of a W4.5 tombstone."""
    return frozenset(
        path
        for path in paths
        if any(
            path == tombstone or path.startswith(f"{tombstone}/")
            for tombstone in W4_RETIRED_PATH_TOMBSTONES
        )
    )


def _retired_w3d_paths(paths: Iterable[str]) -> frozenset[str]:
    """Return paths that reintroduce a W3D owner or a descendant of a retired package root."""
    return frozenset(
        path
        for path in paths
        if any(
            path == tombstone or path.startswith(f"{tombstone}/")
            for tombstone in W3D_RETIRED_PATH_TOMBSTONES
        )
    )


def test_hand_authored_files_stay_below_the_physical_line_limit() -> None:
    """Every new or rewritten covered file stays below the frozen migration boundary."""
    oversized = _oversized_hand_authored_files(_tracked_files())
    config = _guardrail_config()
    violations = _oversized_file_inventory_violations(
        oversized,
        entries=config.oversized_file_entries,
        baseline_counts=_frozen_oversized_file_counts(),
        changed_paths=_changed_paths_since_frozen_baseline(),
        historical_tombstones=_historical_tombstoned_oversized_paths(),
    )
    assert not violations, (
        "hand-authored files must contain fewer than 1,000 physical lines unless they are an "
        "exact active W1 inventory entry at its frozen baseline: "
        f"{violations}"
    )


def test_oversized_file_inventory_is_frozen_at_the_baseline() -> None:
    """The one inventory is an exact, immutable record of the frozen W1 baseline."""
    config = _guardrail_config()
    assert config.oversized_file_baseline_revision == "e7aad17b2f5041769ad8107ab25e77d4e88729ca"
    frozen = _frozen_oversized_file_counts()
    assert len(frozen) == 31
    assert set(config.oversized_file_entries) == set(frozen)
    assert all(
        entry.baseline_lines == frozen[path]
        for path, entry in config.oversized_file_entries.items()
    )


def test_oversized_file_inventory_rejects_new_growth_reactivation_and_history_loss() -> None:
    """The migration gate rejects every invalid state transition directly."""
    baseline = {"wmo/legacy.py": 1000, "wmo/retired.py": 1001}
    entries = {
        "wmo/legacy.py": OversizedFileEntry(1000, "active"),
        "wmo/retired.py": OversizedFileEntry(1001, "tombstoned"),
    }
    new_path = _oversized_file_inventory_violations(
        (("wmo/legacy.py", 1000), ("wmo/new_large.py", 1000)),
        entries=entries,
        baseline_counts=baseline,
        changed_paths=frozenset(),
        historical_tombstones=frozenset(),
    )
    assert "new oversized path: wmo/new_large.py" in new_path
    growth = _oversized_file_inventory_violations(
        (("wmo/legacy.py", 1001),),
        entries=entries,
        baseline_counts=baseline,
        changed_paths=frozenset(),
        historical_tombstones=frozenset(),
    )
    assert any("active path grew beyond its baseline: wmo/legacy.py" in item for item in growth)
    rewritten = _oversized_file_inventory_violations(
        (("wmo/legacy.py", 1000),),
        entries=entries,
        baseline_counts=baseline,
        changed_paths={"wmo/legacy.py"},
        historical_tombstones=frozenset(),
    )
    assert any("rewritten active oversized path" in item for item in rewritten)
    reactivation = _oversized_file_inventory_violations(
        (("wmo/legacy.py", 1000), ("wmo/retired.py", 1001)),
        entries=entries,
        baseline_counts=baseline,
        changed_paths=frozenset(),
        historical_tombstones=frozenset(),
    )
    assert "tombstoned oversized path was reactivated: wmo/retired.py" in reactivation
    history_loss = _oversized_file_inventory_violations(
        (("wmo/legacy.py", 1000),),
        entries={"wmo/legacy.py": OversizedFileEntry(1000, "active")},
        baseline_counts=baseline,
        changed_paths=frozenset(),
        historical_tombstones=frozenset(),
    )
    assert any("inventory history differs" in item for item in history_loss)
    status_reactivation = _oversized_file_inventory_violations(
        (("wmo/legacy.py", 1000), ("wmo/retired.py", 1001)),
        entries={
            "wmo/legacy.py": OversizedFileEntry(1000, "active"),
            "wmo/retired.py": OversizedFileEntry(1001, "active"),
        },
        baseline_counts=baseline,
        changed_paths=frozenset(),
        historical_tombstones={"wmo/retired.py"},
    )
    assert "tombstoned oversized path was reactivated: wmo/retired.py" in status_reactivation


@pytest.mark.parametrize("suffix", sorted(HAND_AUTHORED_SUFFIXES))
def test_line_limit_rejects_a_1000_line_fixture_for_each_covered_suffix(
    tmp_path: Path, suffix: str
) -> None:
    """A 1,000-line fixture is rejected for every covered hand-authored suffix."""
    fixture = tmp_path / f"fixture{suffix}"
    fixture.write_text("line\n" * (MAX_HAND_AUTHORED_LINES + 1), encoding="utf-8")
    assert _line_count(fixture) > MAX_HAND_AUTHORED_LINES
    assert _is_oversized_hand_authored_path(f"fixture{suffix}", fixture)


def test_physical_line_count_handles_final_newlines_without_an_extra_blank_line() -> None:
    """Physical LOC counts an unterminated final line but not a phantom line after a newline."""
    assert _physical_line_count("") == 0
    assert _physical_line_count("line") == 1
    assert _physical_line_count("line\n") == 1
    assert _physical_line_count("line\n\n") == 2


def test_only_exactly_named_generated_outputs_are_exempt() -> None:
    """The line-count exemption does not expand to a suffix or arbitrary generated directory."""
    assert not _is_hand_authored_path("uv.lock")
    assert _is_hand_authored_path("wmo/generated/api_client.py")
    assert _is_hand_authored_path("wmo/generated/large_output.py")


def test_generated_output_inventory_has_only_named_generated_paths() -> None:
    """Every exemption is either the lockfile or an exact path under a generated directory."""
    assert all(
        path == "uv.lock" or "generated" in Path(path).parts
        for path in _guardrail_config().generated_outputs
    )


def test_legacy_paths_match_the_explicit_transition_inventory() -> None:
    """Legacy paths remain exact until their owning deletion PR removes them."""
    actual = _legacy_path_candidates(_tracked_files())
    history = LEGACY_PATH_INVENTORY | LEGACY_PATH_TOMBSTONES
    unexpected = _unknown_legacy_paths(_tracked_files())
    stale = LEGACY_PATH_INVENTORY - actual
    reintroduced = actual & LEGACY_PATH_TOMBSTONES
    assert history == _frozen_legacy_path_inventory()
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


def test_w8_retired_paths_remain_tombstoned() -> None:
    """Deleted GEPA and text-world-model owners cannot return under compatibility paths."""
    assert not _retired_w8_paths(_tracked_files())


def test_w8_tombstones_reject_new_descendants() -> None:
    """A new module below a deleted W8.6 package root is rejected directly."""
    candidate = "wmo/simulation/model/compatibility.py"
    assert _retired_w8_paths((*_tracked_files(), candidate)) == {candidate}


def test_w4_retired_paths_remain_tombstoned() -> None:
    """Deleted harness and duplicate runtime owners cannot return through compatibility paths."""
    assert not _retired_w4_paths(_tracked_files())


def test_w4_tombstones_reject_new_descendants() -> None:
    """A new module below a deleted W4.5 package root is rejected directly."""
    candidate = "wmo/runtime/harness/compatibility.py"
    assert _retired_w4_paths((*_tracked_files(), candidate)) == {candidate}


def test_w3d_retired_paths_remain_tombstoned() -> None:
    """Deleted providers and their compatibility callers cannot return."""
    assert not _retired_w3d_paths(_tracked_files())


def test_w3d_tombstones_reject_new_descendants() -> None:
    """A new compatibility module below the deleted provider root is rejected directly."""
    candidate = "wmo/common/providers/compatibility.py"
    assert _retired_w3d_paths((*_tracked_files(), candidate)) == {candidate}


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
