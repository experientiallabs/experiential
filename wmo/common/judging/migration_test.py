"""Explicit W6.7 deletion blockers inherited from the current W3 stacked base."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _BlockedReference:
    """One current caller that prevents deletion of a legacy judge owner."""

    path: str
    required_text: str


@dataclass(frozen=True)
class _DeferredDeletion:
    """One W6.7 owner whose callers belong to an unmerged legacy migration stream."""

    owner_path: str
    blocked_references: tuple[_BlockedReference, ...]


_DEFERRED_W6_7_DELETIONS = (
    _DeferredDeletion(
        owner_path="wmo/optimize/judge.py",
        blocked_references=(
            _BlockedReference("wmo/optimize/__init__.py", "from wmo.optimize.judge import"),
            _BlockedReference("wmo/cli/app.py", "from wmo.optimize.judge import"),
            _BlockedReference("wmo/optimize/gepa.py", "from wmo.optimize.judge import"),
            _BlockedReference(
                "wmo/optimize/research/gepa_scaling.py", "from wmo.optimize.judge import"
            ),
            _BlockedReference(
                "wmo/optimize/research/pipeline.py", "from wmo.optimize.judge import"
            ),
            _BlockedReference(
                "wmo/optimize/research/seed_stability.py", "from wmo.optimize.judge import"
            ),
            _BlockedReference(
                "wmo/optimize/research/trace_scaling.py", "from wmo.optimize.judge import"
            ),
            _BlockedReference(
                "wmo/simulation/evaluation/grid.py", "from wmo.optimize.judge import"
            ),
            _BlockedReference(
                "wmo/simulation/evaluation/open_loop.py", "from wmo.optimize.judge import"
            ),
            _BlockedReference(
                "wmo/simulation/model/autoconfig.py", "from wmo.optimize.judge import"
            ),
            _BlockedReference("wmo/simulation/model/replay.py", "from wmo.optimize.judge import"),
            _BlockedReference("wmo/simulation/model/build.py", "from wmo.optimize import"),
        ),
    ),
    _DeferredDeletion(
        owner_path="wmo/optimize/reward.py",
        blocked_references=(
            _BlockedReference("wmo/optimize/__init__.py", "from wmo.optimize.reward import"),
            _BlockedReference("wmo/simulation/environment.py", "from wmo.optimize.reward import"),
            _BlockedReference(
                "wmo/simulation/model/world_model.py", "from wmo.optimize.reward import"
            ),
            _BlockedReference(
                "wmo/optimize/routing/evaluation.py", "from wmo.optimize.reward import"
            ),
            _BlockedReference("wmo/optimize/routing/sweep.py", "from wmo.optimize.reward import"),
            _BlockedReference(
                "wmo/simulation/serving/server.py", "from wmo.optimize.reward import"
            ),
        ),
    ),
    _DeferredDeletion(
        owner_path="wmo/simulation/scenarios/verification/judge.py",
        blocked_references=(
            _BlockedReference(
                "wmo/simulation/scenarios/verification/__init__.py",
                "from wmo.simulation.scenarios.verification.judge import",
            ),
            _BlockedReference(
                "wmo/simulation/scenarios/verification/verify.py",
                "from wmo.simulation.scenarios.verification.judge import",
            ),
            _BlockedReference(
                "wmo/simulation/scenarios/builder.py",
                "from wmo.simulation.scenarios.verification import ChecklistJudge",
            ),
            _BlockedReference(
                "wmo/cli/app.py", "from wmo.simulation.scenarios import ChecklistJudge"
            ),
            _BlockedReference(
                "wmo/optimize/research/scenario_fidelity.py",
                "from wmo.simulation.scenarios.verification import ChecklistJudge",
            ),
        ),
    ),
    _DeferredDeletion(
        owner_path="wmo/simulation/evaluation/gold.py",
        blocked_references=(
            _BlockedReference(
                "wmo/simulation/evaluation/__init__.py",
                "from wmo.simulation.evaluation.gold import",
            ),
            _BlockedReference(
                "wmo/simulation/evaluation/closed_loop.py",
                "from wmo.simulation.evaluation.gold import",
            ),
            _BlockedReference(
                "wmo/cli/eval_closed_loop.py", "from wmo.simulation.evaluation.gold import"
            ),
        ),
    ),
    _DeferredDeletion(
        owner_path="wmo/simulation/environment.py",
        blocked_references=(
            _BlockedReference("wmo/cli/route_app.py", "score_on_close=True"),
            _BlockedReference("wmo/cli/optimize_model_app.py", "score_on_close=True"),
            _BlockedReference("wmo/optimize/routing/evaluation.py", "last_score"),
            _BlockedReference("wmo/optimize/routing/sweep.py", "last_score"),
        ),
    ),
)


def test_w6_7_stacked_base_blockers_are_explicit_until_the_legacy_callers_move() -> None:
    """W6.7 cannot silently omit owner deletions while this branch remains stacked on W3."""
    repository_root = Path(__file__).resolve().parents[3]
    for deletion in _DEFERRED_W6_7_DELETIONS:
        assert (repository_root / deletion.owner_path).is_file(), deletion.owner_path
        for reference in deletion.blocked_references:
            source_path = repository_root / reference.path
            assert source_path.is_file(), reference.path
            assert reference.required_text in source_path.read_text(encoding="utf-8"), (
                reference.path
            )
