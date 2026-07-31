"""Run one remote fit-only BigCodeBench router-selection seed.

The command deliberately stops before outer-heldout replay and before publishing a
selection lock. Each seed writes an independently auditable candidate report. A later
latency and artifact audit enriches the five winners before the immutable lock is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Literal

from coding_model_router_bigcodebench_fit import (
    FitData,
    canonical_candidate_config,
    fit_selected_static,
    load_fit_data,
    outer_splits,
    seed_split_provenance,
    select_fit_candidate,
)
from coding_model_router_bigcodebench_select import (
    CandidateSpec,
    CandidateValidation,
    KnnCandidateSpec,
    candidate_grid,
    knn_candidate_grid,
    select_knn_candidate,
    select_knn_economic_refinement,
    select_non_knn_candidate,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.core.files import write_text_atomic

logger = logging.getLogger(__name__)


class CandidateRecord(BaseModel):
    """One grouped fit-only candidate result with canonical configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["knn", "ordinal", "doubly-robust", "empirical-bayes"]
    name: str = Field(min_length=1)
    order: int = Field(ge=0)
    config_json: str = Field(min_length=2)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_reward: float = Field(ge=0.0, le=1.0)
    fit_cost_usd: float = Field(ge=0.0)
    matched_blind_reward: float = Field(ge=0.0, le=1.0)
    matched_blind_cost_usd: float = Field(ge=0.0)
    baseline_reward: float = Field(ge=0.0, le=1.0)
    baseline_cost_usd: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _config_matches_digest(self) -> CandidateRecord:
        value = json.loads(self.config_json)
        if not isinstance(value, dict):
            raise ValueError("candidate config must be one JSON object")
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if canonical != self.config_json:
            raise ValueError("candidate config is not canonical JSON")
        if hashlib.sha256(canonical.encode()).hexdigest() != self.config_sha256:
            raise ValueError("candidate config digest differs")
        return self


class SeedFitReport(BaseModel):
    """Immutable output of one outer seed's fit-only nested selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["bigcodebench-seed-fit-v1"] = "bigcodebench-seed-fit-v1"
    seed: int = Field(ge=0, le=4)
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tasks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcomes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_tasks: int = Field(gt=0)
    heldout_tasks: int = Field(gt=0)
    fit_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    heldout_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_arm: str = Field(min_length=1)
    baseline_fit_reward: float = Field(ge=0.0, le=1.0)
    baseline_fit_cost_usd: float = Field(ge=0.0)
    candidates: list[CandidateRecord] = Field(min_length=1_028, max_length=1_028)
    selected_name: str = Field(min_length=1)
    latency_audit_pending: Literal[True] = True
    target_outcomes_used: Literal[False] = False
    outer_heldout_evaluated: Literal[False] = False

    @model_validator(mode="after")
    def _selected_candidate_exists(self) -> SeedFitReport:
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("seed fit report contains duplicate candidate identities")
        if self.selected_name not in names:
            raise ValueError("selected candidate is absent from the fit report")
        return self


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(worktree: Path) -> str:
    """Return the exact clean source commit used for fitting."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("remote fit worktree is not clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("git did not return a full source commit")
    return commit


def _family(spec: CandidateSpec | KnnCandidateSpec) -> str:
    """Return the lock-compatible family name for one candidate spec."""
    return "knn" if isinstance(spec, KnnCandidateSpec) else spec.family


def candidate_record(result: CandidateValidation) -> CandidateRecord:
    """Convert an in-memory candidate result to its durable canonical record."""
    config_json, config_sha256 = canonical_candidate_config(result.spec.config())
    return CandidateRecord(
        family=_family(result.spec),
        name=result.spec.name,
        order=result.metric.order,
        config_json=config_json,
        config_sha256=config_sha256,
        fit_reward=result.value.reward,
        fit_cost_usd=result.value.cost_usd,
        matched_blind_reward=result.value.matched_blind_reward,
        matched_blind_cost_usd=result.value.matched_blind_cost_usd,
        baseline_reward=result.baseline.reward,
        baseline_cost_usd=result.baseline.cost_usd,
    )


def select_family_winner(
    non_knn: CandidateValidation,
    knn: CandidateValidation,
) -> CandidateValidation:
    """Mechanically select between the two fit-only family winners."""
    if abs(non_knn.baseline.reward - knn.baseline.reward) > 1e-12:
        raise ValueError("family winners used different fit-only baselines")
    metric = select_fit_candidate(
        [non_knn.metric, knn.metric],
        baseline_reward=non_knn.baseline.reward,
    )
    return non_knn if metric.name == non_knn.metric.name else knn


def run_seed_selection(
    root: Path,
    *,
    seed: int,
    work_dir: Path,
    output: Path,
    worktree: Path,
) -> SeedFitReport:
    """Run one complete nested fit-only seed and atomically persist its report."""
    if output.exists():
        raise FileExistsError(f"seed fit report already exists: {output}")
    data: FitData = load_fit_data(root)
    split = next(split for split in outer_splits(data.groups) if split.seed == seed)
    baseline = fit_selected_static(data, split.train_indices)
    non_knn_selected, non_knn_results = select_non_knn_candidate(
        data,
        split.train_indices,
        candidate_grid(),
        seed=seed,
    )
    knn_base, knn_results = select_knn_candidate(
        data,
        split.train_indices,
        knn_candidate_grid(),
        seed=seed,
        work_dir=work_dir / "knn-base",
    )
    knn_selected, economic_results = select_knn_economic_refinement(
        data,
        split.train_indices,
        knn_base,
        seed=seed,
        work_dir=work_dir / "knn-economic",
    )
    selected = select_family_winner(non_knn_selected, knn_selected)
    fit_ids_sha256, heldout_ids_sha256 = seed_split_provenance(data, split)
    all_results = [*non_knn_results, *knn_results, *economic_results]
    report = SeedFitReport(
        seed=seed,
        code_commit=_git_commit(worktree),
        tasks_sha256=_sha256(root / "tasks.jsonl"),
        scores_sha256=_sha256(root / "scores.jsonl"),
        outcomes_sha256=_sha256(root / "outcomes.jsonl"),
        oracle_report_sha256=_sha256(root / "oracle-report.json"),
        fit_tasks=len(split.train_indices),
        heldout_tasks=len(split.test_indices),
        fit_ids_sha256=fit_ids_sha256,
        heldout_ids_sha256=heldout_ids_sha256,
        baseline_arm=baseline.name,
        baseline_fit_reward=baseline.reward,
        baseline_fit_cost_usd=baseline.cost_usd,
        candidates=[candidate_record(result) for result in all_results],
        selected_name=selected.spec.name,
    )
    write_text_atomic(output, report.model_dump_json(indent=2) + "\n")
    logger.info(
        "seed=%d fit selection complete selected=%s reward=%.6f cost_usd=%.6f",
        seed,
        selected.spec.name,
        selected.value.reward,
        selected.value.cost_usd,
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse the remote seed-selection command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one fit-only outer seed from a clean checked-out worktree."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_seed_selection(
        args.root.resolve(),
        seed=args.seed,
        work_dir=args.work_dir.resolve(),
        output=args.output.resolve(),
        worktree=Path.cwd().resolve(),
    )


if __name__ == "__main__":
    main()
