"""Audit, lock, replay, and promote the broad SWE-smith router experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, cast

import coding_model_router_autoresearch as autoresearch
import coding_model_router_swe_smith_select as selection
import joblib
import numpy as np

logger = logging.getLogger("coding-router-swe-smith-postfit")

AUDIT_DECISIONS = 10_000
TIE_BREAK_DECISIONS = 1_000
MAX_P50_MS = 5.0
MAX_P95_MS = 20.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260731


def _sha256_file(path: Path) -> str:
    """Hash one local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Atomically write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, object]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): value for key, value in payload.items()}


def _read_rows(path: Path) -> list[dict[str, object]]:
    """Read one compact paired source."""
    return selection._read_rows(path)


def _arrays(
    rows: list[dict[str, object]],
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    """Convert compact source rows to routing arrays."""
    return selection._arrays(rows)


def _as_float(value: object) -> float:
    """Convert one JSON number to float without accepting another type."""
    if not isinstance(value, int | float):
        raise TypeError(f"expected a JSON number, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    """Convert one JSON integer without accepting another type."""
    if not isinstance(value, int):
        raise TypeError(f"expected a JSON integer, got {type(value).__name__}")
    return value


def _spec(payload: dict[str, object]) -> autoresearch.CandidateSpec:
    """Reconstruct one frozen numeric candidate specification."""
    raw = cast(dict[str, object], payload["spec"])
    return autoresearch.CandidateSpec(
        name=str(raw["name"]),
        analyzer=cast(Literal["word", "char", "hashing", "structural"], raw["analyzer"]),
        components=_as_int(raw["components"]),
        estimator=cast(
            Literal[
                "ridge-uplift",
                "ridge-heads",
                "extra-heads",
                "hist-heads",
                "irt-difficulty",
                "profile-uplift",
            ],
            raw["estimator"],
        ),
        alpha=_as_float(raw.get("alpha", 1.0)),
        min_leaf=_as_int(raw.get("min_leaf", 10)),
        label_mode=cast(
            Literal["observed", "shuffled", "task-blind"],
            raw.get("label_mode", "observed"),
        ),
        profile_mode=cast(
            Literal[
                "none",
                "difficulty",
                "intent",
                "pr-category",
                "difficulty-intent",
                "difficulty-pr-category",
            ],
            raw.get("profile_mode", "none"),
        ),
        profile_temperature=_as_float(raw.get("profile_temperature", 0.1)),
        profile_prior_strength=_as_float(raw.get("profile_prior_strength", 50.0)),
        min_profile_tasks=_as_int(raw.get("min_profile_tasks", 25)),
    )


def _artifact_scores(payload: dict[str, object], texts: list[str]) -> np.ndarray:
    """Score prompts with one persisted numeric or kNN artifact."""
    family = payload.get("family")
    if family == "numeric":
        transformer = cast(autoresearch.FeatureTransformer, payload["transformer"])
        features = np.asarray(transformer.transform(texts), dtype=np.float64)
        estimators = cast(
            tuple[autoresearch.FittedRegressor, autoresearch.FittedRegressor | None],
            payload["estimators"],
        )
        return autoresearch._predict_score(_spec(payload), estimators, features)
    if family != "knn":
        raise ValueError(f"unsupported artifact family: {family}")
    raw_config = cast(dict[str, object], payload["config"])
    config = selection.KnnConfig(
        dim=_as_int(raw_config["dim"]),
        neighbors=_as_int(raw_config["neighbors"]),
        relative_similarity=_as_float(raw_config["relative_similarity"]),
        guard_z=_as_float(raw_config["guard_z"]),
        minimum_support=_as_int(raw_config["minimum_support"]),
    )
    features = autoresearch.HashingFeatureTransformer(config.dim).fit_transform(texts)
    result = selection._knn_fold_scores(
        np.asarray(payload["fit_features"], dtype=np.float64),
        features,
        np.asarray(payload["fit_uplift"], dtype=np.float64),
        [config],
    )
    return result[config.name]


def _artifact_routes(payload: dict[str, object], texts: list[str]) -> np.ndarray:
    """Return strong-arm route decisions for one artifact."""
    threshold = _as_float(payload["threshold"])
    return _artifact_scores(payload, texts) >= threshold


def _route_digest(routes: np.ndarray) -> str:
    """Hash a boolean route vector."""
    return hashlib.sha256(np.asarray(routes, dtype=np.uint8).tobytes()).hexdigest()


def _latency(payload: dict[str, object], texts: list[str], decisions: int) -> dict[str, float]:
    """Measure single-request route latency over a deterministic prompt cycle."""
    if not texts:
        raise ValueError("latency audit has no prompts")
    timings = np.empty(decisions, dtype=np.float64)
    for index in range(decisions):
        started = time.perf_counter_ns()
        route = _artifact_routes(payload, [texts[index % len(texts)]])
        timings[index] = (time.perf_counter_ns() - started) / 1_000_000.0
        if len(route) != 1:
            raise AssertionError("single-request route returned a non-singleton vector")
    return {
        "decisions": decisions,
        "p50_ms": float(np.quantile(timings, 0.50)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "max_ms": float(np.max(timings)),
    }


def _audit(args: argparse.Namespace) -> None:
    """Reload and audit one fit-only artifact without heldout access."""
    source = args.fit_source.resolve()
    report_path = args.report.resolve()
    artifact_path = args.artifact.resolve()
    report = _read_object(report_path)
    winner = cast(dict[str, object], report["winner"])
    rows = _read_rows(source)
    _, _, texts, weak, strong = _arrays(rows)
    persisted = cast(dict[str, object], joblib.load(artifact_path))
    with tempfile.TemporaryDirectory() as temporary:
        recreated_path = Path(temporary) / "recreated.joblib"
        selection._fit_artifact(winner, texts, weak, strong, recreated_path)
        recreated = cast(dict[str, object], joblib.load(recreated_path))
        recreated_routes = _artifact_routes(recreated, texts)
    network_attempts: list[str] = []

    def _deny_network(event: str, _arguments: tuple[object, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
            network_attempts.append(event)
            raise RuntimeError(f"route attempted network operation: {event}")

    sys.addaudithook(_deny_network)
    persisted_routes = _artifact_routes(persisted, texts)
    latency = _latency(persisted, texts[:32], AUDIT_DECISIONS)
    parity = np.array_equal(persisted_routes, recreated_routes)
    passed = (
        parity
        and not network_attempts
        and args.internet_disabled
        and latency["p50_ms"] < MAX_P50_MS
        and latency["p95_ms"] < MAX_P95_MS
        and persisted.get("target_outcomes_used") is False
    )
    result = {
        "schema": "swe-smith-broad-artifact-audit-v1",
        "seed": report["seed"],
        "source_commit": report["source_commit"],
        "report_sha256": _sha256_file(report_path),
        "fit_source_sha256": _sha256_file(source),
        "artifact_sha256": _sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "winner_name": winner["name"],
        "route_digest": _route_digest(persisted_routes),
        "strong_routes": int(np.sum(persisted_routes)),
        "route_parity_after_independent_refit": parity,
        "zero_network_attempts": not network_attempts,
        "internet_disabled": args.internet_disabled,
        "latency": latency,
        "latency_gate_ms": {"p50": MAX_P50_MS, "p95": MAX_P95_MS},
        "heldout_outcomes_used": False,
        "target_outcomes_used": False,
        "passed": passed,
    }
    _write_json(args.output.resolve(), result)
    if not passed:
        raise RuntimeError("SWE-smith artifact failed its serving audit")
    logger.info(
        "artifact audit passed seed=%s winner=%s p50_ms=%.4f p95_ms=%.4f",
        report["seed"],
        winner["name"],
        latency["p50_ms"],
        latency["p95_ms"],
    )


def _canonical_order() -> dict[str, int]:
    """Return the frozen non-control candidate order."""
    names = [spec.name for spec in selection._candidate_specs()]
    names.extend(config.name for config in selection._knn_configs())
    return {name: index for index, name in enumerate(names)}


def _reports(directory: Path) -> list[dict[str, object]]:
    """Read and validate five immutable seed reports."""
    reports = [_read_object(directory / f"seed-{seed}.json") for seed in range(5)]
    inventories = [
        {str(row["name"]) for row in cast(list[dict[str, object]], report["leaderboard"])}
        for report in reports
    ]
    if any(inventory != inventories[0] for inventory in inventories[1:]):
        raise ValueError("fit reports have different candidate inventories")
    if any(report.get("target_outcomes_used") is not False for report in reports):
        raise ValueError("fit report does not preserve the target boundary")
    return reports


def _candidate_matrix(
    reports: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Index every non-control candidate across all five fit reports."""
    matrix: dict[str, list[dict[str, object]]] = {}
    for report in reports:
        for row in cast(list[dict[str, object]], report["leaderboard"]):
            if bool(row["is_control"]):
                continue
            matrix.setdefault(str(row["name"]), []).append(row)
    if any(len(rows) != 5 for rows in matrix.values()):
        raise ValueError("candidate matrix is not dense across five seeds")
    return matrix


def _candidate_summary(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize one candidate's fit-only operating point across seeds."""
    primary = [cast(dict[str, object], row["primary"]) for row in rows]
    retentions = [_as_float(value["retention"]) for value in primary]
    return {
        "name": name,
        "family": rows[0]["family"],
        "config": rows[0]["config"],
        "seed_retentions": retentions,
        "seed_strong_traffic": [_as_float(value["strong_traffic"]) for value in primary],
        "seed_router_reward": [_as_float(value["router_reward"]) for value in primary],
        "minimum_retention": min(retentions),
        "mean_retention": float(np.mean(retentions)),
        "mean_strong_traffic": float(
            np.mean([_as_float(value["strong_traffic"]) for value in primary])
        ),
        "mean_router_reward": float(
            np.mean([_as_float(value["router_reward"]) for value in primary])
        ),
        "fit_quality_feasible": all(value >= 0.95 for value in retentions),
    }


def _fit_tie_candidate(
    name: str,
    candidate_rows: list[dict[str, object]],
    sources_dir: Path,
    temporary: Path,
) -> dict[str, object]:
    """Fit and time one consensus finalist on every outer-fit partition."""
    latencies: list[float] = []
    sizes: list[int] = []
    route_digests: list[str] = []
    artifact_paths: list[str] = []
    for seed, candidate in enumerate(candidate_rows):
        source = sources_dir / f"seed-{seed}-fit.json"
        rows = _read_rows(source)
        _, _, texts, weak, strong = _arrays(rows)
        artifact = temporary / name / f"seed-{seed}.joblib"
        selection._fit_artifact(candidate, texts, weak, strong, artifact)
        payload = cast(dict[str, object], joblib.load(artifact))
        routes = _artifact_routes(payload, texts)
        latency = _latency(payload, texts[:32], TIE_BREAK_DECISIONS)
        latencies.append(latency["p50_ms"])
        sizes.append(artifact.stat().st_size)
        route_digests.append(_route_digest(routes))
        artifact_paths.append(str(artifact))
    return {
        "name": name,
        "mean_p50_ms": float(np.mean(latencies)),
        "seed_p50_ms": latencies,
        "mean_artifact_bytes": float(np.mean(sizes)),
        "seed_artifact_bytes": sizes,
        "seed_route_digests": route_digests,
        "temporary_artifacts": artifact_paths,
    }


def _lock(args: argparse.Namespace) -> None:
    """Choose and fit one five-seed consensus before any heldout replay."""
    reports_dir = args.reports_dir.resolve()
    audits_dir = args.audits_dir.resolve()
    sources_dir = args.sources_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    reports = _reports(reports_dir)
    winner_audits = [_read_object(audits_dir / f"seed-{seed}.json") for seed in range(5)]
    if any(audit.get("passed") is not True for audit in winner_audits):
        raise ValueError("a fit-selected winner has not passed its artifact audit")
    matrix = _candidate_matrix(reports)
    order = _canonical_order()
    summaries = [_candidate_summary(name, rows) for name, rows in matrix.items()]
    feasible = [summary for summary in summaries if bool(summary["fit_quality_feasible"])]
    if feasible:
        best_traffic = min(_as_float(summary["mean_strong_traffic"]) for summary in feasible)
        traffic_ties = [
            summary
            for summary in feasible
            if _as_float(summary["mean_strong_traffic"]) == best_traffic
        ]
        best_reward = max(_as_float(summary["mean_router_reward"]) for summary in traffic_ties)
        finalists = [
            summary
            for summary in traffic_ties
            if _as_float(summary["mean_router_reward"]) == best_reward
        ]
        consensus_feasible = True
    else:
        best_minimum = max(_as_float(summary["minimum_retention"]) for summary in summaries)
        minimum_ties = [
            summary
            for summary in summaries
            if _as_float(summary["minimum_retention"]) == best_minimum
        ]
        best_reward = max(_as_float(summary["mean_router_reward"]) for summary in minimum_ties)
        finalists = [
            summary
            for summary in minimum_ties
            if _as_float(summary["mean_router_reward"]) == best_reward
        ]
        consensus_feasible = False
    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name)
        tie_audits = [
            _fit_tie_candidate(
                str(summary["name"]),
                matrix[str(summary["name"])],
                sources_dir,
                temporary,
            )
            for summary in finalists
        ]
        tie_audits.sort(
            key=lambda row: (
                _as_float(row["mean_p50_ms"]),
                _as_float(row["mean_artifact_bytes"]),
                order[str(row["name"])],
            )
        )
        chosen_tie = tie_audits[0]
        consensus_name = str(chosen_tie["name"])
        consensus_rows = matrix[consensus_name]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        consensus_reports: list[dict[str, object]] = []
        for seed, candidate in enumerate(consensus_rows):
            source = sources_dir / f"seed-{seed}-fit.json"
            rows = _read_rows(source)
            _, _, texts, weak, strong = _arrays(rows)
            artifact = artifact_dir / f"seed-{seed}.joblib"
            selection._fit_artifact(candidate, texts, weak, strong, artifact)
            consensus_report = {
                "schema": "swe-smith-broad-consensus-fit-v1",
                "seed": seed,
                "source_commit": reports[seed]["source_commit"],
                "fit_source_sha256": _sha256_file(source),
                "winner": candidate,
                "artifact_sha256": _sha256_file(artifact),
                "consensus_name": consensus_name,
                "heldout_outcomes_used": False,
                "target_outcomes_used": False,
            }
            report_path = artifact_dir / f"seed-{seed}.json"
            _write_json(report_path, consensus_report)
            consensus_reports.append(
                {
                    "seed": seed,
                    "report_sha256": _sha256_file(report_path),
                    "artifact_sha256": _sha256_file(artifact),
                }
            )
    chosen_summary = next(summary for summary in summaries if summary["name"] == consensus_name)
    result = {
        "schema": "swe-smith-broad-selection-lock-v1",
        "source_commit": reports[0]["source_commit"],
        "fit_report_sha256": [_sha256_file(reports_dir / f"seed-{seed}.json") for seed in range(5)],
        "winner_audit_sha256": [
            _sha256_file(audits_dir / f"seed-{seed}.json") for seed in range(5)
        ],
        "candidate_inventory_sha256": hashlib.sha256(
            ("\n".join(sorted(matrix)) + "\n").encode()
        ).hexdigest(),
        "candidate_count": len(matrix),
        "consensus_feasible": consensus_feasible,
        "consensus_name": consensus_name,
        "consensus": chosen_summary,
        "tie_finalists": tie_audits,
        "consensus_seed_artifacts": consensus_reports,
        "tie_break_decisions_per_seed": TIE_BREAK_DECISIONS,
        "outer_heldout_evaluated": False,
        "target_outcomes_used": False,
    }
    _write_json(args.output.resolve(), result)
    logger.info(
        "selection lock complete consensus=%s feasible=%s finalists=%d",
        consensus_name,
        consensus_feasible,
        len(finalists),
    )


def _parser() -> argparse.ArgumentParser:
    """Build the postfit command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--fit-source", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--artifact", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--internet-disabled", action="store_true")
    lock = subparsers.add_parser("lock")
    lock.add_argument("--sources-dir", type=Path, required=True)
    lock.add_argument("--reports-dir", type=Path, required=True)
    lock.add_argument("--audits-dir", type=Path, required=True)
    lock.add_argument("--artifact-dir", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Dispatch one leakage-safe postfit phase."""
    args = _parser().parse_args()
    if args.command == "audit":
        _audit(args)
    else:
        _lock(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
