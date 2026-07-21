"""Durable, self-contained evidence for one completed population optimization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wmh.harness.live_session import SessionEvent
from wmh.harness.population import PopulationOptimizationResult
from wmh.harness.runtime import TokenUsage
from wmh.harness.source_tree import HarnessSourceTree

_SCHEMA_VERSION = 1


def write_population_archive(
    destination: str | Path,
    result: PopulationOptimizationResult,
) -> Path:
    """Write verified reports, artifacts, sources, and proposal outcomes once.

    The destination must not exist. ``manifest.json`` is written last, so its presence is the
    completion marker; an interrupted copy leaves inspectable evidence but cannot be mistaken for
    a complete archive.
    """
    root = Path(destination)
    if root.exists():
        raise FileExistsError(f"population archive destination already exists: {root}")
    if not result.population:
        raise ValueError("population archive requires an evaluated seed")
    if result.best not in result.population:
        raise ValueError("population archive winner is not in its evaluated population")

    request = result.population[0].score.report.request
    if any(item.score.report.request != request for item in result.population[1:]):
        raise ValueError("population archive candidates use different score requests")

    root.mkdir(parents=True)
    population_entries: list[dict[str, object]] = []
    population_ids: set[str] = set()
    for index, evaluated in enumerate(result.population):
        if evaluated.candidate_id in population_ids:
            raise ValueError("population archive contains duplicate candidate identities")
        population_ids.add(evaluated.candidate_id)
        candidate_dir = root / "population" / f"{index:04d}"
        source_dir = candidate_dir / "source"
        _write_source_tree(source_dir, evaluated.source)
        report_path = candidate_dir / "score.json"
        _write_text(report_path, evaluated.score.report.model_dump_json(indent=2))
        artifacts_dir = candidate_dir / "artifacts"
        for artifact in evaluated.score.report.artifacts:
            content = evaluated.score.artifacts.read_bytes(artifact.path)
            if len(content) != artifact.size_bytes:
                raise ValueError(f"artifact {artifact.path!r} size differs from its score manifest")
            digest = "sha256:" + hashlib.sha256(content).hexdigest()
            if digest != artifact.content_hash:
                raise ValueError(
                    f"artifact {artifact.path!r} content differs from its score manifest"
                )
            target = artifacts_dir / artifact.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        population_entries.append(
            {
                "index": index,
                "candidate_id": evaluated.candidate_id,
                "document_hash": evaluated.candidate.doc_hash,
                "source_tree_hash": evaluated.source.tree_hash,
                "source_path": _relative(root, source_dir),
                "score_path": _relative(root, report_path),
                "artifacts_path": _relative(root, artifacts_dir),
            }
        )

    iteration_entries: list[dict[str, object]] = []
    for expected_index, iteration in enumerate(result.iterations, 1):
        if iteration.index != expected_index:
            raise ValueError("population archive iterations must be contiguous and one-indexed")
        iteration_dir = root / "iterations" / f"{iteration.index:04d}"
        events_path = iteration_dir / "events.json"
        entry: dict[str, object] = {}
        if iteration.error is not None:
            events = iteration.error.events
            usage = iteration.error.worker_usage
            entry.update(
                {
                    "index": iteration.index,
                    "candidate_id": iteration.error.candidate_id,
                    "outcome": "invalid",
                    "error": str(iteration.error),
                    "worker_usage": _usage(usage),
                    "events_path": _relative(root, events_path),
                }
            )
            if iteration.error.source is not None:
                source_dir = iteration_dir / "source"
                _write_source_tree(source_dir, iteration.error.source)
                entry["source_tree_hash"] = iteration.error.source.tree_hash
                entry["source_path"] = _relative(root, source_dir)
        else:
            assert iteration.proposal is not None
            assert iteration.evaluation is not None
            events = iteration.proposal.events
            entry.update(
                {
                    "index": iteration.index,
                    "candidate_id": iteration.proposal.candidate_id,
                    "outcome": "scored",
                    "document_hash": iteration.proposal.candidate.doc_hash,
                    "source_tree_hash": iteration.proposal.source.tree_hash,
                    "worker_usage": _usage(iteration.proposal.worker_usage),
                    "events_path": _relative(root, events_path),
                }
            )
        _write_events(events_path, events)
        iteration_entries.append(entry)

    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "score_request": request.model_dump(mode="json"),
        "best_candidate_id": result.best.candidate_id,
        "best_score": result.best_score,
        "population": population_entries,
        "iterations": iteration_entries,
    }
    manifest_temp = manifest_path.with_name(f"{manifest_path.name}.tmp")
    _write_text(manifest_temp, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_temp.replace(manifest_path)
    return manifest_path


def _write_source_tree(destination: Path, source: HarnessSourceTree) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.files:
        target = destination / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")


def _write_events(path: Path, events: tuple[SessionEvent, ...]) -> None:
    payload = [{"kind": event.kind, "payload": event.payload} for event in events]
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _usage(usage: TokenUsage | None) -> dict[str, int] | None:
    return usage.model_dump(mode="json") if usage is not None else None


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
