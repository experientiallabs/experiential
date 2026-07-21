"""Durable, self-contained evidence for one completed population optimization."""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import JsonObject
from wmh.harness.archive_io import (
    copy_score_artifacts,
    publish_json_manifest,
    relative_path,
    write_source_tree,
    write_text,
)
from wmh.harness.live_session import SessionEvent
from wmh.harness.population import PopulationOptimizationResult
from wmh.harness.runtime import TokenUsage

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
    population_entries: list[JsonObject] = []
    population_ids: set[str] = set()
    for index, evaluated in enumerate(result.population):
        if evaluated.candidate_id in population_ids:
            raise ValueError("population archive contains duplicate candidate identities")
        population_ids.add(evaluated.candidate_id)
        candidate_dir = root / "population" / f"{index:04d}"
        source_dir = candidate_dir / "source"
        write_source_tree(source_dir, evaluated.source)
        report_path = candidate_dir / "score.json"
        write_text(report_path, evaluated.score.report.model_dump_json(indent=2))
        artifacts_dir = candidate_dir / "artifacts"
        copy_score_artifacts(artifacts_dir, evaluated.score)
        population_entries.append(
            {
                "index": index,
                "candidate_id": evaluated.candidate_id,
                "document_hash": evaluated.candidate.doc_hash,
                "source_tree_hash": evaluated.source.tree_hash,
                "source_path": relative_path(root, source_dir),
                "score_path": relative_path(root, report_path),
                "artifacts_path": relative_path(root, artifacts_dir),
            }
        )

    iteration_entries: list[JsonObject] = []
    for expected_index, iteration in enumerate(result.iterations, 1):
        if iteration.index != expected_index:
            raise ValueError("population archive iterations must be contiguous and one-indexed")
        iteration_dir = root / "iterations" / f"{iteration.index:04d}"
        events_path = iteration_dir / "events.json"
        entry: JsonObject = {}
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
                    "events_path": relative_path(root, events_path),
                }
            )
            if iteration.error.source is not None:
                source_dir = iteration_dir / "source"
                write_source_tree(source_dir, iteration.error.source)
                entry["source_tree_hash"] = iteration.error.source.tree_hash
                entry["source_path"] = relative_path(root, source_dir)
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
                    "events_path": relative_path(root, events_path),
                }
            )
        _write_events(events_path, events)
        iteration_entries.append(entry)

    manifest_path = root / "manifest.json"
    manifest: JsonObject = {
        "schema_version": _SCHEMA_VERSION,
        "score_request": request.model_dump(mode="json"),
        "best_candidate_id": result.best.candidate_id,
        "best_score": result.best_score,
        "population": population_entries,
        "iterations": iteration_entries,
    }
    return publish_json_manifest(manifest_path, manifest)


def _write_events(path: Path, events: tuple[SessionEvent, ...]) -> None:
    payload = [{"kind": event.kind, "payload": event.payload} for event in events]
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _usage(usage: TokenUsage | None) -> JsonObject | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "calls": usage.calls,
    }
