"""Tests for the serve-side build manager: lifecycle, events, SSE framing, and failure paths."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from wmh.config import HarnessConfig
from wmh.config.card import load_card
from wmh.engine.reporting import BuildReporter
from wmh.serving.builds import (
    BuildFn,
    BuildManager,
    BuildRouteRequest,
    BuildStatus,
)


def _request(tmp_path: Path, name: str = "fresh") -> BuildRouteRequest:
    traces = tmp_path / "traces.jsonl"
    traces.write_text("{}\n", encoding="utf-8")
    return BuildRouteRequest(
        name=name,
        file=str(traces),
        title="Fresh model",
        description="Built from the test corpus.",
    )


def _ok_build_fn(config: HarnessConfig, *, file: str, root: str, reporter: BuildReporter) -> None:
    """A stand-in build: emits the real reporter sequence and writes a minimal artifact."""
    reporter.ingest_done(3, 12)
    reporter.split_done(2, 1)
    reporter.index_done(12)
    reporter.optimize_start(2)
    reporter.rollout(1, 2, 0.5)
    reporter.optimize_done(0.5, 1, 2)
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "config.toml").write_text("", encoding="utf-8")


def _failing_build_fn(
    config: HarnessConfig, *, file: str, root: str, reporter: BuildReporter
) -> None:
    reporter.ingest_done(1, 1)
    raise RuntimeError("provider exploded")


def _manager(
    tmp_path: Path,
    build_fn: BuildFn = _ok_build_fn,
    *,
    name_taken: Callable[[str], bool] | None = None,
) -> tuple[BuildManager, list[str]]:
    registered: list[str] = []
    manager = BuildManager(
        store_root=tmp_path / ".wmh",
        build_fn=build_fn,
        verify_fn=lambda config: None,  # no live provider ping in tests
        name_taken=name_taken,
        register=lambda name, model_dir: registered.append(name),
    )
    return manager, registered


def test_successful_build_reports_events_and_writes_card(tmp_path: Path) -> None:
    manager, registered = _manager(tmp_path)
    build_id = manager.start(_request(tmp_path))
    snapshot = manager.wait(build_id)

    assert snapshot.status is BuildStatus.SUCCEEDED
    assert registered == ["fresh"]
    kinds = [event.type for event in snapshot.events]
    assert kinds[0] == "ingest_done"
    assert kinds[-1] == "done"

    card = load_card(tmp_path / ".wmh" / "models" / "fresh")
    assert card is not None
    assert card.title == "Fresh model"
    assert card.corpus.traces == 3
    assert card.corpus.steps == 12
    assert card.built_at is not None


def test_failed_build_surfaces_error(tmp_path: Path) -> None:
    manager, registered = _manager(tmp_path, build_fn=_failing_build_fn)
    build_id = manager.start(_request(tmp_path))
    snapshot = manager.wait(build_id)

    assert snapshot.status is BuildStatus.FAILED
    assert snapshot.error is not None and "provider exploded" in snapshot.error
    assert registered == []
    assert snapshot.events[-1].type == "error"


def test_start_rejects_missing_traces_file(tmp_path: Path) -> None:
    request = _request(tmp_path)
    Path(request.file).unlink()
    manager, _ = _manager(tmp_path)
    with pytest.raises(FileNotFoundError, match="traces"):
        manager.start(request)


def test_start_rejects_existing_model_name(tmp_path: Path) -> None:
    model_dir = tmp_path / ".wmh" / "models" / "fresh"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("", encoding="utf-8")
    manager, _ = _manager(tmp_path)
    with pytest.raises(FileExistsError, match="fresh"):
        manager.start(_request(tmp_path))


def test_start_rejects_name_taken_in_another_root(tmp_path: Path) -> None:
    # name_taken spans all served roots, not just this manager's writable store.
    manager, _ = _manager(tmp_path, name_taken=lambda name: name == "fresh")
    with pytest.raises(FileExistsError, match="fresh"):
        manager.start(_request(tmp_path))


def test_concurrent_same_name_builds_rejected_before_persist(tmp_path: Path) -> None:
    # TOCTOU guard: the second start() is rejected while the first is still in-flight (nothing
    # on disk yet). A build_fn that blocks keeps the first "running" past the second's start().
    gate = threading.Event()

    def _blocking(config, *, file: str, root: str, reporter) -> None:  # noqa: ANN001
        gate.wait(5)
        _ok_build_fn(config, file=file, root=root, reporter=reporter)

    manager, _ = _manager(tmp_path, build_fn=_blocking)
    first = manager.start(_request(tmp_path))
    with pytest.raises(FileExistsError, match="building"):
        manager.start(_request(tmp_path))
    gate.set()
    assert manager.wait(first).status is BuildStatus.SUCCEEDED


def test_failed_build_removes_partial_artifact(tmp_path: Path) -> None:
    def _persist_then_fail(config, *, file: str, root: str, reporter) -> None:  # noqa: ANN001
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "config.toml").write_text("", encoding="utf-8")  # pipeline wrote artifact
        raise RuntimeError("register blew up after persist")

    manager, _ = _manager(tmp_path, build_fn=_persist_then_fail)
    manager.wait(manager.start(_request(tmp_path)))
    # The half-written model must be gone so retries don't 409 and serve doesn't load it.
    assert not (tmp_path / ".wmh" / "models" / "fresh").exists()


def test_failed_build_keeps_a_preexisting_model(tmp_path: Path) -> None:
    # A build whose name matches a model already on disk must be rejected — and even if it ran,
    # a failure must never delete that pre-existing model (data loss).
    model_dir = tmp_path / ".wmh" / "models" / "fresh"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("real model", encoding="utf-8")

    def _fail(config, *, file: str, root: str, reporter) -> None:  # noqa: ANN001
        raise RuntimeError("boom")

    manager, _ = _manager(tmp_path, build_fn=_fail)
    # Rejected up front because the name exists on disk (the writable store guard).
    with pytest.raises(FileExistsError):
        manager.start(_request(tmp_path))
    # And the real model is untouched.
    assert (model_dir / "config.toml").read_text(encoding="utf-8") == "real model"


def test_verify_failure_aborts_start_before_thread(tmp_path: Path) -> None:
    def _bad_verify(config) -> None:  # noqa: ANN001
        raise ValueError("bad creds")

    registered: list[str] = []
    manager = BuildManager(
        store_root=tmp_path / ".wmh",
        build_fn=_ok_build_fn,
        verify_fn=_bad_verify,
        register=lambda name, model_dir: registered.append(name),
    )
    with pytest.raises(ValueError, match="bad creds"):
        manager.start(_request(tmp_path))
    # Name reservation released on verify failure, so a corrected retry can reuse the name.
    manager._verify_fn = lambda config: None  # type: ignore[method-assign]
    assert manager.start(_request(tmp_path))


def test_wait_raises_on_timeout(tmp_path: Path) -> None:
    gate = threading.Event()

    def _blocking(config, *, file: str, root: str, reporter) -> None:  # noqa: ANN001
        gate.wait(5)

    manager, _ = _manager(tmp_path, build_fn=_blocking)
    build_id = manager.start(_request(tmp_path))
    with pytest.raises(TimeoutError):
        manager.wait(build_id, timeout=0.1)
    gate.set()


def test_unknown_build_id_raises(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    with pytest.raises(KeyError):
        manager.snapshot("nope")


def _parse_sse(frames: list[str]) -> list[tuple[int, dict]]:
    """Return (id, payload) for each `id: N\\ndata: {...}` frame, skipping keepalive comments."""
    out = []
    for frame in frames:
        if frame.startswith(":"):
            continue
        id_line, data_line = frame.strip().split("\n")
        out.append((int(id_line[len("id: ") :]), json.loads(data_line[len("data: ") :])))
    return out


def test_sse_stream_emits_indexed_events_and_terminates(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    build_id = manager.start(_request(tmp_path))
    manager.wait(build_id)

    events = _parse_sse(list(manager.sse_events(build_id)))
    ids = [i for i, _ in events]
    assert ids == list(range(len(ids)))  # contiguous ids from 0
    payloads = [p for _, p in events]
    assert payloads[0]["type"] == "ingest_done"
    assert payloads[0]["traces"] == 3
    assert payloads[-1]["type"] == "done"


def test_sse_resume_from_start_index_skips_replayed_events(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    build_id = manager.start(_request(tmp_path))
    manager.wait(build_id)

    full = _parse_sse(list(manager.sse_events(build_id)))
    # Reconnect after the first 3 events (Last-Event-ID = 2 -> start_index 3): no replay of 0..2.
    resumed = _parse_sse(list(manager.sse_events(build_id, start_index=3)))
    assert [i for i, _ in resumed] == [i for i, _ in full][3:]
