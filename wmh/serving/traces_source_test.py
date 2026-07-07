"""Tests for serve-side trace access: Hub URL, local resolution, scenarios, and downloads."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from wmh.config.card import TracesSource
from wmh.serving.traces_source import (
    TRACES_FILENAME,
    DownloadStatus,
    TracesDownloader,
    local_traces_path,
    resolve_url,
    scenarios_from_traces,
)


def _write_otel(path: Path) -> None:
    span_llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    path.write_text(json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8")


def test_resolve_url_dataset_and_model() -> None:
    ds = TracesSource(repo="org/wmh-tau", path="traces.otel.jsonl")
    assert (
        resolve_url(ds)
        == "https://huggingface.co/datasets/org/wmh-tau/resolve/main/traces.otel.jsonl"
    )
    mdl = TracesSource(repo="org/m", path="t.jsonl", revision="v1", kind="model")
    assert resolve_url(mdl) == "https://huggingface.co/org/m/resolve/v1/t.jsonl"


def test_local_traces_prefers_downloaded_over_sibling(tmp_path: Path) -> None:
    model_dir = tmp_path / "examples" / "tau" / "models" / "tau"
    model_dir.mkdir(parents=True)
    sibling = tmp_path / "examples" / "tau" / TRACES_FILENAME
    sibling.write_text("{}\n", encoding="utf-8")
    assert local_traces_path(model_dir) == sibling  # falls back to the example sibling
    downloaded = model_dir / TRACES_FILENAME
    downloaded.write_text("{}\n", encoding="utf-8")
    assert local_traces_path(model_dir) == downloaded  # a downloaded copy supersedes
    assert local_traces_path(tmp_path / "nowhere") is None


def test_scenarios_from_traces_groups_by_trace(tmp_path: Path) -> None:
    path = tmp_path / TRACES_FILENAME
    _write_otel(path)
    scenarios = scenarios_from_traces(path)
    assert len(scenarios) == 1
    steps = scenarios[0].steps
    assert steps[0].action.name == "get_user"
    assert steps[0].action_label.startswith("get_user")
    assert steps[0].observation == "found u1"
    assert steps[0].is_error is False


def _fake_fetch(chunks: list[bytes], total: int | None) -> Callable[..., None]:
    def fetch(url: str, dest: Path, on_progress) -> None:  # noqa: ANN001
        done = 0
        with dest.open("wb") as fh:
            for c in chunks:
                fh.write(c)
                done += len(c)
                on_progress(done, total)

    return fetch


def test_downloader_streams_to_dest_atomically(tmp_path: Path) -> None:
    dl = TracesDownloader(fetch=_fake_fetch([b"hello ", b"world"], 11))
    dest = tmp_path / "models" / "m" / TRACES_FILENAME
    dl.start("m", "http://x", dest)
    # Poll to completion (background thread).
    import time

    for _ in range(100):
        p = dl.progress("m")
        if p and p.status is not DownloadStatus.RUNNING:
            break
        time.sleep(0.02)
    p = dl.progress("m")
    assert p is not None and p.status is DownloadStatus.DONE
    assert p.downloaded == 11 and p.total == 11
    assert dest.read_bytes() == b"hello world"
    assert not dest.with_suffix(dest.suffix + ".part").exists()  # temp cleaned up


def test_downloader_reports_failure_and_leaves_no_partial(tmp_path: Path) -> None:
    def boom(url: str, dest: Path, on_progress) -> None:  # noqa: ANN001
        dest.write_bytes(b"partial")
        raise RuntimeError("network died")

    dl = TracesDownloader(fetch=boom)
    dest = tmp_path / "m" / TRACES_FILENAME
    dl.start("m", "http://x", dest)
    import time

    for _ in range(100):
        p = dl.progress("m")
        if p and p.status is not DownloadStatus.RUNNING:
            break
        time.sleep(0.02)
    p = dl.progress("m")
    assert p is not None and p.status is DownloadStatus.FAILED
    assert "network died" in (p.error or "")
    assert not dest.exists()  # the final path never gets a partial file


def test_progress_none_before_any_download(tmp_path: Path) -> None:
    assert TracesDownloader().progress("m") is None
