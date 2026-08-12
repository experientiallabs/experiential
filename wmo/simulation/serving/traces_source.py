"""Serve-side trace access: read a model's local traces, or fetch them from the Hugging Face Hub.

The raw trace corpus (`traces.otel.jsonl`) is large and need not be committed. When it is present
locally it is used directly; otherwise, if the model's card declares a `traces_hf` source, the
backend streams it from the Hub's public resolve URL (no auth, no client-side Hub API) into the
model directory, reporting byte progress the website can poll. A local copy always supersedes the
Hub. Recorded traces are grouped by task into replayable scenarios for the Explore-traces tab.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from wmo.common.config.card import TracesSource
from wmo.common.core.artifacts import JsonObject
from wmo.simulation.ingest.otlp import OtlpTraceFormatError, load_otlp_file

TRACES_FILENAME = "traces.otel.jsonl"


def resolve_url(source: TracesSource) -> str:
    """The public Hub resolve URL for a traces source (works unauthenticated for public repos)."""
    prefix = "datasets/" if source.kind == "dataset" else ""
    return f"https://huggingface.co/{prefix}{source.repo}/resolve/{source.revision}/{source.path}"


def local_traces_path(model_dir: Path) -> Path | None:
    """The traces file for a model, if present: a downloaded copy, else the example sibling."""
    downloaded = model_dir / TRACES_FILENAME
    if downloaded.is_file():
        return downloaded
    # <task>/traces.otel.jsonl sits two levels above <task>/models/<name>/.
    sibling = model_dir.parent.parent / TRACES_FILENAME
    if sibling.is_file():
        return sibling
    return None


class TraceSummaryStep(BaseModel):
    """One canonical normalized span rendered for local trace inspection."""

    name: str
    detail: str
    is_error: bool


class TraceSummary(BaseModel):
    """One canonical trace rendered for bounded local inspection."""

    id: str
    label: str
    task: str | None
    steps: list[TraceSummaryStep]


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def trace_summaries_from_otlp(
    path: Path, *, max_traces: int = 6, max_steps: int = 10
) -> list[TraceSummary]:
    """Read a local OTLP export once and render bounded canonical trace summaries.

    This inspect-only surface is intentionally separate from ``wmo build``. It invokes the same
    strict OTLP normalizer directly and never reconstructs the removed scenario contracts.

    Args:
        path: Local OTLP JSON or JSONL export to inspect.
        max_traces: Maximum trace summaries to return.
        max_steps: Maximum canonical spans included per trace summary.

    Returns:
        Bounded canonical trace summaries in deterministic source-normalized order.

    Raises:
        ValueError: The selected local evidence cannot be normalized as OTLP.
    """
    try:
        traces = load_otlp_file(path).traces
    except OtlpTraceFormatError as exc:
        raise ValueError(f"cannot normalize local OTLP trace evidence {path}: {exc}") from exc
    out: list[TraceSummary] = []
    for index, trace in enumerate(traces):
        steps: list[TraceSummaryStep] = []
        for span in trace.spans:
            steps.append(
                TraceSummaryStep(
                    name=span.name,
                    detail=_span_detail(span.attributes),
                    is_error=span.failure is not None,
                )
            )
            if len(steps) >= max_steps:
                break
        if not steps:
            continue
        out.append(
            TraceSummary(
                id=trace.trace_id,
                label=_trace_label(trace.task, index),
                task=trace.task,
                steps=steps,
            )
        )
        if len(out) >= max_traces:
            break
    return out


def _span_detail(attributes: JsonObject) -> str:
    """Extract a short inspectable detail from a canonical span's public attributes."""
    for key in (
        "gen_ai.tool.message",
        "gen_ai.tool.result",
        "gen_ai.completion",
        "gen_ai.response.text",
        "error.message",
    ):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return _clip(value, 160)
    return _clip(json.dumps(attributes, sort_keys=True), 160) if attributes else ""


def _trace_label(task: str, index: int) -> str:
    """Render a compact canonical trace task label for the local inspection response."""
    if not task:
        return f"Trace {index + 1}"
    try:
        parsed = json.loads(task)
        text = parsed.get("reason_for_call") or parsed.get("task_instructions") or task
    except (json.JSONDecodeError, AttributeError):
        text = task
    return _clip(text, 68)


class DownloadStatus(StrEnum):
    """Lifecycle of one background trace download."""

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class DownloadProgress(BaseModel):
    """How far a trace download has got, and why it stopped when it failed."""

    status: DownloadStatus
    downloaded: int = 0
    total: int | None = None  # bytes, when the server reports Content-Length
    error: str | None = None


class _DownloadState:
    def __init__(self) -> None:
        self.progress = DownloadProgress(status=DownloadStatus.RUNNING)
        self.lock = threading.Lock()


class TracesDownloader:
    """Runs Hub trace downloads on background threads, one in flight per model name."""

    def __init__(
        self, *, fetch: Callable[[str, Path, Callable[[int, int | None], None]], None] | None = None
    ) -> None:
        self._fetch = fetch or _stream_to_file
        self._states: dict[str, _DownloadState] = {}
        self._lock = threading.Lock()

    def start(self, name: str, url: str, dest: Path) -> None:
        with self._lock:
            existing = self._states.get(name)
            if existing and existing.progress.status is DownloadStatus.RUNNING:
                return  # already downloading; the client just polls
            self._states[name] = _DownloadState()
        thread = threading.Thread(
            target=self._run, args=(name, url, dest), name=f"wmo-traces-{name}", daemon=True
        )
        thread.start()

    def _run(self, name: str, url: str, dest: Path) -> None:
        state = self._states[name]

        def on_progress(downloaded: int, total: int | None) -> None:
            with state.lock:
                state.progress.downloaded = downloaded
                state.progress.total = total

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            self._fetch(url, tmp, on_progress)
            tmp.replace(dest)  # atomic: a partial download never looks complete
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            tmp.unlink(missing_ok=True)
            with state.lock:
                state.progress.status = DownloadStatus.FAILED
                state.progress.error = str(exc)
            return
        with state.lock:
            state.progress.status = DownloadStatus.DONE

    def progress(self, name: str) -> DownloadProgress | None:
        state = self._states.get(name)
        if state is None:
            return None
        with state.lock:
            return state.progress.model_copy()


def _stream_to_file(url: str, dest: Path, on_progress: Callable[[int, int | None], None]) -> None:
    """Stream a public Hub file to disk in chunks, following the CDN redirect."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        raw = resp.headers.get("content-length")
        total = int(raw) if raw and raw.isdigit() else None
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                on_progress(downloaded, total)


class TracesResponse(BaseModel):
    """What the Explore-traces tab needs: local scenarios if present, else a Hub download offer."""

    source: str  # "local" | "hub" | "none"
    downloadable: bool
    scenarios: list[TraceSummary] = Field(default_factory=list)
    download: DownloadProgress | None = None
