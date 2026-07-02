"""Serve-side world-model builds: kick off, watch (SSE), and register the finished model.

The website's build-your-own flow POSTs a build request; the manager runs the normal build
pipeline (`wmh.engine.build.build`) on a background thread, mirrors every `BuildReporter` event
into an in-memory log, and exposes that log both as a snapshot and as a Server-Sent-Events
stream. On success it writes the model's `card.json` and hands the fresh artifact to `register`
(the server adds it to the live serving set). One manager owns one writable store root.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from wmh.config import ArtifactPaths, HarnessConfig, WorldModelStore
from wmh.config.card import make_build_card, save_card
from wmh.engine.build import build as run_build
from wmh.engine.reporting import BuildReporter
from wmh.providers import get_provider, verify_all
from wmh.providers.base import EmbedderKind, ProviderKind
from wmh.tracking import MeteredProvider, RunTracker, classify_build_call, save_run

logger = logging.getLogger(__name__)

# How often the SSE follower wakes to re-check for events / client disconnect (seconds). A finite
# wait means a disconnected client's worker thread is freed on the next tick instead of parking
# forever on a build that has gone quiet (e.g. a hung provider call).
_SSE_POLL_SECONDS = 10.0


class BuildFn(Protocol):
    """The build pipeline the manager drives; injectable so tests never hit a provider."""

    def __call__(
        self, config: HarnessConfig, *, file: str, root: str, reporter: BuildReporter
    ) -> None: ...


class BuildRouteRequest(BaseModel):
    """Inputs for one serve-side build — the `wmh build` wizard's fields over HTTP."""

    name: str
    file: str  # server-local path to exported traces (use the uploads route from a browser)
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    provider: str = "bedrock"
    model: str = "us.anthropic.claude-opus-4-8"
    region: str | None = None
    gepa_budget: int = 50
    train_split: float = 0.8
    embed_dim: int = 512


class BuildStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BuildEvent(BaseModel):
    """One reporter callback, flattened for the event log ({"type": ..., **fields})."""

    type: str
    traces: int | None = None
    steps: int | None = None
    train: int | None = None
    test: int | None = None
    budget: int | None = None
    done: int | None = None
    score: float | None = None
    held_out_accuracy: float | None = None
    frontier_size: int | None = None
    rollouts: int | None = None
    error: str | None = None
    name: str | None = None


class BuildSnapshot(BaseModel):
    """Point-in-time view of one build (poll target and reconnect state for the SSE stream)."""

    build_id: str
    name: str
    status: BuildStatus
    error: str | None = None
    events: list[BuildEvent] = Field(default_factory=list)


class _BuildState:
    """Mutable record of one build; `condition` signals appended events / terminal status."""

    def __init__(self, build_id: str, name: str) -> None:
        self.build_id = build_id
        self.name = name
        self.status = BuildStatus.RUNNING
        self.error: str | None = None
        self.events: list[BuildEvent] = []
        self.condition = threading.Condition()

    def append(self, event: BuildEvent) -> None:
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def finish(self, status: BuildStatus, error: str | None = None) -> None:
        with self.condition:
            self.status = status
            self.error = error
            self.condition.notify_all()

    def snapshot(self) -> BuildSnapshot:
        with self.condition:
            return BuildSnapshot(
                build_id=self.build_id,
                name=self.name,
                status=self.status,
                error=self.error,
                events=list(self.events),
            )


class _RecordingReporter(BuildReporter):
    """BuildReporter that mirrors every pipeline event into the build's event log."""

    def __init__(self, state: _BuildState) -> None:
        self._state = state
        self.ingested_traces = 0
        self.ingested_steps = 0

    def ingest_done(self, traces: int, steps: int) -> None:
        self.ingested_traces, self.ingested_steps = traces, steps
        self._state.append(BuildEvent(type="ingest_done", traces=traces, steps=steps))

    def split_done(self, train: int, test: int) -> None:
        self._state.append(BuildEvent(type="split_done", train=train, test=test))

    def index_done(self, steps: int) -> None:
        self._state.append(BuildEvent(type="index_done", steps=steps))

    def optimize_start(self, budget: int) -> None:
        self._state.append(BuildEvent(type="optimize_start", budget=budget))

    def rollout(self, done: int, budget: int, score: float | None) -> None:
        self._state.append(BuildEvent(type="rollout", done=done, budget=budget, score=score))

    def optimize_done(self, held_out_accuracy: float, frontier_size: int, rollouts: int) -> None:
        self._state.append(
            BuildEvent(
                type="optimize_done",
                held_out_accuracy=held_out_accuracy,
                frontier_size=frontier_size,
                rollouts=rollouts,
            )
        )


def _default_build_fn(
    config: HarnessConfig, *, file: str, root: str, reporter: BuildReporter
) -> None:
    """Run the pipeline metered exactly like `wmh build`, persisting a run record.

    Wrapping the serve provider in `MeteredProvider` captures GEPA + judge cost/tokens (the same
    boundary the CLI meters at), and `save_run` writes `runs/<id>.json` so serve-side builds are
    accounted for identically to CLI builds.
    """
    tracker = RunTracker(run_id=uuid.uuid4().hex, kind="build")
    metered = MeteredProvider(
        get_provider(config.serve_provider_config()), tracker, classify=classify_build_call
    )
    with tracker.timed():
        run_build(config, file=file, root=root, serve_provider=metered, reporter=reporter)
    save_run(tracker.record_summary(), ArtifactPaths(root).runs)


def _default_verify(config: HarnessConfig) -> None:
    """Ping the serve provider before spending rollouts; raise on failure (like the CLI guard)."""
    results = verify_all([config.serve_provider_config()])
    bad = [r for r in results if not r.ok]
    if bad:
        detail = "; ".join(f"{r.kind.value} ({r.model}): {r.detail}" for r in bad)
        raise ValueError(f"provider verification failed, not starting build: {detail}")


class BuildManager:
    """Runs builds on background threads against one writable store root."""

    def __init__(
        self,
        store_root: str | Path,
        *,
        build_fn: BuildFn | None = None,
        verify_fn: Callable[[HarnessConfig], None] | None = None,
        name_taken: Callable[[str], bool] | None = None,
        register: Callable[[str, Path], None],
    ) -> None:
        self._store = WorldModelStore(store_root)
        self._build_fn: BuildFn = build_fn or _default_build_fn
        self._verify_fn = verify_fn or _default_verify
        # Whether a name already belongs to the live serving set (across ALL roots, supplied by
        # the server); falls back to this store's own disk when serving a single root.
        self._name_taken = name_taken or self._store.exists
        self._register = register
        self._builds: dict[str, _BuildState] = {}
        # Names of builds that are launched but not yet on disk — closes the TOCTOU where two
        # same-name builds both pass the existence check before either writes config.toml.
        self._reserved: set[str] = set()
        self._lock = threading.Lock()

    @property
    def uploads_dir(self) -> Path:
        return self._store.root / "uploads"

    def start(self, request: BuildRouteRequest) -> str:
        """Validate and launch a build, returning its id. Raises before spending anything."""
        if not Path(request.file).is_file():
            raise FileNotFoundError(
                f"no traces file at {request.file!r}; upload one via "
                "POST /world_models/builds/uploads or pass a server-local path"
            )
        config = HarnessConfig.for_build(
            serve_provider=ProviderKind(request.provider),
            serve_model=request.model,
            region=request.region,
            embed_provider=EmbedderKind.HASHING,
            embed_model=None,
            embed_dim=request.embed_dim,
            gepa_budget=request.gepa_budget,
            train_split=request.train_split,
        )
        # Reserve the name atomically against in-flight builds, the served set (`name_taken`), AND
        # the writable root's disk — the last catches a model built earlier but not currently
        # served (via `--name`), which would otherwise be silently overwritten.
        with self._lock:
            if (
                self._name_taken(request.name)
                or self._store.exists(request.name)
                or request.name in self._reserved
            ):
                raise FileExistsError(
                    f"world model {request.name!r} already exists or is building; pick another name"
                )
            self._reserved.add(request.name)
        # Fail fast on bad creds BEFORE launching the thread, so the client gets a real error
        # instead of a build that "succeeds" with a useless held-out-0.0 model.
        try:
            self._verify_fn(config)
        except Exception:
            with self._lock:
                self._reserved.discard(request.name)
            raise
        state = _BuildState(uuid.uuid4().hex, request.name)
        with self._lock:
            self._builds[state.build_id] = state
        thread = threading.Thread(
            target=self._run,
            args=(state, request, config),
            name=f"wmh-build-{request.name}",
            daemon=True,  # never block `wmh serve` shutdown on an in-flight build
        )
        thread.start()
        return state.build_id

    def _run(self, state: _BuildState, request: BuildRouteRequest, config: HarnessConfig) -> None:
        reporter = _RecordingReporter(state)
        model_dir = self._store.model_dir(request.name)
        # If the dir already exists, this build did not create it — never delete it on failure
        # (it could be a real, previously-built model that just wasn't in the served set).
        preexisting = model_dir.exists()
        try:
            # Only the pipeline itself can leave a partial/broken artifact worth cleaning up.
            self._build_fn(config, file=request.file, root=str(model_dir), reporter=reporter)
        except Exception as exc:  # noqa: BLE001 - report any failure to the client, never a dead silent thread
            # Remove the partial artifact so a failed build doesn't leave a model that `exists()`
            # then treats as real (bricking retries with 409 and serving a broken model) — but
            # only if this build created the dir, never a pre-existing model.
            if not preexisting:
                shutil.rmtree(model_dir, ignore_errors=True)
            state.append(BuildEvent(type="error", error=str(exc)))
            state.finish(BuildStatus.FAILED, error=str(exc))
            return
        finally:
            with self._lock:
                self._reserved.discard(request.name)
        # The model artifact is complete on disk. Writing the card and joining the live serving
        # set are best-effort: a failure here logs a warning but must NOT delete the finished
        # build (the card is additive metadata; an un-registered model still serves on restart).
        try:
            save_card(
                make_build_card(
                    name=request.name,
                    provider=request.provider,
                    model_id=request.model,
                    traces=reporter.ingested_traces,
                    steps=reporter.ingested_steps,
                    built_at=datetime.now(UTC).isoformat(),
                    source=Path(request.file).name,
                    title=request.title,
                    description=request.description,
                    tags=request.tags,
                ),
                model_dir,
            )
            self._register(request.name, model_dir)
        except Exception:  # noqa: BLE001 - never destroy a completed build over card/register
            logger.warning("post-build card/register failed for %s", request.name, exc_info=True)
        state.append(BuildEvent(type="done", name=request.name))
        state.finish(BuildStatus.SUCCEEDED)

    def _state(self, build_id: str) -> _BuildState:
        with self._lock:
            return self._builds[build_id]  # KeyError -> route maps to 404

    def snapshot(self, build_id: str) -> BuildSnapshot:
        return self._state(build_id).snapshot()

    def wait(self, build_id: str, timeout: float = 60.0) -> BuildSnapshot:
        """Block until the build reaches a terminal status, or raise TimeoutError.

        Used by tests and CLI callers; a timeout is an explicit error rather than a silently
        still-RUNNING snapshot that a caller might misread as failed.
        """
        state = self._state(build_id)
        with state.condition:
            reached = state.condition.wait_for(
                lambda: state.status is not BuildStatus.RUNNING, timeout
            )
        if not reached:
            raise TimeoutError(f"build {build_id} still running after {timeout}s")
        return state.snapshot()

    def sse_events(self, build_id: str, start_index: int = 0) -> Iterator[str]:
        """Yield build events as SSE frames from `start_index`, ending when the build finishes.

        Each frame carries `id: <index>` so a reconnecting `EventSource` resumes from
        `Last-Event-ID` instead of replaying (and duplicating) the whole log. Follows live
        events with a bounded wait so a disconnected client's worker thread is freed on the next
        poll (a heartbeat comment is emitted on each idle tick; the next send after a disconnect
        raises and closes the generator) rather than parking forever on a quiet build.
        """
        state = self._state(build_id)
        cursor = start_index
        while True:
            with state.condition:
                got = state.condition.wait_for(
                    lambda seen=cursor: (
                        len(state.events) > seen or state.status is not BuildStatus.RUNNING
                    ),
                    _SSE_POLL_SECONDS,
                )
                start = cursor
                fresh = state.events[cursor:]
                cursor = len(state.events)
                finished = state.status is not BuildStatus.RUNNING
            if not got and not fresh:
                yield ": keepalive\n\n"  # detects client disconnect; frees the worker on next tick
                continue
            for offset, event in enumerate(fresh):
                payload = event.model_dump(exclude_none=True)
                yield f"id: {start + offset}\ndata: {json.dumps(payload)}\n\n"
            if finished and cursor == len(state.events):
                return
