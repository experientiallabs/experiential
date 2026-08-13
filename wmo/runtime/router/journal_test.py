"""Adversarial tests for the routed-interaction journal and idempotency service."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import stat
import threading
import time
from datetime import UTC, datetime, timedelta
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    stable_id,
)
from wmo.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    RoutedCandidateSnapshot,
)
from wmo.common.project import ProjectPaths
from wmo.common.routing import RouterFeatureExtractor, RoutingDecision
from wmo.runtime.router.journal import (
    JournaledRouterRuntime,
    RuntimeAcceptance,
    RuntimeAcceptedEvent,
    RuntimeAttemptFailedEvent,
    RuntimeIdempotencyConflictError,
    RuntimeInteractionInProgressError,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    _completed_event,
    _interaction_identity,
)
from wmo.runtime.router.runtime import RoutedModelResponse, RouterRuntime

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 13, tzinfo=UTC)


def _snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        provider="openai",
        model_id="gpt-test",
        capabilities_sha256=_DIGEST,
        connection_sha256="b" * 64,
    )


def _request(content: str = "Help me") -> ModelRequest:
    return ModelRequest(messages=(ModelMessage(role="user", content=content),))


def _response() -> ModelResponse:
    return ModelResponse(
        output=AssistantAction(content="Done"),
        model=_snapshot(),
        economics=OperationEconomics(),
    )


def _decision(lineage_id: str, request: ModelRequest | None = None) -> RoutingDecision:
    episode_sha256 = hashlib.sha256(lineage_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    feature = RouterFeatureExtractor().from_request(request or _request())
    material = {
        "policy_id": "router-policy-a",
        "policy_sha256": _DIGEST,
        "request_sha256": hashlib.sha256(
            feature.encode("utf-8"), usedforsecurity=False
        ).hexdigest(),
        "episode_id_sha256": episode_sha256,
        "selected_alias": "candidate-a",
        "baseline_alias": "candidate-a",
        "neighbor_count": 8,
        "paired_count": 8,
        "best_similarity": 1.0,
        "estimated_quality_difference": None,
        "uncertainty": None,
        "fallback_reason": None,
    }
    return RoutingDecision(
        decision_id=stable_id("routing-decision", material),
        **material,
    )


class _FakeRuntime:
    """Small deterministic runtime with a shareable target-call counter."""

    def __init__(
        self,
        *,
        counter: Synchronized | None = None,
        delay: float = 0.0,
        fail_once: bool = False,
    ) -> None:
        self.policy = SimpleNamespace(
            candidates=(RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot()),)
        )
        self.counter = counter
        self.delay = delay
        self.fail_once = fail_once
        self.select_calls = 0
        self.complete_calls = 0
        self.decisions: list[RoutingDecision] = []

    def select(self, request: ModelRequest, *, episode_id: str | None = None) -> RoutingDecision:
        del request
        assert episode_id is not None
        self.select_calls += 1
        decision = _decision(episode_id)
        self.decisions.append(decision)
        return decision

    def complete(
        self,
        request: ModelRequest,
        *,
        episode_id: str | None = None,
        decision: RoutingDecision | None = None,
        provider_idempotency_key: str | None = None,
    ) -> RoutedModelResponse:
        del request, episode_id
        assert decision is not None
        assert provider_idempotency_key is not None
        self.complete_calls += 1
        if self.counter is not None:
            with self.counter.get_lock():
                self.counter.value += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail_once:
            self.fail_once = False
            raise TimeoutError("secret provider detail")
        return RoutedModelResponse(decision=decision, response=_response())


def _service(
    root: Path,
    runtime: _FakeRuntime,
    *,
    wait_timeout_seconds: float = 2.0,
    stale_after_seconds: float = 30.0,
) -> JournaledRouterRuntime:
    journal = RuntimeInteractionJournal(ProjectPaths(root=root, project_id="support-agent"))
    return JournaledRouterRuntime(
        cast(RouterRuntime, runtime),
        journal,
        wait_timeout_seconds=wait_timeout_seconds,
        stale_after_seconds=stale_after_seconds,
    )


def test_completed_interaction_replays_after_restart_without_provider_or_selection(
    tmp_path: Path,
) -> None:
    """A completed key returns its canonical response without any new runtime work."""
    first_runtime = _FakeRuntime()
    first = _service(tmp_path / ".wmo", first_runtime)
    raw_key = "sk-this-is-a-fake-key-material-123456789"
    initial = first.complete(_request(), idempotency_key=raw_key)

    restarted_runtime = _FakeRuntime()
    restarted = _service(tmp_path / ".wmo", restarted_runtime)
    replayed = restarted.complete(_request(), idempotency_key=raw_key)

    assert replayed == initial
    assert first_runtime.select_calls == first_runtime.complete_calls == 1
    assert restarted_runtime.select_calls == restarted_runtime.complete_calls == 0
    payload = restarted.journal.path.read_text(encoding="utf-8")
    assert raw_key not in payload
    assert [event.event for event in restarted.journal.read_events()] == [
        "accepted",
        "completed",
    ]


def test_same_key_with_different_request_or_lineage_conflicts(tmp_path: Path) -> None:
    """One key cannot silently change its canonical request or conversation lineage."""
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="same-key", conversation_id="conversation-a")

    with pytest.raises(RuntimeIdempotencyConflictError):
        service.complete(
            _request("Different"),
            idempotency_key="same-key",
            conversation_id="conversation-a",
        )
    with pytest.raises(RuntimeIdempotencyConflictError):
        service.complete(
            _request(),
            idempotency_key="same-key",
            conversation_id="conversation-b",
        )


def test_provider_failure_has_no_target_and_retry_reuses_pinned_decision(tmp_path: Path) -> None:
    """Retryable failures are durable and never trigger a second routing selection."""
    runtime = _FakeRuntime(fail_once=True)
    service = _service(tmp_path / ".wmo", runtime)

    with pytest.raises(TimeoutError, match="secret provider detail"):
        service.complete(_request(), idempotency_key="retry-key")
    events = service.journal.read_events()
    assert [event.event for event in events] == ["accepted", "attempt_failed"]
    failed = cast(RuntimeAttemptFailedEvent, events[-1])
    assert failed.retryable
    assert failed.failure.message == "routed model provider attempt failed"
    assert "response" not in failed.model_dump(mode="json")
    assert "secret provider detail" not in service.journal.path.read_text(encoding="utf-8")

    result = service.complete(_request(), idempotency_key="retry-key")
    retried = service.journal.read_events()
    assert result.response == _response()
    assert [event.event for event in retried] == [
        "accepted",
        "attempt_failed",
        "accepted",
        "completed",
    ]
    assert runtime.select_calls == 1
    assert (
        cast(RuntimeAcceptedEvent, retried[0]).decision
        == cast(RuntimeAcceptedEvent, retried[2]).decision
    )


def test_two_threads_with_one_key_dispatch_target_once(tmp_path: Path) -> None:
    """The journal serializes same-key threads while provider work runs outside the lock."""
    runtime = _FakeRuntime(delay=0.15)
    service = _service(tmp_path / ".wmo", runtime)
    barrier = threading.Barrier(3)
    results: list[RoutedModelResponse] = []

    def call() -> None:
        barrier.wait()
        results.append(service.complete(_request(), idempotency_key="thread-key"))

    threads = (threading.Thread(target=call), threading.Thread(target=call))
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert results[0] == results[1]
    assert runtime.complete_calls == 1


def _process_call(root: str, counter: Synchronized) -> None:
    """Complete one shared-key request in a separate process."""
    runtime = _FakeRuntime(counter=counter, delay=0.2)
    _service(Path(root), runtime, wait_timeout_seconds=3.0).complete(
        _request(), idempotency_key="process-key"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="cross-process lock probe needs fork")
def test_two_processes_with_one_key_dispatch_target_once(tmp_path: Path) -> None:
    """Independent processes observe one accepted attempt and one completed response."""
    context = multiprocessing.get_context("fork")
    counter = context.Value("i", 0)
    root = tmp_path / ".wmo"
    processes = (
        context.Process(target=_process_call, args=(str(root), counter)),
        context.Process(target=_process_call, args=(str(root), counter)),
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert counter.value == 1


def test_torn_tail_is_ignored_but_interior_corruption_fails_closed(tmp_path: Path) -> None:
    """Only a non-newline-terminated final record may be treated as a crash tear."""
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="tail-key")
    path = service.journal.path
    clean = path.read_bytes()
    path.write_bytes(clean + b'{"event":"accepted"')

    assert len(service.journal.read_events()) == 2
    service.complete(_request(), idempotency_key="after-torn-key")
    assert len(service.journal.read_events()) == 4

    repaired = path.read_bytes()
    path.write_bytes(repaired.splitlines(keepends=True)[0] + b"not-json\n" + repaired)
    with pytest.raises(RuntimeJournalError, match="invalid interior line"):
        service.journal.read_events()


def test_duplicate_transition_and_digest_drift_fail_closed(tmp_path: Path) -> None:
    """A valid JSON rewrite cannot bypass event IDs, digests, ordinals, or transitions."""
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="corrupt-key")
    path = service.journal.path
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(f"{lines[0]}\n{lines[0]}\n", encoding="utf-8")
    with pytest.raises(RuntimeJournalError, match="ordinals|event ID|transition"):
        service.journal.read_events()

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    corrupted = lines[1].replace('"response_sha256":"', '"response_sha256":"f')
    path.write_text(f"{lines[0]}\n{corrupted}\n", encoding="utf-8")
    with pytest.raises(RuntimeJournalError, match="invalid interior line"):
        service.journal.read_events()


def test_stale_attempt_gets_failure_then_retry_with_original_route(tmp_path: Path) -> None:
    """A stale live attempt is closed explicitly before a pinned retry is dispatched."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "stale-key", request, None)
    decision = _decision(identity.lineage_id)
    acceptance = RuntimeAcceptance(
        decision=decision,
        selected_alias=decision.selected_alias,
        selected_model=_snapshot(),
        policy_input=ArtifactInput(artifact_id=decision.policy_id, sha256=decision.policy_sha256),
    )
    first = journal.claim(
        identity,
        acceptance,
        now=_TIME,
        stale_after=timedelta(seconds=1),
    )
    retry = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )

    assert first.status == retry.status == "dispatch"
    assert retry.accepted is not None
    assert retry.accepted.attempt_ordinal == 2
    assert [event.event for event in journal.read_events()] == [
        "accepted",
        "attempt_failed",
        "accepted",
    ]


def test_original_provider_success_wins_after_concurrent_stale_takeover(tmp_path: Path) -> None:
    """A late original success commits once and prevents a replacement response."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "takeover-key", request, None)
    decision = _decision(identity.lineage_id)
    acceptance = RuntimeAcceptance(
        decision=decision,
        selected_alias=decision.selected_alias,
        selected_model=_snapshot(),
        policy_input=ArtifactInput(artifact_id=decision.policy_id, sha256=decision.policy_sha256),
    )
    original = journal.claim(
        identity,
        acceptance,
        now=_TIME,
        stale_after=timedelta(seconds=1),
    )
    replacement = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )
    assert original.accepted is not None
    assert replacement.accepted is not None

    winning = journal.record_completed(
        original.accepted,
        _response(),
        completed_at=_TIME + timedelta(seconds=3),
    )
    replay = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=4),
        stale_after=timedelta(seconds=1),
    )

    assert replay.status == "completed"
    assert replay.completed == winning
    replacement_observed = journal.record_completed(
        replacement.accepted,
        _response().model_copy(update={"output": AssistantAction(content="replacement")}),
        completed_at=_TIME + timedelta(seconds=5),
    )
    assert replacement_observed == winning
    assert [event.event for event in journal.read_events()] == [
        "accepted",
        "attempt_failed",
        "accepted",
        "completed",
    ]


def test_replacement_permanent_failure_wins_over_late_original_success(tmp_path: Path) -> None:
    """A replacement's permanent failure remains canonical after a late original success."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "permanent-key", request, None)
    decision = _decision(identity.lineage_id)
    acceptance = RuntimeAcceptance(
        decision=decision,
        selected_alias=decision.selected_alias,
        selected_model=_snapshot(),
        policy_input=ArtifactInput(artifact_id=decision.policy_id, sha256=decision.policy_sha256),
    )
    original = journal.claim(
        identity,
        acceptance,
        now=_TIME,
        stale_after=timedelta(seconds=1),
    )
    replacement = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )
    assert original.accepted is not None
    assert replacement.accepted is not None
    permanent = StructuredFailure(
        code=FailureCode.PROVIDER,
        message="routed model provider attempt failed permanently",
        retryable=False,
        attribution=FailureAttribution.MODEL,
    )
    terminal = journal.record_failure(
        replacement.accepted,
        permanent,
        failed_at=_TIME + timedelta(seconds=3),
    )

    observed = journal.record_completed(
        original.accepted,
        _response(),
        completed_at=_TIME + timedelta(seconds=4),
    )

    assert observed == terminal
    failed_claim = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=5),
        stale_after=timedelta(seconds=1),
    )
    assert failed_claim.status == "failed"
    assert [event.event for event in journal.read_events()] == [
        "accepted",
        "attempt_failed",
        "accepted",
        "attempt_failed",
    ]

    corrupted_completion = _completed_event(
        original.accepted,
        _response(),
        ordinal=5,
        completed_at=_TIME + timedelta(seconds=4),
    )
    journal._append_unlocked(corrupted_completion)
    with pytest.raises(RuntimeJournalError, match="permanent interaction failure"):
        journal.read_events()


def test_live_attempt_wait_is_bounded_and_retryable(tmp_path: Path) -> None:
    """A live attempt produces a bounded retryable conflict instead of hanging."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "live-key", request, None)
    decision = _decision(identity.lineage_id)
    journal.claim(
        identity,
        RuntimeAcceptance(
            decision=decision,
            selected_alias=decision.selected_alias,
            selected_model=_snapshot(),
            policy_input=ArtifactInput(
                artifact_id=decision.policy_id,
                sha256=decision.policy_sha256,
            ),
        ),
        now=_TIME,
        stale_after=timedelta(hours=1),
    )
    service = JournaledRouterRuntime(
        cast(RouterRuntime, _FakeRuntime()),
        journal,
        clock=lambda: _TIME,
        wait_timeout_seconds=0,
        stale_after_seconds=3600,
    )

    with pytest.raises(RuntimeInteractionInProgressError) as caught:
        service.complete(request, idempotency_key="live-key")
    assert caught.value.retryable


def test_append_flushes_and_fsyncs_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every accepted and completed line reaches fsync before the service returns."""
    calls: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="fsync-key")

    assert len(calls) >= 2


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_first_append_fsyncs_file_and_new_directory_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first accepted claim persists the file and each newly named parent entry."""
    fsync_calls = 0
    directory_opens: list[Path] = []
    real_fsync = os.fsync
    real_open = os.open

    def record_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        real_fsync(descriptor)

    def record_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if flags & getattr(os, "O_DIRECTORY", 0):
            directory_opens.append(Path(os.fsdecode(path)))
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "open", record_open)
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="directory-fsync-key")

    assert fsync_calls >= 6
    assert {"runtime", "support-agent", "projects", ".wmo"}.issubset(
        {path.name for path in directory_opens}
    )


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_append_retries_directory_fsync_after_first_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later append repairs directory durability after first creation fsync fails."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "fsync-retry-key", request, None)
    decision = _decision(identity.lineage_id)
    acceptance = RuntimeAcceptance(
        decision=decision,
        selected_alias=decision.selected_alias,
        selected_model=_snapshot(),
        policy_input=ArtifactInput(artifact_id=decision.policy_id, sha256=decision.policy_sha256),
    )
    directory_failures = 0
    successful_directory_fsyncs = 0
    real_fsync = os.fsync

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal directory_failures, successful_directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            if directory_failures == 0:
                directory_failures += 1
                raise OSError("simulated directory fsync failure")
            successful_directory_fsyncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_directory_fsync)
    with pytest.raises(RuntimeJournalError, match="cannot persist runtime journal directory"):
        journal.claim(
            identity,
            acceptance,
            now=_TIME,
            stale_after=timedelta(seconds=1),
        )

    retry = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )

    assert retry.status == "dispatch"
    assert successful_directory_fsyncs >= 8
    assert [event.event for event in journal.read_events()] == [
        "accepted",
        "attempt_failed",
        "accepted",
    ]
