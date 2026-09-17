"""Real SQLite/controller orchestration with a deterministic idempotent runtime fixture."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.models import AssistantAction
from exp.optimize.claas.buffer.store import ExperienceBuffer
from exp.optimize.claas.buffer.store_test import item, result
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.service.contracts import RunMode
from exp.optimize.claas.service.controller import LearningController
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingResult,
)
from exp.optimize.claas.training_contracts_test import spec


class Runtime:
    """Count distinct update calls using idempotent fixture receipts, without executing training."""

    def __init__(self) -> None:
        """Create explicit observations and optional train/cleanup blocking gates."""
        self.policy_revision = "policy-0"
        self.receipts: dict[str, TrainingResult] = {}
        self.open_count = 0
        self.close_count = 0
        self.generate_count = 0
        self.optimizations = 0
        self.train_entered = asyncio.Event()
        self.train_gate: asyncio.Event | None = None
        self.close_entered = asyncio.Event()
        self.close_gate: asyncio.Event | None = None
        self.after_commit_error: Exception | None = None
        self.directory = Path("/unused")

    async def open(
        self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None, *, mode: RunMode
    ) -> Runtime:
        """Record one factory allocation while retaining committed idempotency receipts."""
        self.open_count += 1
        self.policy_revision = resume.policy_revision if resume else spec.initial_policy_revision
        return self

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return original deterministic tokens bound to the currently loaded fixture policy."""
        self.generate_count += 1
        tokens = item(policy=self.policy_revision).experience.exact_tokens
        assert tokens is not None
        return GenerationResult(
            response_id=f"response-{request.request_id}",
            action=AssistantAction(content="answer"),
            exact_tokens=tokens,
            raw_text="answer",
        )

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        """Replay a committed batch exactly, including a prior lost acknowledgement."""
        self.train_entered.set()
        if self.train_gate is not None:
            await self.train_gate.wait()
        receipt = self.receipts.get(batch.batch_id)
        if receipt is None:
            self.optimizations += 1
            receipt = result(batch, self.directory, step=self.optimizations)
            history = tuple(f"policy-{index}" for index in range(self.optimizations, -1, -1))
            receipt = receipt.model_copy(
                update={
                    "checkpoint": receipt.checkpoint.model_copy(update={"policy_history": history})
                }
            )
            self.receipts[batch.batch_id] = receipt
        self.policy_revision = receipt.checkpoint.policy_revision
        if self.after_commit_error is not None:
            error, self.after_commit_error = self.after_commit_error, None
            raise error
        return receipt

    async def checkpoint(self) -> TrainingCheckpoint:
        """Return the last acknowledged fixture checkpoint."""
        return tuple(self.receipts.values())[-1].checkpoint

    async def close(self) -> None:
        """Expose cleanup completion so cancellation tests can observe ownership."""
        self.close_entered.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.close_count += 1


def test_burst_opens_once_drains_partial_and_excludes_later_imports(tmp_path: Path) -> None:
    """A burst consumes its initial ready snapshot across updates and reports a later tail."""

    async def run() -> None:
        """Drive the complete public controller boundary without mocks of its implementation."""
        runtime = Runtime()
        config = RunConfiguration(mode="burst", minimum_ready_examples=2)
        recipe = spec().model_copy(update={"max_batch_examples": 2})
        controller = LearningController(tmp_path, recipe, runtime, config)
        await controller.import_examples((item("one"), item("two"), item("three")))
        await controller.start()
        await controller.import_examples((item("later"),))
        report = await controller.drain()
        assert (runtime.open_count, runtime.optimizations, runtime.close_count) == (1, 2, 1)
        assert report.status.buffer.consumed == 3
        assert report.status.buffer.ready == 1
        assert report.status.stop_reason == "drained"
        with pytest.raises(ValueError, match="burst mode"):
            await controller.generate(
                GenerationRequest(request_id="x", model="tiny-model", prompt="x")
            )

    asyncio.run(run())


def test_empty_burst_avoids_gpu_and_update_limit_preserves_ready(tmp_path: Path) -> None:
    """No-work bursts allocate nothing; finite update limits retain an explicit queue tail."""

    async def run() -> None:
        """Run empty and bounded bursts against independent durable queues."""
        runtime = Runtime()
        empty = LearningController(
            tmp_path / "empty", spec(), runtime, RunConfiguration(mode="burst")
        )
        assert (await empty.drain()).status.stop_reason == "drained"
        assert runtime.open_count == 0
        controller = LearningController(
            tmp_path / "limited",
            spec().model_copy(update={"max_batch_examples": 1}),
            runtime,
            RunConfiguration(mode="burst", maximum_updates=1, minimum_ready_examples=1),
        )
        await controller.import_examples((item("one"), item("two")))
        report = await controller.drain()
        assert report.status.stop_reason == "maximum_updates"
        assert report.status.buffer.ready == 1
        assert report.status.buffer.consumed == 1

    asyncio.run(run())


def test_run_generates_replays_and_trains_feedback_without_blocking_status(tmp_path: Path) -> None:
    """Later feedback triggers background work while HTTP-equivalent status stays responsive."""

    async def run() -> None:
        """Observe an in-flight optimizer without bypassing the queue or scheduler."""
        runtime = Runtime()
        runtime.train_gate = asyncio.Event()
        controller = LearningController(
            tmp_path, spec(), runtime, RunConfiguration(minimum_ready_examples=1)
        )
        await controller.start()
        request = GenerationRequest(request_id="one", model="tiny-model", prompt="question")
        generated = await controller.generate(request)
        assert await controller.generate(request) == generated
        assert runtime.generate_count == 1
        await controller.submit_feedback(generated.response_id, text_feedback="Correct")
        await asyncio.wait_for(runtime.train_entered.wait(), 1)
        status = await asyncio.wait_for(controller.status(), 0.1)
        assert status.buffer.inflight == 1
        await asyncio.wait_for(
            controller.submit_feedback(generated.response_id, text_feedback="Correct"), 0.1
        )
        runtime.train_gate.set()
        for _ in range(100):
            if (await controller.status()).buffer.consumed:
                break
            await asyncio.sleep(0.001)
        report = await controller.close()
        assert report.status.buffer.consumed == 1
        assert runtime.open_count == runtime.optimizations == runtime.close_count == 1

    asyncio.run(run())


def test_lost_training_ack_replays_same_batch_without_second_optimization(tmp_path: Path) -> None:
    """A committed worker result plus caller failure survives restart exactly once."""

    async def run() -> None:
        """Lose a result after the fake worker commits it, then recover through the public API."""
        runtime = Runtime()
        runtime.after_commit_error = OSError("lost acknowledgement")
        config = RunConfiguration(mode="burst")
        first = LearningController(tmp_path, spec(), runtime, config)
        await first.import_examples((item(),))
        with pytest.raises(OSError, match="lost acknowledgement"):
            await first.drain()
        inflight = first.buffer.inflight()
        assert inflight is not None
        assert first.buffer.checkpoint() is None
        second = LearningController(tmp_path, spec(), runtime, config)
        report = await second.drain()
        assert runtime.optimizations == 1
        assert runtime.open_count == runtime.close_count == 2
        assert report.status.buffer.consumed == 1
        assert second.buffer.inflight() is None

    asyncio.run(run())


def test_deadline_and_cancel_preserve_unconsumed_lease_and_release_compute(tmp_path: Path) -> None:
    """Cancelling active training leaves its exact lease recoverable and closes the runtime."""

    async def run() -> None:
        """Cancel during a real awaited training call, then inspect the durable database."""
        runtime = Runtime()
        runtime.train_gate = asyncio.Event()
        controller = LearningController(tmp_path, spec(), runtime, RunConfiguration(mode="burst"))
        await controller.import_examples((item(),))
        task = asyncio.create_task(controller.drain())
        await asyncio.wait_for(runtime.train_entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.close_count == 1
        assert (await controller.status()).buffer.inflight == 1
        recovered = ExperienceBuffer(tmp_path / "experiences.sqlite", spec(), RunConfiguration())
        assert recovered.inflight() is not None

    asyncio.run(run())


def test_idle_deadline_closes_and_process_ownership_is_exclusive(tmp_path: Path) -> None:
    """Even idle service runs stop on time, and a second controller cannot acquire compute."""

    async def run() -> None:
        """Verify the actual file lock and background deadline using one isolated path."""
        runtime = Runtime()
        controller = LearningController(
            tmp_path, spec(), runtime, RunConfiguration(maximum_run_seconds=0.05)
        )
        with pytest.raises(ValueError, match="another learner"):
            LearningController(tmp_path, spec(), runtime, RunConfiguration())
        await controller.start()
        await asyncio.wait_for(runtime.close_entered.wait(), 1)
        report = await controller.close()
        assert report.status.stop_reason == "deadline"
        assert runtime.close_count == 1

    asyncio.run(run())


def test_cancellation_during_close_joins_cleanup_before_unlocking(tmp_path: Path) -> None:
    """Repeated caller cancellation cannot abandon cleanup or release ownership early."""

    async def run() -> None:
        """Hold cleanup, cancel its waiter twice, then prove the original cancellation wins."""
        runtime = Runtime()
        runtime.close_gate = asyncio.Event()
        controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
        await controller.start()
        closing = asyncio.create_task(controller.close())
        await runtime.close_entered.wait()
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        assert not closing.done()
        runtime.close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert runtime.close_count == 1
        other = LearningController(tmp_path, spec(), runtime, RunConfiguration())
        await other.start()
        await other.close()

    asyncio.run(run())


def test_deadline_during_training_retains_batch_for_recovery(tmp_path: Path) -> None:
    """A resident run's hard deadline closes compute without pretending in-flight work drained."""

    async def run() -> None:
        """Let the actual background train wait expire and inspect its complete status."""
        runtime = Runtime()
        runtime.train_gate = asyncio.Event()
        controller = LearningController(
            tmp_path,
            spec(),
            runtime,
            RunConfiguration(minimum_ready_examples=1, maximum_run_seconds=0.05),
        )
        await controller.import_examples((item(),))
        await controller.start()
        await asyncio.wait_for(runtime.train_entered.wait(), 1)
        await asyncio.wait_for(runtime.close_entered.wait(), 1)
        report = await controller.close()
        assert report.status.stop_reason == "deadline"
        assert report.status.buffer.inflight == 1
        assert report.status.buffer.consumed == 0
        assert runtime.close_count == 1

    asyncio.run(run())


def test_persistence_failure_before_training_does_not_execute_update(tmp_path: Path) -> None:
    """A leased batch cannot dispatch until its hosting persistence barrier succeeds."""

    async def run() -> None:
        """Fail the lease flush, then resume and replay the exact durable lease."""
        runtime = Runtime()
        fail = False

        async def persist() -> None:
            """Fail only after a lease appears, before the runtime sees it."""
            if fail and controller.buffer.inflight() is not None:
                raise OSError("remote storage unavailable")

        controller = LearningController(
            tmp_path, spec(), runtime, RunConfiguration(mode="burst"), persist=persist
        )
        await controller.import_examples((item(),))
        await controller.start()
        fail = True
        with pytest.raises(OSError, match="storage unavailable"):
            await controller.drain()
        assert runtime.optimizations == 0
        assert controller.buffer.inflight() is not None
        assert (await controller.status()).cleanup_failure_type == "OSError"
        # An uncertain persistence barrier keeps the process lock until this owner exits.
        with pytest.raises(ValueError, match="another learner"):
            LearningController(tmp_path, spec(), runtime, RunConfiguration(mode="burst"))

    asyncio.run(run())


def test_runtime_revision_mismatch_does_not_consume_batch(tmp_path: Path) -> None:
    """A reported checkpoint is insufficient if the resident runtime selects another policy."""

    class WrongRevision(Runtime):
        """Return a valid-looking receipt while violating loaded-policy acknowledgement."""

        async def train(self, batch: TrainingBatch) -> TrainingResult:
            """Exercise validation before atomic queue consumption."""
            receipt = await super().train(batch)
            self.policy_revision = "wrong"
            return receipt

    async def run() -> None:
        """Drive invalid training completion through the public burst entrypoint."""
        runtime = WrongRevision()
        controller = LearningController(tmp_path, spec(), runtime, RunConfiguration(mode="burst"))
        await controller.import_examples((item(),))
        with pytest.raises(ValueError, match="did not select"):
            await controller.drain()
        status = await controller.status()
        assert status.buffer.inflight == 1
        assert status.buffer.consumed == 0
        assert controller.buffer.checkpoint() is None

    asyncio.run(run())


def test_train_trigger_returns_before_optimizer_and_accepts_adapter_alias(tmp_path: Path) -> None:
    """HTTP-style train triggers acknowledge enqueue while the optimizer remains asynchronous."""

    async def run() -> None:
        """Exercise an adapter-named request, later feedback, and an explicitly blocked update."""
        runtime = Runtime()
        runtime.train_gate = asyncio.Event()
        controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
        await controller.start()
        request = GenerationRequest(request_id="alias", model=spec().adapter_id, prompt="question")
        generated = await controller.generate(request)
        await controller.submit_feedback(generated.response_id, text_feedback="Correct")
        status = await asyncio.wait_for(controller.trigger_train(), 0.1)
        assert status.updates == 0
        await asyncio.wait_for(runtime.train_entered.wait(), 1)
        assert (await controller.status()).buffer.inflight == 1
        runtime.train_gate.set()
        await controller.drain()
        assert (await controller.status()).buffer.consumed == 1
        assert controller.buffer.replay(request) == generated
        await controller.close()

    asyncio.run(run())


def test_constructor_and_cpu_import_respect_existing_owner(tmp_path: Path) -> None:
    """A resident owner excludes construction and previously opened standalone buffer writers."""

    async def run() -> None:
        """Exercise the file lock before initialization, during execution, and after close."""
        standalone = ExperienceBuffer(tmp_path / "experiences.sqlite", spec(), RunConfiguration())
        runtime = Runtime()
        controller = LearningController(tmp_path, spec(), runtime, RunConfiguration())
        before = (tmp_path / "experiences.sqlite").read_bytes()
        with pytest.raises(ValueError, match="another learner"):
            LearningController(tmp_path, spec(), runtime, RunConfiguration())
        assert (tmp_path / "experiences.sqlite").read_bytes() == before
        with pytest.raises(ValueError, match="another learner"):
            standalone.import_examples((item("outside"),))
        await controller.import_examples((item("inside"),))
        assert (await controller.status()).buffer.ready == 1
        await controller.close()
        standalone.import_examples((item("after-close"),))
        with pytest.raises(ValueError, match="no longer owns writes"):
            await controller.import_examples((item("closed-controller"),))
        with pytest.raises(ValueError, match="no longer owns writes"):
            await controller.submit_feedback("inside", text_feedback="Correct")
        assert standalone.status().ready == 2
        assert runtime.open_count == 0

    asyncio.run(run())


def test_train_trigger_during_persistence_cannot_lose_wakeup(tmp_path: Path) -> None:
    """A trigger received while the scheduler checks capacity remains queued for its next turn."""

    async def run() -> None:
        """Block the first background persistence barrier and enqueue a partial update."""
        runtime = Runtime()
        runtime.train_gate = asyncio.Event()
        checking = asyncio.Event()
        release = asyncio.Event()
        block = False

        async def persist() -> None:
            """Hold exactly one background eligibility check after startup returns."""
            nonlocal block
            if block:
                block = False
                checking.set()
                await release.wait()

        controller = LearningController(
            tmp_path,
            spec(),
            runtime,
            RunConfiguration(),
            persist=persist,
        )
        await controller.import_examples((item(),))
        await controller.start()
        block = True
        await asyncio.wait_for(checking.wait(), 1)
        await controller.trigger_train()
        release.set()
        await asyncio.wait_for(runtime.train_entered.wait(), 1)
        runtime.train_gate.set()
        await controller.close()

    asyncio.run(run())


def test_failed_ack_flush_retries_persistence_without_second_update(tmp_path: Path) -> None:
    """A lost persistence acknowledgement cannot turn one committed batch into two updates."""

    async def run() -> None:
        """Fail after the atomic queue commit, flush during cleanup, and reopen the same queue."""
        runtime = Runtime()
        failed = False

        async def persist() -> None:
            """Lose exactly the first acknowledgement after a checkpoint enters SQLite."""
            nonlocal failed
            if first.buffer.checkpoint() is not None and not failed:
                failed = True
                raise OSError("checkpoint flush unavailable")

        limits = RunConfiguration(mode="burst")
        first = LearningController(tmp_path, spec(), runtime, limits, persist=persist)
        await first.import_examples((item(),))
        with pytest.raises(OSError, match="flush unavailable"):
            await first.drain()
        assert (await first.status()).buffer.consumed == 1
        second = LearningController(tmp_path, spec(), runtime, limits)
        report = await second.drain()
        assert report.status.buffer.consumed == 1
        assert runtime.optimizations == 1
        assert runtime.open_count == 1
        assert runtime.close_count == 1

    asyncio.run(run())


class GatedStartupRuntime(Runtime):
    """Delay publication of an allocated runtime while a caller requests shutdown."""

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    async def open(
        self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None, *, mode: RunMode
    ) -> Runtime:
        self.entered.set()
        await self.release.wait()
        return await super().open(spec, resume, mode=mode)


@pytest.mark.parametrize("phase", ["persist", "open"])
def test_shutdown_during_startup_keeps_ownership_until_open_is_joined(
    tmp_path: Path, phase: str
) -> None:
    """Closing cannot free the directory or let startup publish a new worker afterward."""

    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        first_persist = True

        async def persist() -> None:
            nonlocal first_persist
            if phase == "persist" and first_persist:
                first_persist = False
                entered.set()
                await release.wait()

        runtime = GatedStartupRuntime(entered, release) if phase == "open" else Runtime()
        config = RunConfiguration()
        controller = LearningController(tmp_path, spec(), runtime, config, persist=persist)
        startup = asyncio.create_task(controller.start())
        await asyncio.wait_for(entered.wait(), 1)
        shutdown = asyncio.create_task(controller.close())
        await asyncio.sleep(0)
        assert (await controller.status()).state == "closing"
        assert not shutdown.done()
        with pytest.raises(ValueError, match="another learner"):
            LearningController(tmp_path, spec(), Runtime(), config)
        release.set()
        started, closed = await asyncio.wait_for(asyncio.gather(startup, shutdown), 1)
        assert started.state == closed.status.state == "closed"
        assert runtime.open_count == runtime.close_count == (1 if phase == "open" else 0)
        with pytest.raises(ValueError, match="not accepting"):
            await controller.generate(
                GenerationRequest(request_id="late", model="tiny-model", prompt="x")
            )
        replacement = LearningController(tmp_path, spec(), Runtime(), config)
        await replacement.close()

    asyncio.run(run())
