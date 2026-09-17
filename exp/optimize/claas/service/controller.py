"""One finite resident learner run, independent of production serving or gateway admission."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from filelock import FileLock, Timeout

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.optimize.claas.buffer.store import ExperienceBuffer
from exp.optimize.claas.service.configuration import RunConfiguration, RunReport, RunStatus
from exp.optimize.claas.service.contracts import LearnerRuntime, LearnerRuntimeFactory
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    ClaasTrainingSpec,
    TrainingExample,
)


async def _await_cleanup[T](task: asyncio.Task[T]) -> T:
    """Join owned cleanup despite repeated cancellation, then preserve caller cancellation."""
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
        except Exception:  # noqa: BLE001 - retrieve the owned task failure below
            break
    if cancelled is not None:
        # Retrieve a cleanup exception without replacing the caller's cancellation.
        if not task.cancelled():
            task.exception()
        raise cancelled
    return task.result()


class LearningController:
    """Coordinate exact generation, delayed feedback, and recoverable optimizer updates."""

    def __init__(
        self,
        directory: Path,
        spec: ClaasTrainingSpec,
        factory: LearnerRuntimeFactory,
        configuration: RunConfiguration,
        *,
        persist: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Bind local durable state without allocating a GPU or beginning a run."""
        self.directory = directory.resolve()
        self.spec = spec
        self.factory = factory
        self.configuration = configuration
        if configuration.minimum_ready_examples > spec.max_batch_examples:
            raise ValueError("minimum_ready_examples must not exceed max_batch_examples")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._process_lock = FileLock(
            self.directory / "run.lock", timeout=0, mode=0o600, thread_local=False
        )
        try:
            self._process_lock.acquire()
        except Timeout:
            raise ValueError(
                "another learner owns this run directory; stop it before resuming"
            ) from None
        try:
            self.buffer = ExperienceBuffer(
                self.directory / "experiences.sqlite",
                spec,
                configuration,
                process_lock=self._process_lock,
            )
        except BaseException:
            self._process_lock.release()
            raise
        self._persist_callback = persist
        self._persistence = asyncio.Lock()
        self._gpu = asyncio.Lock()
        self._startup = asyncio.Lock()
        self._training = asyncio.Lock()
        self._wake = asyncio.Event()
        self._force_requested = False
        self._runtime: LearnerRuntime | None = None
        self._worker: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[RunReport] | None = None
        self._deadline = 0.0
        self._updates = 0
        self._state: Literal["created", "starting", "running", "closing", "closed", "failed"] = (
            "created"
        )
        self._failure: str | None = None
        self._cleanup_failure: str | None = None
        self._stop_reason: str | None = None
        self._burst_ids: tuple[str, ...] | None = None

    async def _persist(self) -> None:
        """Flush mounted storage after each SQLite commit before external acknowledgement."""
        if self._persist_callback is not None:
            async with self._persistence:
                deadline = (
                    asyncio.get_running_loop().time() + self.configuration.cleanup_timeout_seconds
                )
                if self._state in {"starting", "running"}:
                    deadline = min(deadline, self._deadline)
                async with asyncio.timeout_at(deadline):
                    await self._persist_callback()

    @contextmanager
    def hold_directory(self) -> Iterator[None]:
        """Retain this run's ownership through outer receipt writes after compute closes."""
        self._require_writable()
        with self._process_lock:
            yield

    async def start(self) -> RunStatus:
        """Open the runtime once, recover an unacknowledged update, then admit work."""
        if self._state != "created":
            raise ValueError(
                "this controller has already started; create another controller to resume"
            )
        self._state = "starting"
        self._deadline = asyncio.get_running_loop().time() + self.configuration.maximum_run_seconds
        try:
            async with self._startup:
                await self._initialize()
            if self._state == "starting":
                await self.close(reason="drained")
            elif self._state == "closing":
                await self.close()
            return await self.status()
        except BaseException as error:
            self._failure = type(error).__name__
            try:
                await self.close(reason="failed")
            except Exception:  # noqa: BLE001 - preserve the initiating failure
                pass
            raise

    async def _initialize(self) -> None:
        """Publish a runtime and worker only while startup still owns admission."""
        await self._persist()
        if self._state != "starting":
            return
        if self.configuration.mode == "burst":
            self._burst_ids = self.buffer.ready_ids()
            if not self._burst_ids and self.buffer.inflight() is None:
                return
        async with asyncio.timeout_at(self._deadline):
            self._runtime = await self.factory.open(
                self.spec, self.buffer.checkpoint(), mode=self.configuration.mode
            )
            if self._state != "starting":
                return
            if self.buffer.inflight() is not None:
                await self._train_one(force=True)
        if self._state != "starting":
            return
        self._state = "running"
        self._worker = asyncio.create_task(self._background(), name="claas-learning-run")
        self._wake.set()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a persisted original sample, or replay the same request without resampling."""
        if self.configuration.mode != "run":
            raise ValueError("burst mode does not generate; import exact examples before draining")
        self._require_running()
        async with self._gpu:
            self._require_running()
            await self._persist()
            replay = self.buffer.replay(request)
            if replay is not None:
                return replay
            if request.model not in {self.spec.base_model, self.spec.adapter_id}:
                raise ValueError("generation model differs from this run's configured student")
            runtime = self._require_runtime()
            async with asyncio.timeout_at(self._deadline):
                result = await runtime.generate(request)
            token = result.exact_tokens
            if (
                token.policy_revision != runtime.policy_revision
                or token.model_id != self.spec.base_model
                or token.model_revision != self.spec.model_revision
                or token.tokenizer_id != self.spec.tokenizer_id
                or token.tokenizer_revision != self.spec.tokenizer_revision
                or len(token.response_token_ids) > request.maximum_output_tokens
            ):
                raise ClaasTrainingError(
                    "runtime generation evidence differs from the requested student or limit"
                )
            self.buffer.record_generation(request, result)
            await self._persist()
            return result

    async def submit_feedback(
        self,
        response_id: str,
        *,
        scalar_reward: float | None = None,
        text_feedback: str | None = None,
    ) -> RunStatus:
        """Persist later signals without waiting for the GPU or acknowledging partial writes."""
        self._require_writable()
        self.buffer.feedback(response_id, scalar_reward=scalar_reward, text_feedback=text_feedback)
        await self._persist()
        self._wake.set()
        return await self.status()

    async def import_examples(self, examples: tuple[TrainingExample, ...]) -> RunStatus:
        """Atomically retain exact external examples, including explicit rejection records."""
        self._require_writable()
        self.buffer.import_examples(examples)
        await self._persist()
        self._wake.set()
        return await self.status()

    async def status(self) -> RunStatus:
        """Read queue counts while generation or training is in flight."""
        checkpoint = self.buffer.checkpoint()
        return RunStatus(
            mode=self.configuration.mode,
            state=self._state,
            updates=self._updates,
            policy_revision=checkpoint.policy_revision
            if checkpoint
            else self.spec.initial_policy_revision,
            buffer=self.buffer.status(),
            stop_reason=self._stop_reason,
            failure_type=self._failure,
            cleanup_failure_type=self._cleanup_failure,
        )

    async def trigger_train(self) -> RunStatus:
        """Queue a partial update without waiting for compute; inspect status for completion."""
        self._require_running()
        self._force_requested = True
        self._wake.set()
        return await self.status()

    async def drain(self) -> RunReport:
        """Drain a frozen ready snapshot, forcing partial batches, then close a burst run."""
        if self._state == "created":
            await self.start()
        if self._state in {"closed", "failed"}:
            return RunReport(status=await self.status(), checkpoint=self.buffer.checkpoint())
        ids = self._burst_ids if self._burst_ids is not None else self.buffer.ready_ids()
        try:
            while self._state == "running" and self._limit_reason() is None:
                if not await self._train_one(force=True, allowed_ids=ids):
                    break
        except BaseException as error:
            self._failure = type(error).__name__
            try:
                await self.close(reason="failed")
            except Exception:  # noqa: BLE001 - preserve the initiating failure
                pass
            raise
        if self.configuration.mode == "burst":
            return await self.close(reason=self._limit_reason() or "drained")
        return RunReport(status=await self.status(), checkpoint=self.buffer.checkpoint())

    async def _train_one(self, *, force: bool, allowed_ids: tuple[str, ...] | None = None) -> bool:
        """Serialize lease, optimizer execution, and atomic checkpoint acknowledgement."""
        async with self._training, self._gpu:
            if self._state not in {"starting", "running"} or self._limit_reason() is not None:
                return False
            await self._persist()
            if not force and self.buffer.status().ready < self.configuration.minimum_ready_examples:
                return False
            batch = self.buffer.lease(allowed_ids=allowed_ids)
            await self._persist()
            if batch is None:
                return False
            runtime = self._require_runtime()
            async with asyncio.timeout_at(self._deadline):
                result = await runtime.train(batch)
            if runtime.policy_revision != result.checkpoint.policy_revision:
                raise ClaasTrainingError(
                    "runtime did not select its acknowledged optimizer revision"
                )
            self.buffer.acknowledge(batch, result)
            await self._persist()
            self._updates += 1
            return True

    def _limit_reason(self) -> str | None:
        """Name the first finite compute bound reached by this controller."""
        if self._updates >= self.configuration.maximum_updates:
            return "maximum_updates"
        if self._deadline and asyncio.get_running_loop().time() >= self._deadline:
            return "deadline"
        return None

    def _require_running(self) -> None:
        """Reject generation after admission closes or its lifetime expires."""
        if self._state != "running" or self._limit_reason() is not None:
            raise ValueError(
                "learning run is not accepting work; inspect status or start a new run"
            )

    def _require_writable(self) -> None:
        """Keep CPU imports and feedback inside this controller's actual process ownership."""
        if self._state not in {"created", "starting", "running"}:
            raise ValueError("learning run no longer owns writes; create a new controller")

    def _require_runtime(self) -> LearnerRuntime:
        """Require the single resident runtime opened by this controller."""
        if self._runtime is None:
            raise ClaasTrainingError("learning runtime has not been opened")
        return self._runtime

    async def _background(self) -> None:
        """Train feedback-ready work while retaining a deadline even when the queue is idle."""
        try:
            while self._state == "running":
                self._wake.clear()
                reason = self._limit_reason()
                if reason is not None:
                    self._begin_shutdown(reason)
                    return
                force = self._force_requested
                self._force_requested = False
                if (self.configuration.mode == "run" or force) and await self._train_one(
                    force=force, allowed_ids=self._burst_ids
                ):
                    continue
                try:
                    async with asyncio.timeout_at(self._deadline):
                        await self._wake.wait()
                except TimeoutError:
                    self._begin_shutdown("deadline")
                    return
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - background failures become run status
            if isinstance(error, TimeoutError) and self._limit_reason() == "deadline":
                self._begin_shutdown("deadline")
            else:
                self._failure = type(error).__name__
                self._begin_shutdown("failed")

    def _begin_shutdown(self, reason: str) -> asyncio.Task[RunReport]:
        """Create one cleanup owner without making its background task join itself."""
        if self._shutdown_task is None:
            self._stop_reason = reason
            self._state = "closing"
            self._shutdown_task = asyncio.create_task(self._shutdown(asyncio.current_task()))
            self._shutdown_task.add_done_callback(self._observe_shutdown)
        return self._shutdown_task

    @staticmethod
    def _observe_shutdown(task: asyncio.Task[RunReport]) -> None:
        """Retrieve autonomous cleanup failures; status and explicit close retain the error."""
        if not task.cancelled():
            task.exception()

    async def close(self, *, reason: str = "closed") -> RunReport:
        """Join owned operations and cleanup; leave every unacknowledged lease recoverable."""
        return await _await_cleanup(self._begin_shutdown(reason))

    async def _shutdown(self, owner: asyncio.Task[object] | None) -> RunReport:
        """Close compute before releasing process ownership; uncertain cleanup keeps the lock."""
        worker = self._worker
        if worker is not None and worker is not owner and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        try:
            async with self._startup, self._gpu:
                if self._runtime is not None:
                    async with asyncio.timeout(self.configuration.cleanup_timeout_seconds):
                        await self._runtime.close()
                await self._persist()
        except BaseException as error:
            self._cleanup_failure = type(error).__name__
            self._failure = self._failure or self._cleanup_failure
            self._state = "failed"
            raise
        self._state = "failed" if self._failure else "closed"
        report = RunReport(status=await self.status(), checkpoint=self.buffer.checkpoint())
        self._process_lock.release()
        return report
