"""Serialized ownership of resident upstream training and optional rollout engines.

Cancellation joins native work to avoid releasing GPU ownership prematurely.
In-process timeouts are soft; local process hosting and remote container lifetimes
provide the hard termination boundary for a stalled CUDA call.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Literal, Protocol, cast

import torch
from peft import LoraConfig
from verl.workers.config import HFModelConfig
from verl.workers.rollout.replica import TokenOutput

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.optimize.claas.backends.checkpoints import verify_checkpoint
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.generation import generation_result, prompt_ids
from exp.optimize.claas.backends.verl.native import worker_config
from exp.optimize.claas.backends.verl.resident import ResidentTrainer
from exp.optimize.claas.service.contracts import RunMode
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingResult,
)


async def _join_owned[T](future: asyncio.Future[T]) -> T:
    """Retain ownership until work exits, preserving cancellation over later worker failure."""
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        joined = asyncio.gather(future, return_exceptions=True)
        while not joined.done():
            try:
                await asyncio.shield(joined)
            except asyncio.CancelledError:
                continue
        joined.result()
        raise


class Rollout(Protocol):
    """The upstream rollout extension is loaded only for explicitly selected run mode."""

    async def initialize(self, model: HFModelConfig, spec: ClaasTrainingSpec) -> None:
        """Initialize one rollout engine while training state is CPU-offloaded."""
        ...

    async def synchronize(
        self, weights: dict[str, torch.Tensor], config: LoraConfig, step: int
    ) -> None:
        """Install one complete adapter before admitting generation."""
        ...

    async def generate(
        self, prompt: tuple[int, ...], maximum: int, request_id: str, step: int
    ) -> TokenOutput:
        """Sample exact original tokens under the selected update identity."""
        ...

    async def sleep(self) -> None:
        """Release rollout memory before training uses the shared device."""
        ...

    async def close(self) -> None:
        """Terminate all owned rollout workers."""
        ...


class ResidentVerlFactory:
    """Open one single-GPU resident runtime per isolated service process."""

    def __init__(self, settings: ResidentVerlSettings) -> None:
        """Bind explicit settings without downloading weights or acquiring resources."""
        self.settings = settings

    async def open(
        self,
        spec: ClaasTrainingSpec,
        resume: TrainingCheckpoint | None = None,
        *,
        mode: RunMode,
    ) -> ResidentVerlRuntime:
        """Initialize once and retain engine state until the returned runtime closes."""
        if mode not in {"burst", "run"}:
            raise ValueError("resident runtime mode must be burst or run")
        runtime = ResidentVerlRuntime(spec, self.settings)
        try:
            await runtime.initialize(resume, mode)
        except BaseException:
            await runtime.close()
            raise
        return runtime


class ResidentVerlRuntime:
    """Serialize generation and updates while feedback ingestion remains independent."""

    def __init__(self, spec: ClaasTrainingSpec, settings: ResidentVerlSettings) -> None:
        """Create local ownership handles before allocating any model or process group."""
        self.spec, self.settings = spec, settings
        self.trainer = ResidentTrainer(spec, settings)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claas-verl")
        self._lock = asyncio.Lock()
        self._rollout: Rollout | None = None
        self._closed = False
        self._failed = False
        self._close_task: asyncio.Task[None] | None = None

    async def _call[T](self, operation: Callable[[], T]) -> T:
        """Join owned native work even when its async caller is canceled."""
        future = asyncio.get_running_loop().run_in_executor(self._executor, operation)
        return await _join_owned(future)

    async def initialize(self, resume: TrainingCheckpoint | None, mode: RunMode) -> None:
        """Load the resident trainer, then optionally start and synchronize rollout once."""
        async with self._lock, asyncio.timeout(self.settings.startup_timeout_seconds):
            await self._call(partial(self.trainer.initialize, resume))
            if mode == "run":
                # Selecting the optional rollout plugin must fail rather than silently
                # replacing genuine rollout evidence with another generation backend.
                module = importlib.import_module("exp.optimize.claas.backends.verl.rollout")
                self._rollout = cast(Rollout, module.ResidentRollout(self.settings))
                if self.trainer.model_path is None or self.trainer.tokenizer_path is None:
                    raise ClaasTrainingError("resident model snapshots are unavailable")
                model = worker_config(
                    self.spec, self.trainer.model_path, self.trainer.tokenizer_path, teacher=False
                ).model_config
                await self._rollout.initialize(model, self.spec)
                await self._synchronize()

    @property
    def policy_revision(self) -> str:
        """Expose only the trainer's last durably committed policy."""
        return self.trainer.policy_revision

    def _require_open(self) -> None:
        """Reject further work after close or an uncertain engine operation."""
        if self._closed or self._failed:
            raise ClaasTrainingError(
                "resident runtime is closed or failed; reopen its latest checkpoint"
            )

    async def _synchronize(self) -> None:
        """Transfer a CPU adapter snapshot after actor memory has been released."""
        if self._rollout is not None:
            weights, config = await self._call(self.trainer.adapter)
            step = self.trainer.checkpoint.step if self.trainer.checkpoint else 0
            await self._rollout.synchronize(weights, config, step)

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        """Persist one update and synchronize rollout without recreating either engine."""
        async with self._lock:
            self._require_open()
            try:
                if self._rollout is not None:
                    await self._rollout.sleep()
                async with asyncio.timeout(self.settings.operation_timeout_seconds):
                    result = await self._call(partial(self.trainer.train, batch))
                await self._synchronize()
                return result
            except BaseException:
                self._failed = True
                raise

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Pin a revision for the complete token-in/token-out sampling operation."""
        async with self._lock:
            self._require_open()
            if self._rollout is None:
                raise ClaasTrainingError("burst mode has no generation engine; open run mode")
            tokenizer = self.trainer.tokenizer
            if tokenizer is None:
                raise ClaasTrainingError("resident tokenizer is unavailable")
            prompt = prompt_ids(request, tokenizer, self.spec, self.settings)
            revision = self.policy_revision
            step = self.trainer.checkpoint.step if self.trainer.checkpoint else 0
            try:
                result = await self._rollout.generate(
                    prompt, request.maximum_output_tokens, request.request_id, step
                )
                finish_reason = result.extra_fields.get("finish_reason")
                if finish_reason not in {"stop", "length"}:
                    raise ClaasTrainingError("rollout omitted its native finish reason")
                return generation_result(
                    request,
                    prompt,
                    tuple(result.token_ids),
                    tuple(result.log_probs or ()),
                    tokenizer,
                    self.spec,
                    self.settings,
                    revision,
                    cast(Literal["stop", "length"], finish_reason),
                )
            except BaseException:
                self._failed = True
                raise

    async def checkpoint(self) -> TrainingCheckpoint:
        """Return only verified committed state under the same update lock."""
        async with self._lock:
            self._require_open()
            receipt = self.trainer.checkpoint
            if receipt is None:
                raise ClaasTrainingError("no checkpoint exists before a committed update")
            await self._call(partial(verify_checkpoint, receipt, self.spec))
            return receipt

    async def close(self) -> None:
        """Join one cleanup owner before cancellation can release GPU ownership."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await _join_owned(self._close_task)

    async def _close(self) -> None:
        """Release rollout before training state and its process group exactly once."""
        async with self._lock:
            self._closed = True
            try:
                if self._rollout is not None:
                    await self._rollout.close()
            finally:
                try:
                    await self._call(self.trainer.close)
                finally:
                    self._executor.shutdown(wait=True, cancel_futures=True)
