"""Async ownership tests with inert workers, distinct from CUDA training proof."""

import asyncio
import threading
from pathlib import Path
from typing import cast

import pytest
from tokenizers import AddedToken
from verl.workers.rollout.replica import TokenOutput

from exp.common.claas.generation import GenerationRequest
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.decoding import UnfinishedReasoningError
from exp.optimize.claas.backends.verl.native_test import tokenizer
from exp.optimize.claas.backends.verl.runtime import ResidentVerlRuntime, Rollout
from exp.optimize.claas.training_contracts import ClaasTrainingError, TrainingBatch, TrainingResult
from exp.optimize.claas.training_contracts_test import job, spec


@pytest.mark.parametrize("outcome", ["length", "stop", "missing", "engine"])
def test_native_generation_reason_gates_retention_and_runtime_failure(
    tmp_path: Path, outcome: str
) -> None:
    """Only observed length keeps unfinished reasoning and leaves the resident runtime usable."""

    async def exercise() -> None:
        """Inject only the engine boundary while driving the real runtime and decoder."""
        settings = ResidentVerlSettings(checkpoint_root=tmp_path, decoder="hermes")
        runtime = ResidentVerlRuntime(spec(), settings)
        model_tokenizer = tokenizer()
        raw = "<think>private unfinished reasoning"
        model_tokenizer.add_special_tokens(
            {"additional_special_tokens": [AddedToken(raw, normalized=False)]}
        )
        runtime.trainer.tokenizer = model_tokenizer
        response = model_tokenizer.encode(raw, add_special_tokens=False)
        engine_error = RuntimeError("engine failed before a terminal result")

        class SampledRollout:
            """Expose only the inert native output and close boundary used by this test."""

            async def generate(
                self, prompt: tuple[int, ...], maximum: int, request_id: str, step: int
            ) -> TokenOutput:
                """Supply native metadata without claiming that this fixture sampled on a GPU."""
                assert prompt == (1,) and maximum == 1
                if outcome == "engine":
                    raise engine_error
                return TokenOutput(
                    token_ids=response,
                    log_probs=[-0.25],
                    stop_reason="completed",
                    extra_fields={} if outcome == "missing" else {"finish_reason": outcome},
                )

            async def close(self) -> None:
                """Close the inert fixture without any runtime or GPU resource."""

        runtime._rollout = cast(Rollout, SampledRollout())
        request = GenerationRequest(
            request_id="reasoning", model=spec().base_model, prompt="a", maximum_output_tokens=1
        )
        try:
            if outcome == "length":
                result = await runtime.generate(request)
                assert result.finish_reason == "length" and result.action.content == ""
                assert result.raw_text == raw
                assert not runtime._failed
                repeated = await runtime.generate(request)
                assert repeated.exact_tokens == result.exact_tokens
                assert repeated.action == result.action
            else:
                expected = {
                    "engine": RuntimeError,
                    "stop": UnfinishedReasoningError,
                    "missing": ClaasTrainingError,
                }[outcome]
                with pytest.raises(expected) as caught:
                    await runtime.generate(request)
                if outcome == "engine":
                    assert caught.value is engine_error
                assert runtime._failed
                with pytest.raises(ClaasTrainingError, match="failed"):
                    await runtime.generate(request)
        finally:
            await runtime.close()

    asyncio.run(exercise())


def test_cancellation_joins_native_failure_before_releasing_ownership(tmp_path: Path) -> None:
    """Repeated cancellation cannot abandon an owned optimizer or replace cancellation."""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def operation() -> None:
        """Hold an inert native worker then fail after caller cancellation."""
        entered.set()
        assert release.wait(10)
        finished.set()
        raise ValueError("late worker failure")

    async def exercise() -> None:
        """Observe cancellation while the native worker still owns its resources."""
        runtime = ResidentVerlRuntime(spec(), ResidentVerlSettings(checkpoint_root=tmp_path))
        task = asyncio.create_task(runtime._call(operation))
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        await runtime.close()

    asyncio.run(exercise())


def test_train_calls_serialize_and_reuse_the_resident_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission holds one worker until its operation exits, with no per-update reset."""
    entered, release = threading.Event(), threading.Event()
    observed: list[str] = []

    async def exercise() -> None:
        """Queue two distinct updates behind one inert retained trainer."""
        runtime = ResidentVerlRuntime(spec(), ResidentVerlSettings(checkpoint_root=tmp_path))

        def train(batch: TrainingBatch) -> TrainingResult:
            """Record serialization without pretending to execute an optimizer."""
            observed.append(batch.batch_id)
            if batch.batch_id == "batch-1":
                entered.set()
                assert release.wait(10)
            return cast(TrainingResult, None)

        monkeypatch.setattr(runtime.trainer, "train", train)
        first = asyncio.create_task(runtime.train(job(tmp_path).batch))
        assert await asyncio.to_thread(entered.wait, 10)
        second = asyncio.create_task(
            runtime.train(job(tmp_path).batch.model_copy(update={"batch_id": "batch-2"}))
        )
        await asyncio.sleep(0)
        assert observed == ["batch-1"]
        release.set()
        await asyncio.gather(first, second)
        assert observed == ["batch-1", "batch-2"]
        await runtime.close()
        with pytest.raises(ClaasTrainingError, match="closed"):
            await runtime.train(job(tmp_path).batch)

    asyncio.run(exercise())


def test_close_cancellation_joins_one_cleanup_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canceled close waits until native state is released and never invokes close twice."""
    entered, release = threading.Event(), threading.Event()
    count = 0

    async def exercise() -> None:
        """Cancel a close waiter while its owned worker is still cleaning up."""
        runtime = ResidentVerlRuntime(spec(), ResidentVerlSettings(checkpoint_root=tmp_path))

        def close() -> None:
            """Hold inert cleanup so the test can check the ownership boundary."""
            nonlocal count
            count += 1
            entered.set()
            assert release.wait(10)

        monkeypatch.setattr(runtime.trainer, "close", close)
        task = asyncio.create_task(runtime.close())
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await runtime.close()
        assert count == 1

    asyncio.run(exercise())
