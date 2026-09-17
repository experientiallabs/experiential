"""Async ownership tests with inert workers, distinct from CUDA training proof."""

import asyncio
import threading
from pathlib import Path
from typing import cast

import pytest

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.runtime import ResidentVerlRuntime
from exp.optimize.claas.training_contracts import ClaasTrainingError, TrainingBatch, TrainingResult
from exp.optimize.claas.training_contracts_test import job, spec


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
