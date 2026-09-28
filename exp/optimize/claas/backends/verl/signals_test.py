"""Real CPU Ray and operating-system signals preserve launcher cleanup and run receipts."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from threading import Thread

import pytest
import ray

from exp.optimize.claas.backends.verl.signals import preserve_stop_handlers


def test_failed_ray_initialization_restores_caller_handlers() -> None:
    """Upstream validation runs after installing SIGTERM, so failure must restore ownership."""
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match="Unknown keyword argument"):
        with preserve_stop_handlers():
            ray.init(unknown_test_argument=True)
    assert {number: signal.getsignal(number) for number in before} == before
    assert not ray.is_initialized()


def test_signal_preservation_rejects_worker_thread_before_entering() -> None:
    """An unsupported caller cannot begin Ray work before discovering it cannot restore signals."""
    failures: list[Exception] = []

    def worker() -> None:
        """Attempt lifecycle entry from a thread without process signal authority."""
        try:
            with preserve_stop_handlers():
                raise AssertionError("Ray operation must not start")
        except RuntimeError as error:
            failures.append(error)

    thread = Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(failures) == 1 and "main thread" in str(failures[0])


@pytest.mark.skipif(os.name != "posix", reason="learner process signals require a POSIX host")
def test_real_ray_sigterm_preserves_asyncio_shutdown_and_run_report(tmp_path: Path) -> None:
    """A real SIGTERM after real Ray startup closes the learner instead of exiting through Ray."""
    script = """
import asyncio, json, os, pathlib, signal, socket, sys
import ray
from exp.optimize.claas.backends.verl.signals import preserve_stop_handlers
from exp.optimize.claas.service.controller_test import Runtime
from exp.optimize.claas.service.execution_test import configuration
from exp.optimize.claas.service.launcher import _process

class CpuRayRuntime(Runtime):
    '''Own a real CPU Ray cluster while reusing a provider-free learner fixture.'''
    async def open(self, spec, resume=None, *, mode):
        '''Initialize Ray, prove the loop's signal bindings survive, then request stop.'''
        before = {n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM)}
        with preserve_stop_handlers():
            ray.init(address='local', num_cpus=1, num_gpus=0, include_dashboard=False,
                     object_store_memory=80*1024*1024)
        assert ray.is_initialized()
        assert {n:signal.getsignal(n) for n in before} == before
        asyncio.get_running_loop().call_later(0.1, os.kill, os.getpid(), signal.SIGTERM)
        return await super().open(spec, resume, mode=mode)

    async def close(self):
        '''Join the owned cluster before completing the learner's shutdown receipt.'''
        before = {n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM)}
        with preserve_stop_handlers():
            ray.shutdown(wait_for_processes=True)
        assert {n:signal.getsignal(n) for n in before} == before
        assert not ray.is_initialized()
        await super().close()

directory = pathlib.Path(sys.argv[1])
runtime = CpuRayRuntime()
with socket.socket() as reserved:
    reserved.bind(('127.0.0.1', 0))
    port = reserved.getsockname()[1]
report = asyncio.run(_process(configuration(directory,mode='run',port=port), runtime,
                              'fixture-key-at-least-16', None))
saved = json.loads((directory/'run-report.json').read_text())
assert saved['status']['state'] == report.status.state == 'closed', saved
assert saved['status']['failure_type'] is None, saved
assert saved['status']['cleanup_failure_type'] is None, saved
assert runtime.open_count == runtime.close_count == 1
assert runtime.optimizations == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "run")],
        env={**os.environ, "RAY_USAGE_STATS_ENABLED": "0"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
