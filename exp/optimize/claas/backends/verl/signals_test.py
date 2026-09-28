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
    """Production rollout startup and cleanup retain launcher authority over a real SIGTERM."""
    script = """
import asyncio, importlib.metadata, json, os, pathlib, signal, socket, sys, types
from unittest.mock import patch
import ray
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.service.controller_test import Runtime
from exp.optimize.claas.service.execution_test import configuration
from exp.optimize.claas.service.launcher import _process

class CpuServer:
    '''Substitute only GPU engine allocation while keeping actual Ray actor ownership.'''
    def __init__(self, **kwargs):
        '''Accept the production server construction fields without allocating CUDA.'''
        self.options = kwargs
    async def launch_server(self):
        '''Expose the actual production startup await on a CPU Ray actor.'''
        return None

class CpuAdapter:
    '''Substitute CUDA-specific adapter behavior without replacing Ray lifecycle calls.'''
    def __init__(self, config, model, mesh, *, replica_rank):
        '''Retain observable adapter state for the production initialize path.'''
        self.server_handle = None
        self.released = False
    async def release(self):
        '''Record the real initialize path's release of rollout memory.'''
        self.released = True

package = types.ModuleType('verl.workers.rollout.vllm_rollout')
package.__path__ = []
adapter_module = types.ModuleType(package.__name__ + '.vllm_rollout')
adapter_module.ServerAdapter = CpuAdapter
server_module = types.ModuleType(package.__name__ + '.vllm_async_server')
server_module.vLLMHttpServer = CpuServer
sys.modules[package.__name__] = package
sys.modules[adapter_module.__name__] = adapter_module
sys.modules[server_module.__name__] = server_module
from exp.optimize.claas.backends.verl import rollout

version = importlib.metadata.version
real_init = ray.init

def installed_version(name):
    '''Allow the CPU fixture through the explicit pinned GPU dependency preflight.'''
    return '0.22.0' if name == 'vllm' else version(name)

def initialize_cpu(**kwargs):
    '''Start actual Ray with a small object store suitable for bounded CPU validation.'''
    return real_init(**kwargs, object_store_memory=80*1024*1024)

class CpuRayRuntime(Runtime):
    '''Drive the production rollout initializer and closer with real CPU Ray ownership.'''
    async def open(self, spec, resume=None, *, mode):
        '''Initialize production Ray wiring, prove stop bindings, and send actual SIGTERM.'''
        before = {n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM)}
        self.rollout = rollout.ResidentRollout(ResidentVerlSettings(checkpoint_root=directory))
        model = types.SimpleNamespace(tokenizer_path='fixture-unused-by-cpu-adapter')
        with patch.object(rollout.importlib.metadata,'version',installed_version), \
             patch.object(rollout.ray,'init',initialize_cpu), \
             patch.object(rollout,'ExactVllmHttpServer',CpuServer), \
             patch.object(rollout,'init_device_mesh',lambda *_args: None), \
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            await self.rollout.initialize(model,spec)
        assert ray.is_initialized()
        assert self.rollout.server is not None
        assert self.rollout.adapter.released
        assert {n:signal.getsignal(n) for n in before} == before
        asyncio.get_running_loop().call_later(0.1,os.kill,os.getpid(),signal.SIGTERM)
        return await super().open(spec,resume,mode=mode)

    async def close(self):
        '''Join production Ray actor/runtime cleanup before completing the learner receipt.'''
        before = {n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM)}
        await self.rollout.close()
        assert {n:signal.getsignal(n) for n in before} == before
        assert not ray.is_initialized()
        assert self.rollout.server is None
        assert self.rollout.adapter is None
        await super().close()

directory=pathlib.Path(sys.argv[1])
runtime=CpuRayRuntime()
with socket.socket() as reserved:
    reserved.bind(('127.0.0.1',0))
    port=reserved.getsockname()[1]
report=asyncio.run(_process(configuration(directory,mode='run',port=port),runtime,
                            'fixture-key-at-least-16',None))
saved=json.loads((directory/'run-report.json').read_text())
assert saved['status']['state']==report.status.state=='closed',saved
assert saved['status']['failure_type'] is None,saved
assert saved['status']['cleanup_failure_type'] is None,saved
assert runtime.open_count==runtime.close_count==1
assert runtime.optimizations==0
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "run")],
            env={
                **os.environ,
                "RAY_USAGE_STATS_ENABLED": "0",
                # Use this installed interpreter even when pytest has a `uv run` ancestor.
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            },
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            "CPU Ray signal regression exceeded 90 seconds. "
            f"stdout tail: {(error.stdout or b'')[-8192:]!r}; "
            f"stderr tail: {(error.stderr or b'')[-8192:]!r}",
            pytrace=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
