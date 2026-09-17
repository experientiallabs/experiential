"""Resident public veRL vLLM rollout with upstream CUDA IPC adapter synchronization.

This module is loaded only for explicit run mode. The upstream server owns its
vLLM engine and worker lifetime; CLaaS never launches a standalone vLLM command.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
from typing import cast
from uuid import uuid4

import ray
import torch
from peft import LoraConfig
from torch.distributed.device_mesh import init_device_mesh
from verl.workers.config import HFModelConfig
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.rollout_configuration import rollout_config
from exp.optimize.claas.backends.verl.server import ExactVllmHttpServer
from exp.optimize.claas.training_contracts import ClaasTrainingSpec


class ResidentRollout:
    """Own one upstream async server and transfer only complete LoRA tensors between phases."""

    def __init__(self, settings: ResidentVerlSettings) -> None:
        """Defer Ray and rollout initialization until the trainer is CPU-offloaded."""
        self.settings = settings
        self.server: ray.actor.ActorHandle | None = None
        self.adapter: ServerAdapter | None = None
        self._owns_ray = False
        self._previous_local_world: str | None = None

    async def initialize(self, model: HFModelConfig, spec: ClaasTrainingSpec) -> None:
        """Launch the public veRL server once, retaining CUDA graphs throughout this run."""
        if importlib.metadata.version("vllm") != "0.22.0":
            raise ValueError("run mode requires vllm==0.22.0; install experiential[claas-rollout]")
        if ray.is_initialized():
            raise ValueError(
                "resident run mode owns a private Ray runtime; use an isolated service process"
            )
        config = rollout_config(self.settings, model, spec)
        ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False)
        self._owns_ray = True
        self._previous_local_world = os.environ.get("RAY_LOCAL_WORLD_SIZE")
        os.environ["RAY_LOCAL_WORLD_SIZE"] = "1"
        try:
            server = cast(
                ray.actor.ActorHandle,
                ray.remote(ExactVllmHttpServer)
                .options(
                    name="claas-rollout-" + uuid4().hex,
                    num_cpus=0,
                    runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                )
                .remote(
                    config=config,
                    model_config=model,
                    rollout_mode=RolloutMode.HYBRID,
                    workers=[],
                    replica_rank=0,
                    node_rank=0,
                    gpus_per_node=1,
                    nnodes=1,
                    cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                ),
            )
            self.server = server
            async with asyncio.timeout(self.settings.startup_timeout_seconds):
                await server.launch_server.remote()
                self.adapter = ServerAdapter(
                    config, model, init_device_mesh("cuda", (1,)), replica_rank=0
                )
                self.adapter.server_handle = server
                await self.adapter.release()
        except BaseException:
            await self.close()
            raise

    async def synchronize(
        self, weights: dict[str, torch.Tensor], config: LoraConfig, step: int
    ) -> None:
        """Wake once and install a complete adapter through veRL's bucketed transport."""
        if self.adapter is None:
            raise ValueError("rollout engine is not initialized")
        async with asyncio.timeout(self.settings.operation_timeout_seconds):
            await self.adapter.resume(tags=["weights"])
            await self.adapter.update_weights(
                ((name, tensor.to("cuda")) for name, tensor in weights.items()),
                peft_config=config.to_dict(),
                base_sync_done=True,
                global_steps=step,
            )
            await self.adapter.resume(tags=["kv_cache"])

    async def generate(
        self, prompt: tuple[int, ...], maximum: int, request_id: str, step: int
    ) -> TokenOutput:
        """Return original sampled token IDs and log probabilities from the upstream engine."""
        if self.server is None:
            raise ValueError("rollout engine is not initialized")
        async with asyncio.timeout(self.settings.operation_timeout_seconds):
            output = await self.server.generate_exact.remote(
                list(prompt),
                {
                    "max_tokens": maximum,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "logprobs": True,
                },
                request_id,
            )
        if (
            not isinstance(output, TokenOutput)
            or output.stop_reason != "completed"
            or output.log_probs is None
        ):
            raise ValueError("veRL rollout did not complete with original token probabilities")
        if output.extra_fields.get("global_steps") != step:
            raise ValueError("rollout engine served an unexpected training revision")
        return output

    async def sleep(self) -> None:
        """Offload rollout weights and KV memory before resident training becomes active."""
        if self.adapter is not None:
            async with asyncio.timeout(self.settings.operation_timeout_seconds):
                await self.adapter.release()

    async def close(self) -> None:
        """Terminate the owned Ray actor and runtime before releasing the run's GPU authority."""
        if self.server is not None:
            ray.kill(self.server, no_restart=True)
            self.server = None
        self.adapter = None
        if self._owns_ray:
            ray.shutdown(wait_for_processes=True)
            self._owns_ray = False
            if self._previous_local_world is None:
                os.environ.pop("RAY_LOCAL_WORLD_SIZE", None)
            else:
                os.environ["RAY_LOCAL_WORLD_SIZE"] = self._previous_local_world
