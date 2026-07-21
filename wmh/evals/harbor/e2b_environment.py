"""Harbor E2B task environment with safe template builds and paced sandbox creates.

`WmhE2BEnvironment` subclasses harbor's E2B backend to fix three operational hazards, keeping
everything else (mounts, network policy, uploads, verification) harbor's:

- **Resource-qualified template aliases.** Harbor names templates by environment content only,
  so tasks sharing content but differing in cpu/memory collide. The alias here embeds the full
  resource identity (see `e2b_template_policy`), byte-compatible with the templates already
  built on the account.
- **Single-submit builds.** Harbor wraps its combined submit-and-wait `AsyncTemplate.build` in
  a tenacity retry (harbor/environments/e2b.py `_create_template`), so one transport blip during
  a long build replays the submission and pays for a duplicate build. This class submits exactly
  once via `build_in_background`, then polls the idempotent `get_build_status` GET, retrying
  ONLY transport/rate-limit errors there. A per-alias single-flight lock stops two trials of the
  same task from both seeing "missing" and double-submitting, and a global semaphore keeps
  concurrent builds under the account limit.
- **Create pacing.** Every sandbox create routes through the same process-wide 4/sec admission
  gate as wmh's own pi-worker sandboxes, so the two consumers cannot jointly exceed E2B's
  published account rate.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, override

import httpx
from e2b import AsyncSandbox, AsyncTemplate, Template
from e2b.exceptions import BuildException, RateLimitException
from e2b.template.main import TemplateClass
from e2b.template.types import BuildInfo, TemplateBuildStatus, TemplateBuildStatusResponse
from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
    TpuSpec,
)
from harbor.models.trial.config import ResourceMode, ServiceVolumeConfig
from harbor.models.trial.paths import TrialPaths

from wmh.evals.harbor.e2b_template_policy import (
    E2B_TEMPLATE_BUILD_CONCURRENCY,
    E2B_TEMPLATE_BUILD_STATUS_POLL_INTERVAL_MS,
    E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS,
    E2BTemplateResources,
    qualify_harbor_e2b_template_name,
    resolve_e2b_template_resources,
)
from wmh.harness.e2b_sandbox import acquire_e2b_create_slot_async

# Harbor's own _create_sandbox contract: two attempts with a short pause. Replicated here so
# routing through the create gate does not change harbor's retry behavior.
_CREATE_ATTEMPTS = 2
_CREATE_RETRY_DELAY_S = 1.0

# Per-alias single-flight locks plus one build semaphore, keyed BY EVENT LOOP: harbor's base
# start() does exists -> build, so without the lock two racing attempts of the same task both
# see "missing" and double-submit one paid build. The keying matters: each HarborScorer.score()
# runs its job under a fresh asyncio.run loop, and an asyncio primitive that gained waiters in
# one loop binds to it (Python 3.12), so reusing it from the next job's loop raises
# "bound to a different event loop". Concurrent trials of one job share one loop, which is the
# only place the race exists; the registry entry dies with its loop (WeakKeyDictionary).
_CONTROL_GUARD = threading.Lock()
_LOOP_CONTROLS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    tuple[dict[str, asyncio.Lock], asyncio.Semaphore],
] = weakref.WeakKeyDictionary()


def _template_controls(template_name: str) -> tuple[asyncio.Lock, asyncio.Semaphore]:
    """The running loop's single-flight lock for `template_name` plus its build semaphore."""
    loop = asyncio.get_running_loop()
    with _CONTROL_GUARD:
        controls = _LOOP_CONTROLS.get(loop)
        if controls is None:
            controls = ({}, asyncio.Semaphore(E2B_TEMPLATE_BUILD_CONCURRENCY))
            _LOOP_CONTROLS[loop] = controls
        locks, semaphore = controls
        lock = locks.get(template_name)
        if lock is None:
            lock = asyncio.Lock()
            locks[template_name] = lock
        return lock, semaphore


# e2b converts a non-2xx status GET into BuildException(f"{status_code}: ...") (only 429 becomes
# RateLimitException, 401 AuthenticationException); a transient server-side 5xx is as retryable
# as a transport error, while 4xx and terminal build states stay fatal.
_SERVER_ERROR_BUILD_EXCEPTION = re.compile(r"^5\d\d: ")


def _is_retryable_status_error(error: Exception) -> bool:
    if isinstance(error, (httpx.TransportError, RateLimitException)):
        return True
    return isinstance(error, BuildException) and bool(
        _SERVER_ERROR_BUILD_EXCEPTION.match(str(error))
    )


async def _get_template_build_status(
    build_info: BuildInfo,
    *,
    logs_offset: int,
) -> TemplateBuildStatusResponse:
    """Retry only the idempotent GET for one already-submitted exact build."""
    for attempt in range(len(E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS) + 1):
        try:
            return await AsyncTemplate.get_build_status(build_info, logs_offset=logs_offset)
        except Exception as error:  # noqa: BLE001 - classified below; non-transient re-raises
            if not _is_retryable_status_error(error):
                raise
            if attempt == len(E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS):
                raise
            await asyncio.sleep(E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS[attempt] / 1_000)
    raise AssertionError("unreachable template build status retry state")


async def _wait_for_template_build(build_info: BuildInfo) -> None:
    """Wait for one submitted build without ever replaying its submission."""
    logs_offset = 0
    while True:
        response = await _get_template_build_status(build_info, logs_offset=logs_offset)
        if (
            response.template_id != build_info.template_id
            or response.build_id != build_info.build_id
        ):
            raise RuntimeError("E2B template build status identity disagreement")
        logs_offset += len(response.log_entries)
        if response.status is TemplateBuildStatus.READY:
            return
        if response.status is TemplateBuildStatus.ERROR:
            message = response.reason.message if response.reason else "E2B template build failed"
            raise BuildException(message)
        if response.status not in (
            TemplateBuildStatus.BUILDING,
            TemplateBuildStatus.WAITING,
        ):
            raise BuildException("E2B template build returned an unknown status")
        await asyncio.sleep(E2B_TEMPLATE_BUILD_STATUS_POLL_INTERVAL_MS / 1_000)


class WmhE2BEnvironment(E2BEnvironment):
    """Harbor's E2B environment with qualified aliases, safe builds, and paced creates."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        override_tpu: TpuSpec | None = None,
        cpu_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        memory_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        persistent_env: dict[str, str] | None = None,
        mounts: list[ServiceVolumeConfig] | None = None,
        network_policy: NetworkPolicy | None = None,
        phase_network_policies: Sequence[NetworkPolicy] | None = None,
        extra_docker_compose: Sequence[Path | str] | None = None,
        **_ignored: object,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config.model_copy(deep=True),
            logger=logger,
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            override_storage_mb=override_storage_mb,
            override_gpus=override_gpus,
            override_tpu=override_tpu,
            cpu_enforcement_policy=cpu_enforcement_policy,
            memory_enforcement_policy=memory_enforcement_policy,
            persistent_env=persistent_env,
            mounts=mounts,
            network_policy=network_policy,
            phase_network_policies=phase_network_policies,
            extra_docker_compose=extra_docker_compose,
        )
        # Build and create always pass explicit resources: E2B applies account defaults to
        # omitted values, and an implicit default inside a shared alias would be a silent
        # resource collision. IGNORE keeps harbor's semantics (task values not enforced).
        effective_cpu = (
            None if self._cpu_resource_mode is ResourceMode.IGNORE else self.task_env_config.cpus
        )
        effective_memory = (
            None
            if self._memory_resource_mode is ResourceMode.IGNORE
            else self.task_env_config.memory_mb
        )
        self._template_resources = resolve_e2b_template_resources(
            cpu_count=effective_cpu,
            memory_mb=effective_memory,
        )
        self._build_source_kind: Literal["docker_image", "dockerfile"] = (
            "docker_image" if self.task_env_config.docker_image else "dockerfile"
        )
        self._build_source_reference = self.task_env_config.docker_image or self.environment_id
        self._template_name = qualify_harbor_e2b_template_name(
            self._template_name,
            environment_id=self.environment_id,
            build_source_kind=self._build_source_kind,
            build_source_reference=self._build_source_reference,
            resources=self._template_resources,
        )

    @property
    def template_name(self) -> str:
        """Return the resource-qualified E2B template name."""
        return self._template_name

    @property
    def template_resources(self) -> E2BTemplateResources:
        """Return the exact numeric resources used for build and create."""
        return self._template_resources

    @property
    @override
    def _effective_cpus(self) -> int:
        return self._template_resources.cpu_count

    @property
    @override
    def _effective_memory_mb(self) -> int:
        return self._template_resources.memory_mb

    def _template_definition(self) -> TemplateClass:
        if self.task_env_config.docker_image:
            return Template().from_image(image=self.task_env_config.docker_image)
        return Template(file_context_path=str(self.environment_dir)).from_dockerfile(
            dockerfile_content_or_path=str(self._environment_definition_path)
        )

    @override
    async def _create_template(self) -> BuildInfo:
        """Submit once and poll the exact build without replaying submission.

        A submission transport failure has an unknown outcome and propagates: retrying it is
        exactly the duplicate-paid-build bug this override exists to prevent. Once E2B returns
        ``BuildInfo``, only idempotent status GETs for that identity are retried.
        """
        build_info = await AsyncTemplate.build_in_background(
            template=self._template_definition(),
            name=self._template_name,
            cpu_count=self._template_resources.cpu_count,
            memory_mb=self._template_resources.memory_mb,
        )
        await _wait_for_template_build(build_info)
        return build_info

    @override
    async def _create_sandbox(self) -> None:
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
        }
        self._sandbox = None
        for attempt in range(_CREATE_ATTEMPTS):
            await acquire_e2b_create_slot_async()
            try:
                sandbox = await AsyncSandbox.create(
                    template=self._template_name,
                    metadata=metadata,
                    envs=self._startup_env(),
                    timeout=86_400,
                    allow_internet_access=(
                        self.network_policy.network_mode != NetworkMode.NO_NETWORK
                    ),
                    network=self._sandbox_create_network_options(),
                )
            except Exception:  # noqa: BLE001 - preserve Harbor's retry-all create contract
                if attempt + 1 == _CREATE_ATTEMPTS:
                    raise
                await asyncio.sleep(_CREATE_RETRY_DELAY_S)
                continue
            # One cheap identity check: the sandbox must really come from the qualified alias.
            # It lives INSIDE the create retry so one transient get_info failure retries the
            # whole create instead of failing it; a genuine mismatch stays immediately fatal.
            try:
                info = await sandbox.get_info()
            except BaseException as error:
                await self._kill_quietly(sandbox)
                if isinstance(error, Exception) and attempt + 1 < _CREATE_ATTEMPTS:
                    await asyncio.sleep(_CREATE_RETRY_DELAY_S)
                    continue
                raise
            if info.name is not None and info.name != self._template_name:
                await self._kill_quietly(sandbox)
                raise RuntimeError("E2B sandbox template name mismatch")
            self._sandbox = sandbox
            return
        raise RuntimeError("E2B sandbox create returned no sandbox")

    async def _kill_quietly(self, sandbox: AsyncSandbox) -> None:
        """Best-effort kill of a sandbox this environment is abandoning."""
        try:
            await sandbox.kill()
        except Exception:  # noqa: BLE001 - the create/identity error stays authoritative
            self.logger.warning("failed to kill an abandoned E2B sandbox", exc_info=True)

    @override
    async def start(self, force_build: bool) -> None:
        """Harbor's exists-then-build start, made race-free and concurrency-bounded."""
        lock, build_semaphore = _template_controls(self._template_name)
        async with lock:
            if force_build or not await self._does_template_exist():
                self.logger.debug(f"Creating template {self._template_name}")
                async with build_semaphore:
                    await self._create_template()
        await self._create_sandbox()
        if not self._sandbox:
            raise RuntimeError("Sandbox not found but was just created. This should never happen.")
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()
