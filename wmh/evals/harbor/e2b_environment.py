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
  ONLY transport/rate-limit errors there. A per-loop per-alias single-flight lock plus a
  process-wide submitted-build registry stop concurrent trials AND concurrent scorer loops from
  both seeing "missing" and double-submitting (followers poll the already-paid build), and a
  process-wide bounded semaphore keeps concurrent builds under the account limit.
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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _SubmittedBuild:
    """The provider identity of one already-submitted template build."""

    template_id: str
    build_id: str


@dataclass(frozen=True)
class _AmbiguousSubmission:
    """A submission whose outcome is unknown: it failed before the client got a build id.

    E2B may have accepted the request (a paid build may be running) even though the client saw
    an exception, so the alias must stay claimed; the next attempt reconciles with the control
    plane before any resubmission.
    """

    error: str


# Build coordination happens at two levels.
#
# Per-loop, per-alias asyncio locks give in-loop single-flight: harbor's base start() does
# exists -> build, so without the lock two racing attempts of the same task both see "missing"
# and double-submit one paid build. They are keyed BY EVENT LOOP because each
# HarborScorer.score() runs its job under a fresh asyncio.run loop, and an asyncio primitive
# that gained waiters binds to its loop (Python 3.12); the entry dies with the loop.
#
# Process-wide, thread-safe state covers CONCURRENT loops (two scorer threads, two candidates):
# a registry of submitted build identities per alias (plain data, safe to share across loops)
# so a second loop polls the first loop's already-paid build instead of resubmitting, and one
# bounded semaphore capping concurrent builds account-wide regardless of how many loops exist.
# The threading guard only ever protects fast dict operations; nothing awaits under it.
_CONTROL_GUARD = threading.Lock()
_LOOP_ALIAS_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()
# alias -> identity of the submitted build; None marks a submission in flight (claimed);
# _AmbiguousSubmission marks a failed submission with an unknown provider-side outcome.
_SUBMITTED_BUILDS: dict[str, _SubmittedBuild | _AmbiguousSubmission | None] = {}
_PROCESS_BUILD_SLOTS = threading.BoundedSemaphore(E2B_TEMPLATE_BUILD_CONCURRENCY)
_BUILD_SLOT_POLL_INTERVAL_S = 0.25
_REGISTRY_POLL_INTERVAL_S = 0.05
# Reconciling an ambiguous submission checks alias visibility over a bounded window because E2B
# alias visibility is eventually consistent: a build that E2B accepted can be invisible for a
# few seconds, so one False must not be read as "rejected" and trigger a duplicate paid build.
# ~6 checks x 5s covers the observed propagation lag with margin.
_AMBIGUOUS_VISIBILITY_CHECKS = 6
_AMBIGUOUS_VISIBILITY_DELAY_S = 5.0


def _template_lock(template_name: str) -> asyncio.Lock:
    """The running loop's single-flight lock for `template_name`."""
    loop = asyncio.get_running_loop()
    with _CONTROL_GUARD:
        locks = _LOOP_ALIAS_LOCKS.get(loop)
        if locks is None:
            locks = {}
            _LOOP_ALIAS_LOCKS[loop] = locks
        lock = locks.get(template_name)
        if lock is None:
            lock = asyncio.Lock()
            locks[template_name] = lock
        return lock


def _claim_build(
    alias: str,
) -> _SubmittedBuild | _AmbiguousSubmission | Literal["owner", "pending"]:
    """Atomically claim the right to submit `alias`, or report what already happened to it.

    An ambiguous entry is converted back into a live claim on return, so exactly one caller
    owns its reconciliation while everyone else sees "pending".
    """
    with _CONTROL_GUARD:
        if alias not in _SUBMITTED_BUILDS:
            _SUBMITTED_BUILDS[alias] = None  # claimed: this caller owns the submission
            return "owner"
        entry = _SUBMITTED_BUILDS[alias]
        if isinstance(entry, _AmbiguousSubmission):
            _SUBMITTED_BUILDS[alias] = None  # claimed: this caller owns the reconciliation
            return entry
        return "pending" if entry is None else entry


def _record_submitted_build(alias: str, entry: _SubmittedBuild | _AmbiguousSubmission) -> None:
    with _CONTROL_GUARD:
        _SUBMITTED_BUILDS[alias] = entry


def _clear_submitted_build(alias: str, expected: _SubmittedBuild | None) -> None:
    """Drop the registry entry (only if it still matches `expected`, when given)."""
    with _CONTROL_GUARD:
        if expected is None or _SUBMITTED_BUILDS.get(alias) == expected:
            _SUBMITTED_BUILDS.pop(alias, None)


async def _acquire_build_slot() -> None:
    """Take one process-wide build slot without blocking the event loop or a worker thread."""
    while not _PROCESS_BUILD_SLOTS.acquire(blocking=False):
        await asyncio.sleep(_BUILD_SLOT_POLL_INTERVAL_S)


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


async def _await_submitted_build(alias: str, submitted: _SubmittedBuild) -> None:
    """Follower path: poll a build another loop already paid for; never submit."""
    build_info = BuildInfo(
        template_id=submitted.template_id,
        build_id=submitted.build_id,
        name=alias,
        alias=alias,
    )
    try:
        await _wait_for_template_build(build_info)
    except BaseException:
        # Clear only if the entry still names the failed build, so a fresh rebuild started by
        # another caller is never clobbered.
        _clear_submitted_build(alias, submitted)
        raise


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
        build_info = await self._submit_template_build()
        await _wait_for_template_build(build_info)
        return build_info

    async def _submit_template_build(self) -> BuildInfo:
        """Submit the build exactly once; never wrapped in any retry."""
        return await AsyncTemplate.build_in_background(
            template=self._template_definition(),
            name=self._template_name,
            cpu_count=self._template_resources.cpu_count,
            memory_mb=self._template_resources.memory_mb,
        )

    async def _ensure_template_built(self, *, force_build: bool) -> None:
        """Build the alias once per process, letting concurrent loops share one paid build.

        The caller holds this loop's per-alias lock, so within one loop this runs once at a
        time. Across loops, the process-wide registry decides: the first claimant submits
        (inside a process-wide build slot) and records the build identity BEFORE polling, so
        any other loop finds the identity and polls THAT build to READY instead of paying for
        a second one. A terminal build failure clears the entry so a later attempt can
        rebuild; a submission whose outcome is unknown stays claimed as ambiguous and must be
        reconciled with the control plane before anyone may resubmit.
        """
        alias = self._template_name
        if force_build:
            _clear_submitted_build(alias, None)
        while True:
            claim = _claim_build(alias)
            if claim == "owner":
                await self._acquire_slot_and_build(alias)
                return
            if isinstance(claim, _SubmittedBuild):
                self.logger.debug(f"Awaiting template {alias} submitted by a concurrent job")
                await _await_submitted_build(alias, claim)
                return
            if isinstance(claim, _AmbiguousSubmission):
                await self._reconcile_ambiguous_submission(alias, claim)
                return
            # "pending": another caller's submission or reconciliation is in flight; wait for
            # its identity to land (poll it) or for its failure to release the claim.
            await asyncio.sleep(_REGISTRY_POLL_INTERVAL_S)

    async def _acquire_slot_and_build(self, alias: str) -> None:
        """Owner path: submit under a process-wide build slot and poll to READY."""
        await _acquire_build_slot()
        try:
            self.logger.debug(f"Creating template {alias}")
            try:
                build_info = await self._submit_template_build()
            except BaseException as error:
                # E2B may have accepted the request before the failure, so a paid build may be
                # running without us holding its id. Keep the alias CLAIMED as ambiguous: no
                # waiter may submit a second paid build until a later attempt reconciles with
                # the control plane. The failure itself propagates (never auto-resubmitted).
                _record_submitted_build(
                    alias,
                    _AmbiguousSubmission(error=f"{type(error).__name__}: {error}"),
                )
                raise
            _record_submitted_build(
                alias,
                _SubmittedBuild(
                    template_id=build_info.template_id,
                    build_id=build_info.build_id,
                ),
            )
            try:
                await _wait_for_template_build(build_info)
            except BaseException:
                _clear_submitted_build(alias, None)  # terminal failure: allow a rebuild
                raise
        finally:
            _PROCESS_BUILD_SLOTS.release()

    async def _reconcile_ambiguous_submission(
        self,
        alias: str,
        ambiguous: _AmbiguousSubmission,
    ) -> None:
        """Resolve an earlier submission of unknown outcome before allowing any resubmission.

        `_claim_build` converted the ambiguous entry into a live claim, so this caller is the
        single reconciler and holds the claim for the whole window (no other loop can resubmit
        meanwhile). The control plane is the authority, but E2B alias visibility is eventually
        consistent: an accepted build can be invisible for a few seconds, so a single False
        must NOT be read as "rejected". Poll alias visibility over a bounded window; if the
        alias appears at any check, the earlier submission went through, so release the claim
        with nothing re-paid. Only if it stays absent across the ENTIRE window may this
        claimant clear the marker and submit fresh. An ambiguous submission never yielded a
        build id, so alias resolution is the completion evidence; there is no exact build to
        poll.
        """
        self.logger.debug(
            f"Reconciling an ambiguous earlier submission of template {alias} ({ambiguous.error})"
        )
        for check in range(_AMBIGUOUS_VISIBILITY_CHECKS):
            try:
                exists = await self._does_template_exist()
            except BaseException:
                _record_submitted_build(alias, ambiguous)  # still unknown: restore the marker
                raise
            if exists:
                _clear_submitted_build(alias, None)
                return
            if check + 1 < _AMBIGUOUS_VISIBILITY_CHECKS:
                await asyncio.sleep(_AMBIGUOUS_VISIBILITY_DELAY_S)
        # Absent across the whole eventual-consistency window: the original submission was
        # truly rejected. Safe to submit a fresh build.
        await self._acquire_slot_and_build(alias)

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
        async with _template_lock(self._template_name):
            if force_build or not await self._does_template_exist():
                await self._ensure_template_built(force_build=force_build)
        await self._create_sandbox()
        if not self._sandbox:
            raise RuntimeError("Sandbox not found but was just created. This should never happen.")
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()
