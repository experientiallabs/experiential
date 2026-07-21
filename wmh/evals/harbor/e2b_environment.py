"""Harbor E2B task environment using WMH's shared sandbox-create admission gate."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType
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
    E2B_TEMPLATE_BUILD_STATUS_POLL_INTERVAL_MS,
    E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS,
    E2BTemplateResources,
    e2b_template_resource_digest,
    qualify_harbor_e2b_template_name,
    resolve_e2b_template_resources,
)
from wmh.harness.e2b_sandbox import acquire_e2b_create_slot_async

_CREATE_ATTEMPTS = 2
_CREATE_RETRY_DELAY_S = 1.0
_PREPARATION_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


async def _get_template_build_status(
    build_info: BuildInfo,
    *,
    logs_offset: int,
) -> TemplateBuildStatusResponse:
    """Retry only the idempotent GET for one already-submitted exact build."""
    for attempt in range(len(E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS) + 1):
        try:
            return await AsyncTemplate.get_build_status(build_info, logs_offset=logs_offset)
        except (httpx.TransportError, RateLimitException):
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


@dataclass(frozen=True)
class PreparedE2BTemplate:
    """Verified provider identity and resources for one prepared template."""

    template_id: str
    build_id: str
    cpu_count: int
    memory_mb: int


_PREPARED_TEMPLATES: dict[str, Mapping[str, PreparedE2BTemplate]] = {}
_PREPARED_TEMPLATES_LOCK = RLock()


def register_prepared_e2b_templates(
    preparation_digest: str,
    templates: Mapping[str, PreparedE2BTemplate],
) -> None:
    """Atomically register one fully verified readiness manifest for scoring."""
    if _PREPARATION_DIGEST_PATTERN.fullmatch(preparation_digest) is None:
        raise ValueError("preparation_digest must be a sha256 digest")
    if not templates:
        raise ValueError("prepared E2B template registry cannot be empty")
    snapshot: dict[str, PreparedE2BTemplate] = {}
    for template_name, record in templates.items():
        if not template_name or not record.template_id or not record.build_id:
            raise ValueError("prepared E2B template identities must be nonempty")
        if record.cpu_count < 1 or record.memory_mb < 128:
            raise ValueError("prepared E2B template resources are invalid")
        snapshot[template_name] = record
    with _PREPARED_TEMPLATES_LOCK:
        existing = _PREPARED_TEMPLATES.get(preparation_digest)
        if existing is not None and dict(existing) != snapshot:
            raise ValueError("preparation digest is already registered with different templates")
        _PREPARED_TEMPLATES[preparation_digest] = MappingProxyType(snapshot)


def _prepared_e2b_template(
    preparation_digest: str | None,
    template_name: str,
) -> PreparedE2BTemplate | None:
    if preparation_digest is None:
        return None
    with _PREPARED_TEMPLATES_LOCK:
        return _PREPARED_TEMPLATES.get(preparation_digest, {}).get(template_name)


class WmhE2BEnvironment(E2BEnvironment):
    """Preserve Harbor's E2B behavior while pacing every provider create attempt."""

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
        require_prebuilt: bool = False,
        preparation_digest: str | None = None,
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
        if require_prebuilt and preparation_digest is None:
            raise ValueError("require_prebuilt requires preparation_digest")
        if not require_prebuilt and preparation_digest is not None:
            raise ValueError("preparation_digest requires require_prebuilt")
        if (
            preparation_digest is not None
            and _PREPARATION_DIGEST_PATTERN.fullmatch(preparation_digest) is None
        ):
            raise ValueError("preparation_digest must be a sha256 digest")
        # Harbor's E2B backend deliberately leaves task-declared storage on the
        # provider default: E2B exposes no storage field on template build or
        # sandbox create. Preserve that backend contract, but reject an explicit
        # runtime override so an operator cannot mistake it for an enforced value.
        if self._override_storage_mb is not None:
            raise ValueError("Harbor E2B does not enforce storage overrides")

        effective_cpu = (
            None if self._cpu_resource_mode is ResourceMode.IGNORE else self.task_env_config.cpus
        )
        effective_memory = (
            None
            if self._memory_resource_mode is ResourceMode.IGNORE
            else self.task_env_config.memory_mb
        )
        resources = resolve_e2b_template_resources(
            cpu_count=effective_cpu,
            memory_mb=effective_memory,
        )
        if effective_cpu is not None and self._override_cpus is not None:
            resources = replace(resources, cpu_source="runtime_override")
        if effective_memory is not None and self._override_memory_mb is not None:
            resources = replace(resources, memory_source="runtime_override")
        self._template_resources = resources
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
        self._require_prebuilt = require_prebuilt
        self._preparation_digest = preparation_digest

    @property
    def template_name(self) -> str:
        """Return the resource-qualified E2B template name."""
        return self._template_name

    @property
    def template_resources(self) -> E2BTemplateResources:
        """Return the exact numeric resources used for build and runtime checks."""
        return self._template_resources

    @property
    def build_source_kind(self) -> Literal["docker_image", "dockerfile"]:
        """Return the Harbor-selected E2B build source kind."""
        return self._build_source_kind

    @property
    def build_source_reference(self) -> str:
        """Return the canonical build source identity."""
        return self._build_source_reference

    @property
    def template_resource_digest(self) -> str:
        """Return the complete content, resource, and provider-policy digest."""
        return e2b_template_resource_digest(
            environment_id=self.environment_id,
            build_source_kind=self.build_source_kind,
            build_source_reference=self.build_source_reference,
            resources=self.template_resources,
        )

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

        A submission transport failure has an unknown outcome and propagates. Once E2B
        returns ``BuildInfo``, only idempotent status GETs for that identity are retried.
        A fresh run after an ambiguous submission is safe only after separate provider
        reconciliation proves there are no active builds; an alias 404 is insufficient.
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
        expected = _prepared_e2b_template(
            self._preparation_digest,
            self._template_name,
        )
        if self._require_prebuilt and expected is None:
            raise RuntimeError("E2B template is not registered as prepared")
        if expected is not None and (
            expected.cpu_count != self._template_resources.cpu_count
            or expected.memory_mb != self._template_resources.memory_mb
        ):
            raise RuntimeError("E2B registered resource mismatch")
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
        }
        template_reference = (
            self._template_name
            if expected is None
            else f"{self._template_name}:{expected.build_id}"
        )
        for attempt in range(_CREATE_ATTEMPTS):
            await acquire_e2b_create_slot_async()
            try:
                self._sandbox = await AsyncSandbox.create(
                    template=template_reference,
                    metadata=metadata,
                    envs=self._startup_env(),
                    timeout=86_400,
                    allow_internet_access=(
                        self.network_policy.network_mode != NetworkMode.NO_NETWORK
                    ),
                    network=self._sandbox_create_network_options(),
                )
                break
            except Exception:  # noqa: BLE001 - preserve Harbor's retry-all create contract
                self._sandbox = None
                if attempt + 1 == _CREATE_ATTEMPTS:
                    raise
                await asyncio.sleep(_CREATE_RETRY_DELAY_S)
        if self._sandbox is None:
            raise RuntimeError("E2B sandbox create returned no sandbox")
        try:
            info = await self._sandbox.get_info()
            if expected is None and info.name is not None and info.name != self._template_name:
                raise RuntimeError("E2B sandbox template name mismatch")
            if expected is not None and info.template_id != expected.template_id:
                raise RuntimeError("E2B sandbox template identity mismatch")
            if (
                info.cpu_count != self._template_resources.cpu_count
                or info.memory_mb != self._template_resources.memory_mb
            ):
                raise RuntimeError("E2B sandbox resource mismatch")
        except BaseException:
            sandbox = self._sandbox
            self._sandbox = None
            await sandbox.kill()
            raise

    @override
    async def start(self, force_build: bool) -> None:
        """Start a task environment, failing closed when scoring requires readiness."""
        if not self._require_prebuilt:
            await super().start(force_build)
            return
        if force_build:
            raise ValueError("force_build is incompatible with prepared Harbor E2B scoring")
        if _prepared_e2b_template(self._preparation_digest, self._template_name) is None:
            raise RuntimeError("Harbor E2B template is not prepared for this score request")
        await self._create_sandbox()
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()
