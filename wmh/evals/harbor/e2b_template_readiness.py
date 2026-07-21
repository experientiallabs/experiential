"""Crash-safe preparation and verification of Harbor E2B task templates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from harbor.environments.factory import EnvironmentFactory
from harbor.models.job.config import JobConfig
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.core.types import JsonObject
from wmh.evals.harbor.e2b_environment import (
    PreparedE2BTemplate,
    WmhE2BEnvironment,
    register_prepared_e2b_templates,
)
from wmh.evals.harbor.e2b_template_control import (
    E2BTemplateControlIdentity,
    E2BTemplateNotFound,
    inspect_e2b_template,
)
from wmh.evals.harbor.e2b_template_policy import (
    E2B_TEMPLATE_BUILD_CONCURRENCY,
    e2b_template_readiness_policy_payload,
)
from wmh.evals.harbor.tasks import ResolvedHarborTaskSet

_SCHEMA_VERSION = 1
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class E2BTemplateReadinessEntry(BaseModel):
    """Opaque durable evidence for one resource-qualified provider template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(pattern=_DIGEST_PATTERN)
    resource_digest: str = Field(pattern=_DIGEST_PATTERN)
    template_id: str = Field(min_length=1)
    build_id: str = Field(min_length=1)
    cpu_count: int = Field(strict=True, ge=1)
    memory_mb: int = Field(strict=True, ge=128)


class E2BTemplateResourceCount(BaseModel):
    """Aggregate resource histogram without task or template names."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_count: int = Field(strict=True, ge=1)
    memory_mb: int = Field(strict=True, ge=128)
    count: int = Field(strict=True, ge=1)


class E2BTemplateReadinessReceipt(BaseModel):
    """Atomic partial or complete readiness evidence for one exact plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = _SCHEMA_VERSION
    plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    expected_set_digest: str = Field(pattern=_DIGEST_PATTERN)
    mapping_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    complete: bool
    context_count: int = Field(strict=True, ge=1)
    agent_context_count: int = Field(strict=True, ge=1)
    separate_verifier_context_count: int = Field(strict=True, ge=0)
    unique_template_count: int = Field(strict=True, ge=1)
    ready_before_count: int = Field(strict=True, ge=0)
    built_count: int = Field(strict=True, ge=0)
    build_concurrency: int = Field(strict=True, ge=1)
    context_resource_histogram: tuple[E2BTemplateResourceCount, ...]
    template_resource_histogram: tuple[E2BTemplateResourceCount, ...]
    policy: dict[str, int | str | bool | tuple[int, ...]]
    entries: tuple[E2BTemplateReadinessEntry, ...]

    @field_validator("complete", mode="before")
    @classmethod
    def _require_boolean_complete(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("readiness complete must be boolean")
        return value

    @model_validator(mode="after")
    def _validate_counts(self) -> E2BTemplateReadinessReceipt:
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError("readiness receipt contains duplicate entry keys")
        if self.agent_context_count + self.separate_verifier_context_count != self.context_count:
            raise ValueError("readiness receipt context counts differ")
        if len(self.entries) > self.unique_template_count:
            raise ValueError("readiness receipt has too many entries")
        if self.ready_before_count + self.built_count != len(self.entries):
            raise ValueError("readiness receipt counts differ from completed entries")
        if self.complete and len(self.entries) != self.unique_template_count:
            raise ValueError("complete readiness receipt requires every entry")
        if self.complete != (self.mapping_digest is not None):
            raise ValueError("complete readiness receipt requires exactly one mapping digest")
        return self

    @property
    def receipt_digest(self) -> str:
        """Return the canonical digest used by run evidence and checkpoint inputs."""
        return _digest_json(self.model_dump(mode="json"))


@dataclass(frozen=True)
class _TemplateSpec:
    environment: WmhE2BEnvironment
    key: str
    resource_digest: str
    build_timeout_sec: float


@dataclass(frozen=True)
class _PreparedResult:
    spec: _TemplateSpec
    entry: E2BTemplateReadinessEntry
    built: bool


class E2BTemplateReadinessPlan:
    """One exact opaque set of task and verifier templates required by a score."""

    def __init__(
        self,
        *,
        specs: tuple[_TemplateSpec, ...],
        context_resource_histogram: tuple[E2BTemplateResourceCount, ...],
        agent_context_count: int,
        separate_verifier_context_count: int,
        task_set: ResolvedHarborTaskSet,
    ) -> None:
        context_count = sum(item.count for item in context_resource_histogram)
        if (
            not specs
            or context_count < len(specs)
            or agent_context_count < 1
            or separate_verifier_context_count < 0
            or agent_context_count + separate_verifier_context_count != context_count
        ):
            raise ValueError("E2B readiness plan must contain runtime contexts")
        self._specs = specs
        self.context_count = context_count
        self.agent_context_count = agent_context_count
        self.separate_verifier_context_count = separate_verifier_context_count
        self.unique_template_count = len(specs)
        self._context_resource_histogram = context_resource_histogram
        self._template_resource_histogram = _resource_histogram(
            tuple(spec.environment for spec in specs)
        )
        self._task_set = task_set
        self._expected_by_key = {spec.key: spec for spec in specs}
        if len(self._expected_by_key) != len(specs):
            raise ValueError("E2B readiness plan contains duplicate qualified keys")
        self._identity_payload = self._build_identity_payload()
        self.plan_digest = _digest_json(self._identity_payload)
        self.expected_set_digest = _digest_json(self._template_identity_payload())
        self._receipt: E2BTemplateReadinessReceipt | None = None
        self._receipt_path: Path | None = None
        self._verified = False

    @classmethod
    def create(
        cls,
        *,
        job_config: JobConfig,
        task_set: ResolvedHarborTaskSet,
    ) -> E2BTemplateReadinessPlan:
        """Derive exact task and separate-verifier build contexts without provider calls."""
        if job_config.environment.force_build:
            raise ValueError("prepared Harbor E2B scoring does not allow force_build")
        reserved = {"require_prebuilt", "preparation_digest"} & set(job_config.environment.kwargs)
        if reserved:
            raise ValueError(
                f"prepared Harbor E2B scoring owns reserved environment kwargs: {sorted(reserved)}"
            )
        task_set.verify()
        runtime_config = job_config.environment.model_copy(deep=True)
        timeout_multiplier = (
            job_config.environment_build_timeout_multiplier
            if job_config.environment_build_timeout_multiplier is not None
            else job_config.timeout_multiplier
        )
        contexts: list[tuple[WmhE2BEnvironment, float]] = []
        task_inputs = task_set.task_inputs()
        separate_verifier_context_count = 0
        for definition, task_dir in task_inputs:
            paths = TaskPaths(task_dir)
            build_timeout_sec = definition.environment.build_timeout_sec * timeout_multiplier
            contexts.append(
                (
                    _create_environment(
                        runtime_config=runtime_config,
                        job_config=job_config,
                        environment_dir=paths.environment_dir,
                        task_environment=definition.environment,
                    ),
                    build_timeout_sec,
                )
            )
            steps = definition.steps or [None]
            for step in steps:
                verifier_environment = resolve_effective_verifier_env_config(definition, step)
                if verifier_environment is None:
                    continue
                separate_verifier_context_count += 1
                verifier_dir = paths.tests_dir
                if step is not None and paths.step_tests_dir(step.name).exists():
                    verifier_dir = paths.step_tests_dir(step.name)
                verifier_runtime_config = runtime_config.model_copy(
                    update={"extra_docker_compose": []},
                    deep=True,
                )
                contexts.append(
                    (
                        _create_environment(
                            runtime_config=verifier_runtime_config,
                            job_config=job_config,
                            environment_dir=verifier_dir,
                            task_environment=verifier_environment,
                        ),
                        build_timeout_sec,
                    )
                )

        by_name: dict[str, _TemplateSpec] = {}
        for environment, timeout_sec in contexts:
            key = _digest_text(environment.template_name)
            spec = _TemplateSpec(
                environment=environment,
                key=key,
                resource_digest=f"sha256:{environment.template_resource_digest}",
                build_timeout_sec=timeout_sec,
            )
            existing = by_name.get(environment.template_name)
            if existing is not None:
                if (
                    existing.key != spec.key
                    or existing.resource_digest != spec.resource_digest
                    or existing.environment.template_resources.cpu_count
                    != spec.environment.template_resources.cpu_count
                    or existing.environment.template_resources.memory_mb
                    != spec.environment.template_resources.memory_mb
                ):
                    raise ValueError("qualified E2B template identity collision")
                if spec.build_timeout_sec > existing.build_timeout_sec:
                    by_name[environment.template_name] = spec
                continue
            by_name[environment.template_name] = spec
        specs = tuple(sorted(by_name.values(), key=lambda spec: spec.key))
        context_histogram = _resource_histogram(
            tuple(environment for environment, _timeout_sec in contexts)
        )
        return cls(
            specs=specs,
            context_resource_histogram=context_histogram,
            agent_context_count=len(task_inputs),
            separate_verifier_context_count=separate_verifier_context_count,
            task_set=task_set,
        )

    def identity_payload(self) -> JsonObject:
        """Return the task-opaque semantic plan bound into scorer identity."""
        return json.loads(json.dumps(self._identity_payload))

    def aggregate_payload(self) -> JsonObject:
        """Return outcome-blind aggregate preflight evidence without template identities."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "policy": e2b_template_readiness_policy_payload(),
            "context_count": self.context_count,
            "agent_context_count": self.agent_context_count,
            "separate_verifier_context_count": self.separate_verifier_context_count,
            "unique_template_count": self.unique_template_count,
            "context_resource_histogram": [
                item.model_dump(mode="json") for item in self._context_resource_histogram
            ],
            "template_resource_histogram": [
                item.model_dump(mode="json") for item in self._template_resource_histogram
            ],
        }

    @property
    def mapping_digest(self) -> str:
        """Return the complete actual provider mapping digest after receipt load."""
        if self._receipt is None or self._receipt.mapping_digest is None:
            raise RuntimeError("E2B readiness mapping is not complete")
        return self._receipt.mapping_digest

    @property
    def receipt_digest(self) -> str:
        """Return the complete durable receipt digest after receipt load."""
        if self._receipt is None or not self._receipt.complete:
            raise RuntimeError("E2B readiness receipt is not complete")
        return self._receipt.receipt_digest

    def receipt_bytes(self) -> bytes:
        """Return the exact verified complete receipt bytes for checkpoint ownership."""
        if self._receipt is None or not self._receipt.complete or self._receipt_path is None:
            raise RuntimeError("E2B readiness receipt is not complete")
        raw = _read_regular_text(self._receipt_path)
        try:
            on_disk = E2BTemplateReadinessReceipt.model_validate_json(raw)
        except ValueError as error:
            raise RuntimeError("E2B readiness receipt is invalid") from error
        self._validate_receipt(on_disk)
        if on_disk != self._receipt:
            raise RuntimeError("E2B readiness receipt changed on disk")
        return raw.encode("utf-8")

    @property
    def receipt_file_hash(self) -> str:
        """Return the SHA-256 hash of the exact checkpoint preparation bytes."""
        return _digest_bytes(self.receipt_bytes())

    @property
    def verified(self) -> bool:
        """Whether the current provider mapping was inspected in this process."""
        return self._verified

    def load_complete_receipt(self, receipt_path: Path) -> E2BTemplateReadinessReceipt:
        """Load a complete local receipt without making provider calls."""
        self._task_set.verify()
        receipt = self._load_receipt(receipt_path)
        if not receipt.complete:
            raise RuntimeError("checkpoint resume requires a complete E2B readiness receipt")
        self._receipt = receipt
        self._receipt_path = receipt_path
        self._verified = False
        return receipt

    def rebind_complete_receipt(self, receipt_path: Path) -> None:
        """Adopt an identical checkpoint-owned copy without provider access."""
        if self._receipt is None or not self._receipt.complete:
            raise RuntimeError("E2B readiness receipt is not complete")
        receipt = self._load_receipt(receipt_path)
        if not receipt.complete or receipt != self._receipt:
            raise RuntimeError("checkpoint scorer preparation differs from readiness receipt")
        self._receipt_path = receipt_path
        self._verified = False

    async def prepare(self, receipt_path: Path) -> E2BTemplateReadinessReceipt:
        """Prepare a new mapping, leaving failures as audit-only partial evidence."""
        self._task_set.verify()
        with _exclusive_preparation(receipt_path):
            if os.path.lexists(receipt_path):
                raise RuntimeError(
                    "E2B readiness receipt already exists; use a fresh run path after "
                    "preparation failure"
                )
            receipt = self._new_receipt(entries=(), ready_before_count=0, built_count=0)
            self._write_receipt(receipt_path, receipt)
            entries: dict[str, E2BTemplateReadinessEntry] = {}
            ready_before_count = 0
            built_count = 0
            update_lock = asyncio.Lock()
            semaphore = asyncio.Semaphore(E2B_TEMPLATE_BUILD_CONCURRENCY)

            async def prepare_one(spec: _TemplateSpec) -> _PreparedResult:
                nonlocal ready_before_count, built_count
                async with semaphore:
                    result = await self._prepare_one(spec)
                async with update_lock:
                    if result.spec.key in entries:
                        raise RuntimeError("duplicate E2B readiness result")
                    entries[result.spec.key] = result.entry
                    ready_before_count += int(not result.built)
                    built_count += int(result.built)
                    partial = self._new_receipt(
                        entries=tuple(entries.values()),
                        ready_before_count=ready_before_count,
                        built_count=built_count,
                    )
                    self._write_receipt(receipt_path, partial)
                return result

            results = await asyncio.gather(
                *(prepare_one(spec) for spec in self._specs),
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                raise failures[0]
            self._task_set.verify()
            final = self._new_receipt(
                entries=tuple(entries.values()),
                ready_before_count=ready_before_count,
                built_count=built_count,
                complete=True,
            )
            self._write_receipt(receipt_path, final)
            self._receipt = final
            self._receipt_path = receipt_path
            await self.verify()
            return final

    async def verify(self) -> E2BTemplateReadinessReceipt:
        """Re-inspect and register the exact complete mapping, rejecting any drift."""
        self._task_set.verify()
        if self._receipt is None or not self._receipt.complete or self._receipt_path is None:
            raise RuntimeError("E2B readiness must be complete before verification")
        on_disk = self._load_receipt(self._receipt_path)
        if on_disk != self._receipt:
            raise RuntimeError("E2B readiness receipt changed on disk")
        expected_entries = {entry.key: entry for entry in self._receipt.entries}
        semaphore = asyncio.Semaphore(E2B_TEMPLATE_BUILD_CONCURRENCY)

        async def verify_one(spec: _TemplateSpec) -> tuple[str, E2BTemplateControlIdentity]:
            async with semaphore:
                observed = await _inspect(spec)
            expected = expected_entries[spec.key]
            if _entry(spec, observed) != expected:
                raise RuntimeError("prepared E2B template mapping changed")
            return spec.environment.template_name, observed

        observed = await asyncio.gather(*(verify_one(spec) for spec in self._specs))
        self._task_set.verify()
        register_prepared_e2b_templates(
            self.mapping_digest,
            {
                name: PreparedE2BTemplate(
                    template_id=identity.template_id,
                    build_id=identity.build_id,
                    cpu_count=identity.cpu_count,
                    memory_mb=identity.memory_mb,
                )
                for name, identity in observed
            },
        )
        self._verified = True
        return self._receipt

    async def _prepare_one(
        self,
        spec: _TemplateSpec,
    ) -> _PreparedResult:
        try:
            observed = await _inspect(spec)
        except E2BTemplateNotFound:
            built = await asyncio.wait_for(
                spec.environment._create_template(),
                timeout=spec.build_timeout_sec,
            )
            observed = await _inspect(spec)
            if (
                built.template_id != observed.template_id
                or built.build_id != observed.build_id
                or built.name != spec.environment.template_name
                or built.alias != spec.environment.template_name
            ):
                raise RuntimeError("fresh build identity differs from E2B control plane") from None
            return _PreparedResult(spec=spec, entry=_entry(spec, observed), built=True)
        entry = _entry(spec, observed)
        return _PreparedResult(spec=spec, entry=entry, built=False)

    def _build_identity_payload(self) -> JsonObject:
        return {
            "schema_version": _SCHEMA_VERSION,
            "policy": e2b_template_readiness_policy_payload(),
            "context_count": self.context_count,
            "agent_context_count": self.agent_context_count,
            "separate_verifier_context_count": self.separate_verifier_context_count,
            "unique_template_count": self.unique_template_count,
            "context_resource_histogram": [
                item.model_dump(mode="json") for item in self._context_resource_histogram
            ],
            "template_resource_histogram": [
                item.model_dump(mode="json") for item in self._template_resource_histogram
            ],
            "templates": self._template_identity_payload(),
        }

    def _template_identity_payload(self) -> list[dict[str, str | int | float]]:
        return [
            {
                "key": spec.key,
                "resource_digest": spec.resource_digest,
                "cpu_count": spec.environment.template_resources.cpu_count,
                "memory_mb": spec.environment.template_resources.memory_mb,
                "build_timeout_sec": spec.build_timeout_sec,
            }
            for spec in self._specs
        ]

    def _new_receipt(
        self,
        *,
        entries: tuple[E2BTemplateReadinessEntry, ...],
        ready_before_count: int,
        built_count: int,
        complete: bool = False,
    ) -> E2BTemplateReadinessReceipt:
        ordered = tuple(sorted(entries, key=lambda entry: entry.key))
        mapping_digest = _digest_json([entry.model_dump(mode="json") for entry in ordered])
        return E2BTemplateReadinessReceipt(
            plan_digest=self.plan_digest,
            expected_set_digest=self.expected_set_digest,
            mapping_digest=mapping_digest if complete else None,
            complete=complete,
            context_count=self.context_count,
            agent_context_count=self.agent_context_count,
            separate_verifier_context_count=self.separate_verifier_context_count,
            unique_template_count=self.unique_template_count,
            ready_before_count=ready_before_count,
            built_count=built_count,
            build_concurrency=E2B_TEMPLATE_BUILD_CONCURRENCY,
            context_resource_histogram=self._context_resource_histogram,
            template_resource_histogram=self._template_resource_histogram,
            policy=e2b_template_readiness_policy_payload(),
            entries=ordered,
        )

    def _load_receipt(self, path: Path) -> E2BTemplateReadinessReceipt:
        try:
            receipt = E2BTemplateReadinessReceipt.model_validate_json(_read_regular_text(path))
        except (OSError, ValueError) as error:
            raise RuntimeError("E2B readiness receipt is invalid") from error
        self._validate_receipt(receipt)
        return receipt

    def _validate_receipt(self, receipt: E2BTemplateReadinessReceipt) -> None:
        if (
            receipt.plan_digest != self.plan_digest
            or receipt.expected_set_digest != self.expected_set_digest
            or receipt.context_count != self.context_count
            or receipt.agent_context_count != self.agent_context_count
            or receipt.separate_verifier_context_count != self.separate_verifier_context_count
            or receipt.unique_template_count != self.unique_template_count
            or receipt.policy != e2b_template_readiness_policy_payload()
            or receipt.build_concurrency != E2B_TEMPLATE_BUILD_CONCURRENCY
            or receipt.context_resource_histogram != self._context_resource_histogram
            or receipt.template_resource_histogram != self._template_resource_histogram
        ):
            raise RuntimeError("E2B readiness receipt differs from the current plan")
        for entry in receipt.entries:
            spec = self._expected_by_key.get(entry.key)
            if spec is None or not _entry_matches_spec(entry, spec):
                raise RuntimeError("E2B readiness receipt contains an unexpected template")
        if receipt.complete:
            expected_keys = set(self._expected_by_key)
            if {entry.key for entry in receipt.entries} != expected_keys:
                raise RuntimeError("complete E2B readiness receipt has the wrong entry set")
            expected_mapping_digest = _digest_json(
                [entry.model_dump(mode="json") for entry in receipt.entries]
            )
            if receipt.mapping_digest != expected_mapping_digest:
                raise RuntimeError("E2B readiness receipt mapping digest differs")

    @staticmethod
    def _write_receipt(path: Path, receipt: E2BTemplateReadinessReceipt) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _require_regular_if_present(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt.model_dump(mode="json"), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _create_environment(
    *,
    runtime_config: TrialEnvironmentConfig,
    job_config: JobConfig,
    environment_dir: Path,
    task_environment: TaskEnvironmentConfig,
) -> WmhE2BEnvironment:
    environment = EnvironmentFactory.create_environment_from_config(
        config=runtime_config,
        environment_dir=environment_dir,
        environment_name="wmh-prepare",
        session_id="wmh-prepare",
        trial_paths=TrialPaths(job_config.jobs_dir / ".wmh-e2b-readiness"),
        task_env_config=task_environment,
    )
    if not isinstance(environment, WmhE2BEnvironment):
        raise TypeError("Harbor E2B readiness requires WmhE2BEnvironment")
    return environment


async def _inspect(spec: _TemplateSpec) -> E2BTemplateControlIdentity:
    resources = spec.environment.template_resources
    return await inspect_e2b_template(
        spec.environment.template_name,
        expected_cpu_count=resources.cpu_count,
        expected_memory_mb=resources.memory_mb,
    )


def _entry(
    spec: _TemplateSpec,
    identity: E2BTemplateControlIdentity,
) -> E2BTemplateReadinessEntry:
    return E2BTemplateReadinessEntry(
        key=spec.key,
        resource_digest=spec.resource_digest,
        template_id=identity.template_id,
        build_id=identity.build_id,
        cpu_count=identity.cpu_count,
        memory_mb=identity.memory_mb,
    )


def _entry_matches_spec(entry: E2BTemplateReadinessEntry, spec: _TemplateSpec) -> bool:
    resources = spec.environment.template_resources
    return (
        entry.resource_digest == spec.resource_digest
        and entry.cpu_count == resources.cpu_count
        and entry.memory_mb == resources.memory_mb
    )


def _resource_histogram(
    environments: tuple[WmhE2BEnvironment, ...],
) -> tuple[E2BTemplateResourceCount, ...]:
    histogram = Counter(
        (
            environment.template_resources.cpu_count,
            environment.template_resources.memory_mb,
        )
        for environment in environments
    )
    return tuple(
        E2BTemplateResourceCount(
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            count=count,
        )
        for (cpu_count, memory_mb), count in sorted(histogram.items())
    )


def _digest_text(value: str) -> str:
    return _digest_bytes(value.encode())


def _digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _digest_bytes(encoded)


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


@contextmanager
def _exclusive_preparation(receipt_path: Path) -> Iterator[None]:
    lock_path = receipt_path.with_name(f"{receipt_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("E2B readiness preparation is already active") from error
    os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _require_regular_if_present(path: Path) -> None:
    if not os.path.lexists(path):
        return
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise RuntimeError("E2B readiness receipt cannot be inspected") from error
    if not stat.S_ISREG(mode):
        raise RuntimeError("E2B readiness receipt must be a regular file")


def _read_regular_text(path: Path) -> str:
    _require_regular_if_present(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError("E2B readiness receipt cannot be read") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("E2B readiness receipt must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
