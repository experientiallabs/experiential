"""Explicit per-job operational authority for scoped council runtime construction."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol

from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.provider_grant import RuntimeProviderAuthority
from exp.runtime.models.spend_ceiling import current_spend_ceiling

if TYPE_CHECKING:
    from exp.optimize.evaluation.continuations.job import ContinuationJob


class JobRuntimeProviderAuthority(RuntimeProviderAuthority, Protocol):
    """Operational authority must bind the entire incoming job before runtime source import."""

    def require_job(self, project: ProjectStore, job: ContinuationJob) -> None:
        """Verify actor, checkpoint, counts, horizon, runtime, configuration and lifecycle."""


type RuntimeAuthorityResolver = Callable[
    [ProjectStore, "ContinuationJob"], JobRuntimeProviderAuthority
]


_RESOLVER: ContextVar[RuntimeAuthorityResolver | None] = ContextVar(
    "native-runtime-authority-resolver", default=None
)


@contextmanager
def runtime_authority_scope(resolver: RuntimeAuthorityResolver) -> Iterator[None]:
    """Install a native registered resolver in this context and its copied bounded workers."""
    token = _RESOLVER.set(resolver)
    try:
        yield
    finally:
        _RESOLVER.reset(token)


def runtime_authority(
    project: ProjectStore,
    job: ContinuationJob,
) -> RuntimeProviderAuthority | None:
    """Refuse scoped execution without genuine per-job authority before loading any source."""
    resolver = _RESOLVER.get()
    ceiling = current_spend_ceiling()
    scoped = ceiling is not None and ceiling.envelope is not None
    if resolver is None:
        if scoped:
            raise ValueError("scoped council runtime has no native per-job provider authority")
        return None
    if not scoped:
        raise ValueError("runtime authority cannot be used outside its native spend envelope")
    authority = resolver(project, job)
    authority.require_job(project, job)
    authority.require_configuration(project, job.configuration, runtime_sha256=job.runtime_sha256)
    return authority
