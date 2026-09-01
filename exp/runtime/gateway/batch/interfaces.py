"""Host-installed service seams for the batch lane, with no implementations.

The engine owns job lifecycle and provider protocol; the host owns identity,
persistence, catalog truth, and money. Every protocol here is content-aware
only where the batch product requires it (line bodies inside stored files);
accounting stays content-free, mirroring the synchronous lane's ledger.
"""

from __future__ import annotations

from typing import Protocol

from exp.runtime.gateway.batch.contracts import (
    BatchDeployment,
    BatchFile,
    BatchJob,
    BatchLine,
    BatchLineResult,
)


class BatchStore(Protocol):
    """Durable persistence for batch jobs, installed by the host."""

    def create_job(self, *, job: BatchJob) -> None:
        """Persist one newly submitted job before its public object is returned."""
        ...

    def load_job(self, *, batch_id: str, organization_id: str) -> BatchJob | None:
        """Return one job owned by the organization, or None when absent."""
        ...

    def save_job(self, *, job: BatchJob) -> None:
        """Persist the updated state of one existing job."""
        ...

    def list_jobs(self, *, organization_id: str, limit: int, after: str | None) -> list[BatchJob]:
        """Return the organization's jobs, newest first, paged by batch id."""
        ...

    def open_jobs(self) -> list[BatchJob]:
        """Return every job in a non-terminal status, for the poller."""
        ...

    def begin_dispatch(self, *, batch_id: str) -> bool:
        """Atomically claim the one-time dispatch of one job.

        Returns True exactly once per job: the first caller flips the
        persisted dispatch_started flag from False to True and owns the
        provider submission; every later caller gets False. Hosts implement
        this as one atomic compare-and-set write.
        """
        ...


class BatchFileStore(Protocol):
    """Durable content storage for batch input and output files."""

    def store(self, *, file: BatchFile, content: bytes) -> None:
        """Persist one file's metadata and content atomically."""
        ...

    def load_metadata(self, *, file_id: str, organization_id: str) -> BatchFile | None:
        """Return one file's metadata owned by the organization, or None."""
        ...

    def load_content(self, *, file_id: str, organization_id: str) -> bytes | None:
        """Return one file's raw content owned by the organization, or None."""
        ...


class BatchCatalog(Protocol):
    """Host catalog seam resolving explicit batch-callable models only."""

    def batch_deployment(self, *, model: str) -> BatchDeployment | None:
        """Return the batch deployment for one explicitly named batch model.

        Returns None when the name is unknown or names a synchronous model:
        the caller reports the explicit-request contract violation per line.
        """
        ...


class BatchLedger(Protocol):
    """Content-free per-line money seam, mirroring the synchronous mechanics.

    Reserve happens at submit for every accepted line, settle happens once per
    line at retrieval, and release covers lines that terminally produced no
    provider work (cancellation before dispatch, expiry, job-level failure).
    Every method resolves only after the host's write is durable.
    """

    def reserve_line(self, *, job: BatchJob, line: BatchLine) -> int:
        """Reserve one line's estimated cost; returns reserved micro-USD.

        Raises:
            Exception: Any host budget rejection; the engine converts it into
                a whole-job submit refusal before provider dispatch.
        """
        ...

    def settle_line(self, *, job: BatchJob, line: BatchLine, result: BatchLineResult) -> None:
        """Settle one line's actual usage against its reservation, idempotently."""
        ...

    def release_line(self, *, job: BatchJob, line: BatchLine, reason: str) -> None:
        """Release one line's remaining reservation without usage, idempotently."""
        ...


class BatchSecretResolver(Protocol):
    """Late-bound resolver for provider credential references, host-installed."""

    def resolve(self, reference: str) -> str:
        """Resolve one configured reference without logging or persisting it."""
        ...
