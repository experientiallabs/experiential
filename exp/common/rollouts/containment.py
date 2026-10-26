"""Host containment guarantees that a local executable environment enforces for every session.

A guarantee is named only when the host enforces it for every session the runtime opens; a
runtime that cannot apply a declared guarantee refuses the session instead of running weaker.
The :class:`ContainmentLevel` is recorded in each sandbox cell binding as the execution receipt,
so an evaluation can require guarantees before dispatch and replay compares the exact level.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ContractModel


class ContainmentGuarantee(StrEnum):
    """One independently enforced property of a contained local process session.

    Attributes:
        SESSION_ISOLATION: The command runs in its own session and process group.
        DESCENDANT_CONTAINMENT: Forked, detached, and re-parented descendants stay inside a
            kernel-bounded tree that is killed with the session.
        VERIFIED_CLEANUP: A session closes cleanly only after kernel evidence that no process
            remains and its private workspace is removed; otherwise cleanup fails closed.
        PRIVATE_WORKSPACE: A fresh owner-only working directory that is also ``HOME`` and
            ``TMPDIR``, deleted at close.
        ENVIRONMENT_ALLOWLIST: Only documented locale, path, and timezone variables reach the
            command; parent credentials and descriptors are never inherited.
        CPU_TIME_LIMIT: Each process is killed after its configured CPU seconds.
        MEMORY_LIMIT: Each process is limited to its configured address space.
        FILE_SIZE_LIMIT: No written file may exceed the configured size.
        PROCESS_COUNT_LIMIT: The whole session may hold at most the configured process count.
        NO_NEW_PRIVILEGES: Set-user-ID and file capabilities cannot raise privileges.
        HOST_PROCESS_ISOLATION: Processes outside the session cannot be seen by PID, signaled,
            traced, or have their environment or memory read.
        FILESYSTEM_WRITE_CONFINEMENT: File content writes, truncations, creations, links,
            renames, and deletions are confined to the private workspace (plus the null, zero,
            and full devices). Metadata of existing files (mode, owner, timestamps, extended
            attributes) is not confined: Landlock has no right for it, so a command can still
            change it on any host path the invoking user owns.
        NETWORK_DENIED: No IP network, including loopback, and no abstract Unix sockets outside
            the session. Filesystem Unix sockets the user can reach are not covered.
    """

    SESSION_ISOLATION = "session_isolation"
    DESCENDANT_CONTAINMENT = "descendant_containment"
    VERIFIED_CLEANUP = "verified_cleanup"
    PRIVATE_WORKSPACE = "private_workspace"
    ENVIRONMENT_ALLOWLIST = "environment_allowlist"
    CPU_TIME_LIMIT = "cpu_time_limit"
    MEMORY_LIMIT = "memory_limit"
    FILE_SIZE_LIMIT = "file_size_limit"
    PROCESS_COUNT_LIMIT = "process_count_limit"
    NO_NEW_PRIVILEGES = "no_new_privileges"
    HOST_PROCESS_ISOLATION = "host_process_isolation"
    FILESYSTEM_WRITE_CONFINEMENT = "filesystem_write_confinement"
    NETWORK_DENIED = "network_denied"


def canonical_guarantees(
    values: Iterable[ContainmentGuarantee],
) -> tuple[ContainmentGuarantee, ...]:
    """Return guarantees in their one canonical sorted, duplicate-free order.

    Args:
        values: Guarantees in any order.

    Returns:
        The sorted tuple used by every persisted containment field.
    """
    return tuple(sorted(set(values)))


class ContainmentLevel(ContractModel):
    """The exact containment a runtime enforces for every session, recorded as the receipt.

    Attributes:
        protocol: Versioned identity of the enforcing implementation; a behavior change bumps
            it so receipts produced by different enforcement never compare equal.
        guarantees: Canonically sorted guarantees enforced for every session.
    """

    protocol: str = Field(min_length=1, max_length=128)
    guarantees: tuple[ContainmentGuarantee, ...]

    @field_validator("guarantees")
    @classmethod
    def _require_canonical_order(
        cls,
        value: tuple[ContainmentGuarantee, ...],
    ) -> tuple[ContainmentGuarantee, ...]:
        """Reject an unsorted or repeated guarantee list so equal levels digest equally."""
        if value != canonical_guarantees(value):
            raise ValueError("containment guarantees must be sorted and unique")
        return value

    def missing(
        self,
        required: Iterable[ContainmentGuarantee],
    ) -> tuple[ContainmentGuarantee, ...]:
        """Return the required guarantees this level does not enforce.

        Args:
            required: Guarantees an evaluation requires before dispatch.

        Returns:
            The canonically ordered unmet guarantees, empty when all are enforced.
        """
        return canonical_guarantees(set(required).difference(self.guarantees))


class UnavailableGuarantee(ContractModel):
    """One guarantee the host cannot enforce, with the reason a caller can act on.

    Attributes:
        guarantee: The guarantee that is not enforced.
        reason: What the host lacks, in operator-facing terms.
    """

    guarantee: ContainmentGuarantee
    reason: str = Field(min_length=1, max_length=512)


class ContainmentReport(ContractModel):
    """Startup capability detection for one host, reported before any session runs.

    Attributes:
        platform: The ``sys.platform`` value the report describes.
        level: Enforced protocol and guarantees; empty guarantees when ``refusal`` is set.
        mechanisms: Kernel mechanisms verified at detection and re-verified per session.
        unavailable: Every guarantee not in ``level`` with the reason it is missing.
        refusal: Why local execution is refused entirely on this host, or None when supported.
    """

    platform: str = Field(min_length=1, max_length=64)
    level: ContainmentLevel
    mechanisms: tuple[str, ...]
    unavailable: tuple[UnavailableGuarantee, ...]
    refusal: str | None = None

    @model_validator(mode="after")
    def _require_complete_disjoint_accounting(self) -> ContainmentReport:
        """Require every guarantee to be either enforced or explained, never both."""
        unavailable = tuple(item.guarantee for item in self.unavailable)
        if len(set(unavailable)) != len(unavailable):
            raise ValueError("unavailable containment guarantees must be unique")
        if set(unavailable) & set(self.level.guarantees):
            raise ValueError("a containment guarantee cannot be both enforced and unavailable")
        if set(unavailable) | set(self.level.guarantees) != set(ContainmentGuarantee):
            raise ValueError("a containment report must account for every guarantee")
        if self.refusal is not None and self.level.guarantees:
            raise ValueError("a refused containment report cannot enforce guarantees")
        return self


def containment_report(
    *,
    platform: str,
    protocol: str,
    enforced: Iterable[ContainmentGuarantee],
    reasons: Mapping[ContainmentGuarantee, str],
    mechanisms: Iterable[str],
    refusal: str | None = None,
) -> ContainmentReport:
    """Build a complete report where every unenforced guarantee carries its reason.

    Args:
        platform: The ``sys.platform`` value the report describes.
        protocol: Versioned identity of the enforcing implementation.
        enforced: Guarantees enforced for every session.
        reasons: Reasons for unenforced guarantees; ``refusal`` explains any omitted one.
        mechanisms: Kernel mechanisms verified at detection.
        refusal: Why local execution is refused entirely, or None when supported.

    Returns:
        The validated report with canonical ordering.

    Raises:
        ValueError: An unenforced guarantee has no reason and there is no refusal.
    """
    level = ContainmentLevel(protocol=protocol, guarantees=canonical_guarantees(enforced))
    unavailable: list[UnavailableGuarantee] = []
    for guarantee in ContainmentGuarantee:
        if guarantee in level.guarantees:
            continue
        reason = reasons.get(guarantee, refusal)
        if reason is None:
            raise ValueError(f"containment guarantee {guarantee.value!r} needs a reason")
        unavailable.append(UnavailableGuarantee(guarantee=guarantee, reason=reason))
    return ContainmentReport(
        platform=platform,
        level=level,
        mechanisms=tuple(sorted(set(mechanisms))),
        unavailable=tuple(unavailable),
        refusal=refusal,
    )
