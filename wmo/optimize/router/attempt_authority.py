"""Durable external spend authority for one noninteractive hosted attempt."""

from __future__ import annotations

import secrets
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    canonical_json_bytes,
    sha256_bytes,
    stable_id,
)
from wmo.common.core.files import write_bytes_atomic
from wmo.common.core.locks import file_write_lock
from wmo.common.core.money import USD_ZERO, exact_usd
from wmo.common.project import ExportedProjectBundle, ProjectStage
from wmo.optimize.router.spend import (
    ProviderSpendEntry,
    ProviderSpendLedger,
    ProviderSpendStatus,
)

_AUTHORITY_FILE = "authority.json"
_STATE_FILE = "attempt-state.json"
_HAZARD_FILE = "provider-hazard.json"
_AMBIGUITY_FILE = "provider-ambiguity.json"

_HOSTED_STAGE_ORDER = (
    ProjectStage.BUILDING_WORLD_MODEL,
    ProjectStage.OPTIMIZING_ROUTER,
    ProjectStage.COMPLETING_REPORT,
)


class HostedAttemptAuthorityError(ValueError):
    """A hosted attempt authority is absent, unsafe, mismatched, or already ambiguous."""


class HostedAttemptAuthority(ContractModel):
    """Public non-secret identity of one random write-once external spend authority."""

    schema_version: Literal[1] = 1
    attempt_id: ArtifactId
    authority_sha256: Sha256


class HostedAttemptBinding(ContractModel):
    """Write-once Project and finite-ceiling authorization for one hosted attempt."""

    schema_version: Literal[1] = 1
    project_id: ArtifactId
    attempt_id: ArtifactId
    authority_sha256: Sha256
    ceiling_usd: Decimal = Field(gt=0)

    @field_validator("ceiling_usd", mode="before")
    @classmethod
    def _require_exact_ceiling(cls, value: object) -> Decimal:
        """Return one exact numeric(20,6) ceiling."""
        return exact_usd(value)


class HostedProviderHazard(ContractModel):
    """Source-separated reservations retained outside every portable Project bundle."""

    schema_version: Literal[1] = 1
    project_id: ArtifactId
    attempt_id: ArtifactId
    authority_sha256: Sha256
    stage: ProjectStage
    reservations: tuple[ProviderSpendEntry, ...]

    @field_validator("reservations")
    @classmethod
    def _require_source_separated_reservations(
        cls,
        value: tuple[ProviderSpendEntry, ...],
    ) -> tuple[ProviderSpendEntry, ...]:
        """Require canonical source-specific reservations with no artifact evidence."""
        operation_ids = tuple(item.operation_id for item in value)
        pairs = tuple((item.component, item.billing_source) for item in value)
        if not value:
            raise ValueError("hosted provider hazard requires at least one reservation")
        if operation_ids != tuple(sorted(operation_ids)) or len(set(operation_ids)) != len(value):
            raise ValueError("hosted provider reservations need sorted unique operation IDs")
        if len(set(pairs)) != len(pairs):
            raise ValueError("hosted provider reservations need unique component-source pairs")
        if any(
            item.status != ProviderSpendStatus.RESERVED or item.evidence is not None
            for item in value
        ):
            raise ValueError("hosted provider hazard accepts only evidence-free reservations")
        return value

    @property
    def reserved_usd(self) -> Decimal:
        """Return the exact total reserved across all billing sources."""
        return sum((item.amount_usd for item in self.reservations), start=USD_ZERO)


class HostedStageCommit(ContractModel):
    """External acknowledgment binding one durable stage to its exact verified bundle."""

    schema_version: Literal[1] = 1
    project_id: ArtifactId
    attempt_id: ArtifactId
    authority_sha256: Sha256
    stage: ProjectStage
    bundle_sha256: Sha256
    bundle_size_bytes: int = Field(gt=0)
    spend_ledger: ArtifactInput
    spend_total_usd: Decimal = Field(ge=0)

    @field_validator("spend_total_usd", mode="before")
    @classmethod
    def _require_exact_spend_total(cls, value: object) -> Decimal:
        """Return one exact nonnegative numeric(20,6) committed total."""
        return exact_usd(value, allow_zero=True)

    @model_validator(mode="after")
    def _require_hosted_stage(self) -> HostedStageCommit:
        """Reject a provider-free stage that is outside this attempt state machine."""
        if self.stage not in _HOSTED_STAGE_ORDER:
            raise ValueError("hosted attempt commits only provider-backed stages")
        return self


class HostedAttemptState(ContractModel):
    """Monotonic external state for one bound attempt and latest durable stage pointer."""

    schema_version: Literal[1] = 1
    binding: HostedAttemptBinding
    latest_commit: HostedStageCommit | None = None
    terminal: bool = False

    @model_validator(mode="after")
    def _require_coherent_state(self) -> HostedAttemptState:
        """Bind the latest commit and terminal marker to the immutable attempt binding."""
        commit = self.latest_commit
        if commit is not None and (
            commit.project_id != self.binding.project_id
            or commit.attempt_id != self.binding.attempt_id
            or commit.authority_sha256 != self.binding.authority_sha256
        ):
            raise ValueError("hosted stage commit differs from its attempt binding")
        expected_terminal = commit is not None and commit.stage == ProjectStage.COMPLETING_REPORT
        if self.terminal != expected_terminal:
            raise ValueError("hosted attempt terminal state differs from its latest stage")
        return self


class HostedAttemptAuthorityStore(Protocol):
    """Injected monotonic authority that survives worker and Project-bundle replacement."""

    def load(self, attempt_id: str) -> HostedAttemptAuthority:
        """Return the verified write-once authority for one exact attempt."""

    def bind(
        self,
        authority: HostedAttemptAuthority,
        *,
        project_id: str,
        ceiling_usd: Decimal,
    ) -> HostedAttemptState:
        """Write once or exactly replay the attempt's Project and ceiling binding."""

    def state(self, authority: HostedAttemptAuthority) -> HostedAttemptState:
        """Return the verified monotonic externally committed state for one attempt."""

    def begin(self, hazard: HostedProviderHazard) -> None:
        """Durably record a reservation before its provider dispatch boundary."""

    def mark_ambiguous(self, hazard: HostedProviderHazard) -> None:
        """Permanently close the attempt with write-once ambiguity evidence."""

    def unresolved(
        self,
        authority: HostedAttemptAuthority,
    ) -> HostedProviderHazard | None:
        """Return an active or permanently ambiguous paid-operation reservation."""

    def commit_stage(
        self,
        commit: HostedStageCommit,
        bundle: ExportedProjectBundle,
        ledger: ProviderSpendLedger,
    ) -> None:
        """Atomically commit bundle and spend evidence before clearing its hazard."""


class FileHostedAttemptAuthorityStore:
    """Filesystem-backed authority suitable for one durable shared local volume."""

    def __init__(self, directory: Path) -> None:
        """Bind one caller-owned directory without creating or reopening its authority."""
        self._directory = Path(directory)

    def create(self) -> HostedAttemptAuthority:
        """Create or reopen this store's random write-once authority."""
        return create_hosted_attempt_authority(self._directory)

    def load(self, attempt_id: str) -> HostedAttemptAuthority:
        """Load the exact authority derived for the supplied attempt ID."""
        return load_hosted_attempt_authority(self._directory, attempt_id=attempt_id)

    def bind(
        self,
        authority: HostedAttemptAuthority,
        *,
        project_id: str,
        ceiling_usd: Decimal,
    ) -> HostedAttemptState:
        """Bind this authority to one Project and accepted ceiling exactly once."""
        return bind_hosted_attempt(
            self._directory,
            authority,
            project_id=project_id,
            ceiling_usd=ceiling_usd,
        )

    def state(self, authority: HostedAttemptAuthority) -> HostedAttemptState:
        """Load this authority's monotonic external stage state."""
        return load_hosted_attempt_state(self._directory, authority)

    def begin(self, hazard: HostedProviderHazard) -> None:
        """Write one provider reservation before dispatch."""
        authority = self.load(hazard.attempt_id)
        begin_provider_hazard(self._directory, authority, hazard)

    def mark_ambiguous(self, hazard: HostedProviderHazard) -> None:
        """Write or replay one permanent ambiguity marker."""
        authority = self.load(hazard.attempt_id)
        mark_provider_ambiguity(self._directory, authority, hazard)

    def unresolved(
        self,
        authority: HostedAttemptAuthority,
    ) -> HostedProviderHazard | None:
        """Return this store's active or permanent paid-operation hazard."""
        return unresolved_provider_hazard(self._directory, authority)

    def commit_stage(
        self,
        commit: HostedStageCommit,
        bundle: ExportedProjectBundle,
        ledger: ProviderSpendLedger,
    ) -> None:
        """Persist one bundle-and-ledger acknowledgment before clearing its hazard."""
        authority = self.load(commit.attempt_id)
        commit_provider_stage(self._directory, authority, commit, bundle, ledger)


class _HostedAttemptAuthorityRecord(ContractModel):
    """Private authority record whose random nonce never enters Project state or bundles."""

    schema_version: Literal[1] = 1
    nonce: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    authority: HostedAttemptAuthority

    @model_validator(mode="after")
    def _require_content_identity(self) -> _HostedAttemptAuthorityRecord:
        """Bind public authority identity to the private random nonce."""
        digest = sha256_bytes(self.nonce.encode("ascii"))
        attempt_id = stable_id("hosted-attempt", {"authority_sha256": digest})
        if self.authority.authority_sha256 != digest:
            raise ValueError("hosted attempt authority digest differs from its nonce")
        if self.authority.attempt_id != attempt_id:
            raise ValueError("hosted attempt identity differs from its authority digest")
        return self


def create_hosted_attempt_authority(directory: Path) -> HostedAttemptAuthority:
    """Create or reopen one random write-once spend authority outside Project bundles.

    Args:
        directory: Durable caller-owned directory retained across Project bundle restores.

    Returns:
        Public attempt identity and authority digest safe to bind into spend ledgers.

    Raises:
        HostedAttemptAuthorityError: The directory or an existing authority record is unsafe.
    """
    root = _prepare_directory(directory)
    path = root / _AUTHORITY_FILE
    with file_write_lock(path, what="hosted attempt authority"):
        if path.exists():
            return _load_authority_record(path).authority
        nonce = secrets.token_hex(32)
        digest = sha256_bytes(nonce.encode("ascii"))
        authority = HostedAttemptAuthority(
            attempt_id=stable_id("hosted-attempt", {"authority_sha256": digest}),
            authority_sha256=digest,
        )
        record = _HostedAttemptAuthorityRecord(nonce=nonce, authority=authority)
        write_bytes_atomic(path, canonical_json_bytes(record))
        path.chmod(0o600)
        return authority


def load_hosted_attempt_authority(
    directory: Path,
    *,
    attempt_id: str,
) -> HostedAttemptAuthority:
    """Load and verify the required authority for one exact caller-supplied attempt ID.

    Args:
        directory: Durable authority directory created before workflow execution.
        attempt_id: Expected attempt identity derived from the private authority nonce.

    Returns:
        Verified public authority identity.

    Raises:
        HostedAttemptAuthorityError: The authority is missing, unsafe, corrupt, or mismatched.
    """
    root = _require_directory(directory)
    authority = _load_authority_record(root / _AUTHORITY_FILE).authority
    if authority.attempt_id != attempt_id:
        raise HostedAttemptAuthorityError(
            "hosted attempt ID differs from the durable external spend authority"
        )
    return authority


def bind_hosted_attempt(
    directory: Path,
    authority: HostedAttemptAuthority,
    *,
    project_id: str,
    ceiling_usd: Decimal,
) -> HostedAttemptState:
    """Bind one authority exactly once to its Project and accepted spend ceiling.

    Args:
        directory: Durable caller-owned authority directory.
        authority: Verified random write-once authority identity.
        project_id: Exact Project authorized to consume the ceiling.
        ceiling_usd: Finite total provider-spend authorization for the attempt.

    Returns:
        Newly created or exactly replayed monotonic attempt state.

    Raises:
        HostedAttemptAuthorityError: An existing binding or authority differs.
    """
    root = _verified_authority_directory(directory, authority)
    try:
        binding = HostedAttemptBinding(
            project_id=project_id,
            attempt_id=authority.attempt_id,
            authority_sha256=authority.authority_sha256,
            ceiling_usd=ceiling_usd,
        )
    except ValueError as exc:
        raise HostedAttemptAuthorityError("hosted attempt binding is invalid") from exc
    path = root / _STATE_FILE
    with file_write_lock(path, what="hosted attempt state"):
        if path.exists():
            state = _read_attempt_state(path)
            if state.binding != binding:
                raise HostedAttemptAuthorityError(
                    "hosted attempt already binds another Project or spend ceiling"
                )
            return state
        state = HostedAttemptState(binding=binding)
        write_bytes_atomic(path, canonical_json_bytes(state))
        return state


def load_hosted_attempt_state(
    directory: Path,
    authority: HostedAttemptAuthority,
) -> HostedAttemptState:
    """Load the monotonic Project, ceiling, and latest external bundle binding.

    Args:
        directory: Durable caller-owned authority directory.
        authority: Verified random write-once authority identity.

    Returns:
        Current exact external attempt state.

    Raises:
        HostedAttemptAuthorityError: State is absent, invalid, or names another authority.
    """
    root = _verified_authority_directory(directory, authority)
    state = _read_attempt_state(root / _STATE_FILE)
    _require_state_authority(state, authority)
    return state


def verify_hosted_attempt_resume(
    state: HostedAttemptState,
    *,
    project_id: str,
    selected_stage: ProjectStage | None,
    resume_bundle_sha256: str | None,
) -> None:
    """Require restored Project state to match the exact external stage pointer.

    Args:
        state: Monotonic state returned by the bound external authority store.
        project_id: Project opened for this workflow invocation.
        selected_stage: Latest provider-backed stage selected in that Project.
        resume_bundle_sha256: Exact external bundle restored by the caller, when any.

    Raises:
        HostedAttemptAuthorityError: Project stage or bundle identity differs from the authority.
    """
    if state.binding.project_id != project_id:
        raise HostedAttemptAuthorityError("hosted attempt state belongs to another Project")
    commit = state.latest_commit
    committed_stage = None if commit is None else commit.stage
    if committed_stage != selected_stage:
        raise HostedAttemptAuthorityError(
            "restored Project stage differs from the latest external attempt commit"
        )
    if commit is None:
        if resume_bundle_sha256 is not None:
            raise HostedAttemptAuthorityError(
                "uncommitted hosted attempt cannot resume from a provider-stage bundle"
            )
        return
    if resume_bundle_sha256 != commit.bundle_sha256:
        raise HostedAttemptAuthorityError(
            "restored Project bundle differs from the exact external stage commit"
        )


def begin_provider_hazard(
    directory: Path,
    authority: HostedAttemptAuthority,
    hazard: HostedProviderHazard,
) -> None:
    """Persist a paid-operation reservation before crossing its provider boundary.

    Args:
        directory: Durable authority directory.
        authority: Verified public authority identity.
        hazard: Exact pending paid-operation reservation.
    Raises:
        HostedAttemptAuthorityError: Authority, ambiguity, or active-hazard state conflicts.
    """
    root = _verified_authority_directory(directory, authority)
    _require_hazard_authority(hazard, authority)
    state_path = root / _STATE_FILE
    with file_write_lock(state_path, what="hosted attempt state"):
        state = _read_attempt_state(state_path)
        _require_state_for_hazard(state, hazard)
        if (root / _AMBIGUITY_FILE).exists():
            raise HostedAttemptAuthorityError("hosted attempt is permanently closed by ambiguity")
        path = root / _HAZARD_FILE
        if path.exists():
            raise HostedAttemptAuthorityError(
                "an unresolved hosted provider reservation blocks this attempt"
            )
        write_bytes_atomic(path, canonical_json_bytes(hazard))


def mark_provider_ambiguity(
    directory: Path,
    authority: HostedAttemptAuthority,
    hazard: HostedProviderHazard,
) -> None:
    """Write or exactly replay the permanent ambiguity marker for one closed attempt.

    Args:
        directory: Durable authority directory.
        authority: Verified public authority identity.
        hazard: Last paid-operation reservation whose outcome is unknown.

    Raises:
        HostedAttemptAuthorityError: Existing authority or ambiguity evidence conflicts.
    """
    root = _verified_authority_directory(directory, authority)
    _require_hazard_authority(hazard, authority)
    path = root / _AMBIGUITY_FILE
    state_path = root / _STATE_FILE
    with file_write_lock(state_path, what="hosted attempt state"):
        state = _read_attempt_state(state_path)
        _require_state_for_hazard(state, hazard)
        if path.exists():
            existing = _read_hazard(path, label="ambiguity")
            if existing != hazard:
                raise HostedAttemptAuthorityError(
                    "hosted attempt already records different ambiguity evidence"
                )
            return
        write_bytes_atomic(path, canonical_json_bytes(hazard))


def unresolved_provider_hazard(
    directory: Path,
    authority: HostedAttemptAuthority,
) -> HostedProviderHazard | None:
    """Return permanent ambiguity or an active crash-window reservation, if present.

    Args:
        directory: Durable authority directory.
        authority: Verified public authority identity.

    Returns:
        Ambiguous or active reservation, with permanent ambiguity taking precedence.
    """
    root = _verified_authority_directory(directory, authority)
    state = _read_attempt_state(root / _STATE_FILE)
    _require_state_authority(state, authority)
    ambiguity = root / _AMBIGUITY_FILE
    hazard = root / _HAZARD_FILE
    selected = ambiguity if ambiguity.exists() else hazard
    if not selected.exists():
        return None
    value = _read_hazard(selected, label="provider reservation")
    _require_hazard_authority(value, authority)
    if selected == hazard:
        commit = state.latest_commit
        if commit is not None and commit.stage == value.stage:
            if commit.project_id != value.project_id:
                raise HostedAttemptAuthorityError(
                    "hosted stage commit differs from the active provider reservation"
                )
            hazard.unlink(missing_ok=True)
            return None
    _require_state_for_hazard(state, value)
    return value


def commit_provider_stage(
    directory: Path,
    authority: HostedAttemptAuthority,
    commit: HostedStageCommit,
    bundle: ExportedProjectBundle,
    ledger: ProviderSpendLedger,
) -> None:
    """Acknowledge an externally durable stage bundle before clearing its active hazard.

    Args:
        directory: Durable authority directory.
        authority: Verified public authority identity.
        commit: Exact external stage-pointer acknowledgment to persist write once.
        bundle: Locally verified bundle supplied to the external commit implementation.
        ledger: Exact component ledger committed atomically with the bundle pointer.

    Raises:
        HostedAttemptAuthorityError: Authority, bundle, hazard, or prior commit conflicts.
    """
    root = _verified_authority_directory(directory, authority)
    _require_stage_commit_authority(commit, authority)
    if (
        commit.bundle_sha256 != bundle.sha256
        or commit.bundle_size_bytes != bundle.size_bytes
        or commit.project_id != bundle.manifest.project_id
        or commit.stage != bundle.manifest.completed_stages[-1]
        or commit.spend_ledger not in bundle.manifest.selected_artifacts
        or ledger.ledger_id != commit.spend_ledger.artifact_id
        or ledger.total_usd != commit.spend_total_usd
        or ledger.project_id != commit.project_id
        or ledger.attempt_id != commit.attempt_id
        or ledger.attempt_authority_sha256 != commit.authority_sha256
        or ledger.stage != commit.stage
        or ledger.outcome != "completed"
        or any(item not in bundle.manifest.selected_artifacts for item in ledger.stage_outputs)
    ):
        raise HostedAttemptAuthorityError(
            "hosted stage acknowledgment differs from its verified Project bundle"
        )
    state_path = root / _STATE_FILE
    with file_write_lock(state_path, what="hosted attempt state"):
        state = _read_attempt_state(state_path)
        _require_state_authority(state, authority)
        if ledger.ceiling_usd != state.binding.ceiling_usd:
            raise HostedAttemptAuthorityError(
                "hosted stage ledger differs from its accepted attempt ceiling"
            )
        if (root / _AMBIGUITY_FILE).exists():
            raise HostedAttemptAuthorityError("ambiguous hosted attempt cannot be completed")
        hazard_path = root / _HAZARD_FILE
        hazard = _read_hazard(hazard_path, label="provider reservation")
        _require_hazard_authority(hazard, authority)
        _require_state_for_hazard(state, hazard)
        if hazard.project_id != commit.project_id or hazard.stage != commit.stage:
            raise HostedAttemptAuthorityError(
                "hosted stage acknowledgment differs from the active provider reservation"
            )
        updated = HostedAttemptState(
            binding=state.binding,
            latest_commit=commit,
            terminal=commit.stage == ProjectStage.COMPLETING_REPORT,
        )
        write_bytes_atomic(state_path, canonical_json_bytes(updated))
        hazard_path.unlink(missing_ok=True)


def _prepare_directory(directory: Path) -> Path:
    """Create one caller-owned authority directory without accepting a symlink."""
    root = Path(directory)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise HostedAttemptAuthorityError("hosted attempt authority path must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    return _require_directory(root)


def _require_directory(directory: Path) -> Path:
    """Require an existing regular directory rather than recreating missing authority state."""
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise HostedAttemptAuthorityError(
            "durable hosted attempt authority directory is missing or unsafe"
        )
    return root


def _verified_authority_directory(
    directory: Path,
    authority: HostedAttemptAuthority,
) -> Path:
    """Reopen the private authority and require its exact public identity."""
    root = _require_directory(directory)
    current = _load_authority_record(root / _AUTHORITY_FILE).authority
    if current != authority:
        raise HostedAttemptAuthorityError("hosted attempt authority changed during execution")
    return root


def _load_authority_record(path: Path) -> _HostedAttemptAuthorityRecord:
    """Read one regular private authority file and verify its content identity."""
    if not path.is_file() or path.is_symlink():
        raise HostedAttemptAuthorityError("hosted attempt authority record is missing or unsafe")
    try:
        return _HostedAttemptAuthorityRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise HostedAttemptAuthorityError("hosted attempt authority record is invalid") from exc


def _read_hazard(path: Path, *, label: str) -> HostedProviderHazard:
    """Read one regular hazard or ambiguity record without exposing its local path."""
    if not path.is_file() or path.is_symlink():
        raise HostedAttemptAuthorityError(f"hosted {label} record is missing or unsafe")
    try:
        return HostedProviderHazard.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise HostedAttemptAuthorityError(f"hosted {label} record is invalid") from exc


def _read_attempt_state(path: Path) -> HostedAttemptState:
    """Read one regular monotonic attempt state file."""
    if not path.is_file() or path.is_symlink():
        raise HostedAttemptAuthorityError("hosted attempt state is missing or unsafe")
    try:
        return HostedAttemptState.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise HostedAttemptAuthorityError("hosted attempt state is invalid") from exc


def _require_hazard_authority(
    hazard: HostedProviderHazard,
    authority: HostedAttemptAuthority,
) -> None:
    """Require one hazard to name the exact externally verified attempt authority."""
    if (
        hazard.attempt_id != authority.attempt_id
        or hazard.authority_sha256 != authority.authority_sha256
    ):
        raise HostedAttemptAuthorityError("provider hazard differs from its attempt authority")


def _require_stage_commit_authority(
    commit: HostedStageCommit,
    authority: HostedAttemptAuthority,
) -> None:
    """Require a stage acknowledgment to name the exact verified attempt authority."""
    if (
        commit.attempt_id != authority.attempt_id
        or commit.authority_sha256 != authority.authority_sha256
    ):
        raise HostedAttemptAuthorityError("stage commit differs from its attempt authority")


def _require_state_authority(
    state: HostedAttemptState,
    authority: HostedAttemptAuthority,
) -> None:
    """Require one external state to retain the exact random authority identity."""
    binding = state.binding
    if (
        binding.attempt_id != authority.attempt_id
        or binding.authority_sha256 != authority.authority_sha256
    ):
        raise HostedAttemptAuthorityError("hosted attempt state differs from its authority")


def _require_state_for_hazard(
    state: HostedAttemptState,
    hazard: HostedProviderHazard,
) -> None:
    """Require a provider reservation to be the next stage under the bound ceiling."""
    binding = state.binding
    if (
        hazard.project_id != binding.project_id
        or hazard.attempt_id != binding.attempt_id
        or hazard.authority_sha256 != binding.authority_sha256
    ):
        raise HostedAttemptAuthorityError(
            "provider reservation differs from its Project attempt binding"
        )
    committed_usd = USD_ZERO if state.latest_commit is None else state.latest_commit.spend_total_usd
    if committed_usd + hazard.reserved_usd > binding.ceiling_usd:
        raise HostedAttemptAuthorityError("provider reservation exceeds its attempt ceiling")
    expected_index = (
        0
        if state.latest_commit is None
        else (_HOSTED_STAGE_ORDER.index(state.latest_commit.stage) + 1)
    )
    if state.terminal or expected_index >= len(_HOSTED_STAGE_ORDER):
        raise HostedAttemptAuthorityError("terminal hosted attempt cannot dispatch provider work")
    if hazard.stage != _HOSTED_STAGE_ORDER[expected_index]:
        raise HostedAttemptAuthorityError(
            "provider reservation is not the next externally committed hosted stage"
        )
