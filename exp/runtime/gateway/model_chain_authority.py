"""Bound host authority for serving model chains, separate from draft graph data.

A receipt is the result of an enforcing backend transaction, not permission a
caller can grant itself. Hosts revalidate its durable issuance and stable-alias
floor at acceptance and each physical reservation. Local SQLite has no such
fleet authority and cannot publish or activate populated chains.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator

from exp.common.config.settings import GatewayResourceSettings
from exp.common.core.artifacts import ContractModel, Sha256
from exp.runtime.gateway.ledger_errors import AttemptRejectedError
from exp.runtime.gateway.snapshot_file import (
    PreparedSnapshotFile,
    SnapshotSizeError,
    prepare_snapshot_file,
    read_snapshot_bytes,
)
from exp.runtime.gateway.stream_contracts import GatewayFailure, GatewayFailureClass

if TYPE_CHECKING:
    from exp.common.models import ModelCatalog, NormalizedGatewayCatalog
    from exp.runtime.gateway.contracts import AuthorizationSnapshot
    from exp.runtime.gateway.native_components import NativeGatewayComponents

ModelChainAuthorityMode = Literal["dispatch", "completed_replay"]


class ModelChainAuthority(ContractModel):
    """Immutable binding returned by the host's protected authority transaction.

    Attributes:
        contract_version: Exact integer contract marker, currently 1.
        feature_epoch: Positive durable stable-alias policy floor.
        stable_alias_id: Host-owned alias identity retained across revisions.
        alias_revision_id: Selected immutable alias revision.
        catalog_sha256: Selected normalized catalog digest.
        organization_id: Tenant owning the request and alias.
        request_id: Exact logical request authorized by this receipt.
        identity_id: Authenticated caller identity.
        virtual_key_id: Authenticated non-secret key identifier.
        worker_id: Eligible serving worker identity.
        process_generation: Registered worker process incarnation.
        receipt_id: Durable host issuance identifier, not caller-granted permission.
        expires_at: Aware expiry, checked before dispatch.
        mode: Dispatch authority or exact completed-operation replay only.
    """

    contract_version: Literal[1] = 1
    feature_epoch: int = Field(gt=0, strict=True)
    stable_alias_id: str = Field(min_length=1, max_length=256)
    alias_revision_id: str = Field(min_length=1, max_length=256)
    catalog_sha256: Sha256
    organization_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=256)
    identity_id: str = Field(min_length=1, max_length=256)
    virtual_key_id: str = Field(min_length=1, max_length=256)
    worker_id: str = Field(min_length=1, max_length=256)
    process_generation: UUID
    receipt_id: UUID
    expires_at: AwareDatetime
    mode: ModelChainAuthorityMode

    @field_validator("contract_version", mode="before")
    @classmethod
    def _strict_contract_version(cls, value: object) -> int:
        """A boolean or coerced value cannot claim the native/Python chain contract."""
        if type(value) is not int or value != 1:
            raise ValueError("model-chain contract must be integer 1")
        return value


class ModelChainAuthorityError(ValueError):
    """Serving would depend on model-chain authority this backend cannot prove."""


@runtime_checkable
class ModelChainAuthorityStore(Protocol):
    """An enforcing host operation, never a capability boolean or receipt factory.

    The implementation must validate current key/grant authority, the final
    revision/digest and the registered process against a durable stable-alias
    epoch under the host publication barrier. Its ledger must validate the same
    binding atomically at acceptance and every reservation, including retries.
    """

    def authorize_model_chain(
        self,
        *,
        authorization: AuthorizationSnapshot,
        mode: ModelChainAuthorityMode = "dispatch",
    ) -> AuthorizationSnapshot:
        """Obtain current backend authority without changing the selected request identity."""
        ...


def require_bound_model_chain_authority(
    authorization: AuthorizationSnapshot,
    *,
    mode: ModelChainAuthorityMode = "dispatch",
    require_current: bool = True,
) -> ModelChainAuthority:
    """Validate the exact returned binding; durable enforcement still belongs to the host."""
    receipt = authorization.model_chain_authority
    if (
        receipt is None
        or receipt.mode != mode
        or (require_current and receipt.expires_at <= datetime.now(UTC))
    ):
        raise ModelChainAuthorityError("model chains require current enforcing host authority")
    for field in (
        "organization_id",
        "request_id",
        "identity_id",
        "virtual_key_id",
        "alias_revision_id",
        "catalog_sha256",
    ):
        if getattr(receipt, field) != getattr(authorization, field):
            raise ModelChainAuthorityError(
                "model-chain authority differs from the selected request"
            )
    return receipt


def authorize_model_chain(
    store: object,
    authorization: AuthorizationSnapshot,
    *,
    required: bool,
    mode: ModelChainAuthorityMode = "dispatch",
) -> AuthorizationSnapshot:
    """Consult the configured host even when cached plain content carries no receipt."""
    if not isinstance(store, ModelChainAuthorityStore):
        if not required and authorization.model_chain_authority is None:
            return authorization
        raise ModelChainAuthorityError(
            "model chains require an enforcing host; local SQLite chain serving is unsupported"
        )
    validated = store.authorize_model_chain(authorization=authorization, mode=mode)
    # A backend may issue a new receipt, never silently retarget this request.
    if validated.model_copy(update={"model_chain_authority": None}) != authorization.model_copy(
        update={"model_chain_authority": None}
    ):
        raise ModelChainAuthorityError(
            "model-chain authority changed the selected request identity"
        )
    if (
        required
        or authorization.model_chain_authority is not None
        or validated.model_chain_authority is not None
    ):
        require_bound_model_chain_authority(validated, mode=mode)
    return validated


def authorize_serving_model_chains(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    *,
    mode: ModelChainAuthorityMode = "dispatch",
) -> AuthorizationSnapshot:
    """Check exact normalized and authored views before admission or replay can use them."""
    runtime = components.runtime_catalogs.get(
        (authorization.alias_revision_id, authorization.catalog_sha256)
    )
    required = False
    inspected = False
    if authorization.target.kind == "direct" and runtime is not None:
        try:
            required = runtime.requires_model_chain_authority(pool_id=authorization.target.pool_id)
        except ValueError as exc:
            raise ModelChainAuthorityError("selected authored root cannot be classified") from exc
        inspected = True
    inspect = getattr(components.routes, "requires_model_chain_authority", None)
    if callable(inspect):
        required = bool(inspect(authorization)) or required
        inspected = True
    if not inspected:
        raise ModelChainAuthorityError("selected serving graph cannot be classified")
    if required and authorization.surface not in ("chat_completions", "responses", "messages"):
        raise ModelChainAuthorityError(
            "selected model-chain policy is unsupported on this non-conversational surface"
        )
    return authorize_model_chain(components.store, authorization, required=required, mode=mode)


def refuse_sqlite_chain_authorization(
    connection: sqlite3.Connection,
    organization_id: str,
    alias_revision_id: str,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> None:
    """Check requested and currently active policy before local acceptance or replay."""
    rows = connection.execute(
        """SELECT r.snapshot_ref, active.snapshot_ref FROM alias_revisions r
        JOIN gateway_aliases a ON a.organization_id=r.organization_id AND a.alias_id=r.alias_id
        LEFT JOIN alias_revisions active ON active.organization_id=a.organization_id
          AND active.revision_id=a.active_revision_id
        WHERE r.organization_id=? AND r.revision_id=?""",
        (organization_id, alias_revision_id),
    ).fetchall()
    references = dict.fromkeys(
        str(reference) for row in rows for reference in row if reference is not None
    )
    for reference in references:
        refuse_sqlite_chain_snapshot(connection, reference, maximum_bytes=maximum_bytes)


def refuse_sqlite_chain_snapshot(
    connection: sqlite3.Connection,
    snapshot_ref: str,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> None:
    """Apply the same local refusal inside every low-level registration/activation transaction."""
    database = next(
        (str(row[2]) for row in connection.execute("PRAGMA database_list") if row[1] == "main"), ""
    )
    if database:
        refuse_local_chain_snapshot(
            Path(database).parent, snapshot_ref, maximum_bytes=maximum_bytes
        )


def refuse_local_chain_snapshot(
    state_dir: Path,
    snapshot_ref: str,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> None:
    """Inspect present local serving content without rejecting ordinary remote references.

    Hosted publication classifies remote bytes in its enforcing transaction.
    Local metadata-only remote references remain non-serving until hydration;
    no local absence is treated as permission to execute a populated graph.
    """
    root = state_dir.resolve()
    path = Path(snapshot_ref)
    # Preserve every original component for the no-follow reader. Resolving the
    # child first would hide a symlink from its directory-relative checks.
    try:
        sources = (
            (path, "model_chains"),
            (path.with_suffix(".models.json"), "gateway_model_chains"),
        )
    except ValueError as exc:
        raise ModelChainAuthorityError("catalog snapshot reference is invalid") from exc
    for source, field in sources:
        try:
            raw = json.loads(read_snapshot_bytes(root, str(source), maximum_bytes))
        except FileNotFoundError:
            continue
        except SnapshotSizeError as exc:
            raise _serving_size_error(exc) from exc
        except (OSError, ValueError) as exc:
            raise ModelChainAuthorityError("serving snapshot cannot be classified safely") from exc
        if not isinstance(raw, dict):
            raise ModelChainAuthorityError("serving snapshot must be a catalog object")
        if raw.get(field):
            raise ModelChainAuthorityError("local SQLite cannot activate populated model chains")


def serving_snapshot_limit(explicit: int | None) -> int:
    """Validate an explicit serving limit or use the default without filesystem reads."""
    return (
        GatewayResourceSettings()
        if explicit is None
        else GatewayResourceSettings(serving_snapshot_max_bytes=explicit)
    ).serving_snapshot_max_bytes


def _serving_size_error(error: SnapshotSizeError) -> ModelChainAuthorityError:
    """Name the serving resource policy, never the independent authoring setting."""
    return ModelChainAuthorityError(
        f"serving snapshot needs {error.size} bytes, above the configured "
        f"{error.maximum}-byte resource budget; raise [gateway].serving_snapshot_max_bytes "
        "in <root>/settings.toml and restart the gateway, or pass serving_snapshot_max_bytes "
        "to the SQLite control and ledger constructors"
    )


ChainOperation = Literal["authorize", "accept", "reserve"]
# JSON classification is CPU/GIL-bound across databases in this process. A permit
# bounds simultaneous allocations, never caches policy or spans a write transaction.
_PREFLIGHT_PERMIT = threading.BoundedSemaphore(1)


@contextmanager
def _preflight_budget(remaining_seconds: float) -> Iterator[float]:
    """Wait only within the caller's budget; synchronous parsing is checked, not interrupted."""
    if not math.isfinite(remaining_seconds) or not 0 < remaining_seconds <= threading.TIMEOUT_MAX:
        raise _preflight_timeout()
    deadline = time.monotonic() + remaining_seconds
    if not _PREFLIGHT_PERMIT.acquire(timeout=remaining_seconds):
        raise _preflight_timeout()
    try:
        if time.monotonic() >= deadline:
            raise _preflight_timeout()
        yield deadline
        if time.monotonic() >= deadline:
            raise _preflight_timeout()
    finally:
        _PREFLIGHT_PERMIT.release()


def _preflight_timeout() -> AttemptRejectedError:
    """Reuse the ordinary pre-dispatch timeout shape without permitting fallback."""
    return AttemptRejectedError(
        "request deadline expired during local snapshot preparation",
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TIMEOUT,
            safe_message="request deadline expired before local snapshot preparation completed",
        ),
    )


_AuthorityRows = tuple[tuple[str | None, ...], ...]


class _PlainSnapshotPair:
    """Two classified file generations with at most two retained anti-reuse leaf descriptors."""

    def __init__(self, files: tuple[PreparedSnapshotFile, ...]) -> None:
        """Anchor the exact classified leaves without retaining JSON or ancestor handles."""
        self.generations = tuple(file.generation for file in files)
        self.anchors: list[int | None] = []
        try:
            for file in files:
                self.anchors.append(file.retain_leaf())
                if not file.anchor_matches(self.anchors[-1]):
                    raise ValueError("snapshot anchor cannot preserve classified identity")
        except BaseException:
            self.close()
            raise

    def matches(self, files: tuple[PreparedSnapshotFile, ...]) -> bool:
        """Require the independently opened current pair and still-live anchors to agree."""
        return self.generations == tuple(file.generation for file in files) and all(
            file.anchor_matches(anchor) for file, anchor in zip(files, self.anchors, strict=True)
        )

    def close(self) -> None:
        """Release cache-only anchors, never the live operation's independent handles."""
        for descriptor in self.anchors:
            if descriptor is not None:
                os.close(descriptor)
        self.anchors.clear()


_MemoKey = tuple[str, str, int]


def _close_memo_entries(entries: OrderedDict[_MemoKey, _PlainSnapshotPair]) -> None:
    """Backstop discarded owners without capturing the memo in its own finalizer."""
    while entries:
        entries.popitem()[1].close()


class SnapshotClassificationMemo:
    """Bounded local parsing reuse, never a request, key, alias, or host permission cache.

    Eight snapshot pairs retain at most sixteen leaf descriptors. Every operation
    independently opens and checks the full paths, size policy and exact authority.
    Explicit close owns normal shutdown; the finalizer only backs up discarded owners.
    """

    def __init__(self) -> None:
        """Start an empty eight-pair owner with no descriptors allocated."""
        self._entries: OrderedDict[_MemoKey, _PlainSnapshotPair] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(self, _close_memo_entries, self._entries)

    def matches(self, key: _MemoKey, files: tuple[PreparedSnapshotFile, ...]) -> bool:
        """Return only whether the already-open pair has a reusable plain classification."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or self._closed:
                return False
            if entry.matches(files):
                self._entries.move_to_end(key)
                return True
            self._entries.pop(key).close()
            return False

    def invalidate(self, key: _MemoKey) -> None:
        """Release obsolete anchors after a secure path cannot be opened or classified."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                entry.close()

    def remember(self, key: _MemoKey, files: tuple[PreparedSnapshotFile, ...]) -> None:
        """Keep a proven plain pair; optional anchor retention failure only disables reuse."""
        with self._lock:
            if self._closed:
                return
            previous = self._entries.pop(key, None)
            if previous is not None:
                previous.close()
            if len(self._entries) == 8:
                self._entries.popitem(last=False)[1].close()
            try:
                self._entries[key] = _PlainSnapshotPair(files)
            except (OSError, ValueError):
                # The caller already classified secure operation bytes. Failure
                # to retain optional identity anchors never changes that result.
                return

    def clear(self) -> None:
        """Evict all parse facts and anchors while allowing future operations to repopulate."""
        with self._lock:
            _close_memo_entries(self._entries)

    def close(self) -> None:
        """Idempotently stop retaining entries; later operations must classify fresh bytes."""
        with self._lock:
            self._closed = True
            _close_memo_entries(self._entries)


class LocalSnapshotMemoOwner:
    """Explicit owned-or-borrowed parsing-resource lifetime for SQLite components."""

    def _bind_classification_memo(self, memo: SnapshotClassificationMemo | None) -> None:
        """Borrow the composition owner or allocate a private bounded owner."""
        self.classification_memo = SnapshotClassificationMemo() if memo is None else memo
        self._owns_classification_memo = memo is None

    def close(self) -> None:
        """Close only private resources; borrowed composition resources outlive this component."""
        if self._owns_classification_memo:
            self.classification_memo.close()


def _chain_authority_rows(
    connection: sqlite3.Connection,
    organization_id: str,
    alias_revision_id: str,
) -> _AuthorityRows:
    """Read exact requested and active revision identities, not just their file references."""
    rows = connection.execute(
        """SELECT r.organization_id, r.alias_id, r.revision_id, r.catalog_sha256,
        r.snapshot_ref, a.active_revision_id, active.catalog_sha256, active.snapshot_ref
        FROM alias_revisions r JOIN gateway_aliases a
          ON a.organization_id=r.organization_id AND a.alias_id=r.alias_id
        LEFT JOIN alias_revisions active ON active.organization_id=a.organization_id
          AND active.revision_id=a.active_revision_id
        WHERE r.organization_id=? AND r.revision_id=?""",
        (organization_id, alias_revision_id),
    ).fetchall()
    if not rows:
        raise ModelChainAuthorityError("requested local alias authority is unavailable")
    return tuple(tuple(None if value is None else str(value) for value in row) for row in rows)


def _database_path(connection: sqlite3.Connection) -> str:
    """Bind a proof to the concrete main database, refusing non-file local authority."""
    database = next(
        (str(row[2]) for row in connection.execute("PRAGMA database_list") if row[1] == "main"), ""
    )
    if not database:
        raise ModelChainAuthorityError("local chain preflight requires a file-backed database")
    return database


class SQLiteChainPreflight:
    """Operation-scoped classification with live secure handles, never durable permission.

    Attributes:
        files: At most four distinct requested/active normalized and authored file observations.
    """

    def __init__(
        self,
        database: str,
        rows: _AuthorityRows,
        request_id: str,
        organization_id: str,
        alias_revision_id: str,
        operation: ChainOperation,
        files: tuple[PreparedSnapshotFile, ...],
        deadline: float,
    ) -> None:
        """Bind exact operation identity and already classified, still-open path observations."""
        self._database = database
        self._rows = rows
        self._binding = (request_id, organization_id, alias_revision_id, operation)
        self.files = files
        self._deadline = deadline
        self._closed = False

    @staticmethod
    def require_ledger(
        connection: sqlite3.Connection,
        authorization: AuthorizationSnapshot,
        proof: SQLiteChainPreflight | None,
        operation: ChainOperation,
        *,
        staged: bool,
    ) -> None:
        """Refuse missing local proof or chain policy before the ledger's atomic mutation."""
        if staged or authorization.model_chain_authority is not None or proof is None:
            raise ModelChainAuthorityError(
                "local SQLite requires plain policy and a live chain preflight"
            )
        proof.validate(
            connection,
            request_id=authorization.request_id,
            organization_id=authorization.organization_id,
            alias_revision_id=authorization.alias_revision_id,
            operation=operation,
        )

    def validate(
        self,
        connection: sqlite3.Connection,
        *,
        request_id: str,
        organization_id: str,
        alias_revision_id: str,
        operation: ChainOperation,
    ) -> None:
        """Fence exact DB identities and paths without reading bytes or parsing under the lock."""
        if self._closed or self._binding != (
            request_id,
            organization_id,
            alias_revision_id,
            operation,
        ):
            raise ModelChainAuthorityError(
                "local chain preflight is closed or bound to another operation"
            )
        if time.monotonic() >= self._deadline:
            raise _preflight_timeout()
        if not connection.in_transaction:
            raise ModelChainAuthorityError("local chain proof requires an authority transaction")
        if (
            _database_path(connection) != self._database
            or _chain_authority_rows(connection, organization_id, alias_revision_id) != self._rows
        ):
            raise ModelChainAuthorityError(
                "local alias changed after preflight; retry the operation"
            )
        try:
            for prepared in self.files:
                prepared.validate_current()
        except (OSError, ValueError) as exc:
            raise ModelChainAuthorityError(
                "serving snapshot changed after preflight; retry the operation"
            ) from exc


@contextmanager
def prepare_sqlite_chain_authority(
    connection: sqlite3.Connection,
    organization_id: str,
    alias_revision_id: str,
    *,
    request_id: str,
    operation: ChainOperation,
    maximum_bytes: int,
    remaining_seconds: float,
    classification_memo: SnapshotClassificationMemo | None = None,
) -> Iterator[SQLiteChainPreflight]:
    """Classify both catalog views before BEGIN and retain handles through commit or rollback."""
    if connection.in_transaction:
        raise ModelChainAuthorityError(
            "prepare local chain authority before beginning a transaction"
        )
    database = _database_path(connection)
    rows = _chain_authority_rows(connection, organization_id, alias_revision_id)
    references = dict.fromkeys(
        row[index] for row in rows for index in (4, 7) if row[index] is not None
    )
    if not math.isfinite(remaining_seconds) or not 0 < remaining_seconds <= threading.TIMEOUT_MAX:
        raise _preflight_timeout()
    deadline = time.monotonic() + remaining_seconds
    files: list[PreparedSnapshotFile] = []
    with ExitStack() as stack:
        try:
            for reference in references:
                assert reference is not None
                path = Path(reference)
                pair = tuple(
                    stack.enter_context(
                        prepare_snapshot_file(
                            Path(database).parent,
                            str(source),
                            maximum_bytes,
                            read_content=False,
                        )
                    )
                    for source in (path, path.with_suffix(".models.json"))
                )
                key = (database, reference, maximum_bytes)
                if classification_memo is None or not classification_memo.matches(key, pair):
                    with _preflight_budget(deadline - time.monotonic()):
                        if classification_memo is None or not classification_memo.matches(
                            key, pair
                        ):
                            for prepared, field in zip(
                                pair, ("model_chains", "gateway_model_chains"), strict=True
                            ):
                                content = prepared.read_bytes(maximum_bytes)
                                if content is not None:
                                    raw = json.loads(content)
                                    del content
                                    if not isinstance(raw, dict):
                                        raise ModelChainAuthorityError(
                                            "serving snapshot must be a catalog object"
                                        )
                                    if raw.get(field):
                                        raise ModelChainAuthorityError(
                                            "local SQLite cannot activate populated model chains"
                                        )
                                    del raw
                            if classification_memo is not None:
                                classification_memo.remember(key, pair)
                files.extend(pair)
            if time.monotonic() >= deadline:
                raise _preflight_timeout()
        except SnapshotSizeError as exc:
            if classification_memo is not None:
                for reference in references:
                    assert reference is not None
                    classification_memo.invalidate((database, reference, maximum_bytes))
            raise _serving_size_error(exc) from exc
        except (OSError, ValueError) as exc:
            if classification_memo is not None:
                for reference in references:
                    assert reference is not None
                    classification_memo.invalidate((database, reference, maximum_bytes))
            if isinstance(exc, (ModelChainAuthorityError, AttemptRejectedError)):
                raise
            raise ModelChainAuthorityError("serving snapshot cannot be classified safely") from exc
        proof = SQLiteChainPreflight(
            database,
            rows,
            request_id,
            organization_id,
            alias_revision_id,
            operation,
            tuple(files),
            deadline,
        )
        try:
            yield proof
        finally:
            proof._closed = True


def refuse_unenforced_model_chains(
    catalog: ModelCatalog | NormalizedGatewayCatalog,
) -> None:
    """Refuse local serving publication without rejecting pure draft graph authoring."""
    chains = getattr(catalog, "model_chains", None) or getattr(
        catalog, "gateway_model_chains", None
    )
    if chains:
        raise ModelChainAuthorityError(
            "populated model chains require an enforcing host publication and rollback floor; "
            "local SQLite publication and activation are unsupported"
        )
