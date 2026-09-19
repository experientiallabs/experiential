"""Bound host authority for serving model chains, separate from draft graph data.

A receipt is the result of an enforcing backend transaction, not permission a
caller can grant itself. Hosts revalidate its durable issuance and stable-alias
floor at acceptance and each physical reservation. Local SQLite has no such
fleet authority and cannot publish or activate populated chains.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator

from exp.common.config.settings import GatewayResourceSettings
from exp.common.core.artifacts import ContractModel, Sha256
from exp.runtime.gateway.snapshot_file import read_snapshot_bytes

if TYPE_CHECKING:
    from exp.common.models import ModelCatalog, NormalizedGatewayCatalog
    from exp.runtime.gateway.contracts import AuthorizationSnapshot
    from exp.runtime.gateway.native_components import NativeGatewayComponents

ModelChainAuthorityMode = Literal["dispatch", "completed_replay"]


class ModelChainAuthority(ContractModel):
    """Immutable binding returned by the host's protected authority transaction."""

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
    return authorize_model_chain(components.store, authorization, required=required, mode=mode)


def refuse_sqlite_chain_authorization(
    connection: sqlite3.Connection, organization_id: str, alias_revision_id: str
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
    for row in rows:
        for reference in row:
            if reference is not None:
                refuse_sqlite_chain_snapshot(connection, str(reference))


def refuse_sqlite_chain_snapshot(connection: sqlite3.Connection, snapshot_ref: str) -> None:
    """Apply the same local refusal inside every low-level registration/activation transaction."""
    database = next(
        (str(row[2]) for row in connection.execute("PRAGMA database_list") if row[1] == "main"), ""
    )
    if database:
        refuse_local_chain_snapshot(Path(database).parent, snapshot_ref)


def refuse_local_chain_snapshot(state_dir: Path, snapshot_ref: str) -> None:
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
            raw = json.loads(
                read_snapshot_bytes(
                    root, str(source), GatewayResourceSettings().budget_snapshot_max_bytes
                )
            )
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise ModelChainAuthorityError("serving snapshot cannot be classified safely") from exc
        if not isinstance(raw, dict):
            raise ModelChainAuthorityError("serving snapshot must be a catalog object")
        if raw.get(field):
            raise ModelChainAuthorityError("local SQLite cannot activate populated model chains")


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
