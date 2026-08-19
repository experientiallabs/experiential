"""Compatibility bootstrap for one project-backed gateway alias."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from wmo.cli.gateway.key_output import (
    deliver_key_output,
    recover_key_output,
    settle_key_output,
)
from wmo.optimize.router.activation import verify_automatic_router_policy
from wmo.runtime.gateway.management import GatewayManagement
from wmo.runtime.gateway.project_activation import LocalArtifactProjectActivationRepository
from wmo.runtime.gateway.project_alias import prepare_project_gateway_alias


@dataclass(frozen=True)
class ProjectGatewayCompatibility:
    """Persistent resources used by the project-form gateway launch."""

    alias: str
    alias_revision_id: str
    identity_id: str
    key_file: Path
    policy_id: str
    changed: bool


def prepare_project_gateway(
    project: str,
    root: Path,
    *,
    policy_id: str | None,
) -> ProjectGatewayCompatibility:
    """Materialize one legacy project as ordinary SQLite gateway authority.

    Args:
        project: Existing immutable project name and public compatibility alias.
        root: WMO artifact and gateway root.
        policy_id: Optional exact frozen router policy.

    Returns:
        Exact alias, identity, key-file, and policy resources for normal gateway launch.
    """
    alias = prepare_project_gateway_alias(
        project,
        root,
        policy_id=policy_id,
        project_repository=LocalArtifactProjectActivationRepository(
            root,
            verifier=verify_automatic_router_policy,
        ),
    )
    manager = GatewayManagement(root)
    key_file = manager.state_dir / "compatibility-keys" / f"{alias.identity_id}.txt"
    key_changed = _ensure_compatibility_key(
        manager,
        identity_id=alias.identity_id,
        key_file=key_file,
    )
    return ProjectGatewayCompatibility(
        alias=alias.alias,
        alias_revision_id=alias.alias_revision_id,
        identity_id=alias.identity_id,
        key_file=key_file,
        policy_id=alias.policy_id,
        changed=alias.changed or key_changed,
    )


def _ensure_compatibility_key(
    manager: GatewayManagement,
    *,
    identity_id: str,
    key_file: Path,
) -> bool:
    """Create or recover one private compatibility key file."""
    key_id = f"{identity_id}-key"
    operation_id = f"{identity_id}-key-issue"
    store = manager.require_initialized()
    recovered = recover_key_output(
        key_file,
        store=store,
        organization_id=manager.organization_id,
        identity_id=identity_id,
        key_id=key_id,
        operation_id=operation_id,
        expires_at=None,
    )
    if recovered is not None:
        settle_key_output(key_file)
        return False
    if key_file.exists():
        raw_key = key_file.read_text(encoding="utf-8").strip()
        store.authenticate_key(raw_key=raw_key)
        return False
    manager.issue_key(
        identity_id=identity_id,
        key_id=key_id,
        operation_id=operation_id,
        secret_delivery=partial(deliver_key_output, key_file),
    )
    settle_key_output(key_file)
    return True
