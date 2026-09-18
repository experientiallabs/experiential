"""Local identity-scoped gateway capture composition over the native collector."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from exp.common.claas import CapturePolicy, ClaasScope
from exp.runtime.claas.capture import CaptureBinding, CaptureConfiguration
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration as CollectorConfiguration,
)
from exp.runtime.gateway.native_capture import (
    CaptureController,
    CaptureDeliveryLimits,
)

if TYPE_CHECKING:
    from exp_gateway_native import CaptureCollector

GATEWAY_CAPTURE_APPLICATION = "gateway"


def open_local_capture(configuration: CaptureConfiguration | None) -> CaptureController | None:
    """Bind local policy and SQLite to the common native collector, never a second tap."""
    if configuration is None:
        return None
    from exp_gateway_native import CaptureCollector

    collector_configuration = CollectorConfiguration(
        settlement_required=False,
        delivery=CaptureDeliveryLimits(maximum_records=configuration.queue_capacity),
    )
    collector: CaptureCollector | None = CaptureCollector.sqlite(
        collector_configuration.model_dump_json(), configuration.model_dump_json()
    )
    if collector is None:
        return None
    bindings = {
        (binding.policy.scope.user_id, binding.alias): binding.policy.scope.application_id
        for binding in configuration.bindings
        if binding.policy.enabled
    }

    def application_for(authorization: AuthorizationSnapshot) -> str | None:
        """Use exact authenticated identifiers, never parsed caller-controlled metadata."""
        if authorization.surface.value not in {"chat_completions", "responses"}:
            return None
        return bindings.get((authorization.identity_id, authorization.alias))

    return CaptureController(collector, application_for=application_for)


def local_capture_path(root: Path) -> Path:
    """Return the separate content database, never the accounting database."""
    return (root / "gateway" / "traffic.db").resolve()


def local_capture_configuration(root: Path, *, ghost: bool = False) -> CaptureConfiguration | None:
    """Enable bounded capture for current granted identities unless explicitly disabled.

    This local policy includes own-provider-key traffic. Hosted consent, BYOK
    exclusions and tenant persistence are deliberately not decided here.
    Bindings are a startup snapshot; restart after changing identity grants.
    """
    if ghost:
        return None
    manager = GatewayManagement(root)
    identities = {identity.identity_id for identity in manager.identities() if identity.active}
    aliases = {alias.alias_name for alias in manager.aliases() if alias.active}
    bindings = tuple(
        CaptureBinding(
            alias=grant.alias_name,
            policy=CapturePolicy(
                scope=ClaasScope(
                    user_id=grant.identity_id,
                    application_id=GATEWAY_CAPTURE_APPLICATION,
                ),
                enabled=True,
            ),
        )
        for grant in manager.grants()
        if grant.identity_id in identities and grant.alias_name in aliases
    )
    if not bindings:
        return None
    return CaptureConfiguration(database_path=local_capture_path(root), bindings=bindings)
