"""Organization-and-identity lookup for immutable guardrail policies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from exp.runtime.gateway.contracts import IdentityId, OrganizationId
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy


class GuardrailPolicyStore(Protocol):
    """Resolve at most one assigned policy for an authenticated identity."""

    def policy_for(
        self,
        organization_id: OrganizationId,
        identity_id: IdentityId,
    ) -> GuardrailPolicy | None:
        """Return the assigned policy, or ``None`` when the identity is unguarded."""
        ...


class MappingGuardrailStore:
    """In-memory policy map keyed by organization and identity.

    Missing pairs return ``None`` so the existing gateway hot path stays
    unchanged. The store never inspects request content.
    """

    def __init__(self, policies: Iterable[GuardrailPolicy] = ()) -> None:
        """Index one optional policy per organization-scoped identity.

        Args:
            policies: Immutable policies. Each ``(organization_id, identity_id)``
                pair may appear at most once.

        Raises:
            ValueError: Two policies share the same organization and identity.
        """
        indexed: dict[tuple[str, str], GuardrailPolicy] = {}
        for policy in policies:
            key = (policy.organization_id, policy.identity_id)
            if key in indexed:
                raise ValueError("guardrail policies must be unique per organization and identity")
            indexed[key] = policy
        self._policies = indexed

    def policy_for(
        self,
        organization_id: OrganizationId,
        identity_id: IdentityId,
    ) -> GuardrailPolicy | None:
        """Return the assigned policy, or ``None`` when the pair is unguarded."""
        return self._policies.get((organization_id, identity_id))
