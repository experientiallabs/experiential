"""Identity-keyed lookup for immutable guardrail policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from exp.runtime.gateway.contracts import IdentityId
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy


class GuardrailPolicyStore(Protocol):
    """Resolve at most one assigned policy for an authenticated identity."""

    def policy_for(self, identity_id: IdentityId) -> GuardrailPolicy | None:
        """Return the assigned policy, or ``None`` when the identity is unguarded."""
        ...


class MappingGuardrailStore:
    """In-memory policy map keyed by identity.

    Missing identities return ``None`` so the existing gateway hot path stays
    unchanged. The store never inspects request content.
    """

    def __init__(self, policies: Mapping[str, GuardrailPolicy] | None = None) -> None:
        """Index one optional policy per identity.

        Args:
            policies: Identity ID to immutable policy. Duplicate identities
                keep the last authored entry.

        Raises:
            ValueError: A policy's identity_id does not match its map key.
        """
        indexed: dict[str, GuardrailPolicy] = {}
        for identity_id, policy in (policies or {}).items():
            if policy.identity_id != identity_id:
                raise ValueError("guardrail policy identity must match the store key")
            indexed[identity_id] = policy
        self._policies = indexed

    def policy_for(self, identity_id: IdentityId) -> GuardrailPolicy | None:
        """Return the assigned policy, or ``None`` when the identity is unguarded."""
        return self._policies.get(identity_id)
