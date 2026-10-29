"""Resolve platform and identity policies through one scoped policy store."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy


class GuardrailPolicyStore(Protocol):
    """Resolve a frozen ordered policy set for one authorized request.

    The authorization carries the authenticated organization and identity plus
    the alias and its frozen target, so a store may scope a policy to an
    identity, an alias, or an exact-model pool.
    """

    def policies_for(self, authorization: AuthorizationSnapshot) -> tuple[GuardrailPolicy, ...]:
        """Return every applicable policy, with authenticated decision metadata."""
        ...


class MappingGuardrailStore:
    """Operator-owned scope resolution; identity assignments cannot replace global checks."""

    def __init__(self, policies: Iterable[GuardrailPolicy] = ()) -> None:
        """Index global policies and one assignment per identity, rejecting duplicates."""
        self._global: list[GuardrailPolicy] = []
        self._identities: dict[tuple[str, str], GuardrailPolicy] = {}
        ids: set[str] = set()
        for policy in policies:
            if policy.policy_id in ids:
                raise ValueError("guardrail policy IDs must be unique")
            ids.add(policy.policy_id)
            if policy.organization_id is None or policy.identity_id is None:
                self._global.append(policy)
                continue
            key = (policy.organization_id, policy.identity_id)
            if key in self._identities:
                raise ValueError("guardrail policies must be unique per organization and identity")
            self._identities[key] = policy

    def policies_for(self, authorization: AuthorizationSnapshot) -> tuple[GuardrailPolicy, ...]:
        """Apply global policies first, followed by this identity's optional policy.

        Each policy also applies only to the alias targets it covers: a policy
        with ``target_pool_ids`` skips direct aliases of other exact-model pools,
        so those requests get no classifier call from it.
        """
        organization_id = authorization.organization_id
        identity_id = authorization.identity_id
        target = authorization.target
        assigned = self._identities.get((organization_id, identity_id))
        return tuple(
            p.bind(organization_id, identity_id) for p in self._global if p.applies_to(target)
        ) + (() if assigned is None or not assigned.applies_to(target) else (assigned,))
