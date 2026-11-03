"""Explicit corpus scopes that restrict which fit evidence a world model may learn or retrieve from.

A build's fit-only index holds every fit transition. Some studies need narrower corpora: a world
model constructed from one reserved slice of fit lineages and retrieving from another, possibly
overlapping, slice, while the rest of the fit evidence stays unread for later probing. A
:class:`CorpusScope` is the declaration of one such slice. It names the allowed lineage or trace
identities as a sorted, hashed set and records the identity of the plan that chose them, so the
restriction is part of the artifact identity rather than an unrecorded filter.

A scope never widens access: it is only ever intersected with fit evidence, and every consumer
refuses held-out evidence inside it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json

ScopeKind = Literal["lineage", "trace"]
"""Whether a scope lists leakage lineages or individual source traces."""


class CorpusScope(ContractModel):
    """One explicit, hashed set of evidence identities a corpus is restricted to.

    Attributes:
        name: Stable purpose of the scope, such as ``world-learning-retrieval``.
        kind: ``lineage`` admits every trace of a listed lineage; ``trace`` admits listed traces.
        ids: Sorted, unique, nonempty lineage or trace identities.
        ids_sha256: SHA-256 of the canonical JSON list of ``ids``; must match.
        source_plan: Identity of the plan that chose the scope: an artifact pointer or a
            SHA-256 digest supplied by the caller.
    """

    name: str = Field(min_length=1, max_length=64)
    kind: ScopeKind
    ids: tuple[str, ...] = Field(min_length=1)
    ids_sha256: Sha256
    source_plan: ArtifactInput | Sha256

    @model_validator(mode="after")
    def _require_sorted_hashed_ids(self) -> CorpusScope:
        """Refuse an unordered, repeated, or misdigested identity set.

        Returns:
            The validated scope.

        Raises:
            ValueError: Identities repeat, are unsorted, are empty strings, or differ from
                their recorded digest.
        """
        if any(not item for item in self.ids):
            raise ValueError("corpus scope identities must be nonempty")
        if len(set(self.ids)) != len(self.ids) or self.ids != tuple(sorted(self.ids)):
            raise ValueError("corpus scope identities must be sorted and unique")
        if sha256_json(list(self.ids)) != self.ids_sha256:
            raise ValueError("corpus scope digest differs from its identities")
        return self

    @classmethod
    def declare(
        cls,
        *,
        name: str,
        kind: ScopeKind,
        ids: Iterable[str],
        source_plan: ArtifactInput | str,
    ) -> CorpusScope:
        """Declare a scope over identities in any order, sorting and hashing them.

        Args:
            name: Stable purpose of the scope.
            kind: Whether ``ids`` are lineage or trace identities.
            ids: Allowed identities; repeats are collapsed.
            source_plan: Artifact pointer or SHA-256 digest of the plan that chose them.

        Returns:
            The canonical scope.

        Raises:
            ValueError: No identity is given, or the name or plan identity is invalid.
        """
        ordered = tuple(sorted(set(ids)))
        return cls(
            name=name,
            kind=kind,
            ids=ordered,
            ids_sha256=sha256_json(list(ordered)),
            source_plan=source_plan,
        )

    def admits(self, *, trace_id: str, lineage_id: str) -> bool:
        """Report whether one source trace falls inside the scope.

        Args:
            trace_id: Canonical source trace identity.
            lineage_id: Leakage lineage of that trace.

        Returns:
            True when the scope lists the trace (trace scope) or its lineage (lineage scope).
        """
        return (trace_id if self.kind == "trace" else lineage_id) in self.ids
