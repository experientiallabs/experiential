"""Frozen task-mixture selection with conservative family grouping and explicit deficits."""

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json


class CoverageStratum(ContractModel):
    """One declared future-user need, rather than a statistic inferred from imported traces.

    Attributes:
        id: Stable policy-local name.
        user: Intended beneficiary.
        capability: Required capability, matched to a saved capability brief.
        category: Regression, frontier or exploratory purpose.
        information: Required information condition, declared by the environment author.
        horizon: Required workflow horizon, not an actor's measured number of attempts.
        minimum_families: Distinct source/workflow components required in this stratum.
    """

    id: str = Field(min_length=1, max_length=128)
    user: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    category: Literal["regression", "frontier", "exploratory"]
    information: Literal["available", "discoverable", "conflicting", "unavailable"]
    horizon: Literal["single_decision", "multi_step", "long_horizon"]
    minimum_families: int = Field(ge=1, le=100, strict=True)


class TaskDistribution(ContractModel):
    """A fixed mixture; changing it creates a new experiment, never a retroactive success.

    Attributes:
        strata: Explicit coverage requirements and quotas.
        seed: Deterministic tie-breaking seed; it never supplies evidence of independence.
    """

    strata: tuple[CoverageStratum, ...] = Field(min_length=1, max_length=50)
    seed: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def distinct(self) -> Self:
        """Refuse ambiguous names and an unbounded output population."""
        if len({row.id for row in self.strata}) != len(self.strata):
            raise ValueError("task distribution stratum IDs must be unique")
        if sum(row.minimum_families for row in self.strata) > 200:
            raise ValueError("task distribution exceeds 200 required families")
        return self


class FamilyCandidate(ContractModel):
    """Evidence-derived grouping keys and declared applicable coverage strata.

    Attributes:
        id: Frozen seed candidate identity.
        strata: Strata whose declared semantics this seed is intended to exercise.
        lineages: Source task groups inherited by all variants.
        traces: Source trace identifiers; overlapping sources cannot add family breadth.
        workflow: Intent, tools and verifier signature, independent of task IDs or wording.
    """

    id: str = Field(min_length=1)
    strata: tuple[str, ...] = Field(min_length=1)
    lineages: tuple[str, ...] = Field(min_length=1)
    traces: tuple[str, ...] = ()
    workflow: Sha256

    @model_validator(mode="after")
    def meaningful_groups(self) -> Self:
        """Empty or repeated group labels cannot create apparent source coverage."""
        for rows in (self.strata, self.lineages, self.traces):
            if any(not value.strip() for value in rows) or len(set(rows)) != len(rows):
                raise ValueError("family grouping keys must be nonempty and distinct")
        return self


class DistributionSelection(ContractModel):
    """An assignment and unmet quotas, retaining the full candidate denominator.

    Attributes:
        selected: Stratum to selected seed IDs, with each connected family used at most once.
        deficits: Missing family count per stratum, including zeroes.
        components: Conservative family components across the entire input population.
        candidates: Number of seeds considered, not the number of independent families.
        scope: Limits on the grouping and semantic coverage claims.
    """

    selected: dict[str, tuple[str, ...]]
    deficits: dict[str, int]
    components: tuple[tuple[str, ...], ...]
    candidates: int
    scope: str = (
        "declared coverage; source/workflow connected components, not proven semantic "
        "independence or future-user usefulness; variants and actor attempts add no families"
    )


def family_components(candidates: tuple[FamilyCandidate, ...]) -> tuple[tuple[str, ...], ...]:
    """Close source and workflow overlaps transitively before any coverage selection."""
    if len({row.id for row in candidates}) != len(candidates):
        raise ValueError("family candidate IDs must be unique")
    parents = {row.id: row.id for row in candidates}

    def root(key: str) -> str:
        """Find a deterministic union root with path compression."""
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owners: dict[tuple[str, str], str] = {}
    for row in sorted(candidates, key=lambda item: item.id):
        keys = [
            *(("lineage", value) for value in row.lineages),
            *(("trace", value) for value in row.traces),
            ("workflow", row.workflow),
        ]
        for key in keys:
            previous = owners.setdefault(key, row.id)
            left, right = sorted((root(row.id), root(previous)))
            parents[right] = left
    groups: dict[str, list[str]] = {}
    for row in candidates:
        groups.setdefault(root(row.id), []).append(row.id)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def select_distribution(
    policy: TaskDistribution,
    candidates: tuple[FamilyCandidate, ...],
    *,
    eligible: frozenset[str] | None = None,
) -> DistributionSelection:
    """Fill fixed quotas by bipartite matching, preserving groups even when members fail.

    An augmenting path avoids the greedy failure where a flexible family consumes the only
    slot available to another family. Eligibility can filter outcomes after measurement, but
    grouping is always computed over the original population, including failed bridge nodes.
    """
    components = family_components(candidates)
    ids = {row.id for row in candidates}
    if eligible is not None and not eligible <= ids:
        raise ValueError("eligible families must belong to the frozen candidate population")
    allowed = {row.id for row in policy.strata}
    if any(not set(row.strata) <= allowed for row in candidates):
        raise ValueError("candidate names an undeclared distribution stratum")
    inventory = {row.id: row for row in candidates}
    slots = [(row.id, index) for row in policy.strata for index in range(row.minimum_families)]
    choices: dict[str, tuple[int, ...]] = {}
    representatives: dict[tuple[str, int], str] = {}
    for stratum in policy.strata:
        available = []
        for index, component in enumerate(components):
            members = [
                key
                for key in component
                if stratum.id in inventory[key].strata and (eligible is None or key in eligible)
            ]
            if members:
                available.append(index)
                representatives[(stratum.id, index)] = min(
                    members, key=lambda key: sha256_json({"seed": policy.seed, "candidate": key})
                )
        choices[stratum.id] = tuple(
            sorted(
                available,
                key=lambda index: sha256_json(
                    {"seed": policy.seed, "component": list(components[index])}
                ),
            )
        )
    assigned: dict[int, tuple[str, int]] = {}

    def augment(slot: tuple[str, int], seen: set[int]) -> bool:
        """Find a free family or move its current slot to another compatible family."""
        for component in choices[slot[0]]:
            if component in seen:
                continue
            seen.add(component)
            if component not in assigned or augment(assigned[component], seen):
                assigned[component] = slot
                return True
        return False

    for slot in slots:
        augment(slot, set())
    selected = {
        row.id: tuple(
            sorted(
                representatives[(row.id, index)]
                for index, slot in assigned.items()
                if slot[0] == row.id
            )
        )
        for row in policy.strata
    }
    return DistributionSelection(
        selected=selected,
        deficits={row.id: row.minimum_families - len(selected[row.id]) for row in policy.strata},
        components=components,
        candidates=len(candidates),
    )
