"""Score-independent grouped benchmark partitions with a sealed confirmation view."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, model_validator

PARTITION_MANIFEST_VERSION: Literal["1"] = "1"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class PartitionTask(BaseModel):
    """Immutable identity and score-independent grouping metadata for one task."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    stratum: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _reject_ambiguous_names(self) -> Self:
        for field in ("task_id", "stratum", "group_id"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(f"{field} cannot have leading or trailing whitespace")
        return self


class StratumCount(BaseModel):
    """Canonical count for one score-independent task stratum."""

    model_config = ConfigDict(frozen=True)

    stratum: str = Field(min_length=1)
    count: StrictInt = Field(ge=0)


class DiscoveryPartition(BaseModel):
    """Proposer-safe partition view containing discovery tasks but no held-out identities."""

    model_config = ConfigDict(frozen=True)

    partition_version: Literal["1"]
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    tasks: tuple[PartitionTask, ...]
    confirmation_strata: tuple[StratumCount, ...]
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)

    @property
    def confirmation_counts(self) -> dict[str, int]:
        """Return held-out counts without revealing held-out task identities."""
        return {item.stratum: item.count for item in self.confirmation_strata}


class ConfirmationPartition(BaseModel):
    """Held-out task view opened only after binding an already-frozen candidate."""

    model_config = ConfigDict(frozen=True)

    partition_version: Literal["1"]
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_hash: str = Field(pattern=_DIGEST_PATTERN)
    tasks: tuple[PartitionTask, ...]
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)


class BenchmarkPartitionManifest(BaseModel):
    """Private control-plane manifest for one grouped discovery/confirmation split."""

    model_config = ConfigDict(frozen=True)

    partition_version: Literal["1"] = PARTITION_MANIFEST_VERSION
    tasks: tuple[PartitionTask, ...]
    discovery_strata: tuple[StratumCount, ...]
    selection_seed: str = Field(min_length=1)
    seal_nonce: str = Field(min_length=16, repr=False)
    discovery_task_ids: tuple[str, ...]
    confirmation_task_ids: tuple[str, ...]
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        tasks: tuple[PartitionTask, ...],
        discovery_counts: dict[str, int],
        selection_seed: str,
        seal_nonce: str,
    ) -> BenchmarkPartitionManifest:
        """Select whole groups to satisfy exact per-stratum discovery quotas."""
        canonical_tasks = _canonical_tasks(tasks)
        canonical_counts = _canonical_counts(
            discovery_counts,
            tasks=canonical_tasks,
        )
        discovery_ids, confirmation_ids = _select_partition(
            canonical_tasks,
            canonical_counts,
            seed=selection_seed,
        )
        commitment = _confirmation_commitment(
            tasks=canonical_tasks,
            confirmation_ids=confirmation_ids,
            nonce=seal_nonce,
        )
        return cls(
            tasks=canonical_tasks,
            discovery_strata=tuple(
                StratumCount(stratum=stratum, count=count)
                for stratum, count in canonical_counts.items()
            ),
            selection_seed=selection_seed,
            seal_nonce=seal_nonce,
            discovery_task_ids=discovery_ids,
            confirmation_task_ids=confirmation_ids,
            confirmation_commitment=commitment,
        )

    @property
    def digest(self) -> str:
        """Return the salted identity of the complete private partition manifest."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def discovery_counts(self) -> dict[str, int]:
        """Return exact per-stratum discovery counts."""
        return {item.stratum: item.count for item in self.discovery_strata}

    @property
    def confirmation_counts(self) -> dict[str, int]:
        """Return exact per-stratum confirmation counts."""
        discovery = self.discovery_counts
        totals = Counter(task.stratum for task in self.tasks)
        return {stratum: totals[stratum] - discovery[stratum] for stratum in sorted(totals)}

    def discovery_view(self) -> DiscoveryPartition:
        """Return the only partition view that optimizer search may receive."""
        discovery = frozenset(self.discovery_task_ids)
        return DiscoveryPartition(
            partition_version=self.partition_version,
            partition_manifest_digest=self.digest,
            tasks=tuple(task for task in self.tasks if task.task_id in discovery),
            confirmation_strata=tuple(
                StratumCount(stratum=stratum, count=count)
                for stratum, count in self.confirmation_counts.items()
            ),
            confirmation_commitment=self.confirmation_commitment,
        )

    def open_confirmation(self, *, candidate_hash: str) -> ConfirmationPartition:
        """Bind held-out identities to a candidate frozen before the view is opened."""
        confirmation = frozenset(self.confirmation_task_ids)
        return ConfirmationPartition(
            partition_version=self.partition_version,
            partition_manifest_digest=self.digest,
            candidate_hash=candidate_hash,
            tasks=tuple(task for task in self.tasks if task.task_id in confirmation),
            confirmation_commitment=self.confirmation_commitment,
        )

    @model_validator(mode="after")
    def _validate_derived_partition(self) -> Self:
        canonical_tasks = _canonical_tasks(self.tasks)
        if self.tasks != canonical_tasks:
            raise ValueError("partition tasks must be in canonical order")
        counts = _canonical_counts(self.discovery_counts, tasks=self.tasks)
        if self.discovery_strata != tuple(
            StratumCount(stratum=stratum, count=count) for stratum, count in counts.items()
        ):
            raise ValueError("discovery strata must be unique and in canonical order")
        expected_discovery, expected_confirmation = _select_partition(
            self.tasks,
            counts,
            seed=self.selection_seed,
        )
        if (
            self.discovery_task_ids != expected_discovery
            or self.confirmation_task_ids != expected_confirmation
        ):
            raise ValueError("partition membership does not match the frozen selection_seed")
        expected_commitment = _confirmation_commitment(
            tasks=self.tasks,
            confirmation_ids=self.confirmation_task_ids,
            nonce=self.seal_nonce,
        )
        if self.confirmation_commitment != expected_commitment:
            raise ValueError("confirmation commitment does not match the sealed partition")
        return self


def _canonical_tasks(tasks: tuple[PartitionTask, ...]) -> tuple[PartitionTask, ...]:
    if not tasks:
        raise ValueError("benchmark partition needs at least one task")
    duplicate_ids = sorted(
        task_id for task_id, count in Counter(task.task_id for task in tasks).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"benchmark partition has duplicate task_id values: {duplicate_ids}")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def _canonical_counts(
    counts: dict[str, int],
    *,
    tasks: tuple[PartitionTask, ...],
) -> dict[str, int]:
    totals = Counter(task.stratum for task in tasks)
    if set(counts) != set(totals):
        missing = sorted(set(totals) - set(counts))
        extra = sorted(set(counts) - set(totals))
        raise ValueError(
            f"discovery count strata differ from tasks: missing={missing}, extra={extra}"
        )
    canonical: dict[str, int] = {}
    for stratum in sorted(totals):
        value = counts[stratum]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("discovery counts must be integers")
        if value < 0 or value > totals[stratum]:
            raise ValueError(f"discovery count for {stratum!r} is outside the task roster")
        canonical[stratum] = value
    if sum(canonical.values()) == 0 or sum(canonical.values()) == len(tasks):
        raise ValueError("benchmark partition needs non-empty discovery and confirmation sets")
    return canonical


def _select_partition(
    tasks: tuple[PartitionTask, ...],
    discovery_counts: dict[str, int],
    *,
    seed: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not seed or seed != seed.strip():
        raise ValueError("selection_seed must be a non-empty canonical string")
    strata = tuple(discovery_counts)
    target = tuple(discovery_counts[stratum] for stratum in strata)
    grouped: defaultdict[str, list[PartitionTask]] = defaultdict(list)
    for task in tasks:
        grouped[task.group_id].append(task)
    ranked_groups = sorted(
        grouped,
        key=lambda group_id: (_digest_bytes(seed, "group-order", group_id), group_id),
    )
    zero = (0,) * len(strata)
    states: dict[tuple[int, ...], tuple[int, tuple[str, ...]]] = {zero: (0, ())}
    for group_id in ranked_groups:
        vector_counts = Counter(task.stratum for task in grouped[group_id])
        vector = tuple(vector_counts[stratum] for stratum in strata)
        weight = int.from_bytes(_digest_bytes(seed, "group-weight", group_id)[:8], "big")
        updated = dict(states)
        for current, (score, selected) in states.items():
            candidate_vector = tuple(
                left + right for left, right in zip(current, vector, strict=True)
            )
            if any(value > limit for value, limit in zip(candidate_vector, target, strict=True)):
                continue
            candidate = (score + weight, tuple(sorted((*selected, group_id))))
            incumbent = updated.get(candidate_vector)
            if incumbent is None or candidate < incumbent:
                updated[candidate_vector] = candidate
        states = updated
    solution = states.get(target)
    if solution is None:
        raise ValueError(
            "whole-group benchmark partition cannot satisfy the exact discovery counts"
        )
    selected_groups = frozenset(solution[1])
    discovery = tuple(sorted(task.task_id for task in tasks if task.group_id in selected_groups))
    confirmation = tuple(
        sorted(task.task_id for task in tasks if task.group_id not in selected_groups)
    )
    return discovery, confirmation


def _confirmation_commitment(
    *,
    tasks: tuple[PartitionTask, ...],
    confirmation_ids: tuple[str, ...],
    nonce: str,
) -> str:
    if len(nonce) < 16 or nonce != nonce.strip():
        raise ValueError("seal_nonce must be a canonical secret with at least 16 characters")
    return _canonical_digest(
        {
            "domain": "wmh-benchmark-confirmation-v1",
            "nonce": nonce,
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "confirmation_task_ids": list(confirmation_ids),
        }
    )


def _digest_bytes(*parts: str) -> bytes:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.digest()


def _canonical_digest(value: JsonValue) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()
