"""`HarnessDelta`: the typed update object `wmh harness create` searches through.

The update representation IS the search space: everything the meta-agent can learn about *how to
improve harnesses* is bounded by what the update object can express. A raw file edit expresses
almost nothing — no typed target, no assertion about what it believed it was editing, no per-change
rationale, no verdict. A delta expresses all four:

- **trigger** (`FailureSignature`): the clustered failure mechanism this delta answers to, so the
  archive can ask "which kinds of edits work on which failure classes?" instead of "what changed?".
- **preconditions**: `surface_id -> expected content hash` of every surface the proposer read
  before editing. Application is atomic — ANY mismatch rejects the whole delta before a token of
  eval budget is spent, so a proposal drafted against one parent can never silently misapply to
  another.
- **ops** (`SurfaceOp`): add/replace/remove keyed by surface *identity*, each carrying its own
  rationale bound to the op it justifies.
- **verdict** (`GateRecord`): the acceptance decision and the gate deltas that produced it, written
  back onto the delta. The archive is a lineage of audited deltas, not a pile of snapshots.

`expected_effect` makes every delta a falsifiable prediction: at gate time the trigger cluster is
re-checked and the outcome lands in `verdict.reason`, measuring the proposer's calibration over
time, not just its win rate.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind


class FailureSignature(BaseModel):
    """The clustered failure mechanism a delta answers to: WHY it exists, queryably.

    Built deterministically from a closed-loop report (`wmh.harness.create.cluster_failures`),
    never free-typed by the proposer — so the archive's mechanism labels are comparable across
    deltas and runs.
    """

    mechanism: str  # e.g. the shared unmet assertion, or "none: all tasks pass"
    task_ids: list[str] = Field(default_factory=list)  # the failing tasks exhibiting it
    unmet_assertions: list[str] = Field(default_factory=list)  # deduped, order-stable


class SurfaceOp(BaseModel):
    """One typed edit to one surface, addressed by identity, justified in place."""

    op: Literal["add", "replace", "remove"]
    surface_id: str
    kind: SurfaceKind | None = None  # required on add; on replace must match the existing kind
    content: str | None = None  # the full new content (component rewrite, never a line diff)
    budget: int | None = Field(default=None, ge=1)  # replace: None inherits the existing budget
    rationale: str  # why THIS op, bound to the op — not one motivation string for the whole delta

    @model_validator(mode="after")
    def _validate_shape(self) -> SurfaceOp:
        if self.op == "add" and self.kind is None:
            raise ValueError(f"add of {self.surface_id!r} must declare a kind")
        if self.op in ("add", "replace") and self.content is None:
            raise ValueError(f"{self.op} of {self.surface_id!r} must carry content")
        if self.op == "remove" and self.content is not None:
            raise ValueError(
                f"remove of {self.surface_id!r} carries content; a remove deletes, it never writes"
            )
        if not self.rationale.strip():
            raise ValueError(f"{self.op} of {self.surface_id!r} has no rationale")
        return self


class GateRecord(BaseModel):
    """The acceptance verdict, filled at evaluation time and persisted on the delta."""

    suite_delta: float = 0.0  # regression suite: child - champion (tier 1; >= 0 to pass)
    full_delta: float = 0.0  # full split: child - best seen (tier 2; >= 0 to pass)
    holdout_delta: float | None = None  # held-out split (tier 3; None when no holdout given)
    accepted: bool
    reason: str  # accept/reject reasoning, incl. whether `expected_effect` came true


class HarnessDelta(BaseModel):
    """One proposed update to a `HarnessDoc`, with its audit trail.

    Lineage is by content, not name: `parent_doc_hash` names exactly the document the delta was
    proposed against, and `child_doc_hash` (recorded on successful application) names exactly what
    it produced. A doc is reconstructable by folding accepted deltas from any ancestor snapshot.
    """

    delta_id: str
    parent_doc_hash: str
    trigger: FailureSignature
    # surface_id -> expected content hash of the parent surface the proposer read. Every
    # replace/remove target MUST appear here; application atomically rejects on any mismatch.
    preconditions: dict[str, str] = Field(default_factory=dict)
    ops: list[SurfaceOp] = Field(min_length=1)
    expected_effect: str  # falsifiable: e.g. "the trigger cluster's tasks flip to pass"
    child_doc_hash: str | None = None  # set by apply_delta
    verdict: GateRecord | None = None  # set by the gate; None until evaluated


def compute_delta_id(parent_doc_hash: str, ops: list[SurfaceOp]) -> str:
    """Deterministic delta identity: what it does to what, independent of when it was proposed."""
    joined = parent_doc_hash + "".join(
        f"\x00{op.op}\x00{op.surface_id}\x00{op.content or ''}\x00{op.budget or ''}" for op in ops
    )
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


def apply_delta(parent: HarnessDoc, delta: HarnessDelta, child_name: str) -> HarnessDoc:
    """Apply `delta` to `parent` atomically; returns the child with `delta.child_doc_hash` set.

    Every check runs before anything is built, and the child itself re-validates as a whole
    `HarnessDoc` at construction — an invalid delta must fail here, before any eval budget is
    spent on the variant it would have produced. Raises `ValueError` on any violation.
    """
    _check_lineage(parent, delta)
    _check_preconditions(parent, delta)
    _check_ops(parent, delta)

    surfaces = {s.id: s for s in parent.surfaces}
    for op in delta.ops:
        if op.op == "remove":
            del surfaces[op.surface_id]
            continue
        existing = surfaces.get(op.surface_id)
        # Narrowing only: SurfaceOp's model validator guarantees add carries a kind and
        # add/replace carry content, and _check_ops guarantees replace targets exist.
        kind = op.kind if op.kind is not None else existing.kind if existing else None
        budget = op.budget if op.budget is not None else existing.budget if existing else None
        if kind is None or op.content is None:
            raise ValueError(f"malformed {op.op} of {op.surface_id!r}")
        surfaces[op.surface_id] = Surface(
            id=op.surface_id, kind=kind, content=op.content, budget=budget
        )

    child = HarnessDoc(name=child_name, surfaces=list(surfaces.values()))
    delta.child_doc_hash = child.doc_hash
    return child


def _check_lineage(parent: HarnessDoc, delta: HarnessDelta) -> None:
    if delta.parent_doc_hash != parent.doc_hash:
        raise ValueError(
            f"delta {delta.delta_id} was proposed against doc {delta.parent_doc_hash[:12]}, "
            f"not {parent.doc_hash[:12]} ({parent.name} v{parent.version})"
        )


def _check_preconditions(parent: HarnessDoc, delta: HarnessDelta) -> None:
    for surface_id, expected in delta.preconditions.items():
        surface = parent.surface(surface_id)
        if surface is None:
            raise ValueError(f"precondition on unknown surface {surface_id!r}")
        if surface.content_hash != expected:
            raise ValueError(
                f"precondition mismatch on {surface_id!r}: expected {expected[:12]}, "
                f"parent has {surface.content_hash[:12]} — the delta was drafted against "
                "different content"
            )


def _check_ops(parent: HarnessDoc, delta: HarnessDelta) -> None:
    targets = [op.surface_id for op in delta.ops]
    duplicates = sorted({t for t in targets if targets.count(t) > 1})
    if duplicates:
        raise ValueError(f"multiple ops target the same surface(s): {duplicates}")
    for op in delta.ops:
        existing = parent.surface(op.surface_id)
        if op.op == "add":
            if existing is not None:
                raise ValueError(f"add of {op.surface_id!r}: the surface already exists")
        else:
            if existing is None:
                raise ValueError(f"{op.op} of unknown surface {op.surface_id!r}")
            if op.kind is not None and op.kind is not existing.kind:
                raise ValueError(
                    f"{op.op} of {op.surface_id!r} declares kind {op.kind.value!r}; the surface "
                    f"is {existing.kind.value!r}"
                )
            if op.surface_id not in delta.preconditions:
                raise ValueError(
                    f"{op.op} of {op.surface_id!r} has no precondition: an update must assert "
                    "the content hash of every surface it replaces or removes"
                )
