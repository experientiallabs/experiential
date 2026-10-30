"""The geometry representative-task selection clusters in, chosen explicitly.

:attr:`~exp.simulation.mining.deduplicate.DeduplicatedTrace.selection_vector` is the one vector
core-set geometry reads: clustering, medoids, farthest-first coverage, and nearest-representative
assignment. Duplicate detection, lineage, and partitioning keep using the request-visible
descriptor vector, so changing selection geometry never moves the leakage boundary.

A :data:`SelectionSpec` names which geometry to build:

- **hashing** embeds the request-visible descriptor text. It needs no provider, so it is what a
  provider-free build uses, and it groups cases by how their requests are worded.
- **decision_tag** embeds a decision model's answers about what each case turned out to contain.
  Two episodes of one template that differ only in what the environment revealed sit apart in this
  geometry and together in the hashing one, which is the whole reason it exists.

A ``decision_tag`` spec with no tag matrix raises. It never falls back to hashing, because a
silent geometry fallback is indistinguishable from success: selection still runs, still returns
tasks, and still reports coverage, while measuring something nobody asked for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.simulation.mining.descriptors import (
    HashingDescriptorEmbedder,
    RoutingDescriptor,
    normalized_vectors,
)
from exp.simulation.mining.questions import (
    QuestionBank,
    QuestionBankError,
    packaged_question_bank,
)
from exp.simulation.mining.tagging import TagMatrix

TagTransform = Literal["none", "l2", "hellinger", "logit"]
"""How tag probabilities become coordinates.

``none`` passes the probabilities through unchanged and is the one transform that does not
normalize, so distances keep reflecting how many tags a case triggered. ``l2`` unit-normalizes,
making distance reflect which tags a case triggered rather than how many. ``hellinger`` takes the
elementwise square root first, which is the natural metric between probability vectors and pulls
apart cases whose probabilities are small but not zero. ``logit`` takes clamped log-odds first,
which spreads confident answers away from uncertain ones instead of compressing them at the
interval's ends.
"""

LOGIT_CLAMP = 1e-3
"""Probabilities are clamped into ``[LOGIT_CLAMP, 1 - LOGIT_CLAMP]`` before log-odds, because a
certain answer has infinite log-odds and one such column would otherwise dominate every distance.
"""


class SelectionGeometryError(ValueError):
    """A selection spec cannot be turned into vectors from the supplied evidence."""


class HashingSelectionSpec(ContractModel):
    """Core-set geometry from request-visible descriptor text, with no provider call.

    Attributes:
        kind: Fixed hashing discriminator, defaulting to ``hashing``.
        dimensions: Width of the signed hashing vector. At least 8, as the embedder requires.
    """

    kind: Literal["hashing"] = "hashing"
    dimensions: int = Field(default=64, ge=8)


class DecisionTagSelectionSpec(ContractModel):
    """Core-set geometry from a decision model's answers about each case.

    The bank and families decide the columns, so the geometry is readable: a coordinate is a
    named proposition about the case, not an opaque hash bucket.

    Attributes:
        kind: Fixed decision-tag discriminator, defaulting to ``decision_tag``.
        bank_id: Packaged question bank whose families define the columns and their order.
        families: Families of that bank to use, in bank order. Defaults to ``situation``, the
            family describing what the case turned out to contain.
        transform: How probabilities become coordinates. See :data:`TagTransform`.
    """

    kind: Literal["decision_tag"] = "decision_tag"
    bank_id: str = Field(min_length=1)
    families: tuple[str, ...] = ("situation",)
    transform: TagTransform = "l2"


SelectionSpec = Annotated[
    HashingSelectionSpec | DecisionTagSelectionSpec, Field(discriminator="kind")
]


def selection_vectors(
    spec: SelectionSpec,
    *,
    descriptors: Sequence[RoutingDescriptor],
    tag_matrix: TagMatrix | None = None,
    bank: QuestionBank | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Build one core-set geometry vector per descriptor.

    The result is positional: vector ``i`` belongs to ``descriptors[i]`` and is ready to assign
    to that candidate's ``selection_vector``. Every transform but ``none`` returns unit vectors.

    Args:
        spec: Which geometry to build.
        descriptors: Request-visible descriptors of the selection candidates, in order.
        tag_matrix: Decision-model answers covering every descriptor's trace. Required by a
            ``decision_tag`` spec and unused by a ``hashing`` one.
        bank: The bank whose families name the columns. Omit it for a packaged bank, which is
            read by ``spec.bank_id``; supply it when the bank came from a local file.

    Returns:
        One vector per descriptor, in input order, all of the same width.

    Raises:
        SelectionGeometryError: A ``decision_tag`` spec has no tag matrix, the matrix or the
            bank carries a different bank identity, the matrix omits a descriptor's trace or a
            selected column, or a normalizing transform produced a vector with no direction.
        ValueError: The hashing embedder produced an unusable vector.
    """
    if not descriptors:
        return ()
    if isinstance(spec, HashingSelectionSpec):
        return normalized_vectors(HashingDescriptorEmbedder(spec.dimensions), descriptors)
    if tag_matrix is None:
        raise SelectionGeometryError(
            f"selection spec {spec.kind!r} needs the tag matrix for bank {spec.bank_id!r}; "
            "tag the traces with a TraceTagger and pass tag_matrix, or choose a "
            "HashingSelectionSpec to select on request-visible text alone"
        )
    return _tag_vectors(spec, descriptors, tag_matrix, bank)


def _tag_vectors(
    spec: DecisionTagSelectionSpec,
    descriptors: Sequence[RoutingDescriptor],
    tag_matrix: TagMatrix,
    bank: QuestionBank | None,
) -> tuple[tuple[float, ...], ...]:
    """Read the selected columns out of a tag matrix and transform each descriptor's row."""
    for identity, source in ((tag_matrix.bank_id, "tag matrix"), (bank and bank.bank_id, "bank")):
        if identity is not None and identity != spec.bank_id:
            raise SelectionGeometryError(
                f"selection spec names bank {spec.bank_id!r} but the {source} carries "
                f"{identity!r}; tag the traces with the bank the spec names"
            )
    indexes = _column_indexes(spec, tag_matrix, bank)
    rows: list[tuple[float, ...]] = []
    for descriptor in descriptors:
        try:
            row = tag_matrix.row(descriptor.trace_id)
        except ValueError as exc:
            raise SelectionGeometryError(str(exc)) from exc
        rows.append(tuple(row.values[index] for index in indexes))
    return tuple(
        _transformed(values, spec.transform, descriptor.trace_id)
        for values, descriptor in zip(rows, descriptors, strict=True)
    )


def _column_indexes(
    spec: DecisionTagSelectionSpec, tag_matrix: TagMatrix, bank: QuestionBank | None
) -> tuple[int, ...]:
    """Return the matrix positions of the spec's columns, in bank order.

    Bank order rather than matrix order is the contract, so two matrices tagged from one bank
    produce the same coordinate meaning even if their columns were stored differently.
    """
    try:
        resolved = bank or packaged_question_bank(spec.bank_id)
        selected = resolved.families(*spec.families).question_ids()
    except QuestionBankError as exc:
        raise SelectionGeometryError(str(exc)) from exc
    missing = [name for name in selected if name not in tag_matrix.question_ids]
    if missing:
        raise SelectionGeometryError(
            f"tag matrix for bank {spec.bank_id!r} is missing columns {missing} required by "
            f"families {list(spec.families)}; tag the traces with those families"
        )
    return tuple(tag_matrix.question_ids.index(name) for name in selected)


def _transformed(
    values: tuple[float, ...], transform: TagTransform, trace_id: str
) -> tuple[float, ...]:
    """Apply one transform to a single row's probabilities.

    Args:
        values: One row's selected probabilities, in column order.
        transform: Which transform to apply.
        trace_id: Trace the row describes, named in a failure.

    Returns:
        The row's coordinates. Unit length for every transform but ``none``.

    Raises:
        SelectionGeometryError: A normalizing transform received a row with no direction, which
            happens when every selected probability is zero.
    """
    if transform == "none":
        return values
    if transform == "hellinger":
        return _unit(tuple(math.sqrt(value) for value in values), trace_id, transform)
    if transform == "logit":
        clamped = (min(max(value, LOGIT_CLAMP), 1.0 - LOGIT_CLAMP) for value in values)
        return _unit(tuple(math.log(value / (1.0 - value)) for value in clamped), trace_id, "logit")
    return _unit(values, trace_id, transform)


def _unit(values: tuple[float, ...], trace_id: str, transform: str) -> tuple[float, ...]:
    """Scale one vector to unit length, refusing a vector with no direction."""
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise SelectionGeometryError(
            f"the {transform} transform left trace {trace_id} with no direction: every selected "
            "tag is zero, so this case has no position relative to the others. Use the 'logit' "
            "transform, which keeps a confident no at a finite coordinate, or select families "
            "that this corpus actually triggers"
        )
    return tuple(value / norm for value in values)
