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
    resolve_question_bank,
)
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.mining.views import TraceViewName, TraceViewSpec

TagTransform = Literal["none", "l2", "hellinger", "logit", "indicator"]
"""How tag probabilities become coordinates.

``none`` passes the probabilities through unchanged and is the one transform that does not
normalize, so distances keep reflecting how many tags a case triggered. It is what a clusterer
reading probabilities as probabilities consumes, and it is not a selection geometry: core-set
distance is a dot product of pre-normalized vectors, and raw probability rows saturate its clamp.

``indicator`` rounds every probability to 0.0 or 1.0 at :data:`INDICATOR_THRESHOLD`, so a case
becomes the *set* of propositions it fired. It is the second transform that does not normalize, and
deliberately so: a set-overlap distance such as Jaccard reads how large each set is, and dividing a
row by its own norm destroys exactly that. An all-zero row is a legal position here, meaning "this
case fired nothing", rather than the failure it is for a normalizing transform.

It is an explicit opt-in and must not become a default, for two reasons that compound. These rows
are not unit vectors, so the core-set dot product saturates on them exactly as it does on ``none``:
two cases sharing one fired proposition come out at the clamp, and on a real tagged corpus that is
most pairs. Worse, a selection phase cannot recover from that by rescaling, because every member of
one ``pattern_hierarchy`` cluster cut at ``0.0`` carries, by construction, the identical row. There
is nothing left for a within-cluster medoid or a facility-location pick to measure, so the choice
falls entirely to the identity tie-break. A clusterer that wants these sets therefore thresholds
the untransformed answers itself, which is what
:class:`~exp.simulation.mining.clustering.specs.PatternHierarchyClusterSpec` does, and selection
keeps measuring in a continuous geometry.

``l2`` unit-normalizes, making distance reflect which tags a case triggered rather than how many.

``hellinger`` is the Hellinger embedding of independent Bernoulli tags: each probability ``p``
becomes the pair ``(sqrt(p), sqrt(1 - p))``, so the result has ``2T`` columns and squared Euclidean
distance equals the summed per-tag squared Hellinger distance. Carrying both outcomes is what makes
it a metric between distributions rather than between the ``sqrt(p)`` numbers alone: under the
one-sided form two cases that agree about which tags fired but disagree about how strongly come out
closer than they are, and a case that triggers nothing has no position at all.

``logit`` takes clamped log-odds first, which spreads confident answers away from uncertain ones
instead of compressing them at the interval's ends.
"""

INDICATOR_THRESHOLD = 0.5
"""Probability above which the ``indicator`` transform counts a proposition as fired.

It is a module constant rather than a spec field because the only consumer that needs to choose a
threshold is a clusterer, and
:attr:`~exp.simulation.mining.clustering.specs.PatternHierarchyClusterSpec.threshold` is where that
choice is recorded. A caller selecting on ``indicator`` directly is asking for the ordinary reading
of a probability, which is the even split.
"""

LOGIT_CLAMP = 1e-4
"""Probabilities are clamped into ``[LOGIT_CLAMP, 1 - LOGIT_CLAMP]`` before log-odds, because a
certain answer has infinite log-odds and one such column would otherwise dominate every distance.

``1e-4`` rather than something looser because the clamp is the only thing deciding how far a
certain answer sits from a merely confident one: at ``1e-3`` the extreme coordinate is log(999) and
at ``1e-4`` it is log(9999), and anything that moves this bound changes every distance in the
geometry.
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
        families: Families of that bank to use, in bank order. Empty, the default, uses every
            family the bank declares, so a spec never has to name one bank's internals.
        transform: How probabilities become coordinates. See :data:`TagTransform`. It defaults to
            ``l2``, which keeps the geometry continuous and unit length, because that is what the
            core-set metric measures in and what every selection phase after clustering reads. A
            clusterer that wants the thresholded fired-proposition sets derives them from the
            untransformed answers rather than asking for them here.
        view: Which rendered view the questions are asked about. It defaults to ``task``, the
            request-visible view: it is the one view a live request could also produce, so it is
            the only router-safe one and it is already audited for leakage, and it is about 3.6k
            tokens against the observations view's 11.8k, which is most of what tagging a corpus
            costs. ``observations`` shows what the episode revealed without the agent's own
            commentary, which separates two runs of one template that differ in what the
            environment returned; choose it when that distinction is the one being mined.
        view_spec: Rendering controls for that view. They belong to the spec because they decide
            the exact text that was tagged, and answers about differently capped text are answers
            about different evidence.
    """

    kind: Literal["decision_tag"] = "decision_tag"
    bank_id: str = Field(min_length=1)
    families: tuple[str, ...] = ()
    transform: TagTransform = "l2"
    view: TraceViewName = "task"
    view_spec: TraceViewSpec = TraceViewSpec()


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
    to that candidate's ``selection_vector``. Every transform returns unit vectors except ``none``,
    which passes the probabilities through, and ``indicator``, which returns raw 0/1 membership
    because a set-overlap distance has to read cardinality.

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
    rows = selection_probabilities(spec, descriptors=descriptors, tag_matrix=tag_matrix, bank=bank)
    return tuple(
        _transformed(values, spec.transform, descriptor.trace_id)
        for values, descriptor in zip(rows, descriptors, strict=True)
    )


def selection_probabilities(
    spec: DecisionTagSelectionSpec,
    *,
    descriptors: Sequence[RoutingDescriptor],
    tag_matrix: TagMatrix,
    bank: QuestionBank | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Read the spec's columns out of a tag matrix without transforming them.

    These are the answers as the decision model gave them. A clusterer that models tags as
    probabilities needs them in that form, and deriving them back out of a normalized or log-odds
    geometry is not possible, so the untransformed rows are published alongside the vectors rather
    than reconstructed.

    Args:
        spec: Which bank, families and columns to read.
        descriptors: Request-visible descriptors of the selection candidates, in order.
        tag_matrix: Decision-model answers covering every descriptor's trace.
        bank: The bank whose families name the columns. Omit it for a packaged bank.

    Returns:
        One row of probabilities per descriptor, in input order, in bank column order.

    Raises:
        SelectionGeometryError: The matrix or the bank carries a different bank identity, or the
            matrix omits a descriptor's trace or a selected column.
    """
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
    return tuple(rows)


def _column_indexes(
    spec: DecisionTagSelectionSpec, tag_matrix: TagMatrix, bank: QuestionBank | None
) -> tuple[int, ...]:
    """Return the matrix positions of the spec's columns, in bank order.

    Bank order rather than matrix order is the contract, so two matrices tagged from one bank
    produce the same coordinate meaning even if their columns were stored differently.
    """
    try:
        resolved = bank or resolve_question_bank(spec.bank_id)
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
        The row's coordinates. Unit length for every transform except ``none``, which is returned
        unchanged, and ``indicator``, which returns unnormalized 0/1 membership.

    Raises:
        SelectionGeometryError: A normalizing transform received a row with no direction, which
            happens when every selected probability is zero.
    """
    if transform == "none":
        return values
    if transform == "indicator":
        # Deliberately returned without reaching _unit. A case that fired nothing is a real,
        # readable position in set space, and it is the position a Jaccard backend handles by
        # naming it rather than by refusing it, so the no-direction guard must not see this row.
        return tuple(1.0 if value > INDICATOR_THRESHOLD else 0.0 for value in values)
    if transform == "hellinger":
        # Both outcomes of every tag, which is what makes squared Euclidean distance equal the
        # summed squared Hellinger distance exactly. Every row leaves this with norm sqrt(T),
        # since the two halves sum to T, so the unit scaling below divides every row by the same
        # constant: distances shrink by 1 / T and no ratio, ordering or cosine moves. The
        # reference form's sqrt(w / 2) factor is a second such constant under uniform weights and
        # washes out identically, which is why it is not applied here. Unit length is what the
        # core-set metric requires of a selection vector.
        roots = tuple(math.sqrt(value) for value in values)
        complements = tuple(math.sqrt(max(0.0, 1.0 - value)) for value in values)
        return _unit((*roots, *complements), trace_id, transform)
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
            "tag is zero, so this case has no position relative to the others. Use the 'logit' or "
            "'hellinger' transform, either of which keeps a confident no at a finite coordinate, "
            "or select families that this corpus actually triggers"
        )
    return tuple(value / norm for value in values)
