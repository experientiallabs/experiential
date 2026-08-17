"""One resumable Python service for human rubric review and immutable finalization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel, JsonObject, stable_id
from wmo.common.judging.labels import root_review_object
from wmo.common.judging.proposal import (
    RubricProposal,
    RubricProposalError,
    write_rubric_proposal_evidence,
)
from wmo.common.judging.provenance import (
    JudgingProvenanceError,
    resolve_artifact,
    sorted_verified_inputs,
)
from wmo.common.judging.rubric import Rubric, RubricDimension, ScoreAnchor
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ProjectStore,
    coordinate_completed_build_selection,
)


class RubricReviewError(RuntimeError):
    """Raised when a rubric review transition is invalid or already finalized."""


class RubricReviewEvent(ContractModel):
    """One append-only human review decision made in the local mutable draft."""

    event_id: ArtifactId
    kind: Literal["accept", "reject", "edit", "add", "replace_all", "order", "finalize"]
    dimension_ids: tuple[ArtifactId, ...]
    created_at: AwareDatetime


class RubricReviewDraft(ContractModel):
    """The rubric namespace stored inside the project's sole mutable review draft."""

    schema_version: int = Field(default=1, ge=1)
    source_task_set_id: ArtifactId
    code_revision: str = Field(min_length=1, max_length=256)
    proposals: tuple[RubricProposal, ...] = ()
    dimensions: tuple[RubricDimension, ...] = ()
    rejected_dimension_ids: tuple[ArtifactId, ...] = ()
    events: tuple[RubricReviewEvent, ...] = ()
    status: Literal["draft", "finalized"] = "draft"
    finalized_rubric: Rubric | None = None

    @field_validator("dimensions")
    @classmethod
    def _require_unique_dimensions(
        cls, value: tuple[RubricDimension, ...]
    ) -> tuple[RubricDimension, ...]:
        dimension_ids = tuple(dimension.dimension_id for dimension in value)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("reviewed rubric dimensions must have unique IDs")
        return value

    @field_validator("rejected_dimension_ids")
    @classmethod
    def _require_unique_rejections(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("rejected rubric dimension IDs must not repeat")
        return value

    @field_validator("events")
    @classmethod
    def _require_unique_events(
        cls, value: tuple[RubricReviewEvent, ...]
    ) -> tuple[RubricReviewEvent, ...]:
        event_ids = tuple(event.event_id for event in value)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("rubric review event IDs must not repeat")
        return value

    @model_validator(mode="after")
    def _require_consistent_finalization(self) -> RubricReviewDraft:
        for proposal in self.proposals:
            if proposal.source_task_set_id != self.source_task_set_id:
                raise ValueError("rubric proposal must retain the review task-set identity")
            if set(proposal.source_lineage_ids).intersection(
                proposal.excluded_router_held_out_lineage_ids
            ):
                raise ValueError("rubric proposal cannot use router-held-out evidence")
        if self.status == "finalized" and self.finalized_rubric is None:
            raise ValueError("finalized rubric reviews require a rubric artifact")
        if self.status == "draft" and self.finalized_rubric is not None:
            raise ValueError("draft rubric reviews cannot carry a finalized rubric")
        if self.finalized_rubric is not None:
            if self.finalized_rubric.source_task_set_id != self.source_task_set_id:
                raise ValueError("finalized rubric must retain the review task-set identity")
            if self.finalized_rubric.dimensions != self.dimensions:
                raise ValueError("finalized rubric dimensions must equal the locked review order")
        return self


class RubricReview:
    """Edits one review draft and writes exactly one immutable final rubric artifact."""

    def __init__(
        self,
        store: ProjectStore,
        draft: RubricReviewDraft,
        proposals: tuple[RubricProposal, ...],
        producer_revision: str,
        clock: Callable[[], datetime],
    ) -> None:
        """Construct a review service from validated persisted draft state."""
        self._store = store
        self._draft = draft
        self._proposals = proposals
        self._producer_revision = producer_revision
        self._clock = clock

    @classmethod
    def open(
        cls,
        store: ProjectStore,
        *,
        source_task_set_id: ArtifactId,
        code_revision: str,
        proposals: Sequence[RubricProposal] = (),
        clock: Callable[[], datetime] | None = None,
    ) -> RubricReview:
        """Create or resume the project's rubric-review draft.

        Args:
            store: Project-local storage containing the sole mutable review file.
            source_task_set_id: Task set whose representative evidence seeded this review.
            code_revision: Exact current producer revision for any new artifact.
            proposals: Optional model-proposed cards used when creating a new draft.
            clock: Injectable wall clock for deterministic callers and tests.

        Returns:
            A service that persists every state transition through ``review.json``.

        Raises:
            RubricReviewError: Existing draft state belongs to another task set or proposal set.
        """
        now = _utc_now if clock is None else clock
        supplied_proposals = tuple(proposals)
        selected: list[RubricReviewDraft] = []

        def initialize(current: JsonValue | None) -> JsonObject:
            root = root_review_object(current, error_type=RubricReviewError)
            saved = root.get("rubric_review")
            if saved is None:
                draft = RubricReviewDraft(
                    source_task_set_id=source_task_set_id,
                    code_revision=code_revision,
                    proposals=supplied_proposals,
                )
            else:
                draft = _validated_draft(
                    saved,
                    source_task_set_id=source_task_set_id,
                    proposals=supplied_proposals,
                )
            root["rubric_review"] = draft.model_dump(mode="json")
            selected.append(draft)
            return root

        with coordinate_completed_build_selection(store, task_set_id=source_task_set_id):
            store.update_review(initialize)
        return cls(store, selected[0], supplied_proposals, code_revision, now)

    @property
    def draft(self) -> RubricReviewDraft:
        """Return the current persisted review state."""
        return self._draft

    def accept(self, dimension_id: ArtifactId) -> None:
        """Accept one proposed rubric card in a locked review transaction.

        Args:
            dimension_id: Proposed rubric card to add to the human-owned draft.
        """
        self._mutate(lambda review: review._accept(dimension_id))

    def _accept(self, dimension_id: ArtifactId) -> None:
        """Accept one proposed rubric card into the editable review order.

        Args:
            dimension_id: Proposal card to include in the human-owned rubric draft.

        Raises:
            RubricReviewError: The card is unknown, rejected, already active, or finalized.
        """
        self._ensure_editable()
        candidate = self._candidate(dimension_id)
        active_ids = {dimension.dimension_id for dimension in self._draft.dimensions}
        if dimension_id in active_ids:
            raise RubricReviewError("rubric dimension is already accepted")
        if dimension_id in self._draft.rejected_dimension_ids:
            raise RubricReviewError("rejected rubric dimensions cannot be accepted")
        self._replace(
            dimensions=(*self._draft.dimensions, candidate),
            event_kind="accept",
            event_dimension_ids=(dimension_id,),
        )

    def reject(self, dimension_id: ArtifactId) -> None:
        """Reject one proposed rubric card in a locked review transaction.

        Args:
            dimension_id: Proposed rubric card to remove from the editable draft.
        """
        self._mutate(lambda review: review._reject(dimension_id))

    def _reject(self, dimension_id: ArtifactId) -> None:
        """Reject one proposed card and remove it from the editable rubric if needed.

        Args:
            dimension_id: Proposal card to reject.

        Raises:
            RubricReviewError: The card is not a known proposal or the review is finalized.
        """
        self._ensure_editable()
        self._candidate(dimension_id)
        if dimension_id in self._draft.rejected_dimension_ids:
            raise RubricReviewError("rubric dimension is already rejected")
        dimensions = tuple(
            dimension
            for dimension in self._draft.dimensions
            if dimension.dimension_id != dimension_id
        )
        self._replace(
            dimensions=dimensions,
            rejected_dimension_ids=(*self._draft.rejected_dimension_ids, dimension_id),
            event_kind="reject",
            event_dimension_ids=(dimension_id,),
        )

    def edit(
        self,
        dimension_id: ArtifactId,
        *,
        name: str | None = None,
        description: str | None = None,
        anchors: tuple[ScoreAnchor, ...] | None = None,
    ) -> None:
        """Edit one rubric card in a locked review transaction.

        Args:
            dimension_id: Active or proposed rubric card to change.
            name: Optional replacement display name.
            description: Optional replacement customer-facing definition.
            anchors: Optional complete ordered zero-to-five anchor replacement.
        """
        self._mutate(
            lambda review: review._edit(
                dimension_id,
                name=name,
                description=description,
                anchors=anchors,
            )
        )

    def _edit(
        self,
        dimension_id: ArtifactId,
        *,
        name: str | None = None,
        description: str | None = None,
        anchors: tuple[ScoreAnchor, ...] | None = None,
    ) -> None:
        """Edit one active or proposed rubric card, validating all zero-to-five anchors.

        Args:
            dimension_id: Existing accepted card or a proposal to accept and edit.
            name: Optional replacement display name.
            description: Optional replacement customer-facing definition.
            anchors: Optional complete ordered zero-to-five anchor replacement.

        Raises:
            RubricReviewError: The card is rejected, unknown, unchanged, or finalized.
        """
        self._ensure_editable()
        if dimension_id in self._draft.rejected_dimension_ids:
            raise RubricReviewError("rejected rubric dimensions cannot be edited")
        dimensions = list(self._draft.dimensions)
        position = next(
            (
                index
                for index, dimension in enumerate(dimensions)
                if dimension.dimension_id == dimension_id
            ),
            None,
        )
        if position is None:
            dimensions.append(self._candidate(dimension_id))
            position = len(dimensions) - 1
        current = dimensions[position]
        if name is None and description is None and anchors is None:
            raise RubricReviewError("rubric edits must change a name, description, or anchors")
        dimensions[position] = RubricDimension(
            dimension_id=current.dimension_id,
            name=current.name if name is None else name,
            description=current.description if description is None else description,
            anchors=current.anchors if anchors is None else anchors,
        )
        self._replace(
            dimensions=tuple(dimensions),
            event_kind="edit",
            event_dimension_ids=(dimension_id,),
        )

    def add(self, dimension: RubricDimension) -> None:
        """Add one human-authored scale in a locked review transaction.

        Args:
            dimension: Complete zero-to-five rubric dimension to append.
        """
        self._mutate(lambda review: review._add(dimension))

    def _add(self, dimension: RubricDimension) -> None:
        """Add one human-authored zero-to-five rubric dimension.

        Args:
            dimension: Complete dimension with explicit score anchors.

        Raises:
            RubricReviewError: The ID already exists in the editable review or it is finalized.
        """
        self._ensure_editable()
        if dimension.dimension_id in {item.dimension_id for item in self._draft.dimensions}:
            raise RubricReviewError("rubric dimensions must have unique IDs")
        self._replace(
            dimensions=(*self._draft.dimensions, dimension),
            event_kind="add",
            event_dimension_ids=(dimension.dimension_id,),
        )

    def replace_all(self, dimensions: Sequence[RubricDimension]) -> None:
        """Replace every active scale in a locked review transaction.

        Args:
            dimensions: Complete ordered non-empty set of replacement scales.
        """
        replacement = tuple(dimensions)
        self._mutate(lambda review: review._replace_all(replacement))

    def _replace_all(self, dimensions: Sequence[RubricDimension]) -> None:
        """Replace the editable scale set with a complete human-owned ordering.

        Args:
            dimensions: New ordered non-empty scale set.

        Raises:
            RubricReviewError: The supplied set is empty, has duplicate IDs, or is finalized.
        """
        self._ensure_editable()
        replacement = tuple(dimensions)
        if not replacement:
            raise RubricReviewError("a rubric review needs at least one dimension")
        if len({item.dimension_id for item in replacement}) != len(replacement):
            raise RubricReviewError("replacement rubric dimensions must have unique IDs")
        self._replace(
            dimensions=replacement,
            event_kind="replace_all",
            event_dimension_ids=tuple(item.dimension_id for item in replacement),
        )

    def order(self, dimension_ids: Sequence[ArtifactId]) -> None:
        """Reorder every active scale in a locked review transaction.

        Args:
            dimension_ids: Every active dimension ID once, in the desired output order.
        """
        requested = tuple(dimension_ids)
        self._mutate(lambda review: review._order(requested))

    def _order(self, dimension_ids: Sequence[ArtifactId]) -> None:
        """Set the complete visible order of already accepted rubric dimensions.

        Args:
            dimension_ids: Every active dimension ID once, in the desired output order.

        Raises:
            RubricReviewError: The IDs omit, add, or repeat an active dimension.
        """
        self._ensure_editable()
        requested = tuple(dimension_ids)
        active = {dimension.dimension_id: dimension for dimension in self._draft.dimensions}
        if len(set(requested)) != len(requested) or set(requested) != set(active):
            raise RubricReviewError("rubric order must name every active dimension exactly once")
        self._replace(
            dimensions=tuple(active[dimension_id] for dimension_id in requested),
            event_kind="order",
            event_dimension_ids=requested,
        )

    def finalize(self) -> Rubric:
        """Finalize the current draft inside one locked review transaction.

        Returns:
            The immutable approved rubric produced from the locked review draft.
        """
        finalized: list[Rubric] = []

        def transition(review: RubricReview) -> None:
            finalized.append(review._finalize())

        self._mutate(transition)
        return finalized[0]

    def _finalize(self) -> Rubric:
        """Write one immutable approved rubric artifact and lock the review draft.

        Returns:
            The human-approved immutable rubric artifact contract.

        Raises:
            RubricReviewError: The review has no dimensions or cannot safely finalize.
        """
        if self._draft.status == "finalized":
            if self._draft.finalized_rubric is None:
                raise RubricReviewError("finalized rubric review is missing its artifact")
            return self._draft.finalized_rubric
        if not self._draft.dimensions:
            raise RubricReviewError("cannot finalize a rubric review without dimensions")
        created_at = self._now()
        try:
            _task_set, task_set_input = resolve_artifact(
                self._store,
                artifact_id=self._draft.source_task_set_id,
                expected_artifact_type="task-set",
            )
        except JudgingProvenanceError as exc:
            raise RubricReviewError(
                "cannot finalize a rubric without a completed source task-set artifact"
            ) from exc
        accepted_dimension_ids = {item.dimension_id for item in self._draft.dimensions}
        accepted_proposals = tuple(
            proposal
            for proposal in self._draft.proposals
            if accepted_dimension_ids.intersection(
                item.dimension.dimension_id for item in proposal.dimensions
            )
        )
        try:
            proposal_evidence_ids = tuple(
                sorted(
                    write_rubric_proposal_evidence(
                        self._store,
                        proposal=proposal,
                        source_task_set_input=task_set_input,
                        code_revision=self._producer_revision,
                        created_at=created_at,
                    ).proposal_evidence_id
                    for proposal in accepted_proposals
                )
            )
            proposal_inputs = tuple(
                resolve_artifact(
                    self._store,
                    artifact_id=proposal_evidence_id,
                    expected_artifact_type="rubric-proposal-evidence",
                )[1]
                for proposal_evidence_id in proposal_evidence_ids
            )
        except (JudgingProvenanceError, RubricProposalError) as exc:
            raise RubricReviewError("cannot persist accepted rubric proposal evidence") from exc
        inputs = sorted_verified_inputs((task_set_input, *proposal_inputs))
        rubric_id = stable_id(
            "rubric",
            {
                "source_task_set_id": self._draft.source_task_set_id,
                "dimensions": [item.model_dump(mode="json") for item in self._draft.dimensions],
                "accepted_proposal_evidence_ids": proposal_evidence_ids,
                "inputs": [item.model_dump(mode="json") for item in inputs],
            },
        )
        rubric = Rubric(
            schema_version=1,
            created_at=created_at,
            inputs=inputs,
            code_revision=self._producer_revision,
            rubric_id=rubric_id,
            dimensions=self._draft.dimensions,
            source_task_set_id=self._draft.source_task_set_id,
            accepted_proposal_evidence_ids=proposal_evidence_ids,
            status="human_approved",
            approved_at=created_at,
        )
        try:
            self._store.artifacts.write_json(
                artifact_id=rubric.rubric_id,
                artifact_type="rubric",
                envelope=rubric,
                files={"rubric.json": rubric},
            )
        except ArtifactAlreadyExistsError:
            try:
                stored = Rubric.model_validate_json(
                    self._store.artifacts.read_bytes(rubric.rubric_id, "rubric.json")
                )
            except ValueError as exc:
                raise RubricReviewError(
                    "existing rubric artifact cannot be resumed safely"
                ) from exc
            if not _same_rubric_identity(stored, rubric):
                raise RubricReviewError(
                    "existing rubric artifact conflicts with this review"
                ) from None
            rubric = stored
        event = self._event("finalize", tuple(item.dimension_id for item in rubric.dimensions))
        self._draft = RubricReviewDraft(
            source_task_set_id=self._draft.source_task_set_id,
            code_revision=self._producer_revision,
            proposals=self._draft.proposals,
            dimensions=rubric.dimensions,
            rejected_dimension_ids=self._draft.rejected_dimension_ids,
            events=(*self._draft.events, event),
            status="finalized",
            finalized_rubric=rubric,
        )
        return rubric

    def _candidate(self, dimension_id: ArtifactId) -> RubricDimension:
        """Find one proposed dimension or raise an actionable review error."""
        matches = [
            item.dimension
            for proposal in self._draft.proposals
            for item in proposal.dimensions
            if item.dimension.dimension_id == dimension_id
        ]
        if not matches:
            raise RubricReviewError("rubric proposal dimension does not exist")
        if len(matches) != 1:
            raise RubricReviewError("rubric proposal dimension ID is ambiguous")
        return matches[0]

    def _ensure_editable(self) -> None:
        """Reject review mutations after an immutable artifact has been finalized."""
        if self._draft.status == "finalized":
            raise RubricReviewError("rubric review is finalized and immutable")

    def _replace(
        self,
        *,
        dimensions: tuple[RubricDimension, ...],
        event_kind: Literal["accept", "reject", "edit", "add", "replace_all", "order"],
        event_dimension_ids: tuple[ArtifactId, ...],
        rejected_dimension_ids: tuple[ArtifactId, ...] | None = None,
    ) -> None:
        """Persist one validated draft transition with its append-only review event."""
        event = self._event(event_kind, event_dimension_ids)
        self._draft = RubricReviewDraft(
            source_task_set_id=self._draft.source_task_set_id,
            code_revision=self._producer_revision,
            proposals=self._draft.proposals,
            dimensions=dimensions,
            rejected_dimension_ids=(
                self._draft.rejected_dimension_ids
                if rejected_dimension_ids is None
                else rejected_dimension_ids
            ),
            events=(*self._draft.events, event),
        )

    def _mutate(self, transition: Callable[[RubricReview], None]) -> None:
        """Reload and apply one namespace transition while the review lock is held."""
        selected: list[RubricReviewDraft] = []

        def update(current: JsonValue | None) -> JsonObject:
            root = root_review_object(current, error_type=RubricReviewError)
            saved = root.get("rubric_review")
            if saved is None:
                raise RubricReviewError("rubric review draft disappeared before mutation")
            draft = _validated_draft(
                saved,
                source_task_set_id=self._draft.source_task_set_id,
                proposals=self._proposals,
            )
            working = RubricReview(
                self._store,
                draft,
                self._proposals,
                self._producer_revision,
                self._clock,
            )
            transition(working)
            root["rubric_review"] = working._draft.model_dump(mode="json")
            selected.append(working._draft)
            return root

        with coordinate_completed_build_selection(
            self._store,
            task_set_id=self._draft.source_task_set_id,
        ):
            self._store.update_review(update)
        self._draft = selected[0]

    def _event(
        self,
        kind: Literal["accept", "reject", "edit", "add", "replace_all", "order", "finalize"],
        dimension_ids: tuple[ArtifactId, ...],
    ) -> RubricReviewEvent:
        """Build a deterministic unique event ID for the next persisted transition."""
        created_at = self._now()
        sequence = len(self._draft.events)
        return RubricReviewEvent(
            event_id=stable_id(
                "rubric-review-event",
                {
                    "kind": kind,
                    "dimension_ids": list(dimension_ids),
                    "sequence": sequence,
                    "created_at": created_at.isoformat(),
                },
            ),
            kind=kind,
            dimension_ids=dimension_ids,
            created_at=created_at,
        )

    def _now(self) -> datetime:
        """Return one timezone-aware timestamp or fail before persisting invalid state."""
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RubricReviewError("rubric review clock must return a timezone-aware time")
        return value


def _validated_draft(
    saved: JsonValue,
    *,
    source_task_set_id: ArtifactId,
    proposals: tuple[RubricProposal, ...],
) -> RubricReviewDraft:
    """Validate a persisted rubric namespace against its selected task set and proposals."""
    try:
        draft = RubricReviewDraft.model_validate(saved)
    except ValueError as exc:
        raise RubricReviewError("review.json contains an invalid rubric review draft") from exc
    if draft.source_task_set_id != source_task_set_id:
        raise RubricReviewError("existing rubric review belongs to another task set")
    if proposals and proposals != draft.proposals:
        raise RubricReviewError("existing rubric review has a different proposal set")
    return draft


def _same_rubric_identity(left: Rubric, right: Rubric) -> bool:
    """Compare the stable human-approved rubric content without retry-time provenance."""
    return (
        left.schema_version == right.schema_version
        and left.rubric_id == right.rubric_id
        and left.dimensions == right.dimensions
        and left.source_task_set_id == right.source_task_set_id
        and left.accepted_proposal_evidence_ids == right.accepted_proposal_evidence_ids
        and left.status == right.status == "human_approved"
        and left.code_revision == right.code_revision
        and left.inputs == right.inputs
        and left.source == right.source
    )


def _utc_now() -> datetime:
    """Return the UTC timestamp used when callers do not inject a review clock."""
    return datetime.now(UTC)
