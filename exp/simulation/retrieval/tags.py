"""Decision tags recorded on a retrieval index and the policy that lets them shape ranking.

Tags already decide which traces enter a corpus (a tag selection becomes a
:class:`~exp.simulation.retrieval.scope.CorpusScope`). This module is their second use: they shape
which of the admitted transitions a query retrieves.

An index may carry an :class:`RAGIndexTags` record: for each tagged view, the question bank
identity and revision, every question with its family, the probability threshold the propositions
were fired at, the digest of the tag matrix they were read from, and each indexed trace's fired
propositions. The record is additive. An index without one serializes and is identified exactly as
before.

A query brings its own :class:`TaskQueryTags`, computed for the current task with the same bank and
threshold. Only the ``task`` view may be used for a query: it is what was known before the agent
acted (the router-safe view), while the ``observations`` view describes what the episode revealed
and would leak the outcome of the task being predicted. Observations-view tags may annotate
indexed traces, but a query never carries them and ranking never reads them.

A :class:`TagRetrievalPolicy` declares how tags combine with similarity. The score of a candidate
is its cosine similarity plus ``weight`` times the Jaccard overlap between the query task's fired
tags and the candidate trace's fired task tags. In ``require_shared_family`` mode a candidate is
also dropped unless it shares at least one fired tag from one of the named families. The policy is
frozen and hashed, and :class:`TagRetrievalEvidence` records it beside the query tags so every
retrieval that used tags names exactly how.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.simulation.mining.views import ROUTER_SAFE_VIEWS

TagView = Literal["task", "observations"]
"""Rendered views a recorded tag may describe; matches the mining view names."""

TagRetrievalMode = Literal["boost", "require_shared_family"]
"""Whether tags only re-rank candidates, or also filter out candidates sharing no family tag."""


class RAGTagQuestion(ContractModel):
    """One recorded question column and the family it belongs to.

    Attributes:
        question_id: Stable proposition identity from the question bank.
        family: The bank family of the question, used by family filters.
    """

    question_id: str = Field(min_length=1, max_length=256)
    family: str = Field(min_length=1, max_length=64)


class RAGTraceTags(ContractModel):
    """The propositions that fired for one indexed trace.

    Attributes:
        trace_id: Canonical source trace identity.
        fired: Sorted, unique question identities whose probability was strictly above the
            record's threshold. Empty when nothing fired.
    """

    trace_id: str = Field(min_length=1, max_length=512)
    fired: tuple[str, ...] = ()

    @field_validator("fired")
    @classmethod
    def _sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a canonical fired set.

        Args:
            value: Candidate fired question identities.

        Returns:
            The validated tuple.

        Raises:
            ValueError: Identities repeat or are unsorted.
        """
        if len(set(value)) != len(value) or value != tuple(sorted(value)):
            raise ValueError("fired tags must be sorted and unique")
        return value


class RAGTagView(ContractModel):
    """One view's fired propositions for every tagged trace of an index.

    Attributes:
        bank_id: Identity of the question bank that was answered.
        bank_version: Revision of that bank.
        view: Rendered view the matrix was tagged over.
        threshold: Probability strictly above which a proposition fired, inside (0, 1).
        questions: Recorded columns with their families, in matrix column order.
        matrix_sha256: SHA-256 of the canonical tag matrix the propositions were read from.
        traces: One entry per tagged trace, sorted by trace identity.
    """

    bank_id: str = Field(min_length=1, max_length=128)
    bank_version: int = Field(ge=1)
    view: TagView
    threshold: float = Field(gt=0.0, lt=1.0)
    questions: tuple[RAGTagQuestion, ...] = Field(min_length=1)
    matrix_sha256: Sha256
    traces: tuple[RAGTraceTags, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> RAGTagView:
        """Require unique columns, sorted unique traces, and fired tags inside the columns.

        Returns:
            The validated view record.

        Raises:
            ValueError: A column or trace repeats, traces are unsorted, or a trace fired a
                proposition the record does not declare.
        """
        ids = [question.question_id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.view} tag record repeats a question id")
        traces = [item.trace_id for item in self.traces]
        if len(set(traces)) != len(traces) or traces != sorted(traces):
            raise ValueError(f"{self.view} tag record traces must be sorted and unique")
        known = set(ids)
        for item in self.traces:
            unknown = sorted(set(item.fired) - known)
            if unknown:
                raise ValueError(
                    f"{self.view} tags of trace {item.trace_id} name undeclared questions {unknown}"
                )
        return self

    def fired_by_trace(self) -> dict[str, frozenset[str]]:
        """Return each recorded trace's fired propositions."""
        return {item.trace_id: frozenset(item.fired) for item in self.traces}

    def families(self) -> dict[str, str]:
        """Return each recorded question's family."""
        return {question.question_id: question.family for question in self.questions}

    def restricted_to(self, trace_ids: Iterable[str]) -> RAGTagView:
        """Return the record limited to the given traces, refusing any it does not cover.

        Args:
            trace_ids: Traces the restricted record must describe exactly.

        Returns:
            A record over exactly those traces.

        Raises:
            ValueError: A requested trace has no recorded tags in this view.
        """
        wanted = set(trace_ids)
        recorded = {item.trace_id for item in self.traces}
        missing = sorted(wanted - recorded)
        if missing:
            raise ValueError(
                f"{self.view} tag record has no row for {len(missing)} indexed traces, for "
                f"example {missing[0]}; tag every trace the index holds"
            )
        return self.model_copy(
            update={"traces": tuple(item for item in self.traces if item.trace_id in wanted)}
        )


class RAGIndexTags(ContractModel):
    """Optional tag record of one retrieval index, at most one entry per view.

    Attributes:
        views: Tagged view records sorted by view name. Only the ``task`` view may inform
            query-time ranking; an ``observations`` entry only annotates indexed traces.
    """

    views: tuple[RAGTagView, ...] = Field(min_length=1)

    @field_validator("views")
    @classmethod
    def _one_per_view(cls, value: tuple[RAGTagView, ...]) -> tuple[RAGTagView, ...]:
        """Require at most one record per view, in view order.

        Args:
            value: Candidate view records.

        Returns:
            The validated tuple.

        Raises:
            ValueError: A view repeats or the records are unsorted.
        """
        names = [item.view for item in value]
        if len(set(names)) != len(names) or names != sorted(names):
            raise ValueError("an index tag record holds at most one sorted entry per view")
        return value

    def view(self, name: TagView) -> RAGTagView | None:
        """Return the record for one view, or None when that view is untagged."""
        return next((item for item in self.views if item.view == name), None)

    def traced(self) -> frozenset[str]:
        """Return every trace any view describes."""
        return frozenset(item.trace_id for view in self.views for item in view.traces)

    def restricted_to(self, trace_ids: Iterable[str]) -> RAGIndexTags:
        """Return every view limited to exactly the given traces.

        Args:
            trace_ids: Traces the index holds.

        Returns:
            The restricted record.

        Raises:
            ValueError: Some view lacks a row for one of the traces.
        """
        wanted = frozenset(trace_ids)
        return RAGIndexTags(views=tuple(view.restricted_to(wanted) for view in self.views))


class TaskQueryTags(ContractModel):
    """The current task's fired task-view propositions, supplied by the caller for one query.

    Attributes:
        bank_id: Identity of the bank that was answered for the task.
        bank_version: Revision of that bank.
        view: Must be ``task``. Any other view describes the episode's outcome and is refused.
        threshold: Probability strictly above which a proposition fired.
        matrix_sha256: SHA-256 of the tag matrix the task's propositions were read from.
        fired: Sorted, unique fired question identities. Empty when nothing fired.
    """

    bank_id: str = Field(min_length=1, max_length=128)
    bank_version: int = Field(ge=1)
    view: str = Field(min_length=1)
    threshold: float = Field(gt=0.0, lt=1.0)
    matrix_sha256: Sha256
    fired: tuple[str, ...] = ()

    @field_validator("view")
    @classmethod
    def _router_safe_view(cls, value: str) -> str:
        """Refuse every view a live request could not have seen.

        Args:
            value: Candidate view name.

        Returns:
            The validated view name.

        Raises:
            ValueError: The view is not request-visible.
        """
        if value not in ROUTER_SAFE_VIEWS:
            raise ValueError(
                f"query tags must come from the task view, not {value!r}: the {value} view "
                "describes what the episode revealed and would leak the outcome being "
                "predicted; tag the current task over the task view"
            )
        return value

    @field_validator("fired")
    @classmethod
    def _sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a canonical fired set.

        Args:
            value: Candidate fired question identities.

        Returns:
            The validated tuple.

        Raises:
            ValueError: Identities repeat or are unsorted.
        """
        if len(set(value)) != len(value) or value != tuple(sorted(value)):
            raise ValueError("query fired tags must be sorted and unique")
        return value


class TagRetrievalPolicy(ContractModel):
    """Declared, frozen combination of similarity and task-tag overlap.

    Attributes:
        bank_id: Bank whose task-view tags the policy reads.
        bank_version: Revision of that bank.
        threshold: Fire threshold both the index and the query tags must have used.
        weight: Non-negative finite weight of the Jaccard overlap added to cosine similarity.
        mode: ``boost`` re-ranks only; ``require_shared_family`` also drops candidates sharing
            no fired tag from ``families`` with the query task.
        families: Sorted unique families of the hard filter. Required in
            ``require_shared_family`` mode and refused in ``boost`` mode.
    """

    bank_id: str = Field(min_length=1, max_length=128)
    bank_version: int = Field(ge=1)
    threshold: float = Field(gt=0.0, lt=1.0)
    weight: float = Field(ge=0.0)
    mode: TagRetrievalMode = "boost"
    families: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> TagRetrievalPolicy:
        """Require a finite weight and families exactly when the filter mode needs them.

        Returns:
            The validated policy.

        Raises:
            ValueError: The weight is not finite, or families contradict the mode.
        """
        if not math.isfinite(self.weight):
            raise ValueError("tag retrieval weight must be finite")
        families = self.families
        if len(set(families)) != len(families) or families != tuple(sorted(families)):
            raise ValueError("tag retrieval families must be sorted and unique")
        if self.mode == "require_shared_family" and not families:
            raise ValueError("require_shared_family mode needs at least one named family")
        if self.mode == "boost" and families:
            raise ValueError("boost mode reads every task tag; name families only for a filter")
        return self

    @property
    def identity(self) -> Sha256:
        """Return the SHA-256 identity of the canonical policy."""
        return sha256_json(self)


class TagRetrievalEvidence(ContractModel):
    """What tag-aware ranking read for one retrieval, recorded beside its matches.

    Attributes:
        policy: The exact declared policy.
        policy_sha256: Identity of ``policy``.
        query: The current task's task-view tags.
        index_matrix_sha256: Digest of the tag matrix behind the index's task-view record.
    """

    policy: TagRetrievalPolicy
    policy_sha256: Sha256
    query: TaskQueryTags
    index_matrix_sha256: Sha256

    @model_validator(mode="after")
    def _bound(self) -> TagRetrievalEvidence:
        """Require the recorded policy identity to match the policy."""
        if self.policy.identity != self.policy_sha256:
            raise ValueError("tag retrieval evidence policy digest differs from its policy")
        return self


class TagRanking:
    """Validated pairing of a policy, query tags and an index's task-view record."""

    def __init__(
        self, policy: TagRetrievalPolicy, query: TaskQueryTags, index_tags: RAGIndexTags | None
    ) -> None:
        """Check that the policy, query and index tags were all computed alike.

        Args:
            policy: Declared ranking policy.
            query: The current task's task-view tags.
            index_tags: The index's recorded tags.

        Raises:
            ValueError: The index records no task-view tags, or the bank, revision or threshold
                differ between policy, query and index, or the query or policy names a question
                or family the index does not record.
        """
        record = require_policy_index(policy, index_tags)
        if (query.bank_id, query.bank_version, query.threshold) != (
            policy.bank_id,
            policy.bank_version,
            policy.threshold,
        ):
            raise ValueError(
                f"query tags answer {query.bank_id} v{query.bank_version} at threshold "
                f"{query.threshold}, but the policy reads {policy.bank_id} v{policy.bank_version} "
                f"at {policy.threshold}; tag the task with the policy's bank and threshold"
            )
        families = record.families()
        unknown = sorted(set(query.fired) - families.keys())
        if unknown:
            raise ValueError(
                f"query tags name questions {unknown} the index's task-view record does not "
                "declare; tag the task with the same question columns"
            )
        self.policy = policy
        self.query = query
        self._fired = record.fired_by_trace()
        self._query_fired = frozenset(query.fired)
        self._filter_tags = frozenset(
            question
            for question in self._query_fired
            if families[question] in frozenset(policy.families)
        )
        self.evidence = TagRetrievalEvidence(
            policy=policy,
            policy_sha256=policy.identity,
            query=query,
            index_matrix_sha256=record.matrix_sha256,
        )

    def admits(self, trace_id: str) -> bool:
        """Report whether a candidate trace passes the policy's hard filter.

        Args:
            trace_id: Source trace of the candidate transition.

        Returns:
            True in ``boost`` mode; otherwise whether the trace shares a filter-family tag.
        """
        if self.policy.mode == "boost":
            return True
        return bool(self._filter_tags & self._fired[trace_id])

    def overlap(self, trace_id: str) -> float:
        """Return the Jaccard overlap of the query's and one trace's fired task tags.

        Args:
            trace_id: Source trace of the candidate transition.

        Returns:
            ``|A & B| / |A | B|``, or 0.0 when neither fired anything.
        """
        candidate = self._fired[trace_id]
        union = self._query_fired | candidate
        if not union:
            return 0.0
        return len(self._query_fired & candidate) / len(union)


def require_policy_index(policy: TagRetrievalPolicy, index_tags: RAGIndexTags | None) -> RAGTagView:
    """Return the index's task-view record, refusing one the policy cannot read.

    Args:
        policy: Declared ranking policy.
        index_tags: The index's recorded tags, if any.

    Returns:
        The task-view record the policy ranks with.

    Raises:
        ValueError: The index has no task-view tags, or its bank, revision, threshold or
            families differ from the policy.
    """
    record = None if index_tags is None else index_tags.view("task")
    if record is None:
        raise ValueError(
            "tag-aware retrieval needs an index built with task-view tags; rebuild the index "
            "with trace_tags or retrieve without a tag policy"
        )
    if (record.bank_id, record.bank_version, record.threshold) != (
        policy.bank_id,
        policy.bank_version,
        policy.threshold,
    ):
        raise ValueError(
            f"index task tags answer {record.bank_id} v{record.bank_version} at threshold "
            f"{record.threshold}, but the policy reads {policy.bank_id} v{policy.bank_version} "
            f"at {policy.threshold}"
        )
    missing = sorted(set(policy.families) - set(record.families().values()))
    if missing:
        raise ValueError(
            f"tag retrieval families {missing} have no question in the index's task-view record"
        )
    return record


def verify_index_tags(index_tags: RAGIndexTags, trace_ids: Iterable[str]) -> None:
    """Require every view to describe exactly the traces an index holds.

    Args:
        index_tags: Recorded tags of the index.
        trace_ids: Source traces of the index's transitions.

    Raises:
        ValueError: A view misses an indexed trace or describes a trace the index lacks.
    """
    wanted = frozenset(trace_ids)
    for view in index_tags.views:
        recorded = frozenset(item.trace_id for item in view.traces)
        if recorded != wanted:
            raise ValueError(
                f"{view.view} tag record covers {len(recorded)} traces but the index holds "
                f"{len(wanted)}; it must describe exactly the indexed traces"
            )


class TagRetrievalBinding(ContractModel):
    """A grounded session's tag policy and the query tags of the tasks it may serve.

    Attributes:
        policy: The declared ranking policy.
        tasks: Task-view query tags keyed by task identity. A task without an entry retrieves
            by cosine similarity alone, exactly as without a binding.
    """

    policy: TagRetrievalPolicy
    tasks: dict[str, TaskQueryTags] = Field(default_factory=dict)

    def for_task(self, task_id: str) -> TaskQueryTags | None:
        """Return one task's query tags, or None when the caller supplied none."""
        return self.tasks.get(task_id)

    def identity_for(self, task_id: str) -> Sha256:
        """Return the identity of the policy and one task's query tags.

        Args:
            task_id: Identity of the task a session serves.

        Returns:
            SHA-256 over the policy identity and that task's tags (or None), so a session's
            identity binds exactly the ranking it uses and nothing about other tasks.
        """
        tags = self.for_task(task_id)
        return sha256_json(
            {
                "policy": self.policy.identity,
                "query": None if tags is None else tags.model_dump(mode="json"),
            }
        )
