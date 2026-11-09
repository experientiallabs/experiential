"""Retrieval of recorded transitions whose pre-turn environment state matches the live one.

:func:`build_state_tag_index` tags the pre-turn state of every fit step of a family with the
family's :class:`~exp.simulation.retrieval.state_tags.StateTagBank` through the decision model,
one step-level unit per transition. Every step must carry a fit lineage binding: a held-out trace
is refused by name, never tagged, and never stored. An optional
:class:`~exp.simulation.retrieval.scope.CorpusScope` narrows the steps further and is recorded on
the index. With an embedder (normally the local embedding model), each step's rendered state is
also embedded, for tie-breaking only.

:class:`StateRAG` serves one live step. It tags the live pre-turn state with the same bank, scores
every eligible step by weighted agreement over fired tags, and returns the best. A tag fires when
its probability is above the policy threshold. The agreement of two states is a weighted Jaccard
over the tags either fired: each tag weighs the product of both answers' certainty, where an
answer inside the uncertain band counts ``uncertain_weight`` instead of one, so a confident shared
condition outweighs a hesitant one. Steps of the query's own lineage are excluded. Embedding
similarity only orders steps whose agreement exactly equals another's at or above the cut, and is
computed only then; it never lifts a step over one with higher agreement.

Each retrieval follows one declared lifecycle (:data:`RECALL_MACHINE`): pending, tagged, ranked,
optionally tie-broken, then recorded, or failed when tagging fails. Every move is checked against
that table and written through the receipt sink. Every retrieval writes one
:class:`StateRecallReceipt` (request digests, raw answers, the live
probabilities, what was selected and the tie-break similarities used, and the reported cost), so
:func:`replay_state_recall` recomputes the selection from the receipt with no call at all.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json
from exp.common.project import ArtifactStore, artifact_input
from exp.common.project.errors import ArtifactStoreError
from exp.common.state_machine import MachineRun, StateMachine, TransitionRecord
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS, DecisionRequest
from exp.simulation.mining.tag_clusters import read_record, write_record
from exp.simulation.mining.tagging import TraceTaggingError
from exp.simulation.retrieval.contracts import RAGLineageBinding, RAGTransition
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.retrieval.state_tags import (
    STATE_TAG_PROTOCOL,
    PreTurnState,
    StateTagBank,
    response_usd,
    tag_state,
)

STATE_INDEX_TYPE = "state-tag-index"
"""Artifact type of a saved :class:`StateTagIndex` (``index.json``)."""

Embed = Callable[[Sequence[str]], tuple[tuple[float, ...], ...]]
"""Embed texts into unit vectors, in input order."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateEmbedder:
    """An injected embedder used only to break exact agreement ties.

    Attributes:
        model: Identity of the embedding model, recorded on the index and checked at query.
        embed: Embedding call, normally the local embedding model's.
    """

    model: JsonObject
    embed: Embed


class StateTagRow(ContractModel):
    """One tagged fit step: the state before one recorded transition.

    Attributes:
        transition_id: The recorded transition whose action followed this state.
        trace_id: Its source trace.
        lineage_id: Its leakage lineage.
        step_index: Earlier actions in the episode.
        state_sha256: Identity of the rendered pre-turn state.
        probabilities: One answer per bank question, in bank order.
        vector: Unit embedding of the rendered state, when the index has an embedder.
    """

    transition_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    state_sha256: Sha256
    probabilities: tuple[float, ...] = Field(min_length=1)
    vector: tuple[float, ...] | None = Field(default=None, exclude_if=lambda value: value is None)


class StateTagIndex(ContractModel):
    """Every fit step of a family, tagged by its pre-turn state.

    Attributes:
        protocol: State-tag protocol.
        bank: The questions every step was asked.
        decision_model: Identity of the decision model that answered.
        corpus_scope: The scope the steps were restricted to, when one was given.
        embedder: Identity of the tie-break embedder, when vectors are stored.
        rows: Tagged steps sorted by transition.
        skipped_steps: Fit steps left out because their state or answer could not cross the
            credential boundary, so a replay knows the index is partial. Omitted (and the
            identity unchanged) when no step was skipped.
    """

    protocol: str = STATE_TAG_PROTOCOL
    bank: StateTagBank
    decision_model: JsonObject
    corpus_scope: CorpusScope | None = Field(default=None, exclude_if=lambda value: value is None)
    embedder: JsonObject | None = Field(default=None, exclude_if=lambda value: value is None)
    rows: tuple[StateTagRow, ...] = Field(min_length=1)
    skipped_steps: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)

    @model_validator(mode="after")
    def _consistent(self) -> StateTagIndex:
        """Require sorted unique rows answering every question, with vectors iff embedded."""
        ids = [row.transition_id for row in self.rows]
        if len(set(ids)) != len(ids) or ids != sorted(ids):
            raise ValueError("state tag index rows must be sorted and unique by transition")
        width = len(self.bank.questions)
        for row in self.rows:
            if len(row.probabilities) != width:
                raise ValueError(f"state row {row.transition_id} does not answer every question")
            if (row.vector is None) != (self.embedder is None):
                raise ValueError("state rows carry vectors exactly when the index has an embedder")
            scope = self.corpus_scope
            if scope is not None and not scope.admits(
                trace_id=row.trace_id, lineage_id=row.lineage_id
            ):
                raise ValueError(f"state row {row.transition_id} lies outside the corpus scope")
        return self

    @property
    def identity(self) -> Sha256:
        """Return the SHA-256 identity of the canonical index."""
        return sha256_json(self)


def build_state_tag_index(
    bank: StateTagBank,
    steps: Sequence[tuple[RAGTransition, PreTurnState]],
    bindings: Sequence[RAGLineageBinding],
    *,
    decide: Callable[[DecisionRequest], JsonObject],
    decision_model: JsonObject,
    scope: CorpusScope | None = None,
    embedder: StateEmbedder | None = None,
    concurrency: int = 1,
) -> StateTagIndex:
    """Tag the pre-turn state of every fit step through the decision model.

    Args:
        bank: The family's state questions.
        steps: ``(transition, pre-turn state)`` pairs from :func:`pre_turn_steps`.
        bindings: Frozen lineage assignment of every step's trace.
        decide: Admitted decision call, normally cached so a rebuild makes no call.
        decision_model: Identity of the decision model.
        scope: Optional scope; steps outside it are left out.
        embedder: Optional tie-break embedder of rendered states.
        concurrency: Steps tagged at once; the index is the same for every value.

    Returns:
        The tagged index.

    Raises:
        ValueError: A step's trace has no binding, is held out, or names another lineage, or no
            step lies inside the scope.
    """
    by_trace = {binding.trace_id: binding for binding in bindings}
    kept: list[tuple[RAGTransition, PreTurnState]] = []
    for transition, state in steps:
        binding = by_trace.get(transition.trace_id)
        if binding is None:
            raise ValueError(
                f"trace {transition.trace_id} has no lineage binding; bind every step's trace"
            )
        if binding.partition != "fit":
            raise ValueError(
                f"trace {transition.trace_id} is held out; held-out traces are never tagged or "
                "retrieved for state retrieval, so pass only fit steps"
            )
        if binding.lineage_id != transition.lineage_id:
            raise ValueError(f"transition {transition.transition_id} names another lineage")
        if scope is not None and not scope.admits(
            trace_id=transition.trace_id, lineage_id=transition.lineage_id
        ):
            continue
        kept.append((transition, state))
    if not kept:
        raise ValueError("no fit step lies inside the corpus scope; widen the scope")
    vectors: tuple[tuple[float, ...], ...] | None = None
    if embedder is not None:
        vectors = embedder.embed([state.text() for _, state in kept])

    def answer(pair: tuple[RAGTransition, PreTurnState]) -> tuple[float, ...] | None:
        """Tag one step, or None when its state cannot cross the credential boundary."""
        try:
            return tag_state(bank, pair[1], decide)[0]
        except ArtifactStoreError as error:
            if not _at_credential_boundary(error):
                raise
            return None
        except TraceTaggingError as error:
            if "outcome is unknown" not in str(error):
                raise
            return None  # an earlier run's refused or interrupted decision: never re-bought here

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        answered = list(pool.map(answer, kept))
    refused = sum(1 for row in answered if row is None)
    if refused:
        logger.warning(
            "%d of %d fit steps were left out of the state-tag index: their state or its answer "
            "carries credential-like content the credential boundary refuses",
            refused,
            len(kept),
        )
    pairs = [(pair, row) for pair, row in zip(kept, answered, strict=True) if row is not None]
    if vectors is not None:
        vectors = tuple(v for v, row in zip(vectors, answered, strict=True) if row is not None)
    kept = [pair for pair, _ in pairs]
    answers = [row for _, row in pairs]
    rows = [
        StateTagRow(
            transition_id=transition.transition_id,
            trace_id=transition.trace_id,
            lineage_id=transition.lineage_id,
            step_index=state.step_index,
            state_sha256=state.identity,
            probabilities=probabilities,
            vector=None if vectors is None else tuple(vectors[index]),
        )
        for index, ((transition, state), probabilities) in enumerate(
            zip(kept, answers, strict=True)
        )
    ]
    return StateTagIndex(
        bank=bank,
        decision_model=decision_model,
        corpus_scope=scope,
        embedder=None if embedder is None else embedder.model,
        rows=tuple(sorted(rows, key=lambda row: row.transition_id)),
        skipped_steps=refused,
    )


def save_state_tag_index(
    store: ArtifactStore,
    index: StateTagIndex,
    *,
    code_revision: str,
    inputs: Sequence[ArtifactInput] = (),
) -> ArtifactInput:
    """Save an index as a content-addressed artifact, reusing an identical saved one."""
    identity = "state-tag-index-" + index.identity[:32]
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    return write_record(
        store, identity, STATE_INDEX_TYPE, "index.json", index, code_revision, inputs
    )


def load_state_tag_index(store: ArtifactStore, reference: ArtifactInput) -> StateTagIndex:
    """Read a saved index, refusing a changed or mistyped artifact."""
    return StateTagIndex.model_validate_json(
        read_record(store, reference, STATE_INDEX_TYPE, "index.json")
    )


class StateRetrievalPolicy(ContractModel):
    """Frozen scoring of pre-turn state agreement.

    Attributes:
        threshold: Probability strictly above which a tag fires.
        certain_below: Answers at or below this are a confident no.
        certain_above: Answers at or above this are a confident yes.
        uncertain_weight: Certainty of an answer between the two, from 0 to 1.
        top_k: Most transitions retrieved per step.
        minimum_agreement: Agreement a step must exceed to be retrieved.
        history_steps: Earlier steps shown in a live pre-turn state.
        observation_characters: Characters kept of each earlier observation.
    """

    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    certain_below: float = Field(default=0.2, ge=0.0, lt=1.0)
    certain_above: float = Field(default=0.8, gt=0.0, le=1.0)
    uncertain_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    top_k: int = Field(default=4, ge=1, le=32, strict=True)
    minimum_agreement: float = Field(default=0.0, ge=0.0, lt=1.0)
    history_steps: int = Field(default=4, ge=0, le=32, strict=True)
    observation_characters: int = Field(default=2_000, ge=100, le=50_000, strict=True)

    @model_validator(mode="after")
    def _ordered(self) -> StateRetrievalPolicy:
        """Require the uncertain band to straddle the threshold."""
        if not self.certain_below < self.threshold < self.certain_above:
            raise ValueError("state retrieval needs certain_below < threshold < certain_above")
        return self

    @property
    def identity(self) -> Sha256:
        """Bind the protocol and every scoring parameter."""
        return sha256_json({"protocol": STATE_TAG_PROTOCOL, "policy": self.model_dump(mode="json")})

    def certainty(self, probability: float) -> float:
        """Return one answer's weight: one when confident, ``uncertain_weight`` otherwise."""
        if probability <= self.certain_below or probability >= self.certain_above:
            return 1.0
        return self.uncertain_weight


def state_agreement(
    query: Sequence[float], candidate: Sequence[float], policy: StateRetrievalPolicy
) -> float:
    """Return the certainty-weighted Jaccard agreement of two states' fired tags.

    Args:
        query: Live answers, in bank order.
        candidate: A recorded step's answers, in bank order.
        policy: Scoring parameters.

    Returns:
        Weight of tags both fired over weight of tags either fired; 0.0 when neither fired any.
    """
    shared = either = 0.0
    for left, right in zip(query, candidate, strict=True):
        fired_left, fired_right = left > policy.threshold, right > policy.threshold
        if not (fired_left or fired_right):
            continue
        weight = policy.certainty(left) * policy.certainty(right)
        either += weight
        if fired_left and fired_right:
            shared += weight
    return shared / either if either else 0.0


class RecallState(StrEnum):
    """Lifecycle of one live step's state retrieval."""

    PENDING = "pending"
    TAGGED = "tagged"
    RANKED = "ranked"
    TIE_BROKEN = "tie_broken"
    RECORDED = "recorded"
    FAILED = "failed"


RECALL_MACHINE = StateMachine(
    name="pre-turn-state-recall",
    states=RecallState,
    initial=frozenset({RecallState.PENDING}),
    transitions=(
        (RecallState.PENDING, "tagged", RecallState.TAGGED),
        (RecallState.PENDING, "tag_failed", RecallState.FAILED),
        (RecallState.TAGGED, "ranked", RecallState.RANKED),
        (RecallState.RANKED, "tie_broken", RecallState.TIE_BROKEN),
        (RecallState.RANKED, "recorded", RecallState.RECORDED),
        (RecallState.TIE_BROKEN, "recorded", RecallState.RECORDED),
    ),
)
"""The one transition table of a step's retrieval: tagged, ranked, maybe tie-broken, recorded."""


class StateMatch(ContractModel):
    """One retrieved step and why it ranked where it did.

    Attributes:
        transition_id: The recorded transition.
        trace_id: Its source trace.
        step_index: Earlier actions in its episode.
        agreement: Weighted agreement of its pre-turn state with the live one.
        similarity: Embedding similarity, recorded only when it broke a tie.
    """

    transition_id: str
    trace_id: str
    step_index: int
    agreement: float = Field(ge=0.0, le=1.0)
    similarity: float | None = Field(default=None, exclude_if=lambda value: value is None)


class StateRecallReceipt(ContractModel):
    """What one live step's state retrieval read, decided and cost.

    Attributes:
        protocol: State-tag protocol.
        index_sha256: Identity of the index retrieved from.
        policy_sha256: Identity of the scoring policy.
        state_sha256: Identity of the live pre-turn state.
        excluded_lineage_ids: Lineages left out, the query's own included.
        request_sha256: Digest of each decision request.
        responses: Raw decision responses.
        probabilities: Live answers, in bank order.
        fired: Bank questions the live state fired.
        eligible: Steps scored after exclusions.
        similarities: Embedding similarity of each step that was tied at the cut.
        selected: Retrieved steps in rank order.
        usd: Reported cost of the decisions, when every response carried one.
        transitions: The step's lifecycle moves, ending in ``recorded``.
    """

    protocol: str = STATE_TAG_PROTOCOL
    index_sha256: Sha256
    policy_sha256: Sha256
    state_sha256: Sha256
    excluded_lineage_ids: tuple[str, ...] = ()
    request_sha256: tuple[Sha256, ...] = ()
    responses: tuple[JsonObject, ...] = ()
    probabilities: tuple[float, ...] = ()
    fired: tuple[str, ...] = ()
    eligible: int = Field(default=0, ge=0)
    similarities: dict[str, float] = Field(default_factory=dict)
    selected: tuple[StateMatch, ...] = ()
    usd: float | None = None
    transitions: tuple[TransitionRecord, ...] = ()


def rank_states(
    index: StateTagIndex,
    policy: StateRetrievalPolicy,
    probabilities: Sequence[float],
    excluded_lineage_ids: Sequence[str],
    similarities: Mapping[str, float] | None = None,
) -> tuple[tuple[StateMatch, ...], tuple[str, ...], int]:
    """Rank every eligible step; a pure function of recorded answers.

    Args:
        index: Tagged fit steps.
        policy: Scoring parameters.
        probabilities: Live answers, in bank order.
        excluded_lineage_ids: Lineages never retrieved.
        similarities: Embedding similarity of tied steps, when the tie was broken.

    Returns:
        The selected matches, the steps sharing an agreement with another step at or above the
        cut (which an embedder orders), and the number of eligible steps.
    """
    excluded = frozenset(excluded_lineage_ids)
    scored = [
        (row, state_agreement(probabilities, row.probabilities, policy))
        for row in index.rows
        if row.lineage_id not in excluded
    ]
    passing = [(row, value) for row, value in scored if value > policy.minimum_agreement]
    passing.sort(key=lambda pair: (-pair[1], pair[0].transition_id))
    tied: tuple[str, ...] = ()
    if passing:
        cut = passing[: policy.top_k][-1][1]
        counts = Counter(value for _, value in passing if value >= cut)
        tied = tuple(row.transition_id for row, value in passing if counts.get(value, 0) > 1)
    known = similarities or {}
    passing.sort(
        key=lambda pair: (
            -pair[1],
            -known.get(pair[0].transition_id, 0.0),
            pair[0].transition_id,
        )
    )
    matches = tuple(
        StateMatch(
            transition_id=row.transition_id,
            trace_id=row.trace_id,
            step_index=row.step_index,
            agreement=value,
            similarity=known.get(row.transition_id),
        )
        for row, value in passing[: policy.top_k]
    )
    return matches, tied, len(scored)


class StateRAG:
    """Retrieve recorded transitions by the agreement of their pre-turn state with the live one."""

    def __init__(
        self,
        index: StateTagIndex,
        policy: StateRetrievalPolicy,
        transitions: Mapping[str, RAGTransition],
        *,
        decide: Callable[[DecisionRequest], JsonObject],
        record: Callable[[JsonObject], None],
        embedder: StateEmbedder | None = None,
    ) -> None:
        """Bind a tagged index, its transitions, the policy and an admitted decision boundary.

        Args:
            index: Tagged fit steps.
            policy: Frozen scoring parameters.
            transitions: Every indexed transition by identity, normally the fit RAG index's.
            decide: Native decision call with the caller's cache, accounting and deadline.
            record: Durable receipt sink; it returns only after recording.
            embedder: Tie-break embedder; must match the index's when the index stores vectors.

        Raises:
            ValueError: A row's transition is missing or disagrees, or the embedder differs.
        """
        for row in index.rows:
            transition = transitions.get(row.transition_id)
            if transition is None or (transition.trace_id, transition.lineage_id) != (
                row.trace_id,
                row.lineage_id,
            ):
                raise ValueError(
                    f"state row {row.transition_id} has no matching transition; bind the index "
                    "the state tags were built from"
                )
        if embedder is not None and embedder.model != index.embedder:
            raise ValueError("state retrieval embedder differs from the index's embedder")
        self.index, self.policy = index, policy
        self._transitions = {
            row.transition_id: transitions[row.transition_id] for row in index.rows
        }
        self._vectors = {row.transition_id: row.vector for row in index.rows}
        self._decide, self._record, self._embedder = decide, record, embedder
        self._index_identity = index.identity

    @property
    def identity(self) -> Sha256:
        """Bind the index, the scoring policy and the recall lifecycle table."""
        return sha256_json(
            {
                "index": self._index_identity,
                "policy": self.policy.identity,
                "machine": RECALL_MACHINE.identity,
            }
        )

    @property
    def decisions_per_step(self) -> int:
        """Return the decision calls one live step takes before caching."""
        return self.index.bank.decisions_per_state

    def retrieve(
        self, state: PreTurnState, excluded_lineage_ids: Sequence[str] = ()
    ) -> tuple[tuple[RAGTransition, ...], StateRecallReceipt]:
        """Tag the live pre-turn state, retrieve the best-matching steps and record the receipt.

        Args:
            state: The live state before the assistant's turn.
            excluded_lineage_ids: Lineages never retrieved, the query's own included.

        Returns:
            The retrieved transitions in rank order, and the receipt.
        """
        run = MachineRun(RECALL_MACHINE, "step/" + state.identity, self._record)
        try:
            probabilities, digests, responses = tag_state(self.index.bank, state, self._decide)
        except Exception:
            run.fire("tag_failed")
            raise
        run.fire("tagged")
        excluded = tuple(sorted(set(excluded_lineage_ids)))
        matches, tied, eligible = rank_states(self.index, self.policy, probabilities, excluded)
        run.fire("ranked")
        similarities: dict[str, float] = {}
        if tied and self._embedder is not None:
            (live,) = self._embedder.embed([state.text()])
            for transition_id in tied:
                stored = self._vectors[transition_id]
                assert stored is not None  # vectors exist whenever the embedder matched
                similarities[transition_id] = _cosine(live, stored)
            matches, _, _ = rank_states(
                self.index, self.policy, probabilities, excluded, similarities
            )
            run.fire("tie_broken")
        run.fire("recorded")
        questions = self.index.bank.questions
        receipt = StateRecallReceipt(
            index_sha256=self._index_identity,
            policy_sha256=self.policy.identity,
            state_sha256=state.identity,
            excluded_lineage_ids=excluded,
            request_sha256=digests,
            responses=responses,
            probabilities=probabilities,
            fired=tuple(
                question.question_id
                for question, value in zip(questions, probabilities, strict=True)
                if value > self.policy.threshold
            ),
            eligible=eligible,
            similarities=similarities,
            selected=matches,
            usd=response_usd(responses),
            transitions=tuple(run.history),
        )
        self._record(receipt.model_dump(mode="json"))
        return tuple(self._transitions[match.transition_id] for match in matches), receipt


def replay_state_recall(
    index: StateTagIndex, policy: StateRetrievalPolicy, receipt: StateRecallReceipt
) -> tuple[StateMatch, ...]:
    """Recompute a recorded selection from its receipt, without any call.

    Args:
        index: The index the receipt names.
        policy: The policy the receipt names.
        receipt: A recorded retrieval.

    Returns:
        The selection, equal to the recorded one.

    Raises:
        ValueError: The index or policy differ from the receipt's, or the recomputed selection
            differs from the recorded one.
    """
    if (index.identity, policy.identity) != (receipt.index_sha256, receipt.policy_sha256):
        raise ValueError("state recall receipt names another index or policy")
    matches, _, _ = rank_states(
        index, policy, receipt.probabilities, receipt.excluded_lineage_ids, receipt.similarities
    )
    if matches != receipt.selected:
        raise ValueError("replayed state recall differs from the recorded selection")
    return matches


def recalled_transition(transition: RAGTransition, match: StateMatch) -> JsonObject:
    """The content injected for one retrieved step: its action, result and agreement."""
    return {
        "transition_id": transition.transition_id,
        "step_index": match.step_index,
        "agreement": round(match.agreement, 4),
        "action": transition.action.model_dump(mode="json"),
        "observation": transition.observation.model_dump(mode="json"),
    }


class StateRAGCalls(ContractModel):
    """Planned calls of one family's state retrieval, before caching.

    Attributes:
        authoring_completions: Completions authoring the bank.
        corpus_decisions: Decisions tagging every fit step.
        live_decisions: Decisions tagging the planned live steps.
    """

    authoring_completions: int = Field(ge=0)
    corpus_decisions: int = Field(ge=0)
    live_decisions: int = Field(ge=0)

    @property
    def decisions(self) -> int:
        """Return every planned decision."""
        return self.corpus_decisions + self.live_decisions


def plan_state_rag_calls(
    *, authoring_calls: int, questions: int, corpus_steps: int, live_steps: int
) -> StateRAGCalls:
    """Count the calls a family's state retrieval can make, for pricing before consent.

    Price ``authoring_completions`` with the authoring alias's completion tariff and
    ``decisions`` with the decision alias's tariff. Cached decisions replay free, so the count
    is an upper bound.

    Args:
        authoring_calls: Completions per bank (drafts plus reconcile), or 0 when it is saved.
        questions: Questions in the bank.
        corpus_steps: Fit steps to tag.
        live_steps: Live steps the simulator may serve.
    """
    per_state = math.ceil(questions / MAX_DECISION_QUESTIONS)
    return StateRAGCalls(
        authoring_completions=authoring_calls,
        corpus_decisions=corpus_steps * per_state,
        live_decisions=live_steps * per_state,
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _at_credential_boundary(error: ArtifactStoreError) -> bool:
    """Return whether a store error is the credential boundary refusing one request or answer."""
    text = str(error)
    return "credential boundary" in text or "secret boundary" in text
