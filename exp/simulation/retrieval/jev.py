"""Query-dependent JEV evidence ranking through the maintained decision protocol."""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.common.models import ModelMessage
from exp.runtime.gateway.decisions_contracts import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)
from exp.simulation.retrieval.contracts import RAGMatch, RAGQuery
from exp.simulation.retrieval.selection import HashingOriginals, SelectionQuery

_DEPENDENCIES = ("entity-state", "semantic lookup", "tool behavior", "unclear")
_FIELD = (
    "Does this record explain a field needed for this action? "
    "Rate actual relevance, not whether the tool name merely matches. "
    "A superseded field is 0. A related tool without the needed field is 1. "
    "An observation directly establishing the required field or behavior is 2. "
)
_HISTORY = (
    "Is it connected to this action's history? "
    "Use explicit episode/entity/update links, not similarity alone. "
    "An empty current_episode_history gives no evidence of an episode link: 0. "
    "A different episode using the same tool is 0, not 2. "
    "An earlier observed update/read of this same entity in this episode is 2. "
)


class JEVPolicy(ContractModel):
    """Frozen ranking limits and prompt guidance for an admitted evidence pool.

    Attributes:
        version: Ranking protocol and scoring semantics.
        candidate_limit: Maximum candidates; two questions each plus one category fit 32.
        guidance: Tunable clarification under the fixed three-question protocol.
        field_weight: Contribution of the normalized field-relevance answer.
    """

    version: Literal["jev-cues-v1"] = "jev-cues-v1"
    candidate_limit: int = Field(default=10, ge=1, le=15, strict=True)
    guidance: str = ""
    field_weight: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)

    @property
    def identity(self) -> Sha256:
        """Bind scoring parameters and every fixed and tunable instruction."""
        return sha256_json(
            {"config": self.model_dump(mode="json"), "field": _FIELD, "history": _HISTORY}
        )


class RecordScore(ContractModel):
    """Raw JEV answers and normalized ranking score for one original.

    Attributes:
        transition_id: Exact original transition identity.
        field: Unmodified expected ordinal relevance in the range 0 through 2.
        history: Unmodified expected ordinal history connection in the same range.
        combined: Weighted normalized score in the range 0 through 1.
    """

    transition_id: str
    field: float = Field(ge=0, le=2, allow_inf_nan=False)
    history: float = Field(ge=0, le=2, allow_inf_nan=False)
    combined: float = Field(ge=0, le=1, allow_inf_nan=False)


class JEVSelector:
    """Rerank eligible originals without supplying HTTP, retry or budget ownership."""

    def __init__(
        self,
        *,
        policy: JEVPolicy,
        decide: Callable[[DecisionRequest], JsonObject],
        record: Callable[[JsonObject], None],
    ) -> None:
        """Inject an admitted native-decision caller and durable receipt recorder.

        Args:
            policy: Frozen ranking configuration bound in the simulator artifact.
            decide: Maintained decision-wire call, with shared accounting and deadlines.
            record: Retain the exact request, raw response and resulting selection.
        """
        self.policy, self._decide, self._record = policy, decide, record

    def select(
        self,
        query: RAGQuery,
        candidates: Sequence[RAGMatch],
        *,
        state: JsonObject,
        history: Sequence[ModelMessage],
        top_k: int,
    ) -> tuple[RAGMatch, ...]:
        """Score each record for this action, asking the dependency category once.

        Args:
            query: Permitted query; source lineages were excluded before candidate generation.
            candidates: Ordered exact originals from the immutable index.
            state: Candidate-owned current world, never a reference poststate.
            history: Current episode observations available before this action.
            top_k: Final evidence count, fixed by the simulator artifact.

        Returns:
            Ranked originals with normalized scores; malformed answers fail visibly.
        """
        if not candidates:
            return ()
        if len(candidates) > self.policy.candidate_limit or top_k < 1:
            raise ValueError("JEV candidate or result count exceeds the frozen policy")
        ids = [match.transition.transition_id for match in candidates]
        if len(set(ids)) != len(ids) or any(
            match.transition.lineage_id in query.excluded_lineage_ids for match in candidates
        ):
            raise ValueError("JEV candidates repeat or include an excluded lineage")
        questions: dict[str, ScoreQuestion | ChoiceQuestion | NoulQuestion] = {
            "dependency": ChoiceQuestion(
                instructions="What does this action mainly depend on? " + self.policy.guidance,
                criteria={
                    "entity-state": "Current entity identity, status, fields or version",
                    "semantic lookup": "Finding semantically related content",
                    "tool behavior": "General operation semantics or interface",
                    "unclear": "The permitted context does not establish a dependency",
                },
            )
        }
        for index in range(len(candidates)):
            questions[f"field_{index}"] = ScoreQuestion(
                instructions=_FIELD + f"Score records[{index}]. " + self.policy.guidance,
                criteria=("0=irrelevant/superseded", "1=indirect/uncertain", "2=direct evidence"),
            )
            questions[f"history_{index}"] = ScoreQuestion(
                instructions=_HISTORY + f"Score records[{index}]. " + self.policy.guidance,
                criteria=("0=unrelated", "1=weak connection", "2=linked update/read"),
            )
        request = DecisionRequest(
            state={
                "task": query.task,
                "current_action": query.action.model_dump(mode="json"),
                "known_current_state": state,
                "initial_context": query.initial_context,
                "current_episode_history": [m.model_dump(mode="json") for m in history],
                "records": [m.transition.model_dump(mode="json") for m in candidates],
                "authority": "Historical evidence, not current facts. No future outcomes supplied.",
            },
            questions=questions,
        )
        response = self._decide(request)
        receipt: JsonObject = {
            "policy_sha256": self.policy.identity,
            "request": request.model_dump(mode="json"),
            "response": response,
        }
        # Persist raw bytes even if semantic decoding fails; never hide an invalid decision.
        self._record(receipt)
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("JEV answer inventory differs from the registered questions")
        category = answers["dependency"]
        if (
            not isinstance(category, dict)
            or category.get("type") != "choice"
            or category.get("choice") not in _DEPENDENCIES
        ):
            raise ValueError("JEV dependency answer is not a declared category")
        scores = []
        for index, match in enumerate(candidates):
            field = _score(answers[f"field_{index}"])
            linked = _score(answers[f"history_{index}"])
            scores.append(
                RecordScore(
                    transition_id=match.transition.transition_id,
                    field=field,
                    history=linked,
                    combined=(
                        self.policy.field_weight * field + (1 - self.policy.field_weight) * linked
                    )
                    / 2,
                )
            )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1].combined, pair[1].transition_id),
        )
        selected = tuple(
            match.model_copy(update={"score": score.combined}) for match, score in ranked[:top_k]
        )
        self._record(
            {
                "request_sha256": sha256_json(request),
                "scores": [score.model_dump(mode="json") for score in scores],
                "dependency": category,
                "selected_ids": [match.transition.transition_id for match in selected],
            }
        )
        return selected


def _score(value: object) -> float:
    """Decode a finite ordinal score without coercing strings or boolean answers."""
    if not isinstance(value, dict) or value.get("type") != "score":
        raise ValueError("JEV expected a score answer")
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 2:
        raise ValueError("JEV score must be a finite number from 0 through 2")
    return float(score)


class JEVCueSelector:
    """Rank admitted memory records with query-relative field and history tags.

    Shortlisting uses the existing local hashing control. Native JEV decisions and
    exact record resolution keep their existing owners; no additional HTTP client exists.
    """

    def __init__(
        self,
        *,
        policy: JEVPolicy,
        top_k: int,
        decide: Callable[[DecisionRequest], JsonObject],
        record: Callable[[JsonObject], None],
    ) -> None:
        """Bind the three-question rubric, a finite shortlist and an admitted caller."""
        if top_k < 1 or top_k > policy.candidate_limit:
            raise ValueError("cue top_k must fit the JEV candidate limit")
        self.policy, self._top_k = policy, top_k
        self._decide, self._record = decide, record

    @property
    def identity(self) -> Sha256:
        """Bind the shared rubric, normalized weights and deterministic shortlist."""
        return sha256_json(
            {
                "version": "jev-memory-cues-v1",
                "rubric": self.policy.identity,
                "top_k": self._top_k,
                "shortlist": "raw-originals-signed-token-hashing-v1",
            }
        )

    def select(self, query: SelectionQuery, catalog: Mapping[str, JsonValue]) -> tuple[str, ...]:
        """Select relevant active hypotheses without promoting their authority to truth."""
        if not catalog:
            return ()
        ids = HashingOriginals(top_k=self.policy.candidate_limit).select(query, catalog)
        questions: dict[str, ScoreQuestion | ChoiceQuestion | NoulQuestion] = {
            "dependency": ChoiceQuestion(
                instructions="What does this action mainly depend on? " + self.policy.guidance,
                criteria={
                    "entity-state": "Current entity identity, status, fields or version",
                    "semantic lookup": "Finding semantically related content",
                    "tool behavior": "General operation semantics or interface",
                    "unclear": "The permitted context does not establish a dependency",
                },
            )
        }
        for index in range(len(ids)):
            questions[f"field_{index}"] = ScoreQuestion(
                instructions=_FIELD + f"Score records[{index}]. " + self.policy.guidance,
                criteria=("0=irrelevant/superseded", "1=indirect/uncertain", "2=direct evidence"),
            )
            questions[f"history_{index}"] = ScoreQuestion(
                instructions=_HISTORY + f"Score records[{index}]. " + self.policy.guidance,
                criteria=("0=unrelated", "1=weak connection", "2=linked update/read"),
            )
        request = DecisionRequest(
            state={
                "current_action": query.action.model_dump(mode="json"),
                "known_current_state": query.state,
                "current_episode_history": list(query.history),
                "records": [{"id": key, "record": catalog[key]} for key in ids],
                "authority": "Past disclosed replies plus provisional research hypotheses. "
                "Hypotheses are not native truth. No future outcomes supplied.",
            },
            questions=questions,
        )
        self._record(
            {
                "policy_sha256": self.identity,
                "request": request.model_dump(mode="json"),
                "eligible_ids": list(catalog),
                "shortlist_ids": list(ids),
            }
        )
        response = self._decide(request)
        self._record({"request_sha256": sha256_json(request), "response": response})
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("JEV memory answer inventory differs from the request")
        dependency = answers["dependency"]
        if (
            not isinstance(dependency, dict)
            or dependency.get("type") != "choice"
            or (dependency.get("choice") not in _DEPENDENCIES)
        ):
            raise ValueError("JEV memory dependency is not a declared category")
        scores = []
        for index, key in enumerate(ids):
            field, history = _score(answers[f"field_{index}"]), _score(answers[f"history_{index}"])
            scores.append(
                RecordScore(
                    transition_id=key,
                    field=field,
                    history=history,
                    combined=(
                        self.policy.field_weight * field + (1 - self.policy.field_weight) * history
                    )
                    / 2,
                )
            )
        ranked = sorted(scores, key=lambda row: (-row.combined, row.transition_id))
        selected = tuple(row.transition_id for row in ranked if row.combined > 0)[: self._top_k]
        self._record(
            {
                "request_sha256": sha256_json(request),
                "dependency": dependency,
                "scores": [row.model_dump(mode="json") for row in scores],
                "selected_ids": list(selected),
            }
        )
        return selected
