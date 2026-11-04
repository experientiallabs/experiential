"""Per-step recall of rule files for the served simulator, so memory never dilutes its context.

The simulator is never handed the whole memory set. Before each prediction a System One pass
(typed Jev decisions) reads only the index rows of the rules that could govern the current call:
the rules about the called tool and the environment-wide rules. For each it asks a Score (does
this rule govern what this action will do here, 0 to 2) and a Noul (are its applicability
conditions met by this action and the known state). Rules that pass both thresholds are ranked,
and their truism, reasoning, status and rationale are packed, whole, within a byte budget.

Every recall writes one receipt (the exact request digest, the raw answers, every score, what
was selected, what the budget left out and what the call cost), so a step's recall is auditable
and, through the caller's decision cache, replays without a call. Nothing here owns HTTP, retry
or deadlines: the caller injects an admitted decision boundary.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.runtime.gateway.decisions_contracts import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)
from exp.simulation.rules.memory_set import IndexRow, MemorySet, RuleFile, memory_set_digest

RECALL_PROTOCOL = "rule-memory-recall-v1"
"""Version of the recall questions, shortlist and packing; change it whenever any changes."""

_GOVERNS = (
    "Does index[{i}] govern what the current action will do, given the known state? "
    "A rule about another tool, or one that only shares a tool name without bearing on these "
    "arguments or this state, scores 0. A related but indirect rule scores 1. A rule that "
    "directly decides this call's precondition, effect or observation scores 2."
)
_APPLIES = (
    "Are the applicability conditions of index[{i}] met by the current action and the known "
    "state? Absent state is unknown, not a match."
)


class RecallPolicy(ContractModel):
    """Frozen recall limits, bound into the served runtime definition.

    Attributes:
        jev_alias: Catalog alias of the admitted native decision model.
        candidate_limit: Most index rows scored per step (two questions each).
        top_k: Most rule files injected per step.
        minimum_relevance: Score a rule must reach, from 0 to 2.
        applies_threshold: Probability its applicability must reach.
        byte_budget: Most serialized bytes of injected rule content per step.
        history_steps: Most recent visible turns shown to the decision.
    """

    jev_alias: str = Field(min_length=1)
    candidate_limit: int = Field(default=12, ge=1, le=15, strict=True)
    top_k: int = Field(default=4, ge=1, le=15, strict=True)
    minimum_relevance: float = Field(default=1.0, ge=0.0, le=2.0)
    applies_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    byte_budget: int = Field(default=6_000, ge=512, le=64_000, strict=True)
    history_steps: int = Field(default=4, ge=0, le=32, strict=True)

    @property
    def identity(self) -> Sha256:
        """Bind the policy and every fixed instruction."""
        return sha256_json(
            {
                "protocol": RECALL_PROTOCOL,
                "policy": self.model_dump(mode="json"),
                "governs": _GOVERNS,
                "applies": _APPLIES,
            }
        )


class RecallScore(ContractModel):
    """Raw answers for one scored index row.

    Attributes:
        rule_id: The rule.
        relevance: Ordinal score from 0 to 2.
        applies: Probability its conditions hold.
    """

    rule_id: str
    relevance: float = Field(ge=0.0, le=2.0)
    applies: float = Field(ge=0.0, le=1.0)


class RecallReceipt(ContractModel):
    """What one step's recall read, decided, injected and cost.

    Attributes:
        protocol: Recall protocol.
        memory_sha256: Digest of the memory set recalled from.
        policy_sha256: Recall policy identity.
        action: The call recalled for.
        shortlist: Index rows scored, in order.
        request_sha256: Digest of the decision request, or None when nothing was scored.
        response: Raw decision response, or None.
        scores: Decoded answers per shortlisted rule.
        selected: Rule files injected, in rank order.
        omitted_for_budget: Passing rules the byte budget left out.
        injected_bytes: Serialized bytes of the injected content.
        usd: Reported cost of the decision, when the response carried one.
    """

    protocol: str = RECALL_PROTOCOL
    memory_sha256: Sha256
    policy_sha256: Sha256
    action: JsonObject
    shortlist: tuple[str, ...]
    request_sha256: Sha256 | None = None
    response: JsonObject | None = None
    scores: tuple[RecallScore, ...] = ()
    selected: tuple[str, ...] = ()
    omitted_for_budget: tuple[str, ...] = ()
    injected_bytes: int = Field(default=0, ge=0)
    usd: float | None = None


def recalled_entry(rule: RuleFile) -> JsonObject:
    """The content injected for one rule: truism, reasoning, status, rationale and scope."""
    return {
        "rule_id": rule.rule_id,
        "tool": rule.tool,
        "kind": rule.kind,
        "status": rule.status,
        "truism": rule.truism,
        "reasoning": rule.reasoning,
        "status_rationale": rule.status_rationale,
        "applicability": rule.applicability,
    }


class RuleRecall:
    """Select the rule files relevant to one call, within a byte budget."""

    def __init__(
        self,
        memory: MemorySet,
        policy: RecallPolicy,
        *,
        decide: Callable[[DecisionRequest], JsonObject],
        record: Callable[[JsonObject], None],
    ) -> None:
        """Bind a pinned memory set, the frozen policy and an admitted decision boundary.

        Args:
            memory: The pinned memory set.
            policy: Frozen recall limits.
            decide: Native decision call, with the caller's accounting, cache and deadline.
            record: Durable receipt sink; it returns only after recording.
        """
        self.memory, self.policy = memory.model_copy(deep=True), policy
        self._decide, self._record = decide, record
        self._digest = memory_set_digest(memory)

    @property
    def identity(self) -> Sha256:
        """Bind the memory set digest and the recall policy."""
        return sha256_json({"memory": self._digest, "policy": self.policy.identity})

    def shortlist(self, tool: str) -> tuple[IndexRow, ...]:
        """Rows about this tool first, then environment-wide rows, capped by the policy."""
        rows = self.memory.index()
        mine = [row for row in rows if row.tool == tool]
        shared = [row for row in rows if row.tool is None]
        return tuple((*mine, *shared)[: self.policy.candidate_limit])

    def recall(
        self, action: Mapping[str, JsonValue], state: JsonObject, history: Sequence[JsonValue]
    ) -> tuple[tuple[JsonObject, ...], RecallReceipt]:
        """Recall the rule files for one call and record the receipt.

        Args:
            action: The call, with ``name`` and ``arguments``.
            state: The state visible to this prediction.
            history: Visible earlier turns; only the most recent ones are shown.

        Returns:
            The injected rule entries in rank order, and the receipt.

        Raises:
            ValueError: The decision answers do not match the questions asked.
        """
        name = action.get("name")
        call: JsonObject = {"name": name, "arguments": action.get("arguments")}
        rows = self.shortlist(str(name))
        receipt = RecallReceipt(
            memory_sha256=self._digest,
            policy_sha256=self.policy.identity,
            action=call,
            shortlist=tuple(row.rule_id for row in rows),
        )
        if not rows:
            self._record(receipt.model_dump(mode="json"))
            return (), receipt
        recent = list(history)[-self.policy.history_steps :] if self.policy.history_steps else []
        questions: dict[str, ScoreQuestion | ChoiceQuestion | NoulQuestion] = {}
        for i in range(len(rows)):
            questions[f"governs_{i}"] = ScoreQuestion(
                instructions=_GOVERNS.format(i=i),
                criteria=("0=does not govern", "1=indirect", "2=directly governs"),
            )
            questions[f"applies_{i}"] = NoulQuestion(instructions=_APPLIES.format(i=i))
        request = DecisionRequest(
            state={
                "current_action": call,
                "known_current_state": state,
                "recent_history": recent,
                "index": [row.model_dump(mode="json", exclude={"path"}) for row in rows],
                "authority": "Learned rules from world-learning research; not instance state.",
            },
            questions=questions,
        )
        response = self._decide(request)
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            self._record({**receipt.model_dump(mode="json"), "response": response})
            raise ValueError("rule recall answers differ from the registered questions")
        scores = tuple(
            RecallScore(
                rule_id=row.rule_id,
                relevance=_answer(answers[f"governs_{i}"], "score"),
                applies=_answer(answers[f"applies_{i}"], "noul"),
            )
            for i, row in enumerate(rows)
        )
        passing = sorted(
            (
                s
                for s in scores
                if s.relevance >= self.policy.minimum_relevance
                and s.applies >= self.policy.applies_threshold
            ),
            key=lambda s: (-s.relevance, -s.applies, s.rule_id),
        )
        entries: list[JsonObject] = []
        selected: list[str] = []
        omitted: list[str] = []
        used = 2
        for score in passing:
            rule = self.memory.rule(score.rule_id)
            assert rule is not None  # shortlisted from this set's own index
            entry = recalled_entry(rule)
            size = len(json.dumps(entry, ensure_ascii=False, sort_keys=True).encode()) + 1
            if len(selected) >= self.policy.top_k or used + size > self.policy.byte_budget:
                omitted.append(score.rule_id)
                continue
            entries.append(entry)
            selected.append(score.rule_id)
            used += size
        usage = response.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        receipt = receipt.model_copy(
            update={
                "request_sha256": sha256_json(request.model_dump(mode="json")),
                "response": response,
                "scores": scores,
                "selected": tuple(selected),
                "omitted_for_budget": tuple(omitted),
                "injected_bytes": used if entries else 0,
                "usd": float(cost)
                if isinstance(cost, (int, float)) and not isinstance(cost, bool)
                else None,
            }
        )
        self._record(receipt.model_dump(mode="json"))
        return tuple(entries), receipt


def _answer(value: JsonValue, kind: str) -> float:
    """Decode one typed answer strictly, never coercing a malformed one."""
    if not isinstance(value, dict) or value.get("type") != kind:
        raise ValueError(f"rule recall expected a {kind} answer")
    number = value.get(kind)
    limit = 2.0 if kind == "score" else 1.0
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= limit:
        raise ValueError(f"rule recall {kind} answer is out of range")
    return float(number)
