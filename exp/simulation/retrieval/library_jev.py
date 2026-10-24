"""Infer prior-library applicability from new target trajectories using native JEV."""

import json
from collections.abc import Callable

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.common.project import ArtifactStore
from exp.runtime.gateway.decisions_contracts import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)
from exp.simulation.reconstruction.library import LibraryEntry, resolve_library_entry
from exp.simulation.retrieval.embedding import HashingRAGEmbedder


class LibraryJEVPolicy(ContractModel):
    """Frozen bounds for selecting prior knowledge, independent of current-world state.

    Attributes:
        candidate_limit: Native JEV bound after the recorded lexical cue shortlist.
        top_k: Maximum records delivered to the next construction.
        minimum_relevance: Combined ordinal score strictly exceeded before delivery.
        field_weight: Contribution of field relevance versus an actual history connection.
        maximum_input_bytes: Conservative complete decision-frame bound before dispatch.
    """

    candidate_limit: int = Field(default=15, ge=1, le=15, strict=True)
    top_k: int = Field(default=3, ge=1, le=15, strict=True)
    minimum_relevance: float = Field(default=1, ge=0, le=2, allow_inf_nan=False)

    field_weight: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    maximum_input_bytes: int = Field(default=24_000, ge=4_096, le=24_000, strict=True)

    @property
    def identity(self) -> Sha256:
        """Bind the configuration and target-trace-only applicability protocol."""
        return sha256_json(
            {"version": "library-jev-v4-structural-cues", "policy": self.model_dump(mode="json")}
        )


class JEVLibrarySelector:
    """Rank authorized prior sources/hypotheses without a hand-authored environment router."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        policy: LibraryJEVPolicy,
        decide: Callable[[DecisionRequest], JsonObject],
        record: Callable[[JsonObject], None],
    ) -> None:
        """Bind the maintained artifact and native-decision owners, without another HTTP client."""
        self._store, self.policy, self._decide, self._record = store, policy, decide, record

    def __call__(self, query: JsonObject, entries: tuple[LibraryEntry, ...]) -> tuple[str, ...]:
        """Choose reusable material using raw target trajectories and admitted exact payloads.

        Args:
            query: Target traces rendered by retrieve_library, without future outcomes.
            entries: Eligible originals, accepted rules/skills and labelled research.

        Returns:
            At most top_k entries above the fixed relevance threshold; possibly none.
        """
        if not entries:
            return ()
        if any(
            entry.kind != "research" and entry.disposition not in ("source", "accepted")
            for entry in entries
        ):
            raise ValueError("unaccepted rules are not eligible prior knowledge")
        if len(entries) > self.policy.candidate_limit:
            vectors = HashingRAGEmbedder().embed(
                [json.dumps(query, sort_keys=True), *(" ".join(e.cues) for e in entries)]
            )
            query_vector = vectors[0].values
            scores = [
                sum(a * b for a, b in zip(query_vector, vector.values, strict=True))
                for vector in vectors[1:]
            ]
            ranked = sorted(range(len(entries)), key=lambda i: (-scores[i], entries[i].entry_id))
            retained = ranked[: self.policy.candidate_limit]
            self._record(
                {
                    "candidate_policy": "signed-token-hashing-cues-v1-256",
                    "query_sha256": sha256_json(query),
                    "eligible_count": len(entries),
                    "scores": [
                        {"entry_id": entry.entry_id, "score": score}
                        for entry, score in zip(entries, scores, strict=True)
                    ],
                    "shortlist": [entries[i].entry_id for i in retained],
                }
            )
            entries = tuple(entries[i] for i in retained)
        records = [
            {
                "entry": entry.model_dump(mode="json"),
                "content_cues": _content_cues(
                    resolve_library_entry(self._store, entry), entry.kind
                ),
            }
            for entry in entries
        ]
        target = {
            "query_sha256": sha256_json(query),
            "structural_cues": _content_cues(query, "trajectory"),
            "authority": "Field/type/tool cues only. Full originals resolve after selection. "
            "A field's presence does not establish a behavioral law or its current value.",
        }
        questions: dict[str, ChoiceQuestion | ScoreQuestion | NoulQuestion] = {
            "dependency": ChoiceQuestion(
                instructions="What does the observed target environment mainly require?",
                criteria={
                    "entity-state": "Identity/status/field tracking",
                    "semantic lookup": "Finding related information",
                    "tool behavior": "Operation semantics or interface reconstruction",
                    "unclear": "Insufficient target evidence",
                },
            )
        }
        for i in range(len(entries)):
            questions[f"relevance_{i}"] = ScoreQuestion(
                instructions=(
                    f"Does records[{i}] explain a field or behavioral distinction needed by the "
                    "target traces? Historical entity values cannot initialize target state. "
                    "A shared tool name alone is indirect evidence. Rejected investigations can "
                    "explain a failed approach, but cannot establish native behavior."
                ),
                criteria=("0=irrelevant/superseded", "1=indirect/uncertain", "2=direct evidence"),
            )
            questions[f"history_{i}"] = ScoreQuestion(
                instructions=(
                    f"Is records[{i}] connected to the target traces' observed history? "
                    "Use actual source-episode, entity or update/read links. Shared tool names "
                    "or a similar environment alone do not establish a linked episode. "
                    "A different environment's episode with no explicit link scores 0."
                ),
                criteria=("0=unrelated", "1=weak connection", "2=linked update/read"),
            )
        request = DecisionRequest(
            state={"target": target, "prior_records": records}, questions=questions
        )
        omitted = []
        while len(request.model_dump_json().encode()) + 1024 > self.policy.maximum_input_bytes:
            if len(entries) == 1:
                raise ValueError("library JEV cue frame exceeds the registered input budget")
            omitted.append(entries[-1].entry_id)
            entries, records = entries[:-1], records[:-1]
            questions.pop(f"relevance_{len(entries)}")
            questions.pop(f"history_{len(entries)}")
            request = DecisionRequest(
                state={"target": target, "prior_records": records}, questions=questions
            )
        self._record(
            {
                "policy_sha256": self.policy.identity,
                "input_view": "structural-cues-v1",
                "request_sha256": sha256_json(request),
                "request_bytes_with_reserve": len(request.model_dump_json().encode()) + 1024,
                "omitted_for_budget": omitted,
                "request": request.model_dump(mode="json"),
            }
        )
        response = self._decide(request)
        self._record(
            {
                "policy_sha256": self.policy.identity,
                "request": request.model_dump(mode="json"),
                "response": response,
            }
        )
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("library JEV answer inventory differs from the registered request")
        category = answers["dependency"]
        if (
            not isinstance(category, dict)
            or category.get("choice")
            not in ("entity-state", "semantic lookup", "tool behavior", "unclear")
            or category.get("type") != "choice"
        ):
            raise ValueError("library JEV dependency is not a registered category")
        scored = []
        for i, entry in enumerate(entries):
            field = _ordinal(answers[f"relevance_{i}"])
            history = _ordinal(answers[f"history_{i}"])
            score = self.policy.field_weight * field + (1 - self.policy.field_weight) * history
            scored.append((score, entry.entry_id, field, history))
        selected = tuple(
            key
            for score, key, field, history in sorted(scored, key=lambda row: (-row[0], row[1]))
            if score > self.policy.minimum_relevance
        )[: self.policy.top_k]
        self._record(
            {
                "request_sha256": sha256_json(request),
                "selected": list(selected),
                "scores": [
                    {"entry_id": key, "field": field, "history": history, "combined": score}
                    for score, key, field, history in scored
                ],
            }
        )
        return selected


def _ordinal(answer: JsonValue) -> float:
    """Decode native JEV expected ordinal scores without coercing malformed answers."""
    if not isinstance(answer, dict) or answer.get("type") != "score":
        raise ValueError("library JEV relevance requires an ordinal score")
    score = answer.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 2:
        raise ValueError("library JEV score must be finite and between zero and two")
    return float(score)


def _content_cues(value: JsonValue, kind: str) -> JsonObject:
    """Expose bounded source structure, not entire historical entity snapshots to JEV.

    Original payloads remain unchanged and resolve in full after selection. These cues
    intentionally omit instance values; matching field names is not proof of behavior.
    """
    if kind in ("rule", "skill") and isinstance(value, dict):
        knowledge = value.get("knowledge", value)
        if isinstance(knowledge, dict):
            return {
                key: knowledge[key]
                for key in ("rules", "text_rules", "unknowns", "source_trace_ids")
                if key in knowledge
            }
    if kind == "research" and isinstance(value, dict):
        return {
            key: value[key]
            for key in ("content", "disposition", "source_trace_ids", "investigation_sha256")
            if key in value
        }
    fields: set[str] = set()
    tools: set[str] = set()
    episodes: set[str] = set()
    tasks: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str) and item[:1] in ("{", "["):
            try:
                item = json.loads(item)
            except ValueError:
                continue
        if isinstance(item, dict):
            fields.update(item)
            tool = item.get("gen_ai.tool.name")
            if isinstance(tool, str):
                tools.add(tool)
            action = item.get("action")
            if isinstance(action, dict) and isinstance(action.get("name"), str):
                tools.add(action["name"])
            episode = item.get("trace_id")
            if isinstance(episode, str):
                episodes.add(episode)
            elif "records" in item and isinstance(item.get("id"), str):
                episodes.add(item["id"])
            task = item.get("task")
            if isinstance(task, str):
                tasks.add(task)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return {
        "observed_tool_names": sorted(tools),
        "observed_field_names": sorted(fields)[:256],
        "field_names_omitted": max(0, len(fields) - 256),
        "source_episode_ids": sorted(episodes),
        "task_excerpts": [text[:256] for text in sorted(tasks)[:10]],
        "task_count": len(tasks),
        "view": "Structural cues only; instance values and behavioral effects omitted",
    }
