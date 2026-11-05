"""Exact frozen LLM-judge feedback on the initial group, before point selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.evidence import EvidencePacket
from exp.optimize.evaluation.variance.judged import judge_study, judged_episode, slot_rewards

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec

INITIAL_FEEDBACK_PROTOCOL = "council-initial-llm-feedback-v1"


def initial_feedback(
    project: ProjectStore,
    spec: ActorCriticSpec,
    group: ArtifactInput,
    packet: EvidencePacket,
    *,
    code_revision: str,
) -> tuple[EvidencePacket, ArtifactInput]:
    """Judge or replay the original group and expose exact labels and feedback with citations."""
    reference, judged = judge_study(
        project,
        group,
        spec.episode_judge,
        configuration=spec.initial.configuration,
        code_revision=code_revision,
    )
    rewards = slot_rewards(project, judged.branches, judged.judge)
    labels = [reward for reward in rewards if reward is not None]
    historical = packet.payload["group"]
    summary: JsonObject = {
        "protocol": INITIAL_FEEDBACK_PROTOCOL,
        "attempts_planned": len(judged.branches),
        "attempts_graded": len(labels),
        "passes": sum(labels),
        "unknown": len(rewards) - len(labels),
        "success_rate": None if not labels else sum(labels) / len(labels),
        "label_authority": "frozen full-episode LLM judge",
        "judge": judged.judge,
        "judged_study": reference.model_dump(mode="json"),
        "historical_verifier": historical,
    }
    documents = dict(packet.documents)
    documents["group"] = summary
    feedback: dict[str, JsonValue] = {}
    verdicts: dict[int, JsonObject] = {}
    for slot in judged.branches:
        assessment = judged_episode(project, slot, judged.judge)
        content: JsonObject = {
            "attempt": slot.position,
            "slot": slot.model_dump(mode="json"),
            "assessment": None if assessment is None else assessment.model_dump(mode="json"),
            "label_authority": "frozen full-episode LLM judge",
        }
        locator = f"judgment:{slot.position}"
        verdicts[slot.position] = {
            "llm_verdict": None if assessment is None else assessment.verdict,
            "llm_status": None if assessment is None else assessment.status,
            "llm_feedback": locator,
        }
        attempt = documents.get(f"attempt:{slot.position}")
        if isinstance(attempt, dict):
            documents[f"attempt:{slot.position}"] = {**attempt, **verdicts[slot.position]}
        documents[locator] = content
        feedback[locator] = content
    payload = {**packet.payload, "group": summary, "episode_judgments": feedback}
    overview = None if packet.overview is None else {**packet.overview, "group": summary}
    for rendered in (payload, overview):
        if rendered is not None:
            attempts = rendered.get("attempts")
            if isinstance(attempts, list):
                rendered["attempts"] = [
                    {**row, **verdicts.get(row["attempt"], {})}
                    if isinstance(row, dict) and isinstance(row.get("attempt"), int)
                    else row
                    for row in attempts
                ]
    for rendered in (payload, overview):
        if rendered is None:
            continue
        old = rendered.get("evidence_inventory")
        if isinstance(old, dict):
            supplied = old.get("supplied")
            unavailable = old.get("unavailable")
            rendered["evidence_inventory"] = {
                **old,
                "supplied": [
                    *(supplied if isinstance(supplied, list) else []),
                    {"material": "frozen_episode_judgments", "locator": "judgment:<attempt>"},
                ],
                "unavailable": [
                    row
                    for row in (unavailable if isinstance(unavailable, list) else [])
                    if not isinstance(row, dict)
                    or row.get("material") != "independent_correctness_labels"
                ],
            }
    if overview is not None:
        overview["locators"] = sorted(documents)
    return packet.model_copy(
        update={
            "payload": payload,
            "overview": overview,
            "documents": documents,
            "refs": frozenset(documents),
        }
    ), reference
