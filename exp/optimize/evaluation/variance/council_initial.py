"""Exact frozen LLM-judge feedback on the initial group, before point selection."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, NamedTuple

from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactInput, JsonObject, stable_id
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.variance.council_checkpoint import CouncilAdmission, admit_phase
from exp.optimize.evaluation.variance.evidence import (
    EvidencePacket,
    TraceSearch,
    load_references,
    open_trace_search,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judged import (
    JUDGED_LABELS_PROTOCOL,
    JUDGED_STUDY_TYPE,
    JudgedStudy,
    judge_study,
    judged_episode,
    slot_rewards,
)
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint
from exp.simulation.retrieval.embedding import RAGEmbedderBinding

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec, PickRecord

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
    return render_initial_feedback(project, packet, reference, judged)


def saved_initial_feedback(
    project: ProjectStore,
    spec: ActorCriticSpec,
    group: ArtifactInput,
    reference: ArtifactInput,
    packet: EvidencePacket,
) -> EvidencePacket:
    """Restore exactly the recorded LLM feedback without any possible model dispatch."""
    identity = stable_id(
        JUDGED_STUDY_TYPE,
        {
            "protocol": JUDGED_LABELS_PROTOCOL,
            "judge": spec.episode_judge.identity,
            "study": group.model_dump(mode="json"),
        },
    )
    stored = project.artifacts.read(identity)
    judged = JudgedStudy.model_validate_json(stored.payloads["judged.json"])
    if (
        artifact_input(stored.manifest) != reference
        or stored.manifest.artifact_type != JUDGED_STUDY_TYPE
        or judged.study != group
        or judged.judge != spec.episode_judge.identity
        or len(judged.branches) != spec.initial.study.repeats
    ):
        raise ValueError("native selector feedback differs from the exact initial LLM judgments")
    return render_initial_feedback(project, packet, reference, judged)[0]


def render_initial_feedback(
    project: ProjectStore,
    packet: EvidencePacket,
    reference: ArtifactInput,
    judged: JudgedStudy,
) -> tuple[EvidencePacket, ArtifactInput]:
    """Render committed full-episode labels and supplied citations without model construction."""
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


class PreparedCouncil(NamedTuple):
    """Native initial group, judged feedback and read-only selected evidence."""

    group: ContinuationStudy
    source: ArtifactInput
    initial_judged: ArtifactInput
    pick: PickRecord
    packet: EvidencePacket
    loader: Callable[[ArtifactInput], ContinuationCheckpoint]
    search: TraceSearch | None


def prepare_initial(
    project: ProjectStore,
    spec: ActorCriticSpec,
    identity: str,
    *,
    exposure: ExposurePlan,
    code_revision: str,
    embedder: RAGEmbedderBinding | None,
    admission: CouncilAdmission | None,
) -> PreparedCouncil:
    """Sample and LLM-grade a new group within one native wave, or grade its saved actors.

    A cached TraceML actor group needs only the initial-judging wave. New groups' actor-wave
    inventory includes their LLM labels and this owner performs both before its scope exits.
    """
    from exp.optimize.evaluation.variance.actor_critic import _pick

    store = project.artifacts
    group = saved_continuation_job(project, spec.initial, code_revision=code_revision)
    source = (
        None
        if group is None
        else continuation_result_reference(
            project,
            spec.initial,
            code_revision=code_revision,
        )
    )
    phase = "actors" if group is None else "initial_judging"
    with admit_phase(admission, spec, identity, phase, group=source):
        if group is None:
            group = execute_continuation_job(project, spec.initial, code_revision=code_revision)
            source = continuation_result_reference(
                project, spec.initial, code_revision=code_revision
            )
        assert source is not None
        loader = cache(lambda ref: load_continuation_checkpoint(store, ref))
        references = load_references(store, spec.references)
        search = open_trace_search(store, r, exposure, embedder) if (r := spec.retrieval) else None
        pick, packet = _pick(group, identity, spec, loader, references)
        packet, judged = initial_feedback(
            project, spec, source, packet, code_revision=code_revision
        )
    return PreparedCouncil(group, source, judged, pick, packet, loader, search)
