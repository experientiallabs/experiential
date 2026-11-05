"""Model-targeted behavior curriculum using the existing typed trace-tagging question banks.

The LLM-curated library supplies detection semantics. Compilation only preserves those
semantics and their evidence/provenance; a tag is diagnostic, never an automatic failure.
"""

from datetime import UTC, datetime
from typing import Literal

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelSnapshot
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.authoring_packet import AuthoringPacket, load_packet
from exp.optimize.evaluation.variance.behavior_profiles import (
    BehaviorLibrary,
    LibraryBinding,
    load_library,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.rubric_author import load_rubric
from exp.simulation.mining.questions import Question, QuestionBank

BEHAVIOR_QUESTIONS_PROTOCOL = "council-behavior-questions-v1"


class ModelBehaviorBank(ContractModel):
    """Frozen diagnostic bank and separate model/checkpoint targeting provenance.

    Attributes:
        protocol: Version of the compilation and evidence contract.
        packet: Same immutable coalition evidence used by the rubric author.
        rubric: Corresponding full-episode LLM judge rubric.
        library: Exact accepted LLM-authored behavior definitions.
        target_tier: Curriculum target, never inserted into the correctness proposition.
        target_model: Actual sampled policy snapshot.
        exposure_sha256: Frozen split under which the examples were admitted.
        bank: Standard question bank consumed by DecisionsTraceTagger and its shared cache.
        evidence: Current-family positive locators supporting each targeted question.
        diagnostic_only: Tags cannot replace the episode judge or impose a blanket hard failure.
    """

    protocol: Literal["council-behavior-questions-v1"] = BEHAVIOR_QUESTIONS_PROTOCOL
    packet: ArtifactInput
    rubric: ArtifactInput
    library: ArtifactInput
    target_tier: str
    target_model: ModelSnapshot
    exposure_sha256: Sha256
    bank: QuestionBank
    evidence: dict[str, tuple[str, ...]]
    diagnostic_only: Literal[True] = True


def compile_behavior_questions(
    packet: AuthoringPacket,
    library: BehaviorLibrary,
    *,
    target_tier: str,
    target_model: ModelSnapshot,
    exposure: ExposurePlan,
) -> tuple[QuestionBank, dict[str, tuple[str, ...]]] | None:
    """Compile LLM-authored detection questions supported by this target's mining evidence.

    Generic correctness wording is shared across policies. Only the frozen curriculum metadata
    and positive-example selection are model-conditioned. Missing evidence stays uncertain.
    """
    coalition = packet.coalition
    coalition.check_exposure(exposure)
    tiers = {row.tier: row for row in coalition.tiers}
    if target_tier not in tiers or tiers[target_tier].model != target_model.model_id:
        raise ValueError("behavior target must match an actual sampled coalition model")
    attempts = {
        row.attempt_id
        for row in coalition.attempts
        if row.tier == target_tier and exposure.roles[row.lineage] == "mining_probes"
    }
    relevant = {f"attempt:{identity}" for identity in attempts}
    relevant |= {
        f"pattern:{row.pattern_id}"
        for row in coalition.patterns
        if attempts.intersection(row.attempts)
    }
    questions: list[Question] = []
    evidence: dict[str, tuple[str, ...]] = {}
    for row in library.linters:
        if row.status != "active":
            continue
        positives = tuple(
            sorted(
                {
                    example.locator
                    for example in row.examples
                    if example.family_id == packet.family_id
                    and example.polarity == "positive"
                    and example.locator in relevant
                }
            )
        )
        if not positives:
            continue
        text = (
            f"Does the trace exhibit this behavior: {row.description}\n"
            f"Detection: {row.detection}\nApplicable when: {row.applies_when}\n"
            f"Required correctness only when: {row.required_when}\n"
            "Evidence needed: "
            + "; ".join(f"{e.kind}: {e.detail}" for e in row.evidence_needed)
            + "\nAcceptable look-alikes: "
            + "; ".join(row.false_positive_notes)
            + "\nCounterexamples: "
            + "; ".join(example.note for example in row.examples if example.polarity == "negative")
            + "\nReturn a probability of the behavior in the rendered view. Missing records are "
            "uncertainty, not proof. A diagnostic tag alone does not decide task success."
        )
        if coalition.mentions((text,)):
            raise ValueError("behavior propositions must not name actor models or tiers")
        identity = stable_id("behavior", {"linter": row.linter_id, "text": text})
        questions.append(Question(question_id=identity, family="behavior", text=text))
        evidence[identity] = positives
    if not questions:
        return None
    return QuestionBank(
        bank_id=stable_id(
            "behavior-bank",
            {"questions": [question.model_dump(mode="json") for question in questions]},
        ),
        version=1,
        questions=tuple(questions),
    ), evidence


def save_behavior_questions(
    project: ProjectStore,
    *,
    packet: ArtifactInput,
    rubric: ArtifactInput,
    library: LibraryBinding,
    target_tier: str,
    target_model: ModelSnapshot,
    exposure: ExposurePlan,
    code_revision: str,
) -> ArtifactInput | None:
    """Bind both council outputs to exact native evidence without buying another classifier."""
    source = load_packet(project, packet)
    saved_library = load_library(project, library.reference)
    if saved_library != library.library:
        raise ValueError("behavior library binding does not match its saved version")
    judge = load_rubric(project, rubric)
    if judge.packet_sha256 != source.identity or judge.family_id != source.family_id:
        raise ValueError("behavior bank and rubric must share the identical authoring packet")
    if judge.library != library.citation():
        raise ValueError("behavior bank and rubric must share the identical behavior library")
    compiled = compile_behavior_questions(
        source,
        saved_library,
        target_tier=target_tier,
        target_model=target_model,
        exposure=exposure,
    )
    if compiled is None:
        return None
    bank, evidence = compiled
    body = ModelBehaviorBank(
        packet=packet,
        rubric=rubric,
        library=library.reference,
        target_tier=target_tier,
        target_model=target_model,
        exposure_sha256=sha256_json(exposure),
        bank=bank,
        evidence=evidence,
    )
    identity = stable_id(
        "council-behavior-bank", {"body": body.model_dump(mode="json"), "producer": code_revision}
    )
    inputs = tuple(sorted((packet, rubric, library.reference), key=lambda ref: ref.artifact_id))
    if project.artifacts.exists(identity):
        stored = project.artifacts.read(identity)
        if (
            stored.manifest.inputs != inputs
            or ModelBehaviorBank.model_validate_json(stored.payloads["bank.json"]) != body
        ):
            raise ValueError("saved behavior bank has incompatible evidence or contents")
        return artifact_input(stored.manifest)
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="council-behavior-bank",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=inputs,
        ),
        files={"bank.json": body},
    )
    return artifact_input(manifest)
