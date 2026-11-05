"""Provider-free native phase inventories; deliberately not a spending authority.

A phase quote does not reserve headroom across parallel owners or expand preparation authority.
A physical child-scope ledger and durable outstanding envelopes are required before a conditional
parent policy can use these quotes to dispatch under a different parent ceiling.
"""

from __future__ import annotations

from typing import Literal

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json, stable_id
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.catalog_binding import load_runtime_catalog
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, catalog_model_cost
from exp.optimize.evaluation.fidelity.retrieval_binding import rag_index_embedder_alias
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.council_checkpoint import CouncilCheckpoint, point_job
from exp.optimize.evaluation.variance.evidence_tools import AnalysisTranscript
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls
from exp.optimize.evaluation.variance.judged import (
    JUDGED_LABELS_PROTOCOL,
    JUDGED_STUDY_TYPE,
    JudgedStudy,
)
from exp.optimize.evaluation.variance.pipeline_plan import CostLine
from exp.optimize.evaluation.variance.role_prompts import assign_prompts
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.retrieval.store import load_rag_index


class CouncilPhaseInventory(ContractModel):
    """Exact bounded native jobs/roles and query embeddings outstanding in one checkpoint.

    Attributes:
        protocol: Inventory reader protocol.
        spec_sha256: Exact registered spec including operational dispatch metadata.
        checkpoint: Native phase and logical remaining-work record that was priced.
        costs: Full native phase upper bounds, never a free-inference dollar claim.
        replayed: Exact committed owners omitted from pending cost, with provenance.
        authority: Structurally explicit distinction between a quote and physical authorization.
    """

    protocol: Literal["council-native-phase-inventory-v1"] = "council-native-phase-inventory-v1"
    spec_sha256: Sha256
    checkpoint: CouncilCheckpoint
    costs: tuple[CostLine, ...]
    replayed: tuple[ArtifactInput, ...] = ()
    authority: Literal["inventory_only"] = "inventory_only"

    @property
    def attempts(self) -> int:
        """Full physical-request upper bound, including known-price dependencies."""
        return sum(row.attempts for row in self.costs)

    @property
    def unpriced_attempts(self) -> int:
        """Unpriced request bound; unknown prices never become zero-cost guarantees."""
        return sum(row.attempts for row in self.costs if row.usd is None)

    @property
    def known_usd(self) -> float:
        """Known-rate part of the upper bound, separate from unpriced liabilities."""
        return sum(row.usd for row in self.costs if row.usd is not None)


def phase_inventory(
    project: ProjectStore,
    spec: ActorCriticSpec,
    checkpoint: CouncilCheckpoint,
    *,
    code_revision: str,
) -> CouncilPhaseInventory:
    """Price only exact native owners; read saved cache provenance without dispatching.

    Settled checkpoints dispatch nothing. Job and role replay are recognized only through
    their own committed owner artifacts; a saved response alone is not a completed stage.
    """
    if checkpoint.run_identity != spec.identity(code_revision):
        raise ValueError("checkpoint does not belong to the exact native council registration")
    costs: list[CostComponent] = []
    replayed: list[ArtifactInput] = []

    def judgments(source: ArtifactInput | None, count: int) -> None:
        """Use the native judged-study identity and immutable record, not a guessed alias."""
        if source is not None:
            identity = stable_id(
                JUDGED_STUDY_TYPE,
                {
                    "protocol": JUDGED_LABELS_PROTOCOL,
                    "judge": spec.episode_judge.identity,
                    "study": source.model_dump(mode="json"),
                },
            )
            if project.artifacts.exists(identity):
                stored = project.artifacts.read(identity)
                saved = JudgedStudy.model_validate_json(stored.payloads["judged.json"])
                if (
                    stored.manifest.artifact_type != JUDGED_STUDY_TYPE
                    or saved.study != source
                    or saved.judge != spec.episode_judge.identity
                    or len(saved.branches) != count
                ):
                    raise ValueError("saved LLM group judging differs from the native owner")
                replayed.append(artifact_input(stored.manifest))
                return
        calls = count * judge_model_calls(spec.episode_judge.definition.profile)
        costs.extend(
            row.scaled(calls) for row in evaluation_costs(project, spec.episode_judge.settings)
        )

    def study(job: ContinuationJob) -> None:
        """Price an uncommitted native job and its required LLM labels exactly once."""
        saved = saved_continuation_job(project, job, code_revision=code_revision)
        source = None
        if saved is None:
            costs.extend(job_costs(project, job))
        else:
            source = continuation_result_reference(project, job, code_revision=code_revision)
            replayed.append(source)
        judgments(source, job.study.repeats)

    phase = checkpoint.phase
    if phase == "actors":
        if checkpoint.jobs != (spec.initial,):
            raise ValueError("actor wave inventory must contain the exact original group")
        study(spec.initial)
    elif phase == "initial_judging":
        if checkpoint.group is None:
            raise ValueError("initial LLM grading requires the committed native group")
        expected = continuation_result_reference(project, spec.initial, code_revision=code_revision)
        if checkpoint.group != expected:
            raise ValueError("initial LLM grading cannot substitute a different group")
        judgments(checkpoint.group, spec.initial.study.repeats)
    elif phase in ("selectors", "critics"):
        selector = phase == "selectors"
        settings = spec.judges if selector else spec.critics
        profile = spec.selector_profile if selector else spec.critic_profile
        distribution = spec.selector_prompts if selector else spec.critic_prompts
        prompts = assign_prompts(distribution, seed=spec.seed, draws=len(settings))
        if checkpoint.draws != tuple(range(len(settings))):
            raise ValueError("role wave cannot drop or duplicate any requested draw")
        uncached = 0
        for draw, role in enumerate(settings):
            kind = "judge" if selector else "critic"
            identity = f"{checkpoint.run_identity}-{kind}-{draw}-transcript"
            if project.artifacts.exists(identity):
                stored = project.artifacts.read(identity)
                transcript = AnalysisTranscript.model_validate_json(
                    stored.payloads["transcript.json"]
                )
                if (
                    stored.manifest.code_revision != code_revision
                    or transcript.profile != profile
                    or transcript.prompt != prompts[draw]
                    or transcript.answer_tool
                    != ("choose_points" if selector else "estimate_points")
                ):
                    raise ValueError("role replay does not match its native profile or producer")
                replayed.append(artifact_input(stored.manifest))
            else:
                uncached += 1
                costs.extend(
                    row.scaled(profile.budget.model_calls)
                    for row in evaluation_costs(project, role)
                )
        if spec.retrieval is not None and uncached:
            loaded = load_rag_index(project.artifacts, spec.retrieval.index.artifact_id)
            if artifact_input(loaded.manifest) != spec.retrieval.index:
                raise ValueError("search index differs from the native registered reference")
            binding = spec.retrieval.catalog_binding
            catalog = (
                load_model_catalog(project.model_catalog_path)
                if binding is None
                else load_runtime_catalog(project, binding)
            )
            alias = rag_index_embedder_alias(
                project,
                catalog,
                loaded.index,
                configured_alias=None if binding is None else binding.embedder_alias,
            )
            if alias is not None:
                calls = uncached * profile.budget.tool_calls * RetryPolicy().maximum_attempts
                costs.append(catalog_model_cost(catalog, alias, calls, 0))
    elif phase == "continuations":
        if len(checkpoint.jobs) != len(checkpoint.wave) or len(set(checkpoint.wave)) != len(
            checkpoint.wave
        ):
            raise ValueError("each continuation boundary requires one distinct native job")
        for job in checkpoint.jobs:
            if job != point_job(spec, job.study.checkpoint):
                raise ValueError("continuation wave cannot reduce repeats or change the horizon")
            study(job)
    return CouncilPhaseInventory(
        spec_sha256=sha256_json(spec),
        checkpoint=checkpoint,
        costs=tuple(CostLine(name=row.name, attempts=row.attempts, usd=row.usd) for row in costs),
        replayed=tuple(sorted(set(replayed), key=lambda ref: ref.artifact_id)),
    )
