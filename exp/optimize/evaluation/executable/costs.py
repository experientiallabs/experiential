"""Credential-free planning costs for the complete declared evaluation schedule.

These estimates price full context capacity and every physical allowance, including simulator
calls. Their known-rate subtotal becomes the operation's consented spend ceiling, which every
paid request reserves against at dispatch (``exp.runtime.models.spend_ceiling``). Unknown
tariffs and arbitrary trusted runtime code stay unpriced and need explicit consent, and existing
dispatch ledgers enforce the attempt limits.
"""

from pathlib import Path

from exp.common.core.artifacts import JsonObject, Sha256, sha256_bytes
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.analysis.actor_decisions import ActorDecisionSpec
from exp.optimize.evaluation.analysis.cascade_confirmation import (
    CascadeConfirmationSpec,
    confirmation_panel,
)
from exp.optimize.evaluation.analysis.decision_cascade import CascadeFitSpec
from exp.optimize.evaluation.analysis.memory_analyst import MemoryAnalystSpec
from exp.optimize.evaluation.continuations.action_contrast import ActionContrastSpec
from exp.optimize.evaluation.continuations.actors import ActorPanelSpec, panel_jobs
from exp.optimize.evaluation.continuations.campaign import ContinuationCampaignSpec
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.continuations.critic import CriticSpec
from exp.optimize.evaluation.continuations.experiment import CheckpointExperimentSpec
from exp.optimize.evaluation.continuations.framing import FramingStudySpec
from exp.optimize.evaluation.continuations.intervention import (
    ActionIntervention,
    ActionInterventionSpec,
)
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.observation_contrast import ObservationContrastSpec
from exp.optimize.evaluation.continuations.proposals import CheckpointProposalSpec
from exp.optimize.evaluation.continuations.selection import load_selection_plan, selection_jobs
from exp.optimize.evaluation.equivalence.rubric import RubricDerivationSpec
from exp.optimize.evaluation.equivalence.runner import EquivalenceSpec
from exp.optimize.evaluation.executable import grounded as grounded_runtime
from exp.optimize.evaluation.executable import induced as induced_runtime
from exp.optimize.evaluation.executable import mixed as mixed_runtime
from exp.optimize.evaluation.executable.model_costs import (
    CostComponent,
    decision_cost,
    model_cost,
)
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.factory.candidates import CandidateBatchSpec
from exp.optimize.evaluation.factory.hardening import (
    TaskHardeningSpec,
    hardening_panel,
    seed_proposals,
)
from exp.optimize.evaluation.factory.proposals import load_task_proposals
from exp.optimize.evaluation.factory.quality_proxy import proxy_settings
from exp.optimize.evaluation.factory.runner import TaskFactorySpec
from exp.optimize.evaluation.factory.validity import TaskValiditySpec, ValiditySettings
from exp.optimize.evaluation.factory.variants import load_task_variant
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.hillclimb.proposals import EditProposalSpec
from exp.optimize.evaluation.hillclimb.report import ClimbReportSpec, report_jobs
from exp.optimize.evaluation.hillclimb.rounds import ClimbRound, round_jobs
from exp.optimize.evaluation.induction.service import InductionSpec
from exp.optimize.evaluation.tasks.challenge_runs import ChallengeRunSpec, challenge_attempts
from exp.runtime.agents.cascade import DeciderDefinition
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

# Every paid payload with a declared cost inventory; evaluation_costs refuses anything else.
CostedSpec = (
    ContinuationJob
    | AuthorSettings
    | CriticSpec
    | MemoryAnalystSpec
    | CheckpointProposalSpec
    | ActorDecisionSpec
    | CaptureDefinition
    | ActionInterventionSpec
    | FramingStudySpec
    | ActionContrastSpec
    | ObservationContrastSpec
    | ActorPanelSpec
    | CheckpointExperimentSpec
    | ContinuationCampaignSpec
    | RubricDerivationSpec
    | EquivalenceSpec
    | CandidateBatchSpec
    | TaskFactorySpec
    | ChallengeRunSpec
    | ClimbRound
    | EditProposalSpec
    | ClimbReportSpec
    | TaskValiditySpec
    | TaskHardeningSpec
    | InductionSpec
    | CascadeFitSpec
    | CascadeConfirmationSpec
)


def runtime_costs(
    project: ProjectStore, digest: Sha256, configuration: JsonObject
) -> tuple[CostComponent, ...]:
    """Recognize maintained factory bytes without executing a caller's runtime module."""
    if digest == sha256_bytes(Path(__file__).with_name("workbook.py").read_bytes()):
        # Generic cost review must not import optional workbook dependencies; only the
        # selected maintained owner needs its typed model.
        from exp.optimize.evaluation.executable.workbook import WorkbookRuntimeDefinition

        WorkbookRuntimeDefinition.model_validate(configuration)
        return ()
    if digest == sha256_bytes(Path(mixed_runtime.__file__).read_bytes()):
        mixed = mixed_runtime.MixedRuntimeDefinition.model_validate(configuration)
        return (
            *runtime_costs(project, mixed.native.runtime_sha256, mixed.native.configuration),
            *_grounded_costs(project, mixed.simulated),
        )
    if digest == sha256_bytes(Path(induced_runtime.__file__).read_bytes()):
        # Induced calls make no provider request; the predictor's allowance is the upper bound.
        return _grounded_costs(
            project, induced_runtime.InducedRuntimeDefinition.model_validate(configuration).grounded
        )
    if digest == sha256_bytes(Path(grounded_runtime.__file__).read_bytes()):
        return _grounded_costs(
            project, grounded_runtime.GroundedRuntimeDefinition.model_validate(configuration)
        )
    return (CostComponent("trusted custom runtime: provider allowance undeclared", 0, None),)


def _grounded_costs(
    project: ProjectStore, spec: grounded_runtime.GroundedRuntimeDefinition
) -> tuple[CostComponent, ...]:
    """Include predictor, query embedding and every enabled recall/reader route."""
    catalog = load_model_catalog(project.model_catalog_path)
    world = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    components = [
        model_cost(
            project, world.model_alias, spec.maximum_physical_attempts, spec.maximum_output_tokens
        )
    ]
    embedder = index_embedder_alias(project, catalog, world)
    if embedder is not None:
        components.append(model_cost(project, embedder, spec.maximum_physical_attempts, 0))
    if spec.memory is not None:
        components.append(
            decision_cost(project, spec.memory.recall.jev_alias, spec.maximum_physical_attempts)
        )
    if spec.prediction_reading is not None:
        components.append(
            decision_cost(
                project, spec.prediction_reading.reader_alias, spec.maximum_physical_attempts
            )
        )
    # All roles share one ledger. Summing each role at that ceiling overestimates the mixture,
    # but cannot omit an expensive role or assume a favorable allocation of requests.
    return tuple(components)


def job_costs(project: ProjectStore, job: ContinuationJob) -> tuple[CostComponent, ...]:
    """Include actor attempts, any cascade decider and the runtime's lifetime allowance."""
    return (
        model_cost(
            project, job.actor_alias, job.maximum_physical_attempts, job.study.maximum_output_tokens
        ),
        *decider_costs(
            project,
            None if job.study.cascade is None else job.study.cascade.decider,
            job.maximum_physical_attempts,
        ),
        *runtime_costs(project, job.runtime_sha256, job.configuration),
    )


def decider_costs(
    project: ProjectStore, decider: DeciderDefinition | None, attempts: int
) -> tuple[CostComponent, ...]:
    """Price a cascade decider at the attempt ceiling its decisions draw from.

    In an episode the actor and its decider share one ledger, so pricing each at the whole
    ceiling overestimates the mixture but cannot assume a favorable split of the attempts.
    """
    if decider is None:
        return ()
    if decider.kind == "systemone":
        return (decision_cost(project, decider.alias, attempts),)
    return (model_cost(project, decider.alias, attempts, decider.maximum_output_tokens),)


def evaluation_costs(project: ProjectStore, spec: CostedSpec) -> tuple[CostComponent, ...]:
    """Expand maintained command contracts into complete bounded role inventories."""
    if isinstance(spec, ContinuationJob):
        return job_costs(project, spec)
    if isinstance(spec, AuthorSettings):
        return (
            model_cost(
                project, spec.model_alias, spec.maximum_physical_attempts, spec.output_tokens
            ),
        )
    if isinstance(spec, CriticSpec):
        return (
            model_cost(
                project,
                spec.critic_alias,
                spec.maximum_physical_attempts,
                spec.critic_output_tokens,
            ),
        )
    if isinstance(spec, InductionSpec):
        return (
            model_cost(
                project,
                spec.inducer_alias,
                spec.maximum_physical_attempts,
                spec.settings.maximum_output_tokens,
            ),
        )
    if isinstance(spec, MemoryAnalystSpec):
        return (
            model_cost(
                project, spec.analyst_alias, spec.maximum_physical_attempts, spec.output_tokens
            ),
        )
    if isinstance(spec, CheckpointProposalSpec):
        return (model_cost(project, spec.selector_alias, 1, spec.output_tokens),)
    if isinstance(spec, ActorDecisionSpec):
        return tuple(
            model_cost(
                project, spec.actor_alias, 3 * spec.repeats, case.request.maximum_output_tokens or 0
            )
            for case in spec.cases
        )
    if isinstance(spec, CaptureDefinition):
        return runtime_costs(project, spec.runtime_sha256, spec.configuration)
    if isinstance(spec, ActionInterventionSpec):
        return runtime_costs(project, spec.base.runtime_sha256, spec.base.configuration)
    if isinstance(spec, FramingStudySpec):
        return tuple(row.scaled(2 * spec.pairs) for row in job_costs(project, spec.base))
    if isinstance(spec, ActionContrastSpec):
        # Imposed actions can themselves invoke the simulator, before the suffix job begins.
        return tuple(
            row.scaled(2 * spec.base.study.repeats)
            for row in (
                *job_costs(project, spec.base),
                *runtime_costs(project, spec.base.runtime_sha256, spec.base.configuration),
            )
        )
    if isinstance(spec, ObservationContrastSpec):
        effect = ActionIntervention.model_validate_json(
            project.artifacts.read_bytes(spec.intervention.artifact_id, "intervention.json")
        )
        return tuple(row.scaled(2 * spec.pairs) for row in job_costs(project, effect.spec.base))
    if isinstance(spec, ActorPanelSpec):
        return tuple(
            row for cell in panel_jobs(spec, "estimate") for row in job_costs(project, cell.job)
        )
    if isinstance(spec, CascadeFitSpec):
        return decider_costs(project, spec.decider, spec.maximum_physical_attempts)
    if isinstance(spec, CascadeConfirmationSpec):
        return evaluation_costs(project, confirmation_panel(project, spec)[1])
    if isinstance(spec, CheckpointExperimentSpec):
        plan = load_selection_plan(project, spec.selection)
        revision = project.artifacts.read(spec.selection.artifact_id).manifest.code_revision
        jobs = selection_jobs(plan, code_revision=revision, plan_reference=spec.selection)
        return tuple(
            row
            for _, job in jobs
            for row in (
                *job_costs(project, job),
                *(evaluation_costs(project, spec.critic) if spec.critic is not None else ()),
            )
        )
    if isinstance(spec, ContinuationCampaignSpec):
        count = 2 * spec.points * len(spec.selectors)
        suffix = spec.initial.model_copy(
            update={"maximum_physical_attempts": spec.continuation_physical_attempts}
        )
        return (
            *job_costs(project, spec.initial),
            *(model_cost(project, role.alias, 1, role.output_tokens) for role in spec.selectors),
            *(
                row.scaled(count)
                for critic in spec.critics
                for row in (*job_costs(project, suffix), *evaluation_costs(project, critic))
            ),
        )
    if isinstance(spec, RubricDerivationSpec):
        return (
            model_cost(
                project, spec.proposer.alias, spec.maximum_attempts, spec.proposer.output_tokens
            ),
        )
    if isinstance(spec, EquivalenceSpec):
        return (
            *(
                row
                for case in spec.cases
                for job in (case.reference, case.reference, case.candidate)
                for row in job_costs(project, job)
            ),
            *(
                model_cost(
                    project, reviewer.alias, spec.maximum_review_attempts, reviewer.output_tokens
                )
                for reviewer in (spec.judge, *spec.analysts)
            ),
        )
    if isinstance(spec, CandidateBatchSpec):
        proposals = load_task_proposals(project, spec.proposals)
        jobs = tuple(
            load_executable_task(project, load_task_variant(project, ref).task).job
            for ref in proposals.variants
        )
        return (
            *tuple(row for job in jobs for row in job_costs(project, job)),
            *evaluation_costs(project, ActorPanelSpec(cases=jobs, actors=spec.actors)),
            *(
                model_cost(
                    project,
                    settings.model_alias,
                    settings.maximum_physical_attempts,
                    settings.output_tokens,
                )
                for ref in proposals.variants
                for settings in (
                    spec.proxy_for(
                        load_executable_task(project, load_task_variant(project, ref).task)
                    ),
                )
            ),
        )
    if isinstance(spec, TaskFactorySpec):
        return tuple(
            row
            for family in spec.families
            for settings in (
                proxy_settings(
                    family.quality_proxy,
                    load_executable_task(project, family.proposal.base),
                    spec.actors[-1],
                ),
            )
            for row in (
                *evaluation_costs(project, family.proposal),
                model_cost(
                    project,
                    settings.model_alias,
                    4 * settings.maximum_physical_attempts,
                    settings.output_tokens,
                ),
                *(
                    line.scaled(4)
                    for line in job_costs(
                        project, load_executable_task(project, family.proposal.base).job
                    )
                ),
                *(
                    line.scaled(4)
                    for line in evaluation_costs(
                        project,
                        ActorPanelSpec(
                            cases=(load_executable_task(project, family.proposal.base).job,),
                            actors=spec.actors,
                        ),
                    )
                ),
            )
        )
    if isinstance(spec, ChallengeRunSpec):
        return tuple(
            model_cost(project, alias, attempts, spec.output_tokens)
            for alias, attempts in challenge_attempts(spec).items()
        )
    if isinstance(spec, ClimbRound):
        # Both stages, both arms: confirmation is priced although screening may stop the round.
        return tuple(row for job in round_jobs(project, spec) for row in job_costs(project, job))
    if isinstance(spec, ClimbReportSpec):
        return tuple(row for job in report_jobs(project, spec) for row in job_costs(project, job))
    if isinstance(spec, EditProposalSpec):
        return (
            model_cost(
                project, spec.analyst_alias, spec.maximum_physical_attempts, spec.output_tokens
            ),
        )
    if isinstance(spec, TaskValiditySpec):
        return _validity_costs(project, spec)
    if isinstance(spec, TaskHardeningSpec):
        # Level zero checks one seed candidate; each harder level buys one author draw and
        # may qualify all four proposed worlds, each priced at the seed variant's actor job.
        seed = seed_proposals(project, spec)
        candidate = (
            *_validity_costs(project, spec.validity),
            *evaluation_costs(project, hardening_panel(project, spec)),
        )
        level = (
            *evaluation_costs(project, spec.author),
            *runtime_costs(
                project,
                seed.spec.template.capture.runtime_sha256,
                seed.spec.template.capture.configuration,
            ),
            *(row.scaled(4) for row in candidate),
        )
        return (*candidate, *(row.scaled(spec.maximum_levels) for row in level))
    raise ValueError(f"no cost inventory is defined for {type(spec).__name__}")


def _validity_costs(project: ProjectStore, spec: ValiditySettings) -> tuple[CostComponent, ...]:
    """One leak-judge draw plus every registered closed-book attempt."""
    return (
        *evaluation_costs(project, spec.leak_judge),
        *(
            row.scaled(spec.closed_book_attempts)
            for row in evaluation_costs(project, spec.closed_book)
        ),
    )
