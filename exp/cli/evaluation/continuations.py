"""Bounded continuation execution and credential-free witnessed SFT export."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from rich.console import Console

from exp.cli.evaluation.equivalence import run_equivalence_command
from exp.cli.evaluation.executable import ExecutableCommand, run_executable_command
from exp.cli.evaluation.spend import review_cost
from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ArtifactStoreError, ProjectStore
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.analysis.actor_decisions import (
    ActorDecisionSpec,
    actor_decision_interfaces,
    actor_decisions_need_dispatch,
    preflight_actor_decisions,
    run_actor_decisions,
    saved_actor_decisions,
)
from exp.optimize.evaluation.analysis.decision_calibration import (
    DecisionCalibration,
    calibrate_decisions,
)
from exp.optimize.evaluation.analysis.failures import analyse_failures, persist_failure_analysis
from exp.optimize.evaluation.analysis.memory_analyst import (
    MemoryAnalystSpec,
    preflight_memory_analyst,
    run_memory_analysis,
    saved_memory_analysis,
)
from exp.optimize.evaluation.analysis.memory_evidence import MemoryInspection, inspect_memory
from exp.optimize.evaluation.analysis.research import (
    FailureResearchSpec,
    check_failure_research,
    persist_failure_research,
)
from exp.optimize.evaluation.continuations.action_contrast import (
    ActionContrastSpec,
    preflight_action_contrast,
    run_action_contrast,
    saved_action_contrast,
)
from exp.optimize.evaluation.continuations.actors import (
    ActorPanelSpec,
    preflight_actor_panel,
    run_actor_panel,
    saved_actor_panel,
)
from exp.optimize.evaluation.continuations.admission import (
    ContinuationAdmission,
    admit_continuations,
)
from exp.optimize.evaluation.continuations.campaign import (
    ContinuationCampaignSpec,
    campaign_requires_execution,
    preflight_continuation_campaign,
    run_continuation_campaign,
    saved_continuation_campaign,
)
from exp.optimize.evaluation.continuations.capture import (
    CaptureDefinition,
    capture_initial_world,
    preflight_capture,
    saved_capture,
)
from exp.optimize.evaluation.continuations.critic import (
    CriticSpec,
    load_critic_prediction,
    predict_success,
    preflight_critic,
)
from exp.optimize.evaluation.continuations.experiment import (
    CheckpointExperimentSpec,
    preflight_checkpoint_experiment,
    run_checkpoint_experiment,
    saved_checkpoint_experiment,
)
from exp.optimize.evaluation.continuations.framing import (
    FramingStudySpec,
    run_framing_study,
    saved_framing_study,
)
from exp.optimize.evaluation.continuations.inference import campaign_summary, experiment_summary
from exp.optimize.evaluation.continuations.intervention import (
    ActionInterventionSpec,
    capture_action_intervention,
    preflight_intervention,
    saved_intervention,
)
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.observation_contrast import (
    ObservationContrastSpec,
    preflight_observation_contrast,
    run_observation_contrast,
    saved_observation_contrast,
)
from exp.optimize.evaluation.continuations.proposals import (
    CheckpointProposalSpec,
    propose_checkpoints,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.selection import (
    CheckpointSelectionSpec,
    load_selection_plan,
    match_checkpoints,
    persist_selection_plan,
    selection_jobs,
)
from exp.optimize.evaluation.equivalence.prefixes import (
    ActorPrefixSpec,
    admit_actor_prefix,
    persist_actor_prefix,
)
from exp.optimize.evaluation.equivalence.runner import EquivalenceSpec
from exp.optimize.evaluation.executable.scenarios import (
    RecordScenarioSpec,
    RecordWitnessSpec,
    prepare_record_scenario,
    prepare_record_witness,
    saved_record_scenario,
)
from exp.optimize.evaluation.factory.candidates import (
    CandidateBatchSpec,
    preflight_candidate_batch,
    run_candidate_batch,
    saved_candidate_batch,
)
from exp.optimize.evaluation.factory.proposals import (
    TaskProposalSpec,
    preflight_task_proposals,
    proposal_request,
    propose_tasks,
    saved_task_proposals,
)
from exp.optimize.evaluation.factory.scenario_search import (
    ScenarioSearchSpec,
    preflight_scenario_search,
    run_scenario_search,
)
from exp.optimize.evaluation.factory.variants import (
    TaskVariantSpec,
    load_task_variant,
    preflight_task_variant,
    publish_task_variant,
)
from exp.optimize.evaluation.learning.repair import LearningRepairSpec, repair_learning
from exp.optimize.model.sft.builder import build_sft_dataset, write_sft_dataset
from exp.optimize.model.sft.contracts import SFTBuildSpec
from exp.simulation.capability_briefs import (
    CapabilityBrief,
    persist_capability_brief,
    validate_brief_sources,
)


class ContinuationExport(ContractModel):
    """Saved development studies to independently re-grade and export without model calls.

    Attributes:
        studies: Exact complete results; every branch remains in their admission reports.
        dataset: Existing SFT split controls. Related continuations share one leakage group.
    """

    studies: tuple[ArtifactInput, ...] = Field(min_length=1)
    dataset: SFTBuildSpec = Field(default_factory=SFTBuildSpec)


class ContinuationCommand(ContractModel):
    """One continuation CLI operation with an explicit typed payload.

    Attributes:
        action: Exactly one selected operation; unrelated payloads are refused.
        analysis: Complete saved study for credential-free failure localization.
        observations: Saved-reply substitution with an identical world and equal suffix budgets.
        variant: Authored drill, rephrasing or executed state variation for make-variant.
        contrast: Paired imposed decisions and equal-budget suffixes for compare-actions.
        proposals: Bounded public-context prompt candidates for propose-tasks.
        batch: Complete candidate measurement and optional packaging for evaluate-candidates.
        brief: Source-linked need and assumptions for register-brief.
        intervention: Scripted tool decision for intervene; actor inference allowance stays fixed.
        selection: Proposed checkpoints and matching protocol for match-checkpoints.
        experiment: Registered selection and optional critic for run-selection.
        critic: Prospective estimate settings for predict-success.
        actors: Frozen multi-model comparison for compare-actors.
        capture: Initial actor/world capture definition.
        job: One resumed continuation study.
        export: Independently admitted studies and dataset split settings.
        framing: Two authored incidental cues and their shared base job.
        harbor: Executable task registration, qualification, export or serving.
        scenario: Native setup and public initialization for prepare-scenario.
        scenario_search: Bounded native constructions and discriminating control scripts.
        record_witness: Explicit native script and independent predicate for witness-record.
        memory: Saved canonical calls to inspect without model calls or memory mutation.
        decisions: Frozen fixed-question calibration inventory, never a serving approval.
        memory_analyst: Opt-in bounded diagnostic using the existing trace-analysis rubric.
        actor_decisions: Trace-only next-decision probes with a repeated-reference control.
        prefix: Extract a complete retained actor input and bind a saved simulator observation.
        research: Check cited counterexample outcomes and retain a provisional research revision.
        checkpoint_proposals: Retrospective point selector producing an existing matched plan.
        campaign: Registered source attempts, selectors, prospective critics and continuations.
        learning_repair: Explicit observation-only settlement of failed learning.
        equivalence: Paired executable reference/candidate studies and diagnostic trace council.
    """

    action: Literal[
        "compare-environments",
        "repair-learning",
        "admit-actor-prefix",
        "check-failure-research",
        "propose-checkpoints",
        "run-continuation-campaign",
        "probe-actor-decisions",
        "inspect-memory",
        "measure-decisions",
        "analyse-memory",
        "analyse-failures",
        "compare-observations",
        "capture",
        "run",
        "export",
        "compare-framing",
        "compare-actors",
        "predict-success",
        "harbor",
        "make-variant",
        "intervene",
        "match-checkpoints",
        "run-selection",
        "compare-actions",
        "propose-tasks",
        "evaluate-candidates",
        "register-brief",
        "prepare-scenario",
        "search-scenarios",
        "witness-record",
    ]
    analysis: ArtifactInput | None = None
    observations: ObservationContrastSpec | None = None
    variant: TaskVariantSpec | None = None
    contrast: ActionContrastSpec | None = None
    proposals: TaskProposalSpec | None = None
    batch: CandidateBatchSpec | None = None
    brief: CapabilityBrief | None = None
    intervention: ActionInterventionSpec | None = None
    selection: CheckpointSelectionSpec | None = None
    experiment: CheckpointExperimentSpec | None = None
    critic: CriticSpec | None = None
    actors: ActorPanelSpec | None = None
    capture: CaptureDefinition | None = None
    job: ContinuationJob | None = None
    export: ContinuationExport | None = None
    framing: FramingStudySpec | None = None
    harbor: ExecutableCommand | None = None
    scenario: RecordScenarioSpec | None = None
    scenario_search: ScenarioSearchSpec | None = None
    record_witness: RecordWitnessSpec | None = None
    memory: MemoryInspection | None = None
    decisions: DecisionCalibration | None = None
    memory_analyst: MemoryAnalystSpec | None = None
    actor_decisions: ActorDecisionSpec | None = None
    prefix: ActorPrefixSpec | None = None
    research: FailureResearchSpec | None = None
    checkpoint_proposals: CheckpointProposalSpec | None = None
    campaign: ContinuationCampaignSpec | None = None
    learning_repair: LearningRepairSpec | None = None
    equivalence: EquivalenceSpec | None = None

    @model_validator(mode="after")
    def exact_payload(self) -> "ContinuationCommand":
        """Reject unrelated payloads before any operation, including dry runs and replay."""
        fields = {
            "compare-environments": "equivalence",
            "repair-learning": "learning_repair",
            "admit-actor-prefix": "prefix",
            "check-failure-research": "research",
            "propose-checkpoints": "checkpoint_proposals",
            "run-continuation-campaign": "campaign",
            "probe-actor-decisions": "actor_decisions",
            "inspect-memory": "memory",
            "measure-decisions": "decisions",
            "analyse-memory": "memory_analyst",
            "prepare-scenario": "scenario",
            "search-scenarios": "scenario_search",
            "witness-record": "record_witness",
            "analyse-failures": "analysis",
            "compare-observations": "observations",
            "register-brief": "brief",
            "evaluate-candidates": "batch",
            "propose-tasks": "proposals",
            "compare-actions": "contrast",
            "make-variant": "variant",
            "intervene": "intervention",
            "capture": "capture",
            "run": "job",
            "export": "export",
            "compare-framing": "framing",
            "compare-actors": "actors",
            "predict-success": "critic",
            "harbor": "harbor",
            "match-checkpoints": "selection",
            "run-selection": "experiment",
        }
        selected = fields[self.action]
        if any(
            (getattr(self, field) is not None) != (field == selected) for field in fields.values()
        ):
            raise ValueError(f"{self.action} requires only its {selected} payload")
        return self


def run_continuations(
    project: str, spec_path: Path, *, root: Path, yes: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Run one bounded study or export its independently verified development targets."""
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError, ArtifactStoreError):
        command = ContinuationCommand.model_validate_json(spec_path.read_bytes())
        store = ProjectStore(root, project)
        revision = installed_execution_revision()
        if command.equivalence is not None:
            run_equivalence_command(
                store,
                command.equivalence,
                console=console,
                revision=revision,
                root=root,
                project=project,
                spec_path=spec_path,
                yes=yes,
                non_interactive=non_interactive,
                dry_run=dry_run,
            )
            return
        if command.learning_repair is not None:
            result = repair_learning(
                store, command.learning_repair, code_revision=revision, dry_run=dry_run
            )
            console.print_json(data=result)
            return
        if command.campaign is not None:
            campaign = command.campaign
            completed = saved_continuation_campaign(store, campaign, code_revision=revision)
            needs_execution = campaign_requires_execution(store, campaign, code_revision=revision)
            if needs_execution:
                preflight_continuation_campaign(store, campaign)
            console.print(
                f"Continuation campaign: {campaign.initial.study.repeats} initial attempts; "
                f"{len(campaign.selectors)} selectors, {len(campaign.critics)} critics; "
                f"up to {campaign.points} points each, {campaign.continuations} continuations "
                f"per arm. Actor/selector/critic HTTP ceiling: {campaign.attempt_bound()}. "
                "Simulator calls retain their separate pinned ceiling. No training."
            )
            if dry_run:
                console.print(
                    "Known profiles validated; future evidence packets are checked whole."
                )
                return
            if needs_execution and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, campaign),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            result = completed or run_continuation_campaign(store, campaign, code_revision=revision)
            console.print_json(data=campaign_summary(store, result))
            return
        if command.checkpoint_proposals is not None:
            selector = command.checkpoint_proposals
            propose_checkpoints(store, selector, code_revision=revision, dry_run=True)
            console.print("Retrospective checkpoint selector: at most one new provider attempt.")
            if dry_run:
                console.print("Complete source inventory validated; no model calls or publication.")
                return
            has_response = store.artifacts.exists(selector.identity(revision) + "-response")
            if not has_response and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, selector),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            selected = propose_checkpoints(store, selector, code_revision=revision)
            assert selected is not None
            console.print_json(data=selected.model_dump(mode="json"))
            return
        if command.prefix is not None:
            case = admit_actor_prefix(store.artifacts, command.prefix)
            console.print_json(data=case.model_dump(mode="json"))
            if not dry_run:
                reference = persist_actor_prefix(
                    store.artifacts,
                    command.prefix,
                    created_at=datetime.now(UTC),
                    code_revision=revision,
                )
                console.print_json(data={"saved_prefix": reference.model_dump(mode="json")})
            return
        if command.research is not None:
            research_report = check_failure_research(store.artifacts, command.research)
            console.print_json(data=research_report.model_dump(mode="json"))
            if not dry_run:
                reference = persist_failure_research(
                    store.artifacts,
                    command.research,
                    created_at=datetime.now(UTC),
                    code_revision=revision,
                )
                console.print_json(data={"saved_research": reference.model_dump(mode="json")})
            return
        if command.actor_decisions is not None:
            decision_spec = command.actor_decisions
            saved_decisions = saved_actor_decisions(store, decision_spec, code_revision=revision)
            needs_dispatch = saved_decisions is None and actor_decisions_need_dispatch(
                store, decision_spec, code_revision=revision
            )
            if needs_dispatch:
                preflight_actor_decisions(store, decision_spec)
            console.print(
                f"Actor next-decision probes: {len(decision_spec.cases)} cases, "
                f"{decision_spec.repeats} triples each; maximum physical attempts: "
                f"{len(decision_spec.cases) * decision_spec.repeats * 3}. "
                "No tools executed; development sensitivity, not full-episode equivalence."
            )
            console.print_json(
                data={"interface_diagnostics": actor_decision_interfaces(decision_spec)}
            )
            if dry_run:
                console.print("Complete requests and sources validated; no provider calls.")
                return
            if needs_dispatch and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, decision_spec),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            decisions_report = saved_decisions or run_actor_decisions(
                store, decision_spec, code_revision=revision
            )
            console.print_json(data=decisions_report.metrics())
            console.print(f"Saved decisions: {decision_spec.identity(revision)}-result")
            return
        if command.memory_analyst is not None:
            analyst = command.memory_analyst
            saved_analysis = saved_memory_analysis(store, analyst, code_revision=revision)
            if saved_analysis is None:
                preflight_memory_analyst(store, analyst)
            if dry_run:
                console.print("Memory diagnostic sources and profile verified; no model call.")
                return
            if saved_analysis is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, analyst),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            diagnostic = saved_analysis or run_memory_analysis(
                store, analyst, code_revision=revision
            )
            console.print_json(
                data={
                    "calibration": "unqualified diagnostic",
                    "analysis": diagnostic.model_dump(mode="json"),
                }
            )
            return
        if command.memory is not None:
            report = inspect_memory(store.artifacts, command.memory)
            console.print_json(data=report.model_dump(mode="json"))
            return
        if command.decisions is not None:
            measurement = calibrate_decisions(store.artifacts, command.decisions)
            console.print_json(data=measurement.model_dump(mode="json"))
            return
        if command.record_witness is not None:
            preflight_capture(store, command.record_witness.capture)
            if dry_run:
                console.print(
                    "Record witness inputs parsed; no runtime imported or script executed."
                )
                return
            console.print_json(
                data=prepare_record_witness(store, command.record_witness, code_revision=revision)
            )
            return
        if command.scenario_search is not None:
            search = command.scenario_search
            if dry_run:
                preflight_scenario_search(store, search)
                console.print("Native scenario search pins checked; no runtime executed.")
            else:
                console.print_json(
                    data=run_scenario_search(store, search, code_revision=revision).model_dump(
                        mode="json"
                    )
                )
            return
        if command.scenario is not None:
            scenario = command.scenario
            saved_scenario = saved_record_scenario(store, scenario, code_revision=revision)
            if saved_scenario is None:
                preflight_capture(store, scenario.capture)
            if dry_run:
                console.print(
                    "Native scenario parsed and source pins checked; no runtime imported."
                )
                return
            result = saved_scenario or prepare_record_scenario(
                store, scenario, code_revision=revision
            )
            console.print_json(data=result.model_dump(mode="json"))
            return
        if command.observations is not None:
            spec = command.observations
            saved_observations = saved_observation_contrast(store, spec, code_revision=revision)
            if saved_observations is None:
                attempts = preflight_observation_contrast(store, spec)
            else:
                attempts = sum(
                    cell.job.maximum_physical_attempts for cell in saved_observations.cells
                )
            console.print(
                f"Observation comparison: {spec.pairs * 2} draws; "
                f"maximum actor HTTP attempts: {attempts}; original world retained."
            )
            console.print(
                "Recorded reply retains source authority; authored alternative is unverified."
            )
            if dry_run:
                console.print(
                    "Sources and suffix policy validated without runtime imports or calls."
                )
                return
            if saved_observations is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, spec),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            report = saved_observations or run_observation_contrast(
                store, spec, code_revision=revision
            )
            console.print_json(data={"scope": report.scope, **report.metrics()})
            return
        if command.analysis is not None:
            report = analyse_failures(store.artifacts, command.analysis)
            if dry_run:
                console.print(
                    "Source inventory and rewind boundaries verified; no report published."
                )
                return
            reference = persist_failure_analysis(
                store.artifacts,
                command.analysis,
                created_at=datetime.now(UTC),
                code_revision=revision,
            )
            console.print_json(
                data={
                    "reference": reference.model_dump(mode="json"),
                    **report.model_dump(mode="json"),
                }
            )
            return
        if command.brief is not None:
            validate_brief_sources(store.artifacts, command.brief)
            if dry_run:
                console.print(
                    "Brief sources validated; assumptions retain their declared authority."
                )
                return
            reference = persist_capability_brief(
                store.artifacts, command.brief, created_at=datetime.now(UTC), code_revision=revision
            )
            console.print_json(
                data={
                    "reference": reference.model_dump(mode="json"),
                    "qualification_blockers": command.brief.blockers(),
                }
            )
            return
        if command.batch is not None:
            batch = command.batch
            saved_batch = saved_candidate_batch(store, batch, code_revision=revision)
            if saved_batch is None:
                preflight_candidate_batch(store, batch, code_revision=revision)
            if dry_run:
                console.print("Every candidate and model validated; no calls or packages created.")
                return
            if saved_batch is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, batch),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            batch_result = saved_batch or run_candidate_batch(store, batch, code_revision=revision)
            console.print_json(data=batch_result.model_dump(mode="json"))
            return
        if command.proposals is not None:
            proposals = command.proposals
            saved_proposals = saved_task_proposals(store, proposals, code_revision=revision)
            response_saved = store.artifacts.exists(proposals.identity(revision) + "-response")
            if saved_proposals is None and not response_saved:
                preflight_task_proposals(store, proposals)
            elif saved_proposals is None:
                proposal_request(store, proposals)
            if dry_run:
                console.print("Public authoring context validated; no proposals purchased.")
                return
            if (
                saved_proposals is None
                and not response_saved
                and not require_spend_consent(
                    console,
                    root=root,
                    yes=yes,
                    estimated_cost_usd=review_cost(console, store, proposals),
                    cost_is_upper_bound=False,
                    command=f"exp eval {project} --continuations {spec_path}",
                    non_interactive=non_interactive,
                )
            ):
                return
            proposal_result = saved_proposals or propose_tasks(
                store, proposals, code_revision=revision
            )
            console.print_json(data=proposal_result.model_dump(mode="json"))
            return
        if command.contrast is not None:
            contrast = command.contrast
            saved_contrast = saved_action_contrast(store, contrast, code_revision=revision)
            if saved_contrast is None:
                preflight_action_contrast(store, contrast, code_revision=revision)
            if dry_run:
                console.print("Paired decisions validated; no tools or actor calls executed.")
                return
            if saved_contrast is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, contrast),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            contrast_result = saved_contrast or run_action_contrast(
                store, contrast, code_revision=revision
            )
            console.print_json(data=contrast_result.model_dump(mode="json"))
            return
        if command.variant is not None:
            preflight_task_variant(store, command.variant)
            if dry_run:
                console.print(
                    "Variant sources and context validated; no artifact or world created."
                )
                return
            reference = publish_task_variant(
                store, command.variant, created_at=datetime.now(UTC), code_revision=revision
            )
            variant = load_task_variant(store, reference)
            console.print_json(
                data={
                    "reference": reference.model_dump(mode="json"),
                    **variant.model_dump(mode="json"),
                }
            )
            return
        if command.intervention is not None:
            intervention = command.intervention
            saved_action = saved_intervention(store, intervention, code_revision=revision)
            if saved_action is None:
                preflight_intervention(store, intervention)
            if dry_run:
                console.print("Intervention validated; no runtime imported or tool executed.")
                return
            if saved_action is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, intervention),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            action_result = saved_action or capture_action_intervention(
                store, intervention, code_revision=revision
            )
            console.print_json(data=action_result.model_dump(mode="json"))
            return
        if command.selection is not None:
            selection = command.selection
            identity = selection.identity(revision)
            if store.artifacts.exists(identity):
                from exp.common.project import artifact_input

                plan = load_selection_plan(
                    store, artifact_input(store.artifacts.read(identity).manifest)
                )
            else:
                plan = match_checkpoints(store, selection)
            if dry_run:
                console.print_json(data=plan.model_dump(mode="json"))
                return
            reference = persist_selection_plan(
                store, selection, created_at=datetime.now(UTC), code_revision=revision
            )
            console.print_json(
                data={
                    "reference": reference.model_dump(mode="json"),
                    "matches": len(plan.matches),
                    "unmatched": sum(pair.control is None for pair in plan.matches),
                }
            )
            return
        if command.experiment is not None:
            experiment = command.experiment
            saved_experiment = saved_checkpoint_experiment(
                store, experiment, code_revision=revision
            )
            if saved_experiment is None:
                preflight_checkpoint_experiment(store, experiment)
            plan = load_selection_plan(store, experiment.selection)
            plan_revision = store.artifacts.read(
                experiment.selection.artifact_id
            ).manifest.code_revision
            jobs = selection_jobs(
                plan, code_revision=plan_revision, plan_reference=experiment.selection
            )
            attempts = sum(job.maximum_physical_attempts for _, job in jobs)
            if experiment.critic is not None:
                attempts += len(jobs) * experiment.critic.maximum_physical_attempts
            console.print(
                f"Matched cells: {len(jobs)}; maximum actor/critic HTTP attempts: {attempts}"
            )
            if dry_run:
                console.print("Validated without runtime imports or provider calls.")
                return
            if saved_experiment is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, experiment),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            report = saved_experiment or run_checkpoint_experiment(
                store, experiment, code_revision=revision
            )
            console.print_json(data=experiment_summary(store, report))
            return
        if command.critic is not None:
            exists = store.artifacts.exists(command.critic.identity(revision))
            if not exists:
                preflight_critic(store, command.critic)
            console.print(
                f"Critic maximum HTTP attempts: {command.critic.maximum_physical_attempts}"
            )
            if dry_run:
                console.print("Public critic context validated; no provider call.")
                return
            if not exists and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, command.critic),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            reference = predict_success(store, command.critic, code_revision=revision)
            prediction = load_critic_prediction(store.artifacts, reference)
            console.print_json(
                data={
                    "reference": reference.model_dump(mode="json"),
                    "predicted_success": prediction.estimate.probability,
                    "rationale": prediction.estimate.rationale,
                }
            )
            return
        if command.actors is not None:
            saved_panel = saved_actor_panel(store, command.actors, code_revision=revision)
            if saved_panel is None:
                preflight_actor_panel(store, command.actors)
            attempts = sum(job.maximum_physical_attempts for job in command.actors.cases)
            attempts *= len(command.actors.actors)
            console.print(
                f"Actor panel: {len(command.actors.cases)} cases, "
                f"{len(command.actors.actors)} models; "
                f"maximum actor HTTP attempts: {attempts}"
            )
            if dry_run:
                console.print("Validated without runtime imports or provider calls.")
                return
            if saved_panel is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, command.actors),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            panel = saved_panel or run_actor_panel(store, command.actors, code_revision=revision)
            console.print_json(data={"scope": panel.scope, **panel.metrics})
            return
        if command.capture is not None:
            saved = saved_capture(store, command.capture, code_revision=revision)
            if saved is None:
                preflight_capture(store, command.capture)
            if dry_run:
                console.print(
                    "Capture validated without importing runtime code or opening a world."
                )
                return
            if saved is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, command.capture),
                cost_is_upper_bound=False,
                non_interactive=non_interactive,
                command=f"exp eval {project} --continuations {spec_path}",
            ):
                return
            reference = saved or capture_initial_world(
                store, command.capture, code_revision=revision
            )
            console.print_json(data=reference.model_dump(mode="json"))
            return
        if command.harbor is not None:
            run_executable_command(
                store,
                command.harbor,
                console=console,
                dry_run=dry_run,
                yes=yes,
                non_interactive=non_interactive,
                code_revision=revision,
                spec_path=spec_path,
            )
            return
        if command.framing is not None:
            spec = command.framing
            saved_framing = saved_framing_study(store, spec, code_revision=revision)
            if saved_framing is None:
                preflight_continuation_job(store, spec.base)
            console.print(
                f"Framing cells: {spec.pairs * 2}; maximum HTTP attempts: "
                f"{spec.base.maximum_physical_attempts * spec.pairs * 2}"
            )
            if dry_run:
                console.print("Validated without runtime imports or provider calls.")
                return
            if saved_framing is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, spec),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            report = saved_framing or run_framing_study(store, spec, code_revision=revision)
            console.print_json(data=report.metrics())
            return
        if command.job is not None:
            job = command.job
            saved = saved_continuation_job(store, job, code_revision=revision)
            if saved is None:
                preflight_continuation_job(store, job)
            console.print(
                f"Planned branches: {job.study.repeats}; "
                f"maximum HTTP attempts: {job.maximum_physical_attempts}"
            )
            console.print(f"Trusted local runtime: {job.runtime_file}", markup=False)
            if dry_run:
                console.print("Validated without importing runtime code or calling providers.")
                return
            if saved is None and not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=review_cost(console, store, job),
                cost_is_upper_bound=False,
                command=f"exp eval {project} --continuations {spec_path}",
                non_interactive=non_interactive,
            ):
                return
            result = saved or execute_continuation_job(store, job, code_revision=revision)
            console.print_json(
                data={
                    **result.metrics(),
                    "checkpoints": [
                        {"branch": branch.index, "reference": ref.model_dump(mode="json")}
                        for branch in result.branches
                        for ref in branch.checkpoints
                    ],
                    "reference": continuation_result_reference(
                        store, job, code_revision=revision
                    ).model_dump(mode="json"),
                }
            )
            return
        assert command.export is not None
        for reference in command.export.studies:
            load_study(store.artifacts, reference)
        if dry_run:
            console.print("Study inventories validated. Admission and dataset publication not run.")
            return
        accepted = tuple(
            admit_continuations(
                store.artifacts, reference, created_at=datetime.now(UTC), code_revision=revision
            )
            for reference in command.export.studies
        )
        for reference in accepted:
            report = ContinuationAdmission.model_validate_json(
                store.artifacts.read_bytes(reference.artifact_id, "admission.json")
            )
            console.print(
                f"Admission {reference.artifact_id}: "
                f"{sum(row.admitted for row in report.branches)}/{len(report.branches)} branches"
            )
            console.print(report.scope)
            if report.evidence is None:
                console.print("No task qualification attached; observed-outcome checks only.")
            else:
                console.print(report.evidence.qualification_scope)
                if report.evidence.comparison_scope is not None:
                    console.print(report.evidence.comparison_scope)
        dataset = build_sft_dataset(
            store=store,
            production_sources=(),
            teacher_sources=(),
            continuation_sources=accepted,
            spec=command.export.dataset,
            created_at=datetime.now(UTC),
            code_revision=revision,
        )
        write_sft_dataset(store, dataset)
        console.print(f"Dataset: {dataset.dataset.dataset_id} ({dataset.dataset.status})")
        console.print(
            f"Assistant targets: {len(dataset.rows)}; "
            f"exclusions: {len(dataset.inspection.exclusions)}"
        )
        console.print(
            "Prefix and tool observations are context only. No training or promotion run."
        )
