"""The ``exp eval --continuations`` contract, its action registry and dispatch.

Each action is implemented by the handler module of the workflow that owns it. This module only
validates the spec, resolves the shared invocation context once and dispatches to that handler.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer
from pydantic import model_validator
from rich.console import Console
from rich.markup import escape
from rich.text import Text

from exp.cli.evaluation.agent_memory import handle_learn_agent_memory
from exp.cli.evaluation.analysis import (
    handle_analyse_failures,
    handle_analyse_memory,
    handle_check_failure_research,
    handle_confirm_decision_cascade,
    handle_fit_decision_cascade,
    handle_inspect_memory,
    handle_measure_decisions,
    handle_probe_actor_decisions,
)
from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.continuation_studies import (
    ContinuationExport,
    handle_capture,
    handle_compare_actions,
    handle_compare_actors,
    handle_compare_framing,
    handle_compare_observations,
    handle_export,
    handle_intervene,
    handle_match_checkpoints,
    handle_predict_success,
    handle_propose_checkpoints,
    handle_run,
    handle_run_continuation_campaign,
    handle_run_selection,
)
from exp.cli.evaluation.development import (
    DevelopmentTaskQuery,
    GroundRuntime,
    handle_development_tasks,
    handle_ground_runtime,
)
from exp.cli.evaluation.equivalence import (
    handle_admit_actor_prefix,
    handle_compare_environments,
    handle_derive_equivalence_rubric,
    handle_freeze_equivalence_rubric,
    handle_register_admission_policy,
)
from exp.cli.evaluation.executable import (
    ExecutableCommand,
    handle_harbor,
    handle_prepare_scenario,
    handle_register_specification,
    handle_witness_record,
)
from exp.cli.evaluation.factory import (
    handle_evaluate_candidates,
    handle_make_variant,
    handle_propose_tasks,
    handle_register_brief,
    handle_search_scenarios,
)
from exp.cli.evaluation.hill_climb import (
    handle_climb_report,
    handle_climb_round,
    handle_propose_edits,
)
from exp.cli.evaluation.induction import handle_induce_transitions, handle_validate_transitions
from exp.cli.evaluation.instance_facts import handle_derive_facts
from exp.cli.evaluation.judge_challenges import handle_challenge_judges
from exp.cli.evaluation.learning_repair import handle_repair_learning
from exp.cli.evaluation.memory_forget import handle_forget_summary
from exp.cli.evaluation.operator_inputs import (
    ArtifactReferenceQuery,
    ModelSnapshotQuery,
    NativeWorkbookPayload,
    RubricTrajectoryRegistration,
    handle_artifact_reference,
    handle_model_snapshots,
    handle_native_workbook,
    handle_register_rubric_trajectory,
)
from exp.cli.evaluation.sample_count import handle_sample_count
from exp.cli.evaluation.state_retention import handle_collect_garbage, handle_retire_checkpoint
from exp.cli.evaluation.task_hardening import handle_check_task_validity, handle_harden_tasks
from exp.cli.evaluation.trajectory_traces import handle_export_traces
from exp.cli.shared.options import usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import ArtifactInput, ContractModel, stable_id
from exp.common.project import ArtifactStoreError, ProjectStore
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.analysis.actor_decisions import ActorDecisionSpec
from exp.optimize.evaluation.analysis.cascade_confirmation import CascadeConfirmationSpec
from exp.optimize.evaluation.analysis.decision_calibration import DecisionCalibration
from exp.optimize.evaluation.analysis.decision_cascade import CascadeFitSpec
from exp.optimize.evaluation.analysis.memory_analyst import MemoryAnalystSpec
from exp.optimize.evaluation.analysis.memory_evidence import MemoryInspection
from exp.optimize.evaluation.analysis.research import FailureResearchSpec
from exp.optimize.evaluation.continuations.action_contrast import ActionContrastSpec
from exp.optimize.evaluation.continuations.actors import ActorPanelSpec
from exp.optimize.evaluation.continuations.campaign import ContinuationCampaignSpec
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.continuations.critic import CriticSpec
from exp.optimize.evaluation.continuations.experiment import CheckpointExperimentSpec
from exp.optimize.evaluation.continuations.framing import FramingStudySpec
from exp.optimize.evaluation.continuations.intervention import ActionInterventionSpec
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.observation_contrast import ObservationContrastSpec
from exp.optimize.evaluation.continuations.proposals import CheckpointProposalSpec
from exp.optimize.evaluation.continuations.selection import CheckpointSelectionSpec
from exp.optimize.evaluation.continuations.traces import TrajectoryTraceExport
from exp.optimize.evaluation.equivalence.policy import AdmissionPolicy
from exp.optimize.evaluation.equivalence.prefixes import ActorPrefixSpec
from exp.optimize.evaluation.equivalence.rubric import RubricDerivationSpec, RubricReview
from exp.optimize.evaluation.equivalence.runner import EquivalenceSpec
from exp.optimize.evaluation.executable.controls import SpecificationRegistration
from exp.optimize.evaluation.executable.scenarios import RecordScenarioSpec, RecordWitnessSpec
from exp.optimize.evaluation.factory.candidates import CandidateBatchSpec
from exp.optimize.evaluation.factory.hardening import TaskHardeningSpec
from exp.optimize.evaluation.factory.proposals import TaskProposalSpec
from exp.optimize.evaluation.factory.scenario_search import ScenarioSearchSpec
from exp.optimize.evaluation.factory.validity import TaskValiditySpec
from exp.optimize.evaluation.factory.variants import TaskVariantSpec
from exp.optimize.evaluation.fidelity.fact_derivation import FactsDerivation
from exp.optimize.evaluation.hillclimb.proposals import EditProposalSpec
from exp.optimize.evaluation.hillclimb.report import ClimbReportSpec
from exp.optimize.evaluation.hillclimb.rounds import ClimbRound
from exp.optimize.evaluation.induction.service import InductionSpec, TransitionValidationSpec
from exp.optimize.evaluation.learning.agent_memory_contracts import AgentMemorySpec
from exp.optimize.evaluation.learning.forgetting import SummaryForget
from exp.optimize.evaluation.learning.lessons import LearningRepair
from exp.optimize.evaluation.learning.sample_count import SampleCountRequest
from exp.optimize.evaluation.spending import (
    SpendPolicy,
    save_spend_receipt,
)
from exp.optimize.evaluation.state_retention import CheckpointRetirement, StateCollection
from exp.optimize.evaluation.tasks.challenge_runs import ChallengeRunSpec
from exp.runtime.models.spend_ceiling import (
    SpendCeilingReached,
    UnpricedDispatchRefused,
    current_spend_ceiling,
    operation_spend_scope,
)
from exp.simulation.capability_briefs import CapabilityBrief


@dataclass(frozen=True)
class ContinuationAction[Payload]:
    """One ``--continuations`` action, its only payload field and its owning handler.

    Attributes:
        name: Action name accepted in the spec's ``action`` field.
        payload: ``ContinuationCommand`` field that carries this action's payload.
        handler: Owner-module handler that runs the action with the validated payload.
    """

    name: str
    payload: str
    handler: Callable[[ContinuationContext, Payload], None]


CONTINUATION_ACTIONS = (
    ContinuationAction(
        "derive-equivalence-rubric", "rubric_derivation", handle_derive_equivalence_rubric
    ),
    ContinuationAction(
        "freeze-equivalence-rubric", "rubric_review", handle_freeze_equivalence_rubric
    ),
    ContinuationAction("compare-environments", "equivalence", handle_compare_environments),
    ContinuationAction(
        "register-admission-policy", "admission_policy", handle_register_admission_policy
    ),
    ContinuationAction("model-snapshots", "snapshots", handle_model_snapshots),
    ContinuationAction("artifact-reference", "references", handle_artifact_reference),
    ContinuationAction("native-workbook", "native_workbook", handle_native_workbook),
    ContinuationAction(
        "register-rubric-trajectory", "rubric_trajectory", handle_register_rubric_trajectory
    ),
    ContinuationAction("repair-learning", "learning_repair", handle_repair_learning),
    ContinuationAction("learn-agent-memory", "agent_memory", handle_learn_agent_memory),
    ContinuationAction("sample-count", "sample_count", handle_sample_count),
    ContinuationAction("admit-actor-prefix", "prefix", handle_admit_actor_prefix),
    ContinuationAction("check-failure-research", "research", handle_check_failure_research),
    ContinuationAction("propose-checkpoints", "checkpoint_proposals", handle_propose_checkpoints),
    ContinuationAction("run-continuation-campaign", "campaign", handle_run_continuation_campaign),
    ContinuationAction("probe-actor-decisions", "actor_decisions", handle_probe_actor_decisions),
    ContinuationAction("inspect-memory", "memory", handle_inspect_memory),
    ContinuationAction("forget-summary", "forget", handle_forget_summary),
    ContinuationAction("retire-checkpoint", "retirement", handle_retire_checkpoint),
    ContinuationAction("collect-garbage", "collection", handle_collect_garbage),
    ContinuationAction("measure-decisions", "decisions", handle_measure_decisions),
    ContinuationAction("challenge-judges", "challenge", handle_challenge_judges),
    ContinuationAction("fit-decision-cascade", "cascade_fit", handle_fit_decision_cascade),
    ContinuationAction(
        "confirm-decision-cascade", "cascade_confirmation", handle_confirm_decision_cascade
    ),
    ContinuationAction("analyse-memory", "memory_analyst", handle_analyse_memory),
    ContinuationAction("analyse-failures", "analysis", handle_analyse_failures),
    ContinuationAction("compare-observations", "observations", handle_compare_observations),
    ContinuationAction("capture", "capture", handle_capture),
    ContinuationAction("run", "job", handle_run),
    ContinuationAction("export", "export", handle_export),
    ContinuationAction("export-traces", "traces", handle_export_traces),
    ContinuationAction("compare-framing", "framing", handle_compare_framing),
    ContinuationAction("compare-actors", "actors", handle_compare_actors),
    ContinuationAction("predict-success", "critic", handle_predict_success),
    ContinuationAction("harbor", "harbor", handle_harbor),
    ContinuationAction("make-variant", "variant", handle_make_variant),
    ContinuationAction("intervene", "intervention", handle_intervene),
    ContinuationAction("match-checkpoints", "selection", handle_match_checkpoints),
    ContinuationAction("run-selection", "experiment", handle_run_selection),
    ContinuationAction("compare-actions", "contrast", handle_compare_actions),
    ContinuationAction("propose-tasks", "proposals", handle_propose_tasks),
    ContinuationAction("evaluate-candidates", "batch", handle_evaluate_candidates),
    ContinuationAction("register-brief", "brief", handle_register_brief),
    ContinuationAction("prepare-scenario", "scenario", handle_prepare_scenario),
    ContinuationAction("search-scenarios", "scenario_search", handle_search_scenarios),
    ContinuationAction("witness-record", "record_witness", handle_witness_record),
    ContinuationAction("register-specification", "specification", handle_register_specification),
    ContinuationAction("development-tasks", "development_tasks", handle_development_tasks),
    ContinuationAction("ground-runtime", "grounded", handle_ground_runtime),
    ContinuationAction("climb-round", "climb_round", handle_climb_round),
    ContinuationAction("propose-edits", "edit_proposals", handle_propose_edits),
    ContinuationAction("climb-report", "climb_report", handle_climb_report),
    ContinuationAction("check-task-validity", "task_validity", handle_check_task_validity),
    ContinuationAction("harden-tasks", "hardening", handle_harden_tasks),
    ContinuationAction("derive-facts", "facts", handle_derive_facts),
    ContinuationAction("induce-transitions", "induction", handle_induce_transitions),
    ContinuationAction(
        "validate-transitions", "transition_validation", handle_validate_transitions
    ),
)
_ACTIONS_BY_NAME = {action.name: action for action in CONTINUATION_ACTIONS}

# The registry is the one list of action names. Pydantic validates ``action`` against exactly
# that list as a literal, keeping the literal's schema and error message; static checkers see
# the names as data and type the field as ``str``.
if TYPE_CHECKING:
    ContinuationActionName = str
else:
    ContinuationActionName = Literal[tuple(action.name for action in CONTINUATION_ACTIONS)]


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
        traces: Saved trajectory generators to convert into trace imports for export-traces.
        framing: Two authored incidental cues and their shared base job.
        harbor: Executable task registration, qualification, export or serving.
        scenario: Native setup and public initialization for prepare-scenario.
        scenario_search: Bounded native constructions and discriminating control scripts.
        record_witness: Explicit native script and independent predicate for witness-record.
        memory: Saved canonical calls to inspect without model calls or memory mutation.
        forget: One stale memory summary to forget from the next version, with its reason.
        retirement: One unreferenced world checkpoint whose session-history pins to release.
        collection: Session-state collection; with ``--dry-run`` a report of pins and reclaim.
        decisions: Frozen fixed-question calibration inventory, never a serving approval.
        challenge: Registered fixed-output judge challenge study with one model per arm.
        cascade_fit: Development trajectories, decider, grid and objective for a cascade fit.
        cascade_confirmation: Saved cascade fit and held-out cases for closed-loop confirmation.
        memory_analyst: Opt-in bounded diagnostic using the existing trace-analysis rubric.
        actor_decisions: Trace-only next-decision probes with a repeated-reference control.
        prefix: Extract a complete retained actor input and bind a saved simulator observation.
        research: Check cited counterexample outcomes and retain a provisional research revision.
        checkpoint_proposals: Retrospective point selector producing an existing matched plan.
        campaign: Registered source attempts, selectors, prospective critics and continuations.
        learning_repair: Failed-learning settlement, lesson screening or library evolution.
        agent_memory: Outcome-labelled trace export to learn agent memory from, and its folder.
        sample_count: Trajectory-count registration, development selection or confirmation.
        rubric_derivation: Admitted development trajectories and the bounded rubric proposer.
        rubric_review: Explicit acceptance of one saved provisional rubric draft.
        equivalence: Paired executable reference/candidate studies and diagnostic trace council.
        specification: Independent acceptance specification and its source evidence to save.
        development_tasks: Build fit tasks to print, optionally one with its public tools.
        grounded: Simulated world over the build's simulator, emitted as a capture operation.
        climb_round: One prompt edit screened on train and confirmed on test lineages.
        edit_proposals: Analyst edits from train failures and keep bits only, never applied.
        climb_report: Baseline against the best kept prompt on fresh test repeats.
        task_validity: Leak, closed-book and control checks for one qualified variant.
        hardening: Bounded harder-world levels for one qualified seed until a panel separates.
        facts: Backend-derived instance facts for declared task input files.
        induction: Fit-evidence induction of exact per-tool transition programs.
        transition_validation: Provider-free re-validation of saved programs on fit evidence.
        admission_policy: Per-dimension trajectory admission policy to register before results.
        spend: Explicit spend-ceiling choices: a dollar ceiling replacing the displayed
            known-rate estimate and an unpriced-attempt ceiling. Not part of the operation's
            identity, so raising them and resuming continues the same spend ledger.
        snapshots: Catalog aliases whose digest and exact model snapshots to print.
        references: Saved artifact IDs or types whose exact references to print.
        native_workbook: Starting workbooks, task and answer ranges for a native capture.
        rubric_trajectory: Development chronology to register as a rubric source.
    """

    action: ContinuationActionName
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
    traces: TrajectoryTraceExport | None = None
    framing: FramingStudySpec | None = None
    harbor: ExecutableCommand | None = None
    scenario: RecordScenarioSpec | None = None
    scenario_search: ScenarioSearchSpec | None = None
    record_witness: RecordWitnessSpec | None = None
    memory: MemoryInspection | None = None
    forget: SummaryForget | None = None
    retirement: CheckpointRetirement | None = None
    collection: StateCollection | None = None
    decisions: DecisionCalibration | None = None
    challenge: ChallengeRunSpec | None = None
    cascade_fit: CascadeFitSpec | None = None
    cascade_confirmation: CascadeConfirmationSpec | None = None
    memory_analyst: MemoryAnalystSpec | None = None
    actor_decisions: ActorDecisionSpec | None = None
    prefix: ActorPrefixSpec | None = None
    research: FailureResearchSpec | None = None
    checkpoint_proposals: CheckpointProposalSpec | None = None
    campaign: ContinuationCampaignSpec | None = None
    learning_repair: LearningRepair | None = None
    agent_memory: AgentMemorySpec | None = None
    sample_count: SampleCountRequest | None = None
    rubric_derivation: RubricDerivationSpec | None = None
    rubric_review: RubricReview | None = None
    equivalence: EquivalenceSpec | None = None
    specification: SpecificationRegistration | None = None
    development_tasks: DevelopmentTaskQuery | None = None
    grounded: GroundRuntime | None = None
    climb_round: ClimbRound | None = None
    edit_proposals: EditProposalSpec | None = None
    climb_report: ClimbReportSpec | None = None
    task_validity: TaskValiditySpec | None = None
    hardening: TaskHardeningSpec | None = None
    facts: FactsDerivation | None = None
    induction: InductionSpec | None = None
    transition_validation: TransitionValidationSpec | None = None
    admission_policy: AdmissionPolicy | None = None
    spend: SpendPolicy = SpendPolicy()
    snapshots: ModelSnapshotQuery | None = None
    references: ArtifactReferenceQuery | None = None
    native_workbook: NativeWorkbookPayload | None = None
    rubric_trajectory: RubricTrajectoryRegistration | None = None

    @model_validator(mode="after")
    def exact_payload(self) -> "ContinuationCommand":
        """Reject unrelated payloads before any operation, including dry runs and replay.

        Returns:
            This command, unchanged, once only the selected action's payload is set.

        Raises:
            ValueError: The selected payload is missing or another action's payload is set.
        """
        selected = _ACTIONS_BY_NAME[self.action].payload
        if any(
            (getattr(self, action.payload) is not None) != (action.payload == selected)
            for action in CONTINUATION_ACTIONS
        ):
            raise ValueError(f"{self.action} requires only its {selected} payload")
        return self


def run_continuations(
    project: str, spec_path: Path, *, root: Path, yes: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Validate one ``--continuations`` spec and run its action through the owning handler.

    Args:
        project: Existing project named on the command line.
        spec_path: JSON ``ContinuationCommand`` file; also named in spend prompts.
        root: EXP root that owns the project store and spend settings.
        yes: Explicit authorization of the displayed estimate, including budget overruns.
        non_interactive: Fail rather than prompt when paid work lacks authorization.
        dry_run: Validate inputs without provider calls, runtime execution or publication.

    Returns:
        None. The handler prints its result; readable failures become usage errors.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError, ArtifactStoreError):
        command = ContinuationCommand.model_validate_json(spec_path.read_bytes())
        context = ContinuationContext(
            store=ProjectStore(root, project),
            console=console,
            revision=installed_execution_revision(),
            root=root,
            project=project,
            spec_path=spec_path,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
            operation=operation_identity(command),
            spend=command.spend,
        )
        action = _ACTIONS_BY_NAME[command.action]
        with operation_spend_scope():
            try:
                action.handler(context, getattr(command, action.payload))
            except (SpendCeilingReached, UnpricedDispatchRefused) as stop:
                console.print(f"[red]error[/red] {escape(str(stop))}")
                _spend_receipt(context, announce=True)
                console.print(
                    "Completed calls are saved. Raise spend.ceiling_usd or "
                    "spend.unpriced_attempts in the operation file, then run it again to resume."
                )
                raise typer.Exit(code=1) from None
            _spend_receipt(context)


def operation_identity(command: ContinuationCommand) -> str:
    """Identify one operation and its resumes by its exact payload, excluding spend choices.

    Args:
        command: Validated continuation command.

    Returns:
        A content-addressed ID naming the operation's spend ledger directory.
    """
    return stable_id("operation", command.model_dump(mode="json", exclude={"spend"}))


def _spend_receipt(context: ContinuationContext, *, announce: bool = False) -> None:
    """Save the operation's final spend receipt when a ceiling was consented.

    Args:
        context: The invocation whose operation owns the receipt.
        announce: Also print the accounting and the receipt path, as after a spend stop.
    """
    ceiling = current_spend_ceiling()
    if ceiling is None:
        return
    path = save_spend_receipt(context.store, context.operation, ceiling, context.command)
    if not announce:
        return
    summary = ceiling.summary()
    context.console.print(
        Text(
            f"Spend: ${summary.settled_usd:.6f} settled and ${summary.unknown_usd:.6f} kept for "
            f"unknown outcomes, of a ${summary.ceiling_usd:.6f} ceiling; "
            f"{summary.unpriced_attempts}/{summary.unpriced_attempt_ceiling} unpriced attempts."
        )
    )
    context.console.print(Text(f"Spend receipt: {path}"))
