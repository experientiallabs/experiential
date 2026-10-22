"""The ``exp eval --continuations`` contract, its action registry and dispatch.

Each action is implemented by the handler module of the workflow that owns it. This module only
validates the spec, resolves the shared invocation context once and dispatches to that handler.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import model_validator
from rich.console import Console

from exp.cli.evaluation.analysis import (
    handle_analyse_failures,
    handle_analyse_memory,
    handle_check_failure_research,
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
from exp.cli.evaluation.equivalence import handle_admit_actor_prefix, handle_compare_environments
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
from exp.cli.evaluation.learning_repair import handle_repair_learning
from exp.cli.evaluation.trajectory_traces import handle_export_traces
from exp.cli.shared.options import usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ArtifactStoreError, ProjectStore
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.analysis.actor_decisions import ActorDecisionSpec
from exp.optimize.evaluation.analysis.decision_calibration import DecisionCalibration
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
from exp.optimize.evaluation.equivalence.prefixes import ActorPrefixSpec
from exp.optimize.evaluation.equivalence.runner import EquivalenceSpec
from exp.optimize.evaluation.executable.controls import SpecificationRegistration
from exp.optimize.evaluation.executable.scenarios import RecordScenarioSpec, RecordWitnessSpec
from exp.optimize.evaluation.factory.candidates import CandidateBatchSpec
from exp.optimize.evaluation.factory.proposals import TaskProposalSpec
from exp.optimize.evaluation.factory.scenario_search import ScenarioSearchSpec
from exp.optimize.evaluation.factory.variants import TaskVariantSpec
from exp.optimize.evaluation.learning.lessons import LearningRepair
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
    ContinuationAction("compare-environments", "equivalence", handle_compare_environments),
    ContinuationAction("repair-learning", "learning_repair", handle_repair_learning),
    ContinuationAction("admit-actor-prefix", "prefix", handle_admit_actor_prefix),
    ContinuationAction("check-failure-research", "research", handle_check_failure_research),
    ContinuationAction("propose-checkpoints", "checkpoint_proposals", handle_propose_checkpoints),
    ContinuationAction("run-continuation-campaign", "campaign", handle_run_continuation_campaign),
    ContinuationAction("probe-actor-decisions", "actor_decisions", handle_probe_actor_decisions),
    ContinuationAction("inspect-memory", "memory", handle_inspect_memory),
    ContinuationAction("measure-decisions", "decisions", handle_measure_decisions),
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
        decisions: Frozen fixed-question calibration inventory, never a serving approval.
        memory_analyst: Opt-in bounded diagnostic using the existing trace-analysis rubric.
        actor_decisions: Trace-only next-decision probes with a repeated-reference control.
        prefix: Extract a complete retained actor input and bind a saved simulator observation.
        research: Check cited counterexample outcomes and retain a provisional research revision.
        checkpoint_proposals: Retrospective point selector producing an existing matched plan.
        campaign: Registered source attempts, selectors, prospective critics and continuations.
        learning_repair: Failed-learning settlement, lesson screening or library evolution.
        equivalence: Paired executable reference/candidate studies and diagnostic trace council.
        specification: Independent acceptance specification and its source evidence to save.
        development_tasks: Build fit tasks to print, optionally one with its public tools.
        grounded: Simulated world over the build's simulator, emitted as a capture operation.
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
    decisions: DecisionCalibration | None = None
    memory_analyst: MemoryAnalystSpec | None = None
    actor_decisions: ActorDecisionSpec | None = None
    prefix: ActorPrefixSpec | None = None
    research: FailureResearchSpec | None = None
    checkpoint_proposals: CheckpointProposalSpec | None = None
    campaign: ContinuationCampaignSpec | None = None
    learning_repair: LearningRepair | None = None
    equivalence: EquivalenceSpec | None = None
    specification: SpecificationRegistration | None = None
    development_tasks: DevelopmentTaskQuery | None = None
    grounded: GroundRuntime | None = None

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
        )
        action = _ACTIONS_BY_NAME[command.action]
        action.handler(context, getattr(command, action.payload))
