"""Continuation actions owned by bounded continuation studies and witnessed SFT export."""

from datetime import UTC, datetime

from pydantic import Field

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import artifact_input
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
from exp.optimize.evaluation.equivalence.policy import load_admission_policy
from exp.optimize.model.sft.builder import build_sft_dataset, write_sft_dataset
from exp.optimize.model.sft.contracts import SFTBuildSpec


class ContinuationExport(ContractModel):
    """Saved development studies to independently re-grade and export without model calls.

    Attributes:
        studies: Exact complete results; every branch remains in their admission reports.
        policy: Registered admission policy, registered before every study; each branch's
            per-dimension verdicts are saved with its admission.
        dataset: Existing SFT split controls. Related continuations share one leakage group.
    """

    studies: tuple[ArtifactInput, ...] = Field(min_length=1)
    policy: ArtifactInput
    dataset: SFTBuildSpec = Field(default_factory=SFTBuildSpec)


def handle_run_continuation_campaign(
    context: ContinuationContext, campaign: ContinuationCampaignSpec
) -> None:
    """Run or replay registered source attempts, selectors, critics and continuations.

    Args:
        context: Validated ``--continuations`` invocation.
        campaign: Registered campaign with its selectors, critics and continuation budgets.

    Returns:
        None. The campaign inference summary is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
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
    if context.dry_run:
        console.print("Known profiles validated; future evidence packets are checked whole.")
        return
    if needs_execution and not context.consent(campaign):
        return
    result = completed or run_continuation_campaign(store, campaign, code_revision=revision)
    console.print_json(data=campaign_summary(store, result))


def handle_propose_checkpoints(
    context: ContinuationContext, selector: CheckpointProposalSpec
) -> None:
    """Select retrospective checkpoints with at most one new provider attempt.

    Args:
        context: Validated ``--continuations`` invocation.
        selector: Retrospective point selector producing a matched selection plan.

    Returns:
        None. The selected plan is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    propose_checkpoints(store, selector, code_revision=revision, dry_run=True)
    console.print("Retrospective checkpoint selector: at most one new provider attempt.")
    if context.dry_run:
        console.print("Complete source inventory validated; no model calls or publication.")
        return
    has_response = store.artifacts.exists(selector.identity(revision) + "-response")
    if not has_response and not context.consent(selector):
        return
    selected = propose_checkpoints(store, selector, code_revision=revision)
    assert selected is not None
    console.print_json(data=selected.model_dump(mode="json"))


def handle_compare_observations(
    context: ContinuationContext, spec: ObservationContrastSpec
) -> None:
    """Compare a recorded reply with an authored alternative under equal suffix budgets.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Saved-reply substitution with an identical world and equal suffix budgets.

    Returns:
        None. The comparison scope and metrics are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_observations = saved_observation_contrast(store, spec, code_revision=revision)
    if saved_observations is None:
        attempts = preflight_observation_contrast(store, spec)
    else:
        attempts = sum(cell.job.maximum_physical_attempts for cell in saved_observations.cells)
    console.print(
        f"Observation comparison: {spec.pairs * 2} draws; "
        f"maximum actor HTTP attempts: {attempts}; original world retained."
    )
    console.print("Recorded reply retains source authority; authored alternative is unverified.")
    if context.dry_run:
        console.print("Sources and suffix policy validated without runtime imports or calls.")
        return
    if saved_observations is None and not context.consent(spec):
        return
    report = saved_observations or run_observation_contrast(store, spec, code_revision=revision)
    console.print_json(data={"scope": report.scope, **report.metrics()})


def handle_compare_actions(context: ContinuationContext, contrast: ActionContrastSpec) -> None:
    """Run or replay paired imposed decisions followed by equal-budget suffixes.

    Args:
        context: Validated ``--continuations`` invocation.
        contrast: Paired imposed decisions and their shared suffix job.

    Returns:
        None. The contrast result is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_contrast = saved_action_contrast(store, contrast, code_revision=revision)
    if saved_contrast is None:
        preflight_action_contrast(store, contrast, code_revision=revision)
    if context.dry_run:
        console.print("Paired decisions validated; no tools or actor calls executed.")
        return
    if saved_contrast is None and not context.consent(contrast):
        return
    result = saved_contrast or run_action_contrast(store, contrast, code_revision=revision)
    console.print_json(data=result.model_dump(mode="json"))


def handle_intervene(context: ContinuationContext, intervention: ActionInterventionSpec) -> None:
    """Capture one scripted tool decision without changing the actor inference allowance.

    Args:
        context: Validated ``--continuations`` invocation.
        intervention: Scripted tool decision applied to a saved checkpoint.

    Returns:
        None. The intervention result is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_action = saved_intervention(store, intervention, code_revision=revision)
    if saved_action is None:
        preflight_intervention(store, intervention)
    if context.dry_run:
        console.print("Intervention validated; no runtime imported or tool executed.")
        return
    if saved_action is None and not context.consent(intervention):
        return
    result = saved_action or capture_action_intervention(
        store, intervention, code_revision=revision
    )
    console.print_json(data=result.model_dump(mode="json"))


def handle_match_checkpoints(
    context: ContinuationContext, selection: CheckpointSelectionSpec
) -> None:
    """Match proposed checkpoints to controls and, unless dry-run, register the plan.

    Args:
        context: Validated ``--continuations`` invocation.
        selection: Proposed checkpoints and the matching protocol.

    Returns:
        None. The plan, or its saved reference and match counts, is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    identity = selection.identity(revision)
    if store.artifacts.exists(identity):
        plan = load_selection_plan(store, artifact_input(store.artifacts.read(identity).manifest))
    else:
        plan = match_checkpoints(store, selection)
    if context.dry_run:
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


def handle_run_selection(
    context: ContinuationContext, experiment: CheckpointExperimentSpec
) -> None:
    """Run or replay a registered selection plan with its optional prospective critic.

    Args:
        context: Validated ``--continuations`` invocation.
        experiment: Registered selection plan and optional critic.

    Returns:
        None. The experiment inference summary is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_experiment = saved_checkpoint_experiment(store, experiment, code_revision=revision)
    if saved_experiment is None:
        preflight_checkpoint_experiment(store, experiment)
    plan = load_selection_plan(store, experiment.selection)
    plan_revision = store.artifacts.read(experiment.selection.artifact_id).manifest.code_revision
    jobs = selection_jobs(plan, code_revision=plan_revision, plan_reference=experiment.selection)
    attempts = sum(job.maximum_physical_attempts for _, job in jobs)
    if experiment.critic is not None:
        attempts += len(jobs) * experiment.critic.maximum_physical_attempts
    console.print(f"Matched cells: {len(jobs)}; maximum actor/critic HTTP attempts: {attempts}")
    if context.dry_run:
        console.print("Validated without runtime imports or provider calls.")
        return
    if saved_experiment is None and not context.consent(experiment):
        return
    report = saved_experiment or run_checkpoint_experiment(
        store, experiment, code_revision=revision
    )
    console.print_json(data=experiment_summary(store, report))


def handle_predict_success(context: ContinuationContext, critic: CriticSpec) -> None:
    """Record one prospective success estimate from public critic context.

    Args:
        context: Validated ``--continuations`` invocation.
        critic: Prospective estimate settings.

    Returns:
        None. The saved reference, probability and rationale are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    exists = store.artifacts.exists(critic.identity(revision))
    if not exists:
        preflight_critic(store, critic)
    console.print(f"Critic maximum HTTP attempts: {critic.maximum_physical_attempts}")
    if context.dry_run:
        console.print("Public critic context validated; no provider call.")
        return
    if not exists and not context.consent(critic):
        return
    reference = predict_success(store, critic, code_revision=revision)
    prediction = load_critic_prediction(store.artifacts, reference)
    console.print_json(
        data={
            "reference": reference.model_dump(mode="json"),
            "predicted_success": prediction.estimate.probability,
            "rationale": prediction.estimate.rationale,
        }
    )


def handle_compare_actors(context: ContinuationContext, panel: ActorPanelSpec) -> None:
    """Run or replay a frozen multi-model comparison over the same continuation cases.

    Args:
        context: Validated ``--continuations`` invocation.
        panel: Frozen cases and the actor models compared on each.

    Returns:
        None. The panel scope and metrics are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_panel = saved_actor_panel(store, panel, code_revision=revision)
    if saved_panel is None:
        preflight_actor_panel(store, panel)
    attempts = sum(job.maximum_physical_attempts for job in panel.cases)
    attempts *= len(panel.actors)
    console.print(
        f"Actor panel: {len(panel.cases)} cases, "
        f"{len(panel.actors)} models; "
        f"maximum actor HTTP attempts: {attempts}"
    )
    if context.dry_run:
        console.print("Validated without runtime imports or provider calls.")
        return
    if saved_panel is None and not context.consent(panel):
        return
    result = saved_panel or run_actor_panel(store, panel, code_revision=revision)
    console.print_json(data={"scope": result.scope, **result.metrics})


def handle_capture(context: ContinuationContext, capture: CaptureDefinition) -> None:
    """Capture or replay the initial actor and world state for later continuations.

    Args:
        context: Validated ``--continuations`` invocation.
        capture: Initial actor and world capture definition.

    Returns:
        None. The saved capture reference is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_capture(store, capture, code_revision=revision)
    if saved is None:
        preflight_capture(store, capture)
    if context.dry_run:
        console.print("Capture validated without importing runtime code or opening a world.")
        return
    if saved is None and not context.consent(capture):
        return
    reference = saved or capture_initial_world(store, capture, code_revision=revision)
    console.print_json(data=reference.model_dump(mode="json"))


def handle_compare_framing(context: ContinuationContext, spec: FramingStudySpec) -> None:
    """Run or replay two authored incidental cues over one shared base job.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Two authored cues and their shared base job.

    Returns:
        None. The framing metrics are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_framing = saved_framing_study(store, spec, code_revision=revision)
    if saved_framing is None:
        preflight_continuation_job(store, spec.base)
    console.print(
        f"Framing cells: {spec.pairs * 2}; maximum HTTP attempts: "
        f"{spec.base.maximum_physical_attempts * spec.pairs * 2}"
    )
    if context.dry_run:
        console.print("Validated without runtime imports or provider calls.")
        return
    if saved_framing is None and not context.consent(spec):
        return
    report = saved_framing or run_framing_study(store, spec, code_revision=revision)
    console.print_json(data=report.metrics())


def handle_run(context: ContinuationContext, job: ContinuationJob) -> None:
    """Run or resume one continuation study on its trusted local runtime.

    Args:
        context: Validated ``--continuations`` invocation.
        job: One resumable continuation study.

    Returns:
        None. Metrics, saved checkpoints and the result reference are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_continuation_job(store, job, code_revision=revision)
    if saved is None:
        preflight_continuation_job(store, job)
    console.print(
        f"Planned branches: {job.study.repeats}; "
        f"maximum HTTP attempts: {job.maximum_physical_attempts}"
    )
    console.print(f"Trusted local runtime: {job.runtime_file}", markup=False)
    if context.dry_run:
        console.print("Validated without importing runtime code or calling providers.")
        return
    if saved is None and not context.consent(job):
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


def handle_export(context: ContinuationContext, export: ContinuationExport) -> None:
    """Independently admit saved studies and publish their witnessed SFT dataset.

    Args:
        context: Validated ``--continuations`` invocation.
        export: Saved studies and the SFT split settings.

    Returns:
        None. Admission scopes and the published dataset are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    load_admission_policy(store.artifacts, export.policy)
    for reference in export.studies:
        load_study(store.artifacts, reference)
    if context.dry_run:
        console.print("Study inventories validated. Admission and dataset publication not run.")
        return
    accepted = tuple(
        admit_continuations(
            store.artifacts,
            reference,
            policy=export.policy,
            created_at=datetime.now(UTC),
            code_revision=revision,
        )
        for reference in export.studies
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
        verdicts = [row.policy.status for row in report.branches]
        console.print(
            f"Admission policy {export.policy.artifact_id}: "
            + ", ".join(f"{verdicts.count(name)} {name}" for name in sorted(set(verdicts)))
            + "; per-dimension verdicts saved with the admission.",
            markup=False,
        )
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
        spec=export.dataset,
        created_at=datetime.now(UTC),
        code_revision=revision,
    )
    write_sft_dataset(store, dataset)
    console.print(f"Dataset: {dataset.dataset.dataset_id} ({dataset.dataset.status})")
    console.print(
        f"Assistant targets: {len(dataset.rows)}; exclusions: {len(dataset.inspection.exclusions)}"
    )
    console.print("Prefix and tool observations are context only. No training or promotion run.")
