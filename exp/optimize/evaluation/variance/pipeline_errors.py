"""Error tags, cluster naming and probe selection, wired into the decision-variance stages.

- **Cluster naming** (``variance-build``): when the build names its tag clusters, the stage binds
  an admitted naming client for ``build.naming_alias`` (:func:`cluster_namer`) and prices the
  plan's ``naming_calls`` at that alias's tariff. The dry run lists the error-tag view plan, the
  cluster admission registration and, once saved, the admissions themselves
  (:func:`build_lines`).
- **Probe selection** (``error-mining``): the build ranks mining-probe tasks by their source
  traces' error tags. The registered mining job serves one task, so the stage refuses a job whose
  task the build did not select (:func:`probe_line`); a job per selected task needs a registered
  start per task, which the spec does not carry.
- **Attempt error tags** (``traceml``): with ``traceml.error_bank`` set, each tier's grouped
  attempts are tagged with the error bank over the whole attempt (:func:`tag_tier_errors`, one
  ``observations`` unit per attempt) and over each step (``step`` view). The attempt record
  becomes the coalition's mistake patterns (:func:`coalition_patterns`); the step record ranks
  each attempt's actor-critic points (:func:`attempt_error_points`), saved as an
  ``error-points`` artifact (:func:`save_error_points`) that error mining registers as each
  tier's ``error_points`` selector and critic reference, read with ``read_error_points``. Which
  points the selectors choose stays their judgment.

Every tag is bought through the decision tagging seam and its cache, so a rerun replays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.models import load_model_catalog
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.executable.model_costs import (
    CostComponent,
    decision_cost,
    model_cost,
)
from exp.optimize.evaluation.variance.build_records import VarianceBuildPlan, VarianceBuildReport
from exp.optimize.evaluation.variance.coalition import BehaviorPattern
from exp.optimize.evaluation.variance.evidence import ERROR_POINTS_TYPE, ReferenceSource
from exp.optimize.evaluation.variance.pipeline_inputs import admitted_transport
from exp.optimize.evaluation.variance.pipeline_spec import (
    BuildOutputs,
    DecisionVarianceSpec,
    StageContext,
)
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.error_tags import (
    ErrorPoint,
    bank_for_view,
    error_bank,
    error_patterns,
    error_points,
    load_error_steps,
    load_error_tags,
    persist_error_tags,
    tag_error_steps,
    tag_errors,
    trajectory_unit,
)
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tag_cluster_names import ClusterNamer
from exp.simulation.mining.tagging import TraceTagger
from exp.simulation.mining.traceml import TraceTrajectory, traceml_bank

_QUESTIONS_PER_CALL = 32


def naming_attempts(spec: DecisionVarianceSpec, plan: VarianceBuildPlan) -> int:
    """Physical attempts cluster naming may take: every naming call at the retry ceiling."""
    del spec
    return plan.naming_calls * RetryPolicy().maximum_attempts


def naming_costs(
    project: ProjectStore, spec: DecisionVarianceSpec, plan: VarianceBuildPlan
) -> tuple[CostComponent, ...]:
    """Price the plan's naming calls on the naming alias, or nothing when none are planned."""
    naming = spec.build.spec.cluster_naming
    alias = spec.build.naming_alias
    if plan.saved or naming is None or alias is None or plan.naming_calls == 0:
        return ()
    attempts = naming_attempts(spec, plan)
    return (model_cost(project, alias, attempts, naming.maximum_output_tokens),)


def build_lines(
    project: ProjectStore, spec: DecisionVarianceSpec, plan: VarianceBuildPlan
) -> tuple[str, ...]:
    """Describe the error-tag view plan, the cluster admission and the naming calls.

    Args:
        project: Project whose saved report, when one exists, lists the admissions.
        spec: The registration.
        plan: The build's credential-free summary.

    Returns:
        Plan lines for the dry run and the consent summary.
    """
    build = spec.build.spec
    lines: list[str] = []
    errors = plan.errors
    if errors is None:
        lines.append("Error tags: no error bank; probes are not ranked by errors.")
    else:
        lines.append(
            f"Error tags: {errors.bank_id} v{errors.bank_version} asks {errors.questions} "
            f"propositions over the {errors.view} view of {errors.traces} traces "
            f"({', '.join(errors.roles)}), {errors.decision_calls} decision calls; they rank "
            f"the mining probes{'' if build.probe_limit is None else f' (top {build.probe_limit})'}"
            " and join the clustered tag space.",
        )
    if build.clustering is None:
        lines.append("Clusters: clustering off; world learning is admitted by rules only.")
    else:
        admission = build.cluster_admission
        lines.append(
            f"Clusters: count chosen by the data up to {build.clustering.max_clusters}; admission "
            f"total {admission.total or 'all'}, per cluster {admission.per_cluster or 'all'}, "
            f"minimum size {admission.minimum_cluster_size}"
            + (
                f", excluding {list(admission.exclude_clusters)}"
                if admission.exclude_clusters
                else ""
            )
            + "."
        )
        alias = spec.build.naming_alias
        lines.append(
            "Cluster naming: off."
            if build.cluster_naming is None
            else f"Cluster naming: up to {plan.naming_calls} calls on {alias} "
            f"({len(build.cluster_naming.prompts)} drafts and a reconcile per cluster)."
        )
    if plan.saved:
        report = saved_report(project, plan)
        if report is not None and report.admission is not None:
            quotas = ", ".join(
                f"{row.cluster_id} {row.admitted}/{row.eligible}" for row in report.admission.quotas
            )
            lines.append(
                f"Cluster admissions: {len(report.admission.admitted)} traces admitted "
                f"({quotas}); {len(report.admission.refused)} refused."
            )
        if report is not None and report.selected_probes:
            lines.append(f"Selected probes: {', '.join(report.selected_probes)}.")
    return tuple(lines)


def saved_report(project: ProjectStore, plan: VarianceBuildPlan) -> VarianceBuildReport | None:
    """Read the saved build report of a plan, or None when it is not saved."""
    result = plan.identity + "-result"
    if not project.artifacts.exists(result):
        return None
    return VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(result, "report.json")
    )


def cluster_namer(context: StageContext, plan: VarianceBuildPlan) -> ClusterNamer | None:
    """Bind the admitted naming client when the build names its clusters.

    Raises:
        ValueError: Naming is requested without a naming alias.
    """
    spec = context.spec
    if spec.build.spec.cluster_naming is None or spec.build.spec.clustering is None:
        return None
    alias = spec.build.naming_alias
    if alias is None:
        raise ValueError("cluster naming needs build.naming_alias, an admitted naming model")
    deadline = datetime.now(UTC) + timedelta(hours=spec.build.naming_deadline_hours)
    transport = admitted_transport(context, "naming", naming_attempts(spec, plan), deadline)
    catalog = load_model_catalog(context.project.model_catalog_path)
    resolved = RuntimeModelCatalog(catalog, transport_factory=lambda: transport).resolve(alias)
    return ClusterNamer(client=resolved.client, model=resolved.snapshot)


def probe_line(build: BuildOutputs, task_id: str) -> str:
    """Check the mining job's task against the build's probe selection and describe it.

    Raises:
        ValueError: The build selected probes and the job's task is not one of them.
    """
    if not build.selected_probes:
        return "no error-ranked probe selection"
    if task_id not in build.selected_probes:
        raise ValueError(
            f"the mining job serves task {task_id}, which the variance build did not select as "
            f"a mining probe ({list(build.selected_probes)}); register a job for a selected task"
        )
    rank = next((row for row in build.probe_ranking if row.task_id == task_id), None)
    return f"probe {task_id} selected" + ("" if rank is None else f", error rank {rank.rank}")


def tier_error_bank(spec: DecisionVarianceSpec) -> QuestionBank | None:
    """The error bank attempts are tagged with, or None when attempt error tags are off."""
    settings = spec.traceml
    if settings.error_bank is None:
        return None
    return error_bank(
        settings.error_bank, root=settings.bank_root, families=settings.error_families
    )


def traceml_label_attempts(spec: DecisionVarianceSpec) -> int:
    """Physical request bound for one full tier's action taxonomy, including native retries."""
    bank = traceml_bank(spec.traceml.bank_root)
    steps = (
        spec.mining.attempts
        * spec.mining.registered.maximum_model_calls
        * spec.traceml.steps_per_call
    )
    return (
        steps
        * math.ceil(len(bank.questions) / _QUESTIONS_PER_CALL)
        * RetryPolicy().maximum_attempts
    )


def traceml_decision_attempts(project: ProjectStore, spec: DecisionVarianceSpec) -> int:
    """One shared default tagger covers action labels plus whole/step errors for every tier."""
    return len(spec.mining.actors) * traceml_label_attempts(spec) + sum(
        row.attempts for row in error_tag_costs(project, spec)
    )


def error_tag_costs(project: ProjectStore, spec: DecisionVarianceSpec) -> tuple[CostComponent, ...]:
    """Price tagging every tier's attempts and their steps with the error bank."""
    bank = tier_error_bank(spec)
    if bank is None:
        return ()
    settings, mining = spec.traceml, spec.mining
    per_attempt = math.ceil(
        len(bank_for_view(bank, "observations").questions) / _QUESTIONS_PER_CALL
    )
    per_step = math.ceil(len(bank_for_view(bank, "step").questions) / _QUESTIONS_PER_CALL)
    steps = mining.registered.maximum_model_calls * settings.steps_per_call
    calls = mining.attempts * (per_attempt + steps * per_step)
    return tuple(
        decision_cost(project, settings.decision_alias, calls * RetryPolicy().maximum_attempts)
        for _ in mining.actors
    )


def tag_tier_errors(
    store: ArtifactStore,
    bank: QuestionBank,
    trajectories: Sequence[TraceTrajectory],
    tagger: TraceTagger,
    *,
    code_revision: str,
    inputs: Sequence[ArtifactInput],
    concurrency: int = 1,
) -> tuple[ArtifactInput, ArtifactInput]:
    """Tag one tier's attempts as wholes and step by step, and save both records.

    Returns:
        The saved ``error-tags`` (one row per attempt) and ``error-steps`` records.
    """
    units = [trajectory_unit(trajectory) for trajectory in trajectories]
    tags = tag_errors(units, tagger=tagger, bank=bank, unit_concurrency=concurrency)
    steps = tag_error_steps(trajectories, tagger=tagger, bank=bank, trace_concurrency=concurrency)
    return (
        persist_error_tags(store, tags, code_revision=code_revision, inputs=inputs),
        persist_error_tags(store, steps, code_revision=code_revision, inputs=inputs),
    )


def attempt_error_points(
    store: ArtifactStore, reference: ArtifactInput, *, limit: int = 3
) -> JsonObject:
    """Each attempt's actor-critic points ranked by the errors of the steps decided there."""
    record = load_error_steps(store, reference)
    ranked: dict[str, list[ErrorPoint]] = {
        trace.trace_id: list(error_points(trace, limit=limit)) for trace in record.traces
    }
    return {
        trace: [point.model_dump(mode="json") for point in points]
        for trace, points in ranked.items()
        if points
    }


def save_error_points(
    store: ArtifactStore, reference: ArtifactInput, *, code_revision: str, limit: int = 3
) -> ArtifactInput:
    """Save each attempt's error-ranked points as an ``error-points`` reference artifact.

    Attempt traces are named ``<tier>/attempt:<n>``; the record keys each by ``n``, the index the
    evidence packet uses, so a selector reads attempt ``n``'s points as ``errors:<n>``.

    Args:
        store: Project artifacts.
        reference: The tier's saved ``error-steps`` record.
        code_revision: Producing source identity.
        limit: Most points kept per attempt.

    Returns:
        The saved artifact, adopted when the identical record is already saved.
    """
    ranked = attempt_error_points(store, reference, limit=limit)
    attempts: list[JsonValue] = []
    for trace, points in sorted(ranked.items()):
        index = trace.rsplit(":", 1)[-1]
        if index.isdigit():
            attempts.append({"attempt": int(index), "trace_id": trace, "points": points})
    record: JsonObject = {"steps": reference.model_dump(mode="json"), "attempts": attempts}
    identity = stable_id(ERROR_POINTS_TYPE, {"record": sha256_json(record)})
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=ERROR_POINTS_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(reference,),
            ),
            files={"points.json": record},
        )
    return artifact_input(store.read(identity).manifest)


def error_point_reference(artifact: ArtifactInput) -> ReferenceSource:
    """Register a tier's saved error points as its selectors' and critics' reference."""
    return ReferenceSource(
        kind="error_points", name="error-points", artifact=artifact, file="points.json"
    )


def coalition_patterns(
    store: ArtifactStore, errors: Mapping[str, ArtifactInput], bank: QuestionBank
) -> tuple[BehaviorPattern, ...]:
    """Turn every tier's attempt error tags into coalition mistake patterns.

    Attempt rows are named ``<tier>/attempt:<n>``; the coalition names the same attempt
    ``<tier>.g<n>``. A proposition firing in several tiers becomes one pattern per tier, so a
    pattern never pools tiers.
    """
    patterns: list[BehaviorPattern] = []
    for tier, reference in sorted(errors.items()):
        tags = load_error_tags(store, reference)
        attempts = {
            row.trace_id: f"{tier}.g{row.trace_id.rsplit(':', 1)[1]}"
            for row in tags.traces
            if row.trace_id.startswith(f"{tier}/attempt:")
        }
        for pattern in error_patterns(tags, bank, attempts=attempts):
            row = pattern.model_dump(mode="json")
            row["pattern_id"] = f"{tier}.{pattern.pattern_id}"
            patterns.append(BehaviorPattern.model_validate(row))
    return tuple(patterns)
