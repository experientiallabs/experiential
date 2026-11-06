"""Mixed behavioral tags and reviewed difference groups in the decision-variance pipeline."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.variance.behavior_differences import (
    BEHAVIOR_DIFFERENCES_TYPE,
)
from exp.optimize.evaluation.variance.behavior_pairs import (
    compact_differences,
    run_behavior_differences,
)
from exp.optimize.evaluation.variance.evidence import ReferenceSource
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judged import EpisodeJudge, judge_study
from exp.optimize.evaluation.variance.pipeline_inputs import admitted_transport
from exp.optimize.evaluation.variance.pipeline_spec import (
    DecisionVarianceSpec,
    StageContext,
)
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.questions import QuestionBank, resolve_question_bank
from exp.simulation.mining.tagging import DecisionCache


def behavior_bank(spec: DecisionVarianceSpec) -> QuestionBank | None:
    """Resolve the exact local mixed vocabulary before pricing or provider dispatch."""
    settings = spec.mining.behaviors
    if settings is None:
        return None
    bank = resolve_question_bank(settings.bank_id, root=spec.traceml.bank_root)
    settings.check_bank(bank)
    return bank


def behavior_costs(project: ProjectStore, spec: DecisionVarianceSpec) -> tuple[CostComponent, ...]:
    """Bound per-step tagging plus one alignment and independent audit per proposed pair."""
    settings, bank = spec.mining.behaviors, behavior_bank(spec)
    if settings is None or bank is None:
        return ()
    mining = spec.mining
    steps = mining.registered.maximum_model_calls * (spec.traceml.steps_per_call + 1)
    calls = mining.attempts * steps * math.ceil(len(bank.questions) / 32)
    costs = [
        decision_cost(project, spec.traceml.decision_alias, calls * RetryPolicy().maximum_attempts)
    ]
    for author in (settings.author, settings.reviewer):
        costs.extend(
            row.scaled(settings.maximum_pairs) for row in evaluation_costs(project, author)
        )
    return tuple(row.scaled(len(mining.actors)) for row in costs)


def behavior_lines(spec: DecisionVarianceSpec) -> tuple[str, ...]:
    """Describe the enabled contrast work in the credential-free stage plan."""
    settings = spec.mining.behaviors
    if settings is None:
        return ()
    return (
        f"Behavior differences: mixed bank {settings.bank_id}, up to {settings.maximum_pairs} "
        "same-task pairs per tier, one alignment and independent audit per pair; accepted "
        "Added/Removed groups inform selectors and rubrics, never actor rewards.",
    )


def prepare_behavior_reference(
    context: StageContext,
    job: ContinuationJob,
    judge: EpisodeJudge,
    exposure: ExposurePlan,
) -> ArtifactInput | None:
    """Judge the saved actor group before selectors, then mine its reviewed differences."""
    spec, project = context.spec, context.project
    settings, bank = spec.mining.behaviors, behavior_bank(spec)
    if settings is None or bank is None:
        return None
    result = saved_continuation_job(project, job, code_revision=context.code_revision)
    if result is None:
        execute_continuation_job(project, job, code_revision=context.code_revision)
    source = continuation_result_reference(project, job, code_revision=context.code_revision)
    judged, _ = judge_study(
        project, source, judge, configuration=job.configuration, code_revision=context.code_revision
    )
    tagger = context.dependencies.traceml_tagger
    if tagger is None:
        catalog = load_model_catalog(project.model_catalog_path)
        resolved = RuntimeModelCatalog(catalog).resolve(spec.traceml.decision_alias)
        steps = job.maximum_model_calls * (spec.traceml.steps_per_call + 1)
        calls = spec.mining.attempts * steps * math.ceil(len(bank.questions) / 32)
        transport = admitted_transport(
            context,
            f"behavior-tags-{job.actor_alias}",
            calls * RetryPolicy().maximum_attempts,
            datetime.now(UTC) + timedelta(hours=spec.traceml.deadline_hours),
        ).priced_as(spec.traceml.decision_alias)
        tagger = DecisionsTraceTagger.for_alias(
            resolved,
            transport=transport,
            cache=DecisionCache(project.paths),
            maximum_concurrency=spec.traceml.concurrency,
        )
    ref, _ = run_behavior_differences(
        project,
        settings,
        study=source,
        judged=judged,
        bank=bank,
        tier=job.actor_alias,
        exposure=exposure,
        configuration=job.configuration,
        tagger=tagger,
        code_revision=context.code_revision,
        maximum_steps=job.maximum_model_calls * (spec.traceml.steps_per_call + 1),
        concurrency=spec.traceml.concurrency,
    )
    return ref


def behavior_reference(project: ProjectStore, reference: ArtifactInput) -> ReferenceSource:
    """Create a compact, typed read-only selector reference with the full report as provenance."""
    groups = compact_differences(project, reference)
    record: JsonObject = {
        "source": reference.model_dump(mode="json"),
        "groups": [group.model_dump(mode="json") for group in groups],
        "authority": "reviewed behavioral hypotheses, never episode rewards",
    }
    identity = stable_id("behavior-difference-reference", {"record": sha256_json(record)})
    store = project.artifacts
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=BEHAVIOR_DIFFERENCES_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=store.read(reference.artifact_id).manifest.code_revision,
                inputs=(reference,),
            ),
            files={"mining.json": record},
        )
    return ReferenceSource(
        kind="behavior_differences",
        name="behavior-differences",
        artifact=artifact_input(store.read(identity).manifest),
        file="mining.json",
    )
