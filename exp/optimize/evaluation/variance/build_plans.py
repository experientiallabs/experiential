"""Process-local reuse of a variance build's plan, which is a pure function of immutable inputs.

:func:`~exp.optimize.evaluation.variance.build.plan_variance_build` reads and decodes the whole
completed project build on every call: the trace dataset, the task set, both RAG indexes (their
transitions and vectors) and the embedding checkpoint. On a large project that is about 42 GB of
reads and several minutes of CPU per call, and one front-door start used to plan the same build
about ten times (the families review, each families slot's plan and its saved-result probe, the
preparation plan, then all of it again inside ``run_front_door``).

Everything the plan says is a function of inputs that cannot change under a running process
without changing this key: the project's completed build (immutable artifact references), the
registration, the build's semantic sources, the decision model snapshot, the reviewer, and the
question banks. Two fields read mutable state, and neither is reused blindly:

- ``saved`` (whether the report exists) is re-read from the store on every call;
- ``embedding_inputs_saved`` grows as the build's checkpoint fills, so a plan is kept only once
  the checkpoint holds every input (it then cannot change; checkpoints only add vectors).

This module is deliberately outside the build's import closure (:func:`build_sources`), so it
does not re-key any saved variance build.
"""

from __future__ import annotations

import threading

from exp.common.core.artifacts import Sha256, sha256_json
from exp.common.models import ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_errors import asked_error_bank
from exp.optimize.evaluation.variance.build_records import (
    VarianceBuildPlan,
    VarianceBuildReport,
    VarianceBuildSpec,
)
from exp.optimize.evaluation.variance.build_source import build_source_identity
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.environment_review import CompatibilityReviewer
from exp.simulation.mining.questions import resolve_question_bank
from exp.simulation.mining.tag_cluster_names import ClusterNamer

_PLANS: dict[str, VarianceBuildPlan] = {}
"""Complete plans already computed in this process, by :func:`_key`."""

_LOCK = threading.Lock()


def _key(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    code_revision: str,
    decision_model: ModelSnapshot | None,
    reviewer_model: ModelSnapshot | None,
    reviewer_configuration_sha256: Sha256 | None,
) -> str | None:
    """Digest every input the plan reads, or None when there is no completed build."""
    build = project.load_project().build
    if build is None:
        return None
    if decision_model is None:
        catalog = load_model_catalog(project.model_catalog_path)
        if spec.decision_alias not in catalog.models:
            return None  # let the planner raise its own error
        decision_model = RuntimeModelCatalog(catalog, environment={}).snapshot(spec.decision_alias)[
            0
        ]
    banks = {"task": resolve_question_bank(spec.task_bank).families(*spec.task_families)}
    if spec.observations_bank is not None:
        banks["observations"] = resolve_question_bank(spec.observations_bank).families(
            *spec.observations_families
        )
    return sha256_json(
        {
            "project": str(project.artifacts.project_directory),
            "build": build.model_dump(mode="json"),
            "spec": spec.model_dump(mode="json"),
            "sources": build_source_identity(),
            "code": code_revision,
            "decision_model": decision_model.model_dump(mode="json"),
            "reviewer": None if reviewer_model is None else reviewer_model.model_dump(mode="json"),
            "reviewer_configuration": reviewer_configuration_sha256,
            "banks": {view: sha256_json(bank) for view, bank in sorted(banks.items())},
            "error_bank": sha256_json(asked_error_bank(spec)),
        }
    )


def planned_variance_build(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    *,
    code_revision: str,
    decision_model: ModelSnapshot | None = None,
    reviewer_model: ModelSnapshot | None = None,
    reviewer_configuration_sha256: Sha256 | None = None,
) -> VarianceBuildPlan:
    """Return :func:`plan_variance_build`'s plan, computed once per process and input set.

    Takes the same arguments and returns an equal plan; ``saved`` is always read fresh.
    """
    key = _key(
        project, spec, code_revision, decision_model, reviewer_model, reviewer_configuration_sha256
    )
    with _LOCK:
        cached = None if key is None else _PLANS.get(key)
    if cached is not None:
        saved = project.artifacts.exists(cached.identity + "-result")
        return cached if saved == cached.saved else cached.model_copy(update={"saved": saved})
    plan = plan_variance_build(
        project,
        spec,
        code_revision=code_revision,
        decision_model=decision_model,
        reviewer_model=reviewer_model,
        reviewer_configuration_sha256=reviewer_configuration_sha256,
    )
    if key is not None and plan.embedding_inputs_saved == plan.embedding_inputs:
        with _LOCK:
            _PLANS[key] = plan
    return plan


def replay_or_run_variance_build(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    *,
    code_revision: str,
    tagger: DecisionsTraceTagger | None = None,
    namer: ClusterNamer | None = None,
    reviewer: CompatibilityReviewer | None = None,
) -> VarianceBuildReport:
    """Return the saved report without re-preparing the build, or run it.

    :func:`run_variance_build` prepares the whole build (the same reads as a plan) before it
    finds a saved report and returns it. This returns that same report from a reused plan with
    the run's exact identity inputs, after the same argument checks, and otherwise delegates.
    """
    if (
        spec.construction is None
        and (spec.cluster_naming is None or (namer is not None and spec.clustering is not None))
        and (tagger is None or tagger.cache is not None)
    ):
        plan = planned_variance_build(
            project,
            spec,
            code_revision=code_revision,
            decision_model=None
            if tagger is None
            else ModelSnapshot.model_validate(tagger.model_identity),
            reviewer_model=None if reviewer is None else reviewer.model,
            reviewer_configuration_sha256=None
            if reviewer is None
            else reviewer.configuration_sha256,
        )
        if plan.saved:
            return VarianceBuildReport.model_validate_json(
                project.artifacts.read_bytes(plan.identity + "-result", "report.json")
            )
    return run_variance_build(
        project,
        spec,
        code_revision=code_revision,
        tagger=tagger,
        namer=namer,
        reviewer=reviewer,
    )


def clear_planned_variance_builds() -> None:
    """Forget every reused plan (tests)."""
    with _LOCK:
        _PLANS.clear()
