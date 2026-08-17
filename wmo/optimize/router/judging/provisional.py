"""Provider-free provisional judge calibration for the build wizard."""

from __future__ import annotations

from datetime import datetime

from wmo.common.judging import HumanScoreReview, JudgeCalibration, JudgeCalibrationService
from wmo.common.models import ModelCatalog
from wmo.common.project import ProjectStore, artifact_input
from wmo.optimize.router.judging.artifacts import (
    coordinate_manual_judge_calibration,
    find_provisional_calibration,
    require_review_state,
    write_production_rollout,
    write_review_state,
)
from wmo.optimize.router.judging.contracts import ManualJudgeError
from wmo.optimize.router.judging.service import (
    ManualJudgeCalibrationPlan,
    _read_setup,
    _write_lineage_split,
)
from wmo.runtime.models.registry import RuntimeModelCatalog


@coordinate_manual_judge_calibration
def bootstrap_provisional_judge(
    store: ProjectStore,
    catalog: ModelCatalog,
    plan: ManualJudgeCalibrationPlan,
    *,
    created_at: datetime,
    code_revision: str,
) -> JudgeCalibration:
    """Persist canonical zero-label judge provenance without provider work.

    Args:
        store: Project-local artifact and review store.
        catalog: Static catalog whose selected judge must match the finalized setup.
        plan: Frozen representative real-trace plan for the finalized setup.
        created_at: Time for newly materialized immutable evidence.
        code_revision: Exact producer revision for the evidence.

    Returns:
        Manifest-verified provisional calibration, including exact replay.

    Raises:
        ManualJudgeError: Setup, build, lineage, or existing review state conflicts.
    """
    state = require_review_state(store)
    if state.audit is not None or state.approved_calibration is not None:
        raise ManualJudgeError(
            "completed judge calibration cannot be replaced by provisional state"
        )
    setup = _read_setup(store, state.setup)
    if setup != plan.setup:
        raise ManualJudgeError("provisional judge plan no longer matches finalized setup")
    try:
        selected_judge, _capabilities = RuntimeModelCatalog(catalog).snapshot(setup.judge_alias)
    except ValueError as exc:
        raise ManualJudgeError(str(exc)) from exc
    if selected_judge != setup.judge_model:
        raise ManualJudgeError("configured judge identity changed after finalized setup")
    rollout_inputs = tuple(
        write_production_rollout(
            store,
            setup,
            task,
            trace,
            created_at,
            code_revision,
            allow_provider_free_source=True,
        )
        for task, trace in zip(plan.tasks, plan.traces, strict=True)
    )
    split = _write_lineage_split(
        store,
        setup,
        plan,
        rollout_inputs,
        created_at,
        code_revision,
    )
    provisional = find_provisional_calibration(store, setup, split.split_id)
    if provisional is None:
        empty_labels = HumanScoreReview.open(store).finalize(
            rubric_id=setup.rubric.artifact_id,
            code_revision=code_revision,
            created_at=created_at,
        )
        provisional = JudgeCalibrationService().bootstrap_provisional(
            store,
            rubric_id=setup.rubric.artifact_id,
            label_set_id=empty_labels.label_set_id,
            router_lineage_split_id=split.split_id,
            judge_model=setup.judge_model,
            judge_prompt=setup.prompt_template.prompt,
            created_at=created_at,
            code_revision=code_revision,
        )
    provisional_input = artifact_input(store.artifacts.read(provisional.calibration_id).manifest)
    if state.provisional_calibration not in (None, provisional_input):
        raise ManualJudgeError("selected provisional judge calibration changed")
    write_review_state(
        store,
        state.model_copy(update={"provisional_calibration": provisional_input}),
    )
    return provisional
