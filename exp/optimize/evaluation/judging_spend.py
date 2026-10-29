"""Recover exact judge request coordinates even when response artifact writes failed."""

from exp.common.core.artifacts import ArtifactInput
from exp.common.evaluations.evidence import read_rollout
from exp.common.project import ProjectStore
from exp.optimize.evaluation.judging_resume import read_judging_revision
from exp.optimize.evaluation.prepare import PreparedModelEvaluation, read_evaluation_judge
from exp.optimize.router.judging.protocol import judge_probe_id


def judge_request_coordinates(
    project: ProjectStore,
    prepared: PreparedModelEvaluation,
    revision: ArtifactInput | None,
    rollout_ids: tuple[str, ...],
) -> tuple[tuple[str, str, int], ...]:
    """Enumerate exact original and recovery probe coordinates without reading model output.

    Args:
        project: Owner of immutable rollout and judging evidence.
        prepared: Original frozen evaluation whose ledger is being reconciled.
        revision: Latest reviewed judging pass, or None for original execution.
        rollout_ids: Exact simulation set selected for the report.

    Returns:
        Distinct scope, role, ordinal coordinates for potentially dispatched judge probes.
        The request ledger determines which actually ran and their retained costs. Pairwise
        candidates include historical same-task rollouts so changing reference availability
        cannot hide an earlier paid order. Worker calls with the same model are never included.
    """
    setups = {prepared.judge_setup}
    visited: set[str] = set()
    while revision is not None:
        if revision.artifact_id in visited:
            raise ValueError("judging revision lineage contains a cycle")
        visited.add(revision.artifact_id)
        value = read_judging_revision(project, prepared, revision)
        setups.add(value.setup)
        revision = value.previous
    selected = read_evaluation_judge(project, prepared.judge_setup)
    targets = [read_rollout(project.artifacts, key) for key in rollout_ids]
    candidates = []
    if selected.prompt_template.response_shape == "pairwise":
        for key in project.artifacts.list_ids():
            if project.artifacts.read(key).manifest.artifact_type == "rollout":
                candidates.append(read_rollout(project.artifacts, key))
    scopes: set[str] = set()
    for setup in setups:
        for target, pointer in targets:
            if selected.prompt_template.response_shape != "pairwise":
                scopes.add(judge_probe_id(setup, pointer, None, "single"))
                continue
            for reference, reference_input in candidates:
                if (
                    reference.task_id == target.task_id
                    and reference.rollout_id != target.rollout_id
                ):
                    scopes.add(judge_probe_id(setup, pointer, reference_input, "forward"))
                    scopes.add(judge_probe_id(setup, pointer, reference_input, "reverse"))
    return tuple((scope, "judge", 0) for scope in sorted(scopes))
