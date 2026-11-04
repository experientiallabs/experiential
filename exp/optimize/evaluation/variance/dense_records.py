"""Attach saved dense findings and spans to an eligible family's training records.

:func:`attach_dense` fills each :class:`~exp.optimize.evaluation.variance.records.ContinuationGroup`
``dense`` field from saved dense assessments, one per judged episode, alongside the unchanged
outcome advantages. It checks that each dense assessment judged exactly the evidence the
episode's outcome assessment judged, so the two signals always describe the same episode. An
episode without a dense assessment carries None; its outcome advantage is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.dense_judge import load_dense
from exp.optimize.evaluation.variance.dense_reward import DenseEpisode
from exp.optimize.evaluation.variance.judge import load_assessment
from exp.optimize.evaluation.variance.records import ContinuationGroup, TrainingRecords


def attach_dense(
    project: ProjectStore,
    records: TrainingRecords,
    dense: Mapping[str, ArtifactInput],
    *,
    definition_sha256: str,
) -> TrainingRecords:
    """Return the records with dense findings and spans on every group.

    Args:
        project: Project holding the outcome and dense assessments.
        records: An eligible family's training records.
        dense: Saved dense assessment per episode id.
        definition_sha256: The dense definition every attached assessment must come from.

    Returns:
        The same records with each group's ``dense`` filled in plan order.

    Raises:
        ValueError: A dense assessment is for another episode, definition or evidence.
    """
    groups: dict[str, tuple[ContinuationGroup, ...]] = {}
    for tier, rows in records.groups.items():
        groups[tier] = tuple(
            group.model_copy(
                update={
                    "dense": tuple(
                        None
                        if slot.episode is None or slot.episode.episode_id not in dense
                        else _episode(
                            project,
                            dense[slot.episode.episode_id],
                            slot.episode.assessment,
                            definition_sha256,
                        )
                        for slot in group.episodes
                    )
                }
            )
            for group in rows
        )
    return records.model_copy(update={"groups": groups})


def _episode(
    project: ProjectStore, reference: ArtifactInput, outcome: ArtifactInput, definition: str
) -> DenseEpisode:
    """Load one dense assessment and check it describes the outcome assessment's evidence."""
    export = load_dense(project, reference)
    assessment = load_assessment(project, outcome)
    if export.definition_sha256 != definition:
        raise ValueError(f"dense assessment {reference.artifact_id} uses another definition")
    if (export.episode_id, export.evidence_sha256) != (
        assessment.episode_id,
        assessment.evidence_sha256,
    ):
        raise ValueError(
            f"dense assessment {reference.artifact_id} judged different evidence than the "
            f"episode's outcome assessment {outcome.artifact_id}"
        )
    return DenseEpisode(
        episode_id=export.episode_id,
        assessment=reference,
        definition_sha256=export.definition_sha256,
        findings=export.findings,
        episode=export.episode,
    )
