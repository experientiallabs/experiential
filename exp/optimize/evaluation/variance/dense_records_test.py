"""Training records carry each episode's dense findings and spans beside its advantage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope
from exp.common.project import artifact_input
from exp.optimize.evaluation.variance.dense_candidates import definition_candidates
from exp.optimize.evaluation.variance.dense_judge import judge_dense
from exp.optimize.evaluation.variance.dense_judge_test import (
    SHA,
    VERIFIED,
    evidence,
    fixture,
    trajectory,
    unanimous,
)
from exp.optimize.evaluation.variance.dense_records import attach_dense
from exp.optimize.evaluation.variance.episodes import EpisodeAssessment, JudgedEpisode
from exp.optimize.evaluation.variance.judged import JudgedSlot
from exp.optimize.evaluation.variance.records import ContinuationGroup, TrainingRecords
from exp.optimize.evaluation.variance.records_test import _meaning


def test_groups_carry_dense_findings_for_the_same_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judged slot gets its dense episode; a gap gets None; other evidence is refused."""
    project, judge, _ = fixture(tmp_path, monkeypatch, unanimous(VERIFIED))
    episode = evidence()
    dense_ref, export = judge_dense(
        project,
        judge,
        episode,
        definition_candidates(judge.definition),
        trajectory(),
        code_revision="test",
    )

    def assessment(evidence_sha256: str, name: str) -> JudgedEpisode:
        """Save an outcome assessment of the episode."""
        saved = EpisodeAssessment(
            episode_id=episode.episode_id,
            evidence_sha256=evidence_sha256,
            judge=SHA,
            status="judged",
            verdict="pass",
            decisive_refs=("e5",),
        )
        manifest = project.artifacts.write_json(
            artifact_id=name,
            artifact_type="episode-assessment",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="t", inputs=()
            ),
            files={"assessment.json": saved},
        )
        return JudgedEpisode(
            episode_id=episode.episode_id, assessment=artifact_input(manifest), judge=SHA
        )

    def records(slot: JudgedEpisode) -> TrainingRecords:
        """Records with one group of a judged slot and a gap."""
        group = ContinuationGroup(
            scenario_id="s",
            option_id="a",
            tier="untiered",
            meaning=_meaning(),
            judge=SHA,
            episodes=(JudgedSlot(position=0, episode=slot), JudgedSlot(position=1, gap="none")),
            advantages=(0.0, None),
        )
        return TrainingRecords(policy="p", critics=(), groups={"untiered": (group,)}, judges=())

    attached = attach_dense(
        project,
        records(assessment(episode.identity, "outcome-same")),
        {episode.episode_id: dense_ref},
        definition_sha256=judge.identity,
    )
    (group,) = attached.groups["untiered"]
    first, gap = group.dense
    assert gap is None and first is not None and first.findings == export.findings
    assert group.advantages == (0.0, None)
    with pytest.raises(ValueError, match="different evidence"):
        attach_dense(
            project,
            records(assessment("0" * 64, "outcome-other")),
            {episode.episode_id: dense_ref},
            definition_sha256=judge.identity,
        )
