"""Offline tests for the cross-talk probe: arms, controls, verdicts and replay."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.variance import crosstalk as crosstalk_module
from exp.optimize.evaluation.variance.crosstalk import (
    CROSSTALK_EXEMPLARS_TYPE,
    CROSSTALK_RESULT_TYPE,
    CROSSTALK_SPEC_TYPE,
    ArmResult,
    ContinuationProbeBackend,
    CrosstalkProbeSpec,
    ExemplarSet,
    JudgedRun,
    ProbeEpisode,
    ProbeRun,
    ProbeTarget,
    context_job,
    replay_crosstalk_probe,
    require_matched,
    run_crosstalk_probe,
    select_exemplars,
)
from exp.optimize.evaluation.variance.judged_test import (
    episode_judge,
    judge_settings,
    run_digests,
)
from exp.simulation.continuations import load_continuation_checkpoint

_SOURCE_REWARDS: tuple[float | None, ...] = (0.0, 1.0, None, 1.0, 0.0, 0.0)
"""Judged source best-of-6: positions 1 and 3 pass, position 2 is unscored."""


def _arm_of(job: ContinuationJob) -> str:
    """Name the arm a scripted job belongs to from its delivered prompt."""
    prompt = job.system_prompt or ""
    if "ARM:top" in prompt:
        return "top"
    if "ARM:random" in prompt:
        return "random"
    return "source" if job.study.repeats == len(_SOURCE_REWARDS) else "none"


@dataclass
class _Backend:
    """A scripted backend: no execution, labels per arm, delivery through the prompt.

    Attributes:
        rewards: Per-branch rewards of each target arm.
        calls: Every backend call, in order.
        broken: Make delivery also change the call budget.
    """

    rewards: dict[str, tuple[float | None, ...]]
    calls: list[str] = field(default_factory=list)
    broken: bool = False

    def run(self, job: ContinuationJob) -> ProbeRun:
        """Return a run with one completed episode per planned branch."""
        arm = _arm_of(job)
        self.calls.append(f"run:{arm}")
        digest = sha256_json(job.model_dump(mode="json"))
        episodes = tuple(
            ProbeEpisode(
                position=position,
                status="completed",
                steps=({"action": {"name": "edit", "arguments": {"row": position}}},),
                final_answer=f"answer {position}",
            )
            for position in range(job.study.repeats)
        )
        return ProbeRun(
            study=ArtifactInput(artifact_id=f"study-{arm}-{digest[:8]}", sha256=digest),
            episodes=episodes,
        )

    def judge(self, run: ProbeRun, target: ProbeTarget, job: ContinuationJob) -> JudgedRun:
        """Return the scripted labels of the run's arm."""
        arm = _arm_of(job)
        self.calls.append(f"judge:{arm}")
        rewards = _SOURCE_REWARDS if arm == "source" else self.rewards[arm]
        digest = sha256_json([run.study.artifact_id, target.judge.identity])
        return JudgedRun(
            judged=ArtifactInput(artifact_id=f"judged-{arm}-{digest[:8]}", sha256=digest),
            rewards=rewards,
        )

    def deliver(
        self,
        target: ProbeTarget,
        exemplars: ExemplarSet,
        reference: ArtifactInput,
        spec: CrosstalkProbeSpec,
    ) -> ContinuationJob:
        """Carry the exemplar block in the prompt, keeping everything else."""
        self.calls.append(f"deliver:{exemplars.arm}")
        prompt = f"ARM:{exemplars.arm}\n" + exemplars.block(spec.exemplar_characters)
        update: dict[str, object] = {"system_prompt": prompt}
        if self.broken:
            update["maximum_model_calls"] = target.job.maximum_model_calls + 1
        return target.job.model_copy(update=update)


def _spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, repeats: int = 8
) -> tuple[ProjectStore, CrosstalkProbeSpec]:
    """Register a best-of-6 source and one target with ``repeats`` branches per arm."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    judge = episode_judge(
        project,
        judge_settings(project),
        **run_digests(project, job.study.checkpoint, job.configuration),
    )
    source = job.model_copy(
        update={"study": job.study.model_copy(update={"repeats": len(_SOURCE_REWARDS)})}
    )
    target = job.model_copy(update={"study": job.study.model_copy(update={"repeats": repeats})})
    spec = CrosstalkProbeSpec(
        source=ProbeTarget(job=source, judge=judge),
        targets=(ProbeTarget(job=target, judge=judge),),
        top_k=2,
        seed=7,
        resamples=500,
    )
    return project, spec


def test_top_exemplars_are_the_best_judged_and_the_random_control_ignores_reward() -> None:
    """Unscored attempts never rank; the random draw is seeded and reproducible."""
    run = ProbeRun(
        study=ArtifactInput(artifact_id="study", sha256="0" * 64),
        episodes=tuple(ProbeEpisode(position=n, status="completed") for n in range(6)),
    )
    top, drawn = select_exemplars(run, _SOURCE_REWARDS, 2, 7)
    assert set(top) == {1, 3}
    assert len(drawn) == 2 and select_exemplars(run, _SOURCE_REWARDS, 2, 7)[1] == drawn
    with pytest.raises(ValueError, match="judged reward"):
        select_exemplars(run, (None,) * 6, 2, 7)


def test_a_positive_effect_beating_both_controls_reports_crosstalk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top exemplars lift every paired branch over both controls, so the lower bounds clear 0."""
    project, spec = _spec(tmp_path, monkeypatch)
    backend = _Backend(
        rewards={
            "none": (0.0,) * 8,
            "random": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "top": (1.0,) * 7 + (None,),
        }
    )

    reference, result = run_crosstalk_probe(project, spec, code_revision="test", backend=backend)

    assert result.verdict == "crosstalk"
    assert set(result.top) == {1, 3}
    assert [arm.arm for arm in result.arms] == ["none", "random", "top"]
    top = result.arms[2]
    assert (top.planned, top.scored, top.successes) == (8, 7, 7)
    assert top.bounds == [7 / 8, 1.0]
    by_name = {contrast.name: contrast for contrast in result.contrasts}
    assert by_name["top_vs_none"].summary["independent_units"] == 7
    assert by_name["top_vs_none"].summary["units_without_paired_outcomes"] == 1
    assert (by_name["top_vs_random"].lower or 0.0) > 0.0
    assert backend.calls[:2] == ["run:source", "judge:source"]
    store = project.artifacts
    assert store.read(reference.artifact_id).manifest.artifact_type == CROSSTALK_RESULT_TYPE
    assert store.read(result.spec.artifact_id).manifest.artifact_type == CROSSTALK_SPEC_TYPE
    saved = ExemplarSet.model_validate_json(
        store.read_bytes(result.exemplars[0].artifact_id, "exemplars.json")
    )
    assert saved.arm == "top" and [item.reward for item in saved.exemplars] == [1.0, 1.0]
    manifest = store.read(result.exemplars[1].artifact_id).manifest
    assert manifest.artifact_type == CROSSTALK_EXEMPLARS_TYPE


def test_beating_only_the_no_exemplar_control_is_not_crosstalk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When random exemplars help as much as the top ones, the effect is not reported."""
    project, spec = _spec(tmp_path, monkeypatch)
    backend = _Backend(rewards={"none": (0.0,) * 8, "random": (1.0,) * 8, "top": (1.0,) * 8})

    _, result = run_crosstalk_probe(project, spec, code_revision="test", backend=backend)

    assert result.verdict == "not_detected"
    by_name = {contrast.name: contrast for contrast in result.contrasts}
    assert (by_name["top_vs_none"].lower or 0.0) > 0.0
    assert by_name["top_vs_random"].lower == 0.0


def test_a_null_effect_is_not_detected_and_unscored_arms_have_no_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal arms are not cross-talk; an arm with no scored branch cannot be paired at all."""
    project, spec = _spec(tmp_path, monkeypatch)
    alternating = (0.0, 1.0) * 4
    null = _Backend(rewards={"none": alternating, "random": alternating, "top": alternating})
    assert run_crosstalk_probe(project, spec, code_revision="null", backend=null)[1].verdict == (
        "not_detected"
    )
    unscored = _Backend(rewards={"none": alternating, "random": alternating, "top": (None,) * 8})
    _, result = run_crosstalk_probe(project, spec, code_revision="gap", backend=unscored)
    assert result.verdict == "no_paired_outcomes"
    assert result.arms[2].bounds == [0.0, 1.0]


def test_a_saved_probe_replays_without_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second run and an explicit replay read the saved result; the backend is untouched."""
    project, spec = _spec(tmp_path, monkeypatch)
    rewards: dict[str, tuple[float | None, ...]] = {
        "none": (0.0,) * 8,
        "random": (0.0,) * 8,
        "top": (1.0,) * 8,
    }
    reference, result = run_crosstalk_probe(
        project, spec, code_revision="test", backend=_Backend(rewards=rewards)
    )
    idle = _Backend(rewards={})

    again = run_crosstalk_probe(project, spec, code_revision="test", backend=idle)

    assert again == (reference, result) and idle.calls == []
    assert replay_crosstalk_probe(project, spec, code_revision="test") == result
    with pytest.raises(ValueError, match="no saved result"):
        replay_crosstalk_probe(project, spec, code_revision="other")


def test_an_arm_that_changes_more_than_its_exemplars_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same actor, budget and sampling registration across arms is enforced, not assumed."""
    project, spec = _spec(tmp_path, monkeypatch)
    backend = _Backend(rewards={"none": (0.0,) * 8}, broken=True)
    with pytest.raises(ValueError, match="maximum_model_calls"):
        run_crosstalk_probe(project, spec, code_revision="test", backend=backend)
    control = spec.targets[0].job
    hotter = control.model_copy(
        update={"study": control.study.model_copy(update={"temperature": 0.2})}
    )
    with pytest.raises(ValueError, match="study"):
        require_matched(control, hotter)


def test_a_spec_needs_more_source_attempts_than_exemplars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The random control can only differ from the treatment when the pool exceeds k."""
    _, spec = _spec(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="more attempts than top_k"):
        CrosstalkProbeSpec.model_validate({**spec.model_dump(), "top_k": 6})


def test_context_delivery_starts_a_fresh_checkpoint_with_the_same_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm keeps the task, world and configuration; only the actor's prompt changes."""
    project, spec = _spec(tmp_path, monkeypatch)
    control = spec.targets[0].job
    start = load_continuation_checkpoint(project.artifacts, control.study.checkpoint)
    manifest = project.artifacts.write_json(
        artifact_id="exemplars-fixture",
        artifact_type=CROSSTALK_EXEMPLARS_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"exemplars.json": {"arm": "top", "exemplars": []}},
    )
    reference = artifact_input(manifest)
    if start.actor.calls:
        with pytest.raises(ValueError, match="initial checkpoint"):
            context_job(project, control, "BLOCK", reference, code_revision="test")
        return
    arm = context_job(project, control, "BLOCK", reference, code_revision="test")
    require_matched(control, arm)
    derived = load_continuation_checkpoint(project.artifacts, arm.study.checkpoint)
    assert derived.task == start.task and derived.world == start.world
    assert reference in derived.sources
    assert arm.system_prompt is not None and arm.system_prompt.endswith("BLOCK")
    assert derived.actor.messages[0].content == arm.system_prompt
    assert arm.configuration == control.configuration


def test_file_delivery_needs_a_delivery_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file-delivery registration without a way to expose the file is refused."""
    project, spec = _spec(tmp_path, monkeypatch)
    file_spec = spec.model_copy(update={"delivery": "file"})
    backend = ContinuationProbeBackend(project, code_revision="test")
    exemplars = ExemplarSet.model_validate(
        {"arm": "top", "exemplars": [{"position": 1, "reward": 1.0, "steps": []}]}
    )
    reference = ArtifactInput(artifact_id="exemplars", sha256="0" * 64)
    with pytest.raises(ValueError, match="file_delivery"):
        backend.deliver(file_spec.targets[0], exemplars, reference, file_spec)
    delivered = ContinuationProbeBackend(
        project,
        code_revision="test",
        file_delivery=lambda job, ref: job.model_copy(update={"system_prompt": ref.artifact_id}),
    ).deliver(file_spec.targets[0], exemplars, reference, file_spec)
    assert delivered.system_prompt == "exemplars"


def test_target_arms_run_at_once_and_keep_their_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every target arm is in flight together, yet arms and the result keep a loop's order."""
    rewards: dict[str, tuple[float | None, ...]] = {
        "none": (0.0,) * 8,
        "random": (0.0,) * 7 + (1.0,),
        "top": (1.0,) * 7 + (None,),
    }
    project, spec = _spec(tmp_path / "serial", monkeypatch)
    monkeypatch.setattr(crosstalk_module, "CROSSTALK_CONCURRENCY", 1)
    _, serial = run_crosstalk_probe(
        project, spec, code_revision="test", backend=_Backend(rewards=rewards)
    )
    monkeypatch.setattr(crosstalk_module, "CROSSTALK_CONCURRENCY", 6)
    project, spec = _spec(tmp_path / "parallel", monkeypatch)
    barrier = threading.Barrier(3, timeout=30)

    class _Together(_Backend):
        def run(self, job: ContinuationJob) -> ProbeRun:
            if _arm_of(job) != "source":
                barrier.wait()  # all three arms must be running at once
            return super().run(job)

    _, result = run_crosstalk_probe(
        project, spec, code_revision="test", backend=_Together(rewards=rewards)
    )
    assert [arm.arm for arm in result.arms] == ["none", "random", "top"]

    def shape(arms: tuple[ArmResult, ...]) -> list[tuple[object, ...]]:
        return [(a.target, a.arm, a.planned, a.scored, a.successes, a.bounds) for a in arms]

    assert shape(result.arms) == shape(serial.arms)
    assert [c.summary for c in result.contrasts] == [c.summary for c in serial.contrasts]
    assert result.verdict == serial.verdict
