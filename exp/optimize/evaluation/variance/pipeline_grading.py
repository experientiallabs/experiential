"""Grading every tier's group of attempts by the rubric judge and the behavior tags only.

Attempts are never graded by a start verifier. Right after the tiers' groups run:

1. Each tier's group is judged by the frozen episode judge bound to the rules stage's general
   rubric (:func:`tier_judge`), the judging every later reader reuses.
2. With ``mining.behaviors``, every attempt's steps are answered with the authored behavior bank
   (genuine-work and hack/failure-mode tags), judged passes are paired with judged failures of
   the same task, and the reviewed Added/Removed signatures are grouped
   (:mod:`~.behavior_pairs`). Groups that add a hack or failure-mode tag are the adversarial
   families (:mod:`~.behavior_adversarial`).
3. With ``authoring.adversarial`` (and a behavior library), each family gets an independently
   reviewed linter; accepted linters join a new library version and the general rubric is
   re-authored from it (:func:`revised_rubric`). Every group is then judged again, and its
   differences mined again, under the revised rubric. That rubric is frozen for the rest of
   the epoch: selectors, critics, continuations and families all read it. Linters later
   families add reach the next epoch's rubric through the library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.variance.actor_critic import admitted_concurrency
from exp.optimize.evaluation.variance.behavior_profiles import (
    LibraryBinding,
    load_library,
)
from exp.optimize.evaluation.variance.build_records import load_exposure_plan
from exp.optimize.evaluation.variance.episode_evidence import run_task_sha256, run_world_sha256
from exp.optimize.evaluation.variance.episodes import JudgeProfile, episode_reward
from exp.optimize.evaluation.variance.evidence import JUDGE_VERDICTS_TYPE, ReferenceSource
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls, judge_profile
from exp.optimize.evaluation.variance.judged import (
    EpisodeJudge,
    JudgedStudy,
    judge_study,
    judged_episode,
)
from exp.optimize.evaluation.variance.pipeline_adversarial import adversarial_library
from exp.optimize.evaluation.variance.pipeline_behaviors import behavior_differences
from exp.optimize.evaluation.variance.pipeline_inputs import (
    GeneralRubric,
    mining_rubric,
    rubric_draws,
)
from exp.optimize.evaluation.variance.pipeline_spec import (
    BuildOutputs,
    DecisionVarianceSpec,
    RulesOutputs,
    StageContext,
)
from exp.optimize.evaluation.variance.rubric_author import RUBRIC_FILE, judge_definition
from exp.simulation.continuations import load_continuation_checkpoint


def mining_profile(spec: DecisionVarianceSpec) -> JudgeProfile:
    """Return the episode-judge profile, fixed from the spec before any family rubric exists."""
    judge = spec.mining.judge
    return judge_profile(
        model_alias=judge.model_alias,
        model=judge.model,
        maximum_output_tokens=judge.output_tokens,
        evidence_tool_budget=spec.mining.judge_evidence_tool_budget,
        maximum_input_bytes=judge.maximum_input_bytes,
        prompt_set=spec.mining.judge_prompt_set,
    )


def judge_costs(
    project: ProjectStore, spec: DecisionVarianceSpec, episodes: int
) -> tuple[CostComponent, ...]:
    """Price judging ``episodes`` episodes at the profile's most draws per judgment."""
    calls = judge_model_calls(mining_profile(spec)) * episodes
    return tuple(row.scaled(calls) for row in evaluation_costs(project, spec.mining.judge))


def grading_sha256(spec: DecisionVarianceSpec) -> str:
    """Digest of every setting outside TraceML's slice that its grading reads."""
    mining, authoring = spec.mining, spec.authoring
    adversarial = authoring.adversarial_spec()
    return sha256_json(
        {
            "actors": list(mining.actors),
            "attempts": mining.attempts,
            "judge": mining.judge.model_dump(mode="json"),
            "profile": mining_profile(spec).model_dump(mode="json"),
            "behaviors": None
            if mining.behaviors is None
            else mining.behaviors.model_dump(mode="json"),
            "adversarial": None if adversarial is None else adversarial.model_dump(mode="json"),
            "library_id": authoring.library_id,
            "rubric_author": authoring.rubric_author.model_dump(mode="json"),
            "rubric_profile": authoring.rubric_profile.model_dump(mode="json"),
        }
    )


def tier_judge(
    project: ProjectStore, spec: DecisionVarianceSpec, job: ContinuationJob, rubric: ArtifactInput
) -> EpisodeJudge:
    """Bind the mining judge profile and a rubric to one tier's group job's task and world."""
    start = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    definition = judge_definition(
        project,
        mining_profile(spec),
        rubric,
        task_sha256=run_task_sha256(start),
        world_sha256=run_world_sha256(job.configuration),
    )
    return EpisodeJudge(definition=definition, settings=spec.mining.judge)


@dataclass(frozen=True)
class GroupGrading:
    """Every tier's judged group under the epoch's frozen rubric.

    Attributes:
        rubric: The rubric every group was finally judged by.
        judged: Each tier's saved judged group.
        behavior: Each tier's saved behavior differences, under ``mining.behaviors``.
    """

    rubric: GeneralRubric
    judged: dict[str, ArtifactInput]
    behavior: dict[str, ArtifactInput]

    def references(self) -> tuple[ArtifactInput, ...]:
        """Every saved artifact the grading cites."""
        return (*self.rubric.references(), *self.judged.values(), *self.behavior.values())

    def summary(self) -> JsonObject:
        """The grading's short operator summary."""
        return {
            "graded_by": self.rubric.rubric.artifact_id,
            "rubric_revised": self.rubric.revised,
            "judged": {alias: ref.artifact_id for alias, ref in self.judged.items()},
            "behavior_differences": {
                alias: ref.artifact_id for alias, ref in self.behavior.items()
            },
        }


def _judge_tiers(
    context: StageContext,
    groups: Mapping[str, tuple[ArtifactInput, ContinuationJob]],
    rubric: ArtifactInput,
    packet: ArtifactInput | None = None,
) -> tuple[dict[str, ArtifactInput], dict[str, ArtifactInput]]:
    """Judge every tier's group by ``rubric`` and mine its behavior differences, if enabled."""
    project, spec, revision = context.project, context.spec, context.code_revision
    build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
    exposure = load_exposure_plan(project, build.exposure)
    judged: dict[str, ArtifactInput] = {}
    behavior: dict[str, ArtifactInput] = {}
    for alias, (source, job) in groups.items():
        judge = tier_judge(project, spec, job, rubric)
        judged[alias], _ = judge_study(
            project,
            source,
            judge,
            configuration=job.configuration,
            code_revision=revision,
            concurrency=admitted_concurrency(project, judge.settings.model_alias),
        )
        differences = behavior_differences(
            context, job, source, judged[alias], exposure, rubric=rubric, packet=packet
        )
        if differences is not None:
            behavior[alias] = differences
    return judged, behavior


def revised_rubric(
    context: StageContext, rules: RulesOutputs, behavior: Mapping[str, ArtifactInput]
) -> GeneralRubric | None:
    """Re-author the general rubric once the groups' adversarial families have linters.

    Args:
        context: The running stage.
        rules: The rules stage, holding the general rubric and its library version.
        behavior: Each tier's behavior differences under the general rubric.

    Returns:
        The revised rubric, or None when adversarial linters are off, no family was mined, or
        every accepted linter is already in the rubric's library.
    """
    project, spec = context.project, context.spec
    authoring = spec.authoring
    adversarial = authoring.adversarial_spec()
    if (
        spec.mining.rubric is not None
        or adversarial is None
        or authoring.library_id is None
        or not behavior
    ):
        return None
    base = (
        None
        if rules.rubric_library is None
        else LibraryBinding(
            reference=rules.rubric_library, library=load_library(project, rules.rubric_library)
        )
    )
    start = load_continuation_checkpoint(project.artifacts, rules.job.study.checkpoint)
    library = adversarial_library(
        project,
        adversarial,
        tuple(behavior.values()),
        base,
        library_id=authoring.library_id,
        family=f"mining-{start.task.task_id}",
        code_revision=context.code_revision,
    )
    if library is None or library is base:
        return None
    build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
    rubric, packet = mining_rubric(context, build, rules.job, rules.rulebook, library)
    return GeneralRubric(rubric=rubric, packet=packet, library=library.reference, revised=True)


def grade_groups(
    context: StageContext,
    rules: RulesOutputs,
    groups: Mapping[str, tuple[ArtifactInput, ContinuationJob]],
) -> GroupGrading:
    """Judge every tier's group, mine its behavior signals and freeze the epoch's rubric.

    Args:
        context: The running stage.
        rules: The rules stage, holding the general rubric.
        groups: Each tier's saved group result and the group job it ran under.

    Returns:
        Every tier's judged group (and behavior differences) under the frozen rubric.
    """
    rubric = GeneralRubric(
        rubric=rules.rubric, packet=rules.rubric_packet, library=rules.rubric_library
    )
    judged, behavior = _judge_tiers(context, groups, rubric.rubric, rubric.packet)
    revised = revised_rubric(context, rules, behavior)
    if revised is not None:
        rubric = revised
        judged, behavior = _judge_tiers(context, groups, rubric.rubric, rubric.packet)
    return GroupGrading(rubric=rubric, judged=judged, behavior=behavior)


def grading_costs(
    project: ProjectStore, spec: DecisionVarianceSpec, behavior: tuple[CostComponent, ...]
) -> tuple[tuple[CostComponent, ...], str]:
    """Price judging every tier's group, mining its behavior signals and the revision.

    Args:
        project: Project whose catalog prices the roles.
        spec: The pipeline spec.
        behavior: The bound of one pass of behavior mining over every tier.

    Returns:
        The cost bound and its plan line.
    """
    mining = spec.mining
    one_pass = (*judge_costs(project, spec, mining.attempts * len(mining.actors)), *behavior)
    adversarial = spec.authoring.adversarial_spec()
    revising = (
        adversarial is not None
        and spec.authoring.library_id is not None
        and mining.behaviors is not None
        and mining.rubric is None
    )
    if not revising:
        line = (
            f"Grading: every tier's group judged by the general rubric on "
            f"{mining.judge.model_alias}"
            + (", with behavior differences mined" if mining.behaviors is not None else "")
            + "; no start verifier grades any attempt."
        )
        return one_pass, line
    assert adversarial is not None and mining.behaviors is not None
    assert adversarial.author is not None and adversarial.reviewer is not None
    families = mining.behaviors.maximum_pairs * len(mining.actors)
    calls = adversarial.profile.budget.model_calls
    linters: list[CostComponent] = []
    roles = ((adversarial.author, len(adversarial.stances)), (adversarial.reviewer, 1))
    for settings, draws in roles:
        linters.extend(
            row.scaled(families * draws * calls) for row in evaluation_costs(project, settings)
        )
    author = spec.authoring.rubric_author
    revision = tuple(
        row.scaled(rubric_draws(spec.authoring.rubric_profile))
        for row in evaluation_costs(project, author)
    )
    line = (
        f"Grading: every tier's group judged by the general rubric on {mining.judge.model_alias} "
        "with behavior differences mined; up to one linter per adversarial family "
        f"({adversarial.author.model_alias}, reviewed by {adversarial.reviewer.model_alias}); "
        f"when a linter is accepted the rubric is re-authored by {author.model_alias} and every "
        "group judged and contrasted again. The result is frozen for the epoch; no start verifier "
        "grades any attempt."
    )
    return (*one_pass, *one_pass, *linters, *revision), line


def verdict_reference(project: ProjectStore, judged: ArtifactInput) -> ReferenceSource:
    """Register a judged group's verdicts as a compact selector and critic reference.

    Each attempt's row carries the judge's verdict, reward, criterion findings, decisive events
    and limitations, or why it has no judgment. These verdicts are the only grades selectors
    and critics see.

    Args:
        project: Project holding the judged group and its assessments.
        judged: The saved judged group.

    Returns:
        The ``judge_verdicts`` reference, with the judged group as provenance.
    """
    store = project.artifacts
    labels = JudgedStudy.model_validate_json(store.read_bytes(judged.artifact_id, "judged.json"))
    attempts: list[JsonValue] = []
    for slot in labels.branches:
        assessment = judged_episode(project, slot, labels.judge)
        if assessment is None:
            attempts.append({"attempt": slot.position, "verdict": None, "gap": slot.gap})
            continue
        attempts.append(
            {
                "attempt": slot.position,
                "verdict": assessment.verdict,
                "reward": episode_reward(assessment),
                "findings": [row.model_dump(mode="json") for row in assessment.findings],
                "decisive_refs": list(assessment.decisive_refs),
                "limitations": list(assessment.limitations),
            }
        )
    record: JsonObject = {
        "source": judged.model_dump(mode="json"),
        "judge": labels.judge,
        "attempts": attempts,
        "authority": "frozen rubric judge over the epoch's rubric; the only grade on this path",
    }
    identity = stable_id("judge-verdicts-reference", {"record": sha256_json(record)})
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=JUDGE_VERDICTS_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=store.read(judged.artifact_id).manifest.code_revision,
                inputs=(judged,),
            ),
            files={"verdicts.json": record},
        )
    return ReferenceSource(
        kind="judge_verdicts",
        name="judge-verdicts",
        artifact=artifact_input(store.read(identity).manifest),
        file="verdicts.json",
    )


def rubric_reference(rubric: ArtifactInput) -> ReferenceSource:
    """Register the rubric the groups were judged by as the roles' grader rubric."""
    return ReferenceSource(kind="grader_rubric", name="rubric", artifact=rubric, file=RUBRIC_FILE)
