"""Executed evidence that a varied point is a real decision, and that its family flips.

Structural scenario gates (:mod:`contracts`) only show that a variant is well formed. A point is a
decision only when competing actions executed from the same checkpoint, continued under one frozen
policy and budget, lead to measurably different judged outcomes, and a family qualifies only when
which option is better flips across coherent variants. Spread of success rates between points is
never used as evidence here.

Three steps establish that:

- :func:`run_observation_witness` executes a scenario's witness tool calls in the restored world
  of the variant checkpoint, without any actor model, and checks that every deciding fact really
  appears in an observation, within the actor's remaining calls, and that no observation simply
  states the correct option.
- :func:`run_option_contrast` forces each option as the first action from the checkpoint and runs
  the suffixes through the preregistered action-contrast experiments, so every option gets the
  same frozen actor and budget. Every cell is judged once, as its imposed action followed by its
  suffix, by the variant's frozen episode judge
  (:func:`~exp.optimize.evaluation.variance.judged.judge_cells`); those verdicts are the only
  labels, and unscored cells stay in the denominator.
- :func:`family_flip` compares the per-variant winners: a variant has a winner only when the best
  option's lower bound is above every other option's upper bound, and the family qualifies only
  when at least two variants have different separated winners that match what their scenarios
  declared, with no separated winner contradicting its scenario.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.uncertainty import wilson_interval
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.action_contrast import (
    ActionContrast,
    ActionContrastSpec,
    ContrastCell,
    preflight_action_contrast,
    run_action_contrast,
    saved_action_contrast,
)
from exp.optimize.evaluation.continuations.intervention import (
    ActionInterventionSpec,
    preflight_intervention,
)
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    load_continuation_runtime,
    preflight_continuation_job,
)
from exp.optimize.evaluation.continuations.study import restore_continuation_world
from exp.optimize.evaluation.variance.contracts import (
    DecisionState,
    Scenario,
    WitnessStep,
    resolve_pointer,
)
from exp.optimize.evaluation.variance.judged import (
    EpisodeJudge,
    JudgeCoverage,
    JudgedCell,
    assessment_inputs,
    judge_cells,
    judged_episode,
    slot_coverage,
    slot_rewards,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.interface import Observation
from exp.simulation.continuations import load_continuation_checkpoint

# Witness execution and option contrasts


class WitnessCall(ContractModel):
    """One witness step as executed, with the observation the actor would have seen.

    Attributes:
        step: The authored witness step.
        call: The concrete tool call executed for it.
        observation: The actual observation, or None when execution raised.
        failure: Sanitized error type when execution raised; later steps are not run.
    """

    step: WitnessStep
    call: ToolCall
    observation: Observation | None = None
    failure: str | None = None


class PathEvidence(ContractModel):
    """Whether one deciding fact was shown by an executed observation.

    Attributes:
        path: Deciding JSON pointer of the scenario.
        present: Whether the path exists in the variant's evaluator state.
        value: Its value there, or None when absent.
        declared_by: Indices of witness steps whose ``reveals`` name the path.
        shown_by: Indices of witness calls whose observation content shows the value.
    """

    path: str
    present: bool
    value: JsonValue = None
    declared_by: tuple[int, ...] = ()
    shown_by: tuple[int, ...] = ()


class WitnessResult(ContractModel):
    """An executed observation witness for one scenario at its variant checkpoint.

    Attributes:
        scenario: The scenario whose witness ran.
        checkpoint: The variant checkpoint the world was restored from.
        correct_action: Action text of the scenario's correct option, searched for as a leak.
        state: Evaluator-only state of the restored world before any witness call.
        calls: Every witness step in order, as executed.
        remaining_calls: Actor model calls left at the checkpoint.
        paths: Evidence for each deciding path.
        leaks: Indices of witness calls whose observation states the correct action verbatim.
        problems: Every reason the witness fails; empty when it holds.
        scope: What the witness does and does not establish.
    """

    scenario: Scenario
    checkpoint: ArtifactInput
    correct_action: str
    state: JsonObject
    calls: tuple[WitnessCall, ...]
    remaining_calls: int = Field(ge=0)
    paths: tuple[PathEvidence, ...]
    leaks: tuple[int, ...] = ()
    problems: tuple[str, ...] = ()
    scope: str = (
        "scripted witness calls from the variant checkpoint without an actor model; shows that "
        "deciding values appear in observations, not that an actor will find or use them"
    )

    @property
    def holds(self) -> bool:
        """Whether every deciding fact was shown, within budget and without an answer leak."""
        return not self.problems


class OptionOutcome(ContractModel):
    """Judged episodes after forcing one option as the first action.

    Attributes:
        option_id: Decision option identity.
        action: The concrete forced first action.
        planned: Planned episodes, the denominator.
        assessable: Episodes the judge scored pass or fail.
        successes: Judged passes.
        unknown: Planned episodes without a judged pass or fail.
        not_met: Per rubric criterion, judged episodes that did not meet it; the imposed action
            is part of every judged episode.
        cells: Each planned cell's saved assessment or gap, the source of every count here.
        coverage: How completely the judge labelled the planned cells.
        failures: Sanitized operational failures of unavailable continuations.
        wilson_95: Two-sided 95 percent Wilson interval over assessable continuations only, or
            None when none are assessable.
        bounds: Conservative interval used for separation: the Wilson lower bound with every
            unknown counted as a failure and the Wilson upper bound with every unknown counted as
            a success, both at the planned count.
    """

    option_id: str
    action: ToolCall
    planned: int = Field(ge=1)
    assessable: int = Field(ge=0)
    successes: int = Field(ge=0)
    unknown: int = Field(ge=0)
    not_met: dict[str, int] = Field(default_factory=dict)
    cells: tuple[JudgedCell, ...] = ()
    coverage: JudgeCoverage = JudgeCoverage()
    failures: tuple[str, ...] = ()
    wilson_95: tuple[float, float] | None = None
    bounds: tuple[float, float]


class OptionContrast(ContractModel):
    """Every option of one decision forced from the same checkpoint under one frozen job.

    Attributes:
        step: The decision state's step.
        decision: The decision being contrasted.
        checkpoint: The shared starting checkpoint.
        reference_option: Option paired against every other option; it has one arm per
            contrast, so it carries more planned continuations than the others.
        repeats: Pairs per contrast.
        options: Outcomes in the decision state's option order.
        contrasts: Saved action-contrast results, one per non-reference option.
        judge: Identity of the frozen episode judge that labelled every cell.
        scope: What the contrast does and does not establish.
    """

    step: int = Field(ge=0)
    decision: str
    checkpoint: ArtifactInput
    reference_option: str
    repeats: int = Field(ge=1)
    options: tuple[OptionOutcome, ...] = Field(min_length=2)
    contrasts: tuple[ArtifactInput, ...] = ()
    judge: Sha256
    scope: str = (
        "forced first actions from one checkpoint with equal actor allowances; intervals are over "
        "independent suffix draws at this checkpoint, not over tasks; simulated effects are "
        "predictions and differences do not authorize training"
    )

    @property
    def coverage(self) -> JudgeCoverage:
        """The judge's coverage over every planned cell of every option."""
        total = JudgeCoverage(judge=self.judge)
        for option in self.options:
            total = total.plus(option.coverage)
        return total


def _bind(job: ContinuationJob, checkpoint: ArtifactInput, repeats: int) -> ContinuationJob:
    """Start the frozen job at the given checkpoint with the given repeat count."""
    study = job.study.model_copy(update={"checkpoint": checkpoint, "repeats": repeats})
    return ContinuationJob.model_validate_json(
        job.model_copy(update={"study": study}).model_dump_json()
    )


def _witness_calls(scenario: Scenario) -> tuple[ToolCall, ...]:
    """Give each witness step a call identity unique to its scenario and position."""
    return tuple(
        ToolCall(
            call_id=f"witness-{scenario.scenario_id}-{index}",
            name=step.tool,
            arguments=step.arguments,
        )
        for index, step in enumerate(scenario.witness)
    )


def _correct_action(decision: DecisionState, scenario: Scenario) -> str:
    """Return the correct option's action text, refusing an undeclared option."""
    for option in decision.options:
        if option.option_id == scenario.correct_option:
            return option.action
    raise ValueError(f"scenario {scenario.scenario_id} names an undeclared option")


def witness_identity(
    checkpoint: ArtifactInput, scenario: Scenario, job: ContinuationJob, *, code_revision: str
) -> str:
    """Freeze the witness population before any runtime code loads.

    Args:
        checkpoint: Variant checkpoint.
        scenario: Scenario whose witness runs.
        job: Frozen job already bound to the checkpoint.
        code_revision: Installed implementation identity.

    Returns:
        The stable artifact identity of the witness intent; its result adds ``-result``.
    """
    return stable_id(
        "observation-witness",
        {
            "checkpoint": checkpoint.model_dump(mode="json"),
            "scenario": sha256_json(scenario),
            "job": job.implementation(code_revision),
        },
    )


def saved_observation_witness(
    project: ProjectStore,
    *,
    checkpoint: ArtifactInput,
    scenario: Scenario,
    decision: DecisionState,
    job: ContinuationJob,
    code_revision: str,
) -> WitnessResult | None:
    """Replay a saved witness without loading runtime code, recomputing its verdict.

    Args:
        project: Trusted project.
        checkpoint: Variant checkpoint.
        scenario: Scenario whose witness ran.
        decision: Decision state declaring the scenario's options.
        job: Frozen job supplying the runtime and actor policy.
        code_revision: Installed implementation identity.

    Returns:
        The saved result, or None when no witness was saved.

    Raises:
        ValueError: The saved witness differs from its registration or its recomputed verdict.
    """
    bound = _bind(job, checkpoint, job.study.repeats)
    store = project.artifacts
    identity = witness_identity(checkpoint, scenario, bound, code_revision=code_revision)
    if not store.exists(identity + "-result"):
        return None
    result = WitnessResult.model_validate_json(
        store.read_bytes(identity + "-result", "witness.json")
    )
    expected = _verdict(
        scenario,
        checkpoint,
        _correct_action(decision, scenario),
        result.state,
        result.calls,
        result.remaining_calls,
    )
    if (
        result != expected
        or set(store.read(identity + "-result").manifest.inputs)
        != {checkpoint, artifact_input(store.read(identity).manifest)}
        or tuple(call.call for call in result.calls) != _witness_calls(scenario)
    ):
        raise ValueError("observation witness differs from its registration or evidence")
    return result


def witness_reference(
    project: ProjectStore,
    *,
    checkpoint: ArtifactInput,
    scenario: Scenario,
    job: ContinuationJob,
    code_revision: str,
) -> ArtifactInput:
    """Return the saved witness artifact, for citing it as evidence of an option contrast.

    Args:
        project: Trusted project.
        checkpoint: Variant checkpoint the witness ran from.
        scenario: Scenario whose witness ran.
        job: Frozen job the witness ran under.
        code_revision: Installed implementation identity.

    Returns:
        The ``observation-witness`` artifact.

    Raises:
        ValueError: No witness was saved for this checkpoint, scenario and job; run
            :func:`run_observation_witness` first.
    """
    bound = _bind(job, checkpoint, job.study.repeats)
    identity = witness_identity(checkpoint, scenario, bound, code_revision=code_revision)
    if not project.artifacts.exists(identity + "-result"):
        raise ValueError(
            f"scenario {scenario.scenario_id} has no saved witness at this checkpoint; run "
            "run_observation_witness before citing it"
        )
    return artifact_input(project.artifacts.read(identity + "-result").manifest)


def run_observation_witness(
    project: ProjectStore,
    *,
    checkpoint: ArtifactInput,
    scenario: Scenario,
    decision: DecisionState,
    job: ContinuationJob,
    code_revision: str,
) -> WitnessResult:
    """Execute a scenario's witness calls in the restored variant world and save what they showed.

    No actor model is called. The world is restored from the checkpoint through the job's
    explicitly selected runtime, the evaluator state is probed before any call, then every
    witness call executes in order in that one restored world. A durable intent is written first;
    an interrupted witness is never silently redispatched.

    Args:
        project: Trusted project.
        checkpoint: Variant checkpoint whose world carries the scenario's edits.
        scenario: Scenario whose witness runs.
        decision: Decision state declaring the scenario's options, for the leak check.
        job: Frozen job supplying the runtime file and actor policy; its study is rebound to
            ``checkpoint``.
        code_revision: Installed implementation identity.

    Returns:
        The saved witness; replays exactly once saved.

    Raises:
        ValueError: A witness call is not a permitted, schema-valid task tool, or an earlier
            attempt left an intent without a result.
    """
    bound = _bind(job, checkpoint, job.study.repeats)
    correct_action = _correct_action(decision, scenario)
    store = project.artifacts
    identity = witness_identity(checkpoint, scenario, bound, code_revision=code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_observation_witness(
            project,
            checkpoint=checkpoint,
            scenario=scenario,
            decision=decision,
            job=job,
            code_revision=code_revision,
        )
        if saved is not None:
            return saved
        preflight_continuation_job(project, bound)
        calls = _witness_calls(scenario)
        for call in calls:
            preflight_intervention(
                project,
                ActionInterventionSpec(base=bound, action=call, rationale="observation witness"),
            )
        if store.exists(identity):
            raise ValueError("observation witness has an unknown outcome; no implicit redispatch")
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=(checkpoint,),
        )
        registration = artifact_input(
            store.write_json(
                artifact_id=identity,
                artifact_type="observation-witness-intent",
                envelope=envelope,
                files={"scenario.json": scenario},
            )
        )
        original = load_continuation_checkpoint(store, checkpoint)
        remaining = ChatAgentRuntime(
            maximum_model_calls=bound.maximum_model_calls, system_prompt=bound.system_prompt
        ).remaining_calls(original.actor)
        executed: list[WitnessCall] = []
        with (
            load_continuation_runtime(project, bound) as runtime,
            restore_continuation_world(runtime.world, original.world_task, original.world) as world,
        ):
            state = runtime.probe(world.checkpoint())
            for step, call in zip(scenario.witness, calls, strict=True):
                try:
                    executed.append(
                        WitnessCall(step=step, call=call, observation=world.execute(call))
                    )
                except (ValueError, OSError, RuntimeError) as exc:
                    executed.append(WitnessCall(step=step, call=call, failure=type(exc).__name__))
                    break
        result = _verdict(scenario, checkpoint, correct_action, state, tuple(executed), remaining)
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="observation-witness",
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted((checkpoint, registration), key=lambda ref: ref.artifact_id)
                    )
                }
            ),
            files={"witness.json": result},
        )
        return result


def _verdict(
    scenario: Scenario,
    checkpoint: ArtifactInput,
    correct_action: str,
    state: JsonObject,
    calls: tuple[WitnessCall, ...],
    remaining: int,
) -> WitnessResult:
    """Judge executed witness observations against the variant state; pure and replayable."""
    problems: list[str] = []
    for edit in scenario.edits:
        exists, value = resolve_pointer(state, edit.path)
        if not exists or not _same_object(value, edit.value):
            problems.append(f"checkpoint state does not carry edit {edit.path}; not this variant")
    if len(scenario.witness) + 1 > remaining:
        problems.append(
            f"witness needs {len(scenario.witness)} calls plus the decision but only "
            f"{remaining} actor calls remain"
        )
    for index, call in enumerate(calls):
        if call.failure is not None:
            problems.append(f"witness call {index} failed with {call.failure}")
        elif call.observation is not None and call.observation.is_error:
            problems.append(f"witness call {index} returned an error observation")
    if len(calls) < len(scenario.witness):
        problems.append(f"only {len(calls)} of {len(scenario.witness)} witness calls ran")
    contents = [
        (index, call.observation.content)
        for index, call in enumerate(calls)
        if call.observation is not None
    ]
    paths: list[PathEvidence] = []
    for path in scenario.deciding_paths:
        exists, value = resolve_pointer(state, path)
        declared = tuple(i for i, step in enumerate(scenario.witness) if path in step.reveals)
        shown = tuple(i for i, content in contents if exists and shows_value(value, content))
        paths.append(
            PathEvidence(
                path=path, present=exists, value=value, declared_by=declared, shown_by=shown
            )
        )
        if not exists:
            problems.append(f"deciding fact {path} is absent from the variant state")
        elif not shown:
            problems.append(f"no witness observation shows deciding fact {path}")
    leaks = tuple(i for i, content in contents if states_action(correct_action, content))
    if leaks:
        problems.append(f"witness observations {list(leaks)} state the correct action verbatim")
    return WitnessResult(
        scenario=scenario,
        checkpoint=checkpoint,
        correct_action=correct_action,
        state=state,
        calls=calls,
        remaining_calls=remaining,
        paths=tuple(paths),
        leaks=leaks,
        problems=tuple(problems),
    )


_ABSENT = object()


def shows_value(value: JsonValue, content: str) -> bool:
    """Return whether observation text shows a JSON value.

    Structured content is searched for an equal JSON value at any depth. Text is searched for the
    value's JSON form, or its raw form for strings, at token boundaries so ``12`` is not found in
    ``120`` or ``1.2``.

    Args:
        value: Deciding value from the variant state.
        content: Observation content the actor sees.

    Returns:
        Whether the value appears.
    """
    try:
        parsed: object = json.loads(content)
    except ValueError:
        parsed = _ABSENT
    if parsed is not _ABSENT and _contains(parsed, value):
        return True
    forms = {json.dumps(value), json.dumps(value, separators=(",", ":"))}
    if isinstance(value, str):
        forms.add(value)
    elif isinstance(value, float) and value.is_integer():
        forms.add(str(int(value)))
    return any(form and re.search(_bounded(form), content) for form in forms)


def states_action(action: str, content: str) -> bool:
    """Return whether observation text states an action verbatim, ignoring case and spacing.

    Args:
        action: Option action text.
        content: Observation content.

    Returns:
        Whether the normalized action occurs in the normalized content.
    """
    normalized = " ".join(action.casefold().split())
    return bool(normalized) and normalized in " ".join(content.casefold().split())


def _bounded(form: str) -> str:
    """Build a pattern for a value form that cannot start or end inside a longer token."""
    start = r"(?<![\w.])" if form[0].isalnum() else ""
    end = r"(?!\.?\w)" if form[-1].isalnum() else ""
    return start + re.escape(form) + end


def _contains(node: object, value: JsonValue) -> bool:
    """Search parsed JSON for an exactly equal value."""
    if _same_object(node, value):
        return True
    if isinstance(node, dict):
        return any(_contains(item, value) for item in node.values())
    if isinstance(node, list):
        return any(_contains(item, value) for item in node)
    return False


def _same_object(left: object, right: JsonValue) -> bool:
    """Compare parsed JSON with a JSON value, distinguishing booleans from numbers."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def option_contrast_specs(
    *,
    checkpoint: ArtifactInput,
    decision: DecisionState,
    actions: Mapping[str, ToolCall],
    job: ContinuationJob,
    repeats: int,
    evidence: tuple[ArtifactInput, ...] = (),
) -> tuple[ActionContrastSpec, ...]:
    """Register one preregistered action contrast per non-reference option.

    The first declared option is the reference; every other option is contrasted against it
    under the same frozen job, so all options share checkpoint, actor and budget.

    Args:
        checkpoint: Shared starting checkpoint.
        decision: Decision state whose options are contrasted.
        actions: Concrete forced first action of every option, by option identity.
        job: Frozen job; its study is rebound to the checkpoint and repeat count.
        repeats: Pairs per contrast.
        evidence: Saved sources supporting the hypothesis, such as observation witnesses.

    Returns:
        The contrast specifications in option order.

    Raises:
        ValueError: Actions do not cover exactly the declared options, or two coincide.
    """
    if repeats < 1:
        raise ValueError("an option contrast needs at least one repeat")
    declared = [option.option_id for option in decision.options]
    if set(actions) != set(declared):
        raise ValueError("option actions must name exactly the decision's options")
    signatures = {
        json.dumps([actions[o].name, actions[o].arguments], sort_keys=True) for o in declared
    }
    if len(signatures) != len(declared):
        raise ValueError("two options map to the same concrete action")
    base = _bind(job, checkpoint, repeats)
    reference = declared[0]
    return tuple(
        ActionContrastSpec(
            base=base,
            baseline=actions[reference],
            alternative=actions[option],
            hypothesis=(
                f"At step {decision.step}, deciding {decision.decision!r}: forcing option "
                f"{option!r} instead of {reference!r} changes judged downstream outcomes."
            ),
            evidence=tuple(sorted(set(evidence), key=lambda ref: ref.artifact_id)),
        )
        for option in declared[1:]
    )


def run_option_contrast(
    project: ProjectStore,
    *,
    checkpoint: ArtifactInput,
    decision: DecisionState,
    actions: Mapping[str, ToolCall],
    job: ContinuationJob,
    repeats: int,
    judge: EpisodeJudge,
    code_revision: str,
    evidence: tuple[ArtifactInput, ...] = (),
) -> OptionContrast:
    """Force every option from one checkpoint and judge every frozen-policy episode.

    Suffix running, AB/BA ordering, durable intents and recovery come from
    :func:`run_action_contrast`. Each cell is then judged once by ``judge`` as its forced action
    plus its suffix; the mechanical grade the contrast owner records is diagnostic only. Every
    contrast is preflighted before any of them dispatches, and saved contrasts and assessments
    replay without runtime code or judge calls.

    Args:
        project: Trusted project.
        checkpoint: Shared starting checkpoint.
        decision: Decision state whose options are contrasted.
        actions: Concrete forced first action of every option, by option identity.
        job: Frozen job; its study is rebound to the checkpoint and repeat count.
        repeats: Pairs per contrast.
        judge: The frozen episode judge, fixed before any of its labels exist.
        code_revision: Installed implementation identity.
        evidence: Saved sources supporting the hypothesis, such as observation witnesses.

    Returns:
        Per-option judged outcomes with unscored cells explicit; saved as an ``option-contrast``
        artifact.
    """
    specs = option_contrast_specs(
        checkpoint=checkpoint,
        decision=decision,
        actions=actions,
        job=job,
        repeats=repeats,
        evidence=evidence,
    )
    pending = [
        spec
        for spec in specs
        if saved_action_contrast(project, spec, code_revision=code_revision) is None
    ]
    for spec in pending:
        preflight_action_contrast(project, spec, code_revision=code_revision)
    reports = [run_action_contrast(project, spec, code_revision=code_revision) for spec in specs]
    store = project.artifacts
    references = tuple(
        artifact_input(store.read(spec.identity(code_revision) + "-result").manifest)
        for spec in specs
    )
    judged = [
        judge_cells(project, report, reference, judge, code_revision=code_revision)
        for report, reference in zip(reports, references, strict=True)
    ]
    result = _option_contrast(
        project, checkpoint, decision, actions, repeats, reports, references, judged, judge
    )
    identity = stable_id(
        "option-contrast",
        {
            "specs": [spec.identity(code_revision) for spec in specs],
            "decision": sha256_json(decision),
            "judge": judge.identity,
        },
    )
    if store.exists(identity):
        saved = OptionContrast.model_validate_json(store.read_bytes(identity, "contrast.json"))
        if saved != result:
            raise ValueError("option contrast differs from its saved action contrasts")
        return saved
    store.write_json(
        artifact_id=identity,
        artifact_type="option-contrast",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    {
                        checkpoint,
                        judge.definition.rubric,
                        *references,
                        *assessment_inputs([cell.slot for cells in judged for cell in cells]),
                    },
                    key=lambda ref: ref.artifact_id,
                )
            ),
        ),
        files={"contrast.json": result},
    )
    return result


def _option_contrast(
    project: ProjectStore,
    checkpoint: ArtifactInput,
    decision: DecisionState,
    actions: Mapping[str, ToolCall],
    repeats: int,
    reports: Sequence[ActionContrast],
    references: tuple[ArtifactInput, ...],
    judged: Sequence[tuple[JudgedCell, ...]],
    judge: EpisodeJudge,
) -> OptionContrast:
    """Pool every judged cell by the option it forced; the reference pools its arm."""
    reference = decision.options[0].option_id
    cells: dict[str, list[tuple[ContrastCell, JudgedCell]]] = {
        option.option_id: [] for option in decision.options
    }
    for option, report, labelled in zip(decision.options[1:], reports, judged, strict=True):
        for cell, row in zip(report.cells, labelled, strict=True):
            cells[reference if cell.arm == "baseline" else option.option_id].append((cell, row))
    return OptionContrast(
        step=decision.step,
        decision=decision.decision,
        checkpoint=checkpoint,
        reference_option=reference,
        repeats=repeats,
        options=tuple(
            option_outcome(
                project,
                option.option_id,
                actions[option.option_id],
                cells[option.option_id],
                judge.identity,
            )
            for option in decision.options
        ),
        contrasts=references,
        judge=judge.identity,
    )


def option_outcome(
    project: ProjectStore,
    option_id: str,
    action: ToolCall,
    cells: Sequence[tuple[ContrastCell, JudgedCell]],
    judge: str,
) -> OptionOutcome:
    """Summarize one option's planned cells from their saved assessments.

    Args:
        project: Project holding the assessments.
        option_id: Decision option identity.
        action: The forced first action.
        cells: Every planned cell that forced this option, with its judged slot.
        judge: Identity of the judge that labelled them.

    Returns:
        Judged counts, coverage, a Wilson interval over scored cells, and the separation bounds
        with every unscored cell counted as a failure below and a success above.
    """
    slots = [row.slot for _, row in cells]
    labels = [reward for reward in slot_rewards(project, slots, judge) if reward is not None]
    planned, assessable, successes = len(cells), len(labels), int(sum(labels))
    unknown = planned - assessable
    not_met: dict[str, int] = {}
    for slot in slots:
        assessment = judged_episode(project, slot, judge)
        for finding in () if assessment is None else assessment.findings:
            if finding.status == "not_met":
                not_met[finding.criterion_id] = not_met.get(finding.criterion_id, 0) + 1
    return OptionOutcome(
        option_id=option_id,
        action=action,
        planned=planned,
        assessable=assessable,
        successes=successes,
        unknown=unknown,
        not_met=dict(sorted(not_met.items())),
        cells=tuple(row for _, row in cells),
        coverage=slot_coverage(project, slots, judge),
        failures=tuple(cell.failure for cell, _ in cells if cell.failure is not None),
        wilson_95=wilson_interval(successes / assessable, assessable) if assessable else None,
        bounds=(
            wilson_interval(successes / planned, planned)[0],
            wilson_interval((successes + unknown) / planned, planned)[1],
        ),
    )


# Family flips

VariantStatus = Literal["separated_matches", "separated_mismatch", "unseparated", "unmeasured"]
"""Per-variant result: a separated winner that matches or contradicts its scenario, no
separated winner yet (needs more samples), or no contrast for the variant at all."""


class VariantVerdict(ContractModel):
    """Which option wins in one variant, by interval separation.

    Attributes:
        scenario_id: The variant.
        correct_option: The option its scenario declared correct.
        status: Separation and agreement with the declared option.
        leader: Option with the highest lower bound, or None when unmeasured.
        leader_bounds: The leader's separation bounds.
        runner_up: Other option with the highest upper bound.
        runner_up_bounds: The runner-up's separation bounds.
    """

    scenario_id: str
    correct_option: str
    status: VariantStatus
    leader: str | None = None
    leader_bounds: tuple[float, float] | None = None
    runner_up: str | None = None
    runner_up_bounds: tuple[float, float] | None = None


class FlipVerdict(ContractModel):
    """Whether which option is better flips across a family's variants.

    Attributes:
        qualifies: At least two variants have different separated winners matching their
            scenarios, and no separated winner contradicts its scenario.
        winners: Distinct separated, matching winners across variants.
        variants: One verdict per scenario, in scenario order.
        triage: Variants whose separated winner contradicts the declared option; the author or
            the simulator is wrong and the family cannot qualify until that is resolved.
        undersampled: Variants without a separated winner; more samples may separate them.
    """

    qualifies: bool
    winners: tuple[str, ...]
    variants: tuple[VariantVerdict, ...]
    triage: tuple[str, ...] = ()
    undersampled: tuple[str, ...] = ()


def variant_verdict(scenario: Scenario, contrast: OptionContrast | None) -> VariantVerdict:
    """Decide one variant's winner: the best lower bound above every other upper bound.

    Args:
        scenario: The variant and its declared correct option.
        contrast: Its option contrast, or None when it was not measured.

    Returns:
        The variant's status and leading options.

    Raises:
        ValueError: The contrast does not measure the scenario's declared correct option.
    """
    if contrast is None:
        return VariantVerdict(
            scenario_id=scenario.scenario_id,
            correct_option=scenario.correct_option,
            status="unmeasured",
        )
    if scenario.correct_option not in {option.option_id for option in contrast.options}:
        raise ValueError(f"contrast for {scenario.scenario_id} lacks its correct option")
    leader = max(contrast.options, key=lambda option: option.bounds[0])
    others = [option for option in contrast.options if option.option_id != leader.option_id]
    runner_up = max(others, key=lambda option: option.bounds[1])
    status: VariantStatus
    if leader.bounds[0] <= runner_up.bounds[1]:
        status = "unseparated"
    elif leader.option_id == scenario.correct_option:
        status = "separated_matches"
    else:
        status = "separated_mismatch"
    return VariantVerdict(
        scenario_id=scenario.scenario_id,
        correct_option=scenario.correct_option,
        status=status,
        leader=leader.option_id,
        leader_bounds=leader.bounds,
        runner_up=runner_up.option_id,
        runner_up_bounds=runner_up.bounds,
    )


def family_flip(
    contrasts: Mapping[str, OptionContrast], scenarios: Sequence[Scenario]
) -> FlipVerdict:
    """Decide whether a decision family's better option flips across coherent variants.

    Spread of success between variants is not evidence; only separated per-variant winners are.

    Args:
        contrasts: Option contrast of each measured variant, by scenario identity.
        scenarios: Every variant of the family, with its declared correct option.

    Returns:
        The family verdict with per-variant statuses.

    Raises:
        ValueError: A contrast names an unknown scenario, or variants contrast different options.
    """
    identities = {scenario.scenario_id for scenario in scenarios}
    unknown = sorted(set(contrasts) - identities)
    if unknown:
        raise ValueError(f"contrasts name unknown scenarios {unknown}")
    option_sets = {
        frozenset(option.option_id for option in contrast.options)
        for contrast in contrasts.values()
    }
    if len(option_sets) > 1:
        raise ValueError("every variant must contrast the same decision options")
    variants = tuple(
        variant_verdict(scenario, contrasts.get(scenario.scenario_id)) for scenario in scenarios
    )
    winners = tuple(
        sorted({v.leader for v in variants if v.status == "separated_matches" and v.leader})
    )
    triage = tuple(v.scenario_id for v in variants if v.status == "separated_mismatch")
    return FlipVerdict(
        qualifies=len(winners) >= 2 and not triage,
        winners=winners,
        variants=variants,
        triage=triage,
        undersampled=tuple(v.scenario_id for v in variants if v.status == "unseparated"),
    )
