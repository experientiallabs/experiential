"""One climb round: register one edit, screen it on train, confirm it on test, decide.

The round's edit is written by a person or chosen from an analyst proposal set made after the
previous round. Before any dispatch the round saves the incumbent and candidate prompt versions
and registers them, with the round's deadline and code revision, under the round number. A
resumed round therefore measures the same edit on the same stage panels, reusing their completed
and unknown cells; it can never be re-run with another edit or redrawn under a new deadline or
revision. Screening compares the two prompts on the train lineages;
an edit that passes is confirmed on the test lineages with fresh repeats. The record keeps the
edit only when confirmation passes, and every candidate stays saved either way.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.hillclimb.design import ClimbSpec, ClimbSplit, climb_split
from exp.optimize.evaluation.hillclimb.history import (
    RECORD_TYPE,
    REGISTRATION_TYPE,
    RoundRecord,
    RoundRegistration,
    incumbent_version,
    load_history,
    measured_prompts,
    round_identity,
    saved_record,
    saved_registration,
)
from exp.optimize.evaluation.hillclimb.prompts import (
    PromptEdit,
    PromptVersion,
    edited_version,
    load_prompt_version,
    persist_prompt_version,
)
from exp.optimize.evaluation.hillclimb.proposals import load_edit_proposals
from exp.optimize.evaluation.hillclimb.stages import (
    remeasure,
    run_stage,
    stage_attempts,
    stage_jobs,
    validate_stage,
)


class ProposalChoice(ContractModel):
    """One edit chosen from an analyst proposal set.

    Attributes:
        proposals: Saved proposal set made after the previous round.
        index: Zero-based edit in that set.
    """

    proposals: ArtifactInput
    index: int = Field(ge=0, strict=True)


class ClimbRound(ContractModel):
    """The ``climb-round`` payload: one edit measured in one numbered round.

    Attributes:
        climb: Climb design.
        round: One-based round number; every earlier round must have decided.
        edit: Operator-written edit, or None when choosing an analyst proposal.
        choice: Analyst proposal to measure, or None with an operator edit.
        deadline: Deadline of every continuation the round dispatches.
    """

    climb: ClimbSpec
    round: int = Field(ge=1, le=200, strict=True)
    edit: PromptEdit | None = None
    choice: ProposalChoice | None = None
    deadline: AwareDatetime

    @model_validator(mode="after")
    def one_edit(self) -> Self:
        """Require exactly one edit source."""
        if (self.edit is None) == (self.choice is None):
            raise ValueError("a climb round measures exactly one edit or one proposal choice")
        return self


@dataclass(frozen=True)
class RoundPlan:
    """What a round compares, derived without writes, credentials or provider calls.

    Attributes:
        climb: Climb identity.
        split: Lineage split.
        history: Every earlier decided round.
        incumbent: Prompt the edit changes.
        candidate: Edited prompt.
    """

    climb: str
    split: ClimbSplit
    history: tuple[RoundRecord, ...]
    incumbent: PromptVersion
    candidate: PromptVersion

    @property
    def arms(self) -> tuple[PromptVersion, PromptVersion]:
        """Incumbent then candidate, the stage panel's actor order."""
        return self.incumbent, self.candidate


def plan_round(project: ProjectStore, request: ClimbRound) -> RoundPlan:
    """Resolve the round's incumbent and candidate and check both stages.

    Args:
        project: Project holding the climb.
        request: Round payload.

    Returns:
        The plan, with nothing saved.

    Raises:
        ValueError: An earlier round is undecided, the design is underpowered, the edit was
            already measured or changes nothing, or a case cannot run the edited prompt.
    """
    spec = request.climb
    climb = spec.identity()
    split = climb_split(project, spec)
    history = load_history(project, spec, request.round - 1)
    incumbent = incumbent_version(project, spec, history)
    edit, proposal = request.edit, None
    if request.choice is not None:
        proposals = load_edit_proposals(project.artifacts, request.choice.proposals)
        if (proposals.climb, proposals.after_round) != (climb, request.round - 1):
            raise ValueError(
                f"round {request.round} can only choose from proposals made after round "
                f"{request.round - 1} of this climb"
            )
        if request.choice.index >= len(proposals.proposals.edits):
            raise ValueError("the chosen proposal index is outside the proposal set")
        if load_prompt_version(project.artifacts, proposals.incumbent) != incumbent:
            raise ValueError("the proposals edit another prompt than this round's incumbent")
        edit = proposals.proposals.edits[request.choice.index].edit()
        proposal = request.choice.proposals
    assert edit is not None
    candidate = edited_version(incumbent, edit, proposal=proposal)
    measured = measured_prompts(project, spec, history)
    if candidate.system_prompt in measured:
        raise ValueError(
            f"this exact system prompt was already measured in round "
            f"{measured[candidate.system_prompt]}; re-measuring it would spend fresh test repeats "
            "on a known edit, so propose a different change"
        )
    plan = RoundPlan(
        climb=climb, split=split, history=history, incumbent=incumbent, candidate=candidate
    )
    for stage in ("screen", "confirm"):
        validate_stage(project, spec, split, stage, plan.arms)
    return plan


def round_jobs(project: ProjectStore, request: ClimbRound) -> tuple[ContinuationJob, ...]:
    """Every continuation job a round could dispatch, both arms, for cost review."""
    split = climb_split(project, request.climb)
    return tuple(
        job
        for stage in ("screen", "confirm")
        for job in stage_jobs(request.climb, split, stage, request.deadline)
        for _ in range(2)
    )


def round_attempts(project: ProjectStore, request: ClimbRound) -> int:
    """Maximum actor HTTP attempts of a round when screening passes."""
    split = climb_split(project, request.climb)
    return sum(stage_attempts(request.climb, split, stage) for stage in ("screen", "confirm"))


def saved_round(project: ProjectStore, request: ClimbRound) -> RoundRecord | None:
    """Replay a decided round, refusing a request for another edit or a changed measurement.

    Args:
        project: Project holding the climb.
        request: Round payload.

    Returns:
        The saved record, or None while the round is undecided.
    """
    spec = request.climb
    record = saved_record(project.artifacts, spec.identity(), request.round)
    if record is None:
        return None
    _require_registered_edit(project, request, record.incumbent, record.candidate)
    split = climb_split(project, spec)
    arms = (
        load_prompt_version(project.artifacts, record.incumbent),
        load_prompt_version(project.artifacts, record.candidate),
    )
    for measurement in (record.screen, record.confirm):
        if measurement is not None:
            remeasure(project, spec, split, measurement, arms)
    return record


def _require_registered_edit(
    project: ProjectStore, request: ClimbRound, incumbent: ArtifactInput, candidate: ArtifactInput
) -> None:
    """Refuse a request whose edit differs from the one registered for its round."""
    parent = load_prompt_version(project.artifacts, incumbent)
    registered = load_prompt_version(project.artifacts, candidate)
    if request.choice is not None:
        proposals = load_edit_proposals(project.artifacts, request.choice.proposals)
        if request.choice.index >= len(proposals.proposals.edits):
            raise ValueError("the chosen proposal index is outside the proposal set")
        edit = proposals.proposals.edits[request.choice.index].edit()
        expected = edited_version(parent, edit, proposal=request.choice.proposals)
    else:
        assert request.edit is not None
        expected = edited_version(parent, request.edit)
    if expected != registered:
        raise ValueError(
            f"round {request.round} is registered for another edit; resume it with that edit "
            f"(prompt version {candidate.artifact_id}) or continue with the next round"
        )


def run_climb_round(
    project: ProjectStore, request: ClimbRound, *, code_revision: str
) -> RoundRecord:
    """Register, screen, confirm and decide one round, or replay its saved decision.

    Args:
        project: Project holding the climb.
        request: Round payload.
        code_revision: Installed execution revision of the round's panels.

    Returns:
        The decided round record.

    Raises:
        ValueError: The plan is invalid, the round is registered for another edit, deadline or
            code revision, or the train set is saturated for the incumbent.
        filelock.Timeout: Another process holds this climb or one of its continuations.
    """
    spec, store = request.climb, project.artifacts
    climb = spec.identity()
    with CoordinationLock(store.project_directory / f"{climb}.lock", timeout=0):
        saved = saved_round(project, request)
        if saved is not None:
            return saved
        now = datetime.now(UTC)
        registration = saved_registration(store, climb, request.round)
        if registration is not None:
            _require_registered_edit(
                project, request, registration.incumbent, registration.candidate
            )
            if (registration.deadline, registration.code_revision) != (
                request.deadline,
                code_revision,
            ):
                raise ValueError(
                    f"round {request.round} was registered with deadline "
                    f"{registration.deadline.isoformat()} under code revision "
                    f"{registration.code_revision}; resume it with both so its stage panels are "
                    "reused, never redrawn"
                )
        plan = plan_round(project, request)
        references = tuple(
            persist_prompt_version(store, version, created_at=now, code_revision=code_revision)
            for version in plan.arms
        )
        identity = round_identity(climb, request.round)
        if registration is None:
            store.write_json(
                artifact_id=identity + "-registration",
                artifact_type=REGISTRATION_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=now,
                    code_revision=code_revision,
                    inputs=tuple(sorted(references, key=lambda ref: ref.artifact_id)),
                ),
                files={
                    "registration.json": RoundRegistration(
                        climb=climb,
                        round=request.round,
                        incumbent=references[0],
                        candidate=references[1],
                        deadline=request.deadline,
                        code_revision=code_revision,
                    )
                },
            )
            registration = saved_registration(store, climb, request.round)
            assert registration is not None
        # The saved deadline, not the request's equal instant in another offset, keys both panels.
        screen = run_stage(
            project,
            spec,
            plan.split,
            "screen",
            f"round {request.round} screen",
            plan.arms,
            registration.deadline,
            code_revision=code_revision,
        )
        confirm = None
        if screen.passed:
            confirm = run_stage(
                project,
                spec,
                plan.split,
                "confirm",
                f"round {request.round} confirm",
                plan.arms,
                registration.deadline,
                code_revision=code_revision,
            )
        decided = confirm or screen
        record = RoundRecord(
            climb=climb,
            round=request.round,
            incumbent=references[0],
            candidate=references[1],
            screen=screen,
            confirm=confirm,
            kept=confirm is not None and confirm.passed,
            reason=f"{decided.stage}: {decided.reason}",
        )
        inputs = {
            artifact_input(store.read(identity + "-registration").manifest),
            *references,
            screen.panel,
            *(() if confirm is None else (confirm.panel,)),
        }
        store.write_json(
            artifact_id=identity,
            artifact_type=RECORD_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"round.json": record},
        )
        return record
