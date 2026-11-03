"""A climb's saved rounds: registrations before dispatch, records after the decision.

Round ``n`` of a climb is registered with its incumbent and candidate prompt versions, its deadline
and its code revision before its first panel dispatches, so an interrupted round resumes the same
stage panels with the same edit and can never be swapped for another or redrawn. Its record saves
both stage measurements and the keep decision. Rounds run in order: round ``n`` needs every
earlier round's record, and its incumbent is the last kept candidate (or the baseline).
"""

from collections.abc import Sequence

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, stable_id
from exp.common.project import ArtifactStore, ProjectStore
from exp.optimize.evaluation.hillclimb.design import ClimbSpec
from exp.optimize.evaluation.hillclimb.measurement import StageMeasurement
from exp.optimize.evaluation.hillclimb.prompts import (
    PromptVersion,
    baseline_version,
    load_prompt_version,
    load_reference,
)

REGISTRATION_TYPE = "hill-climb-round-registration"
RECORD_TYPE = "hill-climb-round"


class RoundRegistration(ContractModel):
    """The edit and stage panels a round measures, fixed before its first dispatch.

    Attributes:
        climb: Climb identity.
        round: One-based round number.
        incumbent: Prompt version the edit changes.
        candidate: Edited prompt version.
        deadline: Deadline of every stage continuation; with the prompts, the design and the
            code revision it fixes each stage panel's identity.
        code_revision: Execution revision the stage panels run under.
    """

    climb: str
    round: int = Field(ge=1)
    incumbent: ArtifactInput
    candidate: ArtifactInput
    deadline: AwareDatetime
    code_revision: str = Field(min_length=1)


class RoundRecord(ContractModel):
    """One completed round: its stage measurements and the keep decision.

    Attributes:
        climb: Climb identity.
        round: One-based round number.
        incumbent: Prompt version the edit changed.
        candidate: Edited prompt version, retained whether kept or reverted.
        screen: Train measurement.
        confirm: Test measurement, absent when screening rejected the edit.
        kept: Whether the candidate becomes the incumbent of the next round.
        reason: Why the edit was kept or reverted.
    """

    climb: str
    round: int = Field(ge=1)
    incumbent: ArtifactInput
    candidate: ArtifactInput
    screen: StageMeasurement
    confirm: StageMeasurement | None = None
    kept: bool
    reason: str


def round_identity(climb: str, number: int) -> str:
    """Identity of a round's record; its registration appends ``-registration``."""
    return stable_id(RECORD_TYPE, {"climb": climb, "round": number})


def saved_registration(store: ArtifactStore, climb: str, number: int) -> RoundRegistration | None:
    """A round's saved registration, or None before its first dispatch."""
    identity = round_identity(climb, number) + "-registration"
    if not store.exists(identity):
        return None
    if store.read(identity).manifest.artifact_type != REGISTRATION_TYPE:
        raise ValueError("saved round registration has the wrong artifact type")
    return RoundRegistration.model_validate_json(store.read_bytes(identity, "registration.json"))


def saved_record(store: ArtifactStore, climb: str, number: int) -> RoundRecord | None:
    """A round's saved record, or None until the round decided."""
    identity = round_identity(climb, number)
    if not store.exists(identity):
        return None
    if store.read(identity).manifest.artifact_type != RECORD_TYPE:
        raise ValueError("saved round record has the wrong artifact type")
    record = RoundRecord.model_validate_json(store.read_bytes(identity, "round.json"))
    registration = saved_registration(store, climb, number)
    if (
        registration is None
        or (record.climb, record.round) != (climb, number)
        or (registration.incumbent, registration.candidate) != (record.incumbent, record.candidate)
    ):
        raise ValueError("saved round record differs from its registration")
    return record


def load_history(project: ProjectStore, spec: ClimbSpec, through: int) -> tuple[RoundRecord, ...]:
    """Every decided round from the first through ``through``, in order.

    Args:
        project: Project holding the climb.
        spec: Climb design.
        through: Last round to load; zero loads none.

    Returns:
        The round records.

    Raises:
        ValueError: A round in that range has not decided yet.
    """
    climb = spec.identity()
    history = []
    for number in range(1, through + 1):
        record = saved_record(project.artifacts, climb, number)
        if record is None:
            raise ValueError(
                f"round {number} of this climb has no saved decision; run or resume it first"
            )
        history.append(record)
    return tuple(history)


def incumbent_version(
    project: ProjectStore, spec: ClimbSpec, history: Sequence[RoundRecord]
) -> PromptVersion:
    """The prompt the next round edits: the last kept candidate, or the baseline."""
    for record in reversed(history):
        if record.kept:
            return load_prompt_version(project.artifacts, record.candidate)
    return baseline_version(spec.identity(), spec.baseline_prompt)


def measured_prompts(
    project: ProjectStore, spec: ClimbSpec, history: Sequence[RoundRecord]
) -> dict[str | None, int]:
    """Every prompt the climb has measured, mapped to its round (zero for the baseline)."""
    measured: dict[str | None, int] = {spec.baseline_prompt: 0}
    for record in history:
        version = load_prompt_version(project.artifacts, record.candidate)
        measured.setdefault(version.system_prompt, record.round)
    return measured


def saved_version(project: ProjectStore, version: PromptVersion) -> ArtifactInput:
    """Reference of a version that must already be saved."""
    return load_reference(project.artifacts, version.identity)
