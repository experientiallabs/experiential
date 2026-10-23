"""Bounded hardening of one qualified world: harder variants of the same case until a panel splits.

Level zero is a saved world-proposals experiment and one of its qualified candidates. Each level
checks every qualified candidate for validity, then runs a registered strong-actor panel on the
valid ones. A candidate whose panel separates (some known attempts fail, not all) is kept. When
none is kept and one saturates under the registered rule, the world author is asked for harder
worlds of the same case (more state, distractors and multi-step dependencies; the instruction is
unchanged), and the existing scenario-search controls run again on every new world. The first
saturated candidate in proposal order is the source of the next level, up to a registered number
of levels.

Every level's proposals, searches, validity reports and panels are retained, including failures.
A hardened variant is a scenario variation of the same base task: it keeps the source lineage and
never adds a task family. Pass-rate bands are reported, never used to change a verifier.
"""

from datetime import UTC, datetime
from typing import Literal

from filelock import Timeout
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.actors import (
    ActorPanelSpec,
    PanelActor,
    preflight_actor_panel,
    run_actor_panel,
    saved_actor_panel,
)
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.authoring import AuthorSettings, preflight_author
from exp.optimize.evaluation.factory.difficulty import (
    PanelDifficulty,
    PassBand,
    SaturationRule,
    panel_difficulty,
)
from exp.optimize.evaluation.factory.validity import (
    TaskValidityReport,
    TaskValiditySpec,
    ValiditySettings,
    preflight_task_validity,
    run_task_validity,
    saved_task_validity,
)
from exp.optimize.evaluation.factory.variants import load_task_variant
from exp.optimize.evaluation.factory.worlds import (
    WorldHardening,
    WorldProposals,
    WorldProposalSpec,
    propose_worlds,
    world_proposal_request,
)
from exp.simulation.continuations import load_continuation_checkpoint

Disposition = Literal[
    "unqualified", "invalid", "undetermined", "saturated", "all_fail", "kept", "operational_failure"
]


class TaskHardeningSpec(ContractModel):
    """A registered hardening experiment over one qualified seed world.

    Attributes:
        seed: Saved ``world-proposals`` experiment supplying level zero, the template, the
            independent acceptance and the cited research of this case.
        candidate: Proposal index of the qualified seed candidate within ``seed``.
        author: World author for harder levels; the seed's own draw is never repeated.
        actors: Strong panel, two to ten frozen models, run on every valid candidate.
        repeats: Attempts per actor per candidate.
        minimum_difference: Descriptive adjacent-tier threshold passed to the panel.
        saturation: When the panel counts as solved; registered before any panel runs.
        band: Reporting band for pass rates; never a keep threshold or a tuning target.
        validity: Leak, closed-book and control checks every candidate must pass to be kept.
        maximum_levels: Harder levels the author may be asked for after the seed level.
    """

    seed: ArtifactInput
    candidate: int = Field(ge=0, le=3, strict=True)
    author: AuthorSettings
    actors: tuple[PanelActor, ...] = Field(min_length=2, max_length=10)
    repeats: int = Field(default=4, ge=1, le=100, strict=True)
    minimum_difference: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)
    saturation: SaturationRule = Field(default_factory=SaturationRule)
    band: PassBand = Field(default_factory=PassBand)
    validity: ValiditySettings
    maximum_levels: int = Field(default=2, ge=1, le=5, strict=True)

    def identity(self, code_revision: str) -> str:
        """Bind seed, models, rules and level allowance before any call."""
        return stable_id("task-hardening", {"spec": sha256_json(self), "code": code_revision})


class HardeningCandidate(ContractModel):
    """One proposed world at one level and every stage it reached.

    Attributes:
        proposal: Proposal index within its level's world proposals.
        search: Saved scenario-search result that ran its controls, when one ran.
        cell: Search cell index, when the world was bound.
        variant: Published scenario variant, only when every control passed.
        refusals: Binding or control refusals, retained verbatim.
        validity: Saved validity report, when checks ran.
        rejections: Reasons of every rejecting or unknown validity check, retained verbatim.
        panel: Saved actor-panel report, only for a valid candidate.
        difficulty: Saturation outcome and pass-rate band from that panel.
        failure: Sanitized operational error class.
        disposition: Final disposition under the registered rules.
    """

    proposal: int
    search: ArtifactInput | None = None
    cell: int | None = None
    variant: ArtifactInput | None = None
    refusals: tuple[str, ...] = ()
    validity: ArtifactInput | None = None
    rejections: tuple[str, ...] = ()
    panel: ArtifactInput | None = None
    difficulty: PanelDifficulty | None = None
    failure: str | None = None
    disposition: Disposition


class HardeningLevel(ContractModel):
    """Every candidate of one level, in proposal order.

    Attributes:
        level: Zero for the seed, then each harder level.
        proposals: Saved world proposals for this level.
        source: Previous-level candidate this level hardens; None for the seed level.
        candidates: Every examined candidate; the seed level examines only the seed candidate.
        failure: Sanitized author or binding failure that left the level without candidates.
    """

    level: int
    proposals: ArtifactInput | None = None
    source: int | None = None
    candidates: tuple[HardeningCandidate, ...] = ()
    failure: str | None = None


class TaskHardeningReport(ContractModel):
    """Complete hardening inventory for one case.

    Attributes:
        spec: Exact registration.
        lineage: Source lineage every variant keeps; this experiment is one task family.
        levels: Every level in order.
        status: A candidate was kept, the level limit was reached while saturated, no
            saturated source remained, or an operational failure stopped the experiment.
        counts: Candidates by disposition, with one family regardless of variant count.
        scope: Limits of the result.
    """

    spec: TaskHardeningSpec
    lineage: str
    levels: tuple[HardeningLevel, ...]
    status: Literal["kept", "saturated_at_limit", "exhausted", "operationally_incomplete"]
    counts: JsonObject
    scope: str = (
        "development hardening of one source family: kept variants separate this panel on "
        "this case only; bands describe panel-actor means with intervals at the actor count; "
        "related variants never add task families; requires fresh qualification before delivery"
    )


def _seed(project: ProjectStore, reference: ArtifactInput) -> tuple[WorldProposals, str]:
    """Load and replay the saved seed experiment without buying a draw."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "world-proposals":
        raise ValueError("hardening seed must be an exact saved world-proposals artifact")
    report = WorldProposals.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "proposals.json")
    )
    if not project.artifacts.exists(report.spec.identity(manifest.code_revision) + "-response"):
        raise ValueError("hardening seed lost its saved author response")
    if propose_worlds(project, report.spec, code_revision=manifest.code_revision) != report:
        raise ValueError("hardening seed does not reproduce its saved world proposals")
    return report, manifest.code_revision


def _search_cell(report: WorldProposals, proposal: int) -> int | None:
    """Map a proposal index to its scenario-search cell, or None when it was not bound."""
    if report.cells[proposal].candidate is None or report.search is None:
        return None
    return sum(cell.candidate is not None for cell in report.cells[:proposal])


def _search_reference(
    project: ProjectStore, report: WorldProposals, revision: str
) -> ArtifactInput | None:
    """The saved scenario-search result of one level, when its native search ran."""
    if report.search is None:
        return None
    identity = report.search.spec.identity(revision) + "-result"
    return artifact_input(project.artifacts.read(identity).manifest)


def _panel(
    project: ProjectStore, spec: TaskHardeningSpec, variant: ArtifactInput
) -> ActorPanelSpec:
    """One-case panel over the variant's own job with the registered repeats."""
    job = load_executable_task(project, load_task_variant(project, variant).task).job
    job = job.model_copy(update={"study": job.study.model_copy(update={"repeats": spec.repeats})})
    return ActorPanelSpec(
        cases=(job,), actors=spec.actors, minimum_difference=spec.minimum_difference
    )


def _validity(spec: TaskHardeningSpec, search: ArtifactInput, cell: int) -> TaskValiditySpec:
    """Validity registration for one candidate under the shared registered settings."""
    return TaskValiditySpec(**spec.validity.model_dump(), search=search, cell=cell)


def _author_spec(seed: WorldProposalSpec, spec: TaskHardeningSpec) -> WorldProposalSpec:
    """The seed experiment under the registered hardening author, before hardening context."""
    return WorldProposalSpec.model_validate(
        {**seed.model_dump(mode="json"), **spec.author.model_dump(mode="json"), "hardening": None}
    )


def seed_variant(project: ProjectStore, spec: TaskHardeningSpec) -> ArtifactInput:
    """Resolve the qualified seed variant, refusing an unqualified starting point."""
    report, revision = _seed(project, spec.seed)
    if spec.candidate >= len(report.cells):
        raise ValueError("hardening candidate is outside the seed proposals")
    cell = _search_cell(report, spec.candidate)
    search = _search_reference(project, report, revision)
    if cell is None or search is None or report.search is None:
        raise ValueError("hardening seed candidate was never bound to a native search")
    variant = report.search.cells[cell].variant
    if variant is None:
        raise ValueError("hardening starts from a qualified candidate; its controls did not pass")
    return variant


def seed_proposals(project: ProjectStore, spec: TaskHardeningSpec) -> WorldProposals:
    """The verified saved seed experiment."""
    return _seed(project, spec.seed)[0]


def hardening_panel(project: ProjectStore, spec: TaskHardeningSpec) -> ActorPanelSpec:
    """The registered panel over the seed variant; harder variants share its actor job."""
    return _panel(project, spec, seed_variant(project, spec))


def preflight_task_hardening(project: ProjectStore, spec: TaskHardeningSpec) -> None:
    """Check the seed, its validity and panel inputs, and the author, without any calls."""
    variant = seed_variant(project, spec)
    report, revision = _seed(project, spec.seed)
    search = _search_reference(project, report, revision)
    cell = _search_cell(report, spec.candidate)
    if search is None or cell is None:
        raise ValueError("hardening seed candidate was never bound to a native search")
    preflight_task_validity(project, _validity(spec, search, cell))
    preflight_actor_panel(project, _panel(project, spec, variant))
    author = _author_spec(report.spec, spec)
    preflight_author(project, spec.author, world_proposal_request(project, author))


def _candidate(
    project: ProjectStore,
    spec: TaskHardeningSpec,
    report: WorldProposals,
    revision: str,
    proposal: int,
    code_revision: str,
) -> HardeningCandidate:
    """Take one proposed world through controls, validity and the panel, retaining each stage."""
    binding = report.cells[proposal]
    cell = _search_cell(report, proposal)
    search = _search_reference(project, report, revision)
    if cell is None or search is None or report.search is None:
        reason = binding.refusal or "fewer than two distinct bound worlds; no native search ran"
        return HardeningCandidate(proposal=proposal, refusals=(reason,), disposition="unqualified")
    searched = report.search.cells[cell]
    row = HardeningCandidate(
        proposal=proposal,
        search=search,
        cell=cell,
        variant=searched.variant,
        refusals=searched.refusals,
        disposition="unqualified",
    )
    if searched.variant is None:
        return row
    try:
        validity = run_task_validity(
            project, _validity(spec, search, cell), code_revision=code_revision
        )
        row = row.model_copy(
            update={
                "validity": _artifact(project, validity.spec.identity(code_revision)),
                "rejections": rejections(validity),
            }
        )
        if validity.status != "valid":
            disposition = "invalid" if validity.status == "rejected" else "undetermined"
            return row.model_copy(update={"disposition": disposition})
        panel_spec = _panel(project, spec, searched.variant)
        panel = run_actor_panel(project, panel_spec, code_revision=code_revision)
        difficulty = panel_difficulty(panel.metrics, spec.saturation, spec.band)
    except Timeout:
        raise  # Busy work remains pending; the owning process may still finish it.
    except (ValueError, OSError, RuntimeError) as error:
        return row.model_copy(
            update={"failure": type(error).__name__, "disposition": "operational_failure"}
        )
    return row.model_copy(
        update={
            "panel": _artifact(project, panel_spec.identity(code_revision)),
            "difficulty": difficulty,
            "disposition": "kept" if difficulty.outcome == "separating" else difficulty.outcome,
        }
    )


def rejections(report: TaskValidityReport) -> tuple[str, ...]:
    """Every rejecting or unknown validity finding, prefixed by its check."""
    return tuple(
        check.name + ": " + reason
        for check in report.checks
        if check.status in ("reject", "unknown")
        for reason in check.reasons or ("no reason recorded",)
    )


def _artifact(project: ProjectStore, identity: str) -> ArtifactInput:
    """Reference to a child's saved ``-result`` artifact."""
    return artifact_input(project.artifacts.read(identity + "-result").manifest)


def _proposals(project: ProjectStore, reference: ArtifactInput) -> WorldProposals:
    """Read a level's saved world proposals."""
    return WorldProposals.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "proposals.json")
    )


def _next_spec(
    spec: TaskHardeningSpec, seed: WorldProposalSpec, report: WorldProposals, level: HardeningLevel
) -> tuple[WorldProposalSpec, int] | None:
    """Harden the first saturated candidate in proposal order, or stop when none saturated."""
    for row in level.candidates:
        if row.disposition == "saturated" and row.panel is not None:
            assets = report.cells[row.proposal].assets
            if assets is None:
                raise ValueError("saturated candidate lost its compiled world assets")
            hardening = WorldHardening(level=level.level + 1, assets=assets, panel=row.panel)
            return (
                _author_spec(seed, spec).model_copy(update={"hardening": hardening}),
                row.proposal,
            )
    return None


def _level(
    project: ProjectStore,
    spec: TaskHardeningSpec,
    index: int,
    previous: tuple[WorldProposals, HardeningLevel] | None,
    code_revision: str,
) -> HardeningLevel:
    """Examine the seed candidate, or buy one harder draw and examine every new world."""
    seed, revision = _seed(project, spec.seed)
    if index == 0:
        row = _candidate(project, spec, seed, revision, spec.candidate, code_revision)
        return HardeningLevel(level=0, proposals=spec.seed, candidates=(row,))
    following = None if previous is None else _next_spec(spec, seed.spec, *previous)
    if following is None:
        raise ValueError("hardening level has no saturated source")
    author, source = following
    try:
        report = propose_worlds(project, author, code_revision=code_revision)
    except Timeout:
        raise  # Busy work remains pending; the owning process may still finish it.
    except (ValueError, OSError, RuntimeError) as error:
        return HardeningLevel(level=index, source=source, failure=type(error).__name__)
    return HardeningLevel(
        level=index,
        proposals=artifact_input(project.artifacts.read(author.identity(code_revision)).manifest),
        source=source,
        candidates=tuple(
            _candidate(project, spec, report, code_revision, proposal, code_revision)
            for proposal in range(len(report.cells))
        ),
    )


def _status(
    spec: TaskHardeningSpec, levels: tuple[HardeningLevel, ...]
) -> Literal["kept", "saturated_at_limit", "exhausted", "operationally_incomplete"]:
    """Overall disposition under the registered stopping rule."""
    rows = [row for level in levels for row in level.candidates]
    if any(row.disposition == "kept" for row in rows):
        return "kept"
    last = levels[-1]
    if last.failure is not None or any(
        row.disposition == "operational_failure" for row in last.candidates
    ):
        return "operationally_incomplete"
    if last.level == spec.maximum_levels and any(
        row.disposition == "saturated" for row in last.candidates
    ):
        return "saturated_at_limit"
    return "exhausted"


def _report(
    spec: TaskHardeningSpec, lineage: str, levels: tuple[HardeningLevel, ...]
) -> TaskHardeningReport:
    """Assemble status and counts; variants never add families."""
    rows = [row for level in levels for row in level.candidates]
    dispositions: JsonObject = {
        name: sum(row.disposition == name for row in rows)
        for name in (
            "unqualified",
            "invalid",
            "undetermined",
            "saturated",
            "all_fail",
            "kept",
            "operational_failure",
        )
    }
    return TaskHardeningReport(
        spec=spec,
        lineage=lineage,
        levels=levels,
        status=_status(spec, levels),
        counts={
            "families": 1,
            "levels": len(levels),
            "candidates": len(rows),
            "variants": sum(row.variant is not None for row in rows),
            "dispositions": dispositions,
        },
    )


def _lineage(project: ProjectStore, seed: WorldProposals) -> str:
    """The base task's source lineage, shared by every variant."""
    base = load_executable_task(project, seed.spec.base)
    return load_continuation_checkpoint(
        project.artifacts, base.job.study.checkpoint
    ).task.lineage_group_id


def _level_inputs(level: HardeningLevel) -> set[ArtifactInput]:
    """Every child artifact a level depends on."""
    refs = {level.proposals} if level.proposals is not None else set()
    for row in level.candidates:
        refs.update(ref for ref in (row.search, row.variant, row.validity, row.panel) if ref)
    return refs


def _verify_level(
    project: ProjectStore,
    spec: TaskHardeningSpec,
    level: HardeningLevel,
    lineage: str,
    revision: str,
) -> None:
    """Re-derive candidate outcomes from saved children and refuse a changed lineage."""
    for row in level.candidates:
        if row.variant is not None and load_task_variant(project, row.variant).lineage != lineage:
            raise ValueError("hardened variant left its source lineage")
        if row.validity is not None and row.search is not None and row.cell is not None:
            validity = saved_task_validity(
                project, _validity(spec, row.search, row.cell), code_revision=revision
            )
            if (
                validity is None
                or _artifact(project, validity.spec.identity(revision)) != row.validity
                or rejections(validity) != row.rejections
            ):
                raise ValueError("hardening lost a saved validity report")
        if row.panel is not None and row.variant is not None:
            panel_spec = _panel(project, spec, row.variant)
            panel = saved_actor_panel(project, panel_spec, code_revision=revision)
            if (
                panel is None
                or _artifact(project, panel_spec.identity(revision)) != row.panel
                or panel_difficulty(panel.metrics, spec.saturation, spec.band) != row.difficulty
            ):
                raise ValueError("hardening panel outcome does not reproduce")


def saved_task_hardening(
    project: ProjectStore, spec: TaskHardeningSpec, *, code_revision: str
) -> TaskHardeningReport | None:
    """Re-derive a saved hardening report from saved levels and children, without calls."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    report = TaskHardeningReport.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    if (
        report.spec != spec
        or TaskHardeningSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec
    ):
        raise ValueError("saved hardening report differs from its registration")
    seed, _ = _seed(project, spec.seed)
    if report.lineage != _lineage(project, seed):
        raise ValueError("saved hardening report changed its source lineage")
    for index, level in enumerate(report.levels):
        saved = HardeningLevel.model_validate_json(
            store.read_bytes(f"{identity}-level-{index}", "level.json")
        )
        if saved != level or level.level != index:
            raise ValueError("saved hardening level differs from its report")
        if index:
            previous = report.levels[index - 1]
            expected = None
            if previous.proposals is not None:
                expected = _next_spec(
                    spec, seed.spec, _proposals(project, previous.proposals), previous
                )
            if (
                expected is None
                or level.source != expected[1]
                or (
                    level.proposals is not None
                    and _proposals(project, level.proposals).spec != expected[0]
                )
            ):
                raise ValueError("saved hardening level did not harden its registered source")
        _verify_level(project, spec, level, report.lineage, code_revision)
    if _report(spec, report.lineage, report.levels) != report:
        raise ValueError("saved hardening report does not reproduce its levels")
    return report


def run_task_hardening(
    project: ProjectStore, spec: TaskHardeningSpec, *, code_revision: str
) -> TaskHardeningReport:
    """Run levels in order; saved levels and children replay, unknown outcomes never redraw."""
    spec = TaskHardeningSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_task_hardening(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        seed, _ = _seed(project, spec.seed)
        lineage = _lineage(project, seed)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=(spec.seed,),
        )
        if not store.exists(identity):
            preflight_task_hardening(project, spec)
            store.write_json(
                artifact_id=identity,
                artifact_type="task-hardening-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        levels: list[HardeningLevel] = []
        for index in range(spec.maximum_levels + 1):
            key = f"{identity}-level-{index}"
            if store.exists(key):
                level = HardeningLevel.model_validate_json(store.read_bytes(key, "level.json"))
            else:
                previous = None
                if levels and levels[-1].proposals is not None:
                    previous = (_proposals(project, levels[-1].proposals), levels[-1])
                level = _level(project, spec, index, previous, code_revision)
                store.write_json(
                    artifact_id=key,
                    artifact_type="task-hardening-level",
                    envelope=envelope.model_copy(
                        update={
                            "inputs": tuple(
                                sorted(
                                    {registration, *_level_inputs(level)},
                                    key=lambda ref: ref.artifact_id,
                                )
                            )
                        }
                    ),
                    files={"level.json": level},
                )
            levels.append(level)
            if (
                level.proposals is None
                or index == spec.maximum_levels
                or any(row.disposition == "kept" for row in level.candidates)
                or not any(row.disposition == "saturated" for row in level.candidates)
            ):
                break
        result = _report(spec, lineage, tuple(levels))
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="task-hardening-report",
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted(
                            {
                                registration,
                                *(
                                    artifact_input(store.read(f"{identity}-level-{i}").manifest)
                                    for i in range(len(levels))
                                ),
                            },
                            key=lambda ref: ref.artifact_id,
                        )
                    )
                }
            ),
            files={"report.json": result},
        )
        return result
