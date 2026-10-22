"""Fixed-output judge challenges: invariance to declared-irrelevant changes, response to defects.

A challenge study registers saved, independently labeled outputs from a calibration pack, the
presentation factors its task declares irrelevant, and exact JSON edits of those outputs. Each
edit either varies one declared-irrelevant factor (actor identity, candidate order, explanation
length, a valid alternative action order) or injects one real defect of a kind the study
declares: a changed value at a declared JSON pointer, the same change defended by persuasive
prose, an added entry such as an unauthorized write, or a value of the wrong JSON type, each
under the study's own name (``wrong_cell``, ``wrong_total`` or any other). Every judge arm
then assesses the same blinded outputs through the existing evaluator callback, and the study
reports how often a judgment stays stable where nothing relevant changed and how often it
detects the injected defect, for single judges, repeated same-model judges and panels. A
judgment the evaluator could not obtain (a rejected or undispatched judge output) is counted as
a judge failure, never as an abstention, a flip or a detection.

The study performs no inference itself and never rewrites a label. Saved judgments replay the
report exactly, so judge configurations can be compared on identical outputs without new calls.
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import AxisOutcome, JudgeConfig, WorkflowEvidence
from exp.common.evaluations.uncertainty import missing_outcome_bounds, proportion_summary
from exp.common.project import ProjectStore, artifact_input
from exp.simulation.evaluation.calibration import (
    CalibrationCase,
    CalibrationJudgment,
    CalibrationPack,
    blinded_case,
    required_judge_failed,
    task_decision,
)
from exp.simulation.evaluation.tasks import Verdict, grade
from exp.simulation.reconstruction.models import path_parts, resolve

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
CHALLENGE_PROTOCOL = "judge-challenge-v8"

IrrelevantFactor = Literal[
    "actor_identity", "candidate_order", "explanation_length", "action_order"
]
DefectCheck = Literal["changed_value", "defended_change", "added_entry", "changed_json_type"]
INVARIANCE_FACTORS: tuple[IrrelevantFactor, ...] = (
    "actor_identity",
    "candidate_order",
    "explanation_length",
    "action_order",
)


class DeclaredInvariance(ContractModel):
    """Output locations the task declares irrelevant to its acceptance for one factor.

    Attributes:
        factor: Presentation factor that must not change a judgment.
        pointers: JSON pointers (RFC 6901) into the saved outcome whose values, or whose
            descendants, may vary for this factor.
        rationale: Why the public task contract makes this factor irrelevant.
    """

    factor: IrrelevantFactor
    pointers: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def rooted(self) -> Self:
        """Require nonroot pointers so a declaration never covers the whole outcome."""
        if any(not pointer.startswith("/") or pointer == "/" for pointer in self.pointers):
            raise ValueError("declared irrelevant locations must be nonroot JSON pointers")
        return self


class DeclaredDefect(ContractModel):
    """One real defect kind the study injects, its check and where it lands in the output.

    Attributes:
        kind: Study-chosen name the report measures sensitivity under (for example
            ``wrong_cell`` for a spreadsheet, ``wrong_total`` for an invoice).
        check: What the defect's edits must do. ``changed_value``: every edit replaces a value
            at a declared location with a different one. ``defended_change``: two or more
            replacements with different values, at least one at a declared location and at
            least one of them text defending it. ``added_entry``: at least one edit appends an
            element or adds a member at a declared location, such as an unauthorized write or
            effect. ``changed_json_type``: at least one replacement at a declared location
            changes the value's JSON type.
        pointers: JSON pointers (RFC 6901) into the saved outcome where the defect lands;
            descendants are included.
        rationale: Why the change is a real defect under the public task contract.
    """

    kind: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    check: DefectCheck
    pointers: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def rooted(self) -> Self:
        """Require nonroot pointers and a name apart from every irrelevant factor."""
        if any(not pointer.startswith("/") or pointer == "/" for pointer in self.pointers):
            raise ValueError("declared defect locations must be nonroot JSON pointers")
        if self.kind in INVARIANCE_FACTORS:
            raise ValueError(f"defect kind {self.kind} names an irrelevant factor")
        return self

    def covers(self, pointer: str) -> bool:
        """Whether an edit at ``pointer`` lands inside the declared locations."""
        return any(pointer == root or pointer.startswith(root + "/") for root in self.pointers)


class OutputEdit(ContractModel):
    """One exact edit of a saved outcome.

    Attributes:
        pointer: JSON pointer to an existing value (``replace``), an existing array
            (``append``), or an absent member of an existing object (``add``).
        value: Replacement value, the element appended, or the member added.
        operation: ``replace`` swaps the value; ``append`` adds an element to an array;
            ``add`` adds a member to an object, such as an unrequested effect in a state.
    """

    pointer: str = Field(pattern=r"^/.+")
    value: JsonValue
    operation: Literal["replace", "append", "add"] = "replace"


class Perturbation(ContractModel):
    """One registered change to one saved output.

    Attributes:
        perturbation_id: Unique identity within the study.
        base_case_id: Calibration case whose saved outcome is edited.
        factor: Declared-irrelevant factor varied, or the declared defect kind injected.
        edits: Exact edits; invariance edits stay inside the factor's declared locations and
            defect edits satisfy their kind's declared check.
        description: What changed, for reviewers reading the report.
    """

    perturbation_id: str = Field(min_length=1, max_length=128)
    base_case_id: str = Field(min_length=1)
    factor: str = Field(min_length=1, max_length=64)
    edits: tuple[OutputEdit, ...] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=2000)

    @property
    def kind(self) -> Literal["invariance", "defect"]:
        """Whether the change should leave the judgment unchanged or be detected."""
        return "invariance" if self.factor in INVARIANCE_FACTORS else "defect"


class ChallengeJudge(ContractModel):
    """One judge configuration assessed on every study output.

    Attributes:
        arm_id: Unique arm identity.
        config: Exact judge rubric, prompt and model identity.
        repeats: Independent repeated judgments per output; more than one measures
            same-model consistency, not independence.
    """

    arm_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    config: JudgeConfig
    repeats: int = Field(default=1, ge=1, le=16, strict=True)


class ChallengePanel(ContractModel):
    """A decision rule over several single-judge arms on the same outputs.

    Attributes:
        panel_id: Unique panel identity, distinct from every arm.
        members: Arms whose first repeat is aggregated.
        rule: ``majority`` takes a strict majority of decided members; ``any_defect`` fails
            when any member fails. Neither rule makes correlated members independent.
    """

    panel_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    members: tuple[str, ...] = Field(min_length=2)
    rule: Literal["majority", "any_defect"] = "majority"


class JudgeChallengeStudy(ContractModel):
    """A registered fixed-output study over saved, independently labeled outcomes.

    Attributes:
        pack: Frozen task, verifier, labels and criterion thresholds; its outcomes are the
            saved outputs every arm assesses.
        invariances: Factors the task declares irrelevant, with their locations.
        defects: Defect kinds the study injects, each with its check and locations.
        perturbations: Registered invariance and defect edits.
        judges: Single-judge arms, each run on every output.
        panels: Decision rules over judge arms on the same outputs.
        score_tolerance: Largest criterion score change still counted as stable.
    """

    pack: CalibrationPack
    invariances: tuple[DeclaredInvariance, ...] = ()
    defects: tuple[DeclaredDefect, ...] = ()
    perturbations: tuple[Perturbation, ...] = Field(min_length=1)
    judges: tuple[ChallengeJudge, ...] = Field(min_length=1)
    panels: tuple[ChallengePanel, ...] = ()
    score_tolerance: int = Field(default=0, ge=0, le=4, strict=True)

    @model_validator(mode="after")
    def registered(self) -> Self:
        """Admit only declared-irrelevant variation and real defects on valid outputs."""
        if not self.pack.minimum_scores:
            raise ValueError("challenge studies need registered criterion thresholds")
        cases = {case.challenge.id: case for case in self.pack.cases}
        if len({row.factor for row in self.invariances}) != len(self.invariances):
            raise ValueError("declare each irrelevant factor once")
        if len({row.kind for row in self.defects}) != len(self.defects):
            raise ValueError("declare each defect kind once")
        declared: dict[str, tuple[str, ...]] = {
            row.factor: row.pointers for row in self.invariances
        }
        defects = {row.kind: row for row in self.defects}
        if len({row.perturbation_id for row in self.perturbations}) != len(self.perturbations):
            raise ValueError("perturbation IDs must be unique")
        for row in self.perturbations:
            case = cases.get(row.base_case_id)
            if case is None:
                raise ValueError(f"perturbation {row.perturbation_id} names an unknown case")
            if row.kind == "defect" and row.factor not in defects:
                raise ValueError(
                    f"perturbation {row.perturbation_id} injects {row.factor}, which is neither "
                    "an irrelevant factor nor a defect kind the study declares"
                )
            _variant(case, row, declared, defects)
        arms = [row.arm_id for row in self.judges]
        panels = [row.panel_id for row in self.panels]
        if len(set(arms + panels)) != len(arms) + len(panels):
            raise ValueError("judge arm and panel identities must be unique")
        for panel in self.panels:
            if len(set(panel.members)) != len(panel.members) or set(panel.members) - set(arms):
                raise ValueError("panel members must be distinct registered judge arms")
        return self

    @property
    def identity(self) -> str:
        """Registered study identity, fixed before any judgment is requested."""
        return stable_id(
            "judge-challenge-study",
            {"study": sha256_json(self), "protocol": CHALLENGE_PROTOCOL},
        )


class ChallengeJudgment(ContractModel):
    """One saved judgment of one study output by one arm and repeat.

    Attributes:
        arm_id: Judge arm.
        repeat: Zero-based repeat index.
        item_id: ``base:<case>`` for a saved output, or the perturbation ID for its variant.
        judgment: Saved evaluator result bound to the exact output and configuration.
    """

    arm_id: str
    repeat: int = Field(ge=0)
    item_id: str
    judgment: CalibrationJudgment


class ChallengeOutput(ContractModel):
    """One study output with the label and lineage of the saved output it starts from.

    Attributes:
        item_id: Study output identity.
        case: Calibration case carrying the base label and lineage.
        perturbation: Registered change, absent for a saved base output.
        outcome: Exact outcome shown to every judge.
    """

    item_id: str
    case: CalibrationCase
    perturbation: Perturbation | None
    outcome: WorkflowEvidence | JsonObject


def study_items(study: JudgeChallengeStudy) -> tuple[ChallengeOutput, ...]:
    """Every output each arm judges: perturbed bases first, then their registered variants.

    Args:
        study: Registered challenge study.

    Returns:
        Saved base outputs (only those a perturbation edits) and every variant, in order.
    """
    cases = {case.challenge.id: case for case in study.pack.cases}
    declared: dict[str, tuple[str, ...]] = {row.factor: row.pointers for row in study.invariances}
    defects = {row.kind: row for row in study.defects}
    bases = dict.fromkeys(row.base_case_id for row in study.perturbations)
    return tuple(
        ChallengeOutput(
            item_id="base:" + case_id,
            case=cases[case_id],
            perturbation=None,
            outcome=cases[case_id].challenge.outcome,
        )
        for case_id in bases
    ) + tuple(
        ChallengeOutput(
            item_id=row.perturbation_id,
            case=cases[row.base_case_id],
            perturbation=row,
            outcome=_variant(cases[row.base_case_id], row, declared, defects),
        )
        for row in study.perturbations
    )


def run_challenge_study(
    study: JudgeChallengeStudy,
    judge: Callable[[JudgeConfig, JsonObject, str], CalibrationJudgment],
) -> tuple[ChallengeJudgment, ...]:
    """Request every arm's judgments of the same blinded outputs through the evaluator owner.

    Args:
        study: Registered study.
        judge: Existing evaluator callback, as for rubric development: it receives the exact
            configuration, the blinded output and a stable request key, and retains its own
            receipts. A replaying owner returns saved results for repeated keys.

    Returns:
        One saved judgment per arm, repeat and output, bound to the exact output.

    Raises:
        ValueError: The callback returned a result for another output or configuration.
    """
    saved = []
    items = study_items(study)
    for arm in study.judges:
        for repeat in range(arm.repeats):
            for item in items:
                case = item.case.model_copy(
                    update={
                        "challenge": item.case.challenge.model_copy(
                            update={"outcome": item.outcome}
                        )
                    }
                )
                key = f"{study.identity}:{arm.arm_id}:{repeat}:{item.item_id}"
                result = judge(arm.config, blinded_case(case), key)
                if result.outcome_sha256 != sha256_json(item.outcome) or result.judge_sha256 != (
                    sha256_json(arm.config)
                ):
                    raise ValueError("judge callback returned a different output or configuration")
                saved.append(
                    ChallengeJudgment(
                        arm_id=arm.arm_id,
                        repeat=repeat,
                        item_id=item.item_id,
                        judgment=result.model_copy(update={"case_id": item.item_id}),
                    )
                )
    return tuple(saved)


def assess_challenge_study(
    study: JudgeChallengeStudy, judgments: Sequence[ChallengeJudgment]
) -> JsonObject:
    """Measure stability and defect sensitivity from saved judgments, with no new calls.

    Args:
        study: Registered study.
        judgments: Saved judgments; missing ones stay in every denominator as not run.

    Returns:
        Per-arm and per-panel base-label errors, invariance stability and defect sensitivity
        by factor, with Wilson intervals at the count of independent base lineages, judge
        failures kept apart from abstentions, repeat consistency and defects every panel
        member missed. Each rate's interval is conditional on the judgments obtained; its
        ``planned_bounds`` cover every planned trial, a missing or failed judgment counted
        once as a miss and once as a success.

    Raises:
        ValueError: A judgment repeats, names an unknown arm or output, or is bound to
            another output or configuration.
    """
    items = study_items(study)
    by_id = {item.item_id: item for item in items}
    arms = {arm.arm_id: arm for arm in study.judges}
    decisions: dict[tuple[str, int, str], _Decision] = {}
    for row in judgments:
        item, arm = by_id.get(row.item_id), arms.get(row.arm_id)
        if item is None or arm is None or row.repeat >= arm.repeats:
            raise ValueError("challenge judgment names an unknown arm, repeat or output")
        if row.judgment.outcome_sha256 != sha256_json(
            item.outcome
        ) or row.judgment.judge_sha256 != sha256_json(arm.config):
            raise ValueError("challenge judgment is bound to another output or configuration")
        key = (row.arm_id, row.repeat, row.item_id)
        if key in decisions:
            raise ValueError("challenge judgments must not repeat")
        decisions[key] = _decide(study, item, row.judgment.axes, arm.config)
    report_arms: JsonObject = {}
    for arm in study.judges:
        first = {item.item_id: decisions.get((arm.arm_id, 0, item.item_id)) for item in items}
        repeated = [
            [decisions.get((arm.arm_id, repeat, item.item_id)) for repeat in range(arm.repeats)]
            for item in items
        ]
        # A failed repeat obtained no judgment, so it is neither agreement nor disagreement:
        # it is counted apart, and an item enters with at least two judged repeats.
        judged = [[row for row in rows if row is not None and not row.failed] for rows in repeated]
        consistent = [len({row.judge for row in rows}) == 1 for rows in judged if len(rows) >= 2]
        # Over planned repeats, observed disagreement is a known inconsistency and agreement
        # is known only when every repeat was judged; anything else stays unknown.
        agreeing = [rows for rows in judged if len({row.judge for row in rows}) <= 1]
        known_consistent = sum(len(rows) == arm.repeats for rows in agreeing)
        report_arms[arm.arm_id] = {
            "kind": "single",
            "repeats": arm.repeats,
            **_measure(study, items, first),
            "repeat_consistency": None
            if arm.repeats == 1
            else {
                **proportion_summary(sum(consistent), len(consistent)),
                "planned_bounds": missing_outcome_bounds(
                    known_consistent, len(agreeing) - known_consistent, len(items)
                ),
                "failed_repeats": sum(
                    row is not None and row.failed for rows in repeated for row in rows
                ),
                "missing_repeats": sum(row is None for rows in repeated for row in rows),
                "items_without_two_judged_repeats": len(items) - len(consistent),
            },
        }
    report_panels: JsonObject = {}
    for panel in study.panels:
        combined = {
            item.item_id: _panel_decision(
                panel, [decisions.get((member, 0, item.item_id)) for member in panel.members]
            )
            for item in items
        }
        measured = _measure(study, items, combined)
        report_panels[panel.panel_id] = {
            "kind": "panel",
            "rule": panel.rule,
            "members": list(panel.members),
            **measured,
            "missed_by_every_member": [
                item.item_id
                for item in items
                if item.perturbation is not None
                and item.perturbation.kind == "defect"
                and all(
                    (decision := decisions.get((member, 0, item.item_id))) is not None
                    and decision.judge == "pass"
                    for member in panel.members
                )
            ],
        }
    expected = sum(arm.repeats for arm in study.judges) * len(items)
    return {
        "study_id": study.identity,
        "protocol": CHALLENGE_PROTOCOL,
        "task": study.pack.task.model_dump(mode="json"),
        "pack_sha256": sha256_json(study.pack),
        "declared_invariances": [row.model_dump(mode="json") for row in study.invariances],
        "declared_defects": [row.model_dump(mode="json") for row in study.defects],
        "outputs": {
            "bases": sum(item.perturbation is None for item in items),
            "invariance": sum(
                item.perturbation is not None and item.perturbation.kind == "invariance"
                for item in items
            ),
            "defects": sum(
                item.perturbation is not None and item.perturbation.kind == "defect"
                for item in items
            ),
        },
        "planned_judgments": expected,
        "not_run": expected - len(decisions),
        "arms": report_arms,
        "panels": report_panels,
        "scope": (
            "fixed saved outputs and registered edits only; stability covers the declared "
            "factors, sensitivity covers the injected defect kinds; intervals use independent "
            "base lineages; agreement among correlated judges is not independent evidence"
        ),
    }


def record_challenge_study(
    project: ProjectStore,
    study: JudgeChallengeStudy,
    judgments: Sequence[ChallengeJudgment],
    *,
    code_revision: str,
    artifact_id: ArtifactId | None = None,
    registration: ArtifactInput | None = None,
) -> ArtifactInput:
    """Persist the registered study, its saved judgments and the replayable report.

    Args:
        project: Project that owns the immutable result.
        study: Registered study.
        judgments: Saved judgments of every arm.
        code_revision: Producing implementation identity.
        artifact_id: Run-owned result namespace, or the study identity for callback-only studies.
        registration: Immutable run configuration included in the result provenance, if supplied.

    Returns:
        The immutable result reference; an identical earlier result is returned unchanged.

    Raises:
        ValueError: A saved result for this study holds different judgments or a report its
            judgments no longer reproduce.
    """
    report = assess_challenge_study(study, judgments)
    saved = tuple(judgments)
    identity = artifact_id or study.identity
    if project.artifacts.exists(identity):
        if saved_challenge_judgments(project, study, artifact_id=identity) != saved:
            raise ValueError("challenge study already holds different saved judgments")
        return artifact_input(project.artifacts.read(identity).manifest)
    inputs = {study.pack.task, *(row.judgment.report for row in saved)}
    if registration is not None:
        inputs.add(registration)
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="judge-challenge-study",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda row: row.artifact_id)),
        ),
        files={
            "study.json": study.model_dump(mode="json"),
            "judgments.json": [row.model_dump(mode="json") for row in saved],
            "report.json": report,
        },
    )
    return artifact_input(manifest)


def saved_challenge_judgments(
    project: ProjectStore,
    study: JudgeChallengeStudy,
    *,
    artifact_id: ArtifactId | None = None,
) -> tuple[ChallengeJudgment, ...] | None:
    """Read a saved study's judgments after proving its registration and report reproduce.

    Args:
        project: Project that may own the result.
        study: Registered study.
        artifact_id: Run-owned result namespace, or the study identity for callback-only studies.

    Returns:
        The saved judgments, or None when the study has no saved result.

    Raises:
        ValueError: The saved registration differs or its judgments no longer reproduce the
            saved report.
    """
    store, identity = project.artifacts, artifact_id or study.identity
    if not store.exists(identity):
        return None
    if JudgeChallengeStudy.model_validate_json(store.read_bytes(identity, "study.json")) != study:
        raise ValueError("saved challenge study differs from its registration")
    rows = tuple(
        ChallengeJudgment.model_validate(row)
        for row in _json_list(store.read_bytes(identity, "judgments.json"))
    )
    if assess_challenge_study(study, rows) != json.loads(store.read_bytes(identity, "report.json")):
        raise ValueError("saved challenge judgments no longer reproduce their report")
    return rows


class _Decision(ContractModel):
    """Decisions derived from one judgment of one output.

    Attributes:
        judge: Judge-only decision: required criteria against thresholds, critical flags fail.
        task: Mechanically gated task decision, as calibration grades it.
        verifier: Mechanical verifier verdict on the same output.
        scores: Assessed criterion scores by criterion.
        failed: The evaluator obtained no judgment of a required criterion (rejected, unknown
            or undispatched output), as distinct from a judge that abstained on purpose. An
            optional criterion's failure cannot block the decision and is not counted.
    """

    judge: Verdict
    task: Verdict
    verifier: Verdict
    scores: dict[str, int]
    failed: bool = False


def _decide(
    study: JudgeChallengeStudy,
    item: ChallengeOutput,
    axes: tuple[AxisOutcome, ...],
    config: JudgeConfig,
) -> _Decision:
    """Derive judge-only, task and verifier decisions for one saved judgment."""
    registered = {dimension.dimension_id for dimension in config.definition.dimensions}
    if {axis.dimension_id for axis in axes} != registered or len(axes) != len(registered):
        raise ValueError("challenge judgment criteria differ from the arm's registered rubric")
    verifier = grade(study.pack.candidate, item.outcome).status
    return _Decision(
        judge=task_decision("pass", axes, study.pack.minimum_scores),
        task=task_decision(verifier, axes, study.pack.minimum_scores),
        verifier=verifier,
        scores={
            axis.dimension_id: axis.score
            for axis in axes
            if axis.status == "assessed" and axis.score is not None
        },
        failed=required_judge_failed(axes, study.pack.minimum_scores),
    )


def _panel_decision(panel: ChallengePanel, members: list[_Decision | None]) -> _Decision | None:
    """Aggregate member decisions; any missing member leaves the panel decision missing.

    A panel fails operationally only when its rule cannot decide and a member failed.
    """
    if any(member is None for member in members):
        return None
    known = [member for member in members if member is not None]

    def combine(values: list[Verdict]) -> Verdict:
        """Apply the panel rule to one decision channel."""
        failed, passed = values.count("fail"), values.count("pass")
        if panel.rule == "any_defect":
            return "fail" if failed else "pass" if passed == len(values) else "unassessable"
        if failed * 2 > len(values):
            return "fail"
        return "pass" if passed * 2 > len(values) else "unassessable"

    judge = combine([member.judge for member in known])
    return _Decision(
        judge=judge,
        task=combine([member.task for member in known]),
        verifier=known[0].verifier,
        scores={},
        failed=judge == "unassessable" and any(member.failed for member in known),
    )


def _measure(
    study: JudgeChallengeStudy,
    items: tuple[ChallengeOutput, ...],
    decisions: dict[str, _Decision | None],
) -> JsonObject:
    """Compute base agreement and errors, stability per factor and sensitivity per kind."""
    bases = [item for item in items if item.perturbation is None]
    judged = [(item, decisions[item.item_id]) for item in bases]
    base_known = [
        (item, decision)
        for item in bases
        if (decision := decisions[item.item_id]) is not None and decision.judge != "unassessable"
    ]
    invariance: JsonObject = {}
    for factor in INVARIANCE_FACTORS:
        rows = _variants(items, factor)
        if rows:
            invariance[factor] = _stability(study, rows, decisions)
    sensitivity: JsonObject = {}
    for kind in (row.kind for row in study.defects):
        rows = _variants(items, kind)
        if rows:
            sensitivity[kind] = _sensitivity(rows, decisions)
    return {
        "base_label_agreement": proportion_summary(
            sum(decision.judge == item.case.label for item, decision in base_known),
            len(base_known),
            units=len({item.case.lineage for item, _ in base_known}) or None,
        ),
        "base_false_acceptances": sum(
            decision.judge == "pass" and item.case.label == "fail" for item, decision in base_known
        ),
        "base_false_rejections": sum(
            decision.judge == "fail" and item.case.label == "pass" for item, decision in base_known
        ),
        "base_judge_failures": sum(
            decision is not None and decision.failed for _, decision in judged
        ),
        "invariance": invariance,
        "sensitivity": sensitivity,
    }


def _variants(items: tuple[ChallengeOutput, ...], factor: str) -> list[ChallengeOutput]:
    """Registered variants of one factor or defect kind."""
    return [
        item
        for item in items
        if item.perturbation is not None and item.perturbation.factor == factor
    ]


def _stability(
    study: JudgeChallengeStudy,
    rows: list[ChallengeOutput],
    decisions: dict[str, _Decision | None],
) -> JsonObject:
    """Count unchanged decisions and bounded score shifts against each variant's base.

    A pair where either judgment failed is not compared: a lost judgment is not a flip.
    """
    stable = flips = compared = failures = missing = 0
    shifts: list[int] = []
    lineages: set[str] = set()
    for item in rows:
        variant = decisions[item.item_id]
        base = decisions["base:" + item.case.challenge.id]
        if variant is None or base is None:
            missing += 1
            continue
        if variant.failed or base.failed:
            failures += 1
            continue
        compared += 1
        lineages.add(item.case.lineage)
        shift = max(
            (
                abs(variant.scores[key] - base.scores[key])
                for key in base.scores.keys() & variant.scores.keys()
            ),
            default=0,
        )
        shifts.append(shift)
        same = variant.judge == base.judge and base.scores.keys() == variant.scores.keys()
        flips += variant.judge != base.judge
        stable += same and shift <= study.score_tolerance
    return {
        "variants": len(rows),
        "compared": compared,
        "judge_failures": failures,
        "decision_flips": flips,
        "largest_score_shift": max(shifts, default=None),
        "stability": proportion_summary(stable, compared, units=len(lineages) or None),
        "stability_planned_bounds": missing_outcome_bounds(stable, failures + missing, len(rows)),
    }


def _sensitivity(rows: list[ChallengeOutput], decisions: dict[str, _Decision | None]) -> JsonObject:
    """Count injected defects the judge, the task decision and the verifier each detect."""
    known = [(item, decision) for item in rows if (decision := decisions[item.item_id]) is not None]
    decided = [(item, decision) for item, decision in known if decision.judge != "unassessable"]
    failed = sum(decision.failed for _, decision in known)
    detected = sum(decision.judge == "fail" for _, decision in decided)
    return {
        "variants": len(rows),
        "judged": len(known),
        "detected_by_judge": sum(decision.judge == "fail" for _, decision in decided),
        "accepted_by_judge": sum(decision.judge == "pass" for _, decision in decided),
        "judge_abstained": len(known) - len(decided) - failed,
        "judge_failed": failed,
        "detected_by_task_decision": sum(decision.task == "fail" for _, decision in known),
        "detected_by_verifier": sum(decision.verifier == "fail" for _, decision in known),
        "judge_sensitivity": proportion_summary(
            detected,
            len(decided),
            units=len({item.case.lineage for item, _ in decided}) or None,
        ),
        # An abstention is a known miss; a failed or missing judgment is an unknown outcome.
        "judge_sensitivity_planned_bounds": missing_outcome_bounds(
            detected, len(rows) - len(known) + failed, len(rows)
        ),
    }


def _variant(
    case: CalibrationCase,
    perturbation: Perturbation,
    declared: dict[str, tuple[str, ...]],
    defects: dict[str, DeclaredDefect],
) -> WorkflowEvidence | JsonObject:
    """Apply one registered perturbation after checking it varies only what it claims to.

    Raises:
        ValueError: An invariance edit leaves its declared locations or is not a valid change
            for its factor, a defect is injected into an output not labeled valid, or a defect
            edit does not change what its kind requires.
    """
    original = case.challenge.outcome
    document: JsonValue = (
        original.model_dump(mode="json")
        if isinstance(original, WorkflowEvidence)
        else dict(original)
    )
    factor = perturbation.factor
    for edit in perturbation.edits:
        before = _before(document, edit)
        if perturbation.kind == "invariance":
            if not any(
                edit.pointer == root or edit.pointer.startswith(root + "/")
                for root in declared.get(factor, ())
            ):
                raise ValueError(
                    f"{perturbation.perturbation_id} edits {edit.pointer}, which the task does not "
                    f"declare irrelevant for {factor}"
                )
            _require_invariant(factor, edit, before, perturbation.perturbation_id)
        document = _apply(document, edit)
    if perturbation.kind == "defect":
        if case.label != "pass":
            raise ValueError("defects are injected into outputs labeled valid")
        _require_defect(perturbation, defects[factor], case)
    if not isinstance(document, dict):
        raise ValueError("perturbed outcomes must remain JSON objects")
    if sha256_json(document) == sha256_json(
        original.model_dump(mode="json") if isinstance(original, WorkflowEvidence) else original
    ):
        raise ValueError(f"{perturbation.perturbation_id} does not change the saved output")
    if isinstance(original, WorkflowEvidence):
        return WorkflowEvidence.model_validate(document)
    return document


def _require_invariant(factor: str, edit: OutputEdit, before: JsonValue, name: str) -> None:
    """Check an irrelevant-factor edit changes only presentation."""
    if edit.operation != "replace":
        raise ValueError(f"{name}: an irrelevant-factor change replaces, never adds, a value")
    if factor in ("candidate_order", "action_order"):
        if (
            not isinstance(before, list)
            or not isinstance(edit.value, list)
            or sorted(map(sha256_json, before)) != sorted(map(sha256_json, edit.value))
        ):
            raise ValueError(f"{name}: an order change must permute the same elements")
    if factor == "explanation_length" and not (
        isinstance(before, str) and isinstance(edit.value, str)
    ):
        raise ValueError(f"{name}: an explanation-length change replaces text with text")


def _require_defect(
    perturbation: Perturbation, defect: DeclaredDefect, case: CalibrationCase
) -> None:
    """Check a defect's edits do what its declared kind requires, where it is declared."""
    original = case.challenge.outcome
    document: JsonValue = (
        original.model_dump(mode="json")
        if isinstance(original, WorkflowEvidence)
        else dict(original)
    )
    pairs = [(edit, _before(document, edit)) for edit in perturbation.edits]
    name, kind = perturbation.perturbation_id, defect.kind
    replaced = all(edit.operation == "replace" and edit.value != before for edit, before in pairs)
    if defect.check == "changed_value" and not (
        replaced and all(defect.covers(edit.pointer) for edit, _ in pairs)
    ):
        raise ValueError(
            f"{name}: a {kind} defect replaces values at its declared locations with different ones"
        )
    if defect.check == "defended_change" and (
        len(pairs) < 2
        or not replaced
        or not any(defect.covers(edit.pointer) for edit, _ in pairs)
        or not any(isinstance(edit.value, str) for edit, _ in pairs)
    ):
        raise ValueError(
            f"{name}: a {kind} defect changes a value at its declared locations and the prose "
            "defending it"
        )
    if defect.check == "added_entry" and not any(
        edit.operation in ("append", "add") and defect.covers(edit.pointer) for edit, _ in pairs
    ):
        raise ValueError(
            f"{name}: a {kind} defect appends or adds an entry at its declared locations"
        )
    if defect.check == "changed_json_type" and not any(
        edit.operation == "replace"
        and defect.covers(edit.pointer)
        and _json_type(before) != _json_type(edit.value)
        for edit, before in pairs
    ):
        raise ValueError(
            f"{name}: a {kind} defect changes a value's JSON type at its declared locations"
        )


def _json_type(value: JsonValue) -> str:
    """JSON type name, with integers and floats both numbers."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _saved(document: JsonValue, pointer: str) -> JsonValue:
    """Return the value at an existing pointer through the shared JSON pointer owner.

    Raises:
        ValueError: The pointer does not name an existing value.
    """
    try:
        return resolve(document, pointer)
    except ValueError as error:
        raise ValueError(f"perturbation pointer {pointer} does not name a saved value") from error


def _before(document: JsonValue, edit: OutputEdit) -> JsonValue:
    """The value an edit changes; an added member has none and its object must exist.

    Raises:
        ValueError: The pointer names no saved value, or an added member already exists.
    """
    if edit.operation != "add":
        return _saved(document, edit.pointer)
    # Escaped tokens contain no "/", so the parent pointer is everything before the last one.
    parent = _saved(document, edit.pointer.rpartition("/")[0])
    if not isinstance(parent, dict) or path_parts(edit.pointer)[-1] in parent:
        raise ValueError(f"add target {edit.pointer} must be a new member of an existing object")
    return None


def _apply(document: JsonValue, edit: OutputEdit) -> JsonValue:
    """Return a copy of ``document`` with one edit applied."""
    tokens = path_parts(edit.pointer)

    def rebuild(node: JsonValue, depth: int) -> JsonValue:
        """Copy the path to the edited value."""
        if edit.operation == "add" and depth == len(tokens) - 1 and isinstance(node, dict):
            return {**node, tokens[-1]: edit.value}
        if depth == len(tokens):
            if edit.operation == "replace":
                return edit.value
            if not isinstance(node, list):
                raise ValueError(f"append target {edit.pointer} is not an array")
            return [*node, edit.value]
        token = tokens[depth]
        if isinstance(node, dict):
            return {**node, token: rebuild(node[token], depth + 1)}
        if isinstance(node, list):
            index = int(token)
            return [*node[:index], rebuild(node[index], depth + 1), *node[index + 1 :]]
        raise ValueError(f"perturbation pointer {edit.pointer} does not name a saved value")

    _before(document, edit)
    return rebuild(document, 0)


def _json_list(payload: bytes) -> list[JsonValue]:
    """Parse a saved JSON array."""
    value = json.loads(payload)
    if not isinstance(value, list):
        raise ValueError("saved challenge records must be a JSON array")
    return value
