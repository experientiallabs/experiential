"""The offline Astra rewrite loop of one target prompt, walked through its declared state machine.

This is an explicit loop, not an automated prompt-optimization library: each round Astra proposes
one rewrite from cited failures, and code keeps it only on a held-out gain.

1. ``opened``: the calibration set is split by lineage (:func:`~.astra_prompt_calibration.split`)
   and the whole loop's bound is reserved on the cumulative ledger; a refusal ends in
   ``budget_refused`` before any call, with nothing saved.
2. ``referenced``: Astra answers every calibration input with the role's own task contract (the
   exact system prompt the role saw, with the code text).
3. ``evaluated``: the runtime model runs the code prompt on every input. Astra judges every
   output blind against its reference; on held-out inputs the second open-weight reviewer judges
   too. Parity is the share judged equivalent or better.
4. ``rewritten``: Astra rewrites the prompt from the cited rewrite-set failures only. A rewrite
   that fails the host's checks is rejected (``rewrite_unusable``); one containing held-out input
   or reference text by the mechanical check is rejected (``leak_refused``) before evaluation.
5. ``candidate_evaluated``: the rewrite is evaluated like the code prompt; a judge target's
   rewrite also runs its labeled acceptance cases.
6. ``accepted_round`` when held-out parity rose above the best so far and, for a judge target, the
   labeled cases still hold; otherwise ``rejected_round``. A rewrite-set gain never counts.
7. ``target_met`` when the best held-out parity reaches the target; ``plateaued`` after
   ``plateau_rounds`` rounds without an accepted gain, after ``maximum_rounds``, or when no
   rewrite-set failure is left to cite.

The record (best prompt, every round, final parity, Astra/reviewer agreement and cited held-out
examples) is saved; a rerun replays it with no call, and every draw also replays exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json, stable_id
from exp.common.project import ProjectStore
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.variance.astra_prompt_calibration import split as lineage_split
from exp.optimize.evaluation.variance.astra_prompt_contracts import (
    ASTRA_PROMPT_PROTOCOL,
    RECORD_TYPE,
    CalibrationSet,
    Evaluation,
    GateResult,
    Judgment,
    LabeledCase,
    RefinementRecord,
    RefinementSettings,
    RoundRecord,
)
from exp.optimize.evaluation.variance.astra_prompt_costs import reserve_target, target_bound
from exp.optimize.evaluation.variance.astra_prompt_draws import (
    forced_draw,
    judge_parity,
    rewrite_prompt,
    run_gate,
    swap,
)
from exp.optimize.evaluation.variance.astra_prompt_leakage import held_out_texts, leaks
from exp.optimize.evaluation.variance.astra_prompt_states import (
    REFINEMENT_MACHINE,
    RefinementState,
)
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger, measure
from exp.optimize.evaluation.variance.prompt_targets import PromptTarget
from exp.optimize.evaluation.variance.refined_prompts import journal_of
from exp.runtime.models.providers.codex_cli import offline_closed_model

S = RefinementState

JudgeName = Literal["astra", "reviewer"]


class _Walk:
    """One refinement's walk, every move checked and journaled."""

    def __init__(self, subject: str, journal: TransitionJournal) -> None:
        """Start in ``opened``."""
        self.subject, self.journal = subject, journal
        self.state = REFINEMENT_MACHINE.move(subject, None, S.OPENED.value, journal)

    def go(self, event: str, target: RefinementState) -> None:
        """Apply one move; an undeclared move is journaled as refused and raised."""
        self.state = REFINEMENT_MACHINE.move(
            self.subject, self.state, event, self.journal, target=target
        )


def record_identity(
    target: PromptTarget,
    calibration: CalibrationSet,
    settings: RefinementSettings,
    cases: Sequence[LabeledCase],
) -> str:
    """The identity a refinement record is saved under, computed before any call."""
    return stable_id(
        RECORD_TYPE,
        {
            "protocol": ASTRA_PROMPT_PROTOCOL,
            "target": target.target_id,
            "base": target.sha256,
            "settings": sha256_json(settings),
            "calibration": sha256_json(calibration),
            "cases": sha256_json([case.model_dump(mode="json") for case in cases]),
        },
    )


def load_record(project: ProjectStore, identity: str) -> RefinementRecord | None:
    """The saved record, or None."""
    if not project.artifacts.exists(identity):
        return None
    stored = project.artifacts.read(identity)
    return RefinementRecord.model_validate_json(stored.payloads["record.json"])


class _Loop:
    """The draws and evaluations of one target's refinement."""

    def __init__(
        self,
        project: ProjectStore,
        target: PromptTarget,
        calibration: CalibrationSet,
        settings: RefinementSettings,
        cases: Sequence[LabeledCase],
        code_revision: str,
    ) -> None:
        """Bind the loop's inputs."""
        self.project, self.target, self.settings = project, target, settings
        self.calibration, self.cases, self.revision = calibration, tuple(cases), code_revision
        self.examples = {row.example_id: row for row in calibration.examples}
        self.references: dict[str, JsonObject] = {}
        self.reference_draws: dict[str, str] = {}

    def reference(self) -> None:
        """Astra answers every input under the role's own (code) contract."""
        for example in self.calibration.examples:
            draw = forced_draw(
                self.project, self.settings.astra, kind="reference", system=example.system,
                user=example.user, answer=self.calibration.answer, sources=(),
                code_revision=self.revision,
            )  # fmt: skip
            if draw.reference is not None:
                self.reference_draws[example.example_id] = draw.reference
            if draw.answer is not None:
                self.references[example.example_id] = draw.answer

    def _judge(
        self, judge: JudgeName, example_id: str, output: JsonObject, prompt_sha: str
    ) -> Judgment:
        """One blind judgment of the candidate output against the reference."""
        example = self.examples[example_id]
        return judge_parity(
            self.project,
            self.settings.astra if judge == "astra" else self.settings.reviewer,
            judge=judge,
            seed=self.settings.seed,
            example_id=example_id,
            contract=example.system,
            user=example.user,
            answer=self.calibration.answer,
            candidate=output,
            reference=self.references[example_id],
            prompt_sha256=prompt_sha,
            code_revision=self.revision,
        )

    def evaluate(self, prompt: str, rewrite: Sequence[str], heldout: Sequence[str]) -> Evaluation:
        """Run the runtime model with ``prompt`` on both sides and judge every output blind."""
        prompt_sha = sha256_json(prompt)
        judgments: list[Judgment] = []
        for example_id in (*rewrite, *heldout):
            example = self.examples[example_id]
            draw = forced_draw(
                self.project, self.settings.runtime, kind="runtime",
                system=swap(example.system, self.target.text, prompt), user=example.user,
                answer=self.calibration.answer, sources=(), code_revision=self.revision,
            )  # fmt: skip
            judges: tuple[JudgeName, ...] = (
                ("astra",) if example_id in rewrite else ("astra", "reviewer")
            )
            for judge in judges:
                if draw.answer is None:
                    judgments.append(
                        Judgment(
                            example_id=example_id,
                            judge=judge,
                            candidate_side="A",
                            verdict="worse",
                            failure=draw.failure or "the runtime model gave no answer",
                        )
                    )
                else:
                    judgments.append(self._judge(judge, example_id, draw.answer, prompt_sha))
        gate = None
        if self.target.judge:
            gate = run_gate(
                self.project, self.settings.runtime, base=self.target.text, candidate=prompt,
                cases=self.cases, answer=self.calibration.answer, code_revision=self.revision,
            )  # fmt: skip
        return _score(prompt_sha, judgments, rewrite, heldout, self.settings.decision, gate)

    def failures(self, evaluation: Evaluation, rewrite: Sequence[str]) -> list[JsonObject]:
        """The cited rewrite-set failures shown to Astra (never a held-out input)."""
        rows: list[JsonObject] = []
        for judgment in evaluation.judgments:
            if judgment.judge != "astra" or judgment.example_id not in rewrite:
                continue
            if judgment.at_parity or len(rows) >= self.settings.maximum_failures:
                continue
            example = self.examples[judgment.example_id]
            rows.append(
                {
                    "failure_id": judgment.example_id,
                    "input": example.user,
                    "reference_output": self.references[judgment.example_id],
                    "differences": [row.model_dump(mode="json") for row in judgment.differences],
                    "candidate_violations": list(judgment.candidate_violations),
                    "judge_failure": judgment.failure,
                }
            )
        return rows


def _share(values: Sequence[bool]) -> float:
    """Share of true values; 0 for none."""
    return sum(values) / len(values) if values else 0.0


def _score(
    prompt_sha: str,
    judgments: Sequence[Judgment],
    rewrite: Sequence[str],
    heldout: Sequence[str],
    decision: str,
    gate: GateResult | None,
) -> Evaluation:
    """Aggregate judgments into parity shares and agreement."""
    astra = {j.example_id: j.at_parity for j in judgments if j.judge == "astra"}
    reviewer = {j.example_id: j.at_parity for j in judgments if j.judge == "reviewer"}
    held_astra = [astra[e] for e in heldout]
    held_reviewer = [reviewer[e] for e in heldout]
    combined = [a and r for a, r in zip(held_astra, held_reviewer, strict=True)]
    return Evaluation.model_validate(
        {
            "prompt_sha256": prompt_sha,
            "rewrite_parity": _share([astra[e] for e in rewrite]),
            "heldout_parity": _share(combined if decision == "both" else held_astra),
            "heldout_astra": _share(held_astra),
            "heldout_reviewer": _share(held_reviewer),
            "agreement": _share([a == r for a, r in zip(held_astra, held_reviewer, strict=True)]),
            "judgments": [j.model_dump(mode="json") for j in judgments],
            "gate": None if gate is None else gate.model_dump(mode="json"),
        }
    )


def _validate(
    target: PromptTarget,
    calibration: CalibrationSet,
    cases: Sequence[LabeledCase],
) -> None:
    """Refuse inputs that do not belong to this target."""
    if calibration.target_id != target.target_id or calibration.base_sha256 != target.sha256:
        raise ValueError("the calibration set was captured for another target or code text")
    for row in calibration.examples:
        if row.system.count(target.text) != 1:
            raise ValueError(f"example {row.example_id} does not hold the target text once")
    if target.judge and not cases:
        raise ValueError(f"{target.target_id} grades episodes; give its labeled acceptance cases")
    for case in cases:
        if case.system.count(target.text) != 1:
            raise ValueError(f"labeled case {case.case_id} does not hold the target text once")


def refine_prompt(
    project: ProjectStore,
    target: PromptTarget,
    calibration: CalibrationSet,
    settings: RefinementSettings,
    *,
    cases: Sequence[LabeledCase] = (),
    code_revision: str,
    ledger: PhaseLedger | None = None,
) -> RefinementRecord:
    """Run, or replay, one target's refinement and save its candidate record.

    Args:
        project: The offline calibration project.
        target: The target prompt.
        calibration: Its calibration inputs.
        settings: Runtime, Astra and reviewer models and the loop bounds.
        cases: A judge target's labeled acceptance cases.
        code_revision: Producing source identity.
        ledger: The cumulative ledger the whole bound is reserved on.

    Returns:
        The record; a ``budget_refused`` record is returned unsaved.

    Raises:
        ValueError: The inputs belong to another target, a judge target has no labeled cases, or
            no held-out or rewrite input has a usable reference.
        TransientDraw: A draw failed transiently or was refused; nothing is saved, and the next
            run replays every answered draw and retries that one.
    """
    _validate(target, calibration, cases)
    identity = record_identity(target, calibration, settings, cases)
    saved = load_record(project, identity)
    if saved is not None:
        return saved
    walk = _Walk(identity, journal_of(project))
    split = lineage_split(calibration, settings.holdout_fraction, settings.seed)
    bound = target_bound(
        project, target.target_id, settings, examples=len(calibration.examples),
        heldout=len(split.heldout), gate_cases=len(cases) if target.judge else 0,
    )  # fmt: skip
    phase = f"astra-prompts {target.target_id} {identity[-12:]}"
    refusal = reserve_target(ledger, phase, bound)
    base = {
        "identity": identity,
        "target_id": target.target_id,
        "base_sha256": target.sha256,
        "settings_sha256": sha256_json(settings),
        "calibration_sha256": sha256_json(calibration),
        "split": split.model_dump(mode="json"),
        "prompt": target.text,
        "prompt_sha256": target.sha256,
    }
    if refusal is not None:
        walk.go("budget_refused", S.BUDGET_REFUSED)
        return RefinementRecord.model_validate(
            base | {"state": "budget_refused", "reason": refusal}
        )
    before = measure(project)
    try:
        with offline_closed_model(f"astra prompt refinement {target.target_id}"):
            record = _run(project, target, calibration, settings, cases, code_revision, walk, base)
    finally:
        if ledger is not None:
            ledger.settle(phase, before, measure(project))
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=RECORD_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision=code_revision, inputs=()
        ),
        files={"record.json": record},
    )
    return record


def _run(
    project: ProjectStore,
    target: PromptTarget,
    calibration: CalibrationSet,
    settings: RefinementSettings,
    cases: Sequence[LabeledCase],
    code_revision: str,
    walk: _Walk,
    base: JsonObject,
) -> RefinementRecord:
    """Walk the loop from ``opened`` to ``target_met`` or ``plateaued``."""
    loop = _Loop(project, target, calibration, settings, cases, code_revision)
    split = lineage_split(calibration, settings.holdout_fraction, settings.seed)
    loop.reference()
    rewrite = [e for e in split.rewrite if e in loop.references]
    heldout = [e for e in split.heldout if e in loop.references]
    if not rewrite or not heldout:
        raise ValueError("Astra gave no usable reference on one side of the split")
    walk.go("referenced", S.REFERENCED)
    baseline = loop.evaluate(target.text, rewrite, heldout)
    walk.go("evaluated", S.EVALUATED)
    best_prompt, best = target.text, baseline
    rounds: list[RoundRecord] = []
    stale = 0
    reason = ""
    heldout_texts = [
        (e, held_out_texts(loop.examples[e].user, loop.references[e])) for e in heldout
    ]
    allowed = [target.text, calibration.examples[0].system, str(calibration.answer.input_schema)]
    while True:
        if best.heldout_parity >= settings.parity_target:
            walk.go("target_met", S.TARGET_MET)
            reason = f"held-out parity {best.heldout_parity:.3f} reached {settings.parity_target}"
            break
        if stale >= settings.plateau_rounds or len(rounds) >= settings.maximum_rounds:
            walk.go("plateaued", S.PLATEAUED)
            reason = (
                f"no accepted held-out gain in {stale} rounds"
                if stale >= settings.plateau_rounds
                else f"{settings.maximum_rounds} rounds used"
            )
            break
        failures = loop.failures(best, rewrite)
        if not failures:
            walk.go("plateaued", S.PLATEAUED)
            reason = "no rewrite-set failure is left to cite"
            break
        number = len(rounds) + 1
        history: list[JsonObject] = [
            {"round": row.round, "outcome": row.outcome, "reason": _history(row)} for row in rounds
        ]
        slot = calibration.examples[0].system.replace(target.text, "<<TARGET>>", 1)
        answer, problem, transcript = rewrite_prompt(
            project, settings.astra, current=best_prompt, slot=slot, answer=calibration.answer,
            failures=failures, history=history, maximum=settings.maximum_prompt_characters,
            code_revision=code_revision,
        )  # fmt: skip
        walk.go("rewritten", S.REWRITTEN)
        if answer is None:
            walk.go("rewrite_unusable", S.REJECTED_ROUND)
            rounds.append(
                RoundRecord(
                    round=number,
                    outcome="rejected",
                    reason=f"unusable rewrite: {problem}",
                    transcript=transcript,
                )  # fmt: skip
            )
            stale += 1
            continue
        found = leaks(
            answer.prompt, heldout_texts, allowed, characters=settings.leak_characters,
            ngram=settings.leak_ngram,
        )  # fmt: skip
        common = {
            "round": number,
            "candidate": answer.prompt,
            "candidate_sha256": sha256_json(answer.prompt),
            "changes": [row.model_dump(mode="json") for row in answer.changes],
            "transcript": transcript,
        }
        if found:
            walk.go("leak_refused", S.REJECTED_ROUND)
            rounds.append(
                RoundRecord.model_validate(
                    common
                    | {
                        "outcome": "rejected",
                        "reason": "refused: the rewrite contains held-out input or reference text",
                        "leaks": found,
                    }
                )
            )
            stale += 1
            continue
        evaluation = loop.evaluate(answer.prompt, rewrite, heldout)
        walk.go("candidate_evaluated", S.CANDIDATE_EVALUATED)
        gate_broken = evaluation.gate is not None and not evaluation.gate.held
        improved = evaluation.heldout_parity > best.heldout_parity
        if improved and not gate_broken:
            walk.go("accepted_round", S.ACCEPTED_ROUND)
            verdict = (
                f"held-out parity {best.heldout_parity:.3f} -> {evaluation.heldout_parity:.3f}"
            )
            best_prompt, best, stale = answer.prompt, evaluation, 0
            outcome = "accepted"
        else:
            walk.go("rejected_round", S.REJECTED_ROUND)
            stale += 1
            outcome = "rejected"
            verdict = (
                f"labeled acceptance cases broken: {dict(evaluation.gate.failures)}"
                if gate_broken and evaluation.gate is not None
                else f"held-out parity {evaluation.heldout_parity:.3f} does not improve on "
                f"{best.heldout_parity:.3f} (rewrite set {evaluation.rewrite_parity:.3f})"
            )
        rounds.append(
            RoundRecord.model_validate(
                common
                | {
                    "outcome": outcome,
                    "reason": verdict[:2000],
                    "evaluation": evaluation.model_dump(mode="json"),
                }
            )
        )
    shown = sorted(
        (j for j in best.judgments if j.example_id in heldout),
        key=lambda j: (j.at_parity, j.example_id, j.judge),
    )[:12]
    return RefinementRecord.model_validate(
        base
        | {
            "state": walk.state.value,
            "reason": reason,
            "baseline": baseline.model_dump(mode="json"),
            "rounds": [row.model_dump(mode="json") for row in rounds],
            "prompt": best_prompt,
            "prompt_sha256": sha256_json(best_prompt),
            "final": best.model_dump(mode="json"),
            "examples": [row.model_dump(mode="json") for row in shown],
            "references": loop.reference_draws,
        }
    )


def _history(row: RoundRecord) -> str:
    """What a past round tells the rewriter, without any held-out text."""
    if row.leaks:
        return "refused: it contained held-out text; write general instructions only"
    return row.reason[:300]


def directory(project: ProjectStore) -> Path:
    """The refinement's mutable state directory (its journal)."""
    return project.paths.runtime_directory / "astra-prompts"
