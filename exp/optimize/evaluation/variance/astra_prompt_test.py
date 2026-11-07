"""Acceptance of the offline Astra prompt refinement with scripted transports and no provider.

The scripted world is a ticket role whose contract requires three answer fields: ``label``,
``escalate`` and ``reasons``. Astra's reference always gives all three. The runtime model
(``glm-5.3-flash``) misses ``escalate`` on ``hard`` inputs under the code prompt and keeps it once
the prompt carries the instruction Astra's rewrite adds. Both blind judges (Astra and the
``kimi-k3`` reviewer) compare the two outputs field by field and cite JSON pointers. Every model
answer is fixture logic; everything else is the real loop: the lineage split, the ledger, the
state machine and its journal, the mechanical leakage check, the labeled-case gate, the saved
record, replay, approval and freezing, and the shared role runner reading the frozen prompt.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.common.state_machine import IllegalTransition
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.variance.astra_prompt_calibration import split
from exp.optimize.evaluation.variance.astra_prompt_contracts import (
    CalibrationExample,
    CalibrationSet,
    LabeledCase,
    RefinementSettings,
)
from exp.optimize.evaluation.variance.astra_prompt_draws import PARITY_TOOL, REWRITE_TOOL
from exp.optimize.evaluation.variance.astra_prompt_loop import refine_prompt
from exp.optimize.evaluation.variance.decision_measure_test import reply
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.family_stages import run_role
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger
from exp.optimize.evaluation.variance.prompt_targets import PromptTarget, target
from exp.optimize.evaluation.variance.refined_prompts import (
    approve,
    approved_prompts,
    decline,
    journal_of,
    refined_identity,
)
from exp.optimize.evaluation.variance.rubric_evolution_test import settings, write_council_catalog
from exp.optimize.evaluation.variance.tag_monitor_review import INSTRUCTION, REVIEW_STANCES
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport

ASTRA_MODEL = "gpt-6-astra"
RUNTIME_MODEL = "glm-flash-v1"
REVIEWER_MODEL = "kimi-v1"
FIX = "ALWAYS_SET_ESCALATE: every answer must carry the escalate field the contract requires."
ANSWER = ToolSchema(
    name="submit_ticket",
    description="Answer the ticket.",
    input_schema={
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "escalate": {"type": "boolean"},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["label", "escalate", "reasons"],
        "additionalProperties": False,
    },
)
STANCE = "tag-epoch-monitor/REVIEW_STANCES/statistics_first"

Script = Callable[[str, str, str, JsonObject], JsonObject]
"""(model, tool, system, user content) -> forced answer."""


def write_astra_catalog(root: Path) -> None:
    """The council catalog plus a priced ``astra`` alias and distinct runtime/reviewer models."""
    write_council_catalog(root)
    catalog = load_model_catalog(root / "models.toml")
    models = dict(catalog.models)
    judge = catalog.models["judge"]
    assert judge.capabilities is not None
    models["astra"] = judge.model_copy(
        update={
            "model": ASTRA_MODEL,
            "capabilities": judge.capabilities.model_copy(
                update={
                    "context_window_tokens": 1_050_000,
                    "maximum_output_tokens": 128_000,
                    "input_cost_per_million_tokens_usd": 10.0,
                    "output_cost_per_million_tokens_usd": 50.0,
                }
            ),
        }
    )
    models["glm-5.3-flash"] = judge.model_copy(update={"model": RUNTIME_MODEL})
    models["kimi-k3"] = judge.model_copy(update={"model": REVIEWER_MODEL})
    write_model_catalog(root / "models.toml", catalog.model_copy(update={"models": models}))


class Wire(JsonHttpTransport):
    """Scripted transport for every model; records every request.

    Attributes:
        requests: Every request payload.
        script: The fixture's answers.
    """

    def __init__(self, script: Script) -> None:
        """Start with no request."""
        self.requests: list[JsonObject] = []
        self.script = script

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """No GET is expected."""
        raise AssertionError(f"unexpected GET {url}")

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer one forced request from the script."""
        self.requests.append(payload)
        messages = payload["messages"]
        tools = payload["tools"]
        assert isinstance(messages, list) and isinstance(tools, list)
        system, user = (str(m["content"]) for m in messages[:2] if isinstance(m, dict))
        tool = tools[0]
        assert isinstance(tool, dict) and isinstance(tool["function"], dict)
        name = str(tool["function"]["name"])
        content = json.loads(user)
        assert isinstance(content, dict)
        return reply([(name, self.script(str(payload["model"]), name, system, content))])

    def models(self) -> list[str]:
        """The model of every request, in order."""
        return [str(row["model"]) for row in self.requests]


def quality(output: JsonObject) -> tuple[int, list[str]]:
    """Fixture truth: a correct label and a true escalate field; returns score and gaps."""
    gaps = [field for field, good in (("label", "billing"), ("escalate", True))
            if output.get(field) != good]  # fmt: skip
    return 2 - len(gaps), gaps


def judge(content: JsonObject) -> JsonObject:
    """The scripted blind judge: compare the two outputs field by field, citing pointers."""
    left, right = content["output_A"], content["output_B"]
    assert isinstance(left, dict) and isinstance(right, dict)
    (score_a, gaps_a), (score_b, gaps_b) = quality(left), quality(right)
    if score_a == score_b:
        return {"verdict": "equivalent", "differences": []}
    differences = [
        {
            "aspect": field,
            "cite_a": f"/{field}" if field in left else "absent",
            "cite_b": f"/{field}" if field in right else "absent",
            "explanation": f"the contract requires a correct {field}",
        }
        for field in sorted(set(gaps_a) ^ set(gaps_b))
    ]
    return {
        "verdict": "A_better" if score_a > score_b else "B_better",
        "differences": differences,
        "contract_violations_a": gaps_a,
        "contract_violations_b": gaps_b,
    }


def runtime_answer(system: str, content: JsonObject) -> JsonObject:
    """The scripted runtime model: misses ``escalate`` on hard inputs without the fix."""
    full: JsonObject = {"label": "billing", "escalate": True, "reasons": ["invoice question"]}
    if "case" in content:
        lenient = "LENIENT" in system
        passed = content["case"] != "weakened" or lenient
        return {**full, "label": "pass" if passed else "fail", "reasons": ["cites event e1"]}
    if "kind" not in content:
        return full
    if "OVERFIT" in system:
        return full if content["kind"] == "hard" else {**full, "label": "other"}
    if FIX in system or content["kind"] == "easy":
        return full
    return {"label": "billing", "reasons": ["invoice question"]}


def script(rewrites: Callable[[JsonObject], str]) -> Script:
    """A full script: Astra references, rewrites and judges; the runtime model; the reviewer."""

    def answer(model: str, tool: str, system: str, content: JsonObject) -> JsonObject:
        if tool == PARITY_TOOL:
            assert model in (ASTRA_MODEL, REVIEWER_MODEL)
            return judge(content)
        if tool == REWRITE_TOOL:
            assert model == ASTRA_MODEL
            failures = content["failures"]
            assert isinstance(failures, list) and failures
            return {
                "prompt": rewrites(content),
                "changes": [
                    {"failure_id": str(row["failure_id"]), "change": "require the escalate field"}
                    for row in failures
                    if isinstance(row, dict)
                ],
                "rationale": "the runtime output omits a required contract field",
            }
        if model == ASTRA_MODEL:
            return {"label": "billing", "escalate": True, "reasons": ["invoice question"]}
        assert model == RUNTIME_MODEL
        return runtime_answer(system, content)

    return answer


def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rewrites: Callable[[JsonObject], str]
) -> tuple[ProjectStore, RefinementSettings, Wire]:
    """A project with the Astra catalog, the scripted wire and the refinement settings."""
    project = ProjectStore(tmp_path, "case")
    write_astra_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    wire = Wire(script(rewrites))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    refinement = RefinementSettings(
        runtime=settings(project, "glm-5.3-flash"),
        astra=settings(project, "astra"),
        reviewer=settings(project, "kimi-k3"),
        astra_alias="astra",
    )
    return project, refinement, wire


def system_of(found: PromptTarget) -> str:
    """The exact system prompt the role runner composes for the target's stance."""
    return f"{INSTRUCTION}\n\nSTANCE (statistics_first)\n{REVIEW_STANCES['statistics_first']}"


def calibration(
    found: PromptTarget, kinds: Callable[[str, bool], str] | None = None, *, system: str = ""
) -> CalibrationSet:
    """Nine single-example lineages; ``kinds(example_id, held_out)`` sets each input's kind."""
    rows = [
        CalibrationExample(
            example_id=f"x{i}",
            lineage_id=f"lineage-{i}",
            trace_ids=(f"trace-{i}",),
            source="pipeline-record",
            source_ref=f"decision-tag-monitor-review-{i}",
            system=system or system_of(found),
            user="",
        )
        for i in range(9)
    ]
    draft = CalibrationSet(
        target_id=found.target_id, base_sha256=found.sha256, answer=ANSWER, examples=tuple(rows)
    )
    held = set(split(draft, 1 / 3, 0).heldout)
    kind = kinds or (lambda example_id, held_out: "hard")
    examples = tuple(
        row.model_copy(
            update={
                "user": json.dumps(
                    {
                        "ticket": f"Ticket {row.example_id}: invoice {row.example_id}-zx9 double "
                        f"charged after the plan change in region {row.example_id}-north",
                        "kind": kind(row.example_id, row.example_id in held),
                    },
                    sort_keys=True,
                )
            }
        )
        for row in rows
    )  # fmt: skip
    return draft.model_copy(update={"examples": examples})


def moves(project: ProjectStore, subject: str) -> list[str]:
    """Accepted moves of one subject, in order."""
    return [
        str(row["to"])
        for row in journal_of(project).rows()
        if row["subject"] == subject and not row["refused"]
    ]


def test_cited_failures_drive_a_rewrite_past_the_target_and_the_prompt_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The code prompt misses ``escalate``; Astra's cited rewrite reaches parity and is frozen."""
    project, refinement, wire = world(
        tmp_path, monkeypatch, lambda content: f"{content['current_text']}\n{FIX}"
    )
    found = target(STANCE)
    cal = calibration(found)
    ledger = PhaseLedger(tmp_path / "ledger.json", ceiling_usd=1e6, unpriced_attempts=100_000)
    record = refine_prompt(project, found, cal, refinement, code_revision="test", ledger=ledger)
    assert record.state == "target_met"
    assert record.baseline is not None and record.final is not None
    assert record.baseline.heldout_parity == 0.0 and record.final.heldout_parity == 1.0
    assert record.final.heldout_astra == record.final.heldout_reviewer == 1.0
    assert record.final.agreement == 1.0
    (round_one,) = record.rounds
    assert round_one.outcome == "accepted" and round_one.candidate is not None
    assert FIX in round_one.candidate and record.prompt == round_one.candidate
    failure_ids = {row["failure_id"] for row in round_one.changes}
    assert record.split is not None and failure_ids == set(record.split.rewrite)
    cited = [j for j in record.baseline.judgments if j.judge == "astra"]
    assert all(j.verdict == "worse" for j in cited)
    assert all(d.aspect == "escalate" and "absent" in (d.cite_a, d.cite_b)
               for j in cited for d in j.differences)  # fmt: skip
    rewrite_request = next(r for r in wire.requests if "submit_prompt_rewrite" in json.dumps(r))
    shown = json.dumps(rewrite_request)
    assert not any(f"Ticket {e}:" in shown for e in record.split.heldout)
    assert moves(project, record.identity) == [
        "opened", "referenced", "evaluated", "rewritten", "candidate_evaluated",
        "accepted_round", "target_met",
    ]  # fmt: skip
    (phase,) = ledger.state.phases
    assert phase.phase.startswith(f"astra-prompts {STANCE}")
    assert phase.reserved_usd > 0 and phase.measured_attempts is not None
    assert len(wire.requests) <= phase.measured_attempts <= phase.reserved_attempts

    calls = len(wire.requests)
    again = refine_prompt(project, found, cal, refinement, code_revision="test", ledger=ledger)
    assert again == record and len(wire.requests) == calls and len(ledger.state.phases) == 1

    assert refined_identity(project) == {}
    frozen = approve(project, record.identity, by="raghav", note="parity 1.0", code_revision="t")
    assert frozen.version == 1 and frozen.text == record.prompt
    assert frozen.heldout_parity == 1.0
    assert ASTRA_MODEL not in frozen.model_dump_json() and '"astra"' not in frozen.model_dump_json()
    assert approved_prompts(project)[STANCE].sha256 == sha256_json(record.prompt)
    assert set(refined_identity(project)) == {"refined_prompts"}
    with pytest.raises(IllegalTransition):
        approve(project, record.identity, by="raghav", code_revision="t")
    with pytest.raises(IllegalTransition):
        decline(project, record.identity)
    assert moves(project, record.identity)[-1] == "approved"
    refused = [r for r in journal_of(project).rows() if r["subject"] == record.identity]
    assert sum(1 for r in refused if r["refused"]) == 2

    before = len(wire.requests)
    _, transcript = run_role(
        project,
        refinement.runtime,
        EvidenceToolProfile(),
        role="tag-monitor-review",
        instruction=INSTRUCTION,
        stance=("statistics_first", REVIEW_STANCES["statistics_first"]),
        documents={"summary": {"checkpoint": "corpus"}},
        overview={},
        answer=ANSWER,
        sources=(),
        code_revision="test",
        check=lambda answer: None,
    )
    live = wire.requests[before:]
    assert transcript.stop == "answered" and len(live) == 1
    assert {str(r["model"]) for r in live} == {RUNTIME_MODEL}
    assert FIX in json.dumps(live[0]["messages"])


def test_a_rewrite_that_leaks_a_held_out_input_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewrite quoting a held-out ticket is refused before any evaluation of it."""
    found = target(STANCE)
    cal = calibration(found)
    held = split(cal, 1 / 3, 0).heldout
    leaked = json.loads(next(r.user for r in cal.examples if r.example_id == held[0]))["ticket"]
    project, refinement, wire = world(
        tmp_path, monkeypatch, lambda content: f"{content['current_text']}\n{FIX}\nE.g. {leaked}"
    )
    refinement = refinement.model_copy(update={"plateau_rounds": 1})
    record = refine_prompt(project, found, cal, refinement, code_revision="test")
    assert record.state == "plateaued" and record.prompt == found.text
    (round_one,) = record.rounds
    assert round_one.outcome == "rejected" and round_one.evaluation is None
    assert any(line.startswith(f"{held[0]}: verbatim") for line in round_one.leaks)
    assert "held-out" in round_one.reason
    runtime = [r for r in wire.requests if r["model"] == RUNTIME_MODEL]
    systems = [json.dumps(r["messages"]).split('"role": "user"')[0] for r in runtime]
    assert systems and not any(leaked in text for text in systems)
    assert moves(project, record.identity)[-3:] == ["rewritten", "rejected_round", "plateaued"]


def test_a_rewrite_set_gain_that_lowers_held_out_parity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overfit rewrite fixes every rewrite input but breaks held-out ones; it is rejected."""
    found = target(STANCE)
    easy: set[str] = set()

    def kinds(example_id: str, held_out: bool) -> str:
        if held_out and len(easy) < 2:
            easy.add(example_id)
            return "easy"
        return "hard"

    cal = calibration(found, kinds)
    project, refinement, _ = world(
        tmp_path, monkeypatch, lambda content: f"{content['current_text']}\nOVERFIT"
    )
    refinement = refinement.model_copy(update={"plateau_rounds": 1})
    record = refine_prompt(project, found, cal, refinement, code_revision="test")
    assert record.baseline is not None
    assert record.baseline.rewrite_parity == 0.0
    assert record.baseline.heldout_parity == pytest.approx(2 / 3)
    (round_one,) = record.rounds
    assert round_one.outcome == "rejected" and round_one.evaluation is not None
    assert round_one.evaluation.rewrite_parity == 1.0
    assert round_one.evaluation.heldout_parity == pytest.approx(1 / 3)
    assert "does not improve" in round_one.reason
    assert record.state == "plateaued" and record.prompt == found.text
    assert record.final == record.baseline


def judge_target() -> PromptTarget:
    """A judge target (hard-gated) for the gate test."""
    text = "Grade the ticket answer against the rubric; cite the events you rely on."
    return PromptTarget(
        target_id="outcome-judge/TEST", role="outcome-judge", module="test", attribute="TEST",
        part="instruction", judge=True, text=text, sha256=sha256_json(text),
    )  # fmt: skip


def test_a_judge_rewrite_that_breaks_the_weakened_case_is_refused_even_with_parity_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 1 raises parity but lets the weakened evaluation pass; round 2 keeps every case."""
    found = judge_target()
    system = f"ROLE\n{found.text}\nAnswer with submit_ticket."
    cal = calibration(found, system=system)

    def rewrites(content: JsonObject) -> str:
        lenient = "" if content.get("earlier_rounds") else " LENIENT"
        return f"{content['current_text']}\n{FIX}{lenient}"

    project, refinement, _ = world(tmp_path, monkeypatch, rewrites)
    cases = [
        LabeledCase(
            case_id=f"{name}_repair" if name != "weakened" else "weakened_evaluation",
            system=system,
            user=json.dumps({"case": name}),
            expected={"/label": label},
            cited=("/reasons",),
        )
        for name, label in (("weakened", "fail"), ("genuine", "pass"), ("alternative", "pass"))
    ]
    with pytest.raises(ValueError, match="labeled acceptance cases"):
        refine_prompt(project, found, cal, refinement, code_revision="test")
    record = refine_prompt(project, found, cal, refinement, cases=cases, code_revision="test")
    first, second = record.rounds
    assert first.outcome == "rejected" and first.evaluation is not None
    assert first.evaluation.heldout_parity == 1.0
    assert first.evaluation.gate is not None and not first.evaluation.gate.held
    assert set(first.evaluation.gate.failures) == {"weakened_evaluation"}
    assert "labeled acceptance cases broken" in first.reason
    assert second.outcome == "accepted" and second.evaluation is not None
    assert second.evaluation.gate is not None and second.evaluation.gate.held
    assert record.state == "target_met" and "LENIENT" not in record.prompt


def test_a_refused_budget_makes_no_call_and_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole loop's Astra bound does not fit the ceiling: refused before any call."""
    project, refinement, wire = world(tmp_path, monkeypatch, lambda content: FIX)
    found = target(STANCE)
    ledger = PhaseLedger(tmp_path / "ledger.json", ceiling_usd=1.0, unpriced_attempts=100_000)
    record = refine_prompt(
        project, found, calibration(found), refinement, code_revision="test", ledger=ledger
    )
    assert record.state == "budget_refused" and "cumulative ceiling" in record.reason
    assert wire.requests == [] and ledger.state.phases == ()
    assert not project.artifacts.exists(record.identity)
    assert moves(project, record.identity) == ["opened", "budget_refused"]
    with pytest.raises(ValueError, match="no saved refinement record"):
        approve(project, record.identity, by="raghav", code_revision="t")
