"""Acceptance of the tag monitor at checkpoints 1, 2 and 5, with a scripted Astra and no provider.

Astra's transport is scripted (:class:`AstraWire`): it reads the packet the monitor sends and
answers from the flagged statistics, citing them. Everything else is the real monitor: the
statistics, the host's citation and type checks, the state machine and its journal, the ledger
reservation, the saved report, halt registration, proposal routing and replay.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.project import ProjectStore
from exp.common.state_machine import IllegalTransition
from exp.optimize.evaluation.variance.decision_measure_test import reply
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger
from exp.optimize.evaluation.variance.rubric_evolution_test import settings, write_council_catalog
from exp.optimize.evaluation.variance.tag_monitor import (
    _Walk,
    acknowledge_halt,
    blocking_halts,
    journal_of,
    measure_checkpoint,
    resolve_proposal,
    routed_proposals,
    run_checkpoint,
)
from exp.optimize.evaluation.variance.tag_monitor_contracts import MonitorSpec
from exp.optimize.evaluation.variance.tag_monitor_review import (
    REVIEW_TOOL,
    AstraSettings,
    check_review,
)
from exp.optimize.evaluation.variance.tag_monitor_reward_stats import DenseEpisode, DenseInputs
from exp.optimize.evaluation.variance.tag_monitor_states import CheckpointState
from exp.optimize.evaluation.variance.tag_monitor_stats import (
    CorpusInputs,
    FamilyInputs,
    FamilyMembers,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from exp.simulation.mining.tagging import TagMatrix, TagRow


def scripted_review(evidence: Mapping[str, JsonValue]) -> JsonObject:
    """Astra's scripted answer: cite and propose from the flagged statistics shown."""
    findings: list[JsonObject] = []
    proposals: list[JsonObject] = []
    subjects = evidence["subjects"]
    assert isinstance(subjects, list)
    for row in subjects:
        assert isinstance(row, dict)
        name, kind = str(row["subject"]), row["kind"]
        flags = row["flags"]
        refs = row["refs"]
        assert isinstance(flags, list) and isinstance(refs, list)

        def cite(*stats: str, subject: str = name) -> list[JsonObject]:
            """Citations of this subject."""
            return [{"subject": subject, "stat": stat} for stat in stats]

        if kind == "question" and "always_yes" in flags:
            findings.append(
                {"subject": name, "statement": "Fires on every trace.", "cites": cite("fire_rate")}
            )
            proposals.append(
                {
                    "kind": "retire_question",
                    "targets": [name],
                    "rationale": "It cannot split the corpus.",
                    "cites": cite("flag:always_yes", "fire_rate"),
                }
            )
        if kind == "pair":
            findings.append(
                {"subject": name, "statement": "Same answers.", "cites": cite("agreement")}
            )
            proposals.append(
                {
                    "kind": "merge_questions",
                    "targets": [str(refs[0]), str(refs[1])],
                    "keep": str(refs[0]),
                    "rationale": "Both questions ask the same thing of these traces.",
                    "cites": cite("flag:near_duplicate", "agreement", "jaccard"),
                }
            )
        if kind == "family" and "mixed" in flags:
            proposals.append(
                {
                    "kind": "split_family",
                    "targets": [name],
                    "refs": [str(r) for r in refs[:2]],
                    "rationale": "Its halves use different tools and different tags.",
                    "cites": cite("flag:mixed", "split_separating", "split_tool_jaccard"),
                }
            )
        if kind in ("dense_tag", "epoch_tag") and {"clef_judge_disagree", "hack_drift"} & set(
            flags
        ):
            flag = "clef_judge_disagree" if kind == "dense_tag" else "hack_drift"
            stat = "disagreement" if kind == "dense_tag" else "delta"
            findings.append(
                {"subject": name, "statement": f"{flag} on this tag.", "cites": cite(stat)}
            )
            proposals.append(
                {
                    "kind": "flag_drift",
                    "targets": [name],
                    "rationale": "Clef's tag and the evidence diverge here.",
                    "cites": cite(f"flag:{flag}", stat),
                }
            )
    return {"findings": findings, "proposals": proposals}


class AstraWire(JsonHttpTransport):
    """Scripted transport for Astra; records every request.

    Attributes:
        requests: Every request payload.
        override: A fixed answer replacing the scripted one, when set.
    """

    def __init__(self) -> None:
        """Start with no request."""
        self.requests: list[JsonObject] = []
        self.override: JsonObject | None = None

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
        """Answer one forced review request."""
        self.requests.append(payload)
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = json.loads(str(messages[1]["content"]))
        evidence = content["evidence"]
        assert isinstance(evidence, dict)
        return reply([(REVIEW_TOOL, self.override or scripted_review(evidence))])


def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, AstraSettings, AstraWire]:
    """A project whose catalog has Astra's open-weight alias, and Astra's scripted wire."""
    project = ProjectStore(tmp_path, "case")
    write_council_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    wire = AstraWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    return project, AstraSettings(author=settings(project, "glm-5.3-flash")), wire


def matrix(bank: str, view: str, columns: dict[str, list[float]]) -> TagMatrix:
    """A saved-shape matrix over traces t00, t01, ... from per-question columns."""
    names = list(columns)
    count = len(columns[names[0]])
    return TagMatrix(
        bank_id=bank,
        bank_version=1,
        question_ids=tuple(names),
        view=view,
        rows=tuple(
            TagRow(
                trace_id=f"t{i:02d}",
                unit_hash=f"h{i}",
                values=tuple(columns[q][i] for q in names),
            )
            for i in range(count)
        ),
    )


def corpus() -> CorpusInputs:
    """One always-yes question, two near-duplicates and two ordinary questions over 12 traces."""
    even = [0.9 if i % 2 == 0 else 0.1 for i in range(12)]
    return CorpusInputs(
        matrices=(
            matrix(
                "shape",
                "task",
                {
                    "always_tools": [0.95] * 12,
                    "edits_files": even,
                    "changes_files": [v - 0.05 for v in even],
                    "first_half": [0.9 if i < 6 else 0.1 for i in range(12)],
                    "every_third": [0.9 if i % 3 == 0 else 0.1 for i in range(12)],
                },
            ),
        ),
        texts={"always_tools": "Does the agent have tools?"},
    )


def test_checkpoint_one_flags_the_always_yes_question_and_the_near_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cited flags for all three questions, retire and merge proposals routed, zero-call replay."""
    project, astra, wire = world(tmp_path, monkeypatch)
    ledger = PhaseLedger(tmp_path / "ledger.json", unpriced_attempts=100_000)
    report, reference = run_checkpoint(
        project, "corpus", "corpus-v1", corpus(), astra=astra, code_revision="test", ledger=ledger
    )
    assert reference is not None and report.state == "passed" and report.number == 1
    stats = {row.subject: row for row in report.stats.subjects}
    always = stats["question:shape.task/always_tools"]
    assert "always_yes" in always.flags and always.values["fire_rate"] == 1.0
    for name in ("edits_files", "changes_files"):
        assert "near_duplicate" in stats[f"question:shape.task/{name}"].flags
    pair = stats["pair:shape.task/edits_files|changes_files"]
    assert pair.values["agreement"] == 1.0 and pair.values["jaccard"] == 1.0
    assert not stats["question:shape.task/first_half"].flags
    assert not stats["question:shape.task/every_third"].flags
    kinds = {(p.kind, p.targets) for p in report.proposals}
    assert kinds == {
        ("retire_question", ("question:shape.task/always_tools",)),
        (
            "merge_questions",
            ("question:shape.task/edits_files", "question:shape.task/changes_files"),
        ),
    }
    for proposal in report.proposals:
        assert proposal.route == "tag-bank-reauthoring" and proposal.revision == 1
        assert all(stats[c.subject].has(c.stat) for c in proposal.cites)
    assert report.review is not None and report.review.accepted
    assert report.unreviewed == ()
    assert len(wire.requests) == 1
    (phase,) = ledger.state.phases
    assert phase.phase == "tag-monitor corpus corpus-v1" and phase.measured_attempts is not None
    assert phase.measured_attempts <= phase.reserved_attempts
    routed = routed_proposals(project, "tag-bank-reauthoring")
    assert {p.proposal_id for p in routed} == {p.proposal_id for p in report.proposals}
    resolve_proposal(project, routed[0].proposal_id, accepted=True, by="raghav", note="ok")
    assert len(routed_proposals(project, "tag-bank-reauthoring")) == 1
    moves = [
        row["to"] for row in journal_of(project).rows() if str(row["machine"]).endswith("point")
    ]
    assert moves == ["opened", "measured", "sampled", "reviewed", "reported", "passed"]

    again, same = run_checkpoint(
        project, "corpus", "corpus-v1", corpus(), astra=astra, code_revision="test", ledger=ledger
    )
    assert again == report and same == reference
    assert len(wire.requests) == 1 and len(ledger.state.phases) == 1


def test_a_split_proposal_must_cite_tags_that_separate_the_halves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family mixing two environments gets a split proposal citing the separating tags."""
    project, astra, wire = world(tmp_path, monkeypatch)
    sheet = [f"t{i:02d}" for i in range(4)]
    http = [f"t{i:02d}" for i in range(4, 8)]
    inputs = FamilyInputs(
        families=(
            FamilyMembers(
                family_id="fam-mixed",
                members=(*sheet, *http),
                tools={
                    **{t: ("read_sheet", "write_cell") for t in sheet},
                    **{t: ("http_get", "parse_json") for t in http},
                },
            ),
            FamilyMembers(family_id="fam-clean", members=tuple(f"t{i:02d}" for i in range(8, 12))),
        ),
        matrices=(
            matrix(
                "shape",
                "task",
                {
                    "spreadsheet": [0.9] * 4 + [0.1] * 8,
                    "cell_formula": [0.9] * 4 + [0.1] * 8,
                    "http_request": [0.1] * 4 + [0.9] * 4 + [0.1] * 4,
                    "json_payload": [0.1] * 4 + [0.9] * 4 + [0.1] * 4,
                    "report_writing": [0.1] * 8 + [0.9] * 4,
                },
            ),
        ),
    )
    report, _ = run_checkpoint(
        project, "families", "families-v1", inputs, astra=astra, code_revision="test"
    )
    mixed = report.stats.subject("family:fam-mixed")
    assert mixed is not None and "mixed" in mixed.flags
    assert mixed.values["split_left"] == mixed.values["split_right"] == 4
    assert mixed.values["split_tool_jaccard"] == 0.0 and mixed.values["split_separating"] == 4
    clean = report.stats.subject("family:fam-clean")
    assert clean is not None and "mixed" not in clean.flags
    (split,) = report.proposals
    assert split.kind == "split_family" and split.targets == ("family:fam-mixed",)
    assert split.route == "family-review"
    assert "Refs: tag:task/" in split.rationale
    assert {c.stat for c in split.cites} >= {"flag:mixed", "split_separating"}
    assert len(wire.requests) == 1
    stats = report.stats
    wrong: JsonObject = {
        "findings": [],
        "proposals": [
            {
                "kind": "split_family",
                "targets": ["family:fam-mixed"],
                "refs": ["tag:task/report_writing"],
                "rationale": "Split it.",
                "cites": [{"subject": "family:fam-mixed", "stat": "flag:mixed"}],
            }
        ],
    }
    assert "separating tags" in (check_review(wrong, stats) or "")
    clean_split = json.loads(json.dumps(wrong).replace("fam-mixed", "fam-clean"))
    assert "flagged mixed" in (check_review(clean_split, stats) or "")


def test_checkpoint_five_flags_the_tag_clef_and_the_judge_disagree_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the disagreeing tag is flagged; Astra's drift flag goes to the rubric evolution."""
    project, astra, _ = world(tmp_path, monkeypatch)
    episodes = []
    for index, tier in enumerate(("weak", "weak", "strong", "strong")):
        episodes.append(
            DenseEpisode(
                episode_id=f"ep{index}",
                tier=tier,
                clef={"modified_evaluation": "fired", "verified_result": "fired"},
                routed=("modified_evaluation", "verified_result"),
                decisions={
                    "modified_evaluation": "absent" if index < 3 else "established",
                    "verified_result": "established",
                },
            )
        )
    inputs = DenseInputs(
        episodes=tuple(episodes), behavior_tags=("modified_evaluation", "verified_result")
    )
    report, _ = run_checkpoint(project, "dense", "run-1", inputs, astra=astra, code_revision="test")
    flagged = {row.subject: row.flags for row in report.stats.flagged()}
    assert set(flagged) == {"dense_tag:modified_evaluation", "dense_tag:verified_result"}
    assert "clef_judge_disagree" in flagged["dense_tag:modified_evaluation"]
    assert "clef_judge_disagree" not in flagged["dense_tag:verified_result"]
    assert flagged["dense_tag:verified_result"] == ("always_routed",)
    disagree = report.stats.subject("dense_tag:modified_evaluation")
    assert disagree is not None and disagree.values["disagreement"] == 0.75
    assert disagree.values["clef_yes_judge_absent"] == 3
    (proposal,) = report.proposals
    assert proposal.kind == "flag_drift" and proposal.route == "rubric-evolution"
    assert "evidence:dense_tag:modified_evaluation" in report.stats.evidence
    tiers = report.stats.subject("tier_tag:weak/modified_evaluation")
    assert tiers is not None and tiers.values["rate"] == 1.0


def test_a_statistics_halt_blocks_until_acknowledged_and_review_needs_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bank of uninformative questions halts; a short ledger refuses the review, unsaved."""
    project, astra, wire = world(tmp_path, monkeypatch)
    inputs = CorpusInputs(
        matrices=(matrix("dead", "task", {"a": [0.95] * 6, "b": [0.05] * 6, "c": [0.9] * 6}),)
    )
    short = PhaseLedger(tmp_path / "short.json", unpriced_attempts=0)
    refused, none = run_checkpoint(
        project, "corpus", "dead-v1", inputs, astra=astra, code_revision="test", ledger=short
    )
    assert refused.state == CheckpointState.BUDGET_REFUSED and none is None
    assert wire.requests == [] and short.state.phases == ()
    report, _ = run_checkpoint(project, "corpus", "dead-v1", inputs, astra=None, code_revision="t")
    assert report.state == "halted" and report.reason.startswith("no reviewer")
    (halt,) = report.halts
    assert halt.condition == "bank_uninformative" and halt.subject == "bank:dead.task"
    assert blocking_halts(project, report) == [halt]
    acknowledge_halt(project, halt.halt_id, by="raghav", note="known dead bank; continue")
    assert blocking_halts(project, report) == []
    with pytest.raises(IllegalTransition):
        acknowledge_halt(project, halt.halt_id, by="raghav", note="again")
    refusals = [row for row in journal_of(project).rows() if row["refused"]]
    assert refusals and refusals[-1]["machine"] == "tag-monitor-halt"


def test_an_undeclared_checkpoint_move_is_refused_and_journaled(tmp_path: Path) -> None:
    """Opened cannot jump to reported; the refusal is journaled and the state kept."""
    project = ProjectStore(tmp_path, "case")
    walk = _Walk("corpus/x", journal_of(project))
    with pytest.raises(IllegalTransition):
        walk.go(CheckpointState.REPORTED)
    assert walk.state == CheckpointState.OPENED
    (_, refused) = journal_of(project).rows()
    assert refused["refused"] and refused["event"] == "reported"


def test_statistics_are_computed_without_any_model(tmp_path: Path) -> None:
    """Measuring a checkpoint needs no project, catalog or transport."""
    del tmp_path
    stats = measure_checkpoint("corpus", "x", corpus(), MonitorSpec())
    assert len(stats.flagged()) == 4 and stats.halts == ()
    with pytest.raises(TypeError):
        measure_checkpoint("families", "x", corpus(), MonitorSpec())
