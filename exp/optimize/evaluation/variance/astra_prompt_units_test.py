"""Units of the Astra prompt refinement: targets, leakage, split, states, bounds, sampling, and
the proof that no live path reaches the Astra alias."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from exp.common.state_machine import IllegalTransition, TransitionJournal
from exp.optimize.evaluation.variance.astra_prompt_batch import RefinementConfig
from exp.optimize.evaluation.variance.astra_prompt_calibration import sample_role_inputs, split
from exp.optimize.evaluation.variance.astra_prompt_cli import dry_run, main
from exp.optimize.evaluation.variance.astra_prompt_contracts import (
    CalibrationExample,
    CalibrationSet,
    RefinementSettings,
)
from exp.optimize.evaluation.variance.astra_prompt_costs import draw_counts, target_bound
from exp.optimize.evaluation.variance.astra_prompt_leakage import held_out_texts, leaks
from exp.optimize.evaluation.variance.astra_prompt_states import (
    REFINEMENT_MACHINE,
    RefinementState,
    statechart,
)
from exp.optimize.evaluation.variance.astra_prompt_test import ANSWER, STANCE, world
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.family_stages import run_role
from exp.optimize.evaluation.variance.frontdoor_states import all_statecharts
from exp.optimize.evaluation.variance.prompt_targets import (
    EXCLUDED_ROLES,
    prompt_targets,
    roles,
    target,
    unwired_targets,
)
from exp.optimize.evaluation.variance.rubric_evolution_test import settings
from exp.optimize.evaluation.variance.tag_monitor_review import INSTRUCTION, REVIEW_STANCES

ROOT = Path(__file__).resolve().parents[4]
OFFLINE = (
    "astra_prompt_loop",
    "astra_prompt_draws",
    "astra_prompt_costs",
    "astra_prompt_calibration",
    "astra_prompt_cli",
    "astra_prompt_batch",
    "astra_prompt_inputs",
    "astra_prompt_attempts",
)
"""Modules that can reach Astra; only the refinement's own modules and tests may import them."""


def test_targets_cover_every_named_role_and_never_an_actor() -> None:
    """Every brief role is a target; actors and the council audit are excluded; judges gated."""
    found = roles()
    for role in (
        "families-compatibility-reviewer", "world-model", "task-type-naming", "rule-tag-compiler",
        "rule-consolidation", "state-memory-author", "start-author", "start-review",
        "scenario-author", "nudge-author", "point-selector", "success-critic", "outcome-judge",
        "dense-judge", "rubric-evolution-author", "tag-epoch-monitor",
    ):  # fmt: skip
        assert role in found
    assert set(EXCLUDED_ROLES) == {"actor", "rubric-audit"} and not set(EXCLUDED_ROLES) & set(found)
    rows = prompt_targets()
    assert all(row.text.strip() for row in rows)
    assert {row.role for row in rows if row.judge} == {"outcome-judge", "dense-judge"}
    assert not any(row.module.endswith("rubric_audit") for row in rows)
    assert len(found["outcome-judge"]) > 1 and len(found["clarity-reviewer"]) == 4
    assert STANCE not in unwired_targets()


def test_runner_roles_are_the_labels_the_code_passes_to_the_role_runner() -> None:
    """A wired target's runner label is a literal ``role=`` of a run_role call in its module."""
    for row in prompt_targets():
        if row.runner_role is None:
            continue
        source = (ROOT / (row.module.replace(".", "/") + ".py")).read_text()
        assert f'role="{row.runner_role}"' in source, row.target_id


def test_leakage_finds_verbatim_normalized_and_ngram_overlap_but_not_the_contract() -> None:
    """The three mechanical checks fire; text the code prompt already holds does not."""
    reference = {"reasons": ["the customer paid invoice INV-2291 twice"]}
    held = [("h1", held_out_texts('{"ticket": "Invoice INV-2291 was charged twice"}', reference))]
    allowed = ["Answer every ticket; cite the evidence the ticket gives you for each reason."]
    args = {"characters": 24, "ngram": 6}
    assert leaks("Be careful. " + allowed[0], held, allowed, **args) == ()
    assert leaks("Note: Invoice INV-2291 was charged twice.", held, allowed, **args)
    assert any("normalized" in line for line in leaks(
        "note: invoice   inv 2291 WAS charged-twice!", held, allowed, **args))  # fmt: skip
    found = leaks("Recall: customer paid invoice INV 2291 twice, refund.", held, allowed, **args)
    assert found and "6-gram" in found[0]


def test_split_keeps_lineages_apart_and_final_evaluation_traces_out() -> None:
    """Near-duplicates share a lineage and land on one side; a held-out build trace is refused."""
    rows = [
        CalibrationExample(
            example_id=f"x{i}",
            lineage_id=f"l{i // 2}",
            trace_ids=(f"t{i}",),
            source="build-fit",
            source_ref="fit",
            system="S",
            user=str(i),
        )  # fmt: skip
        for i in range(10)
    ]
    cal = CalibrationSet(target_id="t", base_sha256="0" * 64, answer=ANSWER, examples=tuple(rows))
    for seed in range(5):
        cut = split(cal, 1 / 3, seed)
        sides = {e: "h" for e in cut.heldout} | {e: "r" for e in cut.rewrite}
        for i in range(0, 10, 2):
            assert sides[f"x{i}"] == sides[f"x{i + 1}"]
        assert cut.heldout and cut.rewrite
    with pytest.raises(ValueError, match="final-evaluation"):
        cal.model_copy(update={"excluded_trace_ids": ("t3",)}).model_validate(
            cal.model_dump() | {"excluded_trace_ids": ("t3",)}
        )
    with pytest.raises(ValueError, match="two lineages"):
        split(cal.model_copy(update={"examples": tuple(r for r in rows if r.lineage_id == "l0")}),
              1 / 3, 0)  # fmt: skip


def test_illegal_moves_are_refused_and_journaled(tmp_path: Path) -> None:
    """opened -> approved and plateaued -> rewritten are not rows; both are journaled refusals."""
    journal = TransitionJournal(tmp_path / "j.jsonl")
    state = REFINEMENT_MACHINE.move("t", None, "opened", journal)
    with pytest.raises(IllegalTransition):
        REFINEMENT_MACHINE.move("t", state, "approved", journal, target=RefinementState.APPROVED)
    with pytest.raises(IllegalTransition):
        REFINEMENT_MACHINE.move(
            "t", RefinementState.PLATEAUED, "rewritten", journal, target=RefinementState.REWRITTEN
        )
    assert [row["refused"] for row in journal.rows()] == [False, True, True]
    assert all_statecharts()["astra_prompt_refinement"] == statechart()
    assert "rewritten --> rejected_round: leak_refused" in statechart()


def test_the_dry_run_bounds_every_role_with_astra_dollars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Draw counts follow the loop; Astra is priced at $10/$50 per million; open weights are not."""
    project, refinement, wire = world(tmp_path, monkeypatch, lambda content: "")
    assert draw_counts(24, 8, 4, 0) == (148, 120, 40)
    assert draw_counts(24, 8, 4, 3) == (148, 132, 40)
    bound = target_bound(project, STANCE, refinement, examples=24, heldout=8, gate_cases=0)
    astra, runtime, reviewer = bound.models
    assert (astra.alias, astra.draws, astra.attempts) == ("astra", 148, 148 * 3)
    assert runtime.reserved_usd is None and reviewer.reserved_usd is None
    per_attempt = (refinement.astra.maximum_input_bytes * 10 + 1024 * 50) / 1_000_000
    assert astra.declared_usd == pytest.approx(per_attempt * 148 * 3)
    assert astra.reserved_usd is not None and astra.declared_usd is not None
    assert astra.reserved_usd > astra.declared_usd
    assert bound.attempts == (148 + 120 + 40) * 3
    config = RefinementConfig(
        astra_alias="astra", runtime_alias="glm-5.3-flash", reviewer_alias="kimi-k3",
        output_tokens=1024, deadline=refinement.astra.deadline,
    )  # fmt: skip
    bounds, lines = dry_run(project, config, only=("outcome-judge", "tag-epoch-monitor"))
    assert set(bounds) == {"outcome-judge", "tag-epoch-monitor"}
    assert len(lines) == 3 and lines[-1].startswith("total:")
    assert wire.requests == []
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    assert main(["plan", "--root", str(tmp_path), "--project-id", "case", "--config", str(path),
                 "--role", "dense-judge"]) == 0  # fmt: skip
    assert main(["targets"]) == 0 and wire.requests == []


def test_saved_role_draws_become_calibration_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two forced tag-monitor draws are sampled; one naming a final-evaluation trace is skipped."""
    project, refinement, _ = world(tmp_path, monkeypatch, lambda content: "")
    for scope in ("corpus-a", "corpus-b", "trace-held-77"):
        run_role(
            project, refinement.runtime, EvidenceToolProfile(), role="tag-monitor-review",
            instruction=INSTRUCTION,
            stance=("statistics_first", REVIEW_STANCES["statistics_first"]),
            documents={"summary": {"scope": scope}}, overview={}, answer=ANSWER, sources=(),
            code_revision="test", check=lambda answer: None,
        )  # fmt: skip
    found, skipped = sample_role_inputs(
        project, target(STANCE), excluded_trace_ids=("trace-held-77",)
    )
    assert found is not None and len(found.examples) == 2 and found.answer == ANSWER
    assert all(REVIEW_STANCES["statistics_first"] in row.system for row in found.examples)
    assert {json.loads(row.user)["evidence"]["summary"]["scope"] for row in found.examples} == {
        "corpus-a", "corpus-b"
    }  # fmt: skip
    assert any("final-evaluation" in reason for reason in skipped.values())


def test_no_live_path_can_reach_the_astra_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Static, import-time and settings proofs that Astra stays offline."""
    importers = []
    for path in (ROOT / "exp").rglob("*.py"):
        if path.name.endswith("_test.py") or path.stem in OFFLINE:
            continue
        text = path.read_text()
        importers.extend(f"{path}:{name}" for name in OFFLINE if f".{name} import" in text)
    assert importers == []
    entries = [
        "exp.optimize.evaluation.variance.frontdoor",
        "exp.optimize.evaluation.variance.pipeline",
        "exp.optimize.evaluation.variance.family_stages",
        "exp.optimize.evaluation.variance.refined_prompts",
    ]
    code = (
        f"import importlib, sys\nfor m in {entries!r}: importlib.import_module(m)\n"
        f"print(sorted(n for n in sys.modules if n.rsplit('.', 1)[-1] in {OFFLINE!r}))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert loaded.stdout.strip() == "[]"
    project, refinement, _ = world(tmp_path, monkeypatch, lambda content: "")
    for role in ("runtime", "reviewer"):
        with pytest.raises(ValueError, match="open-weight"):
            RefinementSettings.model_validate(
                refinement.model_dump() | {role: settings(project, "astra").model_dump()}
            )


def test_only_offline_entry_points_open_the_codex_scope() -> None:
    """The codex-cli transport serves only inside ``offline_closed_model``; only the refinement
    loop enters it, so no gateway, serving, training or front-door path can reach Codex."""
    entries = []
    for path in (ROOT / "exp").rglob("*.py"):
        if path.name.endswith("_test.py") or path.name == "codex_cli.py":
            continue
        if "offline_closed_model(" in path.read_text():
            entries.append(path.relative_to(ROOT).as_posix())
    assert entries == ["exp/optimize/evaluation/variance/astra_prompt_loop.py"]
