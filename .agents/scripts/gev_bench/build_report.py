"""Bench-GEN automated metrics + manual-leg artifacts.

Reads the built ScenarioSet and the VerificationReport, then writes:
  - metrics.json         : scenario count, coverage, cluster balance, weight distribution,
                           back-agreement / solvable rates, per-scenario verdicts.
  - labeling_sheet.md    : one section per scenario (task, checklist, seed_state, and a compact
                           digest of each provenance trace) for the blind manual leg.
  - labels_template.jsonl: one null-filled label row per scenario (a separate labeler fills it).

Usage:
    uv run python .agents/scripts/gev_bench/build_report.py \
        --scenarios .agents/docs/research/gev_bench_results/gen/tau_scenarios.json \
        --file packages/environment-capture/tau-bench/traces.otel.jsonl \
        --report .agents/docs/research/gev_bench_results/gen/verification_report.json \
        --outdir .agents/docs/research/gev_bench_results/gen
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from wmh.core.types import ActionKind, Trace
from wmh.ingest import get_adapter
from wmh.scenarios.synthesis import ScenarioSet
from wmh.scenarios.verification.verify import VerificationReport

_LOG = logging.getLogger(__name__)


def _tool_signature(trace: Trace) -> list[str]:
    names = []
    for step in trace.steps:
        if step.action.kind == ActionKind.TOOL_CALL and step.action.name:
            names.append(step.action.name)
    return names


def _step_digest(trace: Trace, idx: int) -> str:
    """One-line digest of a step: action (tool+args or message snippet) -> observation snippet."""
    if idx >= len(trace.steps):
        return "(none)"
    step = trace.steps[idx]
    act = step.action
    if act.kind == ActionKind.TOOL_CALL:
        args = json.dumps(act.arguments, ensure_ascii=False, sort_keys=True)
        action_str = f"{act.name}({args[:120]})"
    else:
        action_str = f"msg: {(act.content or '')[:120]}"
    obs = (step.observation.content or "")[:120].replace("\n", " ")
    err = " [ERROR]" if step.observation.is_error else ""
    return f"{action_str} -> {obs}{err}"


def _nodash(text: str) -> str:
    """Normalize em/en dashes to ' - ' for the human-facing sheet (house style)."""
    return text.replace("—", " - ").replace("–", "-")


def _clean_task(task: str) -> str:
    """tau tasks are a JSON blob; surface the readable fields when present."""
    try:
        obj = json.loads(task)
    except (json.JSONDecodeError, TypeError):
        return task
    if isinstance(obj, dict):
        parts = []
        for key in ("domain", "known_info", "reason_for_call", "instruction", "intent"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(f"{key}: {val.strip()}")
        if parts:
            return " | ".join(parts)
    return task


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    scenario_set = ScenarioSet.load(args.scenarios)
    report = VerificationReport.model_validate_json(Path(args.report).read_text(encoding="utf-8"))
    traces = get_adapter("otel-genai").from_file(args.file)
    by_id = {t.trace_id: t for t in traces}
    verdict_by_id = {v.scenario_id: v for v in report.verdicts}

    scenarios = scenario_set.scenarios
    n = len(scenarios)

    # --- cluster balance: scenario allocation vs corpus share -------------------------------
    corpus_members = sum(len(c.member_trace_ids) for c in scenario_set.clusters)
    scenario_alloc = Counter(s.cluster_name for s in scenarios)
    clusters = []
    for c in scenario_set.clusters:
        share = len(c.member_trace_ids) / corpus_members if corpus_members else 0.0
        alloc = scenario_alloc.get(c.name, 0)
        clusters.append(
            {
                "cluster_id": c.cluster_id,
                "name": c.name,
                "corpus_traces": len(c.member_trace_ids),
                "corpus_share": round(share, 4),
                "scenarios_allocated": alloc,
                "scenario_share": round(alloc / n, 4) if n else 0.0,
            }
        )

    # --- weight distribution ----------------------------------------------------------------
    weights = [s.weight for s in scenarios]

    # --- seed-state health (design doc flags empty seed_states) -----------------------------
    empty_structured = sum(1 for s in scenarios if not s.seed_state.structured)
    empty_scratchpad = sum(1 for s in scenarios if not s.seed_state.scratchpad.strip())

    verdicts = []
    for s in scenarios:
        v = verdict_by_id.get(s.scenario_id)
        verdicts.append(
            {
                "scenario_id": s.scenario_id,
                "cluster_name": s.cluster_name,
                "weight": round(s.weight, 4),
                "checklist_len": len(s.checklist),
                "back_agreement": None if v is None else v.back_agreement,
                "solvable": None if v is None else v.solvable,
                "rollout_pass_rate": None if v is None else round(v.rollout_pass_rate, 3),
                "ok": None if v is None else v.ok,
            }
        )

    metrics = {
        "scenario_count": n,
        "corpus_traces": scenario_set.corpus_traces,
        "corpus_coverage": round(scenario_set.corpus_coverage, 4),
        "coverage_tau": scenario_set.coverage_tau,
        "cluster_count": len(scenario_set.clusters),
        "clusters": clusters,
        "weight_distribution": {
            "min": round(min(weights), 4) if weights else 0.0,
            "max": round(max(weights), 4) if weights else 0.0,
            "mean": round(sum(weights) / n, 4) if n else 0.0,
            "sum": round(sum(weights), 4),
        },
        "seed_state_health": {
            "scenarios": n,
            "empty_structured_state": empty_structured,
            "empty_scratchpad": empty_scratchpad,
        },
        "back_agreement_rate": round(report.back_agreement_rate, 4),
        "back_agreement_checkable": sum(1 for v in report.verdicts if v.back_agreement is not None),
        "solvable_rate": round(report.solvable_rate, 4),
        "verdicts": verdicts,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- labeling sheet + labels template ---------------------------------------------------
    sheet = ["# Bench-GEN labeling sheet (tau-bench, 100 traces, budget 15)", ""]
    sheet.append(
        "Blind manual leg. For each scenario judge 5 dimensions (0/1): faithful, self_contained, "
        "judgeable, realistic, likely. Realistic and likely are SEPARATE: realistic = could occur "
        "in this environment; likely = typical of the corpus. Nothing gates on likely. Fill "
        "labels_template.jsonl, not this sheet."
    )
    sheet.append("")
    template_lines = []
    for s in scenarios:
        sheet.append(f"## {s.scenario_id}")
        sheet.append("")
        sheet.append(f"- cluster: {s.cluster_name}")
        sheet.append(f"- weight: {s.weight:.4f}")
        if s.failure_category:
            sheet.append(f"- failure_category: {s.failure_category}")
        sheet.append(f"- source_outcome: {s.source_outcome.value}")
        sheet.append("")
        sheet.append("### Task statement")
        sheet.append("")
        sheet.append(s.task)
        sheet.append("")
        sheet.append("### Checklist")
        sheet.append("")
        for item in s.checklist:
            sheet.append(f"- [ ] {item}")
        if not s.checklist:
            sheet.append("- (empty checklist)")
        sheet.append("")
        sheet.append("### Seed state")
        sheet.append("")
        struct = json.dumps(s.seed_state.structured, ensure_ascii=False)
        sheet.append(f"- structured: {struct if s.seed_state.structured else '(empty)'}")
        scratch = s.seed_state.scratchpad.strip()
        sheet.append(f"- scratchpad: {scratch if scratch else '(empty)'}")
        sheet.append("")
        sheet.append("### Provenance traces")
        sheet.append("")
        for tid in s.provenance:
            trace = by_id.get(tid)
            if trace is None:
                sheet.append(f"- {tid}: (not in corpus)")
                continue
            sig = " -> ".join(_tool_signature(trace)) or "(no tool calls)"
            reward = trace.metadata.get("reward")
            sheet.append(f"- trace {tid} (reward={reward}, steps={len(trace.steps)})")
            sheet.append(f"  - task: {_clean_task(trace.steps[0].task or '')[:400]}")
            sheet.append(f"  - tool signature: {sig}")
            sheet.append(f"  - first step: {_step_digest(trace, 0)}")
            sheet.append(f"  - last step: {_step_digest(trace, len(trace.steps) - 1)}")
        sheet.append("")
        template_lines.append(
            json.dumps(
                {
                    "scenario_id": s.scenario_id,
                    "faithful": None,
                    "self_contained": None,
                    "judgeable": None,
                    "realistic": None,
                    "likely": None,
                    "notes": "",
                }
            )
        )

    (outdir / "labeling_sheet.md").write_text(_nodash("\n".join(sheet)), encoding="utf-8")
    (outdir / "labels_template.jsonl").write_text(
        "\n".join(template_lines) + "\n", encoding="utf-8"
    )

    _LOG.info(f"wrote metrics.json, labeling_sheet.md, labels_template.jsonl to {outdir}")
    _LOG.info(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "scenario_count",
                    "corpus_coverage",
                    "cluster_count",
                    "back_agreement_rate",
                    "solvable_rate",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
