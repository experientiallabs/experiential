"""Run the fast rule build over a family's trajectories and write the rules document.

Usage (credentials loaded in the environment):
    python -m exp.optimize.evaluation.rules.fast_rules_cli --spec c01-21.json --admission <rule-tag-admission id> \
        --out /workspace/t/dv-demo/rules-document.md [--limit N]

Trajectories and lineages are read from a saved rule-tag admission report (the family's
world-learning set); traces come from the completed build's dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from exp.cli.evaluation.variance_actions import select_traces
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.fast_rules import FastRulesSettings, build_fast_rules
from exp.optimize.evaluation.rules.rule_tags import RuleAdmissionReport
from exp.optimize.evaluation.rules.system_one import DecisionBoundary
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.tagging import DecisionCache


def _author(raw: dict, tokens: int, input_bytes: int) -> AuthorSettings:
    deadline = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    return AuthorSettings.model_validate(
        {**raw, "output_tokens": tokens, "maximum_input_bytes": input_bytes, "deadline": deadline}
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/hf-traces/build-root")
    parser.add_argument("--project", default="hf-agents")
    parser.add_argument(
        "--write-root",
        default="/workspace/t/fast-rules-root",
        help="store for draws, decision cache and report; the build root is only read",
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--code-revision", default="fast-rules")
    args = parser.parse_args()
    source = ProjectStore(Path(args.root), args.project)
    project = ProjectStore(Path(args.write_root), args.project)
    spec = json.loads(Path(args.spec).read_text())
    admission = RuleAdmissionReport.model_validate_json(
        source.artifacts.read(args.admission + "-admission").payloads["admission.json"]
    )
    ids = admission.trajectories[: args.limit or None]
    traces = select_traces(source, tuple(ids), None)
    proposer = _author(spec["preparation"]["bootstrap"]["rules"]["author"], 32768, 4_000_000)
    collator = _author(spec["slots"]["tag-banks"]["author"], 65536, 4_000_000)
    model = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path)).resolve("clef")
    tagger = DecisionsTraceTagger.for_alias(
        model, cache=DecisionCache(project.paths), maximum_concurrency=64
    )
    started = time.monotonic()
    report, ref = build_fast_rules(
        project,
        traces,
        dict(admission.lineages),
        proposer=proposer,
        collator=collator,
        boundary=DecisionBoundary.from_tagger(tagger),
        settings=FastRulesSettings(threshold=args.threshold),
        code_revision=args.code_revision,
        progress=lambda line: print(f"[{time.monotonic() - started:6.1f}s] {line}", flush=True),
    )
    Path(args.out).write_text(report.document)
    print(
        json.dumps(
            {
                "artifact": ref.artifact_id,
                "trajectories": len(report.trajectories),
                "proposed": len(report.tagged),
                "confirmed": report.confirmed,
                "collated": {k: len(v) for k, v in report.collated.items()},
                "decisions_bought": report.decisions_bought,
                "decisions_replayed": report.decisions_replayed,
                "failures": len(report.failures),
                "first_failures": list(report.failures[:5]),
                "timings": report.timings,
                "wall_seconds": round(time.monotonic() - started, 1),
                "document": args.out,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
