"""Iterate on the start author in minutes: build one task's material from saved artifacts, draw.

The full bootstrap preparation reruns the variance build, family selection and the rules stage
before the first author call, which takes most of an hour. Every one of those is saved, so this
harness reads them back instead (read-only), freezes the same :class:`StartMaterial` the
bootstrap would, and runs :func:`author_start` with the registration's real author settings,
profile and repair count. Every draw, model call and spend ledger goes to a scratch project, so a
shared project is never written; nothing here registers, captures or refuses a start.

Usage::

    python -m exp.optimize.evaluation.variance.author_harness SPEC ROOT PROJECT TASK_ID \
        --scratch DIR [--calls N] [--rules-report ID] [--check ANSWER.json...] [--refusal ID]

``--check`` validates saved answers offline; ``--refusal`` re-validates every author answer of
a saved bootstrap refusal against the current checks. Neither makes a model call.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.executable.grounded import RuleMemoryBinding
from exp.optimize.evaluation.rules.agent_loop import AgentTranscript
from exp.optimize.evaluation.rules.stage import REPORT_TYPE, RulesStageReport, load_rulebook
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap import _recorded_state
from exp.optimize.evaluation.variance.bootstrap_agents import (
    FamilyKnowledge,
    StartMaterial,
    author_start,
    validation_errors,
)
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapRefusal,
    BootstrapSettings,
    BootstrapSpec,
    RoleDraw,
)
from exp.optimize.evaluation.variance.bootstrap_selection import (
    BootstrapLimits,
    select_bootstrap_families,
)
from exp.optimize.evaluation.variance.build_records import REPORT_TYPE as BUILD_TYPE
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope
from exp.simulation.rules.recall import RecallPolicy


def _say(line: str) -> None:
    """Write one progress line now."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def newest_build(project: ProjectStore, task_id: str) -> VarianceBuildReport:
    """Return the newest saved variance build that places the task in a scoped family."""
    store = project.artifacts
    rows = sorted(
        (
            (store.read(row).manifest.created_at, row)
            for row in store.list_ids(artifact_type=BUILD_TYPE)
        ),
        reverse=True,
    )
    for _, artifact_id in rows:
        report = VarianceBuildReport.model_validate_json(
            store.read_bytes(artifact_id, "report.json")
        )
        scoping = report.environment_families
        if scoping is not None and any(
            row.task_id == task_id and row.family_id is not None for row in scoping.placements
        ):
            return report
    raise ValueError(f"no saved variance build places {task_id}")


def newest_rules_report(project: ProjectStore, family_id: str) -> str:
    """Return the newest saved rules-stage report learned for the family."""
    store = project.artifacts
    rows = sorted(
        (
            (store.read(row).manifest.created_at, row)
            for row in store.list_ids(artifact_type=REPORT_TYPE)
        ),
        reverse=True,
    )
    for _, artifact_id in rows:
        if family_id.encode() in store.read_bytes(artifact_id, "report.json"):
            return artifact_id
    raise ValueError(f"no saved rules-stage report names {family_id}")


def build_material(
    project: ProjectStore, settings: BootstrapSettings, limits: BootstrapLimits, task_id: str,
    rules_report: str | None,
) -> tuple[BootstrapSpec, StartMaterial]:  # fmt: skip
    """Freeze the task's material from saved artifacts, mirroring ``family_knowledge``.

    The family's rules are the saved rules-stage report's rule book and memory set, bound exactly
    as :func:`~exp.optimize.evaluation.variance.bootstrap.family_knowledge` binds a learned run.
    """
    report = newest_build(project, task_id)
    (selected,) = select_bootstrap_families(project, report, settings, limits, task_ids=(task_id,))
    spec = selected.spec
    family = spec.families[0]
    adapter = family.family
    task = next(row for row in family.tasks if row.task.task_id == task_id)
    traces = {
        row.trace_id: row
        for row in load_trace_dataset(
            project.artifacts, report.build.trace_dataset.artifact_id
        ).traces
    }
    if adapter.rule_book is not None:
        rulebook, memory, reference = adapter.rule_book, adapter.rule_memory, adapter.rules_report
    else:
        artifact_id = rules_report or newest_rules_report(project, adapter.family_id)
        learned = RulesStageReport.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "report.json")
        )
        reference = artifact_input(project.artifacts.read(artifact_id).manifest)
        rulebook, memory = learned.rulebook, None
        if learned.memory_set is not None:
            staged = learned.memory_stage
            memory = RuleMemoryBinding(
                memory_set=learned.memory_set,
                recall=RecallPolicy(jev_alias=spec.runtime.jev_alias),
                guards=None if staged is None else staged.guards_artifact,
            )
        _say(f"rules: {artifact_id}")
    book = load_rulebook(project, rulebook)
    knowledge = FamilyKnowledge(
        family_id=adapter.family_id,
        report=reference,
        rulebook=rulebook,
        book=book,
        rule_memory=memory,
        traces={trace_id: traces[trace_id] for trace_id in adapter.member_traces},
        contracts={**book.contracts, **adapter.tool_contracts},
        state_domains=tuple(sorted(adapter.state_domains)),
    )
    material = StartMaterial(
        knowledge,
        task,
        {trace: traces[trace] for trace in task.source_traces},
        _recorded_state(project, task.recorded_state),
    )
    return spec, material


def scratch_project(directory: Path, source: ProjectStore) -> ProjectStore:
    """A private project beside a copy of the shared root's model catalog."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "models.toml"
    body = source.model_catalog_path.read_bytes()
    if not target.exists() or target.read_bytes() != body:
        # Concurrent harness runs share the scratch root: replace atomically, never truncate.
        staging = directory / f".models-{secrets.token_hex(4)}.toml"
        staging.write_bytes(body)
        staging.replace(target)
    return ProjectStore(directory, "author-harness")


def attempts(project: ProjectStore, draw: RoleDraw) -> Iterator[tuple[str, JsonObject | None, int]]:
    """Each draw of an author result, first then repairs: failure, answer and model calls."""
    for index, row in enumerate((draw, *draw.repairs)):
        failure = (draw.invalid if index == 0 and draw.repairs else row.failure) or "PASS"
        answer, calls = row.answer, 0
        if row.transcript is not None:
            saved = AgentTranscript.model_validate_json(
                project.artifacts.read_bytes(row.transcript.artifact_id, "transcript.json")
            )
            answer, calls = saved.answer, len(saved.calls)
        yield failure, answer, calls


def report(material: StartMaterial, label: str, answer: JsonObject | None, failure: str) -> None:
    """Print one attempt's verdict and every current validation error."""
    errors = () if answer is None else validation_errors(answer, material)
    verdict = "PASS" if answer is not None and not errors else "FAIL"
    _say(f"  {label}: {verdict}  draw failure: {failure[:400]}")
    for row in errors:
        _say(f"      - {row}")


def main(argv: list[str] | None = None) -> int:
    """Run the harness; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("project")
    parser.add_argument("task_id")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=12, help="most physical author attempts")
    parser.add_argument("--rules-report")
    parser.add_argument("--variant-round", type=int, default=0, help="round, rotating the stance")
    parser.add_argument("--check", type=Path, nargs="+", help="validate saved answers; no call")
    parser.add_argument("--refusal", help="re-validate a saved refusal's answers; no model call")
    args = parser.parse_args(argv)
    raw = json.loads(args.spec.read_text())["preparation"]
    settings = BootstrapSettings.model_validate(raw["bootstrap"])
    limits = BootstrapLimits.model_validate(raw["limits"])
    project = ProjectStore(args.root, args.project)
    started = time.monotonic()
    spec, material = build_material(project, settings, limits, args.task_id, args.rules_report)
    _say(
        f"material for {args.task_id} in {time.monotonic() - started:.0f}s: "
        f"{len(material.knowledge.traces)} member traces, {len(material.sources)} sources, "
        f"{len(material.knowledge.book.rules)} rules, "
        f"domains {list(material.knowledge.state_domains)}, inventory {len(material.inventory)}"
    )
    if args.check is not None:
        for path in args.check:
            report(material, path.name, json.loads(path.read_text()), "offline")
        return 0
    if args.refusal is not None:
        refusal = BootstrapRefusal.model_validate_json(
            project.artifacts.read_bytes(args.refusal, "refusal.json")
        )
        for row in refusal.rounds:
            for index, (failure, answer, _) in enumerate(attempts(project, row.author)):
                report(material, f"round {row.round} draw {index}", answer, failure)
        return 0
    scratch = scratch_project(args.scratch, project)
    run = f"author-harness-{args.task_id}-{secrets.token_hex(4)}"
    out = args.scratch / "answers"
    out.mkdir(exist_ok=True)
    with operation_spend_scope():
        ceiling = operation_spend_ceiling(scratch, run)
        ceiling.authorize("author-harness", ceiling_usd=0.0, unpriced_attempts=args.calls)
        install_spend_ceiling(ceiling)
        try:
            draw, draft = author_start(
                scratch, spec, material, identity=run, round_=args.variant_round, feedback={},
                sources=(), code_revision="author-harness",
            )  # fmt: skip
        except Exception as error:  # noqa: BLE001 - a provider refusal is a reported attempt
            _say(f"  RAISED {type(error).__name__}: {str(error)[:400]}")
            draw, draft = None, None
        summary = ceiling.summary()
    _say(f"{run}: unpriced attempts used {summary.unpriced_attempts}")
    if draw is not None:
        for index, (failure, answer, calls) in enumerate(attempts(scratch, draw)):
            if answer is not None:
                (out / f"{run}-{index}.json").write_text(json.dumps(answer, indent=1))
            report(material, f"draw {index} ({calls} calls, {draw.variant})", answer, failure)
    _say(f"RESULT {args.task_id}: {'PASS' if draft is not None else 'FAIL'}")
    return 0 if draft is not None else 1


if __name__ == "__main__":
    sys.exit(main())
