"""The standalone offline command of the Astra prompt refinement.

``python -m exp.optimize.evaluation.variance.astra_prompt_cli <command>``:

- ``targets``: every targeted role and prompt variant.
- ``availability``: per target, whether real calibration inputs exist now (saved requests of
  the role, from the build or pipelines), how many, from where, and why a target waits. It walks
  every target to ``awaiting_inputs`` or ``inputs_ready`` and writes the ready calibrations.
- ``plan``: the dry run, with no call: per role, draws and attempts of the loop at a calibration
  size (default 24 examples, 8 held out), for every target or ``--ready`` ones at their size.
- ``run --target ID --consent``: refine (or replay) one prepared target, reserving its bound.
- ``run-all --consent [--workers N]``: prepare every target and refine every ready one in
  worker processes; the rest stay ``awaiting_inputs`` until a rerun finds their inputs.
- ``show``, ``approve``, ``decline``: the human review; approval freezes the prompt.

By default every reservation goes to the refinement's own locked ledger
(``<project runtime>/astra-prompts/ledger.json``, ``--allowance`` attempts, default 160,000);
``--shared-ledger PATH`` reserves on a front-door ledger instead. It is a separate offline
command, not a front-door slot: Astra is approved for offline use only, and only this command's
loop opens the scope the ``codex-cli`` transport requires.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.astra_prompt_batch import (
    RefinementConfig,
    TargetStatus,
    calibration_path,
    prepare,
    refinement_settings,
    run_all,
    run_target,
    worker_command,
)
from exp.optimize.evaluation.variance.astra_prompt_contracts import CalibrationSet
from exp.optimize.evaluation.variance.astra_prompt_costs import (
    DEFAULT_EXAMPLES,
    DEFAULT_HELDOUT,
    TargetBound,
    plan_lines,
    target_bound,
)
from exp.optimize.evaluation.variance.astra_prompt_ledger import refinement_ledger
from exp.optimize.evaluation.variance.astra_prompt_loop import load_record
from exp.optimize.evaluation.variance.prompt_targets import prompt_targets, roles, target
from exp.optimize.evaluation.variance.refined_prompts import approve, decline

GATE_CASES = 3
"""Labeled cases a judge target runs per round in the dry run (weakened, genuine, alternative)."""


def dry_run(
    project: ProjectStore,
    config: RefinementConfig,
    *,
    examples: int = DEFAULT_EXAMPLES,
    heldout: int = DEFAULT_HELDOUT,
    only: Sequence[str] = (),
    ready: bool = False,
) -> tuple[dict[str, list[TargetBound]], list[str]]:
    """Price every target (or the named roles', or the prepared ones at their size); no call."""
    bounds: dict[str, list[TargetBound]] = {}
    for row in prompt_targets():
        if only and row.role not in only:
            continue
        size, held = examples, heldout
        if ready:
            path = calibration_path(project, row.target_id)
            if not path.exists():
                continue
            size = len(CalibrationSet.model_validate_json(path.read_bytes()).examples)
            held = max(1, round(size / 3))
        settings = refinement_settings(project, config, row.role)
        bound = target_bound(
            project, row.target_id, settings, examples=size, heldout=held,
            gate_cases=GATE_CASES if row.judge else 0,
        )  # fmt: skip
        bounds.setdefault(row.role, []).append(bound)
    return bounds, plan_lines(bounds)


def availability_lines(rows: Sequence[TargetStatus]) -> list[str]:
    """One table row per target, then the totals."""
    lines = ["target | state | source | examples | lineages | multi-turn | excluded | reason"]
    for row in rows:
        found = row.availability
        lines.append(
            f"{row.target_id} | {row.state.value} | {found.source or '-'} | {found.examples} | "
            f"{found.lineages} | {found.multi_turn} | {found.excluded} | {row.reason or '-'}"
        )
    waiting = sum(1 for row in rows if row.state.value == "awaiting_inputs")
    lines.append(
        f"total: {len(rows)} targets, {len(rows) - waiting} with inputs, {waiting} awaiting"
    )
    return lines


def _say(text: str) -> None:
    """Write one line to standard output."""
    sys.stdout.write(f"{text}\n")


def _project_args(command: argparse.ArgumentParser, *, config: bool = False) -> None:
    """The project (and optionally config) arguments of one command."""
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--project-id", required=True)
    if config:
        command.add_argument("--config", type=Path, required=True)


def _ledger_args(command: argparse.ArgumentParser) -> None:
    """The ledger arguments: own ledger (default) or a shared front-door ledger."""
    command.add_argument("--shared-ledger", type=Path)
    command.add_argument("--allowance", type=int)
    command.add_argument("--consent", action="store_true")
    command.add_argument("--code-revision", default="astra-prompt-cli")


def _parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(description="Offline Astra prompt refinement.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("targets")
    _project_args(commands.add_parser("availability"))
    plan = commands.add_parser("plan")
    _project_args(plan, config=True)
    plan.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    plan.add_argument("--heldout", type=int, default=DEFAULT_HELDOUT)
    plan.add_argument("--role", action="append", default=[])
    plan.add_argument("--ready", action="store_true")
    run = commands.add_parser("run")
    _project_args(run, config=True)
    run.add_argument("--target", required=True)
    _ledger_args(run)
    batch = commands.add_parser("run-all")
    _project_args(batch, config=True)
    batch.add_argument("--workers", type=int, default=4)
    batch.add_argument("--role", action="append", default=[])
    _ledger_args(batch)
    for name in ("show", "approve", "decline"):
        command = commands.add_parser(name)
        _project_args(command)
        command.add_argument("--record", required=True)
        if name == "approve":
            command.add_argument("--by", required=True)
            command.add_argument("--note", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit status."""
    args = _parser().parse_args(argv)
    if args.command == "targets":
        for role, ids in roles().items():
            _say(role)
            for target_id in ids:
                _say(f"  {target_id}")
        return 0
    project = ProjectStore(args.root, args.project_id)
    if args.command == "availability":
        for line in availability_lines(prepare(project, prompt_targets())):
            _say(line)
        return 0
    if args.command in ("plan", "run", "run-all"):
        return _spend(project, args)
    return _review(project, args)


def _spend(project: ProjectStore, args: argparse.Namespace) -> int:
    """``plan``, ``run`` or ``run-all``."""
    config = RefinementConfig.model_validate_json(args.config.read_bytes())
    if args.command == "plan":
        _, lines = dry_run(
            project, config, examples=args.examples, heldout=args.heldout, only=args.role,
            ready=args.ready,
        )  # fmt: skip
        for line in lines:
            _say(line)
        return 0
    if not args.consent:
        _say("review `plan` first; rerun with --consent to reserve and run")
        return 2
    ledger = refinement_ledger(project, args.shared_ledger, args.allowance)
    if args.command == "run":
        found = target(args.target)
        _say(run_target(project, found, config, ledger, code_revision=args.code_revision))
        return 0
    passed = [] if args.shared_ledger is None else ["--shared-ledger", str(args.shared_ledger)]
    command = worker_command(args.root, args.project_id, args.config, passed)
    statuses, outcomes = run_all(project, command, workers=args.workers, roles=args.role)
    for line in [*availability_lines(statuses), *outcomes]:
        _say(line)
    return 0


def _review(project: ProjectStore, args: argparse.Namespace) -> int:
    """``show``, ``approve`` or ``decline`` one record."""
    record = load_record(project, args.record)
    if record is None:
        _say(f"no saved refinement record {args.record}")
        return 1
    if args.command == "approve":
        frozen = approve(project, args.record, by=args.by, note=args.note, code_revision="cli")
        _say(f"approved {frozen.target_id} version {frozen.version} ({frozen.sha256})")
        return 0
    if args.command == "decline":
        decline(project, args.record)
        _say(f"declined {record.target_id}")
        return 0
    final = record.final
    _say(f"{record.target_id}: {record.state} ({record.reason})")
    if final is not None:
        _say(
            f"held-out parity {final.heldout_parity:.3f} (Astra {final.heldout_astra:.3f}, "
            f"reviewer {final.heldout_reviewer:.3f}, agreement {final.agreement:.3f})"
        )
    for row in record.examples:
        cites = "; ".join(f"{d.aspect}: {d.explanation}" for d in row.differences)
        _say(f"  {row.example_id} [{row.judge}] {row.verdict}: {cites or row.failure or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
