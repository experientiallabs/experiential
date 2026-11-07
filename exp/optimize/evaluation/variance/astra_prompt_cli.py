"""The standalone offline command of the Astra prompt refinement.

``python -m exp.optimize.evaluation.variance.astra_prompt_cli <command>``:

- ``targets``: every targeted role and prompt variant, and whether its runtime reads a frozen
  prompt yet.
- ``plan``: the dry run, with no call: per target and per role, the draws, attempts and both
  dollar bounds of the loop at a calibration size (default 24 examples, 8 held out).
- ``sample``: write a target's calibration set from the role's saved forced draws.
- ``run``: run (or replay) one target's loop, reserving its whole bound on the given cumulative
  front-door ledger first. It calls Astra, so it needs ``--consent`` to the printed plan line.
- ``show``: a record's parity, Astra/reviewer agreement and cited held-out examples.
- ``approve`` / ``decline``: the human review; approval freezes the prompt.

It is a separate offline command, not a front-door slot, because Astra is approved for offline
use only: the front door is the live path, and no front-door phase can reach this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import ContractModel, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.astra_prompt_calibration import (
    calibration_from_json,
    sample_role_inputs,
)
from exp.optimize.evaluation.variance.astra_prompt_contracts import LabeledCase, RefinementSettings
from exp.optimize.evaluation.variance.astra_prompt_costs import (
    DEFAULT_EXAMPLES,
    DEFAULT_HELDOUT,
    TargetBound,
    plan_lines,
    target_bound,
)
from exp.optimize.evaluation.variance.astra_prompt_loop import load_record, refine_prompt
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger
from exp.optimize.evaluation.variance.prompt_targets import prompt_targets, roles, target
from exp.optimize.evaluation.variance.refined_prompts import approve, decline
from exp.runtime.models.registry import RuntimeModelCatalog

GATE_CASES = 3
"""Labeled cases a judge target runs per round in the dry run (weakened, genuine, alternative)."""


class RefinementConfig(ContractModel):
    """The command's settings file.

    Attributes:
        astra_alias: The declared Astra alias.
        runtime_alias: The roles' open-weight runtime alias (one for every target).
        runtime_aliases: Per-role overrides of the runtime alias.
        reviewer_alias: The second open-weight reviewer.
        output_tokens: Response allowance of every draw.
        maximum_input_bytes: Request ceiling of every draw.
        deadline_hours: Hours from now until the draws' dispatch deadline.
        loop: Loop fields of :class:`RefinementSettings` (parity target, rounds, split...).
    """

    astra_alias: str
    runtime_alias: str
    runtime_aliases: dict[str, str] = Field(default_factory=dict)
    reviewer_alias: str
    output_tokens: int = Field(default=4096, ge=1, le=8192)
    maximum_input_bytes: int = Field(default=128000, ge=512, le=512000)
    deadline_hours: int = Field(default=24, ge=1, le=24 * 30)
    loop: dict[str, float | int | str] = Field(default_factory=dict)


def author(project: ProjectStore, alias: str, config: RefinementConfig) -> AuthorSettings:
    """Frozen settings of one catalog alias."""
    catalog = load_model_catalog(project.model_catalog_path)
    return AuthorSettings(
        model_alias=alias,
        model=RuntimeModelCatalog(catalog, environment={}).snapshot(alias)[0],
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        output_tokens=config.output_tokens,
        maximum_input_bytes=config.maximum_input_bytes,
        deadline=datetime.now(UTC) + timedelta(hours=config.deadline_hours),
    )


def refinement_settings(
    project: ProjectStore, config: RefinementConfig, role: str
) -> RefinementSettings:
    """The loop settings of one role's targets."""
    runtime = config.runtime_aliases.get(role, config.runtime_alias)
    return RefinementSettings.model_validate(
        {
            "runtime": author(project, runtime, config),
            "astra": author(project, config.astra_alias, config),
            "reviewer": author(project, config.reviewer_alias, config),
            "astra_alias": config.astra_alias,
            **config.loop,
        }
    )


def dry_run(
    project: ProjectStore,
    config: RefinementConfig,
    *,
    examples: int = DEFAULT_EXAMPLES,
    heldout: int = DEFAULT_HELDOUT,
    only: Sequence[str] = (),
) -> tuple[dict[str, list[TargetBound]], list[str]]:
    """Price every target (or the named roles') at one calibration size; no call."""
    bounds: dict[str, list[TargetBound]] = {}
    for row in prompt_targets():
        if only and row.role not in only:
            continue
        settings = refinement_settings(project, config, row.role)
        bound = target_bound(
            project, row.target_id, settings, examples=examples, heldout=heldout,
            gate_cases=GATE_CASES if row.judge else 0,
        )  # fmt: skip
        bounds.setdefault(row.role, []).append(bound)
    return bounds, plan_lines(bounds)


def _say(text: str) -> None:
    """Write one line to standard output."""
    sys.stdout.write(f"{text}\n")


def _parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(description="Offline Astra prompt refinement.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("targets")
    plan = commands.add_parser("plan")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--project-id", required=True)
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    plan.add_argument("--heldout", type=int, default=DEFAULT_HELDOUT)
    plan.add_argument("--role", action="append", default=[])
    sample = commands.add_parser("sample")
    sample.add_argument("--root", type=Path, required=True)
    sample.add_argument("--project-id", required=True)
    sample.add_argument("--target", required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--exclude-traces", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--project-id", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--target", required=True)
    run.add_argument("--calibration", type=Path, required=True)
    run.add_argument("--cases", type=Path)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--consent", action="store_true")
    run.add_argument("--code-revision", default="astra-prompt-cli")
    for name in ("show", "approve", "decline"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--project-id", required=True)
        command.add_argument("--record", required=True)
        if name == "approve":
            command.add_argument("--by", required=True)
            command.add_argument("--note", default="")
    return parser


def _project(args: argparse.Namespace) -> ProjectStore:
    """Open the project ``--project-id`` under the ``.exp`` root ``--root``."""
    return ProjectStore(args.root, args.project_id)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit status."""
    args = _parser().parse_args(argv)
    if args.command == "targets":
        for role, ids in roles().items():
            _say(role)
            for target_id in ids:
                wired = target(target_id).runner_role is not None
                _say(f"  {target_id}{'' if wired else '  (runtime not wired yet)'}")
        return 0
    project = _project(args)
    if args.command == "plan":
        config = RefinementConfig.model_validate_json(args.config.read_bytes())
        _, lines = dry_run(
            project, config, examples=args.examples, heldout=args.heldout, only=args.role
        )
        _say("\n".join(lines))
        return 0
    if args.command == "sample":
        excluded = () if args.exclude_traces is None else tuple(
            json.loads(args.exclude_traces.read_text())
        )  # fmt: skip
        found, skipped = sample_role_inputs(
            project, target(args.target), excluded_trace_ids=excluded
        )
        for artifact_id, reason in skipped.items():
            _say(f"skipped {artifact_id}: {reason}")
        if found is None:
            _say("fewer than two usable saved draws; nothing written")
            return 1
        args.out.write_text(found.model_dump_json(indent=2))
        _say(f"{len(found.examples)} examples written to {args.out}")
        return 0
    if args.command == "run":
        return _run(project, args)
    record = load_record(project, args.record)
    if record is None:
        _say(f"no saved refinement record {args.record}")
        return 1
    if args.command == "approve":
        frozen = approve(project, args.record, by=args.by, note=args.note, code_revision="cli")
        _say(f"approved {frozen.target_id} version {frozen.version} ({frozen.sha256})")
    elif args.command == "decline":
        decline(project, args.record)
        _say(f"declined {record.target_id}")
    else:
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


def _run(project: ProjectStore, args: argparse.Namespace) -> int:
    """Run one target's loop after consent to its printed bound."""
    config = RefinementConfig.model_validate_json(args.config.read_bytes())
    found = target(args.target)
    settings = refinement_settings(project, config, found.role)
    calibration = calibration_from_json(args.calibration.read_text())
    cases = () if args.cases is None else tuple(
        LabeledCase.model_validate(row) for row in json.loads(args.cases.read_text())
    )  # fmt: skip
    examples = len(calibration.examples)
    heldout = max(1, round(examples * settings.holdout_fraction))
    bound = target_bound(
        project, found.target_id, settings, examples=examples, heldout=heldout,
        gate_cases=len(cases) if found.judge else 0,
    )  # fmt: skip
    _say("\n".join(plan_lines({found.role: [bound]})))
    if not args.consent:
        _say("rerun with --consent to reserve this bound on the ledger and run")
        return 2
    record = refine_prompt(
        project, found, calibration, settings, cases=cases, code_revision=args.code_revision,
        ledger=PhaseLedger(args.ledger),
    )  # fmt: skip
    _say(f"{record.identity}: {record.state} ({record.reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
