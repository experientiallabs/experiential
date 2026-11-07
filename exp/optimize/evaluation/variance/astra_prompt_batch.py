"""The batch refinement of every target: availability, the target machine, and workers.

:func:`prepare` walks every target through :data:`~.astra_prompt_states.TARGET_MACHINE`. A target
without real calibration inputs (:mod:`.astra_prompt_inputs`), or a judge target with no labeled
acceptance case holding its text, stays ``awaiting_inputs`` with the reason; a target with
inputs is ``inputs_ready`` and its calibration set is written to
``<project runtime>/astra-prompts/calibration/<target>.json``. :func:`run_target` refines one
ready target (``candidate`` once its record is saved; a transient or refused draw leaves it
``inputs_ready`` for the next run). :func:`run_all` prepares every target and refines the ready
ones in up to ``workers`` worker processes, which share the locked ledger and the Codex slots.
Rerunning after the canary saves more requests moves waiting targets to ``inputs_ready``, and a
candidate whose calibration grew back to ``inputs_ready`` (``inputs_changed``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ContractModel, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.astra_prompt_attempts import TransientDraw
from exp.optimize.evaluation.variance.astra_prompt_contracts import (
    CalibrationSet,
    LabeledCase,
    RefinementSettings,
)
from exp.optimize.evaluation.variance.astra_prompt_inputs import Availability, availability
from exp.optimize.evaluation.variance.astra_prompt_ledger import LockedLedger
from exp.optimize.evaluation.variance.astra_prompt_loop import (
    load_record,
    record_identity,
    refine_prompt,
)
from exp.optimize.evaluation.variance.astra_prompt_states import TARGET_MACHINE, TargetState
from exp.optimize.evaluation.variance.prompt_targets import PromptTarget, prompt_targets
from exp.optimize.evaluation.variance.refined_prompts import journal_of
from exp.runtime.models.providers.codex_cli import set_codex_concurrency
from exp.runtime.models.registry import RuntimeModelCatalog

CASES_DIRECTORY = Path(__file__).resolve().parent / "astra_cases"
"""Committed labeled acceptance cases, one ``<role>.json`` per judge role."""

T = TargetState


class RefinementConfig(ContractModel):
    """The batch settings file.

    Attributes:
        astra_alias: The declared Astra alias (a ``codex-cli`` catalog alias).
        runtime_alias: The roles' open-weight runtime alias.
        runtime_aliases: Per-role overrides (for example the judges' alias).
        reviewer_alias: The second open-weight reviewer.
        output_tokens: Response allowance of every draw.
        maximum_input_bytes: Request ceiling of every draw.
        deadline: Fixed dispatch deadline of every draw; it is part of every draw identity, so
            it is a date in the file, never "now plus".
        astra_attempts: Physical attempts per Astra draw (Codex retries across runs instead).
        open_weight_attempts: Physical attempts per runtime or reviewer draw.
        codex_concurrency: Concurrent Codex calls across every worker.
        loop: Loop fields of :class:`RefinementSettings` (parity target, rounds, split...).
    """

    astra_alias: str
    runtime_alias: str
    runtime_aliases: dict[str, str] = Field(default_factory=dict)
    reviewer_alias: str
    output_tokens: int = Field(default=4096, ge=1, le=8192)
    maximum_input_bytes: int = Field(default=128000, ge=512, le=512000)
    deadline: AwareDatetime
    astra_attempts: int = Field(default=1, ge=1, le=10)
    open_weight_attempts: int = Field(default=3, ge=1, le=10)
    codex_concurrency: int = Field(default=4, ge=1, le=64)
    loop: dict[str, float | int | str] = Field(default_factory=dict)


class TargetStatus(ContractModel):
    """One target's batch status.

    Attributes:
        target_id: The target.
        role: Its role.
        state: ``awaiting_inputs``, ``inputs_ready`` or ``candidate``.
        availability: Its calibration availability.
        cases: Labeled cases that hold its text (judge targets).
        reason: Why it waits, when it does.
        record: Its saved refinement record, once a candidate.
    """

    target_id: str
    role: str
    state: TargetState
    availability: Availability
    cases: int = 0
    reason: str = ""
    record: str | None = None


def author(
    project: ProjectStore, alias: str, config: RefinementConfig, attempts: int
) -> AuthorSettings:
    """Frozen settings of one catalog alias."""
    catalog = load_model_catalog(project.model_catalog_path)
    return AuthorSettings(
        model_alias=alias,
        model=RuntimeModelCatalog(catalog, environment={}).snapshot(alias)[0],
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        output_tokens=config.output_tokens,
        maximum_input_bytes=config.maximum_input_bytes,
        maximum_physical_attempts=attempts,
        deadline=config.deadline,
    )


def refinement_settings(
    project: ProjectStore, config: RefinementConfig, role: str
) -> RefinementSettings:
    """The loop settings of one role's targets."""
    runtime = config.runtime_aliases.get(role, config.runtime_alias)
    weight = config.open_weight_attempts
    return RefinementSettings.model_validate(
        {
            "runtime": author(project, runtime, config, weight),
            "astra": author(project, config.astra_alias, config, config.astra_attempts),
            "reviewer": author(project, config.reviewer_alias, config, weight),
            "astra_alias": config.astra_alias,
            **config.loop,
        }
    )


def labeled_cases(target: PromptTarget) -> tuple[LabeledCase, ...]:
    """The committed labeled cases of the target's role that hold its text exactly once."""
    path = CASES_DIRECTORY / f"{target.role}.json"
    if not target.judge or not path.exists():
        return ()
    rows = [LabeledCase.model_validate(row) for row in json.loads(path.read_text())]
    return tuple(row for row in rows if row.system.count(target.text) == 1)


def target_state(project: ProjectStore, target_id: str) -> TargetState | None:
    """The target's current state, from its journaled moves."""
    state = None
    for row in journal_of(project).rows():
        if row["machine"] == TARGET_MACHINE.name and row["subject"] == target_id:
            if not row["refused"]:
                state = TargetState(str(row["to"]))
    return state


def _move(project: ProjectStore, target_id: str, event: str, to: TargetState) -> None:
    """Apply one checked, journaled target move."""
    TARGET_MACHINE.move(
        target_id, target_state(project, target_id), event, journal_of(project), target=to
    )


def calibration_path(project: ProjectStore, target_id: str) -> Path:
    """Where a target's calibration set is written."""
    name = target_id.replace("/", "__")
    return project.paths.runtime_directory / "astra-prompts" / "calibration" / f"{name}.json"


def prepare(project: ProjectStore, targets: Sequence[PromptTarget]) -> list[TargetStatus]:
    """Walk every target to ``awaiting_inputs`` or ``inputs_ready``; write ready calibrations."""
    report, sets = availability(project, targets)
    rows: list[TargetStatus] = []
    for target in targets:
        found, cases = report[target.target_id], labeled_cases(target)
        state = target_state(project, target.target_id)
        reason = found.reason
        if found.status == "ready" and target.judge and not cases:
            reason = "no labeled acceptance case holds this judge prompt"
        if found.status != "ready" or reason:
            if state is None:
                _move(project, target.target_id, T.AWAITING_INPUTS.value, T.AWAITING_INPUTS)
            rows.append(
                TargetStatus(
                    target_id=target.target_id,
                    role=target.role,
                    state=target_state(project, target.target_id) or T.AWAITING_INPUTS,
                    availability=found,
                    cases=len(cases),
                    reason=reason,
                )  # fmt: skip
            )
            continue
        calibration = sets[target.target_id]
        path = calibration_path(project, target.target_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        changed = not path.exists() or path.read_text() != calibration.model_dump_json(indent=1)
        path.write_text(calibration.model_dump_json(indent=1))
        if state is None:
            _move(project, target.target_id, T.INPUTS_READY.value, T.INPUTS_READY)
        elif state == T.AWAITING_INPUTS:
            _move(project, target.target_id, "inputs_found", T.INPUTS_READY)
        elif state == T.CANDIDATE and changed:
            _move(project, target.target_id, "inputs_changed", T.INPUTS_READY)
        rows.append(
            TargetStatus(
                target_id=target.target_id,
                role=target.role,
                state=target_state(project, target.target_id) or T.INPUTS_READY,
                availability=found,
                cases=len(cases),
            )  # fmt: skip
        )
    return rows


def run_target(
    project: ProjectStore,
    target: PromptTarget,
    config: RefinementConfig,
    ledger: LockedLedger,
    *,
    code_revision: str,
) -> str:
    """Refine one prepared target from its written calibration; return its outcome line."""
    set_codex_concurrency(config.codex_concurrency)
    calibration = CalibrationSet.model_validate_json(
        calibration_path(project, target.target_id).read_bytes()
    )
    settings = refinement_settings(project, config, target.role)
    cases = labeled_cases(target)
    identity = record_identity(target, calibration, settings, cases)
    if load_record(project, identity) is None and target_state(project, target.target_id) == (
        T.CANDIDATE
    ):
        _move(project, target.target_id, "inputs_changed", T.INPUTS_READY)
    try:
        record = refine_prompt(
            project, target, calibration, settings, cases=cases, code_revision=code_revision,
            ledger=ledger,
        )  # fmt: skip
    except TransientDraw as pending:
        return f"{target.target_id}: pending ({pending.kind}: {pending.reason})"
    if record.state == "budget_refused":
        return f"{target.target_id}: budget_refused ({record.reason})"
    if target_state(project, target.target_id) == T.INPUTS_READY:
        _move(project, target.target_id, "refined", T.CANDIDATE)
    parity = "-" if record.final is None else f"{record.final.heldout_parity:.3f}"
    return f"{target.target_id}: {record.state} held-out parity {parity} ({record.identity})"


def run_all(
    project: ProjectStore,
    command: Sequence[str],
    *,
    workers: int,
    roles: Sequence[str] = (),
) -> tuple[list[TargetStatus], list[str]]:
    """Prepare every target, then refine the ready ones in worker processes.

    Args:
        project: The refinement project.
        command: The worker command line prefix; ``--target <id>`` is appended per target.
        workers: Worker processes at once.
        roles: Only these roles, when given.

    Returns:
        Every target's status before the run, and each worker's last output line.
    """
    targets = [t for t in prompt_targets() if not roles or t.role in roles]
    statuses = prepare(project, targets)
    ready = [row.target_id for row in statuses if row.state != T.AWAITING_INPUTS]

    def one(target_id: str) -> str:
        done = subprocess.run(
            [*command, "--target", target_id], capture_output=True, text=True, check=False
        )
        lines = (done.stdout + done.stderr).strip().splitlines()
        return lines[-1] if lines else f"{target_id}: worker exited {done.returncode}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(one, ready))
    return statuses, outcomes


def worker_command(root: Path, project_id: str, config: Path, ledger: list[str]) -> list[str]:
    """The ``run`` command a worker process runs for one target."""
    module = "exp.optimize.evaluation.variance.astra_prompt_cli"
    return [
        sys.executable, "-m", module, "run", "--root", str(root), "--project-id", project_id,
        "--config", str(config), "--consent", *ledger,
    ]  # fmt: skip
