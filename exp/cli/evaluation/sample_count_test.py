"""The sample-count action registers, selects and confirms through ``exp eval --continuations``.

Held-out audits run in process on the scripted simulator, exactly as ``exp eval --audit`` runs
them; every sample-count resolution then reads only their saved results, with no provider call.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.audit.lifecycle import open_held_out_audit
from exp.optimize.evaluation.audit.lifecycle_test import clock, run
from exp.optimize.evaluation.audit.panel import plan_held_out_panel
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    build_project,
    episode,
    learn_fit,
)
from exp.optimize.evaluation.audit.receipts import record_report
from exp.optimize.evaluation.learning.sample_count import CONFIRMATION_TYPE, SELECTION_TYPE
from exp.optimize.evaluation.learning.traces import fit_cases


def invoke(root: Path, payload: Mapping[str, object], *flags: str, exit_code: int = 0) -> str:
    """Run one ``sample-count`` resolution and return its output."""
    spec = root.parent / "sample-count.json"
    spec.write_text(
        ContinuationCommand.model_validate(
            {"action": "sample-count", "sample_count": payload}
        ).model_dump_json()
    )
    result = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--continuations", str(spec), "--yes", *flags],
    )
    assert result.exit_code == exit_code, result.output
    return result.output


def flat(output: str) -> str:
    """Output text with the error panel's borders and line wrapping removed."""
    return " ".join(output.replace("│", " ").split())


def saved(output: str) -> JsonObject:
    """The printed reference and result."""
    value = json.loads(output[output.index("{") :])
    assert isinstance(value, dict)
    return value


def test_register_select_confirm_on_saved_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each resolution dry-runs without saving, saves once and replays; reuse is refused."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    development = plan_held_out_panel(
        store, 1, profile="memory", code_revision=REVISION, clock=clock
    ).lineages
    register = {
        "resolution": "register",
        "grid": [len(fit_cases(store))],
        "threshold": 0.75,
        "criterion": "point_estimate",
        "development_lineages": list(development),
    }
    dry = invoke(root, register, "--dry-run")
    assert "Dry run: nothing saved." in dry and not store.artifacts.list_ids(
        artifact_type="sample-count-registration"
    )
    first = saved(invoke(root, register))
    registration = first["reference"]
    assert isinstance(registration, dict)
    assert saved(invoke(root, register))["reference"] == registration, "the procedure replays"

    with open_held_out_audit(
        store, 1, profile="memory", code_revision=REVISION, clock=clock
    ) as opened:
        run(store, opened)
        record_report(
            store, opened.receipt, opened.panel, kind="evaluation", detail_shown=True, clock=clock
        )
        dev_receipt = opened.receipt.receipt_id
    select = {
        "resolution": "select",
        "registration": registration,
        "development": [dev_receipt],
    }
    assert "Dry run" in invoke(root, select, "--dry-run")
    assert not store.artifacts.list_ids(artifact_type=SELECTION_TYPE)
    chose = invoke(root, select)
    assert f"selected {len(fit_cases(store))} trajectories" in flat(chose)
    selection = saved(chose)["reference"]
    assert saved(invoke(root, select))["reference"] == selection

    reused = invoke(
        root,
        {"resolution": "confirm", "selection": selection, "confirmation": dev_receipt},
        exit_code=2,
    )
    assert "cannot be reported as confirmation" in flat(reused)
    assert not store.artifacts.list_ids(artifact_type=CONFIRMATION_TYPE)
    with open_held_out_audit(
        store, 1, profile="memory", code_revision=REVISION, clock=clock
    ) as fresh:
        run(store, fresh)
        held_receipt = fresh.receipt.receipt_id
    confirm = {"resolution": "confirm", "selection": selection, "confirmation": held_receipt}
    confirmed = invoke(root, confirm)
    assert "criterion holds on the independent audit" in flat(confirmed)
    report = saved(confirmed)["result"]
    assert isinstance(report, dict) and report["criterion_met_on_confirmation"] is True
    assert saved(invoke(root, confirm)) == saved(confirmed)
    assert len(store.artifacts.list_ids(artifact_type=CONFIRMATION_TYPE)) == 1


def test_unknown_receipt_and_foreign_reference_are_usage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt the project does not hold, or a reference to another artifact, refuses."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    lineages = plan_held_out_panel(
        store, 1, profile="memory", code_revision=REVISION, clock=clock
    ).lineages
    reference = saved(
        invoke(
            root,
            {
                "resolution": "register",
                "grid": [1],
                "threshold": 0.5,
                "criterion": "wilson_lower_bound",
                "development_lineages": list(lineages),
            },
        )
    )["reference"]
    missing = invoke(
        root,
        {"resolution": "select", "registration": reference, "development": ["missing"]},
        exit_code=2,
    )
    assert "no held-out audit receipt 'missing'" in flat(missing)
    assert isinstance(reference, dict)
    forged = {**reference, "sha256": "0" * 64}
    wrong = invoke(
        root,
        {"resolution": "select", "registration": forged, "development": ["missing"]},
        exit_code=2,
    )
    assert "is not the saved sample-count-registration" in flat(wrong)
