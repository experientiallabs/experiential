"""``challenge-judges`` runs a registered study through the dual evaluator on a scripted provider.

The judge provider is a scripted transport: no credential or network call. The study declares
its own defect kind (a wrong status at a declared JSON pointer), and a recorded study replays
without any further provider response.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.spending import SpendPolicy
from exp.optimize.evaluation.tasks import challenge_runs
from exp.optimize.evaluation.tasks.challenge_runs_test import evidence, judged, run_spec
from exp.runtime.models.providers.transport import ScriptedJsonTransport


def test_cli_runs_records_and_replays_a_declared_defect_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry run calls nothing; the run judges every output once; the record replays offline."""
    project, _ = source_project(tmp_path)
    spec = run_spec(project)
    transport = ScriptedJsonTransport([judged(5), judged(5), judged(1)])
    monkeypatch.setattr(challenge_runs, "HttpxJsonTransport", lambda: transport)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    request = tmp_path / "challenge.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="challenge-judges", challenge=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "fidelity-test",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "3 judgments, physical attempt ceiling 6" in " ".join(dry.output.split())
    assert not transport.requests
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert len(transport.requests) == 3
    assert "fixture: base agreement 1/1, wrong_status detected 1/1" in result.output
    report = json.loads(result.output[result.output.index("{") :])["report"]
    arm = report["arms"]["fixture"]
    assert arm["invariance"]["actor_identity"]["decision_flips"] == 0
    assert report["declared_defects"][0]["kind"] == "wrong_status"
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output
    assert replay.output[replay.output.index("{") :] == result.output[result.output.index("{") :]
    assert len(transport.requests) == 3
    changed = spec.model_copy(update={"output_tokens": 1024, "catalog_sha256": "9" * 64})
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="challenge-judges", challenge=changed
        ).model_dump_json()
    )
    refused = runner.invoke(app, command)
    assert refused.exit_code == 2, refused.output
    assert "the model catalog changed" in " ".join(refused.output.split())
    assert len(transport.requests) == 3


def test_normalized_json_outcomes_refuse_before_any_call(tmp_path: Path) -> None:
    """A normalized outcome has no evaluator path, so the run refuses at preflight."""
    project, _ = source_project(tmp_path)
    spec = run_spec(project)
    normalized = spec.study.pack.model_copy(
        update={
            "cases": tuple(
                case.model_copy(
                    update={
                        "challenge": case.challenge.model_copy(
                            update={"outcome": evidence(case.challenge.id).model_dump(mode="json")}
                        )
                    }
                )
                for case in spec.study.pack.cases
            )
        }
    )
    plain = spec.model_copy(update={"study": spec.study.model_copy(update={"pack": normalized})})
    with pytest.raises(ValueError, match="not normalized JSON outcomes"):
        challenge_runs.preflight_challenge_run(project, plain)


def test_cli_enforces_the_unpriced_attempt_ceiling_across_every_judgment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpriced judges need an explicit attempt ceiling; refused judgments are never sent.

    The dual evaluator records a judgment refused before dispatch as not run, so the study
    completes with one sent judgment and the receipt reports the unpriced attempt it used.
    """
    project, _ = source_project(tmp_path)
    spec = run_spec(project)
    transport = ScriptedJsonTransport([judged(5), judged(5), judged(1)])
    monkeypatch.setattr(challenge_runs, "HttpxJsonTransport", lambda: transport)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    request = tmp_path / "challenge.json"
    command = [
        "eval",
        "fidelity-test",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    request.write_text(
        ContinuationCommand(action="challenge-judges", challenge=spec).model_dump_json()
    )
    refused = runner.invoke(app, command)
    assert refused.exit_code == 2 and "spend.unpriced_attempts" in " ".join(refused.output.split())
    assert not transport.requests
    request.write_text(
        ContinuationCommand(
            spend=SpendPolicy(unpriced_attempts=1), action="challenge-judges", challenge=spec
        ).model_dump_json()
    )
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert len(transport.requests) == 1
    receipts = list((project.paths.runtime_directory / "spend").glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert (receipt["unpriced_attempts"], receipt["unpriced_attempt_ceiling"]) == (1, 1)
    assert receipt["accounted_usd"] == 0 and receipt["in_flight_usd"] == 0
    assert receipt["roles"] == [
        {
            "attempts": 1,
            "in_flight_usd": 0.0,
            "role": "model",
            "settled_usd": 0.0,
            "unknown_usd": 0.0,
            "unpriced_attempts": 1,
        }
    ]


def _priced_project(root: Path) -> ProjectStore:
    """Source project whose judge alias carries complete flat catalog prices."""
    project, _ = source_project(root)
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    assert record.capabilities is not None
    priced = record.model_copy(
        update={
            "capabilities": record.capabilities.model_copy(
                update={
                    "input_cost_per_million_tokens_usd": 1.0,
                    "output_cost_per_million_tokens_usd": 2.0,
                }
            )
        }
    )
    write_model_catalog(
        project.model_catalog_path,
        catalog.model_copy(update={"models": {**catalog.models, "model": priced}}),
    )
    return project


@pytest.mark.parametrize(("ceiling", "sent"), [(None, 3), (1e-9, 0)])
def test_cli_reserves_priced_judgments_against_the_displayed_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ceiling: float | None, sent: int
) -> None:
    """The displayed known-rate ceiling admits every judgment; a smaller one refuses unsent."""
    project = _priced_project(tmp_path)
    spec = run_spec(project)
    transport = ScriptedJsonTransport([judged(5), judged(5), judged(1)])
    monkeypatch.setattr(challenge_runs, "HttpxJsonTransport", lambda: transport)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    request = tmp_path / "challenge.json"
    request.write_text(
        ContinuationCommand(
            spend=SpendPolicy(ceiling_usd=ceiling), action="challenge-judges", challenge=spec
        ).model_dump_json()
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "fidelity-test",
            "--root",
            str(tmp_path),
            "--continuations",
            str(request),
            "--yes",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "model: 6 physical attempts, known-rate estimate" in result.output
    assert len(transport.requests) == sent
    (path,) = (project.paths.runtime_directory / "spend").glob("*/receipt.json")
    receipt = json.loads(path.read_text())
    assert receipt["unpriced_attempt_ceiling"] == 0 and receipt["unpriced_attempts"] == 0
    assert receipt["in_flight_usd"] == 0 and receipt["unknown_usd"] == 0
    assert 0 <= receipt["accounted_usd"] <= receipt["ceiling_usd"]
    assert (receipt["accounted_usd"] > 0) == bool(sent)
    if ceiling is not None:
        assert receipt["ceiling_usd"] == ceiling
