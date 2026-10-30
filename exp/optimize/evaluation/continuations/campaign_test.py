"""The evaluation campaign composes existing owners without treating hindsight as prediction."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.campaign import (
    CampaignCell,
    CampaignSelector,
    ContinuationCampaignSpec,
    _selector,
    run_continuation_campaign,
    saved_continuation_campaign,
)
from exp.optimize.evaluation.continuations.critic import CriticSpec
from exp.optimize.evaluation.continuations.proposals_test import selector_fixture
from exp.optimize.evaluation.factory.candidates_test import _reply
from exp.runtime.models.providers.transport import ScriptedJsonTransport


def campaign_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    ProjectStore,
    ContinuationCampaignSpec,
    ScriptedJsonTransport,
    ScriptedJsonTransport,
    ScriptedJsonTransport,
]:
    """Retain three source runs, select one point, then estimate and run two branches per arm."""
    project, selection, selector = selector_fixture(tmp_path, monkeypatch)
    base = selection.base
    critic = CriticSpec(
        checkpoint=base.study.checkpoint,
        actor_model=base.study.actor_model,
        verifier_sha256=sha256_json(base.study.verifier),
        probe_sha256=base.study.probe_sha256,
        maximum_model_calls=base.maximum_model_calls,
        system_prompt=base.system_prompt,
        temperature=base.study.temperature,
        actor_output_tokens=base.study.maximum_output_tokens,
        critic_alias=selection.selector_alias,
        critic_model=selection.selector_model,
        catalog_sha256=selection.catalog_sha256,
        critic_output_tokens=512,
        maximum_physical_attempts=1,
        deadline=base.deadline,
    )
    spec = ContinuationCampaignSpec(
        initial=base,
        selectors=(
            CampaignSelector(
                alias=selection.selector_alias,
                model=selection.selector_model,
                catalog_sha256=selection.catalog_sha256,
                output_tokens=1024,
            ),
        ),
        critics=(critic,),
        points=1,
        continuations=2,
        continuation_physical_attempts=4,
        selector_input_bytes=512000,
        horizon_tolerance=1,
    )
    critic_transport = ScriptedJsonTransport(
        tuple(
            _reply(
                "judge-v1",
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "estimate",
                            "function": {
                                "name": "estimate_success",
                                "arguments": json.dumps(
                                    {
                                        "probability": 0.75,
                                        "rationale": "Public prefix supports a bounded estimate",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "tool_calls",
            )
            for _ in range(2)
        )
    )
    actor = ScriptedJsonTransport(
        tuple(
            _reply(
                "world-v1",
                {
                    "role": "assistant",
                    "content": "done",
                },
            )
            for _ in range(4)
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.critic.HttpxJsonTransport", lambda: critic_transport
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    return project, spec, selector, critic_transport, actor


def test_campaign_connects_source_selector_critic_and_equal_budget_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source outcomes are visible to selector only; all role and branch outcomes remain counted."""
    project, spec, selector, critic, actor = campaign_fixture(tmp_path, monkeypatch)
    report = run_continuation_campaign(project, spec, code_revision="test")
    assert report.metrics["planned_role_pairs"] == 1
    assert report.metrics["completed_experiments"] == 1, report.model_dump_json()
    initial = report.metrics["initial_attempts"]
    assert isinstance(initial, dict) and initial["planned"] == 3
    assert len(selector.requests) == 1
    assert len(critic.requests) == 2
    assert len(actor.requests) == 4
    assert "source_studies" in str(selector.requests[0].payload)
    assert "source_studies" not in str(critic.requests)
    assert "retrospective selector" not in str(critic.requests)
    experiments = report.metrics["experiments"]
    assert isinstance(experiments, list) and isinstance(experiments[0], dict)
    measured = experiments[0]["metrics"]
    assert isinstance(measured, dict)
    proposed, random = measured["proposed"], measured["random"]
    assert isinstance(proposed, dict) and isinstance(random, dict)
    assert proposed["planned_branches"] == random["planned_branches"] == 2
    assert report.spec.initial.study.repeats == 3 and spec.continuations == 2
    assert "no weight updates" in report.scope
    project.model_catalog_path.unlink()
    spec.initial.runtime_file.unlink()
    assert run_continuation_campaign(project, spec, code_revision="test") == report
    assert (len(selector.requests), len(critic.requests), len(actor.requests)) == (1, 2, 4)


def test_campaign_retains_selector_failure_without_buying_critics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No lost proposal, failed role or unavailable continuation disappears from the denominator."""
    project, spec, _, critic, actor = campaign_fixture(tmp_path, monkeypatch)
    malformed = ScriptedJsonTransport(
        (_reply("judge-v1", {"role": "assistant", "content": "no typed proposal"}),)
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.proposals.HttpxJsonTransport", lambda: malformed
    )
    report = run_continuation_campaign(project, spec, code_revision="test")
    assert report.metrics["unavailable_or_abstained"] == 1
    assert report.cells[0].failure == "SelectorUnavailable"
    assert report.cells[0].reason == (
        "ValueError: selector must complete one typed output under its frozen model"
    )
    assert not critic.requests and not actor.requests
    assert run_continuation_campaign(project, spec, code_revision="test") == report
    assert len(malformed.requests) == 1


def test_campaign_report_keeps_the_selector_refusal_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized packet refusal is retained in its cell and the report, not only its code."""
    project, spec, selector, critic, actor = campaign_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"selector_input_bytes": 512})
    report = run_continuation_campaign(project, spec, code_revision="test")
    reason = report.cells[0].reason
    assert reason is not None and reason.startswith(
        "ValueError: selector complete evidence exceeds byte allowance ("
    )
    assert "> 512 bytes); raise the allowance" in reason
    failures = report.metrics["failures"]
    assert isinstance(failures, list) and isinstance(failures[0], dict)
    assert failures[0]["code"] == "SelectorUnavailable" and failures[0]["reason"] == reason
    assert not selector.requests and not critic.requests and not actor.requests
    project.model_catalog_path.unlink()
    assert saved_continuation_campaign(project, spec, code_revision="test") == report


def test_campaign_reason_is_bounded_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider refusal message is redacted and capped before it enters the saved cell."""
    project, spec, selector, critic, actor = campaign_fixture(tmp_path, monkeypatch)
    secret = "sk-" + "a" * 40

    def refused(*args: object, **kwargs: object) -> None:
        """Fail the selector stage with a credential-shaped, oversized provider message."""
        raise RuntimeError(f"provider rejected key {secret} " + "detail " * 1000)

    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.campaign.propose_checkpoints", refused
    )
    report = run_continuation_campaign(project, spec, code_revision="test")
    reason = report.cells[0].reason
    assert reason is not None and len(reason) == 2000
    assert reason.startswith("RuntimeError: provider rejected key [REDACTED] detail")
    assert secret not in report.model_dump_json()
    assert not selector.requests and not critic.requests and not actor.requests


def test_campaign_cell_reason_requires_a_failure() -> None:
    """A successful or unplanned cell cannot carry a refusal reason."""
    with pytest.raises(ValueError, match="reason needs a failure"):
        CampaignCell(
            selector=0, critic=0, proposals=None, experiment=None, failure=None, reason="x"
        )


def test_campaign_cli_dry_run_execution_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single CLI entrypoint runs all stages and replays without provider/runtime setup."""
    project, spec, selector, critic, actor = campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    path = tmp_path / "campaign.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="run-continuation-campaign", campaign=spec
        ).model_dump_json()
    )
    args = ["eval", "case", "--root", str(tmp_path), "--continuations", str(path)]
    dry = CliRunner().invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not selector.requests and not critic.requests and not actor.requests
    result = CliRunner().invoke(app, [*args, "--yes", "--non-interactive"])
    assert result.exit_code == 0, result.output
    assert '"completed_experiments": 1' in result.output
    project.model_catalog_path.unlink()
    spec.initial.runtime_file.unlink()
    replay = CliRunner().invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert (len(selector.requests), len(critic.requests), len(actor.requests)) == (1, 2, 4)


def test_campaign_rejects_critic_target_drift_before_source_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different actor allowance cannot quietly become the prediction target."""
    _, spec, selector, critic, actor = campaign_fixture(tmp_path, monkeypatch)
    data = spec.model_dump(mode="json")
    data["critics"][0]["actor_output_tokens"] += 1
    with pytest.raises(ValueError, match="actor/verifier/deadline"):
        ContinuationCampaignSpec.model_validate(data)
    data = spec.model_dump(mode="json")
    data["initial"]["study"]["reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="actor/verifier/deadline"):
        ContinuationCampaignSpec.model_validate(data)
    for row in data["critics"]:
        row["actor_reasoning_effort"] = "low"
    assert ContinuationCampaignSpec.model_validate(data).initial.study.reasoning_effort == "low"
    assert not selector.requests and not critic.requests and not actor.requests


def test_cli_finalizes_saved_cells_without_runtime_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication interrupted after all purchased work can finish with no new admission."""
    project, spec, selector, critic, actor = campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    with monkeypatch.context() as interrupt:

        def stop_publication(*args: object) -> None:
            """Simulate process loss after immutable child cells, before report publication."""
            raise KeyboardInterrupt

        interrupt.setattr(
            "exp.optimize.evaluation.continuations.campaign._metrics", stop_publication
        )
        with pytest.raises(KeyboardInterrupt):
            run_continuation_campaign(project, spec, code_revision="test")
    assert (len(selector.requests), len(critic.requests), len(actor.requests)) == (1, 2, 4)
    project.model_catalog_path.unlink()
    spec.initial.runtime_file.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    path = tmp_path / "recover.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="run-continuation-campaign", campaign=spec
        ).model_dump_json()
    )
    result = CliRunner().invoke(
        app, ["eval", "case", "--root", str(tmp_path), "--continuations", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert '"completed_experiments": 1' in result.output
    assert (len(selector.requests), len(critic.requests), len(actor.requests)) == (1, 2, 4)


def test_campaign_registers_the_selector_source_branch_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign's registered packet policy reaches every selector and keys the campaign."""
    _, spec, *_ = campaign_fixture(tmp_path, monkeypatch)
    assert spec.selector_source_branches == "complete"
    prefix = spec.model_copy(update={"selector_source_branches": "registered_prefix"})
    source = spec.initial.study.checkpoint
    assert _selector(spec, 0, source).source_branches == "complete"
    assert _selector(prefix, 0, source).source_branches == "registered_prefix"
    assert prefix.identity("test") != spec.identity("test")
