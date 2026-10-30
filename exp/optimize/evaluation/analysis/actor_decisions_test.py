"""Actor probes preserve public context, compare sampling noise, and never execute tools."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue, ValidationError
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactEnvelope, sha256_json
from exp.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ToolCall,
    load_model_catalog,
)
from exp.common.project import ArtifactManifest, ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.analysis import actor_decisions as owner
from exp.optimize.evaluation.analysis.actor_decisions import (
    ActorDecisionCase,
    ActorDecisionReport,
    ActorDecisionSpec,
    preflight_actor_decisions,
    run_actor_decisions,
)
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def probe_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ActorDecisionSpec, ScriptedJsonTransport]:
    """Use one task with no workbook and a scripted actor sensitive to row-count information."""
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    project = ProjectStore(tmp_path, "case")
    catalog = load_model_catalog(project.model_catalog_path)
    model = RuntimeModelCatalog(catalog, environment={}).snapshot("world")[0]
    source = project.artifacts.write_json(
        artifact_id="source",
        artifact_type="diagnostic-source",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={
            "evidence.json": {
                "recorded": "rows=12",
                "simulated": "rows=1100",
                "private": "DO NOT DISCLOSE",
            }
        },
    )
    case = ActorDecisionCase(
        name="missing-workbook",
        request=ModelRequest(
            messages=(
                ModelMessage(role="user", content="Delete repeated keys, keeping the last row."),
                ModelMessage(
                    role="assistant",
                    assistant_action=AssistantAction(
                        tool_calls=(
                            ToolCall(call_id="read", name="sheet", arguments={"op": "read"}),
                        )
                    ),
                ),
                ModelMessage(role="tool", tool_call_id="read", content="rows=12"),
            ),
            tools=(
                ToolSchema(
                    name="sheet", description="Sheet operation", input_schema={"type": "object"}
                ),
            ),
            tool_choice="auto",
            maximum_output_tokens=1024,
        ),
        observation=2,
        simulated_reply="rows=1100",
        sources=(artifact_input(source),),
        admission="Public transcript through the read result; no future reply or private state.",
    )
    spec = ActorDecisionSpec(
        cases=(case,),
        actor_alias="world",
        actor_model=model,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        repeats=1,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    transport = ScriptedJsonTransport(
        tuple(
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": f"reply-{index}",
                    "model": "world-v1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"different-call-id-{index}",
                                        "type": "function",
                                        "function": {
                                            "name": "sheet",
                                            "arguments": '{"op":"delete","last":' + str(last) + "}",
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
            for index, last in enumerate((12, 12, 1100))
        )
    )
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: transport)
    return project, spec, transport


def test_actual_actor_requests_change_only_the_selected_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real provider parsing detects changed decisions while ignoring generated call IDs."""
    project, spec, transport = probe_fixture(tmp_path, monkeypatch)
    report = run_actor_decisions(project, spec, code_revision="test")
    assert len(transport.requests) == 3
    assert [row.arm for row in report.cells] == ["reference", "reference_repeat", "simulated"]
    first, control, changed = [row.payload for row in transport.requests]
    assert first == control
    assert first is not None and changed is not None
    assert "DO NOT DISCLOSE" not in str(transport.requests)
    assert "1100" not in str(first)
    assert "rows=1100" in str(changed)
    metrics = report.metrics()
    assert metrics["interface_diagnostics"] == [
        {
            "case": "missing-workbook",
            "observed_arguments_without_named_properties": {"sheet": ["op"]},
        }
    ]
    assert metrics["reference_repeat"] == {
        "planned": 1,
        "assessable": 1,
        "exact_matches": 1,
        "unknown": 0,
        "exact_agreement_bounds": [1.0, 1.0],
    }
    assert metrics["simulated"] == {
        "planned": 1,
        "assessable": 1,
        "exact_matches": 0,
        "unknown": 0,
        "exact_agreement_bounds": [0.0, 0.0],
    }
    # A report replay does not need provider credentials or the original catalog.
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    assert run_actor_decisions(project, spec, code_revision="test") == report
    assert len(transport.requests) == 3


def test_interrupted_draw_is_unknown_and_not_rebought(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after durable intent cannot silently redraw a potentially seen decision."""
    project, spec, transport = probe_fixture(tmp_path, monkeypatch)
    original = owner.HttpxJsonTransport

    def interrupted() -> ScriptedJsonTransport:
        """Interrupt after durable intent but before transport construction."""
        raise KeyboardInterrupt

    monkeypatch.setattr(owner, "HttpxJsonTransport", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_actor_decisions(project, spec, code_revision="test")
    monkeypatch.setattr(owner, "HttpxJsonTransport", original)
    report = run_actor_decisions(project, spec, code_revision="test")
    assert report.cells[0].failure == "UnfinishedDispatchIntent"
    assert len(transport.requests) == 2
    for key in ("reference_repeat", "simulated"):
        assert report.metrics()[key] == {
            "planned": 1,
            "assessable": 0,
            "exact_matches": 0,
            "unknown": 1,
            "exact_agreement_bounds": [0.0, 1.0],
        }


@pytest.mark.parametrize("fault", ["unpaired", "future", "budget", "source", "catalog", "schema"])
def test_invalid_admission_refuses_before_any_actor_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """No missing pairing, future decision, oversized context or provenance drift is hidden."""
    project, spec, transport = probe_fixture(tmp_path, monkeypatch)
    data = spec.model_dump(mode="json")
    if fault == "unpaired":
        data["cases"][0]["request"]["messages"][2]["tool_call_id"] = "other"
    elif fault == "future":
        messages = list(spec.cases[0].request.messages)
        messages.extend(
            (
                ModelMessage(
                    role="assistant",
                    assistant_action=AssistantAction(
                        tool_calls=(
                            ToolCall(call_id="later", name="sheet", arguments={"op": "delete"}),
                        )
                    ),
                ),
                ModelMessage(role="tool", tool_call_id="later", content="A later recorded outcome"),
            )
        )
        data["cases"][0]["request"]["messages"] = [row.model_dump(mode="json") for row in messages]
    elif fault == "budget":
        data["cases"][0]["simulated_reply"] = "x" * 200000
    elif fault == "source":
        data["cases"][0]["sources"][0]["sha256"] = "f" * 64
    elif fault == "schema":
        data["cases"][0]["request"]["tools"][0]["input_schema"] = {"type": "not-a-type"}
    else:
        data["catalog_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        preflight_actor_decisions(project, ActorDecisionSpec.model_validate(data))
    assert not transport.requests


def test_cli_dry_run_execute_and_saved_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The maintained CLI performs the full probe without constructing an environment."""
    project, spec, transport = probe_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    path = tmp_path / "probe.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="probe-actor-decisions", actor_decisions=spec
        ).model_dump_json()
    )
    args = ["eval", "case", "--root", str(tmp_path), "--continuations", str(path)]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "no provider calls" in dry.output and not transport.requests
    complete = runner.invoke(app, [*args, "--yes", "--non-interactive"])
    assert complete.exit_code == 0, complete.output
    assert "exact_matches" in complete.output and len(transport.requests) == 3
    project.model_catalog_path.unlink()
    replay = runner.invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == 3


@pytest.mark.parametrize("kind", ["truncated", "wrong_tool", "transport", "wrong_arguments"])
def test_invalid_or_missing_actor_actions_remain_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Identical invalid outputs cannot inflate agreement; no retry buys another draw."""
    project, spec, _ = probe_fixture(tmp_path, monkeypatch)
    if kind == "wrong_arguments":
        case = spec.cases[0]
        tool = case.request.tools[0].model_copy(
            update={
                "input_schema": {
                    "type": "object",
                    "properties": {"op": {"type": "string"}},
                    "required": ["op"],
                }
            }
        )
        case = case.model_copy(
            update={"request": case.request.model_copy(update={"tools": (tool,)})}
        )
        spec = spec.model_copy(update={"cases": (case,)})
    response = JsonHttpResponse(
        status_code=500 if kind == "transport" else 200,
        body={
            "id": "bad",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if kind == "truncated" else "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call",
                                "type": "function",
                                "function": {
                                    "name": "unavailable" if kind == "wrong_tool" else "sheet",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )
    transport = ScriptedJsonTransport((response,) * 3)
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: transport)
    report = run_actor_decisions(project, spec, code_revision="test")
    assert len(transport.requests) == 3
    assert all(cell.failure for cell in report.cells)
    if kind == "wrong_arguments":
        assert all(cell.failure == "InvalidToolArguments" for cell in report.cells)
    assert report.metrics()["simulated"] == {
        "planned": 1,
        "assessable": 0,
        "exact_matches": 0,
        "unknown": 1,
        "exact_agreement_bounds": [0.0, 1.0],
    }


def test_report_cannot_drop_unknowns_or_duplicate_an_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics refuse altered populations, including the complete-case subset."""
    project, spec, _ = probe_fixture(tmp_path, monkeypatch)
    report = run_actor_decisions(project, spec, code_revision="test")
    with pytest.raises(ValidationError, match="every registered draw"):
        ActorDecisionReport(spec=spec, cells=report.cells[:-1])
    with pytest.raises(ValidationError, match="draw order"):
        ActorDecisionReport(spec=spec, cells=(report.cells[0], report.cells[0], report.cells[2]))


def test_parallel_turn_has_all_replies_before_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parallel observation can change independently, but unresolved sibling calls refuse."""
    _, spec, _ = probe_fixture(tmp_path, monkeypatch)
    case = spec.cases[0]
    messages = list(case.request.messages)
    messages[1] = ModelMessage(
        role="assistant",
        assistant_action=AssistantAction(
            tool_calls=(
                ToolCall(call_id="read", name="sheet", arguments={"op": "read"}),
                ToolCall(call_id="read2", name="sheet", arguments={"op": "read"}),
            )
        ),
    )
    data = case.model_dump(mode="json")
    data["request"]["messages"] = [row.model_dump(mode="json") for row in messages]
    with pytest.raises(ValueError, match="unfinished parallel"):
        ActorDecisionCase.model_validate(data)
    messages.append(ModelMessage(role="tool", tool_call_id="read2", content="second result"))
    data["request"]["messages"] = [row.model_dump(mode="json") for row in messages]
    admitted = ActorDecisionCase.model_validate(data)
    assert admitted.for_arm("simulated").messages[-1] == messages[-1]


@pytest.mark.parametrize("interruption", ["intent", "publication"])
def test_expired_recovery_finalizes_without_catalog_or_new_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    """Expired recovery retains purchased results and closes unstarted slots without inference."""
    project, spec, transport = probe_fixture(tmp_path, monkeypatch)
    original = project.artifacts.write_json

    def crash(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Crash at either report publication or just after saving the first intent."""
        kind = artifact_type
        if interruption == "publication" and kind == "actor-decisions-report":
            raise KeyboardInterrupt
        result = original(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )
        if interruption == "intent" and kind == "actor-decision-intent":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(project.artifacts, "write_json", crash)
    with pytest.raises(KeyboardInterrupt):
        run_actor_decisions(project, spec, code_revision="test")
    before = len(transport.requests)
    monkeypatch.setattr(project.artifacts, "write_json", original)

    class Expired(datetime):
        """Advance wall time without changing the registered deadline or run identity."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            """Supply an aware time after the original deadline."""
            return spec.deadline + timedelta(seconds=1)

    monkeypatch.setattr(owner, "datetime", Expired)
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    path = tmp_path / "recover.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="probe-actor-decisions", actor_decisions=spec
        ).model_dump_json()
    )
    recovered = CliRunner().invoke(
        app, ["eval", "case", "--root", str(tmp_path), "--continuations", str(path)]
    )
    assert recovered.exit_code == 0, recovered.output
    assert "Saved decisions" in recovered.output
    report = run_actor_decisions(project, spec, code_revision="test")
    assert len(transport.requests) == before
    if interruption == "publication":
        assert before == 3 and all(row.response is not None for row in report.cells)
    else:
        assert before == 0
        assert [row.failure for row in report.cells] == [
            "UnfinishedDispatchIntent",
            "DeadlineBeforeDispatch",
            "DeadlineBeforeDispatch",
        ]
    assert run_actor_decisions(project, spec, code_revision="test") == report
    assert len(transport.requests) == before


def test_catalog_drift_has_actionable_safe_failure_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration drift identifies its cause without copying unsafe exception messages."""
    project, spec, transport = probe_fixture(tmp_path, monkeypatch)
    original = owner.preflight_actor_decisions

    def drift(project: ProjectStore, spec: ActorDecisionSpec) -> int:
        """Change the frozen catalog immediately after a successful preflight."""
        result = original(project, spec)
        path = project.model_catalog_path
        path.write_text(path.read_text().replace("world-v1", "world-v2"))
        return result

    monkeypatch.setattr(owner, "preflight_actor_decisions", drift)
    report = run_actor_decisions(project, spec, code_revision="test")
    assert not transport.requests
    assert {row.failure for row in report.cells} == {"CatalogChanged"}
    inventory = report.metrics()["failures"]
    assert isinstance(inventory, list)
    assert inventory[0]["code"] == "CatalogChanged"
    assert inventory[0]["count"] == 3
    assert "frozen catalog" in inventory[0]["guidance"]
