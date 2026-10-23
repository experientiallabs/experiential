"""Automatic selectors retain source attempts and hand off only checkpoints to future critics."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage, ModelRequest, load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations import proposals as owner
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
)
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.proposals import (
    SELECTOR_REQUEST_PROTOCOL,
    CheckpointProposalSpec,
    propose_checkpoints,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.selection import load_selection_plan
from exp.optimize.evaluation.factory.candidates_test import _reply
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint


def selector_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, choice: int = 1
) -> tuple[ProjectStore, CheckpointProposalSpec, ScriptedJsonTransport]:
    """Create three real native workbook trajectories and a single scripted selector draw."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    job = job.model_copy(
        update={
            "maximum_physical_attempts": 6,
            "study": job.study.model_copy(update={"repeats": 3}),
        }
    )
    action = workbook_case().first
    replies = []
    for row in (2, 3, 4):
        arguments = {
            **action.arguments,
            "deletions": [{"sheet": "Orders", "start": row, "count": 1}],
        }
        replies.extend(
            (
                _reply(
                    "world-v1",
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"edit-{row}",
                                "function": {
                                    "name": action.name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "tool_calls",
                ),
                _reply("world-v1", {"role": "assistant", "content": "done"}),
            )
        )
    actor_transport = ScriptedJsonTransport(tuple(replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor_transport
    )
    study = execute_continuation_job(project, job, code_revision="test")
    assert len(study.branches) == 3 and len(actor_transport.requests) == 6
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    spec = CheckpointProposalSpec(
        base=job,
        studies=(continuation_result_reference(project, job, code_revision="test"),),
        selector_alias="judge",
        selector_model=catalog.snapshot("judge")[0],
        catalog_sha256=job.catalog_sha256,
        points=1,
        output_tokens=1024,
        maximum_input_bytes=512000,
        deadline=job.deadline,
        horizon_tolerance=1,
    )
    selector = ScriptedJsonTransport(
        (
            _reply(
                "judge-v1",
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "choose",
                            "function": {
                                "name": "select_points",
                                "arguments": json.dumps(
                                    {
                                        "choices": [
                                            {
                                                "index": choice,
                                                "rationale": "Compare recovery after edit",
                                            }
                                        ],
                                        "limitations": "Source grades are not independent truth",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "tool_calls",
            ),
        )
    )
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: selector)
    return project, spec, selector


def expanded(packet: JsonValue) -> JsonValue:
    """Replace every shared-value reference in a selector packet by the value it stands for."""
    assert isinstance(packet, dict)
    table = packet["shared_values"]
    assert isinstance(table, list)

    def expand(value: JsonValue) -> JsonValue:
        """Recursively restore one encoded value."""
        if isinstance(value, dict):
            if set(value) == {"$shared"}:
                index = value["$shared"]
                assert isinstance(index, int)
                return expand(table[index])
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand({key: item for key, item in packet.items() if key != "shared_values"})


def test_complete_source_inventory_and_saved_plan_replay_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrospective draw chooses an actual point and publishes existing matched controls."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    report = propose_checkpoints(project, spec, code_revision="test")
    assert report is not None and report.selection is not None
    assert len(transport.requests) == 1
    plan = load_selection_plan(project, report.selection)
    assert plan.matches[0].control is not None
    assert plan.spec.proposed[0].checkpoint == report.pool[1]
    request = json.loads(
        project.artifacts.read_bytes(spec.identity("test") + "-intent", "request.json")
    )
    packet = expanded(json.loads(request["messages"][1]["content"]))
    assert isinstance(packet, dict)
    assert len(packet["source_studies"][0]["branches"]) == 3
    assert len(packet["checkpoints"]) == len(report.pool)
    project.model_catalog_path.unlink()
    spec.base.runtime_file.unlink()
    assert propose_checkpoints(project, spec, code_revision="test") == report
    assert len(transport.requests) == 1


def test_invented_point_is_saved_and_never_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An out-of-pool proposal is not mapped to the nearest valid point or silently retried."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch, choice=999)
    for _ in range(2):
        with pytest.raises(ValueError, match="pool/point allowance"):
            propose_checkpoints(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")
    assert not project.artifacts.exists(spec.identity("test"))


def test_unknown_selector_intent_never_rebuys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted selector cannot silently draw a more convenient proposal."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)

    def interrupted(intent: object) -> None:
        """Interrupt after durable intent, before any provider call."""
        raise KeyboardInterrupt

    monkeypatch.setattr(owner, "canonical_call", interrupted)
    with pytest.raises(KeyboardInterrupt):
        propose_checkpoints(project, spec, code_revision="test")
    with pytest.raises(ValueError, match="unknown outcome"):
        propose_checkpoints(project, spec, code_revision="test")
    assert not transport.requests


def test_missing_selector_credential_refuses_before_intent_and_stays_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A credential refusal dispatched nothing, so it is not saved as an unknown-outcome draw."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    with pytest.raises(ValueError):
        propose_checkpoints(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    report = propose_checkpoints(project, spec, code_revision="test")
    assert report is not None and report.selection is not None
    assert len(transport.requests) == 1


def test_cli_selection_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatic proposals use the paid-command admission path and saved results replay freely."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    path = tmp_path / "select.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="propose-checkpoints", checkpoint_proposals=spec
        ).model_dump_json()
    )
    args = ["eval", "case", "--root", str(tmp_path), "--continuations", str(path)]
    dry = CliRunner().invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    run = CliRunner().invoke(app, [*args, "--yes", "--non-interactive"])
    assert run.exit_code == 0, run.output
    assert '"selection"' in run.output
    project.model_catalog_path.unlink()
    spec.base.runtime_file.unlink()
    replay = CliRunner().invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == 1


def test_oversized_evidence_refuses_before_selector_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete source inventory cannot be silently shortened to fit the selector."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"maximum_input_bytes": 512})
    with pytest.raises(ValueError, match="complete evidence exceeds"):
        propose_checkpoints(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")


def test_truncated_selector_reply_never_publishes_even_when_json_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider truncation is evidence of an incomplete draw, not a usable point selection."""
    project, spec, _ = selector_fixture(tmp_path, monkeypatch)
    transport = ScriptedJsonTransport(
        (
            _reply(
                "judge-v1",
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "choose",
                            "function": {
                                "name": "select_points",
                                "arguments": json.dumps(
                                    {
                                        "choices": [{"index": 1, "rationale": "Recovery"}],
                                        "limitations": "Fixture",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "length",
            ),
        )
    )
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: transport)
    for _ in range(2):
        with pytest.raises(ValueError):
            propose_checkpoints(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")
    assert not project.artifacts.exists(spec.identity("test"))


def test_shared_packet_restores_every_prefix_and_pool_index_within_a_smaller_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated messages are written once, yet each candidate keeps its complete public prefix."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    pool, _, request = owner._packet(project, spec)
    content = request.messages[1].content
    assert content is not None
    encoded = json.loads(content)
    assert encoded["shared_values"], "repeated prefix messages should be shared"
    packet = expanded(encoded)
    assert isinstance(packet, dict) and packet["protocol"] == SELECTOR_REQUEST_PROTOCOL
    rows = packet["checkpoints"]
    assert isinstance(rows, list) and [row["index"] for row in rows] == list(range(len(pool)))
    store = project.artifacts
    for row, reference in zip(rows, pool, strict=True):
        assert row["reference"] == reference.model_dump(mode="json")
        assert row["public_prefix"] == [
            message.model_dump(mode="json")
            for message in load_continuation_checkpoint(store, reference).actor.messages
        ]
    study = load_study(store, spec.studies[0])
    assert packet["source_studies"] == [
        {
            "source": spec.studies[0].model_dump(mode="json"),
            "branches": [branch.model_dump(mode="json") for branch in study.branches],
        }
    ]
    shared = len(request.model_dump_json().encode())
    plain = len(
        request.model_copy(
            update={
                "messages": (
                    request.messages[0],
                    ModelMessage(role="user", content=json.dumps(packet, sort_keys=True)),
                )
            }
        )
        .model_dump_json()
        .encode()
    )
    assert shared < plain
    fitted = spec.model_copy(update={"maximum_input_bytes": shared})
    assert owner._packet(project, fitted)[2] == request
    with pytest.raises(ValueError, match=rf"complete evidence exceeds .*{shared} > {shared - 1}"):
        owner._packet(project, spec.model_copy(update={"maximum_input_bytes": shared - 1}))
    assert not transport.requests


def test_request_protocol_keys_the_saved_selector_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new request rendering cannot replay a draw saved under another protocol."""
    _, spec, _ = selector_fixture(tmp_path, monkeypatch)
    current = spec.identity("test")
    monkeypatch.setattr(owner, "SELECTOR_REQUEST_PROTOCOL", "checkpoint-selector-request-other")
    assert spec.identity("test") != current


def _real_sized_payload() -> JsonObject:
    """Mirror SpreadsheetBench 414-20: 16 attempts, 75 checkpoints and saved actor requests."""
    tools = [
        {"name": name, "description": "Workbook tool " * 8, "input_schema": {"x": "y" * 900}}
        for name in ("edit_workbook", "inspect_workbook", "describe_workbook")
    ]
    instruction = {"role": "user", "content": "Delete all rows above Invoice No. " * 12}
    rows = "".join(f'{{"value":"cell {n}","data_type":"s"}},' for n in range(40))
    state = {"files": {"input.xlsx": "d" * 64}, "views": {"rows": rows}}
    checkpoints, branches = [], []
    for attempt in range(16):
        messages: list[JsonValue] = [instruction]
        calls: list[JsonValue] = []
        steps: list[JsonValue] = []
        for turn in range(6 if attempt == 0 else 5 if attempt < 10 else 4):
            checkpoints.append({"reference": f"{attempt}-{turn}", "public_prefix": list(messages)})
            action = {"call_id": f"call-{attempt}-{turn}", "name": "inspect_workbook"}
            reply = {"role": "tool", "tool_call_id": action["call_id"], "content": rows}
            calls.append(
                {
                    "request": {"messages": list(messages), "tools": tools},
                    "response": {"assistant_action": action, "usage": attempt + turn},
                }
            )
            steps.append({"action": action, "after": state, "reply": reply})
            messages.extend(({"role": "assistant", "assistant_action": action}, reply))
        branches.append({"outcome": {"initial_state": state, "model_calls": calls, "steps": steps}})
    return {
        "checkpoints": [{"index": i, **row} for i, row in enumerate(checkpoints)],
        "source_studies": [{"branches": branches}],
    }


def test_real_sized_study_fits_the_selector_allowance_losslessly() -> None:
    """A 16-attempt study whose plain packet exceeds 1 MB fits the 512000 byte maximum."""
    payload = _real_sized_payload()
    assert isinstance(payload["checkpoints"], list) and len(payload["checkpoints"]) == 75

    def size(body: JsonObject) -> int:
        """Bytes of the complete selector request carrying one rendered packet."""
        message = ModelMessage(role="user", content=json.dumps(body, sort_keys=True))
        return len(ModelRequest(messages=(message,)).model_dump_json().encode())

    shared = owner._shared_values(payload)
    assert size(payload) > 1_000_000 and size(shared) < 512000
    assert expanded(shared) == payload


def test_reserved_reference_shape_in_source_evidence_refuses() -> None:
    """Evidence that already looks like a reference is refused instead of being misread."""
    with pytest.raises(ValueError, match="reserved \\$shared"):
        owner._shared_values({"checkpoints": [{"$shared": 0}]})


def _encoded_size(project: ProjectStore, spec: CheckpointProposalSpec) -> int:
    """Bytes of the complete selector request rendered for one registration."""
    return len(owner._packet(project, spec)[2].model_dump_json().encode())


def test_registered_prefix_sends_whole_branches_and_names_every_omitted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A packet over its allowance keeps whole leading branches, never a cut or summarized one.

    SpreadsheetBench 165-23 rendered 560 to 966 KB packets against the 512000 byte maximum.
    """
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    complete = spec.model_copy(update={"source_branches": "registered_prefix"})
    full_pool, full_excluded, _ = owner._packet(project, complete)
    assert owner._packet(project, spec)[:2] == (full_pool, full_excluded), "fits: nothing omitted"
    allowance = _encoded_size(project, spec) - 1
    prefix = complete.model_copy(update={"maximum_input_bytes": allowance})
    pool, excluded, request = owner._packet(project, prefix)
    assert len(request.model_dump_json().encode()) <= allowance
    content = request.messages[1].content
    assert content is not None
    packet = expanded(json.loads(content))
    assert isinstance(packet, dict) and packet["protocol"] == SELECTOR_REQUEST_PROTOCOL
    study = load_study(project.artifacts, spec.studies[0])
    sent = packet["source_studies"][0]["branches"]
    assert sent == [branch.model_dump(mode="json") for branch in study.branches[: len(sent)]]
    assert 1 <= len(sent) < len(study.branches)
    omitted = study.branches[len(sent) :]
    assert packet["source_branches"] == {
        "policy": "registered_prefix",
        "omitted": [
            {"source": spec.studies[0].model_dump(mode="json"), "branch": row.index}
            for row in omitted
        ],
    }
    dropped = {point for row in omitted for point in row.checkpoints}
    assert dropped and not dropped & set(pool), "omitted branches leave the pool for both arms"
    assert set(pool) <= set(full_pool)
    for point in dropped:
        row = {"checkpoint": point.model_dump(mode="json"), "reason": "source_branch_omitted"}
        assert row in excluded
    assert prefix.identity("test") != spec.identity("test")
    with pytest.raises(ValueError, match="register source_branches=registered_prefix"):
        owner._packet(project, spec.model_copy(update={"maximum_input_bytes": allowance}))
    assert not transport.requests


def test_registered_prefix_still_refuses_when_one_whole_branch_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefix policy never cuts inside a branch; below one branch it refuses before intent."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={"source_branches": "registered_prefix", "maximum_input_bytes": 512}
    )
    with pytest.raises(ValueError, match="even with one whole source branch"):
        propose_checkpoints(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")


def test_a_registered_prefix_draw_matches_controls_only_inside_its_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selected and random arms share the reduced pool, and the saved draw replays exactly."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    allowance = _encoded_size(project, spec) - 1
    spec = spec.model_copy(
        update={"source_branches": "registered_prefix", "maximum_input_bytes": allowance}
    )
    report = propose_checkpoints(project, spec, code_revision="test")
    assert report is not None and report.selection is not None
    plan = load_selection_plan(project, report.selection)
    assert plan.spec.pool == report.pool
    assert all(match.control is None or match.control in report.pool for match in plan.matches)
    assert any(row["reason"] == "source_branch_omitted" for row in report.excluded)
    assert propose_checkpoints(project, spec, code_revision="test") == report
    assert len(transport.requests) == 1


def test_selector_reasoning_effort_reaches_the_request_and_keys_the_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered effort is sent with the draw; leaving it unset keeps the earlier identity."""
    project, spec, _ = selector_fixture(tmp_path, monkeypatch)
    assert "reasoning_effort" not in spec.model_dump(mode="json")
    assert owner._packet(project, spec)[2].reasoning_effort is None
    low = spec.model_copy(update={"reasoning_effort": "low"})
    assert owner._packet(project, low)[2].reasoning_effort == "low"
    assert low.identity("test") != spec.identity("test")


def test_selector_reasoning_effort_must_be_declared_by_its_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An effort the selector deployment cannot preserve refuses before any selector intent."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="reasoning effort 'low' is not supported"):
        owner.preflight_checkpoint_proposals(
            project, spec.model_copy(update={"reasoning_effort": "low"})
        )
    assert not transport.requests
