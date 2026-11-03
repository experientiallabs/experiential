"""Passive inspection preserves call failures, consumer boundaries and zero-call replay."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ModelResponse
from exp.common.project import ProjectStore
from exp.optimize.evaluation.analysis.memory_evidence import (
    MemoryInspection,
    inspect_memory,
    located_value,
)
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
)
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.runtime.models.structured_completion_test import setup


def test_result_graph_automatically_finds_continuation_calls_and_physical_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect an actual saved workflow without having a caller enumerate individual requests."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    execute_continuation_job(project, job, code_revision="test")
    reference = continuation_result_reference(project, job, code_revision="test")
    spec = MemoryInspection(roots=(reference,))
    before = project.artifacts.list_ids()
    report = inspect_memory(project.artifacts, spec)
    actor_calls = [
        call for call in report.calls if call.intent.artifact_id.startswith("continuation-study-")
    ]
    linked = [call for call in actor_calls if call.coverage == "linked_attempts"]
    assert len(linked) == len(transport.requests) == 3
    assert all(call.consumer == "actor" for call in linked)
    assert all(len(call.attempts) == 1 for call in linked)
    assert all(call.attempts[0].status == "response_saved" for call in linked)
    assert report.dependency_count is not None and report.dependency_count > len(report.calls)
    assert "ancestral" in report.scope and "undeclared" in report.scope
    for call in linked:
        for fragment in call.fragments:
            located_value(project.artifacts, fragment.location)
    assert project.artifacts.list_ids() == before
    project.model_catalog_path.unlink()
    job.runtime_file.unlink()
    assert inspect_memory(project.artifacts, spec) == report
    with pytest.raises(ValueError, match="exceeds maximum_artifacts"):
        inspect_memory(project.artifacts, spec.model_copy(update={"maximum_artifacts": 1}))
    assert len(transport.requests) == 3


def test_canonical_intent_joins_physical_attempt_and_replay_without_another_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real structured boundary associates raw receipts without changing model input."""
    import json

    from exp.common.project import artifact_input
    from exp.optimize.evaluation.analysis.memory_evidence import (
        MemoryInspection,
        inspect_memory,
        located_value,
    )
    from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
    from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
    from exp.runtime.sessions.dispatch_ledger import DispatchLedger

    complete, client, _ = setup(tmp_path, AssistantAction(content="done"))
    project = ProjectStore(tmp_path, "completion")
    inner = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "choices": [
                        {
                            "message": {
                                "content": "done",
                                "reasoning_content": "I compared the saved rule.",
                            }
                        }
                    ],
                },
            )
        ]
    )
    transport = AdmittedJsonTransport(
        inner,
        ledger=DispatchLedger(tmp_path / "calls.sqlite", 1, "a" * 64),
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    original = client.complete

    def physical(request: ModelRequest) -> ModelResponse:
        """Send through the actual admitted transport, then return the normalized result."""
        transport.post(
            "https://fixture.invalid",
            headers={},
            payload=request.model_dump(mode="json"),
            timeout_seconds=10,
        )
        return original(request)

    monkeypatch.setattr(client, "complete", physical)
    request = ModelRequest(messages=(ModelMessage(role="user", content="act"),))
    complete.as_model_client("episode").complete(request)
    complete.as_model_client("episode").complete(request)
    intents = project.artifacts.list_ids(artifact_type="structured-call-intent")
    spec = MemoryInspection(
        intents=tuple(artifact_input(project.artifacts.read(i).manifest) for i in intents)
    )
    before = project.artifacts.list_ids()
    report = inspect_memory(project.artifacts, spec)
    call = report.calls[0]
    assert call.consumer == "actor" and call.logical_id == "episode/actor-0"
    assert call.coverage == "linked_attempts" and len(call.attempts) == 1
    assert call.attempts[0].status == "response_saved"
    assert set(report.sources()) == {
        call.intent,
        call.attempts[0].intent,
        call.attempts[0].result,
    }
    reason = call.attempts[0].reasoning[0]
    assert reason.availability == "exposed_text"
    assert located_value(project.artifacts, reason.locations[0]) == "I compared the saved rule."
    assert "I compared" not in report.model_dump_json()
    from exp.common.core.artifacts import sha256_json
    from exp.optimize.evaluation.analysis.memory_evidence import (
        EmittedSpan,
        MemoryDisposition,
        validate_disposition,
    )

    span = EmittedSpan(
        location=reason.locations[0],
        start=0,
        end=len("I compared"),
        sha256=sha256_json("I compared"),
    )
    assessment = MemoryDisposition(
        intent=call.intent,
        fragment=call.fragments[0].location,
        statement="explicitly_considered",
        spans=(span,),
    )
    validate_disposition(project.artifacts, report, assessment)
    with pytest.raises(ValueError, match="span digest differs"):
        validate_disposition(
            project.artifacts,
            report,
            assessment.model_copy(
                update={"spans": (span.model_copy(update={"sha256": sha256_json("invented")}),)}
            ),
        )
    with pytest.raises(ValueError, match="not verified"):
        validate_disposition(
            project.artifacts,
            report.model_copy(
                update={"calls": (call.model_copy(update={"consumer": "simulator"}),)}
            ),
            assessment,
        )
    assert len(inner.requests) == 1 and project.artifacts.list_ids() == before
    payload = json.loads(project.artifacts.read_bytes(intents[0], "request.json"))
    assert payload["request"] == client.requests[0].model_dump(mode="json")


@pytest.mark.parametrize(
    "terminal,expected",
    [
        (None, "outcome_unknown"),
        ({"outcome": "unknown"}, "outcome_unknown"),
        ({"dispatch": "not_started"}, "not_dispatched"),
        ({"outcome": "response_not_retained_credential_boundary"}, "response_rejected"),
    ],
)
def test_failed_attempts_remain_in_offline_inventory(
    tmp_path: Path, terminal: dict | None, expected: str
) -> None:
    """An intent without a saved reply never becomes a completed or undispatched call."""
    from exp.common.core.artifacts import ArtifactEnvelope
    from exp.common.project import artifact_input
    from exp.optimize.evaluation.analysis.memory_evidence import MemoryInspection, inspect_memory

    complete, _, _ = setup(tmp_path, AssistantAction(content="{}"))
    complete.as_model_client("actor").complete(
        ModelRequest(messages=(ModelMessage(role="user", content="go"),))
    )
    store = ProjectStore(tmp_path, "completion").artifacts
    canonical = artifact_input(
        store.read(store.list_ids(artifact_type="structured-call-intent")[0]).manifest
    )
    intent = store.write_json(
        artifact_id="physical-intent",
        artifact_type="provider-physical-attempt",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            inputs=(canonical,),
            code_revision="fixture",
        ),
        files={"receipt.json": {"canonical_request": canonical.model_dump(mode="json")}},
    )
    if terminal is not None:
        store.write_json(
            artifact_id="physical-result",
            artifact_type="provider-physical-attempt",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                inputs=(artifact_input(intent),),
                code_revision="fixture",
            ),
            files={"receipt.json": terminal},
        )
    report = inspect_memory(store, MemoryInspection(intents=(canonical,)))
    assert len(report.calls) == 1 and len(report.calls[0].attempts) == 1
    assert report.calls[0].attempts[0].status == expected
    assert report.calls[0].attempts[0].reasoning[0].availability == "incomplete"


def test_simulator_memory_is_not_actor_memory_and_locators_verify(tmp_path: Path) -> None:
    """Identical prompt text in different consumer calls cannot cross the assessment boundary."""
    import json

    from exp.common.core.artifacts import sha256_json
    from exp.common.project import artifact_input
    from exp.optimize.evaluation.analysis.memory_evidence import (
        MemoryInspection,
        inspect_memory,
        located_value,
    )
    from exp.runtime.models.call_evidence import call_consumer

    complete, _, _ = setup(tmp_path, AssistantAction(content="{}"))
    body = {
        "environment_state": {"status": "closed"},
        "memory_observations": [{"id": "observation"}],
        "research_history": {
            "prior_trajectories": [{"id": "rule", "claim": "closed cannot cancel"}]
        },
    }
    request = ModelRequest(messages=(ModelMessage(role="user", content=json.dumps(body)),))
    with call_consumer("simulator"):
        complete.complete_request(request, "prediction")
    complete.as_model_client("actor").complete(request)
    store = ProjectStore(tmp_path, "completion").artifacts
    refs = tuple(
        artifact_input(store.read(i).manifest)
        for i in store.list_ids(artifact_type="structured-call-intent")
    )
    report = inspect_memory(store, MemoryInspection(intents=refs))
    simulator = next(row for row in report.calls if row.consumer == "simulator")
    actor = next(row for row in report.calls if row.consumer == "actor")
    memory = next(
        f for f in simulator.fragments if f.channel == "research_history.prior_trajectories"
    )
    assert located_value(store, memory.location) == body["research_history"]["prior_trajectories"]
    assert all(not f.channel.startswith("research_history") for f in actor.fragments)
    assert all(row.coverage == "physical_join_unavailable" for row in report.calls)
    with pytest.raises(ValueError, match="digest differs"):
        located_value(store, memory.location.model_copy(update={"sha256": sha256_json("wrong")}))


@pytest.mark.parametrize(
    "body,expected",
    [
        (
            {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 12}},
            },
            {"usage_only"},
        ),
        (
            {
                "output": [
                    {
                        "type": "reasoning",
                        "encrypted_content": "opaque",
                        "summary": [{"type": "summary_text", "text": "summary"}],
                    }
                ]
            },
            {"opaque_carrier", "provider_summary"},
        ),
        ({"content": [{"type": "redacted_thinking", "data": "opaque"}]}, {"opaque_carrier"}),
        (
            {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]},
            {"incomplete"},
        ),
        ({"unknown_provider_field": "text"}, {"not_captured"}),
    ],
)
def test_reasoning_availability_never_invents_hidden_text(
    tmp_path: Path, body: dict, expected: set[str]
) -> None:
    """Opaque carriers, token counts and incomplete responses keep their actual authority."""
    from exp.common.core.artifacts import ArtifactEnvelope
    from exp.common.project import artifact_input
    from exp.optimize.evaluation.analysis.memory_evidence import _reasoning, located_value

    project = ProjectStore(tmp_path, "reasoning")
    from exp.common.project import ProjectConfig

    project.initialize(ProjectConfig(project_id="reasoning"))
    source = project.artifacts.write_json(
        artifact_id="raw",
        artifact_type="provider-physical-attempt",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), inputs=(), code_revision="fixture"
        ),
        files={"receipt.json": {"response_body": body}},
    )
    evidence = _reasoning(project.artifacts, artifact_input(source))
    assert {item.availability for item in evidence} == expected
    for item in evidence:
        for location in item.locations:
            value = located_value(project.artifacts, location)
            if item.availability == "provider_summary":
                assert value == "summary"
