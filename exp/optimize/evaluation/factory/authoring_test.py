"""Shared author calls retain exact requests and finite dispatch authority."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import ModelRequest, ReasoningEffort, load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.authoring import author_response, preflight_author
from exp.optimize.evaluation.factory.proposals import (
    TaskProposalSpec,
    proposal_request,
    propose_tasks,
)
from exp.optimize.evaluation.factory.proposals_test import proposal_fixture
from exp.runtime.models.providers.errors import ProviderRetryableResponseError
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def test_saved_author_response_cannot_be_rebound_to_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing an identity with changed context refuses instead of relabeling a saved draw."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    propose_tasks(project, spec, code_revision="test")
    request = proposal_request(project, spec)
    assert request.maximum_output_tokens is not None
    changed = request.model_copy(
        update={"maximum_output_tokens": request.maximum_output_tokens + 1}
    )
    with pytest.raises(ValueError, match="original request"):
        author_response(
            project,
            spec,
            changed,
            identity=spec.identity("test"),
            sources=(spec.base, spec.brief),
            code_revision="test",
        )
    assert len(transport.requests) == 1


def test_expired_author_has_no_intent_or_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deadlines block new dispatch but do not consume a draw or falsely create ambiguity."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"deadline": spec.deadline - timedelta(days=1)})
    with pytest.raises(ValueError, match="deadline expired"):
        propose_tasks(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")


def _reasoning_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reasoning: bool = True,
    supported: tuple[ReasoningEffort, ...] | None = ("none", "low", "high"),
) -> tuple[ProjectStore, TaskProposalSpec, ScriptedJsonTransport]:
    """Pin a high-reasoning catalog while retaining the exact original provider snapshot."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models[spec.model_alias]
    assert record.capabilities is not None
    capabilities = record.capabilities.model_copy(
        update={"supports_reasoning": reasoning, "reasoning_effort": "high" if reasoning else None}
    )
    catalog = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                spec.model_alias: record.model_copy(
                    update={
                        "capabilities": capabilities,
                        "supported_reasoning_efforts": supported if reasoning else None,
                    }
                ),
            }
        }
    )
    write_model_catalog(project.model_catalog_path, catalog)
    model, _ = RuntimeModelCatalog(catalog, environment={}).snapshot(spec.model_alias)
    assert model == spec.model
    return (
        project,
        spec.model_copy(
            update={
                "catalog_sha256": sha256_json(catalog.model_dump(mode="json")),
            }
        ),
        transport,
    )


@pytest.mark.parametrize("effort, wire", [(None, "high"), ("none", "none"), ("low", "low")])
def test_author_effort_is_scoped_to_the_draw_and_replays_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: ReasoningEffort | None, wire: str
) -> None:
    """Per-call reasoning changes the actual wire and frozen receipt, never the source catalog."""
    project, original, transport = _reasoning_fixture(tmp_path, monkeypatch)
    spec = original.model_copy(update={"reasoning_effort": effort})
    assert "reasoning_effort" not in original.model_dump(mode="json")
    assert (spec.identity("test") == original.identity("test")) == (effort is None)
    before = project.model_catalog_path.read_bytes()
    logical = proposal_request(project, spec)
    report = propose_tasks(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    payload = transport.requests[0].payload
    assert payload is not None and payload["reasoning_effort"] == wire
    assert project.model_catalog_path.read_bytes() == before
    intent = spec.identity("test") + "-intent"
    assert (
        ModelRequest.model_validate_json(project.artifacts.read_bytes(intent, "request.json"))
        == logical
    )
    files = {row.path for row in project.artifacts.read(intent).manifest.files}
    if effort is None:
        assert files == {"spec.json", "request.json"}
    else:
        effective = ModelRequest.model_validate_json(
            project.artifacts.read_bytes(intent, "effective-request.json")
        )
        assert effective.reasoning_effort == effort
        profile = json.loads(project.artifacts.read_bytes(intent, "reasoning-profile.json"))
        assert profile["effective_effort"] == effort and profile["supports_reasoning"] is True
        assert profile["logical_request_sha256"] == sha256_json(logical)
        assert profile["effective_request_sha256"] == sha256_json(effective)
    project.model_catalog_path.unlink()
    assert propose_tasks(project, spec, code_revision="test") == report
    response, _ = author_response(
        project,
        spec,
        logical,
        identity=spec.identity("test"),
        sources=(spec.base, spec.brief),
        code_revision="test",
    )
    assert response == report.response and len(transport.requests) == 1
    different = spec.model_copy(update={"reasoning_effort": "low" if effort != "low" else "none"})
    with pytest.raises(ValueError, match="original request"):
        author_response(
            project,
            different,
            logical,
            identity=spec.identity("test"),
            sources=(spec.base, spec.brief),
            code_revision="test",
        )
    assert len(transport.requests) == 1


def test_author_none_on_nonreasoning_model_is_already_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An off policy neither invents reasoning support nor sends an unsupported field."""
    project, original, transport = _reasoning_fixture(tmp_path, monkeypatch, reasoning=False)
    spec = original.model_copy(update={"reasoning_effort": "none"})
    before = project.model_catalog_path.read_bytes()
    propose_tasks(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    payload = transport.requests[0].payload
    assert payload is not None and "reasoning_effort" not in payload and "reasoning" not in payload
    profile = json.loads(
        project.artifacts.read_bytes(spec.identity("test") + "-intent", "reasoning-profile.json")
    )
    assert profile["requested_effort"] == "none"
    assert profile["effective_effort"] is None and profile["supports_reasoning"] is False
    assert project.model_catalog_path.read_bytes() == before


@pytest.mark.parametrize(
    "reasoning, supported, effort",
    [(False, None, "low"), (True, ("low", "high"), "none"), (True, None, "low")],
)
def test_author_unsupported_effort_refuses_before_intent_or_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reasoning: bool,
    supported: tuple[ReasoningEffort, ...] | None,
    effort: ReasoningEffort,
) -> None:
    """Capability and deployment support must both authorize an explicit reasoning override."""
    project, original, transport = _reasoning_fixture(
        tmp_path, monkeypatch, reasoning=reasoning, supported=supported
    )
    spec = original.model_copy(update={"reasoning_effort": effort})
    request = proposal_request(project, spec)
    with pytest.raises(ValueError, match="reasoning effort"):
        preflight_author(project, spec, request)
    with pytest.raises(ValueError, match="reasoning effort"):
        propose_tasks(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")


def test_author_reasoning_profile_never_redraws_known_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed semantic output remains the original draw under an explicit off profile."""
    project, original, transport = proposal_fixture(tmp_path, monkeypatch, duplicate=True)
    spec = original.model_copy(update={"reasoning_effort": "none"})
    for _ in range(2):
        with pytest.raises(ValueError, match="distinct wording"):
            propose_tasks(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")


def test_author_reasoning_profile_never_redraws_unknown_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost response retains its explicit profile and refuses even without the catalog."""
    project, original, _ = proposal_fixture(tmp_path, monkeypatch)
    spec = original.model_copy(update={"reasoning_effort": "none"})
    transport = ScriptedJsonTransport(())

    def lose_response() -> JsonHttpResponse:
        """Simulate loss after the physical request was admitted."""
        raise RuntimeError("lost author response")

    monkeypatch.setattr(transport, "_answer", lose_response)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    with pytest.raises(RuntimeError, match="lost author response"):
        propose_tasks(project, spec, code_revision="test")
    count = len(transport.requests)
    project.model_catalog_path.unlink()
    with pytest.raises(ValueError, match="unknown outcome"):
        propose_tasks(project, spec, code_revision="test")
    assert len(transport.requests) == count == 1


@pytest.mark.parametrize("effort", [None, "none"])
def test_completed_author_output_exhaustion_is_one_known_draw_and_replays_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: ReasoningEffort | None
) -> None:
    """A paid reasoning-only length result retains its actual body and never becomes unknown."""
    project, original, _ = _reasoning_fixture(tmp_path, monkeypatch)
    spec = original.model_copy(update={"reasoning_effort": effort})
    catalog_bytes = project.model_catalog_path.read_bytes()
    body: JsonObject = {
        "id": "reasoning-only-length",
        "model": "world-v1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": None},
            }
        ],
        "usage": {
            "prompt_tokens": 32,
            "completion_tokens": 4096,
            "completion_tokens_details": {"reasoning_tokens": 4096},
        },
    }
    transport = ScriptedJsonTransport((JsonHttpResponse(status_code=200, body=body),))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    with pytest.raises(ProviderRetryableResponseError, match="output allowance") as first:
        propose_tasks(project, spec, code_revision="test")
    assert first.value.hit_length_limit is True and first.value.response_body == body
    assert len(transport.requests) == 1
    payload = transport.requests[0].payload
    assert payload is not None and payload["max_tokens"] == 4096
    assert payload["reasoning_effort"] == (effort or "high")
    assert project.model_catalog_path.read_bytes() == catalog_bytes
    identity = spec.identity("test")
    intent = artifact_input(project.artifacts.read(identity + "-intent").manifest)
    rejected = project.artifacts.read(identity + "-rejection").manifest
    assert rejected.artifact_type == "author-rejection" and rejected.inputs == (intent,)
    rejection = json.loads(project.artifacts.read_bytes(rejected.artifact_id, "rejection.json"))
    assert rejection["reason"] == "output_allowance_exhausted"
    assert rejection["output_tokens"] == 4096
    assert rejection["response_body"] == body and rejection["response_sha256"] == sha256_json(body)
    assert not project.artifacts.exists(identity + "-response")
    # Physical transport receipts preserve the actual returned body and its author intent ancestry.
    physical = [
        project.artifacts.read(artifact_id).manifest
        for artifact_id in project.artifacts.list_ids(artifact_type="provider-physical-attempt")
        if artifact_id.endswith("-response")
    ]
    owned = [
        row
        for row in physical
        if (
            len(row.inputs) == 1
            and project.artifacts.read(row.inputs[0].artifact_id).manifest.inputs == (intent,)
        )
    ]
    assert len(owned) == 1
    actual = json.loads(project.artifacts.read_bytes(owned[0].artifact_id, "receipt.json"))
    assert actual["http_status"] == 200 and actual["response_body"] == body
    # A caller may mutate its error body without changing immutable evidence or cached replay.
    assert first.value.response_body is not None
    first.value.response_body["id"] = "caller-only-mutation"
    before = project.artifacts.list_ids()
    catalog = load_model_catalog(project.model_catalog_path)
    for connection in catalog.connections.values():
        if connection.api_key_env is not None:
            monkeypatch.delenv(connection.api_key_env, raising=False)
    project.model_catalog_path.unlink()

    def no_transport() -> ScriptedJsonTransport:
        """Forbid new transport construction on known terminal replay."""
        raise AssertionError("known author rejection must not construct a provider transport")

    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", no_transport
    )
    with pytest.raises(ProviderRetryableResponseError, match="output allowance") as replay:
        propose_tasks(project, spec, code_revision="test")
    assert replay.value.hit_length_limit is True and replay.value.response_body == body
    assert project.artifacts.list_ids() == before and len(transport.requests) == 1
    # The known failure cannot be rebound to different settings or logical request contents.
    changed = spec.model_copy(update={"reasoning_effort": "low"})
    logical = proposal_request(project, spec)
    with pytest.raises(ValueError, match="original request"):
        author_response(
            project,
            changed,
            logical,
            identity=identity,
            sources=(spec.base, spec.brief),
            code_revision="test",
        )
    with pytest.raises(ValueError, match="original request"):
        author_response(
            project,
            spec,
            logical.model_copy(update={"maximum_output_tokens": 4095}),
            identity=identity,
            sources=(spec.base, spec.brief),
            code_revision="test",
        )
    assert len(transport.requests) == 1
