"""Physical receipts survive unusable output while credentials stay outside artifacts."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ConnectionConfig, ModelCatalog, ModelRecord
from exp.common.models.client import RequestAdmissionRefused
from exp.common.project import ArtifactStoreError, ProjectConfig, ProjectStore
from exp.runtime.models.providers.admitted_transport import (
    ADMITTED_TRANSPORT_PROVIDERS,
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.attempt_timeouts import ATTEMPT_TIMEOUT_PROTOCOL
from exp.runtime.models.providers.backpressure import (
    AdmissionLimits,
    HostProviderAdmission,
    ProviderAdmissionRefused,
    ProviderRoute,
)
from exp.runtime.models.providers.openai_compatible_test import _request
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import SUPPORTED_PROVIDERS, RuntimeModelCatalog
from exp.runtime.models.registry_test import _catalog as _registry_catalog
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError, DispatchLedger


def test_raw_response_is_retained_before_model_parsing_and_cap_blocks_second_send(
    tmp_path: Path,
) -> None:
    """Even an empty assistant response is saved exactly, without authenticated headers."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    body: JsonObject = {"choices": [{"message": {"content": None}}]}
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body)])
    ledger = DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    headers = {"Authorization": "Bearer never-retain-this"}
    payload: JsonObject = {"model": "fixture"}
    assert (
        transport.post(
            "https://fixture.invalid",
            headers=headers,
            payload=payload,
            timeout_seconds=30,
        ).body
        == body
    )
    with pytest.raises(DispatchCapExhaustedError):
        transport.post(
            "https://fixture.invalid",
            headers=headers,
            payload=payload,
            timeout_seconds=30,
        )
    assert ledger.used() == 1
    artifacts = project.artifacts.list_ids()
    assert len(artifacts) == 2
    response = next(identity for identity in artifacts if identity.endswith("-response"))
    receipt = project.artifacts.read_bytes(response, "receipt.json").decode()
    assert '"content":null' in receipt
    assert "never-retain-this" not in payload and "Authorization" not in receipt


def test_credential_response_is_rejected_with_a_hash_only_receipt(tmp_path: Path) -> None:
    """Receipt recording does not bypass the shared provider-output credential boundary."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    inner = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=200, body={"api_key": "do-not-retain"}),
        ]
    )
    transport = AdmittedJsonTransport(
        inner,
        ledger=DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64),
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    with pytest.raises(ArtifactStoreError, match="secret boundary"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    ids = project.artifacts.list_ids()
    assert len(ids) == 2 and any(identity.endswith("-rejected") for identity in ids)
    assert all(
        b"do-not-retain" not in project.artifacts.read_bytes(identity, "receipt.json")
        for identity in ids
    )


def test_intent_persistence_cannot_extend_dispatch_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow durable intent may consume a slot, but must not start late network I/O."""
    from collections.abc import Mapping

    from exp.common.core.artifacts import ArtifactEnvelope, JsonValue
    from exp.common.project import ArtifactManifest

    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    now = [datetime(2030, 1, 1, tzinfo=UTC)]
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    ledger = DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64)
    original = project.artifacts.write_json

    def slow_intent(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, JsonValue],
    ) -> ArtifactManifest:
        """Advance the clock after the intent was saved, before the transport can send."""
        result = original(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=files,
        )
        if artifact_id.endswith("-intent"):
            now[0] += timedelta(seconds=2)
        return result

    monkeypatch.setattr(project.artifacts, "write_json", slow_intent)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=now[0] + timedelta(seconds=1),
        clock=lambda: now[0],
    )
    with pytest.raises(TimeoutError, match="persisting intent"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert ledger.used() == 1 and not inner.requests
    assert any(name.endswith("-not-dispatched") for name in project.artifacts.list_ids())


def test_a_refusing_guard_runs_before_every_claim_and_stops_the_send(tmp_path: Path) -> None:
    """The guard sees each claim first; once it refuses, no slot is claimed and nothing is sent."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    body: JsonObject = {"choices": []}
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body)] * 2)
    ledger = DispatchLedger(tmp_path / "physical.sqlite", 5, "a" * 64)
    allowed = [True]
    checks: list[int] = []

    def guard() -> None:
        """Record the check and refuse once dispatch is no longer allowed."""
        checks.append(ledger.used())
        if not allowed[0]:
            raise RuntimeError("reservation released")

    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        before_claim=guard,
    )
    transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    allowed[0] = False
    with pytest.raises(RuntimeError, match="reservation released"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert checks == [0, 1] and ledger.used() == 1 and len(inner.requests) == 1


def test_final_transport_body_is_saved_without_headers_or_endpoint_secrets(tmp_path: Path) -> None:
    """The receipt describes the actual provider-protocol input, not a reconstructed prompt."""
    import json

    from exp.common.core.artifacts import sha256_json
    from exp.runtime.models.call_evidence import call_consumer

    project = ProjectStore(tmp_path, "attempt")
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    transport = AdmittedJsonTransport(
        inner,
        ledger=DispatchLedger(tmp_path / "calls.sqlite", 1, "b" * 64),
        artifacts=project.artifacts,
        code_revision="test",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    payload: JsonObject = {
        "messages": [{"role": "user", "content": "post-hook input"}],
        "model": "x",
    }
    with call_consumer("actor"):
        transport.post(
            "https://fixture.invalid/?key=inert-url-secret",
            headers={"Authorization": "inert-header-secret"},
            payload=payload,
            timeout_seconds=5,
        )
    identity = next(x for x in project.artifacts.list_ids() if x.endswith("-intent"))
    encoded = project.artifacts.read_bytes(identity, "receipt.json")
    saved = json.loads(encoded)
    assert saved["request_body"] == inner.requests[0].payload == payload
    assert saved["request_sha256"] == sha256_json(saved["request_body"])
    assert saved["consumer"] == "actor"
    assert b"inert-url-secret" not in encoded and b"inert-header-secret" not in encoded


def test_credential_bearing_request_refuses_before_claim_and_io(tmp_path: Path) -> None:
    """A body that cannot be safely retained cannot be silently dispatched with hash-only proof."""
    project = ProjectStore(tmp_path, "attempt")
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    ledger = DispatchLedger(tmp_path / "calls.sqlite", 1, "b" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="test",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    with pytest.raises(ArtifactStoreError, match="request"):
        transport.post(
            "https://fixture.invalid",
            headers={},
            payload={"messages": [{"role": "user", "content": '{"api_key": "inert-body-secret"}'}]},
            timeout_seconds=5,
        )
    assert not inner.requests and ledger.used() == 0
    assert not project.artifacts.list_ids()


@pytest.mark.parametrize(
    ("url", "payload", "texts"),
    [
        (
            "https://fixture.invalid/v1/embeddings",
            {"model": "embed", "input": ["Authorization: token ${GITHUB_TOKEN}", "ls -la"]},
            ["Authorization: token ${GITHUB_TOKEN}", "ls -la"],
        ),
        (
            "https://fixture.invalid/v1beta/models/e:batchEmbedContents",
            {"requests": [{"content": {"parts": [{"text": "'password': u.password"}]}}]},
            ["'password': u.password"],
        ),
    ],
)
def test_embedding_text_is_retained_by_digest_so_quoted_credentials_still_dispatch(
    tmp_path: Path, url: str, payload: JsonObject, texts: list[str]
) -> None:
    """Embedding inputs quote trace evidence; the receipt cites them, the wire gets them whole."""
    from exp.common.core.artifacts import sha256_bytes, sha256_json

    project = ProjectStore(tmp_path, "attempt")
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    transport = AdmittedJsonTransport(
        inner,
        ledger=DispatchLedger(tmp_path / "calls.sqlite", 1, "b" * 64),
        artifacts=project.artifacts,
        code_revision="test",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    transport.post(url, headers={}, payload=payload, timeout_seconds=5)
    assert inner.requests[0].payload == payload
    identity = next(x for x in project.artifacts.list_ids() if x.endswith("-intent"))
    encoded = project.artifacts.read_bytes(identity, "receipt.json")
    saved = json.loads(encoded)
    assert saved["request_sha256"] == sha256_json(payload)
    assert all(text.encode() not in encoded for text in texts)
    cited = json.dumps(saved["request_body"])
    for text in texts:
        assert sha256_bytes(text.encode("utf-8")) in cited


def test_embedding_route_still_refuses_credentials_outside_its_text_fields(
    tmp_path: Path,
) -> None:
    """Only embedded text is cited; a credential in any other embedding field still refuses."""
    project = ProjectStore(tmp_path, "attempt")
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    ledger = DispatchLedger(tmp_path / "calls.sqlite", 1, "b" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="test",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    with pytest.raises(ArtifactStoreError, match="request"):
        transport.post(
            "https://fixture.invalid/v1/embeddings",
            headers={},
            payload={"input": ["ls"], "user": '{"api_key": "inert-body-secret"}'},
            timeout_seconds=5,
        )
    assert not inner.requests and ledger.used() == 0


def test_late_paid_response_is_retained_but_cannot_authorize_an_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late response is evidence, not permission to continue mutating the world."""
    import json
    from collections.abc import Mapping

    now = [datetime(2030, 1, 1, tzinfo=UTC)]
    body: JsonObject = {"choices": [{"message": {"content": "perform the write"}}]}
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body)])
    original = inner.post

    def late(
        url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Return after the declared boundary despite the requested transport timeout."""
        now[0] += timedelta(seconds=2)
        return original(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(inner, "post", late)
    project = ProjectStore(tmp_path, "attempt")
    ledger = DispatchLedger(tmp_path / "calls.sqlite", 1, "c" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="test",
        deadline=now[0] + timedelta(seconds=1),
        clock=lambda: now[0],
    )
    with pytest.raises(TimeoutError, match="response.*deadline"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert len(inner.requests) == ledger.used() == 1
    identity = next(x for x in project.artifacts.list_ids() if x.endswith("-response"))
    saved = json.loads(project.artifacts.read_bytes(identity, "receipt.json"))
    assert saved["response_body"] == body and saved["returned_after_deadline"] is True
    with pytest.raises(TimeoutError, match="before physical dispatch"):
        transport.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=30)
    assert len(inner.requests) == ledger.used() == 1


def test_each_attempt_intent_records_its_timeout_and_the_derivation_protocol(
    tmp_path: Path,
) -> None:
    """A saved attempt names the bound it ran under and which registered derivation produced it."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={"ok": True})])
    transport = AdmittedJsonTransport(
        inner,
        ledger=DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64),
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=10),
    )
    transport.post("https://fixture.invalid", headers={}, payload={"m": 1}, timeout_seconds=202.4)
    intent = next(i for i in project.artifacts.list_ids() if i.endswith("-intent"))
    receipt = json.loads(project.artifacts.read_bytes(intent, "receipt.json"))
    assert receipt["timeout_seconds"] == pytest.approx(202.4)
    assert receipt["timeout_protocol"] == ATTEMPT_TIMEOUT_PROTOCOL


def _one_alias_catalog(provider: str) -> ModelCatalog:
    """Build an unvalidated one-alias catalog; only the connection provider is read."""
    return ModelCatalog.model_construct(
        connections={"primary": ConnectionConfig.model_construct(provider=provider)},
        models={"fixture": ModelRecord.model_construct(connection="primary", model="fixture")},
    )


def test_every_registry_provider_is_classified_for_admission() -> None:
    """A new registry provider fails here until its transport path is classified."""
    assert ADMITTED_TRANSPORT_PROVIDERS <= SUPPORTED_PROVIDERS
    assert SUPPORTED_PROVIDERS - ADMITTED_TRANSPORT_PROVIDERS == {
        "bedrock",
        "elevenlabs",
        "tinker",
        "typesafe",
        "vertex",
    }


@pytest.mark.parametrize("provider", sorted(ADMITTED_TRANSPORT_PROVIDERS))
def test_admitted_providers_pass(provider: str) -> None:
    """Every provider built on the injected JSON transport is accepted."""
    require_admitted_transport(_one_alias_catalog(provider), "fixture", consumer="critic")


@pytest.mark.parametrize("provider", sorted(SUPPORTED_PROVIDERS - ADMITTED_TRANSPORT_PROVIDERS))
def test_unadmitted_providers_refuse_with_the_remedy(provider: str) -> None:
    """The refusal names the alias and provider, and lists the providers to switch to."""
    with pytest.raises(ValueError) as refused:
        require_admitted_transport(_one_alias_catalog(provider), "fixture", consumer="critic")
    message = str(refused.value)
    assert f"critic alias 'fixture' uses provider {provider!r}" in message
    assert "anthropic, azure, gemini, openai, openai-compatible, openrouter" in message


def _admitted(
    tmp_path: Path, inner: ScriptedJsonTransport, admission: HostProviderAdmission, cap: int = 10
) -> tuple[AdmittedJsonTransport, DispatchLedger, ProjectStore]:
    """Build an admitted transport over a scripted wire and an explicit host admission."""
    project = ProjectStore(tmp_path, "attempt")
    project.initialize(ProjectConfig(project_id="attempt"))
    ledger = DispatchLedger(tmp_path / "physical.sqlite", cap, "a" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=10),
        admission=admission,
    )
    return transport, ledger, project


def test_open_breaker_refuses_before_the_ledger_claim_and_any_io(tmp_path: Path) -> None:
    """Overload responses open the provider breaker; the next send is refused, not dispatched."""
    admission = HostProviderAdmission(
        tmp_path / "admission.sqlite", AdmissionLimits(breaker_threshold=2)
    )
    inner = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=status, body={}) for status in (429, 503)]
    )
    transport, ledger, _ = _admitted(tmp_path, inner, admission)
    route = transport.for_route(ProviderRoute(provider="openrouter", alias="writer"))
    for _ in range(2):
        route.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=5)
    assert admission.breaker_state("openrouter") == "open"
    with pytest.raises(ProviderAdmissionRefused) as refused:
        route.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=5)
    assert isinstance(refused.value, RequestAdmissionRefused)
    assert refused.value.reason == "circuit_open"
    assert ledger.used() == 2 and len(inner.requests) == 2
    assert admission.in_flight(ProviderRoute("openrouter", "writer")) == (0, 0)
    # An unbound transport is scoped by endpoint host, not by the bound provider's breaker.
    assert admission.breaker_state("endpoint:fixture.invalid") == "closed"


def test_a_slot_is_held_only_while_the_request_is_in_flight(tmp_path: Path) -> None:
    """The scripted wire observes its own slot; none remains after success or failure."""
    admission = HostProviderAdmission(tmp_path / "admission.sqlite")
    route = ProviderRoute(provider="openai", alias="writer")
    seen: list[tuple[int, int]] = []

    class _Observing(ScriptedJsonTransport):
        """Record the live slot counts at send time."""

        def post(
            self,
            url: str,
            *,
            headers: Mapping[str, str],
            payload: JsonObject,
            timeout_seconds: float,
        ) -> JsonHttpResponse:
            """Record in-flight counts, then answer from the script."""
            seen.append(admission.in_flight(route))
            return super().post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )

    inner = _Observing(
        [JsonHttpResponse(status_code=200, body={}), ProviderTransportError("connection reset")]
    )
    transport, _, _ = _admitted(tmp_path, inner, admission)
    bound = transport.for_route(route)
    bound.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=5)
    with pytest.raises(ProviderTransportError):
        bound.post("https://fixture.invalid", headers={}, payload={}, timeout_seconds=5)
    assert seen == [(1, 1), (1, 1)]
    assert admission.in_flight(route) == (0, 0)


def test_the_intent_receipt_records_the_idempotency_key_without_other_headers(
    tmp_path: Path,
) -> None:
    """The key a resend must reuse is evidence; the credential header is still never kept."""
    admission = HostProviderAdmission(tmp_path / "admission.sqlite")
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    transport, _, project = _admitted(tmp_path, inner, admission)
    transport.post(
        "https://fixture.invalid",
        headers={"Authorization": "Bearer never-retain", "Idempotency-Key": "exp-intent-abc"},
        payload={},
        timeout_seconds=5,
    )
    intent = next(i for i in project.artifacts.list_ids() if i.endswith("-intent"))
    raw = project.artifacts.read_bytes(intent, "receipt.json")
    assert json.loads(raw)["idempotency_key"] == "exp-intent-abc"
    assert b"never-retain" not in raw


def test_registry_scopes_each_client_by_provider_and_alias(tmp_path: Path) -> None:
    """Through the registry, overload opens the alias's provider breaker mid-retry-loop."""
    admission = HostProviderAdmission(
        tmp_path / "admission.sqlite", AdmissionLimits(breaker_threshold=2)
    )
    inner = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=429, body={"error": {"message": "slow down"}})] * 2
    )
    transport, ledger, _ = _admitted(tmp_path, inner, admission)
    registry = RuntimeModelCatalog(
        _registry_catalog(provider="openrouter"),
        environment={"FIXTURE_API_KEY": "fixture"},
        transport_factory=lambda: transport,
    )
    client = registry.resolve("fixture-model").client
    with pytest.raises(ProviderAdmissionRefused, match="openrouter circuit breaker"):
        client.complete(_request())
    assert len(inner.requests) == 2 and ledger.used() == 2
    assert admission.breaker_state("openrouter") == "open"
    with pytest.raises(ProviderAdmissionRefused):
        client.complete(_request())
    assert len(inner.requests) == 2 and ledger.used() == 2
