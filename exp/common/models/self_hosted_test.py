"""The self-hosted decisions exception: loopback only, explicit, revision-pinned, offline."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
)
from exp.common.models.self_hosted import is_loopback_url
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.providers.typesafe import TypeSafeClient
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS, DecisionsTraceTagger
from exp.simulation.mining.decisions_test import _answers, _bank, _unit

CLEF_REVISION = "cfddda043dd190ef3b01cff449d5ee8cff1e8ec6"


def clef_catalog(base_url: str = "http://127.0.0.1:30001/v1", **connection: object) -> ModelCatalog:
    """Return a catalog with Clef on a self-hosted SGLang decisions server."""
    return ModelCatalog.model_validate(
        {
            "connections": {
                "clef-local": {
                    "provider": "typesafe",
                    "base_url": base_url,
                    "self_hosted": True,
                    **connection,
                }
            },
            "models": {
                DEFAULT_DECISION_ALIAS: ModelRecord(
                    connection="clef-local",
                    model="Cloudflare/clef",
                    revision=CLEF_REVISION,
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(supports_completions=False),
                ).model_dump(mode="json")
            },
        }
    )


def test_the_catalog_accepts_a_self_hosted_loopback_clef_connection() -> None:
    """Clef over plain HTTP on loopback resolves with no credential and a pinned revision."""
    catalog = clef_catalog()
    snapshot, _ = RuntimeModelCatalog(catalog, environment={}).snapshot(DEFAULT_DECISION_ALIAS)
    assert (snapshot.provider, snapshot.model_id, snapshot.revision) == (
        "typesafe",
        "Cloudflare/clef",
        CLEF_REVISION,
    )
    resolved = RuntimeModelCatalog(catalog, environment={}).resolve(DEFAULT_DECISION_ALIAS)
    assert isinstance(resolved.client, TypeSafeClient)
    profile = resolved.client.gateway_wire_profile()
    assert profile.decisions_url == "http://127.0.0.1:30001/v1/systemone"
    assert "Authorization" not in profile.headers
    assert profile.organization_tag_key is None
    assert ModelCatalog.model_validate(catalog.model_dump(mode="json")) == catalog


def test_a_tunnel_port_never_changes_the_model_identity() -> None:
    """The loopback port is incidental; the pinned revision names the served weights."""
    first = RuntimeModelCatalog(clef_catalog(), environment={}).snapshot(DEFAULT_DECISION_ALIAS)
    moved = RuntimeModelCatalog(clef_catalog("http://localhost:30003/v1"), environment={}).snapshot(
        DEFAULT_DECISION_ALIAS
    )
    assert first == moved
    remote = ConnectionConfig(
        provider="typesafe",
        base_url="https://decisions.example.test/v1",
        api_key_env="KEY",
        trusted_custom_origin=True,
    )
    assert remote.identity_sha256() != first[0].connection_sha256


@pytest.mark.parametrize(
    ("connection", "message"),
    [
        ({"provider": "typesafe", "base_url": "http://127.0.0.1:30001/v1"}, "built-in official"),
        (
            {"provider": "typesafe", "base_url": "http://10.0.0.5:30001/v1", "self_hosted": True},
            "loopback",
        ),
        (
            {
                "provider": "typesafe",
                "base_url": "https://clef.example.test/v1",
                "self_hosted": True,
            },
            "loopback",
        ),
        ({"provider": "typesafe", "self_hosted": True}, "explicit loopback base_url"),
        (
            {
                "provider": "openai-compatible",
                "base_url": "http://127.0.0.1:30001/v1",
                "self_hosted": True,
            },
            "native decisions provider",
        ),
        (
            {
                "provider": "typesafe",
                "base_url": "http://127.0.0.1:30001/v1",
                "self_hosted": True,
                "trusted_custom_origin": True,
            },
            "cannot be combined",
        ),
        (
            {
                "provider": "typesafe",
                "base_url": "http://127.0.0.1:30001/v1",
                "trusted_custom_origin": True,
            },
            "https base_url",
        ),
    ],
)
def test_unsafe_decision_connections_are_refused(
    connection: dict[str, object], message: str
) -> None:
    """HTTP or a custom origin needs the explicit flag, and the flag needs loopback."""
    with pytest.raises(ValidationError, match=message):
        ConnectionConfig.model_validate(connection)


def test_a_self_hosted_model_must_pin_its_weights_revision() -> None:
    """Without a revision the snapshot could not tell two servings of different weights apart."""
    document = clef_catalog().model_dump(mode="json")
    document["models"][DEFAULT_DECISION_ALIAS]["revision"] = None
    with pytest.raises(ValidationError, match="pin the exact weights"):
        ModelCatalog.model_validate(document)


def test_the_tagger_reaches_the_loopback_server_through_the_catalog() -> None:
    """A scripted SGLang server answers a tagging call made through the catalog alias."""
    bank = _bank()
    transport = ScriptedJsonTransport([JsonHttpResponse(200, _answers(bank))])
    resolved = RuntimeModelCatalog(clef_catalog(), environment={}).resolve(DEFAULT_DECISION_ALIAS)
    tagger = DecisionsTraceTagger.for_alias(resolved, transport=transport)
    matrix = tagger.tag([_unit("trace-1")], bank)
    assert len(matrix.rows) == 1
    (request,) = transport.requests
    assert request.url == "http://127.0.0.1:30001/v1/systemone"
    assert request.payload["model"] == "Cloudflare/clef"
    assert "Authorization" not in request.headers


def test_loopback_detection() -> None:
    """Only localhost and loopback literals count."""
    assert is_loopback_url("http://localhost:1/v1")
    assert is_loopback_url("http://127.4.0.1/v1")
    assert is_loopback_url("http://[::1]:8000/v1")
    assert not is_loopback_url("http://localhost.example.test/v1")
    assert not is_loopback_url("http://192.168.1.4/v1")
