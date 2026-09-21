"""Local tests for the decision-only TypeSafe provider wire."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    BillingSource,
    EmbeddingClient,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
)
from exp.runtime.models.providers.base import ProviderHttpClient
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.providers.typesafe import TYPESAFE_BASE_URL, TypeSafeClient


def _model() -> ModelSnapshot:
    """Return a provider identity with no credential-bearing fields."""
    return ModelSnapshot(
        provider="typesafe",
        model_id="systemone",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256=sha256_json({}),
        connection_sha256=sha256_json({"provider": "typesafe"}),
    )


def test_typesafe_exposes_only_native_systemone_wire() -> None:
    """Constructing the native wire never dispatches or impersonates chat/embeddings."""
    client = TypeSafeClient(
        model=_model(), api_key="provider-secret-canary", transport=ScriptedJsonTransport()
    )
    profile = client.gateway_wire_profile()
    assert profile.dialect == "typesafe_systemone"
    assert profile.url == profile.decisions_url == f"{TYPESAFE_BASE_URL}/systemone"
    assert profile.headers == {
        "Authorization": "Bearer provider-secret-canary",
        "Content-Type": "application/json",
    }
    assert profile.model_id == "systemone"
    assert profile.billing_customer_managed is True
    assert profile.embeddings_url is None
    assert profile.images_url is None
    assert profile.supports_temperature is False
    assert profile.supports_top_p is False
    assert isinstance(client, ModelClient)
    assert not isinstance(client, EmbeddingClient)
    assert not isinstance(client, ProviderHttpClient)
    assert "provider-secret-canary" not in repr(client)
    assert "provider-secret-canary" not in repr(profile)


def test_typesafe_completion_fails_locally_without_transport() -> None:
    """The ordinary ModelClient seam rejects chat instead of issuing an HTTP request."""
    client = TypeSafeClient(model=_model(), api_key="provider-secret-canary")
    with pytest.raises(ProviderCapabilityError) as raised:
        client.complete(ModelRequest(messages=(ModelMessage(role="user", content="state-canary"),)))
    assert raised.value.capability == "completions"
    assert "state-canary" not in str(raised.value)
    assert "provider-secret-canary" not in str(raised.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://api.typesafe.ai/v1",
        "https://key:secret@example.test/v1",
        "https://example.test/v1?key=secret",
        "https://example.test/v1#secret",
        "/v1",
    ],
)
def test_typesafe_rejects_unsafe_endpoint_shapes(url: str) -> None:
    """Invalid direct construction cannot send a bearer key over an unsafe URL."""
    with pytest.raises(ValueError, match="HTTPS base URL"):
        TypeSafeClient(model=_model(), api_key="provider-secret-canary", base_url=url)


def test_typesafe_refuses_an_empty_api_key() -> None:
    """A missing released credential fails before a wire profile can be obtained."""
    with pytest.raises(ValueError, match="non-empty API key"):
        TypeSafeClient(model=_model(), api_key="")
