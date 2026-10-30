"""Per-call reasoning views validate the override and keep every other catalog binding."""

import pytest

from exp.common.models import (
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ReasoningEffort,
)
from exp.common.models.catalog import GatewayDeploymentMetadata
from exp.common.models.token_cost_test import prices
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.reasoning_view import reasoning_catalog
from exp.runtime.models.registry import ModelConnectionError, RuntimeModelCatalog
from exp.runtime.models.registry_test import _DEFAULT_CAPABILITIES, _catalog


@pytest.mark.parametrize("effort, wire", [(None, "high"), ("none", "none"), ("low", "low")])
def test_reasoning_catalog_is_local_and_preserves_model_prices_and_actual_wire(
    effort: ReasoningEffort | None, wire: str
) -> None:
    """A per-call view reaches client defaults while keeping every unrelated catalog binding."""
    capabilities = _DEFAULT_CAPABILITIES.model_copy(
        update={"supports_reasoning": True, "reasoning_effort": "high"}
    )
    catalog = _catalog(
        provider="openai-compatible",
        base_url="https://fixture.invalid/v1",
        capabilities=capabilities,
    )
    record = catalog.models["fixture-model"].model_copy(
        update={
            "supported_reasoning_efforts": ("none", "low", "high"),
            "gateway": GatewayDeploymentMetadata(prices=prices()),
        }
    )
    catalog = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "fixture-model": record,
                "unrelated": record.model_copy(update={"model": "other"}),
            }
        }
    )
    original = catalog.model_dump(mode="json")
    snapshot = RuntimeModelCatalog(catalog, environment={}).snapshot("fixture-model")[0]
    adjusted = reasoning_catalog(catalog, "fixture-model", effort)
    assert (adjusted is catalog) == (effort is None)
    assert catalog.model_dump(mode="json") == original
    assert adjusted.models["unrelated"] is catalog.models["unrelated"]
    assert adjusted.connections is catalog.connections and adjusted.roles is catalog.roles
    assert adjusted.models["fixture-model"].token_prices == record.token_prices
    assert RuntimeModelCatalog(adjusted, environment={}).snapshot("fixture-model")[0] == snapshot
    transport = ScriptedJsonTransport(
        (
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "local-profile",
                    "model": "fixture-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
        )
    )
    resolved = RuntimeModelCatalog(
        adjusted,
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=lambda: transport,
    ).resolve("fixture-model")
    # No request-level override: this proves the local catalog client default itself changed.
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="one call"),), maximum_output_tokens=8
    )
    assert request.reasoning_effort is None
    resolved.client.complete(request)
    assert len(transport.requests) == 1
    payload = transport.requests[0].payload
    assert payload is not None and payload["reasoning_effort"] == wire
    assert resolved.snapshot == snapshot and resolved.token_prices == record.token_prices
    assert catalog.model_dump(mode="json") == original


def test_reasoning_catalog_no_override_or_already_off_needs_no_capability_invention() -> None:
    """A non-reasoning alias is already off and non-off requests fail before any client exists."""
    catalog = _catalog()
    assert reasoning_catalog(catalog, "fixture-model", None) is catalog
    assert reasoning_catalog(catalog, "fixture-model", "none") is catalog
    assert catalog.models["fixture-model"].capabilities == _DEFAULT_CAPABILITIES
    with pytest.raises(ValueError, match="reasoning effort"):
        reasoning_catalog(catalog, "fixture-model", "low")
    with pytest.raises(ModelConnectionError, match="unknown model alias"):
        reasoning_catalog(catalog, "absent", "none")


@pytest.mark.parametrize("supported", [None, (), ("low", "high")])
def test_reasoning_catalog_refuses_unadvertised_effort(
    supported: tuple[ReasoningEffort, ...] | None,
) -> None:
    """Reasoning capability alone does not authorize an undeclared disabled effort."""
    catalog = _catalog(
        capabilities=ModelCapabilities(supports_reasoning=True, reasoning_effort="high")
    )
    record = catalog.models["fixture-model"].model_copy(
        update={"supported_reasoning_efforts": supported}
    )
    catalog = catalog.model_copy(update={"models": {"fixture-model": record}})
    original = catalog.model_dump(mode="json")
    with pytest.raises(ValueError, match="reasoning effort 'none' is not supported"):
        reasoning_catalog(catalog, "fixture-model", "none")
    assert catalog.model_dump(mode="json") == original
