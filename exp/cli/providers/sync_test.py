"""Hosted login synchronization keeps account identities and catalog metadata together."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.cli.providers.declaration import eligible_for_role
from exp.cli.providers.experiential_cloud import hosted_connection
from exp.cli.providers.model_picker import configured_models
from exp.cli.providers.sync import sync_account_models
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    SetupRole,
    load_model_catalog,
    serves_role,
    write_model_catalog,
)
from exp.runtime.models.providers import HttpProviderModelLister
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport


@pytest.mark.parametrize("structured_output", [False, True])
def test_login_refreshes_unknown_models_from_the_cloud_catalog_without_new_aliases(
    tmp_path: Path,
    structured_output: bool,
) -> None:
    """A login repairs cached identity-only metadata without calls to inference endpoints."""
    connection = hosted_connection({})
    write_model_catalog(
        tmp_path / "models.toml",
        ModelCatalog(
            connections={connection.name: connection.catalog_config()},
            models={
                "my-model": ModelRecord(
                    connection=connection.name,
                    model="cloud-chat",
                    billing_source=BillingSource.HOST_MANAGED,
                    capabilities=ModelCapabilities(supports_completions=False),
                )
            },
        ),
    )
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=200, body={"data": [{"id": "cloud-chat"}]}),
            JsonHttpResponse(
                status_code=200,
                body={
                    "models": [
                        {
                            "model": {"slug": "cloud-chat", "output_modalities": ["text"]},
                            "providers": [
                                {
                                    "id": "primary",
                                    "status": "active",
                                    "input_nano_usd_per_million": 200000000,
                                    "output_nano_usd_per_million": 1200000000,
                                    "capabilities": {
                                        "supports_structured_output": structured_output,
                                        "supports_reasoning": True,
                                        "reasoning_default_effort": "high",
                                        "supported_reasoning_efforts": ["low", "high", "max"],
                                    },
                                }
                            ],
                            "default_provider_ids": ["primary"],
                        }
                    ],
                    "total": 1,
                    "offset": 0,
                },
            ),
        ]
    )

    aliases = sync_account_models(
        tmp_path,
        connection=connection,
        api_key="secret-key",
        console=ScriptedConsole(""),
        lister=HttpProviderModelLister(transport=transport),
    )

    assert aliases == ("my-model",)
    saved = load_model_catalog(tmp_path / "models.toml")
    capabilities = saved.models["my-model"].capabilities
    assert capabilities is not None
    assert serves_role(capabilities, SetupRole.JUDGE) is structured_output
    assert saved.models["my-model"].discovery is not None
    assert saved.models["my-model"].discovery.supports_structured_output is structured_output
    available = configured_models(
        saved.models, connection_providers={connection.name: connection.provider}
    )
    assert eligible_for_role(available[0], SetupRole.JUDGE) is structured_output
    assert capabilities.input_cost_per_million_tokens_usd == 0.2
    assert capabilities.output_cost_per_million_tokens_usd == 1.2
    assert saved.models["my-model"].supported_reasoning_efforts == ("low", "high", "max")
    assert capabilities.reasoning_effort == "high"
    assert "secret-key" not in (tmp_path / "models.toml").read_text()
    assert len(transport.requests) == 2
    assert all(not request.payload for request in transport.requests)
