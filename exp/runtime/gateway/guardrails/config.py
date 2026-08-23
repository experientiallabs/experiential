"""Load an optional identity-scoped guardrail engine from the gateway root."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import httpx
from pydantic import Field, ValidationError

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject
from exp.runtime.gateway.guardrails.classifiers import (
    ClassifierRegistry,
    KeywordClassifier,
)
from exp.runtime.gateway.guardrails.client import DirectClassifierClient, InspectingClassifier
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.http_json import (
    DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES,
    HttpJsonClassifier,
    validate_bearer_env_name,
    validate_classifier_url,
)
from exp.runtime.gateway.guardrails.preset import policy_from_authored
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore

_CONFIG_NAME = "guardrails.json"
_FORBIDDEN_ADAPTER_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "bearer",
        "credential",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)


class KeywordAdapterDocument(ContractModel):
    """Local coarse keyword adapter declared in ``guardrails.json``."""

    adapter_id: ArtifactId
    kind: str
    needles: tuple[str, ...]


class HttpJsonAdapterDocument(ContractModel):
    """Dedicated HTTP JSON adapter declared in ``guardrails.json``."""

    adapter_id: ArtifactId
    kind: str
    url: str
    bearer_env: str | None = None
    max_response_bytes: int = Field(
        default=DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES,
        ge=1,
        le=1_048_576,
    )


def load_guardrail_engine(
    root: Path,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> GuardrailEngine | None:
    """Return an engine when ``ROOT/gateway/guardrails.json`` exists.

    Missing files leave the gateway unguarded. The file assigns policies by
    organization and identity. It may register ``keyword`` adapters for local
    tests and ``http_json`` adapters for dedicated classifier endpoints. It
    never stores raw prompts, responses, detector payloads, or credentials.

    Args:
        root: Initialized EXP root that contains the ``gateway`` directory.
        http_client: Optional shared client injected into every ``http_json``
            adapter. Production loads omit this and use the process pool.

    Returns:
        A composed engine, or ``None`` when no file is present.

    Raises:
        ValueError: The file exists but is not a valid policy document.
    """
    path = root / "gateway" / _CONFIG_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("gateway guardrail configuration is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("gateway guardrail configuration must be a JSON object")
    return engine_from_document(cast(JsonObject, payload), http_client=http_client)


def engine_from_document(
    payload: JsonObject,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> GuardrailEngine:
    """Compose one engine from an already parsed configuration object.

    Args:
        payload: Policy and optional adapter document.
        http_client: Optional shared client for ``http_json`` adapters.

    Returns:
        An engine with a mapping store and registered adapters.

    Raises:
        ValueError: Policies or adapters are malformed.
    """
    adapters = _adapters(payload, http_client=http_client)
    policies = _policies(payload, frozenset(adapters))
    return GuardrailEngine(
        store=MappingGuardrailStore(policies),
        client=DirectClassifierClient(ClassifierRegistry(adapters)),
        monotonic=time.monotonic,
    )


def _policies(payload: JsonObject, adapter_ids: frozenset[str]) -> tuple[GuardrailPolicy, ...]:
    """Parse the authored policy list and expand any standard presets."""
    raw = payload.get("policies", [])
    if not isinstance(raw, list):
        raise ValueError("guardrail policies must be a list")
    policies = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each guardrail policy must be an object")
        policies.append(policy_from_authored(cast(Mapping[str, object], item), adapter_ids))
    return tuple(policies)


def _adapters(
    payload: JsonObject,
    *,
    http_client: httpx.AsyncClient | None,
) -> dict[str, InspectingClassifier]:
    """Register keyword and http_json adapters declared in the document."""
    raw = payload.get("adapters", [])
    if raw == []:
        return {}
    if not isinstance(raw, list):
        raise ValueError("guardrail adapters must be a list")
    adapters: dict[str, InspectingClassifier] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each guardrail adapter must be an object")
        adapter = _adapter(cast(dict[str, object], item), http_client=http_client)
        adapter_id = _adapter_id(item)
        if adapter_id in adapters:
            raise ValueError(f"duplicate guardrail adapter_id: {adapter_id}")
        adapters[adapter_id] = adapter
    return adapters


def _adapter(
    item: dict[str, object],
    *,
    http_client: httpx.AsyncClient | None,
) -> InspectingClassifier:
    """Build one declared adapter after rejecting credential literals."""
    _reject_credential_literals(item)
    kind = item.get("kind")
    if kind == "keyword":
        return _keyword_adapter(item)
    if kind == "http_json":
        return _http_json_adapter(item, http_client=http_client)
    raise ValueError("adapter kind must be keyword or http_json")


def _keyword_adapter(item: dict[str, object]) -> KeywordClassifier:
    """Parse one coarse local keyword adapter."""
    try:
        authored = KeywordAdapterDocument.model_validate(item)
    except ValidationError as exc:
        raise ValueError("keyword adapter is malformed") from exc
    return KeywordClassifier(authored.needles)


def _http_json_adapter(
    item: dict[str, object],
    *,
    http_client: httpx.AsyncClient | None,
) -> HttpJsonClassifier:
    """Parse one dedicated HTTP JSON adapter."""
    try:
        authored = HttpJsonAdapterDocument.model_validate(item)
    except ValidationError as exc:
        raise ValueError("http_json adapter is malformed") from exc
    validate_classifier_url(authored.url)
    if authored.bearer_env is not None:
        validate_bearer_env_name(authored.bearer_env)
    return HttpJsonClassifier(
        adapter_id=authored.adapter_id,
        url=authored.url,
        bearer_env=authored.bearer_env,
        max_response_bytes=authored.max_response_bytes,
        client=http_client,
    )


def _adapter_id(item: dict[str, object]) -> str:
    """Return the adapter identity after a cheap presence check."""
    adapter_id = item.get("adapter_id")
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("guardrail adapter_id must be a non-empty string")
    return adapter_id


def _reject_credential_literals(item: dict[str, object]) -> None:
    """Reject adapter keys that look like inline credentials.

    Args:
        item: One adapter object from the configuration document.

    Raises:
        ValueError: A forbidden credential field is present.
    """
    forbidden = sorted(
        key for key in item if key.lower().replace("-", "_") in _FORBIDDEN_ADAPTER_KEYS
    )
    if forbidden:
        raise ValueError(
            "adapter documents cannot include credential fields; use bearer_env "
            "for an environment variable name"
        )
