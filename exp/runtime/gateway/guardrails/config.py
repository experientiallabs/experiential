"""Load an optional identity-scoped guardrail engine from the gateway root."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, KeywordClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore

_CONFIG_NAME = "guardrails.json"


def load_guardrail_engine(root: Path) -> GuardrailEngine | None:
    """Return an engine when ``ROOT/gateway/guardrails.json`` exists.

    Missing files leave the gateway unguarded. The file assigns policies by
    organization and identity and may register local keyword adapters. It
    never stores raw prompts, responses, or detector payloads.

    Args:
        root: Initialized EXP root that contains the ``gateway`` directory.

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
    return engine_from_document(cast(JsonObject, payload))


def engine_from_document(payload: JsonObject) -> GuardrailEngine:
    """Compose one engine from an already parsed configuration object.

    Args:
        payload: Policy and optional adapter document.

    Returns:
        An engine with a mapping store and registered keyword adapters.

    Raises:
        ValueError: Policies or adapters are malformed.
    """
    policies = _policies(payload)
    registry = ClassifierRegistry(_keyword_adapters(payload))
    return GuardrailEngine(
        store=MappingGuardrailStore(policies),
        client=DirectClassifierClient(registry),
        monotonic=time.monotonic,
    )


def _policies(payload: JsonObject) -> tuple[GuardrailPolicy, ...]:
    """Parse the authored policy list."""
    raw = payload.get("policies", [])
    if not isinstance(raw, list):
        raise ValueError("guardrail policies must be a list")
    policies: list[GuardrailPolicy] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each guardrail policy must be an object")
        policies.append(GuardrailPolicy.model_validate(item))
    return tuple(policies)


def _keyword_adapters(payload: JsonObject) -> dict[str, KeywordClassifier]:
    """Register local keyword adapters declared in the document."""
    raw = payload.get("adapters", [])
    if raw == []:
        return {}
    if not isinstance(raw, list):
        raise ValueError("guardrail adapters must be a list")
    adapters: dict[str, KeywordClassifier] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each guardrail adapter must be an object")
        adapter_id = item.get("adapter_id")
        kind = item.get("kind")
        needles = item.get("needles")
        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("guardrail adapter_id must be a non-empty string")
        if kind != "keyword":
            raise ValueError("only the keyword adapter kind is built in; inject others in code")
        if not isinstance(needles, list) or not all(isinstance(needle, str) for needle in needles):
            raise ValueError("keyword adapter needles must be a list of strings")
        adapters[adapter_id] = KeywordClassifier(tuple(needles))
    return adapters
