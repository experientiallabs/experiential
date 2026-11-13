"""Manifest coverage and SDK-surface drift gate for the generateContent surface."""

from __future__ import annotations

import inspect
import re

from pydantic import BaseModel

from exp.runtime.gateway.compatibility import CompatibilityDisposition
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.google_protocol.manifest import GENERATE_CONTENT_MANIFEST
from exp.runtime.google_protocol.requests import _GenerateContentRequest, _GenerationConfig


def test_manifest_binds_the_surface_with_unique_fields() -> None:
    """One disposition per field, both JSON spellings, on the right surface."""
    assert GENERATE_CONTENT_MANIFEST.surface == GatewayApiSurface.GENERATE_CONTENT
    paths = [field.field_path for field in GENERATE_CONTENT_MANIFEST.fields]
    assert len(paths) == len(set(paths))
    assert {"contents", "systemInstruction", "system_instruction", "generationConfig"} <= set(paths)


def _aliases(model: type[BaseModel]) -> set[str]:
    """Every JSON spelling a wire model accepts for its fields."""
    names: set[str] = set()
    for name, info in model.model_fields.items():
        alias = info.validation_alias
        if alias is None:
            names.add(name)
        else:
            names.update(str(choice) for choice in alias.choices)  # type: ignore[attr-defined]
    return names


def test_every_sdk_top_level_field_is_consciously_classified() -> None:
    """The SDK-surface drift gate: a ``google-genai`` bump may add request fields.

    The official SDK spreads ``GenerateContentConfig`` onto the wire body in
    two places: fields written to the request's top level must appear in the
    manifest, and fields written inside ``generationConfig`` must be known to
    the closed wire model, so a dependency bump turns new Google surface into
    a loud decision instead of a silent customer-facing 400.
    """
    from google.genai import models

    source = inspect.getsource(models._GenerateContentConfig_to_mldev)
    top_level = set(re.findall(r"setv\(\s*parent_object,\s*\['(\w+)'\]", source))
    nested = set(re.findall(r"setv\(\s*to_object,\s*\['(\w+)'\]", source))
    decided = {field.field_path for field in GENERATE_CONTENT_MANIFEST.fields}
    assert top_level <= decided, f"undecided top-level fields: {sorted(top_level - decided)}"
    known = _aliases(_GenerationConfig)
    assert nested <= known, f"unknown generationConfig fields: {sorted(nested - known)}"
    # Every manifest field the wire model can see is one it declares a slot for.
    assert decided <= _aliases(_GenerateContentRequest)


def test_cached_content_is_an_explicit_rejection() -> None:
    """A caller-named Google cache resource is refused, never silently dropped."""
    decisions = {field.field_path: field.disposition for field in GENERATE_CONTENT_MANIFEST.fields}
    assert decisions["cachedContent"] == CompatibilityDisposition.UNSUPPORTED
    assert decisions["cached_content"] == CompatibilityDisposition.UNSUPPORTED
