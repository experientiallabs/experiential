"""Tests for the executable closed OpenAI compatibility manifests."""

from __future__ import annotations

import pytest

from wmo.runtime.gateway.contracts import CompatibilityDisposition, CompatibilityManifest
from wmo.runtime.openai_protocol.manifest import (
    CHAT_MANIFEST,
    RESPONSES_MANIFEST,
    disposition_map,
)


@pytest.mark.parametrize(
    ("manifest", "field"),
    tuple(
        (manifest, item.field_path)
        for manifest in (CHAT_MANIFEST, RESPONSES_MANIFEST)
        for item in manifest.fields
        if item.disposition != CompatibilityDisposition.UNSUPPORTED
    ),
)
def test_every_accepted_field_has_one_executable_manifest_decision(
    manifest: CompatibilityManifest, field: str
) -> None:
    """Every accepted field is explicit and unique rather than SDK-version widened."""
    assert field in disposition_map(manifest)


def test_manifests_classify_explicit_exclusions() -> None:
    """Multimodal, hosted, background, logprob, and multi-choice features stay excluded."""
    chat = disposition_map(CHAT_MANIFEST)
    responses = disposition_map(RESPONSES_MANIFEST)
    assert chat["audio"] == CompatibilityDisposition.UNSUPPORTED
    assert chat["n"] == CompatibilityDisposition.UNSUPPORTED
    assert chat["logprobs"] == CompatibilityDisposition.UNSUPPORTED
    assert responses["background"] == CompatibilityDisposition.UNSUPPORTED
    assert responses["conversation"] == CompatibilityDisposition.UNSUPPORTED
    assert responses["include"] == CompatibilityDisposition.UNSUPPORTED
