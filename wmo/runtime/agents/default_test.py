"""Tests for the default pi agent document wmo ships."""

from __future__ import annotations

from wmo.runtime.agents.default import default_agent
from wmo.runtime.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    MAX_TURNS_ID,
    RUNTIME_KIND_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    SurfaceKind,
)
from wmo.runtime.harness.runtime import DEFAULT_MAX_OUTPUT_TOKENS


def test_default_agent_extends_the_baseline_with_the_pi_node_runtime() -> None:
    doc = default_agent()

    surfaces = {surface.id: surface for surface in doc.surfaces}
    assert doc.name == "default"
    # Everything the baseline contributes is still there.
    assert {"prompt:core", TOOL_POLICY_ID, MAX_TURNS_ID, TEMPERATURE_ID} <= set(surfaces)
    assert surfaces[RUNTIME_KIND_ID].content == "pi-node"
    assert surfaces[MAX_OUTPUT_TOKENS_ID].content == str(DEFAULT_MAX_OUTPUT_TOKENS)


def test_default_agent_carries_the_vendored_pi_source_as_code_surfaces() -> None:
    # The document is the whole agent: without the vendored source the runtime would have nothing
    # to execute, so a broken vendoring must show up here rather than at run time.
    doc = default_agent()

    code = [surface for surface in doc.surfaces if surface.kind is SurfaceKind.CODE]
    assert code, "the default agent ships no pi source"
    assert all(surface.path for surface in code)


def test_default_agent_takes_its_name() -> None:
    assert default_agent("scored-candidate").name == "scored-candidate"


def test_each_call_returns_an_independent_document() -> None:
    # Callers mutate what they get back (the optimizer edits surfaces); shared state would leak
    # one caller's edits into the next agent built from the default.
    first = default_agent()
    second = default_agent()

    assert first is not second
    assert first.surfaces is not second.surfaces
    assert first.doc_hash == second.doc_hash


def test_the_default_agent_is_a_valid_harness_document() -> None:
    # HarnessDoc validates on construction (tools resolve, submit present, params in range), so
    # round-tripping through the model is what proves the shipped default cannot be invalid.
    doc = default_agent()

    assert HarnessDoc.model_validate(doc.model_dump()).doc_hash == doc.doc_hash
