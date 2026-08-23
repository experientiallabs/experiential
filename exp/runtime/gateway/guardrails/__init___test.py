"""Public export smoke test for the guardrails package."""

from __future__ import annotations

from exp.runtime.gateway.guardrails import (
    BoundedSyncClassifier,
    GuardrailEngine,
    GuardrailPolicy,
    MappingGuardrailStore,
    load_guardrail_engine,
)


def test_package_exports_the_operator_facing_types() -> None:
    """The package surface stays small and importable."""
    assert BoundedSyncClassifier is not None
    assert GuardrailEngine is not None
    assert GuardrailPolicy is not None
    assert MappingGuardrailStore is not None
    assert load_guardrail_engine is not None
