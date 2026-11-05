"""Canonical locator preservation and invalid grammar refusals."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.rules.locator_inventory import LocatorInventory, encode_inventory


def test_roundtrip_every_reference_without_other_evidence_changes() -> None:
    """Compression preserves a complete445-reference inventory exactly."""
    refs = [f"trace:trace{i // 32}:span:span{i}" for i in range(440)] + [
        f"contract:tool{i}" for i in range(5)
    ]
    original = {
        "inventory": sorted(refs),
        "source_traces": [{"quote": '雪"\\\n'}],
        "family": {"questions": ["all"]},
    }
    encoded = {**original, "inventory": encode_inventory(refs)}
    assert LocatorInventory.model_validate(encoded["inventory"]).decode() == tuple(sorted(refs))
    assert {k: v for k, v in encoded.items() if k != "inventory"} == {
        k: v for k, v in original.items() if k != "inventory"
    }


@pytest.mark.parametrize(
    "data",
    [
        {"direct": ["contract:x", "contract:x"]},
        {"direct": ["trace:x:span:y"], "trace_spans": {"trace:x": ["y"]}},
        {"trace_spans": {"trace:x:other": ["y"]}},
        {"trace_spans": {"trace:x": [""]}},
    ],
)
def test_noncanonical_or_duplicate_inventory_refused(data: JsonObject) -> None:
    """Ambiguous prefix encodings cannot manufacture canonical references."""
    with pytest.raises(ValueError):
        LocatorInventory.model_validate(data)
