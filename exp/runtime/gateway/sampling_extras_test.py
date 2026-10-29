"""Tests for the OpenRouter sampling extras carrier and its per-rung forwarding."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.sampling_extras import (
    OPENROUTER_SAMPLING_EXTRAS,
    VLLM_SAMPLING_EXTRAS,
    SamplingExtras,
    forward_sampling_extras,
    sampling_extras_of,
)


def test_an_absent_carrier_stays_none() -> None:
    """No extras sent keeps the request field ``None``, so its digest is unchanged."""
    assert sampling_extras_of(SamplingExtras()) is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("repetition_penalty", 0.0),
        ("repetition_penalty", 2.5),
        ("min_p", -0.1),
        ("min_p", 1.5),
        ("top_a", 1.01),
    ),
)
def test_out_of_range_values_are_refused(field: str, value: float) -> None:
    """Each extra keeps OpenRouter's documented range."""
    with pytest.raises(ValidationError):
        SamplingExtras.model_validate({field: value})


def test_accepted_extras_forward_and_the_rest_are_disclosed() -> None:
    """A vLLM wire takes repetition_penalty and min_p; top_a is dropped by name."""
    extras = SamplingExtras(repetition_penalty=1.1, min_p=0.05, top_a=0.2)
    payload, disclosures = forward_sampling_extras({"model": "m"}, extras, VLLM_SAMPLING_EXTRAS)
    assert payload == {"model": "m", "repetition_penalty": 1.1, "min_p": 0.05}
    assert disclosures == ("top_a->dropped(unsupported_by_provider)",)


def test_openrouter_takes_every_extra() -> None:
    """OpenRouter's wire documents all three extras."""
    extras = SamplingExtras(repetition_penalty=1.1, min_p=0.05, top_a=0.2)
    payload, disclosures = forward_sampling_extras({}, extras, OPENROUTER_SAMPLING_EXTRAS)
    assert payload == {"repetition_penalty": 1.1, "min_p": 0.05, "top_a": 0.2}
    assert disclosures == ()


def test_a_strict_wire_drops_every_extra_with_disclosure() -> None:
    """A wire carrying none dispatches the payload unchanged and names each drop."""
    extras = SamplingExtras(repetition_penalty=1.2, min_p=0.1)
    payload, disclosures = forward_sampling_extras({"model": "m"}, extras, frozenset())
    assert payload == {"model": "m"}
    assert disclosures == (
        "repetition_penalty->dropped(unsupported_by_provider)",
        "min_p->dropped(unsupported_by_provider)",
    )
