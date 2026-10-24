"""OpenRouter's open-weight sampling extras: ``repetition_penalty``, ``min_p`` and ``top_a``.

OpenRouter accepts these on every Chat Completions request as standard
sampling parameters, forwards them to upstreams that take them and ignores
them elsewhere. The gateway does the same per rung: a rung whose wire carries
the field receives it verbatim, every other rung dispatches without it and the
drop is disclosed in ``ignored_parameters``. A request is never refused for
asking, but its values stay validated, so an out-of-range value is a named 400.

Which wire accepts which field is decided once per rung in ``native_rungs``:
OpenRouter carries all three, a self-hosted vLLM rung (``upstream_priority``
authored) carries ``repetition_penalty`` and ``min_p`` (vLLM has no ``top_a``),
and every other wire carries none: OpenAI, Azure, Anthropic, Gemini, Bedrock,
the native Responses dialect, and the generic OpenAI-compatible resellers,
some of which refuse unknown body fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from exp.common.core.artifacts import ContractModel, JsonObject

type SamplingExtraName = Literal["repetition_penalty", "min_p", "top_a"]
"""One OpenRouter sampling extra, spelled as the public request field."""

SAMPLING_EXTRA_FIELDS: tuple[SamplingExtraName, ...] = ("repetition_penalty", "min_p", "top_a")
"""Every sampling extra, in the order disclosures are reported."""

OPENROUTER_SAMPLING_EXTRAS: frozenset[SamplingExtraName] = frozenset(SAMPLING_EXTRA_FIELDS)
"""The extras OpenRouter's Chat Completions wire documents and forwards."""

VLLM_SAMPLING_EXTRAS: frozenset[SamplingExtraName] = frozenset({"repetition_penalty", "min_p"})
"""The extras a self-hosted vLLM server's sampling parameters carry."""


class SamplingExtraFields(BaseModel):
    """The three extras as top-level request fields, each at OpenRouter's documented range.

    Shared by the Chat wire model (where a bad value is a field-named 400) and
    the canonical carrier, so both validate the same bounds. Values are strict
    numbers: an integer is accepted, a string or a boolean is refused.

    Attributes:
        repetition_penalty: Penalty on tokens already in the prompt or output,
            in ``(0, 2]``; 1 is neutral.
        min_p: Minimum token probability relative to the most likely token, in ``[0, 1]``.
        top_a: Dynamic probability floor scaled by the top token's probability, in ``[0, 1]``.
    """

    repetition_penalty: float | None = Field(default=None, gt=0, le=2, strict=True)
    min_p: float | None = Field(default=None, ge=0, le=1, strict=True)
    top_a: float | None = Field(default=None, ge=0, le=1, strict=True)


class SamplingExtras(SamplingExtraFields, ContractModel):
    """The caller's validated sampling extras, carried on the canonical request."""

    def present(self) -> tuple[tuple[SamplingExtraName, float], ...]:
        """Return every extra the caller set, in disclosure order."""
        values: dict[SamplingExtraName, float | None] = {
            "repetition_penalty": self.repetition_penalty,
            "min_p": self.min_p,
            "top_a": self.top_a,
        }
        return tuple(
            (name, value) for name in SAMPLING_EXTRA_FIELDS if (value := values[name]) is not None
        )


def sampling_extras_of(fields: SamplingExtraFields) -> SamplingExtras | None:
    """Collect decoded extras, or ``None`` when the caller sent none.

    ``None`` keeps a request without extras serializing and digesting exactly
    as it did before the carrier existed.

    Args:
        fields: The decoded wire request carrying the three fields.

    Returns:
        The extras carrier, or ``None`` when every value is absent.
    """
    extras = SamplingExtras(
        repetition_penalty=fields.repetition_penalty, min_p=fields.min_p, top_a=fields.top_a
    )
    return extras if extras.present() else None


def sampling_extra_disclosure(name: SamplingExtraName) -> str:
    """The ``ignored_parameters`` entry for an extra a rung's wire does not carry."""
    return f"{name}->dropped(unsupported_by_provider)"


def forward_sampling_extras(
    payload: JsonObject,
    extras: SamplingExtras | None,
    accepted: frozenset[SamplingExtraName],
) -> tuple[JsonObject, tuple[str, ...]]:
    """Add the extras one rung's wire accepts and disclose the ones it does not.

    Args:
        payload: The provider payload the rung's dialect built.
        extras: The caller's extras, or ``None``.
        accepted: The extras this rung's wire carries.

    Returns:
        The payload with every accepted extra set, and one disclosure per
        extra the rung dispatches without.
    """
    if extras is None:
        return payload, ()
    forwarded: JsonObject = {}
    disclosures: list[str] = []
    for name, value in extras.present():
        if name in accepted:
            forwarded[name] = value
        else:
            disclosures.append(sampling_extra_disclosure(name))
    return ({**payload, **forwarded} if forwarded else payload), tuple(disclosures)
