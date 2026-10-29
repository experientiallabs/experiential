"""Per-wire form of replayed tool-call arguments that encode no JSON object.

The OpenAI decoders keep such a call's exact text as ``raw_arguments`` and an
object stand-in as ``arguments`` (``ToolCall.arguments_treatment``; see
``exp.runtime.openai_protocol.tool_argument_repairs``). Each rung then sends
the least lossy form its wire can faithfully carry:

- OpenAI's own API and Azure OpenAI, on Chat Completions or Responses: the raw
  text, byte-exact. Both treat ``arguments`` as an opaque string and serve this
  history, so nothing is rewritten and nothing is disclosed.
- Every other string wire (OpenRouter, self-hosted vLLM, the OpenAI-compatible
  resellers, a foreign Responses origin): the stand-in encoded as one JSON
  object. These servers decode history arguments to render a chat template
  (vLLM ``json.loads`` them), so raw text that is no object would move the 400
  upstream, where a provider 400 does not fail over.
- Anthropic Messages (``tool_use.input``), Gemini (``functionCall.args``) and
  Bedrock Converse (``toolUse.input``): the stand-in object, the only form
  those fields take.

Every rung that receives the stand-in discloses it in ``ignored_parameters``,
one entry per treatment, e.g. ``input.arguments->repaired(truncated_json)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never
from urllib.parse import urlsplit

from exp.common.models.model import ToolArgumentsTreatment, ToolCall
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.surfaces import GatewayApiSurface

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

OPAQUE_ARGUMENT_DIALECTS: Final = frozenset({"openai_compatible", "openai_responses"})
"""The wires whose ``arguments`` field is a string an origin may treat as opaque."""

_OPENAI_HOST: Final = "api.openai.com"
_AZURE_OPENAI_HOST_SUFFIXES: Final = (".openai.azure.com", ".cognitiveservices.azure.com")

TREATMENT_ORDER: Final[tuple[ToolArgumentsTreatment, ...]] = (
    "double_encoded_json",
    "truncated_json",
    "non_object_json",
    "invalid_json",
)
"""Every treatment, in the order disclosures are reported."""


def opaque_tool_arguments_wire(profile: GatewayWireProfile) -> bool:
    """Whether a rung's origin serves replayed argument text that encodes no object.

    Args:
        profile: The rung's resolved wire profile.

    Returns:
        True for OpenAI's own API and Azure OpenAI resource hosts on a string wire.
    """
    if profile.dialect not in OPAQUE_ARGUMENT_DIALECTS:
        return False
    host = (urlsplit(profile.url).hostname or "").lower()
    return host == _OPENAI_HOST or host.endswith(_AZURE_OPENAI_HOST_SUFFIXES)


def treatment_disclosure(surface: GatewayApiSurface, treatment: ToolArgumentsTreatment) -> str:
    """The ``ignored_parameters`` entry for one treatment a rung's stand-in carries.

    Args:
        surface: The public surface whose field path the disclosure names.
        treatment: How the stand-in was derived.

    Returns:
        The path->effect disclosure string.
    """
    path = (
        "input.arguments"
        if surface == GatewayApiSurface.RESPONSES
        else "messages.tool_calls.function.arguments"
    )
    match treatment:
        case "double_encoded_json":
            effect = "decoded(double_encoded_json)"
        case "truncated_json":
            effect = "repaired(truncated_json)"
        case "non_object_json":
            effect = "wrapped(non_object_json)"
        case "invalid_json":
            effect = "wrapped(invalid_json)"
        case _:
            assert_never(treatment)
    return f"{path}->{effect}"


def history_tool_argument_disclosures(
    profile: GatewayWireProfile, request: GatewayRequest
) -> tuple[str, ...]:
    """Disclose each treatment whose stand-in this rung sends instead of the raw text.

    Args:
        profile: The rung's resolved wire profile.
        request: The rung's canonical request.

    Returns:
        One disclosure per treatment present, in ``TREATMENT_ORDER``; empty on
        an opaque wire or when no replayed call needed a stand-in.
    """
    if opaque_tool_arguments_wire(profile):
        return ()
    present = {
        call.arguments_treatment
        for message in request.messages
        for call in message.tool_calls
        if call.arguments_treatment is not None
    }
    return tuple(
        treatment_disclosure(request.surface, treatment)
        for treatment in TREATMENT_ORDER
        if treatment in present
    )


def history_tool_arguments_for_wire(
    profile: GatewayWireProfile, request: GatewayRequest
) -> GatewayRequest:
    """Replace replayed raw argument text with its stand-in unless the wire takes it opaque.

    Object wires already read ``arguments``; a non-opaque string wire would
    otherwise send ``raw_arguments`` verbatim, so the raw text is dropped and
    the stand-in encodes in its place.

    Args:
        profile: The rung's resolved wire profile.
        request: The rung's canonical request.

    Returns:
        The request unchanged on an opaque wire or when nothing was treated;
        otherwise a copy whose treated calls carry the stand-in alone.
    """
    if opaque_tool_arguments_wire(profile) or not any(
        call.arguments_treatment is not None
        for message in request.messages
        for call in message.tool_calls
    ):
        return request
    messages = tuple(
        message.model_copy(update={"tool_calls": tuple(_stand_in(c) for c in message.tool_calls)})
        if any(call.arguments_treatment is not None for call in message.tool_calls)
        else message
        for message in request.messages
    )
    return request.model_copy(update={"messages": messages})


def _stand_in(call: ToolCall) -> ToolCall:
    """Return a treated call carrying only its object stand-in."""
    if call.arguments_treatment is None:
        return call
    return call.model_copy(update={"raw_arguments": None, "arguments_treatment": None})
