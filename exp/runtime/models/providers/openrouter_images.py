"""OpenRouter's dedicated Images API request mapping."""

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.models.providers.openai_compatible import openai_images_request
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def openrouter_images_request(model_id: str, request: ImagesRequest) -> JsonObject:
    """Map supported controls and reject unavailable output modes before dispatch."""
    for name, unsupported in (
        ("response_format", request.response_format == "url"),
        ("style", request.style is not None),
        ("quality", request.quality in ("standard", "hd")),
    ):
        if unsupported:
            raise OpenAIProtocolError(
                status_code=400,
                code="unsupported_parameter",
                message=f"OpenRouter Images does not support {name}={getattr(request, name)!r}; "
                "omit this parameter. Images are returned as b64_json.",
                param=name,
            )
    body = openai_images_request(model_id, request)
    body.pop("response_format", None)
    moderation = body.pop("moderation", None)
    provider: JsonObject = {"allow_fallbacks": False}
    if moderation is not None:
        provider["options"] = {"openai": {"moderation": moderation}}
    body["provider"] = provider
    if request.size == "auto":
        body.pop("size", None)
        body["aspect_ratio"] = "auto"
    return body
