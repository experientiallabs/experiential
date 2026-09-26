"""Pure, opt-in Google cached-content planning from exact text checkpoints.

A plan is not spending authority or evidence of a cache write. The consumer owns
customer funding, a positive configured allowance, model minimums, pricing,
account isolation, resource lifecycle, and truthful cache-use disclosure. No
provider request or credential lookup happens here. Unsupported shapes return
``None`` and retain the existing implicit-cache and dropped-marker behavior.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, canonical_json_bytes, sha256_json
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.cache_policy import cache_markers

_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_GEMINI_PATH = re.compile(rf"/v1beta/models/({_SEGMENT}):streamGenerateContent")
_VERTEX_PATH = re.compile(
    rf"/v1/(projects/({_SEGMENT})/locations/([a-z0-9-]+))"
    rf"/publishers/google/models/({_SEGMENT}):streamGenerateContent"
)
_VERTEX_CACHE_PATH = re.compile(rf"/v1/projects/({_SEGMENT})/locations/([a-z0-9-]+)/cachedContents")
_REGION = re.compile(r"[a-z]+(?:-[a-z]+)+[0-9]+")
_RESOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_POLICY = "google-explicit-text-prefix-v1"


@dataclass(frozen=True)
class VertexCacheProject:
    """Host-verified identity linking an endpoint project to its resource number.

    Attributes:
        endpoint_project: Exact project ID or number in the admitted Vertex URL.
        project_number: Canonical positive decimal project number returned by Google.
            The host verifies this association for the selected provider account;
            neither caller input nor a cache response establishes the association.
    """

    endpoint_project: str
    project_number: str

    def __post_init__(self) -> None:
        """Reject malformed aliases and any attempt to remap a numeric project."""
        if (
            not isinstance(self.endpoint_project, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,255}", self.endpoint_project)
            or not isinstance(self.project_number, str)
            or not re.fullmatch(r"[1-9][0-9]{0,19}", self.project_number)
            or (self.endpoint_project.isdecimal() and self.endpoint_project != self.project_number)
        ):
            raise ValueError("Vertex cache project needs a verified endpoint and canonical number")


@dataclass(frozen=True)
class GoogleCachePlan:
    """Immutable cache creation and continuation payloads, without authority.

    Attributes:
        create_url: Exact allowlisted cache collection URL. Hidden from repr
            because a Gemini API key may be in its query string.
        model: Provider model resource name included in the cache body.
        resource_prefix: Exact resource collection, including Vertex scope,
            followed by a slash. Bind project-ID plans to host-verified numeric
            scope before claiming resources. ``apply`` accepts one identifier below it.
        prefix_sha256: Canonical sorted-JSON digest of policy and resource body
            excluding TTL. Includes model, text, system instruction and tools;
            excludes generation controls, suffix, credentials and account scope.
            The consumer must separately bind credential/account authority.
        conservative_input_bound: UTF-8 byte length of the complete cache body,
            used as a deliberately conservative text-input reservation proxy,
            not a provider token count or proof of a model's token minimum.
        ttl_seconds: Explicit five-minute lifetime, fixed at 300 seconds.
        _create_payload_json: Immutable private canonical JSON, excluded from
            repr. ``create_payload`` returns a fresh decoded copy on every read.
        _generation_payload_json: Immutable private canonical JSON, excluded
            from repr. Its public property has no ``cachedContent`` until apply.
    """

    create_url: str = field(repr=False)
    model: str
    resource_prefix: str
    prefix_sha256: str
    conservative_input_bound: int
    _create_payload_json: bytes = field(repr=False)
    _generation_payload_json: bytes = field(repr=False)
    ttl_seconds: int = field(default=300, init=False)

    @property
    def create_payload(self) -> JsonObject:
        """Return a fresh cache resource body with explicit ``ttl: 300s``."""
        return cast("JsonObject", json.loads(self._create_payload_json))

    @property
    def generation_payload(self) -> JsonObject:
        """Return a fresh continuation body without a cached resource name."""
        return cast("JsonObject", json.loads(self._generation_payload_json))

    def bind_vertex_project(self, binding: VertexCacheProject | None) -> GoogleCachePlan | None:
        """Bind the exact numeric resource namespace without rewriting the request.

        Project-ID endpoints require a host-verified mapping before any claim or
        provider call. Numeric endpoints already name their canonical project.
        Gemini has no project namespace and rejects an accidental Vertex binding.
        Missing authority returns no plan; contradictory authority fails closed.
        """
        if self.resource_prefix == "cachedContents/":
            if binding is not None:
                raise ValueError("Gemini cache authority must not contain a Vertex project")
            return self
        match = _VERTEX_CACHE_PATH.fullmatch(urlsplit(self.create_url).path)
        if match is None:
            raise ValueError("Vertex cache plan does not have an exact project/location scope")
        project, location = match[1], match[2]
        if binding is None:
            if not re.fullmatch(r"[1-9][0-9]{0,19}", project):
                return None
            number = project
        else:
            if binding.endpoint_project != project:
                raise ValueError(
                    "Vertex cache authority differs from the admitted endpoint project"
                )
            number = binding.project_number
        prefix = f"projects/{number}/locations/{location}/cachedContents/"
        if prefix == self.resource_prefix:
            return self
        return replace(
            self,
            resource_prefix=prefix,
            prefix_sha256=sha256_json({"plan": self.prefix_sha256, "resource_prefix": prefix}),
        )

    def apply(self, resource_name: str) -> JsonObject:
        """Attach one exact-scope provider resource name to a fresh continuation.

        Args:
            resource_name: Provider-returned resource name under this plan's
                collection, never a URL or a resource in another location.

        Returns:
            A new deeply independent native generation body.

        Raises:
            ValueError: The name has a foreign scope, nested path, query,
                fragment, encoded delimiter, or unsupported identifier.
        """
        if not resource_name.startswith(self.resource_prefix) or not _RESOURCE_ID.fullmatch(
            resource_name[len(self.resource_prefix) :]
        ):
            raise ValueError(
                "Google cache resource must be one identifier in the planned collection; "
                "use the exact name returned by cache creation."
            )
        payload = self.generation_payload
        payload["cachedContent"] = resource_name
        return payload


def build_google_cache_plan(
    profile: GatewayWireProfile,
    request: GatewayRequest,
    upstream_payload: JsonObject,
) -> GoogleCachePlan | None:
    """Split a native Google request at its last supported explicit checkpoint.

    Only leading system/user text checkpoints qualify. All original markers
    must be valid ephemeral five-minute hints (omitted TTL means five minutes).
    The whole request must be text-only without tool history, native carriers,
    automatic markers, tool markers, or interleaved instructions. A checkpoint
    inside user text is supported only after exact concatenation verification.
    A system checkpoint must end the entire leading system instruction because
    a cached generation cannot also supply an uncached system instruction.

    Args:
        profile: Resolved native Gemini profile on an official Gemini or Vertex
            endpoint. Signed request bodies cannot be rewritten.
        request: Original canonical request retaining checkpoint positions.
        upstream_payload: Exact already-built native generation body, unchanged
            by this function. Sampling stays in this body's continuation.

    Returns:
        A pure plan, or ``None`` for an unmarked, unsupported, ambiguous, or
        mismatched request. ``None`` never claims the requested resource was
        created: consumers must preserve existing dropped-marker disclosure.
        Funding, configured allowance and minimum-token eligibility are HOST
        decisions, not facts inferred from this return value.
    """
    if profile.dialect != "gemini_generate_content" or profile.signs_request_body:
        return None
    endpoint = _cache_endpoint(profile)
    markers = cache_markers(request)
    if endpoint is None or not markers or any(not _five_minute_marker(m) for m in markers):
        return None
    if (
        request.provider_cache_control is not None
        or request.provider_server_tools
        or request.provider_native_tools
        or request.web_search is not None
        or request.tool_search is not None
        or any(tool.cache_control is not None for tool in request.tools)
    ):
        return None
    checkpoint = _last_checkpoint(request)
    if checkpoint is None or not _payload_matches(request, upstream_payload):
        return None
    message_index, offset = checkpoint
    system_count = sum(message.role == "system" for message in request.messages)
    if message_index < system_count and (
        message_index != system_count - 1
        or offset != len(request.messages[message_index].content or "")
    ):
        return None

    create_url, model, resource_prefix = endpoint
    generation = cast("JsonObject", json.loads(canonical_json_bytes(upstream_payload)))
    resource: JsonObject = {"model": model}
    for key in ("systemInstruction", "tools", "toolConfig"):
        if key in generation:
            resource[key] = generation.pop(key)
    contents = cast("list[JsonObject]", generation["contents"])
    if message_index >= system_count:
        content_index = message_index - system_count
        prefix = contents[:content_index]
        marked_content = contents[content_index]
        cached_parts, remaining_parts = _split_parts(
            cast("list[JsonObject]", marked_content["parts"]), offset
        )
        prefix.append({"role": "user", "parts": cached_parts})
        suffix = contents[content_index + 1 :]
        if remaining_parts:
            suffix.insert(0, {"role": "user", "parts": remaining_parts})
        if not suffix:
            # Do not invent a generation prompt or rely on empty contents.
            return None
        resource["contents"] = prefix
        generation["contents"] = suffix
    digest = sha256_json({"policy": _POLICY, "resource": resource})
    resource["ttl"] = "300s"
    resource_json = canonical_json_bytes(resource)
    return GoogleCachePlan(
        create_url=create_url,
        model=model,
        resource_prefix=resource_prefix,
        prefix_sha256=digest,
        conservative_input_bound=len(resource_json),
        _create_payload_json=resource_json,
        _generation_payload_json=canonical_json_bytes(generation),
    )


def _five_minute_marker(marker: JsonObject) -> bool:
    """Accept only the decoder's explicit ephemeral five-minute marker shape."""
    return (
        set(marker) <= {"type", "ttl"}
        and marker.get("type") == "ephemeral"
        and marker.get("ttl", "5m") == "5m"
    )


def _cache_endpoint(profile: GatewayWireProfile) -> tuple[str, str, str] | None:
    """Derive an exact official cache collection without reading credentials."""
    try:
        url = urlsplit(profile.url)
    except ValueError:
        return None
    if (
        url.scheme != "https"
        or url.fragment
        or any(character.isspace() for character in profile.url)
    ):
        return None
    query = parse_qsl(url.query, keep_blank_values=True)
    if (
        len({key for key, _ in query}) != len(query)
        or any(key not in {"alt", "key"} for key, _ in query)
        or any(key == "alt" and value != "sse" for key, value in query)
        or any(key == "key" and not value for key, value in query)
    ):
        return None
    auth_query = urlencode([(key, value) for key, value in query if key == "key"])
    if url.netloc == "generativelanguage.googleapis.com":
        match = _GEMINI_PATH.fullmatch(url.path)
        if match is None:
            return None
        model = f"models/{match[1]}"
        if profile.model_id not in {match[1], model}:
            return None
        return (
            urlunsplit(("https", url.netloc, "/v1beta/cachedContents", auth_query, "")),
            model,
            "cachedContents/",
        )
    match = _VERTEX_PATH.fullmatch(url.path)
    if match is None or auth_query:
        return None
    scope, location, model_id = match[1], match[3], match[4]
    if location == "global":
        host = "aiplatform.googleapis.com"
    elif _REGION.fullmatch(location):
        host = f"{location}-aiplatform.googleapis.com"
    else:
        return None
    if url.netloc != host or profile.operational_region not in {None, location}:
        return None
    model = f"{scope}/publishers/google/models/{model_id}"
    if profile.model_id not in {
        model_id,
        f"models/{model_id}",
        f"publishers/google/models/{model_id}",
        model,
    }:
        return None
    return (
        urlunsplit(("https", host, f"/v1/{scope}/cachedContents", "", "")),
        model,
        f"{scope}/cachedContents/",
    )


def _last_checkpoint(request: GatewayRequest) -> tuple[int, int] | None:
    """Locate the last original text checkpoint without moving any boundary."""
    checkpoint: tuple[int, int] | None = None
    conversation_started = False
    user_prefix_ended = False
    for index, message in enumerate(request.messages):
        if (
            message.content is None
            or message.role not in {"system", "user", "assistant"}
            or message.content_parts
            or message.tool_calls
            or message.cache_control is not None
            or message.provider_reasoning
            or message.provider_native_item is not None
            or message.provider_anthropic_block is not None
            or message.provider_anthropic_blocks is not None
            or message.provider_item_id is not None
        ):
            return None
        if message.role == "system":
            if conversation_started:
                return None
        else:
            conversation_started = True
        if message.role == "assistant":
            user_prefix_ended = True
        if not message.provider_text_blocks:
            continue
        offsets = _checkpoint_offsets(message)
        if offsets is None:
            return None
        for offset in offsets:
            if user_prefix_ended or offset == 0:
                return None
            checkpoint = index, offset
    return checkpoint


def _checkpoint_offsets(message: GatewayMessage) -> tuple[int, ...] | None:
    """Map retained block boundaries to canonical text using an exact join."""
    texts: list[str] = []
    marked: list[bool] = []
    for block in message.provider_text_blocks:
        text = block.get("text")
        if (
            set(block) - {"type", "text", "cache_control"}
            or block.get("type") != "text"
            or not isinstance(text, str)
            or ("cache_control" in block and not isinstance(block["cache_control"], dict))
        ):
            return None
        texts.append(text)
        marked.append("cache_control" in block)
    if "".join(texts) == message.content:
        separator = ""
    elif message.role == "system" and "\n\n".join(texts) == message.content:
        separator = "\n\n"
    else:
        return None
    offset = 0
    checkpoints: list[int] = []
    for index, (text, has_marker) in enumerate(zip(texts, marked, strict=True)):
        if index:
            offset += len(separator)
        offset += len(text)
        if has_marker:
            checkpoints.append(offset)
    return tuple(checkpoints)


def _text_parts(value: JsonValue) -> list[str] | None:
    """Accept only nonempty lists of native plain-text parts, without carriers."""
    if not isinstance(value, list) or not value:
        return None
    texts: list[str] = []
    for part in value:
        if not isinstance(part, dict) or set(part) != {"text"}:
            return None
        text = part["text"]
        if not isinstance(text, str):
            return None
        texts.append(text)
    return texts


def _payload_matches(request: GatewayRequest, payload: JsonObject) -> bool:
    """Verify native text positions and closed function-tool shapes exactly."""
    if set(payload) - {"contents", "systemInstruction", "tools", "toolConfig", "generationConfig"}:
        return False
    system = [message.content for message in request.messages if message.role == "system"]
    instruction = payload.get("systemInstruction")
    if system:
        if (
            not isinstance(instruction, dict)
            or set(instruction) != {"parts"}
            or _text_parts(instruction["parts"]) != system
        ):
            return False
    elif instruction is not None or "systemInstruction" in payload:
        return False
    contents = payload.get("contents")
    messages = [message for message in request.messages if message.role != "system"]
    if not isinstance(contents, list) or not contents or len(contents) != len(messages):
        return False
    for content, message in zip(contents, messages, strict=True):
        if not isinstance(content, dict) or set(content) != {"role", "parts"}:
            return False
        if content["role"] != ("model" if message.role == "assistant" else "user"):
            return False
        texts = _text_parts(content["parts"])
        if texts is None or "".join(texts) != message.content:
            return False
    return _tools_match(request, payload)


def _tools_match(request: GatewayRequest, payload: JsonObject) -> bool:
    """Allow only function definitions and calling policy from the canonical request."""
    if request.tools:
        expected: list[JsonObject] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description or tool.name,
                        "parametersJsonSchema": tool.parameters,
                    }
                    for tool in request.tools
                ]
            }
        ]
        if payload.get("tools") != expected:
            return False
    elif "tools" in payload:
        return False
    choice = request.tool_choice
    if choice is None:
        return "toolConfig" not in payload
    if isinstance(choice, str):
        policy: JsonObject = {"mode": {"auto": "AUTO", "none": "NONE", "required": "ANY"}[choice]}
    else:
        policy = {"mode": "ANY", "allowedFunctionNames": [choice.name]}
    return payload.get("toolConfig") == {"functionCallingConfig": policy}


def _split_parts(parts: list[JsonObject], offset: int) -> tuple[list[JsonObject], list[JsonObject]]:
    """Split verified plain text at one canonical offset, preserving every byte."""
    prefix: list[JsonObject] = []
    suffix: list[JsonObject] = []
    remaining = offset
    for part in parts:
        text = cast("str", part["text"])
        if remaining >= len(text) and remaining > 0:
            prefix.append(part)
            remaining -= len(text)
        elif remaining > 0:
            prefix.append({"text": text[:remaining]})
            suffix.append({"text": text[remaining:]})
            remaining = 0
        else:
            suffix.append(part)
    return prefix, suffix
