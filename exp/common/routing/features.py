"""One versioned request-visible feature extractor shared by fit and future runtime."""

from __future__ import annotations

from exp.common.core.artifacts import (
    ArtifactId,
    ContractModel,
    JsonObject,
    Sha256,
    canonical_json_bytes,
    sha256_json,
)
from exp.common.models import ModelRequest
from exp.common.tasks import TaskCase, ToolSchema

ROUTER_FEATURE_EXTRACTOR_ID: ArtifactId = "request-visible-v2"
ROUTER_FEATURE_SCHEMA_SHA256: Sha256 = sha256_json(
    {
        "extractor_id": ROUTER_FEATURE_EXTRACTOR_ID,
        "fields": ["initial_user_intent", "initial_context", "tools", "allowed_tags"],
        "tool_fields": ["name", "description", "input_schema"],
        "ordering": "first-user-message-only-tools-by-name-json-keys-sorted",
    }
)


class RouterFeatureRecord(ContractModel):
    """Canonical request-visible content sent to the configured embedding model."""

    initial_user_intent: str
    initial_context: JsonObject
    tools: tuple[ToolSchema, ...]
    allowed_tags: JsonObject


class RouterFeatureExtractor:
    """Render fit tasks and online model requests through one immutable feature schema."""

    @property
    def extractor_id(self) -> ArtifactId:
        """Return the stable feature implementation version."""
        return ROUTER_FEATURE_EXTRACTOR_ID

    @property
    def schema_sha256(self) -> Sha256:
        """Return the exact persisted feature schema digest."""
        return ROUTER_FEATURE_SCHEMA_SHA256

    def from_task(self, task: TaskCase, *, allowed_tags: JsonObject | None = None) -> str:
        """Render only request-visible fields from one representative fit task.

        Args:
            task: Canonical task whose partition, lineage, weight, outcomes, and source IDs are
                deliberately excluded.
            allowed_tags: Optional caller-approved tags available on the corresponding live call.

        Returns:
            Deterministic embedding text under the same schema used for live requests.
        """
        record = RouterFeatureRecord(
            initial_user_intent=task.instruction,
            initial_context=task.initial_context,
            tools=_ordered_tools(task.tools),
            allowed_tags=allowed_tags or {},
        )
        return self.render(record)

    def from_request(
        self,
        request: ModelRequest,
        *,
        initial_context: JsonObject | None = None,
        allowed_tags: JsonObject | None = None,
    ) -> str:
        """Render request-visible live content through the frozen fit-time schema.

        Args:
            request: Provider-neutral messages and tools visible before routing.
            initial_context: Optional policy-approved initial context visible to the live caller.
            allowed_tags: Optional policy-approved routing tags visible to the live caller.

        Returns:
            Deterministic embedding text with no later action, outcome, or episode metadata.

        Raises:
            ValueError: The request has no initial user message.
        """
        initial_user_intent = next(
            (
                message.content
                for message in request.messages
                if message.role == "user" and message.content is not None
            ),
            None,
        )
        if initial_user_intent is None:
            raise ValueError("router feature extraction requires an initial user message")
        record = RouterFeatureRecord(
            initial_user_intent=initial_user_intent,
            initial_context=initial_context or {},
            tools=_ordered_tools(request.tools),
            allowed_tags=allowed_tags or {},
        )
        return self.render(record)

    def render(self, record: RouterFeatureRecord) -> str:
        """Serialize one validated feature record deterministically.

        Args:
            record: Request-visible feature values under the frozen schema.

        Returns:
            Canonical UTF-8 JSON text for the embedding client.
        """
        return canonical_json_bytes(record).decode("utf-8")


def _ordered_tools(tools: tuple[ToolSchema, ...]) -> tuple[ToolSchema, ...]:
    """Canonicalize semantically unordered tool definitions by name."""
    return tuple(sorted(tools, key=lambda tool: tool.name))
