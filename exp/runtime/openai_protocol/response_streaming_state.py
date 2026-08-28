"""Mutable item state shared by the Responses stream encoder."""

from exp.common.core.artifacts import JsonObject


class ResponseReasoningState:
    """Accumulated reasoning-summary item with its provider output identity."""

    def __init__(self, *, item_id: str, output_index: int) -> None:
        """Initialize one reasoning item with no summary parts."""
        self.item_id = item_id
        self.output_index = output_index
        self.parts: dict[int, str] = {}
        self.encrypted_content: str | None = None

    def item(self, *, completed: bool, include_encrypted_content: bool) -> JsonObject:
        """Return the current official Responses reasoning item."""
        item: JsonObject = {
            "id": self.item_id,
            "type": "reasoning",
            "summary": (
                [{"type": "summary_text", "text": text} for _, text in sorted(self.parts.items())]
                if completed
                else []
            ),
            "status": "completed" if completed else "in_progress",
        }
        if include_encrypted_content and self.encrypted_content is not None:
            item["encrypted_content"] = self.encrypted_content
        return item
