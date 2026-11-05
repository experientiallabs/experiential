"""Draw-local, sequential delivery of exact evidence with explicit completion."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import ToolCall
from exp.common.tasks import ToolSchema

EVIDENCE_DELIVERY_PROTOCOL = "evidence-pages-v1"
PAGE_CHARACTERS = 8192
PAGE_INSTRUCTION = """
An unpaged successful read is complete. Paged reads use evidence-pages-v1. Each page contains
an exact chunk of one JSON-text document, its SHA256, Unicode-codepoint offset, total_characters
and next_cursor. Concatenate chunks in offset order
and JSON-decode only when complete=true; partial documents are not citable. Use
continue_evidence with exactly next_cursor. Citations are granted only on complete delivery.
A refused read serves no evidence. Narrow a request that cannot fit retained history; paging
never removes earlier messages or expands the frozen request limit."""
CONTINUE_EVIDENCE = ToolSchema(
    name="continue_evidence",
    description="Read the next exact page of this draw's pending evidence document.",
    input_schema={
        "type": "object",
        "properties": {"cursor": {"type": "string", "maxLength": 160}},
        "required": ["cursor"],
        "additionalProperties": False,
    },
)


@dataclass
class _Document:
    """One trusted result, kept unchanged until all sequential pages are delivered.

    Attributes:
        text: Exact compact JSON serialization of the original result.
        references: Trusted references, granted only on the final page.
        offset: Next required character offset.
    """

    text: str
    references: tuple[str, ...]
    offset: int = 0


class EvidencePages:
    """Deliver exact values without granting partial or undelivered references."""

    def __init__(self) -> None:
        """Start with no pending documents or cursors."""
        self.documents: dict[str, _Document] = {}

    @property
    def incomplete(self) -> bool:
        """Whether a started document is not completely delivered."""
        return any(doc.offset < len(doc.text) for doc in self.documents.values())

    def start(
        self,
        call: ToolCall,
        value: JsonValue,
        references: tuple[str, ...],
        accepts: Callable[[JsonValue, tuple[str, ...]], bool],
        *,
        maximum_pages: int = 4,
    ) -> tuple[Literal["ok", "refused"], JsonValue, tuple[str, ...]]:
        """Admit the complete raw value first, then deliver it or start exact pages.

        The complete-value check is a necessary capacity check, not a promise that unknown
        future assistant outputs and page envelopes will fit. Every page is checked again.
        """
        if not accepts(value, references):
            return (
                "refused",
                "complete result cannot fit cumulative serialized request history; narrow the read",
                (),
            )
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if (
            len(text) <= PAGE_CHARACTERS
            or (len(text) + PAGE_CHARACTERS - 1) // PAGE_CHARACTERS > maximum_pages
        ):
            return "ok", value, references
        identity = sha256_json(
            {"call": call.model_dump(mode="json"), "text": text, "references": list(references)}
        )
        if identity in self.documents:
            return "refused", "document already started; use its exact continuation cursor", ()
        self.documents[identity] = _Document(text, references)
        result = self._page(identity, accepts)
        if result[0] == "refused":
            del self.documents[identity]
        return result

    def resume(
        self,
        cursor: JsonValue,
        accepts: Callable[[JsonValue, tuple[str, ...]], bool],
    ) -> tuple[Literal["ok", "refused"], JsonValue, tuple[str, ...]]:
        """Deliver only the exact next cursor of a document owned by this draw."""
        if not isinstance(cursor, str):
            return "refused", "cursor must be an exact returned string", ()
        identity, separator, offset = cursor.partition(":")
        doc = self.documents.get(identity)
        if not separator or doc is None or offset != str(doc.offset) or doc.offset >= len(doc.text):
            return "refused", "unknown, foreign, completed or out-of-order evidence cursor", ()
        return self._page(identity, accepts)

    def _page(
        self,
        identity: str,
        accepts: Callable[[JsonValue, tuple[str, ...]], bool],
    ) -> tuple[Literal["ok", "refused"], JsonValue, tuple[str, ...]]:
        """Find a nonempty page fitting the exact current assembly, or refuse without progress."""
        doc = self.documents[identity]
        end = min(len(doc.text), doc.offset + PAGE_CHARACTERS)
        while end > doc.offset:
            complete = end == len(doc.text)
            refs = doc.references if complete else ()
            body: JsonObject = {
                "protocol": EVIDENCE_DELIVERY_PROTOCOL,
                "encoding": "json-text",
                "document_sha256": hashlib.sha256(doc.text.encode()).hexdigest(),
                "offset": doc.offset,
                "offset_unit": "unicode-codepoints",
                "total_characters": len(doc.text),
                "chunk": doc.text[doc.offset : end],
                "next_cursor": None if complete else f"{identity}:{end}",
                "complete": complete,
                "citations": list(refs),
            }
            if accepts(body, refs):
                doc.offset = end
                return "ok", body, refs
            end = doc.offset + (end - doc.offset) // 2
        return (
            "refused",
            "no page fits retained history; document remains incomplete and uncitable",
            (),
        )
