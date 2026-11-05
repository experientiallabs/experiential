"""Lossless, model-facing compression of canonical evidence references."""

from collections.abc import Iterable
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject

LOCATOR_PROTOCOL = "locator-prefix-v1"

LOCATOR_GRAMMAR = f"""Inventory grammar {LOCATOR_PROTOCOL}: direct entries are complete canonical
references. Each trace_spans key is a complete trace:<trace_id> prefix; append ':span:' and
one listed suffix to obtain a canonical reference. Cite the reconstructed full reference,
never a suffix or an invented alias. The encoding preserves every reference, not a sample."""


class LocatorInventory(ContractModel):
    """An exactly reversible inventory of canonical references.

    Attributes:
        direct: Complete references that are not trace-span references.
        trace_spans: Full trace prefixes mapped to their exact span suffixes.
    """

    direct: tuple[str, ...] = ()
    trace_spans: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        """Reject ambiguous encodings and duplicate canonical references."""
        for prefix, spans in self.trace_spans.items():
            if not prefix.startswith("trace:") or not prefix[6:] or ":" in prefix[6:]:
                raise ValueError("trace prefix must be trace:<trace_id>")
            if not spans or any(not span or ":" in span for span in spans):
                raise ValueError("span suffixes must be nonempty canonical identifiers")
        decoded = self.decode()
        if len(decoded) != len(set(decoded)):
            raise ValueError("inventory contains duplicate canonical references")
        return self

    def decode(self) -> tuple[str, ...]:
        """Return all canonical references, without inventing aliases."""
        return tuple(
            sorted(
                (
                    *self.direct,
                    *(
                        f"{prefix}:span:{span}"
                        for prefix, spans in self.trace_spans.items()
                        for span in spans
                    ),
                )
            )
        )


def encode_inventory(references: Iterable[str]) -> JsonObject:
    """Encode an inventory without losing or duplicating any canonical reference."""
    refs = tuple(sorted(references))
    direct: list[str] = []
    spans: dict[str, list[str]] = {}
    for reference in refs:
        parts = reference.split(":")
        if len(parts) == 4 and parts[0] == "trace" and parts[2] == "span" and parts[1] and parts[3]:
            spans.setdefault(":".join(parts[:2]), []).append(parts[3])
        else:
            direct.append(reference)
    inventory = LocatorInventory(
        direct=tuple(direct), trace_spans={k: tuple(v) for k, v in spans.items()}
    )
    if inventory.decode() != refs:
        raise ValueError("inventory encoding changed canonical references")
    return inventory.model_dump(mode="json")
