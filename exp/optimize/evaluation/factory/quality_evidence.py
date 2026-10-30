"""Lossless shared-value framing for complete, bounded task-quality evidence.

Only exact repeated JSON values are shared. Every original object and array is tagged, so
untrusted data resembling a reference cannot become one. Source artifact citations remain
visible outside the encoded tree; the decoded original digest binds all original values.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    canonical_json_bytes,
    sha256_json,
)

EVIDENCE_PROTOCOL = "task-quality-evidence-v1"
MINIMUM_SHARED_BYTES = 256
MAXIMUM_ORIGINAL_BYTES = 8_000_000
MAXIMUM_EXPANDED_VALUES = 200_000
MAXIMUM_DEPTH = 128
MAXIMUM_SHARED_VALUES = 10_000


class QualityEvidence(ContractModel):
    """A complete original evidence object represented with exact shared JSON subtrees.

    Attributes:
        protocol: Version of the tagged representation and bounded decoder.
        original_sha256: Digest of the original canonical JSON, before sharing.
        original_bytes: Exact original canonical size; sharing never truncates this object.
        sources: Verbatim original source-citation value, visible without expansion.
        root: Tagged original object; primitives are literal, objects/arrays explicit.
        shared_values: Content-addressed tagged definitions, all reachable from root.
    """

    protocol: Literal["task-quality-evidence-v1"] = EVIDENCE_PROTOCOL
    original_sha256: Sha256
    original_bytes: int = Field(ge=2, le=MAXIMUM_ORIGINAL_BYTES)
    sources: JsonValue
    root: JsonValue
    shared_values: dict[str, JsonValue] = Field(max_length=MAXIMUM_SHARED_VALUES)


@dataclass
class _Budget:
    """Bound actual reconstruction rather than trusting an advertised size."""

    values: int = 0
    size: int = 0

    def visit(self, depth: int) -> None:
        """Refuse excessive depth or repeated-reference expansion before allocating children."""
        self.values += 1
        if depth > MAXIMUM_DEPTH or self.values > MAXIMUM_EXPANDED_VALUES:
            raise ValueError("quality evidence exceeds its depth or expanded-value bound")

    def add(self, size: int) -> None:
        """Charge canonical bytes as values are reconstructed."""
        self.size += size
        if self.size > MAXIMUM_ORIGINAL_BYTES:
            raise ValueError("quality evidence exceeds its expanded-byte bound")


def compact_quality_evidence(original: JsonObject) -> QualityEvidence:
    """Encode every original value deterministically; reject excessive inputs, never omit them."""
    counts: Counter[Sha256] = Counter()
    budget = _Budget()

    def inventory(value: JsonValue, depth: int) -> None:
        """Validate original depth before canonicalizing it and count exact repeated values."""
        budget.visit(depth)
        if isinstance(value, dict):
            for child in value.values():
                inventory(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                inventory(child, depth + 1)
        encoded = canonical_json_bytes(value)
        if len(encoded) > MAXIMUM_ORIGINAL_BYTES:
            raise ValueError("quality evidence exceeds its original-byte bound")
        if isinstance(value, (dict, list, str)) and len(encoded) >= MINIMUM_SHARED_BYTES:
            counts[sha256_json(value)] += 1

    inventory(original, 0)
    canonical = canonical_json_bytes(original)
    shared: dict[str, JsonValue] = {}

    def children(value: JsonValue) -> JsonValue:
        """Tag containers so original reference-shaped dictionaries remain ordinary data."""
        if isinstance(value, dict):
            return {"object": [[key, encode(value[key])] for key in sorted(value)]}
        if isinstance(value, list):
            return {"array": [encode(child) for child in value]}
        return value

    def encode(value: JsonValue) -> JsonValue:
        """Replace an exactly repeated subtree with its content address, once inventoried."""
        digest = sha256_json(value)
        if counts[digest] > 1:
            if digest not in shared:
                shared[digest] = children(value)
            return {"ref": digest}
        return children(value)

    root = encode(original)
    if len(shared) > MAXIMUM_SHARED_VALUES:
        raise ValueError("quality evidence exceeds its shared-value bound")
    return QualityEvidence(
        original_sha256=sha256_json(original),
        original_bytes=len(canonical),
        sources=original.get("sources"),
        root=root,
        shared_values=shared,
    )


def expand_quality_evidence(evidence: QualityEvidence) -> JsonObject:
    """Verify and reconstruct exact evidence, refusing cycles, hidden nodes and expansion bombs."""
    budget = _Budget()
    used: set[str] = set()

    def restore(node: JsonValue, active: frozenset[str], depth: int) -> JsonValue:
        """Decode one tagged value within the shared graph's actual expansion budget."""
        if isinstance(node, dict) and set(node) == {"ref"}:
            key = node["ref"]
            if (
                not isinstance(key, str)
                or key not in evidence.shared_values
                or key in active
                or len(active) > 2 * MAXIMUM_DEPTH
            ):
                raise ValueError("quality evidence has an unknown, cyclic or too-deep reference")
            used.add(key)
            value = restore(evidence.shared_values[key], active | {key}, depth)
            if sha256_json(value) != key:
                raise ValueError("quality evidence shared-value digest differs")
            return value
        budget.visit(depth)
        if not isinstance(node, (dict, list)):
            budget.add(len(canonical_json_bytes(node)))
            return node
        if not isinstance(node, dict) or len(node) != 1:
            raise ValueError("quality evidence requires one explicit container or reference tag")
        if "array" in node:
            items = node["array"]
            if not isinstance(items, list):
                raise ValueError("quality evidence array tag requires a list")
            budget.add(2 + max(0, len(items) - 1))
            return [restore(child, active, depth + 1) for child in items]
        if "object" in node:
            entries = node["object"]
            if not isinstance(entries, list):
                raise ValueError("quality evidence object tag requires ordered key/value entries")
            budget.add(2 + max(0, len(entries) - 1))
            result: JsonObject = {}
            for entry in entries:
                if (
                    not isinstance(entry, list)
                    or len(entry) != 2
                    or not isinstance(entry[0], str)
                    or entry[0] in result
                ):
                    raise ValueError("quality evidence has invalid or duplicate object keys")
                key = entry[0]
                budget.add(len(canonical_json_bytes(key)) + 1)
                result[key] = restore(entry[1], active, depth + 1)
            return result
        raise ValueError("quality evidence has an unknown representation tag")

    original = restore(evidence.root, frozenset(), 0)
    if not isinstance(original, dict):
        raise ValueError("quality evidence root must be an object")
    if used != set(evidence.shared_values):
        raise ValueError("quality evidence contains unreachable shared values")
    if (
        len(canonical_json_bytes(original)) != evidence.original_bytes
        or sha256_json(original) != evidence.original_sha256
        or original.get("sources") != evidence.sources
    ):
        raise ValueError("quality evidence original digest, size or visible sources differ")
    return original
