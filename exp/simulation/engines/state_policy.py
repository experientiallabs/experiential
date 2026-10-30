"""One executable write boundary for prompt construction and proposal admission."""

from jsonschema import Draft202012Validator
from pydantic import Field, StrictStr

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.simulation.engines.transition import ApplyEdits, ProposedTransition


class StateEditPolicy(ContractModel):
    """Permitted subtree roots, without authority to replace their ancestors.

    Attributes:
        roots: Exact object-key paths that may be edited, including descendants.
            An empty inventory permits only an empty edit list.
        read_roots: Additional visible roots permitted as copy sources. Writable roots are
            also readable. Copy cannot reference hidden domains or an ancestor of a root.
    """

    roots: tuple[tuple[StrictStr, ...], ...] = Field(default=())
    read_roots: tuple[tuple[StrictStr, ...], ...] = Field(default=())

    def edit_schema(self) -> JsonObject:
        """Constrain each structural edit to a declared root before generation."""
        if any(not root for root in (*self.roots, *self.read_roots)):
            raise ValueError("state edit roots must not grant whole-document replacement")
        if not self.roots:
            return {"type": "array", "maxItems": 0}
        paths: list[JsonObject] = [
            {
                "type": "array",
                "minItems": len(root),
                "prefixItems": [{"const": part} for part in root],
                "items": {"type": ["string", "integer"]},
            }
            for root in self.roots
        ]
        source_paths: list[JsonObject] = [
            {
                "type": "array",
                "minItems": len(root),
                "prefixItems": [{"const": part} for part in root],
                "items": {"type": ["string", "integer"]},
            }
            for root in (*self.roots, *self.read_roots)
        ]
        return {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op", "path"],
                "properties": {
                    "op": {"enum": ["set", "delete", "append", "remove_value", "copy"]},
                    "path": {"anyOf": paths},
                    "value": {},
                    "from": {"anyOf": source_paths},
                },
                "additionalProperties": False,
                "allOf": [
                    {
                        "if": {"properties": {"op": {"const": "copy"}}},
                        "then": {"required": ["from"], "not": {"required": ["value"]}},
                        "else": {"not": {"required": ["from"]}},
                    },
                    {
                        "if": {"properties": {"op": {"enum": ["set", "append", "remove_value"]}}},
                        "then": {"required": ["value"]},
                    },
                ],
            },
        }

    def require(self, proposal: ProposedTransition) -> None:
        """Refuse forbidden writes even when their net effect would be unchanged."""
        if not isinstance(proposal.effect, ApplyEdits):
            raise ValueError("executable prediction requires bounded edits, not state replacement")
        if not Draft202012Validator(self.edit_schema()).is_valid(proposal.effect.edits):
            raise ValueError(
                "prediction edits exceed the declared writable state paths; preserve the "
                "environment-owned logical clock and all other domains exactly"
            )
