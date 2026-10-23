"""Recognize item ids the OpenAI Responses API itself could have issued.

OpenAI validates a replayed item's ``id`` against its own format and refuses
any other ("Invalid 'input[50].id': 'msg_dffbfe65-bdcc-4ff9-a512-...'.
Expected an ID that begins with 'msg'.", 124 requests a day). Clients and
other gateways mint ids of their own (a ``msg_`` prefix over a UUID, an
``item_`` prefix) for history they replay, and the gateway must not forward
those to an OpenAI Responses rung. A foreign id is never needed there: the
provider resolves nothing by id under ``store=false``, so the item is sent
without it.
"""

from __future__ import annotations

OPENAI_ITEM_ID_PREFIXES = {"message": "msg", "function_call": "fc", "reasoning": "rs"}
"""The id prefix OpenAI issues for each replayable item type."""


def openai_issued_item_id(item_id: str, item_type: str) -> bool:
    """Whether ``item_id`` has the shape OpenAI issues for ``item_type``.

    OpenAI ids are ``<prefix>_`` followed by letters and digits only; the
    gateway's own synthetic ids (``msg_`` plus hex) share that shape. An item
    type without a known prefix (hosted tool calls) is never judged foreign
    here.

    Args:
        item_id: Replayed item id.
        item_type: Responses item type (``message``, ``function_call``, ...).

    Returns:
        False only when the type has a known prefix and the id does not match it.
    """
    prefix = OPENAI_ITEM_ID_PREFIXES.get(item_type)
    if prefix is None:
        return True
    head, separator, rest = item_id.partition("_")
    return (
        head == prefix
        and separator == "_"
        and bool(rest)
        and all(
            character.isascii() and (character.isalnum() or character == "_") for character in rest
        )
    )
