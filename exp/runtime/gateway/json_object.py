"""Shared JSON-object output instruction for provider wires and token reservations."""

JSON_OBJECT_SYSTEM_INSTRUCTION = (
    "Output format: the caller will pass your entire reply to a strict JSON "
    "parser, so it must be exactly one raw JSON object. Begin the reply with "
    "'{' as the very first character and end it with '}' as the very last. "
    "Do not begin with ```json or any code fence, do not use markdown, and do "
    "not add any words before or after the object. A reply that starts with "
    "anything other than '{' is a failure."
)
"""System instruction specifying the schema-free JSON-object response contract."""
