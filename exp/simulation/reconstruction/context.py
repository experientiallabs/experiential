"""Source-authority views shared by construction and simulator prompt assembly."""

from exp.common.core.artifacts import JsonObject


def prior_records(context: JsonObject, kinds: tuple[str, ...]) -> list[JsonObject]:
    """Partition selected originals, generated behavior and investigations.

    Args:
        context: Exact selection from the immutable library resolver.
        kinds: Source categories included in this particular prompt channel.

    Returns:
        Selected payloads with authority metadata. Generated behavior excludes nested
        raw originals, research and evaluation bookkeeping. No source becomes native truth.
    """
    selected = context.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError("prior library selection must contain a record list")
    records: list[JsonObject] = []
    for record in selected:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("entry"), dict)
            or "payload" not in record
        ):
            raise ValueError("prior library record requires typed entry metadata and a payload")
        entry = record["entry"]
        assert isinstance(entry, dict)
        if entry.get("kind") in kinds:
            if entry.get("kind") in ("rule", "skill") and entry.get("disposition") != "accepted":
                continue
            if entry.get("kind") in ("rule", "skill"):
                payload = record["payload"]
                if isinstance(payload, dict):
                    knowledge = payload.get("knowledge", payload)
                    if not isinstance(knowledge, dict):
                        raise ValueError("prior behavior must contain a structured knowledge view")
                    payload = {
                        key: knowledge[key]
                        for key in (
                            "rules",
                            "rule",
                            "text_rules",
                            "skill",
                            "python_source",
                            "contracts",
                            "interface",
                            "claims",
                            "unknowns",
                            "source_trace_ids",
                            "prediction_guidance",
                        )
                        if key in knowledge
                    }
                records.append({**record, "payload": payload})
            else:
                records.append(record)
    return records
