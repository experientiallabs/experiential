"""Choose the native decision model that recalls simulator memory, from the model catalog."""

from exp.common.models import ModelCatalog

# Preferred native decision models for memory recall, best first.
RECALL_PREFERENCE = ("clef-flash", "clef", "jev")


DECISION_PROVIDER = "typesafe"


def resolve_memory_recall(catalog: ModelCatalog) -> str | None:
    """Choose the decision alias that recalls simulator memory, without a flag.

    The ``memory_recall`` role wins when set. Otherwise the catalog's SystemOne decision models
    are ranked by ``RECALL_PREFERENCE`` (Clef Flash first), then by alias.

    Args:
        catalog: The project's model catalog.

    Returns:
        The alias, or None when the catalog has no native decision model.

    Raises:
        ValueError: The ``memory_recall`` role names an alias that is not a decision model.
    """

    def decision(alias: str) -> bool:
        record = catalog.models.get(alias)
        connection = None if record is None else catalog.connections.get(record.connection)
        return connection is not None and connection.provider == DECISION_PROVIDER

    chosen = catalog.roles.memory_recall
    if chosen is not None:
        if not decision(chosen):
            raise ValueError(
                f"memory_recall role {chosen!r} is not a native decision model "
                f"({DECISION_PROVIDER} connection)"
            )
        return chosen

    def rank(alias: str) -> tuple[int, str]:
        model = catalog.models[alias].model.casefold()
        hits = [i for i, name in enumerate(RECALL_PREFERENCE) if name in model]
        return (min(hits) if hits else len(RECALL_PREFERENCE), alias)

    candidates = sorted((a for a in catalog.models if decision(a)), key=rank)
    return candidates[0] if candidates else None
