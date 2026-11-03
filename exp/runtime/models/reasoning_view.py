"""Per-call reasoning views of a frozen catalog that keep every other binding."""

from __future__ import annotations

from exp.common.models import ModelCapabilities, ModelCatalog, ReasoningEffort
from exp.runtime.models.registry import ModelConnectionError


def reasoning_catalog(
    catalog: ModelCatalog, alias: str, effort: ReasoningEffort | None
) -> ModelCatalog:
    """Return a pure per-call reasoning view without changing catalog bindings or defaults.

    Args:
        catalog: Original credential-free catalog.
        alias: Exact alias whose per-call profile may change.
        effort: Explicit override, or None to return the original catalog unchanged.

    Returns:
        A local catalog clone, or the same object for no override or an already-off model.

    Raises:
        ModelConnectionError: The alias is absent.
        ValueError: The capability or deployment does not support the requested effort.
    """
    if effort is None:
        return catalog
    record = catalog.models.get(alias)
    if record is None:
        raise ModelConnectionError(f"unknown model alias {alias!r}")
    capability = record.capabilities or ModelCapabilities()
    if not capability.supports_reasoning:
        if effort == "none":
            return catalog
        raise ValueError(f"model {alias!r} does not support a reasoning effort override")
    supported = record.supported_reasoning_efforts or ()
    if effort not in supported:
        raise ValueError(
            f"reasoning effort {effort!r} is not supported by {alias}; "
            f"supported efforts: {', '.join(supported) or 'none declared'}"
        )
    return catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                alias: record.model_copy(
                    update={
                        "capabilities": capability.model_copy(update={"reasoning_effort": effort}),
                    }
                ),
            }
        }
    )
