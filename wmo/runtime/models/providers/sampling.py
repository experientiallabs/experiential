"""Explicit sampling-parameter serialization from catalog capabilities.

Request builders consult these helpers instead of model-name heuristics or error-string retries.
Unsupported parameters are omitted from the wire payload. The typed WMO request is unchanged.
"""

from __future__ import annotations

from wmo.common.models import ModelCapabilities, ModelRequest


def include_temperature(
    request: ModelRequest,
    capabilities: ModelCapabilities | None,
) -> bool:
    """Return whether temperature may be serialized for this model.

    Args:
        request: Typed WMO request that may name a temperature.
        capabilities: Catalog capabilities for the target model, when known.

    Returns:
        ``True`` only when the request names a temperature and the model has not declared
        that sampling parameter unsupported.
    """
    if request.temperature is None:
        return False
    if capabilities is None or capabilities.supports_temperature is None:
        return True
    return capabilities.supports_temperature
