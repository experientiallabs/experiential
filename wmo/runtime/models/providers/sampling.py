"""Serialize optional sampling fields from catalog-declared request support.

Request builders consult this helper instead of model-name heuristics or error-string
retries. Unsupported fields are omitted from the wire payload. The typed WMO request is
unchanged. Adding a later sampling field means declaring it on ``SamplingSupport`` and
passing that field name here.
"""

from __future__ import annotations

from typing import Literal

from wmo.common.models import ModelCapabilities, ModelRequest

SamplingField = Literal["temperature"]


def include_sampling_field(
    request: ModelRequest,
    capabilities: ModelCapabilities | None,
    field: SamplingField,
) -> bool:
    """Return whether one optional sampling field may be serialized.

    Args:
        request: Typed WMO request that may name the field.
        capabilities: Catalog capabilities for the target model, when known.
        field: Sampling field declared on ``SamplingSupport``.

    Returns:
        ``True`` only when the request names the field and the catalog has not declared
        that field unsupported.
    """
    requested = getattr(request, field)
    if capabilities is None:
        return requested is not None
    return capabilities.sampling.include(field, requested)
