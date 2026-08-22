"""Hosted Experiential Cloud connection for local CLI provider setup.

Experiential Cloud is a setup picker for the hosted Platform gateway. The
persisted catalog provider stays ``openai-compatible``: the CLI does not invent
a new runtime provider family, and it does not rebuild a local gateway
authority for this hosted path. Local ``exp run`` / ``exp config gateway``
workflows stay available for genuine offline serving.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

SETUP_PICKER_NAME = "experiential-cloud"
SETUP_PICKER_LABEL = "Experiential Cloud"
CATALOG_PROVIDER = "openai-compatible"
HOSTED_GATEWAY_DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
HOSTED_GATEWAY_API_KEY_ENV = "EXPLABS_API_KEY"
HOSTED_GATEWAY_URL_ENV = "EXP_GATEWAY_URL"


def hosted_gateway_base_url(environment: Mapping[str, str] | None = None) -> str:
    """Return the hosted Platform ``/v1`` origin for Experiential Cloud setup.

    Args:
        environment: Optional process environment. ``None`` reads ``os.environ``.

    Returns:
        ``EXP_GATEWAY_URL`` when that value is non-empty (preview or staging),
        otherwise the production Platform origin.
    """
    source: Mapping[str, str] = os.environ if environment is None else environment
    value = source.get(HOSTED_GATEWAY_URL_ENV, "").strip()
    return value or HOSTED_GATEWAY_DEFAULT_BASE_URL
