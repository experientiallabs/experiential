"""Tests for the waterfall Adapter seam: the missing-SDK error users actually have to act on.

The `Adapter` protocol itself has nothing to assert: no production code does
`isinstance(x, Adapter)`, so a conformance test here would only restate the declaration and pass
for reasons unrelated to whether the four rungs work. Each rung is covered by its own sibling
suite (`anthropic_test.py`, `azure_openai_test.py`, `bedrock_test.py`, `openai_test.py`), and `ty`
is what holds them to the protocol.
"""

from __future__ import annotations

from wmo.common.vendor.waterfall.adapters.base import missing_sdk_error


def test_missing_sdk_error_names_the_package_and_the_reinstall() -> None:
    # The SDKs are core dependencies, so this error means a partial environment: it must point at
    # `uv sync`, not at an extra the user would look for and not find.
    error = missing_sdk_error("boto3")

    assert isinstance(error, ModuleNotFoundError)
    message = str(error)
    assert "'boto3'" in message
    assert "uv sync" in message
    assert "world-model-optimizer" in message
