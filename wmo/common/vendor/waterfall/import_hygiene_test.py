"""The package must import and construct with zero provider SDKs installed."""

from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Mapping, Sequence

import pytest

import wmo.common.vendor
import wmo.common.vendor.waterfall  # binds the attribute this test has to put back (see below)

_SDK_ROOTS = ("boto3", "botocore", "openai", "anthropic")
_PACKAGE = "wmo.common.vendor.waterfall"


def test_import_and_construct_without_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a bare environment: any SDK import raises, including cached package modules.
    # Otherwise an earlier-collected test's import makes this a cache hit that would mask a
    # top-level SDK import sneaking into the package.
    # Re-importing this package below rebinds the `waterfall` attribute on the parent
    # `wmo.common.vendor` module to a fresh module object, and monkeypatch's sys.modules undo
    # restores sys.modules WITHOUT restoring that attribute. Record it here so teardown puts the
    # original back: a later test's monkeypatch of the waterfall sleep hook resolves through the
    # parent attribute and would otherwise patch a module that nothing
    # under test is running (sleeps then happen for real).
    monkeypatch.setattr(wmo.common.vendor, "waterfall", wmo.common.vendor.waterfall)
    for name in list(sys.modules):
        if name.split(".")[0] in _SDK_ROOTS or name.startswith(_PACKAGE):
            monkeypatch.delitem(sys.modules, name)
    real_import = builtins.__import__

    def blocking_import(
        name: str,
        globals: Mapping[str, object] | None = None,  # noqa: A002 - __import__ signature
        locals: Mapping[str, object] | None = None,  # noqa: A002 - __import__ signature
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> types.ModuleType:
        if name.split(".")[0] in _SDK_ROOTS:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    from wmo.common.vendor.waterfall import Backend, Waterfall

    wf = Waterfall(
        [
            Backend("bedrock", "us.anthropic.claude-opus-4-8", profile="p", region="us-west-2"),
            Backend("openai", "gpt-5.5"),
            Backend("anthropic", "claude-opus-4-8"),
        ]
    )
    assert len(wf.backends) == 3

    # First actual call must fail with a clear "reinstall the environment" message, not a
    # bare import.
    with pytest.raises(ModuleNotFoundError, match=r"'boto3' package is required"):
        wf.complete(system="", messages=[{"role": "user", "content": "hi"}])
