"""Admission serialization for bounded native continuation ownership."""

from dataclasses import replace

from exp.runtime.gateway.native_continuation_registration import continuation_registration
from exp.runtime.gateway.native_responses_test import _context


def test_store_false_transfers_no_content_or_authority() -> None:
    """Caller storage opt-out prevents even native retention-state registration."""
    assert continuation_registration(None) is None
    assert continuation_registration(replace(_context(), retain=False)) is None
