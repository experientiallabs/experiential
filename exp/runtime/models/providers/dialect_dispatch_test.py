"""Inline tests for the dialect dispatch seam.

The request-shaping behaviour is exercised through ``streaming_requests_test``
and every payload-builder suite; this module pins the disclosure wording that
callers read off the wire.
"""

from __future__ import annotations

from exp.runtime.models.providers.dialect_dispatch import CACHE_CONTROL_NOT_FORWARDED_SUFFIX


def test_cache_control_disclosure_names_where_cache_reads_show_up() -> None:
    """The unforwarded-marker disclosure is a stable wire string that never reads as "ignored".

    It travels in ``x-experiential-ignored-parameters`` beside a billed
    ``cache_read_input_tokens`` on OpenAI-compatible routes, so it has to say
    that caching is the provider's decision and where any reads are reported.
    """
    assert CACHE_CONTROL_NOT_FORWARDED_SUFFIX == (
        "->not_forwarded(provider_decides_caching;"
        " cache reads reported in usage.cache_read_input_tokens)"
    )
    assert "ignored" not in CACHE_CONTROL_NOT_FORWARDED_SUFFIX
    assert "usage.cache_read_input_tokens" in CACHE_CONTROL_NOT_FORWARDED_SUFFIX
