"""Tests for the route-bound execution snapshot's host presentation labels."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.native_execution_test import _route


def _with_labels(labels: dict[str, str]) -> ExecutionSnapshot:
    """Validate the three-rung test route's snapshot carrying ``labels``."""
    return ExecutionSnapshot.model_validate(
        {**_route().snapshot.model_dump(), "public_providers": labels}
    )


def test_public_providers_default_to_none() -> None:
    """A snapshot without labels names every rung by its provider id."""
    assert _route().snapshot.public_providers == {}


def test_public_providers_accept_one_header_safe_token_per_rung() -> None:
    """Labels are kept exactly, at most 64 bytes of letters, digits, dot, dash and underscore."""
    labels = {"one": "example_cloud", "two": "Example.Cloud-2", "three": "a" * 64}
    assert _with_labels(labels).public_providers == labels


@pytest.mark.parametrize("label", ["", "-leading", "two words", "line\nbreak", "café", "a" * 65])
def test_public_providers_refuse_a_label_that_is_not_one_header_token(label: str) -> None:
    """A label a header cannot carry exactly is refused at construction."""
    with pytest.raises(ValidationError):
        _with_labels({"one": label})


def test_public_providers_must_name_deployments_of_the_route() -> None:
    """A label for a deployment the route does not dispatch is refused."""
    with pytest.raises(ValidationError, match="public provider labels"):
        _with_labels({"elsewhere": "example_cloud"})
