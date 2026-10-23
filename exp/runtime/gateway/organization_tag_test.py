"""Tests for the opaque per-organization SystemOne scheduling tag."""

from __future__ import annotations

from exp.runtime.gateway.organization_tag import (
    ORGANIZATION_TAG_HEADER,
    organization_tag,
    organization_tag_key,
    tagged_headers,
)

_ORG = "244649dc-c49a-425a-873d-57079c50a706"


def test_tag_is_stable_opaque_and_credential_scoped() -> None:
    """One credential maps an organization to one tag that never contains its id."""
    key = organization_tag_key("provider-secret-canary")
    tag = organization_tag(key, _ORG)
    assert tag == organization_tag(key, _ORG)
    assert len(tag) == 32 and int(tag, 16) >= 0
    assert _ORG not in tag and "244649dc" not in tag
    assert tag != organization_tag(key, "5ca21e30-2453-458c-9118-3a0a5ea61890")
    assert tag != organization_tag(organization_tag_key("rotated-secret"), _ORG)
    assert b"provider-secret-canary" not in key


def test_tagged_headers_add_the_tag_only_on_house_rungs() -> None:
    """Customer credentials and tenant origins never receive a cross-request identifier."""
    key = organization_tag_key("provider-secret-canary")
    static = {"Authorization": "Bearer provider-secret-canary"}
    tagged = tagged_headers(static, key, _ORG, billing_customer_managed=False, tenant_origin=False)
    assert tagged == {**static, ORGANIZATION_TAG_HEADER: organization_tag(key, _ORG)}
    assert static == {"Authorization": "Bearer provider-secret-canary"}
    for managed, tenant in ((True, False), (False, True), (True, True)):
        assert (
            tagged_headers(
                static, key, _ORG, billing_customer_managed=managed, tenant_origin=tenant
            )
            is None
        )
    assert (
        tagged_headers(static, None, _ORG, billing_customer_managed=False, tenant_origin=False)
        is None
    )
