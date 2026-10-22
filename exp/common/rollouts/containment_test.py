"""Tests for the persisted containment guarantee, level, and report contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.rollouts.containment import (
    ContainmentGuarantee,
    ContainmentLevel,
    ContainmentReport,
    UnavailableGuarantee,
    canonical_guarantees,
    containment_report,
)

_NETWORK = ContainmentGuarantee.NETWORK_DENIED
_CLEANUP = ContainmentGuarantee.VERIFIED_CLEANUP


def test_level_requires_canonical_order_and_reports_missing_requirements() -> None:
    """Equal levels digest equally, and requirements resolve to the unmet subset."""
    level = ContainmentLevel(protocol="p-v1", guarantees=canonical_guarantees((_NETWORK, _CLEANUP)))

    assert level.missing((_CLEANUP,)) == ()
    assert level.missing((ContainmentGuarantee.MEMORY_LIMIT, _CLEANUP)) == (
        ContainmentGuarantee.MEMORY_LIMIT,
    )
    with pytest.raises(ValidationError, match="sorted and unique"):
        ContainmentLevel(protocol="p-v1", guarantees=(_CLEANUP, _NETWORK))


def test_report_builder_explains_every_unenforced_guarantee() -> None:
    """Each guarantee is either enforced or carries a reason, never both."""
    report = containment_report(
        platform="linux",
        protocol="p-v1",
        enforced=(_CLEANUP,),
        reasons={
            guarantee: "absent" for guarantee in ContainmentGuarantee if guarantee != _CLEANUP
        },
        mechanisms=("b", "a", "a"),
    )

    assert report.level.guarantees == (_CLEANUP,)
    assert report.mechanisms == ("a", "b")
    assert len(report.unavailable) == len(ContainmentGuarantee) - 1
    with pytest.raises(ValueError, match="needs a reason"):
        containment_report(
            platform="linux", protocol="p-v1", enforced=(), reasons={}, mechanisms=()
        )


def test_report_rejects_overlapping_incomplete_or_refused_enforcement() -> None:
    """A report cannot both enforce and explain a guarantee or enforce under refusal."""
    level = ContainmentLevel(protocol="p-v1", guarantees=(_CLEANUP,))
    complete = tuple(
        UnavailableGuarantee(guarantee=guarantee, reason="absent")
        for guarantee in ContainmentGuarantee
        if guarantee != _CLEANUP
    )
    with pytest.raises(ValidationError, match="both enforced and unavailable"):
        ContainmentReport(
            platform="linux",
            level=level,
            mechanisms=(),
            unavailable=(*complete, UnavailableGuarantee(guarantee=_CLEANUP, reason="x")),
        )
    with pytest.raises(ValidationError, match="account for every guarantee"):
        ContainmentReport(platform="linux", level=level, mechanisms=(), unavailable=complete[1:])
    with pytest.raises(ValidationError, match="refused containment report"):
        ContainmentReport(
            platform="linux", level=level, mechanisms=(), unavailable=complete, refusal="no"
        )
