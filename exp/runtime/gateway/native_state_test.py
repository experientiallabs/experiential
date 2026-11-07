"""Admission views observe the shared native registry, including expiry and reset."""

from exp_gateway_native import NativeAccounting

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_state import NativeHealth, NativeState, NativeSticky


def test_health_views_share_circuits_and_probe_ownership() -> None:
    owner = NativeAccounting(background_recovery=False)
    first, second = NativeHealth(NativeState(owner)), NativeHealth(NativeState(owner))
    key = ("revision", "deployment", "connection")
    first.failed(
        key,
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttle",
            retry_after_seconds=30,
        ),
    )
    assert second.suppressed(key)
    assert not second.claim(key)
    assert second.throttled_remaining_seconds((key,)) is not None
    first.dispatch_opened(key)
    # Reachability does not clear a still-active provider rate window.
    assert second.suppressed(key)
    assert second.claim_throttle_redial(key)
    failed = ("revision", "other-deployment", "connection")
    for _ in range(2):
        first.failed(
            failed,
            GatewayFailure(
                failure_class=GatewayFailureClass.TRANSPORT, safe_message="connection failed"
            ),
        )
    assert second.suppressed(failed)
    first.dispatch_opened(failed)
    assert not second.suppressed(failed)
    assert second.claim(failed)
    assert owner.close()


def test_affinity_views_share_binary_fingerprints_and_expiry() -> None:
    owner = NativeAccounting(background_recovery=False)
    first, second = NativeSticky(NativeState(owner)), NativeSticky(NativeState(owner))
    first.bind(b"\x00\xff", "lane", ttl_seconds=60)
    assert second.bound_deployment(b"\x00\xff") == "lane"
    second.clear(b"\x00\xff")
    assert first.bound_deployment(b"\x00\xff") is None
    assert first.size() == 0
    assert owner.close()
