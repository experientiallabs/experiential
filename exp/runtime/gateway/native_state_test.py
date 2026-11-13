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


def test_throttle_window_judges_only_the_depths_a_reservation_may_dial() -> None:
    owner = NativeAccounting(background_recovery=False)
    health = NativeHealth(NativeState(owner))
    throttled, open_lane = (
        ("revision", "throttled", "connection"),
        ("revision", "open", "connection"),
    )
    health.failed(
        throttled,
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttle",
            retry_after_seconds=30,
        ),
    )
    keys = (throttled, open_lane)
    assert health.throttled_remaining_seconds(keys) is None
    remaining = health.throttled_remaining_seconds(keys, [0])
    assert remaining is not None
    assert 0 < remaining <= 30
    assert health.throttled_remaining_seconds(keys, [1]) is None
    assert health.throttled_remaining_seconds(keys, []) is None
    assert owner.close()


def test_hintless_probe_ownership_crosses_admission_views() -> None:
    """A probe's own successful open lifts the window observed by every native view."""
    owner = NativeAccounting(background_recovery=False)
    first, second = NativeHealth(NativeState(owner)), NativeHealth(NativeState(owner))
    key = ("revision", "deployment", "connection")
    first.failed(
        key,
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED, safe_message="provider throttle"
        ),
    )
    assert second.claim_throttle_probe(key)
    first.release_probe(key, "unrelated")
    second.bind_throttle_probe(key, "probe")
    first.succeeded(key, "older-request")
    assert second.suppressed(key)
    first.dispatch_opened(key, "probe")
    assert not second.suppressed(key)
    assert owner.close()


def test_account_failure_names_only_circuits_it_still_holds() -> None:
    """The native view names the account failure while it is the newest cause everywhere."""
    owner = NativeAccounting(background_recovery=False)
    health = NativeHealth(NativeState(owner))
    account, open_lane = (
        ("revision", "account", "connection"),
        ("revision", "open", "connection"),
    )
    rejected = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
        safe_message="provider rejected the configured credential",
        customer_owned=True,
    )
    health.failed(account, rejected)
    keys = (account, open_lane)
    assert health.circuit_failure(keys, [0]) == rejected
    # Every judged depth must be held open by an account failure.
    assert health.circuit_failure(keys, [0, 1]) is None
    assert health.circuit_failure(keys, []) is None
    # A throttle settled after the account failure is the current cause.
    health.failed(
        account,
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttle",
            retry_after_seconds=30,
        ),
    )
    assert health.circuit_failure(keys, [0]) is None
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
