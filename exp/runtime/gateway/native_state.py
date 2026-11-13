"""Typed admission access to registries owned by the native request engine."""

from __future__ import annotations

import json
from collections.abc import Sequence

from exp_gateway_native import NativeAccounting
from pydantic import TypeAdapter

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayFailure
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry


class NativeState:
    """Validate values returned by the engine's composition-only state interface."""

    def __init__(self, native: NativeAccounting) -> None:
        self.native = native

    def read[T](self, result_type: type[T], operation: str, argument: JsonObject) -> T:
        """Decode one strictly typed view without invoking a request callback."""
        return TypeAdapter(result_type).validate_json(
            self.native.state_json(operation, json.dumps(argument))
        )

    def write(self, operation: str, argument: JsonObject) -> None:
        """Mutate the engine's shared registry during initial admission."""
        self.native.state_json(operation, json.dumps(argument))


class NativeHealth(DeploymentHealthRegistry):
    """The admission and synchronous-ledger views of native lane health."""

    def __init__(self, state: NativeState) -> None:
        self.state = state

    def order(self, keys: tuple[DeploymentHealthKey, ...], candidates: Sequence[int]) -> list[int]:
        """Order candidates using the circuits native dispatch observes."""
        return self.state.read(
            list[int],
            "health_order",
            {"keys": [list(key) for key in keys], "depths": list(candidates)},
        )

    def lane_ejected(self, key: DeploymentHealthKey) -> bool:
        """Read the physical lane circuit across catalog revisions."""
        return self.state.read(bool, "health_ejected", {"key": list(key)})

    def claim(self, key: DeploymentHealthKey) -> bool:
        """Claim the normal circuit probe before admission."""
        return self.state.read(bool, "health_claim", {"key": list(key), "mode": "normal"})

    def claim_last_resort(self, key: DeploymentHealthKey) -> bool:
        """Claim a probe after ordinary eligible candidates are exhausted."""
        return self.state.read(bool, "health_claim", {"key": list(key), "mode": "last_resort"})

    def claim_forced(self, key: DeploymentHealthKey) -> bool:
        """Claim the policy-authorized overflow probe."""
        return self.state.read(bool, "health_claim", {"key": list(key), "mode": "forced"})

    def claim_throttle_redial(self, key: DeploymentHealthKey) -> bool:
        """Claim the separately authorized warm-lane redial."""
        return self.state.read(bool, "health_claim", {"key": list(key), "mode": "throttle_redial"})

    def dispatch_opened(self, key: DeploymentHealthKey) -> None:
        """Publish reachability as soon as the provider opens."""
        self.succeeded(key)

    def succeeded(self, key: DeploymentHealthKey) -> None:
        """Clear the deployment and physical-lane failure circuits."""
        self.state.write("health_succeeded", {"key": list(key)})

    def failed(self, key: DeploymentHealthKey, failure: GatewayFailure) -> None:
        """Apply one observed failure to the native health registry."""
        self.state.write(
            "health_failed", {"key": list(key), "failure": failure.model_dump(mode="json")}
        )

    def exhausted(self, key: DeploymentHealthKey, reset_after_seconds: float) -> None:
        """Retain the provider plan reset window."""
        self.state.write(
            "health_exhausted", {"key": list(key), "reset_seconds": reset_after_seconds}
        )

    def suppressed(self, key: DeploymentHealthKey) -> bool:
        """Read whether initial admission should avoid this deployment."""
        return self.state.read(bool, "health_suppressed", {"key": list(key)})

    def throttled_remaining_seconds(
        self, keys: tuple[DeploymentHealthKey, ...], depths: Sequence[int] | None = None
    ) -> float | None:
        """Read the longest remaining throttle window when every judged deployment is inside one.

        Args:
            keys: One health key per ordered route deployment.
            depths: The route indexes this reservation could claim; ``None``
                judges every deployment.

        Returns:
            The longest remaining window in seconds, or ``None`` when no
            depth is judged or any judged deployment is outside a throttle window.
        """
        judged = keys if depths is None else tuple(keys[depth] for depth in depths)
        return TypeAdapter(float | None).validate_json(
            self.state.native.state_json("health_throttled_remaining", json.dumps({"keys": judged}))
        )

    def circuit_failure(
        self, keys: tuple[DeploymentHealthKey, ...], depths: Sequence[int]
    ) -> GatewayFailure | None:
        """Read the account failure holding every claimable depth's circuit open.

        Args:
            keys: One health key per ordered route deployment.
            depths: The route indexes this reservation could claim.

        Returns:
            The first depth's recorded account failure when every depth's circuit
            refuses a claim and runs no newer throttle window, else ``None``.
        """
        judged = [keys[depth] for depth in depths]
        return TypeAdapter(GatewayFailure | None).validate_json(
            self.state.native.state_json("health_circuit_failure", json.dumps({"keys": judged}))
        )

    def release_probe(self, key: DeploymentHealthKey) -> None:
        """Return an unconsumed health probe."""
        self.state.write("health_release", {"key": list(key)})


class NativeSticky(StickySpillRegistry):
    """Bounded affinity shared by native dispatch and initial admission."""

    def __init__(self, state: NativeState) -> None:
        self.state = state

    def bind(self, fingerprint: bytes, deployment_id: str, *, ttl_seconds: float) -> None:
        """Retain the admitted conversation on its selected deployment."""
        self.state.write(
            "sticky_bind",
            {
                "fingerprint": fingerprint.hex(),
                "deployment_id": deployment_id,
                "ttl_seconds": ttl_seconds,
            },
        )

    def bound_deployment(self, fingerprint: bytes) -> str | None:
        """Read the affinity binding without extending its lifetime."""
        return TypeAdapter(str | None).validate_json(
            self.state.native.state_json(
                "sticky_bound", json.dumps({"fingerprint": fingerprint.hex()})
            )
        )

    def clear(self, fingerprint: bytes) -> None:
        """Discard one conversation binding."""
        self.state.write("sticky_clear", {"fingerprint": fingerprint.hex()})

    def size(self) -> int:
        """Report the native registry size."""
        return self.state.read(int, "sticky_len", {})
