"""Host timeout authority is finite, immutable, and separate from caller input."""

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.generation_timeouts import GatewayGenerationTimeouts
from exp.runtime.gateway.native_rungs_test import _AUTHORIZATION, _deployment, _dispatch, _route


@pytest.mark.parametrize("field", ["first_token_base_seconds", "progress_seconds"])
@pytest.mark.parametrize(
    "value", [0.0, -1.0, 3601.0, float("nan"), float("inf"), float("-inf"), True, "600"]
)
def test_invalid_wait_cannot_be_authorized(field: str, value: float | bool | str) -> None:
    """A host cannot publish an unbounded, negative, or zero wait."""
    with pytest.raises(ValidationError):
        GatewayGenerationTimeouts.model_validate(
            {"first_token_base_seconds": 600.0, "progress_seconds": 600.0, field: value}
        )


def test_host_waits_are_scoped_to_authorization_without_mutating_shared_routes() -> None:
    """Two orgs on the same deployments retain independent waits on every rung."""
    deployments = (_deployment("one", "openai"), _deployment("two", "openai"))
    baseline = _route(deployments)
    authorization = _AUTHORIZATION.model_copy(
        update={
            "organization_id": "long-wait-org",
            "generation_timeouts": GatewayGenerationTimeouts(
                first_token_base_seconds=600, progress_seconds=600
            ),
        }
    )
    scoped = baseline.model_copy(
        update={"snapshot": baseline.snapshot.model_copy(update={"authorization": authorization})}
    )
    for deployment in deployments:
        original = _dispatch(baseline, deployment).wire_entry
        configured = _dispatch(scoped, deployment).wire_entry
        assert configured == {
            **original,
            # Idempotency binds the organization, independently of timeout policy.
            "idempotency_key": configured["idempotency_key"],
            "timeout_seconds": 600.0,
            "time_to_first_token_base_seconds": 600.0,
        }
        assert original["timeout_seconds"] == 60.0
        assert original["time_to_first_token_base_seconds"] is None
        assert _dispatch(baseline, deployment).wire_entry == original
    assert scoped.snapshot.authorization.deadline_monotonic == _AUTHORIZATION.deadline_monotonic
    restored = type(authorization).model_validate_json(authorization.model_dump_json())
    assert restored.generation_timeouts == authorization.generation_timeouts
    with pytest.raises(ValidationError):
        restored.generation_timeouts = None
