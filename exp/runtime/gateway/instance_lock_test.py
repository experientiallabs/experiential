"""One gateway owns a root at a time, including while its workers are torn down."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from exp.runtime.gateway import instance_lock
from exp.runtime.gateway.instance_lock import GatewayTeardownIncomplete, gateway_instance_lock
from exp.runtime.gateway.lifecycle_fields import GatewayLifecycleError


@pytest.fixture(autouse=True)
def _release_retained_locks() -> Iterator[None]:
    """Release locks a test deliberately left held, as process exit would."""
    yield
    while instance_lock._RETAINED_OWNER_LOCKS:
        instance_lock._RETAINED_OWNER_LOCKS.pop().release()


def test_instance_lock_rejects_a_second_owner_for_the_same_root(tmp_path: Path) -> None:
    """A second local process cannot concurrently own one gateway database."""
    with gateway_instance_lock(tmp_path, port=8000) as ownership:
        with pytest.raises(GatewayLifecycleError, match="already owns"):
            with gateway_instance_lock(tmp_path, port=9000):
                raise AssertionError("second lock unexpectedly acquired")
        ownership.teardown_complete()


def test_a_finished_teardown_releases_the_root_even_after_a_serving_error(
    tmp_path: Path,
) -> None:
    """A serving error whose teardown finished lets the next gateway start."""
    with pytest.raises(RuntimeError, match="serving failed"):
        with gateway_instance_lock(tmp_path, port=8000) as ownership:
            try:
                raise RuntimeError("serving failed")
            finally:
                ownership.teardown_complete()
    with gateway_instance_lock(tmp_path, port=9000) as ownership:
        ownership.teardown_complete()


@pytest.mark.parametrize(
    "failure",
    [GatewayTeardownIncomplete("drain timed out"), KeyboardInterrupt()],
    ids=["failed drain", "interrupted drain"],
)
def test_unfinished_teardown_keeps_the_root_owned(tmp_path: Path, failure: BaseException) -> None:
    """Workers that may still write keep their process the owner until it exits."""
    with pytest.raises(type(failure)):
        with gateway_instance_lock(tmp_path, port=8000):
            raise failure
    with pytest.raises(GatewayLifecycleError, match="already owns"):
        with gateway_instance_lock(tmp_path, port=9000):
            raise AssertionError("a second gateway started under live workers")
