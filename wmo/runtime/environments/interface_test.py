"""Tests for the simulator-owned executable-environment interfaces."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType

from wmo.common.models import ToolCall
from wmo.common.tasks import TaskCase
from wmo.runtime.environments.interface import EnvironmentRuntime, EnvironmentSession, Observation


def test_execute_only_session_and_runtime_protocols_accept_conforming_fakes() -> None:
    runtime = _Runtime()
    session = _Session()

    assert isinstance(runtime, EnvironmentRuntime)
    assert isinstance(session, EnvironmentSession)
    assert session.execute(ToolCall(call_id="call-1", name="lookup")) == Observation(content="ok")


def test_observation_round_trips_with_structured_metadata() -> None:
    observation = Observation(content="tool output", metadata={"exit_code": 0})

    assert Observation.model_validate_json(observation.model_dump_json()) == observation


class _Session:
    """A fake session exposing only the one canonical executable operation."""

    def execute(self, action: ToolCall) -> Observation:
        return Observation(content="ok")


class _Runtime:
    """A fake simulator owner that opens a clean executable session."""

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        return _Context(_Session())


class _Context(AbstractContextManager[EnvironmentSession]):
    """A basic cleanup-owning context used to prove the protocol boundary."""

    def __init__(self, session: EnvironmentSession) -> None:
        self._session = session

    def __enter__(self) -> EnvironmentSession:
        return self._session

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False
