"""Tests for the customer whole-episode agent contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import FailureAttribution, FailureCode, StructuredFailure
from wmo.common.models import ModelClient
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.agents.interface import (
    AgentAdapterPreflightError,
    AgentEpisode,
    AgentRuntime,
    preflight_agent_runtime,
)
from wmo.runtime.environments import EnvironmentSession


def test_episode_requires_ordered_unique_events_and_consistent_failure() -> None:
    first = _span("first", datetime(2026, 8, 11, tzinfo=UTC))
    second = _span("second", first.ended_at + timedelta(seconds=1))
    failure = StructuredFailure(
        code=FailureCode.PROVIDER,
        message="candidate model failed",
        attribution=FailureAttribution.MODEL,
    )

    episode = AgentEpisode(
        events=(first, second),
        stop_reason=StopReason.FAILURE,
        failure=failure,
    )

    assert episode.events == (first, second)
    with pytest.raises(ValidationError, match="unique"):
        AgentEpisode(events=(first, first), stop_reason=StopReason.COMPLETED)
    with pytest.raises(ValidationError, match="ordered"):
        AgentEpisode(events=(second, first), stop_reason=StopReason.COMPLETED)
    with pytest.raises(ValidationError, match="require a structured failure"):
        AgentEpisode(stop_reason=StopReason.FAILURE)
    with pytest.raises(ValidationError, match="only failed"):
        AgentEpisode(stop_reason=StopReason.COMPLETED, failure=failure)


def test_preflight_accepts_the_keyword_model_and_environment_seam() -> None:
    preflight_agent_runtime(_ConformingAgent())


def test_preflight_rejects_positional_only_model_injection() -> None:
    with pytest.raises(AgentAdapterPreflightError, match="Add keyword-addressable model"):
        preflight_agent_runtime(cast(AgentRuntime, _PositionalModelAgent()))


class _ConformingAgent:
    """Implements the canonical model-injection seam."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        return AgentEpisode(stop_reason=StopReason.COMPLETED)


class _PositionalModelAgent:
    """Makes model positional-only and therefore unavailable for injection."""

    def run(
        self,
        task: TaskCase,
        model: ModelClient,
        /,
        *,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        return AgentEpisode(stop_reason=StopReason.COMPLETED)


def _span(span_id: str, timestamp: datetime) -> RolloutSpan:
    return RolloutSpan(
        span_id=span_id,
        kind=RolloutEventKind.MESSAGE,
        started_at=timestamp,
        ended_at=timestamp,
        payload={"fixture": span_id},
    )
