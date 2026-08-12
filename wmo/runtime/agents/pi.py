"""Thin adapter from an installed Pi bridge to the WMO agent contract."""

from __future__ import annotations

from collections.abc import Callable
from shutil import which

from pydantic import ValidationError

from wmo.common.core.artifacts import JsonObject
from wmo.common.tasks import TaskCase
from wmo.runtime.agents.interface import AgentEpisode
from wmo.runtime.environments import EnvironmentSession

PiTranscriptRunner = Callable[[TaskCase, object, EnvironmentSession], JsonObject]
"""Converts one installed Pi run into a JSON-compatible WMO episode transcript."""


class PiRuntimePreflightError(RuntimeError):
    """An installed Pi executable or its WMO bridge is unavailable."""


class PiTranscriptError(ValueError):
    """An installed Pi bridge emitted a transcript outside the WMO episode contract."""


class PiAgentRuntime:
    """Runs installed Pi through a narrow model-injected transcript bridge.

    Args:
        executable: Name or path of the externally installed Pi executable.
        transcript_runner: Adapter that invokes installed Pi and returns its transcript as JSON.
            It receives WMO's injected model and execute-only environment. Tests can supply a
            deterministic runner without installing or invoking Pi.
    """

    def __init__(
        self,
        *,
        executable: str = "pi",
        transcript_runner: PiTranscriptRunner | None = None,
    ) -> None:
        if not executable:
            raise ValueError("Pi executable must be a non-empty command or path")
        self._executable = executable
        self._transcript_runner = transcript_runner

    def preflight(self) -> None:
        """Confirm that WMO can call an installed Pi integration without vendored source.

        Raises:
            PiRuntimePreflightError: No deterministic bridge is configured for the installed Pi
                executable, or the executable cannot be found.
        """
        if self._transcript_runner is not None:
            return
        executable_path = which(self._executable)
        if executable_path is None:
            raise PiRuntimePreflightError(
                "PiAgentRuntime could not find an installed Pi executable named "
                f"{self._executable!r}. Install Pi outside WMO, then configure a transcript_runner "
                "that accepts WMO's injected model and execute-only environment. WMO does not "
                "ship Pi source."
            )
        raise PiRuntimePreflightError(
            "PiAgentRuntime found an installed Pi executable at "
            f"{executable_path!r}, but no transcript_runner bridge is configured. Add a bridge "
            "that accepts WMO's injected model and execute-only environment; WMO intentionally "
            "does not vendor Pi source."
        )

    def run(
        self,
        task: TaskCase,
        *,
        model: object,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Run Pi with injected dependencies and normalize its transcript to an episode.

        Args:
            task: Task and tool schemas for the installed Pi process.
            model: Candidate model supplied by WMO's pending common model-client contract.
            environment: Execute-only session supplied by the simulator.

        Returns:
            The canonical in-memory episode reconstructed from Pi's transcript.

        Raises:
            PiRuntimePreflightError: Pi cannot be invoked through the configured bridge.
            PiTranscriptError: The bridge returned an invalid episode transcript.
        """
        self.preflight()
        runner = self._transcript_runner
        if runner is None:
            raise AssertionError("Pi preflight returned without a transcript runner")
        transcript = runner(task, model, environment)
        try:
            return AgentEpisode.model_validate(transcript)
        except ValidationError as exc:
            raise PiTranscriptError(
                "Pi transcript does not satisfy the WMO AgentEpisode contract. "
                "Update the installed Pi bridge to emit ordered events, terminal state, and any "
                "structured failure."
            ) from exc
