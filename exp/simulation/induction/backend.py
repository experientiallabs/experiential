"""An executable-world backend that runs induced programs and defers the rest to a predictor.

The backend is one simulated-side state owner in the world kernel. It composes accepted
induced programs (:class:`InducedCoverage`) with a predictor backend, normally the grounded
world model, that holds the one JSON state of every simulated domain and the visible call
history. Ownership inside the backend follows the kernel's rule that no tool spans owners:

- an induced tool executes as exact code while its call stays inside the program's validated
  domain (validated argument names and code paths, no refusal or failure); the reply and new
  state are adopted into the predictor's state and history, so later predictions see them;
- every other tool, and every call outside a program's domain, goes to the predictor;
- when the predictor executes a call that writes an induced tool's domains, the tool hands its
  domains to the predictor for the rest of the episode, closed so no tool spans owners again.
  Induced domains are therefore only ever written by validated code.

The handed-off partition is part of every captured image, so checkpoints, rollbacks, forks and
restores keep exactly the same induced part. The backend never claims native authority: every
reply it returns is simulated, labelled ``induced`` or ``predicted`` in its metadata.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Protocol, runtime_checkable

from exp.common.core.artifacts import (
    JsonObject,
    Sha256,
    canonical_json_bytes,
    sha256_json,
    strict_equal,
)
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import NativeImage, WorldAccess
from exp.runtime.environments.worlds import WorldBackend, WorldBackendFactory, WorldContext
from exp.simulation.induction.contracts import InducedCoverage
from exp.simulation.induction.runner import RUNNER_PROTOCOL
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.validation import induced_partition

# Identity of the code that decides this backend's behavior, in place of its source text.
# Bump it when behavior changes; exp/tests/code_protocols_test.py guards the covered source.
INDUCED_WORLD_PROTOCOL = "induced-world-v1"
_PART = "induced.json"


@runtime_checkable
class AdoptingBackend(WorldBackend, Protocol):
    """A predictor backend that also accepts exact transitions computed outside it."""

    def state_projection(self) -> JsonObject:
        """Return a detached copy of every simulated domain's current value."""
        ...

    def adopt(self, action: ToolCall, observation: Observation, state: JsonObject) -> None:
        """Commit an externally computed reply and complete next state into history and state."""
        ...


class InducedWorldFactory:
    """Create and restore induced-plus-predictor backends for one coverage and predictor.

    Attributes:
        coverage: Accepted programs, their validated domains and the induced partition.
        fallback: Predictor factory over the same world manifest; its backends hold the state.
        sandbox: Bounded executor, configured with the limits the programs were validated under.
        facts: Starting instance facts (key to value) that every program receives read-only.
    """

    def __init__(
        self,
        coverage: InducedCoverage,
        fallback: WorldBackendFactory,
        sandbox: ProgramSandbox,
        *,
        facts: JsonObject | None = None,
    ) -> None:
        """Bind coverage, predictor and executor without starting any process.

        Raises:
            ValueError: The sandbox limits differ from those the programs were validated under.
        """
        if sandbox.limits != coverage.limits:
            raise ValueError(
                "induced programs were validated under other sandbox limits; open the sandbox "
                "with the coverage's limits"
            )
        self.coverage = coverage
        self.fallback = fallback
        self.sandbox = sandbox
        self.facts = deepcopy(facts or {})
        self._definition: JsonObject = {
            "implementation": INDUCED_WORLD_PROTOCOL,
            "runner": RUNNER_PROTOCOL,
            "coverage": sha256_json(coverage),
            "facts": sha256_json(self.facts),
            "fallback": fallback.definition,
        }

    @property
    def definition(self) -> JsonObject:
        """Return the coverage, facts and predictor identity bound by checkpoints."""
        return self._definition

    def create(self, task: TaskCase) -> InducedWorldBackend:
        """Create a fresh backend in the task's initial state with the full induced partition."""
        return InducedWorldBackend(
            self, _adopting(self.fallback.create(task)), frozenset(self.coverage.induced_tools)
        )

    def restore(self, image: NativeImage) -> InducedWorldBackend:
        """Restore the predictor's image and the induced partition captured with it.

        Raises:
            ValueError: The image lacks the induced part or names tools the coverage lacks.
        """
        if _PART not in image.parts:
            raise ValueError("induced world image has no induced partition part")
        wire = json.loads(image.parts[_PART])
        listed = wire.get("induced_tools") if isinstance(wire, dict) else None
        tools = (
            frozenset(name for name in listed if isinstance(name, str))
            if isinstance(listed, list)
            else frozenset()
        )
        if (
            not isinstance(listed, list)
            or len(tools) != len(listed)
            or not tools <= set(self.coverage.induced_tools)
        ):
            raise ValueError("induced world image names tools outside its coverage")
        partition, _ = induced_partition(self.coverage.manifest, tools)
        if partition != tools:
            raise ValueError("induced world image partition lets a tool span owners")
        parts = {name: payload for name, payload in image.parts.items() if name != _PART}
        backend = self.fallback.restore(NativeImage(parts=parts, domains=image.domains))
        return InducedWorldBackend(self, _adopting(backend), tools)


class InducedWorldBackend:
    """One isolated episode state: induced programs first, then the predictor."""

    def __init__(
        self, factory: InducedWorldFactory, fallback: AdoptingBackend, induced: frozenset[str]
    ) -> None:
        """Own one predictor backend and the tools still executed as induced code."""
        self._factory = factory
        self._fallback = fallback
        self.induced_tools = induced

    def access(self, action: ToolCall) -> WorldAccess:
        """Resolve access through the predictor, which applies the same world manifest."""
        return self._fallback.access(action)

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Run an induced program inside its validated domain, otherwise the predictor."""
        if action.name in self.induced_tools:
            observation = self._induced(action)
            if observation is not None:
                return observation
        policy = self._factory.coverage.manifest.tool(action.name)
        observation = self._fallback.execute(action, context)
        if policy.writes:
            remaining = {
                tool
                for tool in self.induced_tools
                if tool != action.name
                and not self._factory.coverage.manifest.tool(tool).domains() & set(policy.writes)
            }
            self.induced_tools, _ = induced_partition(self._factory.coverage.manifest, remaining)
        return observation.model_copy(
            update={"metadata": {**observation.metadata, "transition": "predicted"}}
        )

    def domain_digests(self) -> dict[str, Sha256]:
        """Return the predictor-held digests of every simulated domain."""
        return self._fallback.domain_digests()

    def state_projection(self) -> JsonObject:
        """Expose detached domain state for the kernel's declared consistency checks."""
        return self._fallback.state_projection()

    def capture(self) -> NativeImage:
        """Capture the predictor's exact image plus the current induced partition."""
        image = self._fallback.capture()
        return NativeImage(
            parts={
                **image.parts,
                _PART: canonical_json_bytes({"induced_tools": sorted(self.induced_tools)}),
            },
            domains=image.domains,
        )

    def close(self) -> None:
        """Release the predictor backend; the shared sandbox stays with its owner."""
        self._fallback.close()

    def _induced(self, action: ToolCall) -> Observation | None:
        """Return the exact induced observation, or None when the call is outside the domain."""
        coverage = self._factory.coverage
        program = coverage.program(action.name)
        policy = coverage.manifest.tool(action.name)
        state = self._fallback.state_projection()
        visible = {name: deepcopy(state[name]) for name in sorted(policy.domains())}
        outcome = self._factory.sandbox.run(
            program, deepcopy(visible), action.arguments, self._factory.facts
        )
        after = outcome.state
        if (
            not coverage.domains[action.name].admits(action.arguments, outcome)
            or after is None
            or set(after) != set(visible)
            or any(
                not strict_equal(after[name], visible[name])
                for name in visible
                if name not in policy.writes
            )
        ):
            return None
        reply = outcome.reply
        observation = Observation(
            content=reply if isinstance(reply, str) else canonical_json_bytes(reply).decode(),
            is_error=outcome.status == "error_reply",
            metadata={
                "call_id": action.call_id,
                "authority": "simulated",
                "transition": "induced",
                "program_sha256": program.sha256,
            },
        )
        try:
            self._fallback.adopt(
                action, observation, {**state, **{name: after[name] for name in policy.writes}}
            )
        except ValueError:
            return None
        return observation


def _adopting(backend: WorldBackend) -> AdoptingBackend:
    """Require a predictor backend that can adopt exact transitions."""
    if not isinstance(backend, AdoptingBackend):
        raise ValueError(
            "the induced world's predictor backend must expose state_projection and adopt; "
            "use the grounded world model or another adopting backend"
        )
    return backend
