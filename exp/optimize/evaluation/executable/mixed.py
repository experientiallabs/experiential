"""Configuration-driven composition of a reviewed native runtime and a frozen learned simulator."""

from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
)
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.grounded import continuation_runtime as grounded_runtime
from exp.optimize.evaluation.executable.grounded import trial_runtime as granted_grounded_runtime
from exp.optimize.evaluation.executable.loader import ContinuationRuntime, load_runtime_definition
from exp.optimize.evaluation.executable.provider_grant import ProviderGrant
from exp.runtime.environments.mixed import (
    MixedCheckpoint,
    MixedEvent,
    MixedRealityEnvironmentRuntime,
    MixedRealityPolicy,
    ToolRoute,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import WorldCheckpoint


class NativeRuntimeBinding(ContractModel):
    """One explicitly selected native implementation with a separately bound private projection.

    Attributes:
        runtime_file: Trusted Python factory already implementing continuation_runtime.
        runtime_sha256: Exact selected source digest, verified before import.
        configuration: Immutable native setup; use digest-bound input artifacts, never credentials.
        probe_sha256: Expected evaluator-only projection definition.
    """

    runtime_file: Path
    runtime_sha256: Sha256
    configuration: JsonObject = Field(default_factory=dict)
    probe_sha256: Sha256


class MixedRuntimeDefinition(ContractModel):
    """Compose existing owners without domain-specific synchronization or new actor execution.

    Attributes:
        native: Reviewed native factory; its manifest must equal the simulator's manifest.
        simulated: Built simulator and pinned memory, with shared physical provider admission.
        routes: Explicit real/native or simulated routes. Shadow fidelity requires its own
            paired state contract and is not inferred by this grounded composition.
    """

    native: NativeRuntimeBinding
    simulated: GroundedRuntimeDefinition
    routes: tuple[ToolRoute, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def explicit_modes(self) -> Self:
        """Require both owners and refuse ambiguous shadow-state initialization."""
        if {route.mode for route in self.routes} != {"real", "simulated"}:
            raise ValueError("mixed grounded factory requires real and simulated routes only")
        # Validate ownership, effects and bridge routing before loading either factory.
        MixedRealityPolicy(
            native_identity="0" * 64,
            candidate_identity="0" * 64,
            routes=self.routes,
            world=self.simulated.manifest,
        )
        return self

    @property
    def probe_sha256(self) -> Sha256:
        """Bind separate native and simulated namespaces; neither silently overwrites the other."""
        return sha256_json(
            {
                "kind": "paired-native-simulated-projection-v1",
                "native": self.native.probe_sha256,
                "simulated": self.simulated.probe_sha256,
            }
        )


def continuation_runtime(project: ProjectStore, configuration: JsonObject) -> ContinuationRuntime:
    """Load explicitly selected owners and expose one paired world through existing workflows.

    Native state remains native-owned. The manifest's private native read bridges condition
    simulated calls but never synchronize predicted writes into native state. Read facts are
    instance observations, not learned rules or simulator-fidelity measurements. Native factory
    imports and dependencies remain trusted operator code; this is not a process sandbox.
    """
    return _runtime(project, configuration)


def trial_runtime(
    project: ProjectStore, configuration: JsonObject, provider_grant: ProviderGrant
) -> ContinuationRuntime:
    """Grant simulator calls for one trial; native environment authority remains unchanged."""
    provider_grant.require_configuration(project, configuration)
    return _runtime(project, configuration, provider_grant)


def _runtime(
    project: ProjectStore, configuration: JsonObject, provider_grant: ProviderGrant | None = None
) -> ContinuationRuntime:
    """Compose the same native and simulated owners under the supplied operational authority."""
    spec = MixedRuntimeDefinition.model_validate(configuration)
    # Validate typed semantics, but pass the registered JSON unchanged across the grant boundary.
    # Re-serializing datetime fields could change +00:00 to Z and falsely look like an override.
    simulated_configuration = configuration["simulated"]
    if not isinstance(simulated_configuration, dict):
        raise ValueError("simulated runtime configuration must be a JSON object")
    with ExitStack() as owned:
        native = owned.enter_context(
            load_runtime_definition(
                project,
                runtime_file=spec.native.runtime_file,
                runtime_sha256=spec.native.runtime_sha256,
                configuration=spec.native.configuration,
                probe_sha256=spec.native.probe_sha256,
            )
        )
        if not isinstance(native.world, ExecutableWorldRuntime) or native.world.owner != "native":
            raise ValueError("mixed composition requires a native executable-world runtime")
        if native.world.manifest != spec.simulated.manifest:
            raise ValueError("native runtime manifest differs from the frozen mixed definition")
        candidate = owned.enter_context(
            grounded_runtime(project, simulated_configuration)
            if provider_grant is None
            else granted_grounded_runtime(project, simulated_configuration, provider_grant)
        )
        if (
            not isinstance(candidate.world, ExecutableWorldRuntime)
            or candidate.world.owner != "simulated"
        ):
            raise ValueError("mixed composition requires an executable learned simulator")
        policy = MixedRealityPolicy(
            native_identity=native.world.backend_sha256,
            candidate_identity=candidate.world.backend_sha256,
            routes=spec.routes,
            world=spec.simulated.manifest,
        )
        release = owned.pop_all().close
    revision = installed_execution_revision()
    definition_sha = sha256_json(spec)

    def record(event: MixedEvent) -> None:
        """Save original authority and grounding evidence before the next mixed dispatch stage."""
        project.artifacts.write_json(
            artifact_id="mixed-" + event.operation_id + "-" + event.kind,
            artifact_type="mixed-executable-event",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=revision,
                inputs=tuple(
                    sorted(
                        (spec.simulated.world_model,)
                        + (() if provider_grant is None else (provider_grant.registration,)),
                        key=lambda reference: reference.artifact_id,
                    )
                ),
            ),
            files={
                "event.json": event,
                "definition.json": {
                    "sha256": definition_sha,
                    "policy": policy.model_dump(mode="json"),
                },
            },
        )

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """Return actual native and simulated projections with explicit separate authority."""
        if (
            not isinstance(checkpoint, MixedCheckpoint)
            or checkpoint.policy_sha256 != sha256_json(policy)
            or checkpoint.native is None
            or checkpoint.candidate is None
        ):
            raise ValueError("mixed probe checkpoint differs from its complete frozen policy")
        return {
            "native": native.probe(checkpoint.native),
            "simulated": candidate.probe(checkpoint.candidate),
        }

    return ContinuationRuntime(
        MixedRealityEnvironmentRuntime(
            policy=policy, native=native.world, candidate=candidate.world, record=record
        ),
        probe,
        spec.probe_sha256,
        release,
    )
