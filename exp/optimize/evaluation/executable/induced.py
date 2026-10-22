"""Built-in executable factory: induced programs over the frozen grounded simulator.

The configuration is a grounded runtime definition plus a saved induction (or re-validation)
whose coverage was validated under the same world manifest. Induced tools run as exact code
inside their validated domain; everything else, and every call outside that domain, goes to
the grounded predictor, which holds the one simulated state and the full call history. The
world stays the simulated owner: no reply this runtime produces is native evidence.

Only ``continuation_runtime`` is provided; per-trial provider grants bind the grounded
definition's own configuration and are not composed here.
"""

import json
import weakref

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.induction.service import load_coverage
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import WorldCheckpoint
from exp.simulation.induction.backend import InducedWorldFactory
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.reconstruction.models import resolve


class InducedRuntimeDefinition(ContractModel):
    """A grounded simulator with induced programs serving the tools they own.

    Attributes:
        grounded: Frozen grounded runtime definition; its manifest must equal the coverage's.
        coverage: Saved induction or re-validation whose coverage the programs execute under.
    """

    grounded: GroundedRuntimeDefinition
    coverage: ArtifactInput

    @property
    def probe_sha256(self) -> Sha256:
        """Bind a projection of simulated state served by induced code and the predictor."""
        return sha256_json(
            {"kind": "induced-simulated-domain-projection", "paths": self.grounded.probe_paths}
        )


def continuation_runtime(project: ProjectStore, configuration: JsonObject) -> ContinuationRuntime:
    """Compose induced programs over the grounded simulator for capture or continuation.

    Raises:
        ValueError: The coverage was validated under another world manifest, or the grounded
            runtime is not the built-in executable grounded world.
    """
    spec = InducedRuntimeDefinition.model_validate(configuration)
    coverage = load_coverage(project.artifacts, spec.coverage)
    if coverage.manifest != spec.grounded.manifest:
        raise ValueError(
            "induced coverage was validated under another world manifest; re-run "
            "validate-transitions with this runtime's manifest"
        )
    base = grounded.continuation_runtime(project, spec.grounded.model_dump(mode="json"))
    world = base.world
    if not isinstance(world, ExecutableWorldRuntime):
        raise ValueError("induced programs compose only with the executable grounded world")
    sandbox = ProgramSandbox(coverage.limits)
    facts = spec.grounded.facts
    starting = None if facts is None else facts.facts.conditioning(facts.first_version)
    factory = InducedWorldFactory(
        coverage,
        world.factory,
        sandbox,
        facts={} if starting is None else {fact.key: fact.value for fact in starting.facts},
    )
    # The runtime has no close hook; the worker stops when the factory is collected or at exit.
    weakref.finalize(factory, sandbox.close)

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """Read only registered simulated state; unresolved fields refuse assessment."""
        if not isinstance(checkpoint, ExecutableCheckpoint):
            raise ValueError("simulated projection requires its executable checkpoint")
        if checkpoint.backend_sha256 != sha256_json(factory.definition):
            raise ValueError("probe checkpoint differs from the frozen induced simulator")
        wire = json.loads(checkpoint.image.parts["predictor.json"])
        return {
            name: resolve(wire["state"], path) for name, path in spec.grounded.probe_paths.items()
        }

    return ContinuationRuntime(
        ExecutableWorldRuntime(
            spec.grounded.manifest, factory, owner="simulated", seed=spec.grounded.seed
        ),
        probe,
        spec.probe_sha256,
    )
