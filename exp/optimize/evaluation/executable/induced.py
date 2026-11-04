"""Built-in executable factory: induced programs over the frozen grounded simulator.

The configuration is a grounded runtime definition plus a saved induction (or re-validation)
whose coverage was validated under the same world manifest. The coverage is bound into the
grounded definition's ``programs`` and served by the grounded factory's one composition: induced
tools run as exact code inside their validated domain; everything else, and every call outside
that domain, goes to the grounded predictor, which holds the one simulated state and the full
call history. The world stays the simulated owner: no reply this runtime produces is native
evidence.

Only ``continuation_runtime`` is provided; per-trial provider grants bind the grounded
definition's own configuration and are not composed here. The returned runtime owns the
program worker; closing it stops the worker and removes its directory.
"""

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.induction.service import load_coverage


class InducedRuntimeDefinition(ContractModel):
    """A grounded simulator with induced programs serving the tools they own.

    Attributes:
        grounded: Frozen grounded runtime definition without programs of its own; its manifest
            must equal the coverage's.
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
            definition already binds programs.
    """
    spec = InducedRuntimeDefinition.model_validate(configuration)
    if spec.grounded.programs is not None:
        raise ValueError(
            "the grounded definition already binds transition programs; bind one coverage, "
            "either inline or through this induced definition"
        )
    coverage = load_coverage(project.artifacts, spec.coverage)
    if coverage.manifest != spec.grounded.manifest:
        raise ValueError(
            "induced coverage was validated under another world manifest; re-run "
            "validate-transitions with this runtime's manifest"
        )
    composed = spec.grounded.model_copy(update={"programs": coverage})
    runtime = grounded.continuation_runtime(project, composed.model_dump(mode="json"))
    return ContinuationRuntime(runtime.world, runtime.probe, spec.probe_sha256, runtime.release)
