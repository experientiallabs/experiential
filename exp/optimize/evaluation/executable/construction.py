"""Trusted, data-only runtime binding for generated assets before native scenario execution."""

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.executable.loader import load_runtime_module
from exp.simulation.world_construction import load_construction


class ConstructionBinding(ContractModel):
    """Exact configuration produced by the operator-selected runtime, never by the author.

    Attributes:
        configuration: Bound runtime data, retaining the adapter's permissions and capabilities.
        probe_sha256: Private probe definition after binding the new assets.
    """

    configuration: JsonObject
    probe_sha256: Sha256


def bind_construction(
    project: ProjectStore, capture: CaptureDefinition, assets: ArtifactInput
) -> ConstructionBinding:
    """Invoke the pinned runtime's binder under the caller's durable binding intent.

    Adapters export ``bind_construction(project, configuration, assets)`` and return this model.
    Binding must be deterministic and perform no external effects or provider calls. This is
    trusted local code, not a sandbox. It must retain tool policy and change only initial assets.
    Runtimes without the hook refuse; generated code is never imported as a substitute.
    """
    load_construction(project.artifacts, assets)
    module = load_runtime_module(capture.runtime_file, capture.runtime_sha256)
    binder = getattr(module, "bind_construction", None)
    if not callable(binder):
        raise ValueError(
            "runtime must export bind_construction; implement a trusted data-only binder"
        )
    binding = binder(project, capture.configuration, assets)
    if not isinstance(binding, ConstructionBinding):
        raise ValueError("runtime binder must return ConstructionBinding")
    return ConstructionBinding.model_validate_json(binding.model_dump_json())
