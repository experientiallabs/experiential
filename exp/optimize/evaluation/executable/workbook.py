"""Configuration-only native workbook worlds through the shared executable runtime contract.

The workbook adapter is a concrete native backend, not a special mixed-reality path. Explicit
private worksheet views let a task constrain collateral edits without putting expected values
or full worksheets in actor context. File identities and selected-file provenance remain visible
to the verifier. Task authors must still define the intended behavior and qualify its predicate.
"""

from operator import not_
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_bytes,
    sha256_json,
    validate_artifact_file_path,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.construction import ConstructionBinding
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.runtime.environments.artifact_world import ArtifactWorldFactory
from exp.runtime.environments.workbook_edits import (
    WorkbookEditAdapter,
    read_supported_workbook,
    workbook_projection,
)
from exp.runtime.environments.world_contracts import ExecutableCheckpoint, WorldEvent, WorldManifest
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import WorldCheckpoint
from exp.simulation.world_construction import load_construction

# Saved probes and the task profiles that contain them bind to this protocol, not to this
# file's bytes. Bump it whenever projection semantics change; exp/tests/code_protocols_test.py
# records the covered source so the choice between a bump and a refactor stays explicit.
PRIVATE_VIEWS_PROTOCOL = "private-workbook-views-v2"


class WorkbookStateView(ContractModel):
    """An explicit worksheet-content view used by a private preservation predicate.

    Attributes:
        input_file: Initial workbook supplying the baseline.
        output_file: Inspect this output; None means inspect input_file. Unless
            baseline_until_written is set, an output the actor has not created is unavailable
            evidence, never the unchanged input.
        sheets: Exact ordered sheet names to hash, or None for all sheets in workbook order.
            Missing names produce unavailable evidence, never an empty successful digest.
        include_rows: Opt in to complete typed sheet rows in the private evaluator projection.
            Actor tool observations remain unchanged; explicit public_fields can disclose them.
        baseline_until_written: Project input_file while output_file is absent, so a preserved
            path compares the input baseline with the written output across every state. Use
            it only for preservation views: a goal over such a view would see the unchanged
            input when the actor never writes. Requires output_file.
    """

    include_rows: bool = Field(default=False, strict=True, exclude_if=not_)
    input_file: str
    output_file: str | None = None
    sheets: tuple[str, ...] | None = Field(default=None, min_length=1, max_length=128)
    baseline_until_written: bool = Field(default=False, strict=True, exclude_if=not_)

    @model_validator(mode="after")
    def unique_sheets(self) -> Self:
        """Refuse duplicate or empty names rather than hashing redundant or imaginary evidence."""
        if self.sheets is not None and (
            len(set(self.sheets)) != len(self.sheets) or any(not name for name in self.sheets)
        ):
            raise ValueError("worksheet view names must be nonempty and unique")
        if self.baseline_until_written and self.output_file is None:
            raise ValueError("a baseline until written needs an output_file to wait for")
        return self


class WorkbookRuntimeDefinition(ContractModel):
    """Frozen artifact bindings, native domain policy and private worksheet projections.

    Attributes:
        input_artifact: Exact immutable bundle containing the original XLSX files.
        files: Native input names mapped to file paths inside that artifact.
        output_paths: Separate native output files, absent until the actor creates them.
        manifest: Shared native/simulated domain and tool policy.
        views: Named private content digests and optional complete rows for verifier predicates.
        seed: Kernel-owned random seed.
        events: Optional deterministic scheduled events, restored with the world.
    """

    input_artifact: ArtifactInput
    files: dict[str, str] = Field(min_length=1, max_length=32)
    output_paths: tuple[str, ...] = Field(min_length=1, max_length=32)
    manifest: WorldManifest
    views: dict[str, WorkbookStateView] = Field(default_factory=dict, max_length=32)
    seed: int = Field(default=0, strict=True)
    events: tuple[WorldEvent, ...] = ()

    @model_validator(mode="after")
    def bound_namespace(self) -> Self:
        """Require exact native ownership and finite private views of declared workbook files."""
        if len(set(self.output_paths)) != len(self.output_paths) or set(self.files) & set(
            self.output_paths
        ):
            raise ValueError("workbook outputs must be unique and separate from immutable inputs")
        names = {*self.files, *self.output_paths}
        for name in names:
            if validate_artifact_file_path(name).as_posix() != name or not name.endswith(".xlsx"):
                raise ValueError("native workbook names must be canonical relative .xlsx paths")
        for source in self.files.values():
            if validate_artifact_file_path(source).as_posix() != source:
                raise ValueError("workbook artifact paths must be canonical relative paths")
        if {row.name for row in self.manifest.domains if row.owner == "native"} != names:
            raise ValueError(
                "native manifest domains must exactly match the input and output files"
            )
        for name, view in self.views.items():
            if (
                not name
                or len(name) > 128
                or view.input_file not in self.files
                or (view.output_file is not None and view.output_file not in self.output_paths)
            ):
                raise ValueError(
                    "worksheet views must name declared input/output files and a bounded key"
                )
        return self

    @property
    def probe_sha256(self) -> Sha256:
        """Bind projections to the view protocol, adapter dependencies and exact configuration."""
        return sha256_json(
            {
                "protocol": PRIVATE_VIEWS_PROTOCOL,
                "definition": self.model_dump(mode="json"),
                "adapter": WorkbookEditAdapter().definition,
            }
        )


def continuation_runtime(project: ProjectStore, configuration: JsonObject) -> ContinuationRuntime:
    """Build a native backend from immutable files; the actor receives no private probe object."""
    spec = WorkbookRuntimeDefinition.model_validate(configuration)
    manifest = project.artifacts.read(spec.input_artifact.artifact_id).manifest
    if artifact_input(manifest) != spec.input_artifact:
        raise ValueError("workbook input artifact changed; bind its exact immutable reference")
    files = {
        name: project.artifacts.read_bytes(spec.input_artifact.artifact_id, source)
        for name, source in spec.files.items()
    }
    for payload in files.values():
        read_supported_workbook(payload)
    world = ExecutableWorldRuntime(
        spec.manifest,
        ArtifactWorldFactory(WorkbookEditAdapter(), files=files, outputs=spec.output_paths),
        seed=spec.seed,
        events=spec.events,
    )

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """Expose file identities and selected-sheet digests only to the evaluator."""
        if not isinstance(checkpoint, ExecutableCheckpoint) or (
            checkpoint.backend_sha256 != world.backend_sha256
            or checkpoint.manifest_sha256 != world.manifest_sha256
            or checkpoint.owner != "native"
        ):
            raise ValueError("workbook probe requires its exact native backend and manifest")
        decoded: dict[str, JsonObject] = {}
        views: JsonObject = {}
        for name, view in spec.views.items():
            selected = (
                view.input_file
                if view.output_file is None
                or (view.baseline_until_written and view.output_file not in checkpoint.image.parts)
                else view.output_file
            )
            if selected not in checkpoint.image.parts:
                views[name] = {"source": None, "unavailable": "registered file is absent"}
                continue
            if selected not in decoded:
                decoded[selected] = workbook_projection(
                    read_supported_workbook(checkpoint.image.parts[selected])
                )
            sheets = decoded[selected]["sheets"]
            if not isinstance(sheets, list):
                raise ValueError("workbook projection has a malformed sheet inventory")
            named = {
                row["name"]: row
                for row in sheets
                if isinstance(row, dict) and isinstance(row.get("name"), str)
            }
            missing = (
                []
                if view.sheets is None
                else [sheet for sheet in view.sheets if sheet not in named]
            )
            if missing:
                views[name] = {"source": selected, "missing_sheets": missing}
                continue
            content = sheets if view.sheets is None else [named[sheet] for sheet in view.sheets]
            projection: JsonObject = {"source": selected, "sha256": sha256_json(content)}
            if view.include_rows:
                projection["sheets"] = content
            views[name] = projection
        return {
            "files": {
                name: sha256_bytes(payload) for name, payload in checkpoint.image.parts.items()
            },
            "views": views,
        }

    return ContinuationRuntime(world, probe, spec.probe_sha256)


def bind_construction(
    project: ProjectStore, configuration: JsonObject, assets: ArtifactInput
) -> ConstructionBinding:
    """Replace only the source bundle; keep files, ownership, probes, events and seed fixed."""
    load_construction(project.artifacts, assets)
    definition = WorkbookRuntimeDefinition.model_validate(configuration)
    bound = WorkbookRuntimeDefinition.model_validate(
        {**definition.model_dump(mode="json"), "input_artifact": assets.model_dump(mode="json")}
    )
    for source in bound.files.values():
        read_supported_workbook(project.artifacts.read_bytes(assets.artifact_id, source))
    return ConstructionBinding(
        configuration=bound.model_dump(mode="json"), probe_sha256=bound.probe_sha256
    )
