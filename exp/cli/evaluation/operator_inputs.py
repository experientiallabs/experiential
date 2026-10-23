"""Continuation actions that print or publish the operator inputs other actions consume.

Specs for jobs, reviews, comparisons and factory batches pin exact model snapshots, the frozen
catalog digest, artifact references, native runtime bindings and registered development
trajectories. These actions produce each of them through the CLI, so an operator writes JSON
specs instead of Python:

- ``model-snapshots`` prints the catalog digest and the exact ``ModelSnapshot`` of each alias.
- ``artifact-reference`` prints exact references of saved artifacts by ID or type, including the
  task of a task variant.
- ``native-workbook`` publishes declared starting workbooks byte for byte and prints a ready
  native ``capture`` operation, an optional answer-range verifier and the untouched projection.
- ``register-rubric-trajectory`` registers one development chronology for rubric derivation
  together with its explicit development classification.

None of them calls a provider.
"""

import json
from pathlib import Path

from pydantic import Field

from exp.cli.evaluation.continuation_context import (
    ContinuationContext,
    require_new_operation_file,
)
from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import artifact_input
from exp.optimize.evaluation.analysis.source_access import declare_source_access
from exp.optimize.evaluation.equivalence.rubric import (
    RubricTrajectory,
    register_rubric_trajectory,
)
from exp.optimize.evaluation.executable.workbook_operator import (
    NativeWorkbook,
    native_workbook_capture,
)
from exp.runtime.models import RuntimeModelCatalog


class ModelSnapshotQuery(ContractModel):
    """Catalog aliases whose exact identities a spec pins.

    Attributes:
        aliases: Configured model aliases, in the order to print.
    """

    aliases: tuple[str, ...] = Field(min_length=1, max_length=64)


class ArtifactReferenceQuery(ContractModel):
    """Saved artifacts to print exact references for.

    Attributes:
        artifact_ids: Exact artifact IDs.
        artifact_types: Artifact types whose every saved artifact is listed.
        contents: Also print the JSON data files of review-facing artifacts (rubric drafts
            and frozen rubrics, review responses and failures, factory, validity and quality
            reports); any other type is refused, so evaluation evidence is never disclosed.
    """

    artifact_ids: tuple[str, ...] = Field(default=(), max_length=256)
    artifact_types: tuple[str, ...] = Field(default=(), max_length=16)
    contents: bool = Field(default=False, strict=True)


# Artifacts an operator must read to review a step: model proposals awaiting an explicit
# freeze, review outcomes and factory reports. Evaluation inputs, held-out tasks, traces,
# checkpoints and world images are never printed.
REVIEWABLE_TYPES = frozenset(
    {
        "actor-rubric-draft",
        "actor-rubric",
        "equivalence-review-response",
        "equivalence-review-failure",
        "scenario-search-result",
        "task-proposals",
        "task-quality-proxy",
        "task-validity-report",
        "candidate-measurement",
        "candidate-batch",
    }
)
MAXIMUM_PRINTED_BYTES = 1_000_000


class NativeWorkbookOperation(NativeWorkbook):
    """A native workbook world for one development task, and where to save its capture.

    Attributes:
        operation_file: Optional new JSON file for the ready ``capture`` operation. Existing
            files are never overwritten; dry-run does not create it.
    """

    operation_file: Path | None = None


class RubricTrajectoryRegistration(ContractModel):
    """One development chronology to register as a rubric-derivation source.

    Attributes:
        trajectory: Development task and its retained chronology.
        rationale: The operator's reason the chronology may be read as development evidence.
    """

    trajectory: RubricTrajectory
    rationale: str = Field(min_length=1, max_length=2000)


def handle_model_snapshots(context: ContinuationContext, query: ModelSnapshotQuery) -> None:
    """Print the frozen catalog digest and the exact snapshot of each alias.

    Args:
        context: Validated ``--continuations`` invocation.
        query: Aliases to resolve.

    Returns:
        None. ``catalog_sha256`` and ``models`` (alias to ``ModelSnapshot``) are printed.
    """
    catalog = load_model_catalog(context.store.model_catalog_path)
    runtime = RuntimeModelCatalog(catalog, environment={})
    context.console.print_json(
        data={
            "catalog_sha256": sha256_json(catalog.model_dump(mode="json")),
            "models": {
                alias: runtime.snapshot(alias)[0].model_dump(mode="json") for alias in query.aliases
            },
        }
    )


def handle_artifact_reference(context: ContinuationContext, query: ArtifactReferenceQuery) -> None:
    """Print exact references of saved artifacts; a task variant also names its task.

    Args:
        context: Validated ``--continuations`` invocation.
        query: Artifact IDs and types to list.

    Returns:
        None. One row per artifact with ``reference``, ``artifact_type``, ``created_at`` and
        ``inputs`` is printed; ``task-variant`` rows add their executable ``task``, and with
        ``contents`` review artifacts add their JSON data files.

    Raises:
        ValueError: The query names nothing, an ID is not saved, or contents were requested
            for an artifact type outside the review allowlist.
    """
    if not query.artifact_ids and not query.artifact_types:
        raise ValueError("name artifact_ids or artifact_types to list")
    store = context.store.artifacts
    identities = list(query.artifact_ids)
    for kind in query.artifact_types:
        identities.extend(store.list_ids(artifact_type=kind))
    rows = []
    for identity in dict.fromkeys(identities):
        if not store.exists(identity):
            raise ValueError(f"no saved artifact {identity!r}")
        manifest = store.read(identity).manifest
        row: JsonObject = {
            "reference": artifact_input(manifest).model_dump(mode="json"),
            "artifact_type": manifest.artifact_type,
            "created_at": manifest.created_at.isoformat(),
            "inputs": [item.model_dump(mode="json") for item in manifest.inputs],
        }
        if query.contents:
            if manifest.artifact_type not in REVIEWABLE_TYPES:
                raise ValueError(
                    f"artifact {identity} is a {manifest.artifact_type}; contents are printed "
                    "only for review artifacts: " + ", ".join(sorted(REVIEWABLE_TYPES))
                )
            files: JsonObject = {}
            for entry in manifest.files:
                payload = store.read_bytes(identity, entry.path)
                if len(payload) > MAXIMUM_PRINTED_BYTES or not entry.path.endswith(".json"):
                    files[entry.path] = {"bytes": len(payload), "sha256": entry.sha256}
                else:
                    files[entry.path] = json.loads(payload)
            row["contents"] = files
        if manifest.artifact_type == "task-variant":
            paths = [entry.path for entry in manifest.files]
            if len(paths) == 1:
                variant = json.loads(store.read_bytes(identity, paths[0]))
                if isinstance(variant, dict) and isinstance(variant.get("task"), dict):
                    row["task"] = ArtifactInput.model_validate(variant["task"]).model_dump(
                        mode="json"
                    )
        rows.append(row)
    context.console.print_json(data={"artifacts": rows})


def handle_native_workbook(context: ContinuationContext, request: NativeWorkbookOperation) -> None:
    """Publish the starting workbooks and print a ready native capture operation.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` publishes nothing and
            reports a placeholder input reference and no projection.
        request: Development task, starting files, views, reference answers and output file.

    Returns:
        None. ``input_artifact``, ``capture``, ``verifier`` and ``initial_projection`` are
        printed; the capture operation is saved to ``operation_file`` unless dry-run.
    """
    require_new_operation_file(request.operation_file)
    prepared = native_workbook_capture(
        context.store, request, revision=context.revision, publish=not context.dry_run
    )
    if request.operation_file is not None and not context.dry_run:
        with request.operation_file.open("x", encoding="utf-8") as stream:
            json.dump(prepared["capture"], stream, indent=2)
            stream.write("\n")
    context.console.print_json(data=prepared)


def handle_register_rubric_trajectory(
    context: ContinuationContext, registration: RubricTrajectoryRegistration
) -> None:
    """Register one development chronology and classify it for development diagnostics.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` registers nothing.
        registration: The chronology and the operator's classification rationale.

    Returns:
        None. ``trajectory`` (the ``derive-equivalence-rubric`` source) and ``source_access``
        are printed. A repeated registration replays the saved artifacts.
    """
    if context.dry_run:
        context.console.print("Dry run: not registered.")
        return
    source = register_rubric_trajectory(
        context.store, registration.trajectory, inputs=(), code_revision=context.revision
    )
    access = declare_source_access(
        context.store.artifacts,
        source,
        scope="development",
        rationale=registration.rationale,
        code_revision=context.revision,
    )
    context.console.print_json(
        data={
            "trajectory": source.model_dump(mode="json"),
            "source_access": access.model_dump(mode="json"),
        }
    )
