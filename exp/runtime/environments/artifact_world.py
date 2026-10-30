"""Native artifact files as an executable-world backend, edited by a typed artifact adapter.

Each declared file path is one state domain: the initial input files plus the outputs the
adapter may produce. The adapter resolves which files a call reads and writes before it runs,
then transforms detached bytes; the native checkpoint is the exact file bytes. Adapter
refusals (unsupported features, out-of-range edits) are error observations with no change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from exp.common.core.artifacts import (
    JsonObject,
    Sha256,
    sha256_bytes,
    sha256_json,
    validate_artifact_file_path,
)
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.artifact_edits import ArtifactEditAdapter
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import NativeImage, WorldAccess
from exp.runtime.environments.worlds import WorldContext

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
ARTIFACT_WORLD_PROTOCOL = "artifact-world-v2"

_ABSENT = sha256_json({"absent": True})


@runtime_checkable
class ArtifactAccessAdapter(ArtifactEditAdapter, Protocol):
    """An artifact adapter that also resolves each call's file access before execution."""

    def access(self, action: ToolCall) -> WorldAccess:
        """Return the files one call reads and writes, without applying it."""


class ArtifactWorldFactory:
    """Create and restore artifact worlds over one initial file set and declared outputs."""

    def __init__(
        self,
        adapter: ArtifactAccessAdapter,
        *,
        files: Mapping[str, bytes],
        outputs: tuple[str, ...],
    ) -> None:
        """Bind the adapter, the immutable initial files and the writable output paths.

        Raises:
            ValueError: A path is not canonical or no outputs are declared.
        """
        if not outputs:
            raise ValueError("an artifact world declares at least one output path")
        for path in (*files, *outputs):
            if validate_artifact_file_path(path).as_posix() != path:
                raise ValueError(f"artifact path {path!r} is not a canonical relative path")
        self._adapter = adapter
        self._files = dict(files)
        self._outputs = tuple(outputs)
        self._domains = tuple(sorted({*files, *outputs}))
        self._definition: JsonObject = {
            "name": ARTIFACT_WORLD_PROTOCOL,
            "adapter": adapter.definition,
            "files": {path: sha256_bytes(payload) for path, payload in sorted(files.items())},
            "outputs": list(self._outputs),
        }

    @property
    def definition(self) -> JsonObject:
        """Return the adapter, initial file and output identity bound by checkpoints."""
        return self._definition

    def create(self, task: TaskCase) -> ArtifactWorld:
        """Create a world holding a private copy of the initial files."""
        del task
        return ArtifactWorld(self._adapter, dict(self._files), self._domains, self._outputs)

    def restore(self, image: NativeImage) -> ArtifactWorld:
        """Create a world holding a private copy of the captured file bytes."""
        unknown = sorted(set(image.parts) - set(self._domains))
        if unknown:
            raise ValueError(f"artifact image contains undeclared files {unknown!r}")
        return ArtifactWorld(self._adapter, dict(image.parts), self._domains, self._outputs)


class ArtifactWorld:
    """One private file set driven by the executable-world kernel."""

    def __init__(
        self,
        adapter: ArtifactAccessAdapter,
        files: dict[str, bytes],
        domains: tuple[str, ...],
        outputs: tuple[str, ...],
    ) -> None:
        """Own one file mapping; it is never shared with another world."""
        self._adapter = adapter
        self._files = files
        self._domains = domains
        self._outputs = outputs

    def access(self, action: ToolCall) -> WorldAccess:
        """Resolve one call's files through the adapter."""
        return self._adapter.access(action)

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Apply one call to detached bytes and adopt only the declared outputs it produced."""
        del context
        try:
            output = self._adapter.apply(dict(self._files), (action,), self._outputs)
        except ValueError as exc:
            return Observation(content=str(exc), is_error=True)
        self._files.update(output.files)
        written = {
            path: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
            for path, payload in sorted(output.files.items())
        }
        receipt: JsonObject = {"written": written}
        if not output.files:
            receipt["projection"] = output.projection
        return Observation(
            content=json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def domain_digests(self) -> dict[str, Sha256]:
        """Digest every declared file, marking files that do not exist yet."""
        return {
            path: sha256_bytes(self._files[path]) if path in self._files else _ABSENT
            for path in self._domains
        }

    def capture(self) -> NativeImage:
        """Return the exact bytes of every existing file and the per-file digests."""
        return NativeImage(parts=dict(self._files), domains=self.domain_digests())

    def state_projection(self) -> JsonObject:
        """Expose file existence and byte identity, without claiming semantic file coverage."""
        return {
            path: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
            for path, payload in sorted(self._files.items())
        }

    def close(self) -> None:
        """Drop the private file mapping."""
        self._files = {}
