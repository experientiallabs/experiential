"""Load an operator-selected trusted runtime file into an executable continuation world.

Only the explicitly selected source bytes execute, after their digest is checked. A plain load
calls the file's ``continuation_runtime`` factory; a load under a ``ProviderGrant`` calls its
``trial_runtime`` factory with that per-trial provider authority. Private probes remain evaluator
capabilities. Dependencies imported by the trusted factory are the operator's responsibility.

A runtime may own processes or files (an induced-program worker, the parts of a composition).
Whoever loads one closes it: use it as a context manager, or call ``close``. A runtime that the
loader refuses after its factory ran is closed before the refusal propagates.
"""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Self

from exp.common.core.artifacts import JsonObject, Sha256, sha256_bytes
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.study import WorldRuntime
from exp.optimize.evaluation.executable.provider_grant import ProviderGrant
from exp.simulation.continuations import WorldCheckpoint


def _owns_nothing() -> None:
    """Release nothing: the factory started no process and holds no file."""


@dataclass(frozen=True)
class ContinuationRuntime:
    """Trusted environment implementation selected explicitly by the operator.

    Attributes:
        world: Existing executable or mixed runtime with exact restore semantics.
        probe: Evaluator-only state projection, never passed to the actor.
        probe_sha256: Digest of the projection definition checked against registration.
        release: Stops what the factory started; idempotent. Factories that own nothing
            leave the default.
    """

    world: WorldRuntime
    probe: Callable[[WorldCheckpoint], JsonObject]
    probe_sha256: Sha256
    release: Callable[[], None] = _owns_nothing

    def close(self) -> None:
        """Release every process and file the factory opened; later calls do nothing."""
        self.release()

    def __enter__(self) -> Self:
        """Return this runtime; resources are released when the block exits."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close on every exit path, including exceptions and interrupts."""
        del exception_type, exception, traceback
        self.close()


def load_runtime_definition(
    project: ProjectStore,
    *,
    runtime_file: Path,
    runtime_sha256: Sha256,
    configuration: JsonObject,
    probe_sha256: Sha256,
    provider_grant: ProviderGrant | None = None,
) -> ContinuationRuntime:
    """Execute verified factory bytes for initial capture or exact checkpoint restoration.

    Args:
        project: Trusted project passed to the factory.
        runtime_file: Explicit trusted Python file exporting the factory.
        runtime_sha256: Exact reviewed source digest.
        configuration: Frozen factory inputs, without private credentials.
        probe_sha256: Registered probe identity the runtime must report.
        provider_grant: Per-trial provider authority; selects ``trial_runtime`` when given.

    Returns:
        The factory's runtime, after its type and probe identity are checked. The caller
        closes it.
    """
    if provider_grant is not None:
        provider_grant.require_configuration(project, configuration, runtime_sha256=runtime_sha256)
    module = load_runtime_module(runtime_file, runtime_sha256)
    entrypoint = "continuation_runtime" if provider_grant is None else "trial_runtime"
    factory = getattr(module, entrypoint, None)
    if not callable(factory):
        raise ValueError(f"runtime must export {entrypoint} for this execution authority")
    runtime = (
        factory(project, configuration)
        if provider_grant is None
        else factory(project, configuration, provider_grant)
    )
    if not isinstance(runtime, ContinuationRuntime):
        raise ValueError("runtime factory must return ContinuationRuntime")
    if runtime.probe_sha256 != probe_sha256:
        runtime.close()
        raise ValueError("runtime probe differs from the frozen definition")
    return runtime


def load_runtime_module(runtime_file: Path, runtime_sha256: Sha256) -> ModuleType:
    """Import only the explicitly selected source bytes, never model-generated executable code."""
    module_spec = importlib.util.spec_from_file_location(
        "exp_continuation_runtime_" + runtime_sha256, runtime_file
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError("runtime file cannot be loaded as a Python module")
    module = importlib.util.module_from_spec(module_spec)
    source = runtime_file.read_bytes()
    if sha256_bytes(source) != runtime_sha256:
        raise ValueError("runtime source changed after preflight")
    exec(compile(source, str(runtime_file), "exec"), module.__dict__)
    return module
