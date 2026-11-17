"""Canonical-call joins survive errors and remain local to their concurrent caller."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.runtime.models.call_evidence import canonical_call, current_call_intent


def test_nested_call_restores_context_after_failure() -> None:
    """An inner failed call must not relabel a subsequent outer physical attempt."""
    outer = ArtifactInput(artifact_id="outer", sha256="a" * 64)
    inner = ArtifactInput(artifact_id="inner", sha256="b" * 64)
    assert current_call_intent() is None
    with canonical_call(outer):
        with pytest.raises(RuntimeError), canonical_call(inner):
            assert current_call_intent() == inner
            raise RuntimeError("provider failed")
        assert current_call_intent() == outer
    assert current_call_intent() is None


def test_parallel_call_evidence_never_crosses_consumers() -> None:
    """Concurrent actor and simulator sessions retain distinct physical parentage."""
    barrier = Barrier(2)

    def run(name: str) -> ArtifactInput | None:
        """Overlap two active contexts before reading their references."""
        with canonical_call(ArtifactInput(artifact_id=name, sha256="a" * 64)):
            barrier.wait(timeout=10)
            return current_call_intent()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("actor", "simulator")))
    assert [row.artifact_id for row in results if row is not None] == ["actor", "simulator"]
