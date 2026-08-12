"""Shared optimizer contracts plus artifact-specific optimization packages.

Offline guarded router fitting lives in `wmo.optimize.router`, model training lives in
`wmo.optimize.model`, and the harness execution seam lives in `wmo.runtime.harness`.
Shared optimizer contracts and temporary prompt evolution remain at this package root.
"""

from wmo.optimize.base import ArtifactRef, OptimizeMetrics, Optimizer, OptimizeResult
from wmo.optimize.gepa import GEPAOptimizer

__all__ = [
    "ArtifactRef",
    "GEPAOptimizer",
    "OptimizeMetrics",
    "OptimizeResult",
    "Optimizer",
]
