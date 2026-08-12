"""Shared optimizer contracts plus artifact-specific optimization packages.

Routing policy search lives in `wmo.optimize.routing`, model training lives in
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
