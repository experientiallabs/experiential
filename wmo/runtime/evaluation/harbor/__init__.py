"""Ground-truth harness evaluation through Harbor (optional `harbor` extra).

This subpackage imports the `harbor` PyPI package at module scope and is therefore imported
lazily by its consumers, exactly like the e2b extra. Importing `wmo` and
`wmo.simulation.evaluation` must succeed without it. The E2B task-environment path also needs e2b
and is imported only through Harbor's environment factory.
"""

from wmo.runtime.evaluation.harbor.agent import WmoHarborAgent
from wmo.runtime.evaluation.harbor.scorer import (
    HarborJobRunner,
    HarborRewardMissingError,
    HarborRun,
    HarborRunner,
    HarborScorer,
)
from wmo.runtime.evaluation.harbor.tasks import resolve_harbor_tasks

__all__ = [
    "HarborJobRunner",
    "HarborRewardMissingError",
    "HarborRun",
    "HarborRunner",
    "HarborScorer",
    "WmoHarborAgent",
    "resolve_harbor_tasks",
]
