"""Ground-truth harness evaluation through Harbor (optional `harbor` extra).

This subpackage imports the `harbor` PyPI package at module scope and is therefore imported
lazily by its consumers, exactly like the e2b extra: `import wmo` (and `wmo.evals`) must succeed
without it. The E2B task-environment path additionally needs the e2b extra and is itself only
imported through harbor's environment factory (`wmo.evals.harbor.e2b_environment`).
"""

from wmo.evals.harbor.agent import WmoHarborAgent
from wmo.evals.harbor.scorer import (
    HarborJobRunner,
    HarborRewardMissingError,
    HarborRun,
    HarborRunner,
    HarborScorer,
)
from wmo.evals.harbor.tasks import resolve_harbor_tasks

__all__ = [
    "HarborJobRunner",
    "HarborRewardMissingError",
    "HarborRun",
    "HarborRunner",
    "HarborScorer",
    "WmoHarborAgent",
    "resolve_harbor_tasks",
]
