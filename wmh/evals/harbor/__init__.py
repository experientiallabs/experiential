"""Ground-truth harness evaluation through Harbor (optional `harbor` extra).

This subpackage imports the `harbor` PyPI package at module scope and is therefore imported
lazily by its consumers, exactly like the e2b extra: `import wmh` (and `wmh.evals`) must succeed
without it. The E2B task-environment path additionally needs the e2b extra and is itself only
imported through harbor's environment factory (`wmh.evals.harbor.e2b_environment`).
"""

from wmh.evals.harbor.agent import WmhHarborAgent
from wmh.evals.harbor.scorer import (
    HarborJobRunner,
    HarborRewardMissingError,
    HarborRun,
    HarborRunner,
    HarborScorer,
)
from wmh.evals.harbor.tasks import resolve_harbor_tasks

__all__ = [
    "HarborJobRunner",
    "HarborRewardMissingError",
    "HarborRun",
    "HarborRunner",
    "HarborScorer",
    "WmhHarborAgent",
    "resolve_harbor_tasks",
]
