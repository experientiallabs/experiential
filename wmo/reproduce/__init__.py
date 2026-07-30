"""One-command reproduction of published benchmark results (`wmo reproduce`).

Every published number this project quotes should be one command away from a stranger's
terminal. A committed MANIFEST per benchmark pins everything the number depends on - the
dataset (a Hugging Face repo), the protocol (every knob), and the published results with an
explicit tolerance - and the runner replays it and says REPRODUCED or DIVERGED, row by row.

Two manifest kinds, because the benchmarks differ in what "exact" can mean:

- `matrix`: the evidence is a precomputed outcome matrix (RouterBench-style). The runner
  fits and reports in-process with the RECORDED embedding vectors served from a published
  cache, so the reproduction is offline, credential-free, and bit-exact by construction.
- `commands`: the evidence is bought live (a trace corpus is built into a world model and
  swept against real providers). The runner replays the pinned CLI commands verbatim. This
  spends real money (it forecasts and requires --yes) and is PROTOCOL-exact, never
  bit-exact: providers are nondeterministic and prices drift, which is why these manifests
  carry wide tolerances and the verdict says which kind of exactness it is claiming.
"""

from wmo.reproduce.manifest import (
    Manifest,
    PublishedRow,
    load_manifest,
    manifest_names,
)
from wmo.reproduce.runner import ReproduceResult, RowVerdict, run_reproduction

__all__ = [
    "Manifest",
    "PublishedRow",
    "ReproduceResult",
    "RowVerdict",
    "load_manifest",
    "manifest_names",
    "run_reproduction",
]
