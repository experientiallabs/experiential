"""Follow-up: corrected Test 1 (domain recovery) for the scenario-set e2e run.

The main e2e run's original Test 1 was degenerate on this corpus (one rollout per task_id, so
k = n and any clustering scores perfectly). This reruns Test 1 against the non-trivial ground
truth — the tau2 domain (airline / retail / telecom) — and patches the result into
scenario_e2e_results_tau_bench.json under `test1_domain_recovery`.

Usage (from the repo root, after run_scenario_e2e.py finishes):
    uv run python .agents/scripts/run_scenario_test1_domain.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

# Reuse the main script's corpus/provider helpers so both runs see the identical subsample.
sys.path.insert(0, str(REPO / ".agents" / "scripts"))
from run_scenario_e2e import (  # noqa: E402
    OUT_DIR,
    TRACES,
    WORKERS,
    bedrock,
    NOVA_LITE,
    embed_batch,
    subsample,
    titan_embedder,
)

from wmh.ingest import get_adapter  # noqa: E402
from wmh.research.scenario_recovery import ground_truth_labels, recovery_report  # noqa: E402
from wmh.scenarios import FacetExtractor, cluster_facets, trace_digest  # noqa: E402


def main() -> None:
    corpus = subsample(get_adapter("otel-genai").from_file(str(TRACES)))
    lite = bedrock(NOVA_LITE)
    extractor = FacetExtractor(lite)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        facets = list(pool.map(extractor.extract, corpus))
    embedder = titan_embedder()
    facet_embeddings = embed_batch(embedder, [f.embed_text() for f in facets])
    digest_embeddings = embed_batch(embedder, [trace_digest(t)[:8000] for t in corpus])

    truth = [label.split("/")[0] for label in ground_truth_labels(corpus)]
    k = len(set(truth))
    facet_labels, _ = cluster_facets(facets, facet_embeddings, k=k, seed=0)
    digest_labels, _ = cluster_facets(facets, digest_embeddings, k=k, seed=0)
    rec_facet = recovery_report(facet_labels.tolist(), truth)
    rec_digest = recovery_report(digest_labels.tolist(), truth)
    print(f"facet:  purity={rec_facet.purity:.3f} ARI={rec_facet.adjusted_rand_index:.3f}")
    print(f"digest: purity={rec_digest.purity:.3f} ARI={rec_digest.adjusted_rand_index:.3f}")

    results_path = OUT_DIR / "scenario_e2e_results_tau_bench.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results.pop("test1_recovery", None)  # the degenerate task-identity version
    results["test1_domain_recovery"] = {
        "note": "one rollout per task_id in this corpus makes task-identity recovery degenerate; "
        "ground truth here is the tau2 domain, k=3",
        "facet_embedding": rec_facet.model_dump(),
        "raw_digest_embedding_baseline": rec_digest.model_dump(),
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"patched {results_path}")


if __name__ == "__main__":
    main()
