"""Build the METHOD A task manifest + unpack the per-task verifier bundles.

Two jobs:

1. Extract the FULL deep-swe task tree. loaders/deepswe.py deliberately extracts only
   instruction.md/task.toml/manifest.json, so tests/ (test.sh, test.patch, grader.py, config.json)
   -- which is the entire grading contract -- is missing from that cache. We need it.

2. Emit manifest.json in a REPO-GROUPED SHUFFLED order, one entry per task.

Why the order matters: the collector walks the manifest until its spend cap trips, so the order
decides which tasks get measured if we run out of money. Tasks are shuffled by REPOSITORY under a
fixed seed and then emitted repo-block by repo-block, so any prefix of the manifest is a random
sample of repos rather than an alphabetical slice. That keeps the repo-grouped fit/heldout split
and the repo-clustered bootstrap valid on a truncated run, and keeps whole repos together so a
partial run does not split one repo across measured and unmeasured.

    python3 router_methods_a_manifest.py --outdir /scratch/router_a
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import tarfile
import urllib.request

TARBALL = "https://codeload.github.com/datacurve-ai/deep-swe/tar.gz/refs/heads/main"
TASKS_URL = "https://deepswe.datacurve.ai/artifacts/v1/tasks.json"

# Identical on all 113 instruction.md files, so it carries no task signal -- but it DOES tell the
# agent to commit, which our patch extraction (git add -A + diff vs base) does not depend on. Kept
# in the prompt so the scaffold matches what the published trials ran against.
BOILERPLATE = ("\nIMPORTANT: Please work on this in a new branch from main and "
               "commit everything when you are done.\n")


def fetch(url: str, dest: pathlib.Path) -> pathlib.Path:
    if not dest.exists() or dest.stat().st_size == 0:
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    tar_path = fetch(TARBALL, out / "deep-swe-main.tar.gz")
    tasks_json = json.loads(fetch(TASKS_URL, out / "tasks.json").read_text())
    repo_of = {t["id"]: t["repository"] for t in tasks_json["rows"]}

    tree = out / "tree"
    if not (tree / "deep-swe-main" / "tasks").is_dir():
        with tarfile.open(tar_path) as tf:
            tf.extractall(tree, filter="data")   # FULL tree: tests/ included
    troot = tree / "deep-swe-main" / "tasks"

    entries = []
    for d in sorted(troot.iterdir()):
        if not (d / "task.toml").exists():
            continue
        toml = (d / "task.toml").read_text()

        def grab(key: str) -> str | None:
            for line in toml.splitlines():
                if line.strip().startswith(f"{key} ="):
                    return line.split("=", 1)[1].strip().strip('"')
            return None

        image, base = grab("docker_image"), grab("base_commit_hash")
        cfg = d / "tests" / "config.json"
        if not (image and base and cfg.exists()):
            print(f"SKIP {d.name}: image={bool(image)} base={bool(base)} cfg={cfg.exists()}")
            continue
        n_f2p = len(json.loads(cfg.read_text()).get("f2p_node_ids") or [])
        if n_f2p == 0:
            print(f"SKIP {d.name}: no f2p tests, graded reward would be undefined")
            continue
        raw = (d / "instruction.md").read_text()
        entries.append({
            "task_id": d.name,
            "repo": repo_of.get(d.name, d.name),
            "image": image,
            "base_commit": base,
            "n_f2p": n_f2p,
            "prompt": raw if raw.endswith(BOILERPLATE) else raw + BOILERPLATE,
        })

    by_repo: dict[str, list[dict]] = {}
    for e in entries:
        by_repo.setdefault(e["repo"], []).append(e)
    repos = sorted(by_repo)
    random.Random(a.seed).shuffle(repos)
    ordered = [e for r in repos for e in by_repo[r]]

    (out / "manifest.json").write_text(json.dumps(ordered, indent=1))
    print(f"manifest: {len(ordered)} tasks over {len(repos)} repos -> {out / 'manifest.json'}")
    print(f"tests root: {troot}")
    print(f"f2p tests per task: min={min(e['n_f2p'] for e in ordered)} "
          f"max={max(e['n_f2p'] for e in ordered)}")
    print("first 5 repos in walk order:", repos[:5])


if __name__ == "__main__":
    main()
