#!/usr/bin/env python3
"""Run real SWE-bench scenario(s) against the real Docker sandbox.

This builds the environment from scratch when requested: base image, environment image (the real
conda/pip dependency install), instance image (clone repo + checkout + install), then `docker exec`s
the recorded commands.

By default the standup is TRULY COLD: it first purges all local swebench/sweb.* images (so no shared
base layers are reused) and builds with `--no-cache`, so the timed standup is the real from-zero
multi-GB cost every run. Pass `--warm` (optionally with `--cache`) to reuse existing images/layers
for a faster repeat run.

Needs the swebench `.venv` from this directory's README (it imports `swebench` to get the official
Dockerfiles + setup scripts) and a running local Docker daemon. It never imports `wmh`; it reads this
example's `traces.otel.jsonl` and uses the harness's deterministic blake2b split to select held-out
scenarios. If a demo batch is larger than the held-out pool but fits in the full corpus, it switches
to the full corpus so multi-scenario demos can run from stable trace indexes.

By default the stood-up image(s) are wound down in the background after the run (they are multi-GB
and a cold run re-creates them); pass `--keep-image` to keep them.

Usage (from this directory, in the swebench venv):
    .venv/bin/python run_real_scenario.py                          # trace 0, truly cold, then clean up
    .venv/bin/python run_real_scenario.py --warm --cache           # reuse existing image/layers
    .venv/bin/python run_real_scenario.py --trace 2 --keep-image   # a specific trace, keep the image
    .venv/bin/python run_real_scenario.py --trace 0 --scenarios 8 --concurrency 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_DEFAULT_CORPUS = Path(__file__).resolve().parent / "traces.otel.jsonl"


def _attr_map(span: dict[str, Any]) -> dict[str, str]:
    return {a["key"]: a.get("value", {}).get("stringValue", "") for a in span.get("attributes", [])}


def _load_traces(corpus: Path) -> "list[dict[str, Any]]":
    """Group the OTel spans into ordered traces: [{trace_id, instance_id, commands:[...]}].

    Traces are ordered by their earliest span's (startTimeUnixNano, spanId) — IDENTICAL to the wmh
    adapter (wmh/ingest/otel_genai.py), NOT by first appearance in the file. `--trace N` must select
    the SAME scenario index on both sides; ordering by file position would silently diverge from the
    world-model side whenever the corpus is not already start-time sorted.
    """
    spans = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        by_trace.setdefault(span["traceId"], []).append(span)

    def _span_key(span: dict[str, Any]) -> tuple[int, str]:
        return (int(span.get("startTimeUnixNano") or 0), str(span.get("spanId") or ""))

    # Earliest span per trace = its sort key; matches otel_genai's group[0] inter-trace ordering.
    order = sorted(by_trace, key=lambda tid: _span_key(min(by_trace[tid], key=_span_key)))

    traces: list[dict[str, Any]] = []
    for tid in order:
        instance_id = ""
        commands: list[str] = []
        for span in by_trace[tid]:
            attrs = _attr_map(span)
            if "wmh.trace.metadata" in attrs:
                instance_id = json.loads(attrs["wmh.trace.metadata"]).get("instance_id", "")
            args = attrs.get("gen_ai.tool.call.arguments")
            if args:  # an action span (the observation span has no arguments)
                command = json.loads(args).get("command")
                if isinstance(command, str) and command.strip():
                    commands.append(command)
        traces.append({"trace_id": tid, "instance_id": instance_id, "commands": commands})
    return traces


def _holdout(traces: list[dict[str, Any]], train_split: float) -> list[dict[str, Any]]:
    """The held-out traces, by the SAME deterministic blake2b split the wmh harness uses."""
    held: list[dict[str, Any]] = []
    for trace in traces:
        digest = hashlib.blake2b(trace["trace_id"].encode("utf-8"), digest_size=8).digest()
        fraction = int.from_bytes(digest, "big") / 2**64
        if fraction >= train_split:
            held.append(trace)
    return held


def _docker_build(
    tag: str, dockerfile: str, scripts: dict[str, str], platform_str: str, *, no_cache: bool
) -> None:
    """Write a build context (Dockerfile + setup scripts) and `docker build` it, streaming output.

    Raises on a non-zero build. The streamed output IS the real dependency-install log; the caller
    times the whole call so the install cost lands in the comparison.
    """
    with tempfile.TemporaryDirectory(prefix="wmh-swe-build-") as ctx:
        ctx_dir = Path(ctx)
        (ctx_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        for name, body in scripts.items():
            (ctx_dir / name).write_text(body, encoding="utf-8")
        cmd = ["docker", "build", "-t", tag, "--platform", platform_str]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(str(ctx_dir))
        print(f"$ {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f"docker build failed for {tag} (exit {rc})")


def _exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def _pull_image(instance_id: str, platform_str: str, *, no_cache: bool) -> str:
    """`docker pull` the official prebuilt per-instance image, streaming progress; return its tag.

    The standup for the `pull` mode: a multi-GB image download (the environment + its installed
    dependencies, prebuilt) — a real, timed cost, just not a from-scratch compile. This is the path
    that works under emulation (the from-scratch `build` mode's `apt`/conda steps fail under qemu on
    non-x86 hosts). Raises on a failed pull.

    Cold by default (`no_cache=True`): a locally-cached image makes `docker pull` a no-op (~0.5s
    "up to date" check), which would silently hide the real download cost the comparison is about —
    so we remove it first, mirroring the build path's `--no-cache`. Pass `--cache` to reuse it.
    """
    compat = instance_id.replace("__", "_1776_")
    image = f"docker.io/swebench/sweb.eval.x86_64.{compat}:latest".lower()
    if no_cache and _exists(image):
        print(f"--- removing cached image for a cold pull (pass --cache to reuse): {image} ---")
        subprocess.run(["docker", "rmi", "-f", image], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    cmd = ["docker", "pull", "--platform", platform_str, image]
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"docker pull failed for {image} (exit {rc})")
    return image


def _purge_swebench_images() -> int:
    """Remove ALL local swebench/sweb.* images so a pull re-downloads every layer (truly cold).

    `docker rmi <image>` only drops the tag; Docker keeps the underlying layer blobs as long as any
    sibling image references them, so a re-pull of one instance only fetches its ~1 instance-specific
    layer and reuses the ~12 shared base layers (the ~3s "warm-ish" standup). The swebench eval
    images all derive from the same base, so evicting the whole family is what forces the next pull
    to download the full multi-GB image cold. Targeted to the swebench family — it does NOT touch
    unrelated images. Returns the count removed.
    """
    listed = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True,
    )
    refs = [
        r for r in listed.stdout.splitlines()
        if r.startswith("swebench/") or r.startswith(("sweb.eval", "sweb.env", "sweb.base"))
    ]
    removed = 0
    for ref in refs:
        if subprocess.run(["docker", "rmi", "-f", ref],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            removed += 1
    return removed


def _wind_down(images: list[str]) -> None:
    """Remove the stood-up images in a detached background process; return immediately.

    These instance images are multi-GB and a cold run re-creates them anyway, so we don't leave them
    filling the disk. Detached (`Popen`, no wait) so the cleanup doesn't add to the reported run
    time — `docker rmi` can take a moment to unwind layers. `--keep-image` skips this.
    """
    if not images:
        return
    # `docker rmi -f` each image; `|| true` so a missing/shared image doesn't abort the rest.
    script = " ; ".join(f"docker rmi -f {img} >/dev/null 2>&1 || true" for img in images)
    subprocess.Popen(  # noqa: S602 - fixed command, image tags are derived from the dataset spec
        ["bash", "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[winding down {len(images)} image(s) in the background: {', '.join(images)}]")


def _default_mode() -> str:
    """`build` from scratch on a native x86_64 host, else `pull` the prebuilt image.

    SWE-bench's base/env Dockerfiles run `apt`/conda steps that fail under qemu emulation (the apt
    GPG "invalid signature" error) on Apple-Silicon/arm64 hosts, so building from scratch only works
    natively. Everywhere else, pulling the prebuilt image is the robust real-environment standup.
    """
    return "build" if platform.machine().lower() in ("x86_64", "amd64") else "pull"


def _select_traces(
    pool: list[dict[str, Any]], trace_index: int, scenarios: int, *, pool_name: str
) -> list[tuple[int, dict[str, Any]]]:
    """Select traces by the same semantics as the world-model side."""
    if scenarios < 1:
        raise SystemExit("--scenarios must be at least 1")
    if trace_index == -1:
        selected = sorted(enumerate(pool), key=lambda item: len(item[1]["commands"]))[:scenarios]
    elif 0 <= trace_index < len(pool):
        end = trace_index + scenarios
        if end > len(pool):
            raise SystemExit(
                f"--trace {trace_index} with --scenarios {scenarios} exceeds "
                f"{len(pool)} {pool_name} trace(s)"
            )
        selected = [(i, pool[i]) for i in range(trace_index, end)]
    else:
        raise SystemExit(f"--trace {trace_index} out of range; {len(pool)} {pool_name} trace(s)")
    if len(selected) < scenarios:
        raise SystemExit(
            f"requested {scenarios} scenario(s), but only {len(selected)} {pool_name} trace(s) exist"
        )
    return selected


def _run_many(
    args: argparse.Namespace, selected: list[tuple[int, dict[str, Any]]], *, pool_name: str
) -> int:
    """Run several single-scenario invocations concurrently and summarize their timings."""
    # `args.concurrency or args.scenarios` would treat an explicit --concurrency 0 as unset; reject
    # it before defaulting.
    if args.concurrency is not None and args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    workers = args.concurrency or args.scenarios
    warm = args.warm
    cache = args.cache
    if not warm or not cache:
        # A cold single run purges the whole swebench image family. Running that concurrently would
        # make children delete images from under each other, so batch mode uses warm cached semantics.
        warm = True
        cache = True
        print(
            "=== multi-scenario run: forcing --warm --cache to avoid concurrent image purges ==="
        )

    def child_cmd(trace_i: int) -> list[str]:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--corpus",
            str(args.corpus),
            "--trace",
            str(trace_i),
            "--trace-pool",
            pool_name,
            "--train-split",
            str(args.train_split),
            "--dataset",
            str(args.dataset),
            "--mode",
            str(args.mode),
            "--exec-timeout",
            str(args.exec_timeout),
        ]
        if warm:
            cmd.append("--warm")
        if cache:
            cmd.append("--cache")
        if args.keep_image:
            cmd.append("--keep-image")
        return cmd

    print_lock = threading.Lock()

    def stream(pipe, prefix: str) -> None:  # noqa: ANN001 - subprocess pipe object
        try:
            for line in pipe:
                with print_lock:
                    print(f"{prefix}{line}", end="", flush=True)
        finally:
            pipe.close()

    def run_child(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        trace_i, trace = item
        start = time.monotonic()
        proc = subprocess.Popen(
            child_cmd(trace_i),
            cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=stream, args=(proc.stdout, f"[trace {trace_i} out] "), daemon=True
        )
        stderr_thread = threading.Thread(
            target=stream, args=(proc.stderr, f"[trace {trace_i} err] "), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        seconds = time.monotonic() - start
        return {
            "trace_index": trace_i,
            "trace_id": trace["trace_id"],
            "instance_id": trace["instance_id"],
            "commands": len(trace["commands"]),
            "returncode": returncode,
            "seconds": seconds,
        }

    print(
        f"REAL sandbox batch: {len(selected)} SWE-bench scenario(s) from {pool_name}, "
        f"concurrency={workers}\n"
    )
    batch_start = time.monotonic()
    results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_child, item): item for item in selected}
        for fut in cf.as_completed(futures):
            trace_i, trace = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - record a failed child, keep the batch going
                # e.g. the child failed to even spawn (bad interpreter, fd exhaustion). Record it as
                # a failure rather than aborting the batch and orphaning sibling children.
                result = {
                    "trace_index": trace_i,
                    "trace_id": trace["trace_id"],
                    "instance_id": trace["instance_id"],
                    "commands": len(trace["commands"]),
                    "returncode": 1,
                    "seconds": 0.0,
                }
                print(f"[done] trace {trace_i} {trace['instance_id']}: failed to run: {exc}")
                results.append(result)
                continue
            results.append(result)
            status = "ok" if result["returncode"] == 0 else f"exit {result['returncode']}"
            print(
                f"[done] trace {result['trace_index']} {result['instance_id']} "
                f"({result['commands']} commands): {status}, {result['seconds']:.1f}s"
            )

    batch_wall = time.monotonic() - batch_start
    ok = sum(1 for r in results if r["returncode"] == 0)
    work = sum(float(r["seconds"]) for r in results)
    print(
        f"\ndone (REAL sandbox batch): {ok}/{len(results)} ok, "
        f"batch wall {batch_wall:.1f}s, summed runner wall {work:.1f}s"
    )
    for result in sorted(results, key=lambda r: r["trace_index"]):
        status = "ok" if result["returncode"] == 0 else f"exit {result['returncode']}"
        print(
            f"  trace {result['trace_index']}: {result['instance_id']} "
            f"{result['commands']} commands, {status}, {result['seconds']:.1f}s"
        )
        if result["returncode"] != 0:
            print(f"    see streamed trace {result['trace_index']} output above")
    return 0 if ok == len(results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="swe-bench OTel JSONL corpus.")
    parser.add_argument(
        "--trace", type=int, default=0,
        help=(
            "Trace index to replay (default: 0). Uses held-out indexes when possible; switches to "
            "all traces for larger demo batches. Pass -1 for the simplest = fewest commands."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=1,
        help=(
            "Number of scenarios to run. With --trace N, runs N consecutive indexes; with "
            "--trace -1, runs the simplest N traces."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrent scenario runners. Default: --scenarios.",
    )
    parser.add_argument(
        "--trace-pool",
        choices=("auto", "held-out", "all"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trace-id", default=None,
        help="Replay the single trace with this exact trace_id (overrides --trace/--scenarios). "
        "Pins the SAME scenario the world-model side ran, since index order differs between sides.",
    )
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    parser.add_argument(
        "--dataset", default="SWE-bench/SWE-bench_Verified", help="HF dataset for the build spec."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "build", "pull"),
        default="auto",
        help=(
            "How to stand up the env: 'build' compiles base/env/instance from scratch (native "
            "x86_64/Linux only — apt/conda fail under qemu emulation); 'pull' downloads the prebuilt "
            "multi-GB image (works under emulation). 'auto' (default): build on x86_64, else pull."
        ),
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse cached Docker layers (skip already-built images). Default: cold --no-cache.",
    )
    parser.add_argument("--exec-timeout", type=int, default=600, help="Per-command timeout (s).")
    parser.add_argument(
        "--warm",
        action="store_true",
        help=(
            "Skip the truly-cold purge. By default the standup purges ALL local swebench/sweb.* "
            "images first so it re-downloads every layer (no shared-base reuse) and reports the real "
            "cold multi-GB cost. --warm keeps existing images, for a faster repeat run."
        ),
    )
    parser.add_argument(
        "--keep-image",
        action="store_true",
        help=(
            "Keep the stood-up Docker image(s) after the run. Default: wind them down in the "
            "background (these images are multi-GB; a cold run re-creates them anyway)."
        ),
    )
    parser.add_argument(
        "--no-family-purge",
        action="store_true",
        help=(
            "Still cold-fetch THIS instance's own image, but don't purge the whole swebench/sweb.* "
            "family first. Required when several instances run concurrently (each is a distinct "
            "per-instance image, so a family purge would delete the others' images mid-run)."
        ),
    )
    parser.add_argument(
        "--cache-shared",
        action="store_true",
        help=(
            "Reuse the SHARED base + env images across runs (built once, kept), but always "
            "cold-build the per-instance image (clone + checkout + install) and wind only IT down. "
            "This is the correct mode for a fixed-N concurrency sweep, where the same scenarios "
            "repeat at every level: --cache would cache the instance image after the first level and "
            "later levels would rebuild nothing, while fully-cold redundantly rebuilds the shared "
            "base 16x. --cache-shared measures the marginal per-scenario standup at EVERY level. "
            "Implies no family purge (the shared images must survive)."
        ),
    )
    args = parser.parse_args()
    # Docker Desktop/remote daemons can support an older API than the locally installed Docker CLI.
    # Pin the client API unless the caller deliberately chose a different value.
    os.environ.setdefault("DOCKER_API_VERSION", "1.41")

    traces = _load_traces(Path(args.corpus))
    heldout = _holdout(traces, args.train_split)
    if args.trace_id is not None:
        # Pin by stable trace_id: replay EXACTLY the world side's scenario, ignoring pool/index
        # ordering. Always a single scenario, so --scenarios is not consulted.
        by_id = {t["trace_id"]: t for t in traces}
        if args.trace_id not in by_id:
            raise SystemExit(f"--trace-id {args.trace_id} not found in {args.corpus}")
        selected = [(0, by_id[args.trace_id])]
    else:
        if args.trace_pool == "all":
            pool = traces
            pool_name = "all"
        elif args.trace_pool == "held-out":
            pool = heldout
            pool_name = "held-out"
        else:
            pool = heldout or traces
            pool_name = "held-out" if heldout else "all"
        requested_end = args.trace + args.scenarios if args.trace >= 0 else args.scenarios
        if (
            args.trace_pool == "auto"
            and heldout
            and requested_end > len(pool)
            and requested_end <= len(traces)
        ):
            print(
                f"=== requested {args.scenarios} scenario(s) from trace {args.trace}, but only "
                f"{len(pool)} held-out traces exist; using all {len(traces)} traces instead ==="
            )
            pool = traces
            pool_name = "all"
        if not pool:
            raise SystemExit(f"no traces in {args.corpus}; nothing to run")
        selected = _select_traces(pool, args.trace, args.scenarios, pool_name=pool_name)
        if args.scenarios > 1:
            raise SystemExit(_run_many(args, selected, pool_name=pool_name))
    trace = selected[0][1]
    instance_id, commands = trace["instance_id"], trace["commands"]
    if not instance_id:
        raise SystemExit(f"trace {trace['trace_id'][:8]} has no instance_id in metadata")

    mode = _default_mode() if args.mode == "auto" else args.mode

    # Truly-cold by default: evict the whole swebench image family BEFORE the clock starts, so shared
    # base layers can't be reused and the timed standup is a full from-zero download/build. The
    # eviction is teardown of PRIOR state, so it is deliberately not counted in the standup time.
    # `--warm` skips it for a faster repeat run.
    cold = not args.warm
    if cold and not args.no_family_purge and not args.cache_shared:
        print("=== cold standup: purging all local swebench/sweb.* images (no shared-layer reuse) ===")
        n = _purge_swebench_images()
        print(f"--- purged {n} swebench image(s); the standup below is a true cold download ---\n")
    elif cold and args.cache_shared:
        # Shared base+env are reused (built once); only the per-instance image is cold-built below.
        print("=== cold standup (shared base+env reused; per-instance image built cold) ===")
    elif cold:
        # Concurrent-safe cold: this instance's own image is still removed + re-fetched below
        # (no_cache), but we skip the family purge so sibling concurrent runs keep their images.
        print("=== cold standup (this instance only; family purge skipped for concurrency) ===")

    # Start the standup clock HERE — resolving the SWE-bench dataset spec (an HF download + parse on
    # first use) is part of standing the environment up from nothing, so it must be timed, not
    # excluded. (It was previously excluded, which under-counted the real standup by ~100s.)
    start = time.monotonic()
    print(f"=== resolving SWE-bench dataset spec for {instance_id} (first run downloads it) ===")
    # Official SWE-bench build spec: the real base/env/instance Dockerfiles + setup scripts.
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
        from swebench.harness.utils import load_swebench_dataset
    except ImportError as exc:  # pragma: no cover - depends on the isolated venv
        raise SystemExit(
            "swebench is not importable; run via ./run.sh (it sets up the .venv) or install "
            "swebench in packages/environment-capture/swe-bench/.venv. Docker must be running."
        ) from exc

    ds = load_swebench_dataset(args.dataset, "test", instance_ids=[instance_id])
    if not ds:
        raise SystemExit(f"instance {instance_id} not found in {args.dataset}")
    spec = make_test_spec(ds[0])

    print(
        f"\nREAL sandbox: {instance_id} ({len(commands)} commands) — standing up the env "
        f"[mode={mode}], then exec'ing the recorded commands\n"
    )
    # Cold (the default) never reuses anything; --warm + --cache reuses build layers / the image.
    no_cache = cold or not args.cache
    created: list[str] = []  # images this run stood up (for the wind-down cleanup)

    if mode == "build":
        # base image -> env image (the real conda/pip dependency install) -> instance image (clone
        # repo + checkout + install). Each streams its build log and counts toward the clock.
        layers = [
            ("base", spec.base_image_key, spec.base_dockerfile, {}),
            (
                "env (dependency install)",
                spec.env_image_key,
                spec.env_dockerfile,
                {"setup_env.sh": spec.setup_env_script},
            ),
            (
                "instance (repo + install)",
                spec.instance_image_key,
                spec.instance_dockerfile,
                {"setup_repo.sh": spec.install_repo_script},
            ),
        ]
        # --cache-shared: the base + env layers are shared build-once infra (reused across runs and
        # never wound down); only the per-instance layer is cold-built each run and wound down. So a
        # fixed-N sweep measures the marginal per-scenario standup at every level (see the flag help).
        shared_tags = {spec.base_image_key, spec.env_image_key} if args.cache_shared else set()
        for label, tag, dockerfile, scripts in layers:
            reuse = _exists(tag) and (args.cache or (args.cache_shared and tag in shared_tags))
            if reuse:
                print(f"--- {label}: {tag} already built (cached) ---\n")
                continue
            print(f"=== building {label}: {tag} ===")
            # Shared layers may reuse Docker's build cache (built once); the per-instance layer stays
            # truly cold so its clone/checkout/install is a real from-scratch standup every run.
            layer_no_cache = no_cache and tag not in shared_tags
            _docker_build(tag, dockerfile, scripts, spec.platform, no_cache=layer_no_cache)
            if tag not in shared_tags:  # shared base/env persist; only the instance is wound down
                created.append(tag)
            print()
        run_image = spec.instance_image_key
    else:  # pull
        print("=== pulling the prebuilt instance image (multi-GB download — this is the standup) ===")
        run_image = _pull_image(instance_id, spec.platform, no_cache=no_cache)
        created.append(run_image)
        print()
    build_done = time.monotonic()
    print(f"[environment stood up ({mode}) in {build_done - start:.1f}s]\n")

    # Run the recorded scenario in a fresh container off the stood-up instance image.
    container = f"wmh-real-{uuid.uuid4().hex[:8]}"
    rc = subprocess.run(
        ["docker", "run", "-d", "--name", container, "--platform", spec.platform,
         "-w", "/testbed", "--rm", run_image, "sleep", "2h"],
        stdout=subprocess.DEVNULL,
    ).returncode
    if rc != 0:
        raise SystemExit(f"failed to start container (docker run exit {rc})")
    failures = 0
    try:
        for i, command in enumerate(commands):
            print(f"--- step {i} ---\n$ {command}")
            try:
                proc = subprocess.run(
                    ["docker", "exec", "-w", "/testbed", container, "bash", "-lc", command],
                    timeout=args.exec_timeout,
                )
                if proc.returncode != 0:
                    failures += 1
                    print(f"[exit {proc.returncode}]")
            except subprocess.TimeoutExpired:
                failures += 1
                print(f"[timed out after {args.exec_timeout}s]")
            print()
    finally:
        subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    total = time.monotonic() - start
    note = "" if failures == 0 else f" ({failures} command(s) errored/timed out)"
    print(
        f"done (REAL sandbox): standup {build_done - start:.1f}s + "
        f"{len(commands)} commands, {total:.1f}s total{note}"
    )

    # Wind-down: drop the multi-GB image(s) this run stood up, in the background (after timing, so it
    # never counts against the clock). `--keep-image` or `--cache` (you want to reuse it) opt out.
    if not args.keep_image and not args.cache:
        _wind_down(created)


if __name__ == "__main__":
    main()
