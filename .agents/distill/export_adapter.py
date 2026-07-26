"""Export a Tinker LoRA checkpoint to the h100-dev-box-6 vLLM eval server.

Pipeline (run from the Mac; heavy transfers never touch the Mac):
  1. Resolve the tinker:// sampler path to a signed archive URL via the
     Tinker REST client (needs TINKER_API_KEY locally; the URL is passed to
     the box over ssh stdin and never logged).
  2. The box curls the archive directly from Tinker (datacenter bandwidth;
     ~1.5 GB Nano archive took ~108 s) and extracts it under
     ``<remote_dir>/_raw/<name>``.
  3. The box converts it to standard PEFT layout with
     ``tinker_cookbook.weights.build_lora_adapter`` using the local HF
     snapshot of the base model (no base download; FP8-snapshot-based
     conversion is byte-identical to BF16-based conversion — verified
     2026-07-24 on the Nano headline adapter, sha256 match).
  4. POST /v1/load_lora_adapter on the box-local vLLM server (the Azure NSG
     does not expose the vLLM port publicly, so this also goes over ssh).
  5. Print the served model name to use as the eval endpoint ``model``.

Usage (from the distill worktree root; venv has tinker + tinker_cookbook):
  .venv/bin/python .agents/distill/export_adapter.py \
      "tinker://<run-id>/sampler_weights/<ckpt>" --name pi-step-0040

  # dry-run: fetch + convert + validate on the box, but skip the server call
  .venv/bin/python .agents/distill/export_adapter.py "tinker://..." --dry-run

  # unload a previously loaded adapter (server has --max-loras 1: unload the
  # old adapter before loading the next one)
  .venv/bin/python .agents/distill/export_adapter.py --unload pi-step-0040

TINKER_API_KEY is taken from the environment. If it is unset, the script reads
it from an env file (never printed): ``$WMO_ENV_FILE`` when set, otherwise
``<repo>/.env.local`` and then ``<repo>/../platform/.env.local``. A missing env
file is only a warning; the hard failure is the key still being unavailable.

Server ops (box h100-dev-box-6, azureuser@40.80.93.150):
  start:  tmux new-session -d -s nemotron-serve \
              "/nvme/work/nemotron-serve/serve.sh 2>&1 | tee /nvme/work/nemotron-serve/serve.log"
  stop:   tmux kill-session -t nemotron-serve
  logs:   tail -f /nvme/work/nemotron-serve/serve.log
  box-side conversion venv: /nvme/work/nemotron-serve/.export-venv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_BASE_SNAPSHOT = (
    "/nvme/hf_cache/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8/"
    "snapshots/7d7e5797b8a3c7abbab54033b6004e93e8b6bc91"
)
DEFAULT_HOST = "azureuser@40.80.93.150"
DEFAULT_REMOTE_DIR = "/nvme/lora-adapters"
DEFAULT_PORT = 8100
BOX_EXPORT_PY = "/nvme/work/nemotron-serve/.export-venv/bin/python"

# Env file that can supply TINKER_API_KEY when the environment does not. Resolved from the
# repo, never from one machine's home directory: an explicit override first, then this
# checkout's own .env.local, then the sibling platform checkout that holds the shared keys.
ENV_FILE_VAR = "WMO_ENV_FILE"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_file_candidates() -> list[Path]:
    """Env files to search for TINKER_API_KEY, most specific first."""
    override = os.environ.get(ENV_FILE_VAR)
    if override:
        return [Path(override).expanduser()]
    return [REPO_ROOT / ".env.local", REPO_ROOT.parent / "platform" / ".env.local"]


def _ensure_tinker_api_key() -> None:
    """Put TINKER_API_KEY in the environment, reading an env file only as a fallback.

    An env file that is absent or silent about the key is a warning naming that file, not a
    failure: the only hard requirement is that the key ends up in the environment somehow.
    """
    if os.environ.get("TINKER_API_KEY"):
        return
    problems: list[str] = []
    for path in _env_file_candidates():
        if not path.exists():
            problems.append(f"{path} (no such file)")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(?:export\s+)?TINKER_API_KEY=(.*)$", line.strip())
            if m:
                os.environ["TINKER_API_KEY"] = m.group(1).strip().strip("'\"")
                return
        problems.append(f"{path} (no TINKER_API_KEY line)")
    print("warning: no env file supplied TINKER_API_KEY: " + "; ".join(problems))
    sys.exit(
        "TINKER_API_KEY is not set. Export it, add a TINKER_API_KEY=... line to one of the "
        f"files above, or point {ENV_FILE_VAR} at an env file that has one."
    )


def _slug(tinker_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tinker_path.split("/")[-1])


def _ssh(host: str, cmd: str, *, stdin: str | None = None, timeout: int = 900) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", host, cmd],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        sys.exit(
            f"remote command failed (exit {proc.returncode}):\n"
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _server_api(host: str, port: int, route: str, payload: dict) -> dict:
    body = shlex.quote(json.dumps(payload))
    out = _ssh(
        host,
        f"curl -s -m 600 -X POST http://127.0.0.1:{port}{route} "
        f"-H 'Content-Type: application/json' -d {body}",
        timeout=630,
    ).strip()
    try:
        return json.loads(out) if out else {}
    except json.JSONDecodeError:
        # vLLM returns plain text ("Success: ...") on the lora admin routes.
        return {"raw": out}


def _signed_archive_url(tinker_path: str) -> str:
    _ensure_tinker_api_key()
    import tinker  # deferred: import needs the venv

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    resp = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
    return resp.url


def export(args: argparse.Namespace) -> None:
    name = args.name or _slug(args.tinker_path)
    raw_dir = f"{args.remote_dir}/_raw/{name}"
    peft_dir = f"{args.remote_dir}/{name}"

    print(f"[1/4] resolving signed archive URL for {args.tinker_path}")
    url = _signed_archive_url(args.tinker_path)

    print(f"[2/4] box: fetch + extract -> {raw_dir}")
    _ssh(
        args.host,
        f'URL=$(cat); set -e; mkdir -p {shlex.quote(raw_dir)}; '
        f'cd {shlex.quote(raw_dir)}; '
        f'curl -sf -o archive.tar "$URL"; tar xf archive.tar; rm archive.tar; '
        f'ls adapter_model.safetensors adapter_config.json >/dev/null',
        stdin=url,
    )

    print(f"[3/4] box: build_lora_adapter (base={args.base_snapshot}) -> {peft_dir}")
    convert_py = (
        "import sys\n"
        "from tinker_cookbook.weights import build_lora_adapter\n"
        f"build_lora_adapter(base_model={args.base_snapshot!r},\n"
        f"                   adapter_path={raw_dir!r},\n"
        f"                   output_path={peft_dir!r})\n"
        "import json\n"
        f"cfg = json.load(open({peft_dir!r} + '/adapter_config.json'))\n"
        "print('rank', cfg['r'], 'rank_pattern', cfg.get('rank_pattern'))\n"
    )
    # CUDA_VISIBLE_DEVICES="" — conversion is CPU-only; the GPUs belong to the
    # running vLLM server and torch would otherwise OOM trying to use them.
    out = _ssh(
        args.host,
        f"set -e; rm -rf {shlex.quote(peft_dir)}; "
        f"HF_HOME=/nvme/hf_cache TMPDIR=/nvme/tmp CUDA_VISIBLE_DEVICES= "
        f"{BOX_EXPORT_PY} - <<'PYEOF'\n"
        f"{convert_py}PYEOF",
    )
    if out.strip():
        print(f"      {out.strip()}")

    if args.dry_run:
        print("[dry-run] skipping load_lora_adapter")
        print(f"served model name would be: {name}")
        return

    print(f"[4/4] load_lora_adapter name={name}")
    resp = _server_api(
        args.host,
        args.port,
        "/v1/load_lora_adapter",
        {"lora_name": name, "lora_path": peft_dir},
    )
    blob = json.dumps(resp).lower()
    if "error" in blob or "fail" in blob:
        sys.exit(f"load_lora_adapter failed: {json.dumps(resp, indent=2)}")
    print(f"      server response: {resp}")

    print(f"\nserved model name for eval endpoint: {name}")
    print("(base model stays available as: nemotron-3-super)")


def unload(args: argparse.Namespace) -> None:
    resp = _server_api(
        args.host, args.port, "/v1/unload_lora_adapter", {"lora_name": args.unload}
    )
    print(f"unload response: {resp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tinker_path", nargs="?", help="tinker://.../sampler_weights/... path")
    ap.add_argument("--name", help="served adapter name (default: derived from path)")
    ap.add_argument("--base-snapshot", default=DEFAULT_BASE_SNAPSHOT,
                    help="box-local HF snapshot dir of the base model")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + convert on the box; skip the server call")
    ap.add_argument("--unload", metavar="NAME",
                    help="unload a previously loaded adapter and exit")
    args = ap.parse_args()

    if args.unload:
        unload(args)
        return
    if not args.tinker_path:
        ap.error("tinker_path is required unless --unload is given")
    export(args)


if __name__ == "__main__":
    main()
