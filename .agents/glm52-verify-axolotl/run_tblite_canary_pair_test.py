"""Integration test for the matched TBLite pair wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).parent
WRAPPER = HERE / "run_tblite_canary_pair.sh"
REWRITER = HERE / "rewrite_matched_eval_token_budget.py"


def test_configs_are_matched_and_rewritten_before_either_arm_runs(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    cfg_dir = eval_root / "configs"
    jobs_dir = eval_root / "jobs"
    runtime_dir = eval_root / "runtime"
    fake_here = tmp_path / "tblite"
    fake_here.mkdir()
    cfg_dir.mkdir(parents=True)
    jobs_dir.mkdir()
    runtime_dir.mkdir()

    (fake_here / "tblite_env.sh").write_text(
        f'CFG_DIR="{cfg_dir}"\n'
        f'JOBS_DIR="{jobs_dir}"\n'
        f'RUNTIME_DIR="{runtime_dir}"\n'
        f'export HPY="{sys.executable}"\n'
    )
    (fake_here / "make_tblite_cfgs.py").write_text(
        """import argparse, os, pathlib, yaml
p = argparse.ArgumentParser()
p.add_argument('--arm', required=True)
p.add_argument('--served-model', required=True)
p.add_argument('--runs', required=True)
a = p.parse_args()
cfg = {
  'job_name': f'{os.environ["JOB_PREFIX"]}-{a.arm}-run1',
  'n_attempts': 1,
  'n_concurrent_trials': 2,
  'environment': {'type': 'e2b'},
  'agents': [{'name': 'terminus-2', 'override_timeout_sec': 2700,
    'model_name': a.served_model,
    'kwargs': {'temperature': 1.0, 'max_turns': 100, 'enable_summarize': True,
      'model_info': {'max_input_tokens': 53240, 'max_output_tokens': 12288},
      'llm_call_kwargs': {'top_p': 1.0, 'max_tokens': 12288, 'seed': 0}}}],
  'datasets': [{'name': 'openthoughts-tblite', 'ref': '2.0',
    'task_names': ['task-a', 'task-b']}],
}
path = pathlib.Path(os.environ['CFG_DIR']) / f'{cfg["job_name"]}.yaml'
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
"""
    )
    (fake_here / "run_tblite.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python_path="$1"
    "$HPY" - "$python_path" <<'PY'
import json, pathlib, sys, yaml
p = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text())
assert cfg['agents'][0]['kwargs']['model_info']['max_input_tokens'] == 53232
manifest = p.parent.parent / 'runtime' / 'token-budget-manifest.json'
assert manifest.exists()
job = p.parent.parent / 'jobs' / cfg['job_name']
job.mkdir()
(job / 'result.json').write_text(json.dumps({'stats': {'n_completed_trials': 2}}))
PY
"""
    )
    (fake_here / "run_tblite.sh").chmod(0o755)
    (fake_here / "score_tblite.py").write_text(
        """import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument('job')
p.add_argument('--out', required=True)
a = p.parse_args()
pathlib.Path(a.out).write_text('{}\\n')
"""
    )

    env = os.environ | {
        "HERE": str(fake_here),
        "STORAGE_ROOT": str(tmp_path),
        "TOKEN_BUDGET_REWRITER": str(REWRITER),
        "EVAL_ROOT": str(eval_root),
        "CFG_DIR": str(cfg_dir),
        "JOBS_DIR": str(jobs_dir),
        "RUNTIME_DIR": str(runtime_dir),
        "JOB_PREFIX": "candidate",
        "BASE_ARM": "base",
        "BASE_MODEL": "base-model",
        "ADAPTER_ARM": "adapter",
        "ADAPTER_MODEL": "adapter-model",
        "N_RUNS": "1",
        "N_TASKS": "2",
        "TASK_NAMES": "",
    }
    subprocess.run(["bash", str(WRAPPER)], env=env, check=True)

    manifest = json.loads((runtime_dir / "token-budget-manifest.json").read_text())
    assert manifest["safe_max_input_tokens"] == 53232
    assert len(manifest["configs"]) == 2
    for path in cfg_dir.glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert config["agents"][0]["kwargs"]["model_info"]["max_input_tokens"] == 53232
