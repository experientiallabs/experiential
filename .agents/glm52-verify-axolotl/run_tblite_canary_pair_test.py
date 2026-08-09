from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("n_tasks", "task_names", "expected"),
    (("10", "", "10"), ("", "a,b,c", "3")),
)
def test_scorer_uses_selected_task_denominator(
    tmp_path: Path,
    n_tasks: str,
    task_names: str,
    expected: str,
) -> None:
    here = tmp_path / "eval"
    here.mkdir()
    log = tmp_path / "hpy.log"
    hpy = tmp_path / "hpy"
    hpy.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >>"$HPY_LOG"\n')
    hpy.chmod(0o755)
    (here / "tblite_env.sh").write_text(
        'HPY="$STUB_HPY"\n'
        'CFG_DIR="$STUB_ROOT/cfg"\n'
        'JOBS_DIR="$STUB_ROOT/jobs"\n'
        'RUNTIME_DIR="$STUB_ROOT/runtime"\n'
        'mkdir -p "$CFG_DIR" "$JOBS_DIR" "$RUNTIME_DIR"\n'
    )
    (here / "make_tblite_cfgs.py").write_text("")
    run_tblite = here / "run_tblite.sh"
    run_tblite.write_text("#!/usr/bin/env bash\nexit 0\n")
    run_tblite.chmod(0o755)

    script = Path(__file__).with_name("run_tblite_canary_pair.sh")
    env = {
        **os.environ,
        "HERE": str(here),
        "STUB_HPY": str(hpy),
        "HPY_LOG": str(log),
        "STUB_ROOT": str(tmp_path),
        "N_RUNS": "1",
        "N_TASKS": n_tasks,
        "TASK_NAMES": task_names,
        "JOB_PREFIX": "test",
        "ADAPTER_ARM": "adapter",
        "ADAPTER_MODEL": "adapter-model",
    }
    subprocess.run(["bash", str(script)], check=True, env=env)

    score_calls = [
        line for line in log.read_text().splitlines() if "score_tblite.py" in line
    ]
    assert len(score_calls) == 2
    assert all(f"--total-tasks {expected}" in line for line in score_calls)
