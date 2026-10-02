"""Shared ingestion executes without loading runtime or simulation packages."""

import subprocess
import sys
from pathlib import Path

from exp.common.traces.ingest.persistence_test import _source


def test_file_ingestion_is_independent_of_runtime_and_simulation(tmp_path: Path) -> None:
    """A fresh Python process can ingest and restore tool traces using common alone."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from exp.common.traces.ingest.persistence import ingest_traces, read_ingested_traces

root = Path(sys.argv[2])
summary, receipt = ingest_traces(
    "powerset", root=root, source_format="chat-json", path=Path(sys.argv[1])
)
assert receipt is not None
restored = read_ingested_traces(root, receipt.import_id)
assert len(restored.traces) == summary.trace_count == 20
assert len(restored.issues) == 1
assert all(trace.tools[0].name == "lookup" for trace in restored.traces)
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in ("exp.runtime", "exp.simulation", "exp.optimize", "exp.cli", "exp_gateway_native")
)
""",
            str(_source(tmp_path)),
            str(tmp_path / "state"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
