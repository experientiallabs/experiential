from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path(__file__).with_name("router_real_build_wm.py")
    spec = importlib.util.spec_from_file_location("router_real_build_wm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_string_set_accepts_json_array_and_jsonl(tmp_path: Path) -> None:
    module = _module()
    rows = [{"task": "one"}, {"task": "two"}]
    array_path = tmp_path / "tasks.json"
    array_path.write_text(json.dumps(rows), encoding="utf-8")
    jsonl_path = tmp_path / "tasks.jsonl"
    jsonl_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    assert module._string_set(array_path, "task") == {"one", "two"}
    assert module._string_set(jsonl_path, "task") == {"one", "two"}
