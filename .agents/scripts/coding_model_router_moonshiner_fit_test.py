"""Unit tests for the external Moonshiner effort fitter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_moonshiner_fit.py")
    spec = importlib.util.spec_from_file_location("moonshiner_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_family_keeps_related_templates_in_one_group() -> None:
    assert module._family("bash-it-one", "debug") == "bash-it"
    assert (
        module._family("behavior-dependency-planning-0018", "dependency-planning")
        == "behavior-dependency-planning"
    )
    assert module._family("vcf91-0100", "bugfix") == "vcf91"
    assert module._family("w18-sc14-review", "review") == "w18-sc14"


def test_rank_uses_average_tie_ranks() -> None:
    assert module._rank(np.asarray([3.0, 1.0, 1.0, 2.0])).tolist() == [
        3.0,
        0.5,
        0.5,
        2.0,
    ]


def test_monotonicity_detects_lower_pass_followed_by_higher_failure() -> None:
    data = module.Data(
        task_ids=["monotone", "violation"],
        groups=["a", "b"],
        texts=["a", "b"],
        rewards=np.asarray(
            [
                [0, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
            ],
            dtype=np.float64,
        ),
        costs=np.zeros((2, 5), dtype=np.float64),
        structural=np.zeros((2, 20), dtype=np.float64),
    )
    report = module._monotonicity(data)
    assert report["tasks_with_monotonicity_violation"] == 1
    assert report["violation_task_ids"] == ["violation"]
