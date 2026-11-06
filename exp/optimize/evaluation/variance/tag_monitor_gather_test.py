"""Gatherers read saved records only; without them they return None."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
from exp.optimize.evaluation.variance.frontdoor_executors import SlotContext
from exp.optimize.evaluation.variance.tag_monitor_gather import gather_dense, gather_mining


def test_post_pipeline_gatherers_wait_for_saved_records(tmp_path: Path) -> None:
    """No pipeline stage and no hybrid assessment saved means no inputs yet."""
    project = ProjectStore(tmp_path, "case")
    context = SlotContext(project, cast(BootstrapPreparationSpec, None), "test")
    assert gather_mining(context) is None and gather_dense(context) is None
