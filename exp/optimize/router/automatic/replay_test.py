"""Completed automatic-router replay tests."""

from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import cast

from rich.console import Console

from exp.cli.optimize.router_candidates import collect_router_candidates
from exp.common.models import load_model_catalog, write_model_catalog
from exp.optimize.router.automatic.preflight import AutomaticRouterOptions
from exp.optimize.router.automatic.replay import find_persisted_automatic_router_replay
from exp.optimize.router.automatic.service import optimize_project_router
from exp.optimize.router.automatic.service_test import (
    _REVISION,
    _TIME,
    _approve_manual_judge,
    _completed_project,
    _RuntimeCatalog,
)
from exp.runtime.models import RuntimeModelCatalog


def test_persisted_replay_requires_the_confirmed_models_and_reasoning(tmp_path: Path) -> None:
    """New candidate, incumbent, and reasoning choices cannot replay an old router."""
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    catalog = load_model_catalog(store.model_catalog_path)
    plan = collect_router_candidates(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO()),
    )
    optimize_project_router(
        store,
        plan,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        options=AutomaticRouterOptions(
            maximum_model_calls=1,
            maximum_judgments=20,
            simulation_maximum_output_tokens=8_000,
        ),
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )
    catalog = load_model_catalog(store.model_catalog_path)
    replay = find_persisted_automatic_router_replay(
        store, judgment_status="human_calibrated", code_revision=_REVISION
    )
    assert replay is not None
    embeddings = tuple(state.embedding_calls)
    completions = tuple(state.completion_calls)
    changes = (
        catalog.roles.model_copy(update={"candidates": ("candidate-a", "world")}),
        catalog.roles.model_copy(update={"incumbent": "candidate-b"}),
        catalog.roles.model_copy(update={"candidate_reasoning_efforts": {"candidate-b": "high"}}),
        catalog.roles.model_copy(update={"world_model_reasoning_effort": "high"}),
        catalog.roles.model_copy(update={"judge_reasoning_effort": "high"}),
    )
    for roles in changes:
        write_model_catalog(store.model_catalog_path, catalog.model_copy(update={"roles": roles}))
        assert (
            find_persisted_automatic_router_replay(
                store, judgment_status="human_calibrated", code_revision=_REVISION
            )
            is None
        ), roles
    write_model_catalog(store.model_catalog_path, catalog)
    assert (
        find_persisted_automatic_router_replay(
            store, judgment_status="human_calibrated", code_revision=_REVISION
        )
        == replay
    )
    assert tuple(state.embedding_calls) == embeddings
    assert tuple(state.completion_calls) == completions
