"""The closed Astra alias authors build-time tags only; every other role refuses it."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.astra_role import refuse_astra_outside_its_roles
from exp.optimize.evaluation.variance.pipeline_spec import COUNCIL_ACTORS
from exp.optimize.evaluation.variance.rubric_evolution import EvolutionSpec
from exp.optimize.evaluation.variance.rubric_evolution_test import settings, write_council_catalog
from exp.optimize.evaluation.variance.tag_monitor_epoch import TagEpochMonitor
from exp.optimize.evaluation.variance.tag_monitor_review import AstraSettings
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.tagging import TraceTaggingError

ASTRA = "astra"


def _project(tmp_path: Path) -> ProjectStore:
    """A project whose catalog adds a closed ``astra`` chat alias beside the council."""
    project = ProjectStore(tmp_path, "case")
    project.model_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    write_council_catalog(project.model_catalog_path.parent)
    catalog = load_model_catalog(project.model_catalog_path)
    models = dict(catalog.models)
    models[ASTRA] = catalog.models["judge"].model_copy(update={"model": "gpt-6-astra"})
    write_model_catalog(project.model_catalog_path, catalog.model_copy(update={"models": models}))
    return project


def _registration(**changes: JsonValue) -> JsonObject:
    """A registration-shaped document with Astra in all its build-time roles."""
    document: JsonObject = {
        "preparation": {"build": {"spec": {"decision_alias": "clef"}}},
        "pipeline": {
            "traceml": {"decision_alias": "clef-hosted"},
            "mining": {"actors": ["actor-glmflash"], "judge": {"model_alias": "deepseek-judge"}},
        },
        "slots": {
            "tag-banks": {"author": {"model_alias": ASTRA}, "decision_alias": "clef-hosted"},
            "state-rag": {"author_alias": ASTRA, "decision_alias": "clef-hosted"},
            "tag-monitor-corpus": {
                "astra": {"author": {"model_alias": ASTRA}, "astra_alias": ASTRA}
            },
        },
        "astra_alias": ASTRA,
    }
    document.update(changes)
    return document


def test_astra_is_accepted_in_the_build_time_tag_roles() -> None:
    """Tag-bank author, state-tag author and a checkpoint 1 reviewer may be the closed alias."""
    refuse_astra_outside_its_roles(_registration(), ASTRA)


@pytest.mark.parametrize(
    ("section", "role"),
    [
        ("pipeline", {"mining": {"actors": ["actor-glmflash", ASTRA]}}),
        ("pipeline", {"mining": {"judge": {"model_alias": ASTRA}}}),
        ("pipeline", {"traceml": {"decision_alias": ASTRA}}),
        ("preparation", {"build": {"spec": {"decision_alias": ASTRA}}}),
        ("preparation", {"build": {"reviewer": {"model_alias": ASTRA}}}),
    ],
)
def test_astra_is_refused_in_actor_judge_decision_and_review_roles(
    section: str, role: JsonObject
) -> None:
    """An actor, a judge, a decision alias or the families reviewer naming Astra is refused."""
    with pytest.raises(ValueError, match="build-time tags only"):
        refuse_astra_outside_its_roles(_registration(**{section: role}), ASTRA)


def test_astra_is_refused_in_the_rubric_evolution_author_and_epoch_monitor_slot() -> None:
    """Training-time roles inside the rubric-evolution slot refuse the closed alias."""
    slots = _registration()["slots"]
    assert isinstance(slots, dict)
    for setting in (
        {"registration": {"spec": {"author": {"model_alias": ASTRA}}}},
        {"monitor": {"astra": {"author": {"model_alias": ASTRA}}}},
    ):
        document = _registration(slots={**slots, "rubric-evolution": setting})
        with pytest.raises(ValueError, match="build-time tags only"):
            refuse_astra_outside_its_roles(document, ASTRA)


def test_a_slot_cannot_declare_another_astra_alias() -> None:
    """Astra settings must declare the registration's own alias."""
    with pytest.raises(ValueError, match="declares Astra alias"):
        refuse_astra_outside_its_roles(_registration(), None)


def test_closed_authors_are_refused_by_astra_settings_epoch_monitor_and_rubric_author(
    tmp_path: Path,
) -> None:
    """Undeclared, Astra refuses a closed author; checkpoint 6 and the rubric author always do."""
    project = _project(tmp_path)
    closed = settings(project, ASTRA)
    with pytest.raises(ValueError, match="open-weight author or the declared astra_alias"):
        AstraSettings(author=closed)
    declared = AstraSettings(author=closed, astra_alias=ASTRA)
    assert not declared.open_weight
    with pytest.raises(ValueError, match="checkpoint 6"):
        TagEpochMonitor(astra=declared)
    open_weight = AstraSettings(author=settings(project, "glm-5.3-flash"))
    assert TagEpochMonitor(astra=open_weight).astra == open_weight
    with pytest.raises(ValueError, match="open-weight author"):
        EvolutionSpec(
            auditors=tuple(settings(project, alias) for alias in COUNCIL_ACTORS),
            author=closed,
        )


def test_a_closed_chat_alias_cannot_be_a_decision_model(tmp_path: Path) -> None:
    """The decision tagger refuses an alias that is not a native decision model."""
    project = _project(tmp_path)
    catalog = load_model_catalog(project.model_catalog_path)
    resolved = RuntimeModelCatalog(catalog, environment={"HELD_OUT_TEST_KEY": "x"}).resolve(ASTRA)
    with pytest.raises(TraceTaggingError, match="native decision client|typed decision"):
        DecisionsTraceTagger.for_alias(resolved)
