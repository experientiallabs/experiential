"""Tests for the command line's mining geometry choices."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from exp.cli.build.mining_options import (
    CLUSTER_WITH_CHOICES,
    CLUSTER_WITH_OPTION,
    MINE_WITH_CHOICES,
    MINE_WITH_OPTION,
    SAMPLE_WITH_CHOICES,
    SAMPLE_WITH_OPTION,
    TAGGER_CONCURRENCY_OPTION,
    build_tagger,
    mining_geometry,
    mining_preparation_settings,
)
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelSnapshot,
)
from exp.common.project import ProjectTracePreparationSettings
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.registry import ResolvedModel, RuntimeModelCatalog
from exp.simulation.build import mining_spec_for
from exp.simulation.mining.clustering import (
    AtomsHierarchyClusterSpec,
    BernoulliMixtureClusterSpec,
    DescriptorClusterSpec,
    KMeansClusterSpec,
    PatternHierarchyClusterSpec,
)
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.features import DecisionTagSelectionSpec, HashingSelectionSpec
from exp.simulation.mining.questions import EXAMPLE_BANK_ID
from exp.simulation.mining.sampling import (
    DiversitySamplingSpec,
    MedoidSamplingSpec,
    ProportionalSamplingSpec,
    RepresentativeSamplingSpec,
    StratifiedSamplingSpec,
)
from exp.simulation.mining.service import MiningSpec


def _spec(**overrides: str | None) -> MiningSpec:
    """Build a spec from the declared flag defaults with the named flags changed."""
    arguments: dict[str, str | None] = {
        "mine_with": "hashing",
        "cluster_with": None,
        "sample_with": None,
        "tagger": None,
    }
    arguments.update(overrides)
    if arguments["mine_with"] == "decision-tag" and "question_bank" not in overrides:
        arguments["question_bank"] = EXAMPLE_BANK_ID
    return mining_geometry(**arguments)  # ty: ignore[invalid-argument-type]


def test_naming_hashing_is_the_provider_free_default() -> None:
    """--mine-with hashing is the one-flag way back to a build that calls nothing."""
    assert _spec() == MiningSpec()


def test_the_declared_defaults_are_the_tag_route_and_its_settled_pipeline() -> None:
    """What a bare `exp build` mines is a product decision, so the flag defaults are pinned.

    The library default stays the provider-free one, because a stored task set mined with
    ``MiningSpec()`` has to keep reproducing itself. The command line is where the product choice
    lives, and it chooses the paid route.
    """
    assert MINE_WITH_OPTION.default == "decision-tag"
    assert CLUSTER_WITH_OPTION.default is None
    assert SAMPLE_WITH_OPTION.default is None

    spec = _spec(mine_with="decision-tag", tagger="clef-decisions")

    assert spec.cluster == PatternHierarchyClusterSpec()
    assert spec.sampling == RepresentativeSamplingSpec()
    assert isinstance(MiningSpec().selection, HashingSelectionSpec)
    assert MiningSpec().cluster == DescriptorClusterSpec()
    assert MiningSpec().sampling == MedoidSamplingSpec()


def test_each_cluster_name_selects_its_backend() -> None:
    """A name that silently mapped to the default would be worse than an error."""
    assert _spec(cluster_with="kmeans").cluster == KMeansClusterSpec(seed=42)
    assert _spec(cluster_with="atoms-hierarchy").cluster == AtomsHierarchyClusterSpec(seed=42)
    assert _spec(cluster_with="descriptor").cluster == DescriptorClusterSpec()
    tagged = _spec(
        mine_with="decision-tag", tagger="clef-decisions", cluster_with="bernoulli-mixture"
    )
    assert tagged.cluster == BernoulliMixtureClusterSpec(seed=42)
    patterns = _spec(
        mine_with="decision-tag", tagger="clef-decisions", cluster_with="pattern-hierarchy"
    )
    assert patterns.cluster == PatternHierarchyClusterSpec()


def test_each_sampling_name_selects_its_strategy() -> None:
    """These are the discretionary half of selection, so the mapping has to be exact."""
    assert _spec(sample_with="medoid").sampling == MedoidSamplingSpec()
    assert _spec(sample_with="proportional").sampling == ProportionalSamplingSpec()
    assert _spec(sample_with="diversity").sampling == DiversitySamplingSpec()
    assert _spec(sample_with="stratified").sampling == StratifiedSamplingSpec()


def test_the_decision_tag_geometry_records_the_bank_it_was_given() -> None:
    """Which bank was read is what makes a tag geometry reproducible, so it is never defaulted."""
    spec = _spec(mine_with="decision-tag", tagger="clef-decisions")

    assert isinstance(spec.selection, DecisionTagSelectionSpec)
    assert spec.selection.bank_id == EXAMPLE_BANK_ID
    assert spec.selection.view == "task"
    assert spec.selection.transform == "l2"
    assert isinstance(MiningSpec().selection, HashingSelectionSpec)


def test_the_decision_tag_geometry_refuses_to_run_without_a_bank() -> None:
    """Discovering a missing bank after a corpus has been tagged costs real money."""
    with pytest.raises(typer.BadParameter, match="needs --question-bank"):
        _spec(mine_with="decision-tag", tagger="clef-decisions", question_bank=None)


def test_a_bank_without_the_tag_geometry_is_refused() -> None:
    """A flag that is read by nothing is a misunderstanding worth reporting."""
    with pytest.raises(typer.BadParameter, match="only read by"):
        _spec(question_bank=EXAMPLE_BANK_ID)


def test_a_bank_nobody_installed_is_refused_at_parse_time() -> None:
    """The bank is local configuration, so a build must stop before it tags anything."""
    with pytest.raises(typer.BadParameter, match="no question bank named"):
        _spec(mine_with="decision-tag", tagger="clef-decisions", question_bank="absent-v1")


def test_the_seed_reaches_the_backend_that_uses_it() -> None:
    """Reproducing a mined task set depends on the seed being recorded, not defaulted."""
    arguments: dict[str, object] = {
        "mine_with": "hashing",
        "cluster_with": "kmeans",
        "sample_with": "medoid",
        "tagger": None,
        "seed": 7,
    }

    assert mining_geometry(**arguments).cluster == KMeansClusterSpec(seed=7)  # ty: ignore[invalid-argument-type]


def test_a_tag_geometry_without_a_tagger_is_refused_before_any_work() -> None:
    """Catching this at parse time costs nothing; catching it later costs a tagged corpus."""
    with pytest.raises(typer.BadParameter, match="needs --tagger ALIAS"):
        _spec(mine_with="decision-tag")


def test_a_tagger_that_would_go_unused_is_refused() -> None:
    """Naming a paid alias that is never called is a misconfiguration, not a preference."""
    with pytest.raises(typer.BadParameter, match="only used by --mine-with decision-tag"):
        _spec(tagger="clef-decisions")


def test_the_mixture_backend_without_a_tag_geometry_is_refused() -> None:
    """It reads probabilities, and the hashing geometry has none to give it."""
    with pytest.raises(typer.BadParameter, match="decision_tag selection geometry"):
        _spec(cluster_with="bernoulli-mixture")


def test_the_pattern_backend_without_a_tag_geometry_is_refused() -> None:
    """It thresholds per-tag probabilities, and the hashing geometry has none to give it."""
    with pytest.raises(typer.BadParameter, match="decision_tag selection geometry"):
        _spec(cluster_with="pattern-hierarchy")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("mine_with", "tags"),
        ("cluster_with", "spectral"),
        ("sample_with", "random"),
    ],
)
def test_an_unknown_name_lists_the_ones_that_exist(flag: str, value: str) -> None:
    """An error message is part of the interface, so it has to say what to type."""
    with pytest.raises(typer.BadParameter, match="must be one of"):
        _spec(**{flag: value})


def test_the_choice_lists_match_the_specs_they_map_to() -> None:
    """A name added to one list and not the mapping would fail only when someone used it."""
    assert MINE_WITH_CHOICES == ("hashing", "decision-tag")
    for name in CLUSTER_WITH_CHOICES:
        tagged = name in {"bernoulli-mixture", "pattern-hierarchy"}
        spec = _spec(
            cluster_with=name,
            mine_with="decision-tag" if tagged else "hashing",
            tagger="clef-decisions" if tagged else None,
        )
        assert spec.cluster.kind == name.replace("-", "_")
    for name in SAMPLE_WITH_CHOICES:
        assert _spec(sample_with=name).sampling.kind == name


def test_preparation_settings_round_trip_to_the_same_spec() -> None:
    """A project stores names, so rebuilding from them must give back the same geometry."""
    from exp.simulation.build import mining_spec_for

    settings = mining_preparation_settings(
        source_kind="otlp",
        mine_with="decision-tag",
        cluster_with="bernoulli-mixture",
        sample_with="stratified",
        tagger="clef-decisions",
        question_bank=EXAMPLE_BANK_ID,
        seed=11,
    )

    assert settings.mining_geometry == "decision_tag"
    assert settings.mining_tagger_alias == "clef-decisions"
    assert settings.mining_bank == EXAMPLE_BANK_ID
    assert mining_spec_for(settings) == _spec(
        mine_with="decision-tag",
        cluster_with="bernoulli-mixture",
        sample_with="stratified",
        tagger="clef-decisions",
    ).model_copy(update={"cluster": BernoulliMixtureClusterSpec(seed=11)})


def _decision_alias() -> ResolvedModel:
    """Resolve a fake alias that serves the typed decisions wire and no chat."""
    return ResolvedModel(
        alias="clef-decisions",
        snapshot=ModelSnapshot(
            provider="fixture",
            model_id="decisions-fixture",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities_sha256="a" * 64,
            connection_sha256="b" * 64,
        ),
        capabilities=ModelCapabilities(),
        client=_DecisionClient(),  # ty: ignore[invalid-argument-type]
        embedding_client=None,
    )


class _DecisionClient:
    """A resolved client exposing one typed decisions wire profile."""

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the typed decisions profile this alias serves."""
        return GatewayWireProfile(
            dialect="typesafe_systemone",
            url="https://example.test/v1/systemone",
            decisions_url="https://example.test/v1/systemone",
            model_id="clef",
        )


def _catalog() -> ModelCatalog:
    """Return the smallest valid catalog, which the patched resolver never reads from."""
    return ModelCatalog(
        connections={"fixture": ConnectionConfig(provider="openai", api_key_env="FIXTURE_API_KEY")},
        models={
            "clef-decisions": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="fixture",
                model="decisions-fixture",
            )
        },
    )


def _built_tagger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: int
) -> DecisionsTraceTagger:
    """Build a tagger over a fake resolved alias, with no provider client and no credential."""
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "resolve",
        lambda self, alias, **kwargs: _decision_alias(),
    )
    tagger = build_tagger(
        "clef-decisions",
        catalog=_catalog(),
        root=tmp_path,
        project="fixture",
        **overrides,
    )
    assert isinstance(tagger, DecisionsTraceTagger)
    return tagger


def test_a_built_tagger_parallelizes_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tagged build is hours of pure waiting when it buys one decision at a time."""
    assert _built_tagger(monkeypatch, tmp_path).maximum_concurrency == 8


def test_the_concurrency_choice_reaches_the_tagger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The flag is only worth having if the number a user types is the number used."""
    assert _built_tagger(monkeypatch, tmp_path, maximum_concurrency=3).maximum_concurrency == 3
    assert _built_tagger(monkeypatch, tmp_path, maximum_concurrency=1).maximum_concurrency == 1


def test_the_concurrency_option_defaults_to_the_shared_allowance() -> None:
    """The flag and the library default are one number, not two that can drift apart."""
    assert TAGGER_CONCURRENCY_OPTION.default == 8


def test_concurrency_never_enters_the_mined_evidence() -> None:
    """It cannot change an answer, so it must not change a task set's identity either.

    An execution knob inside the mining spec or the persisted preparation settings would travel
    into every digest built from them, and two builds that differ only in how fast they tagged
    would stop replaying each other.
    """
    fields = (*MiningSpec.model_fields, *ProjectTracePreparationSettings.model_fields)

    assert not [name for name in fields if "concurrency" in name]
    assert _spec(mine_with="decision-tag", tagger="clef-decisions") == mining_spec_for(
        mining_preparation_settings(
            source_kind="otlp",
            mine_with="decision-tag",
            cluster_with=None,
            sample_with=None,
            tagger="clef-decisions",
            question_bank=EXAMPLE_BANK_ID,
        )
    )


def test_an_unnamed_backend_takes_each_geometry_s_default_and_a_named_one_is_literal() -> None:
    """A name is never reinterpreted, which is the whole reason the unset value is not a name.

    A run that asks for descriptor clustering on the tag route gets descriptor clustering. What
    the tag route defaults to is only what it does when nothing was asked for.
    """
    tagged: dict[str, str | None] = {"mine_with": "decision-tag", "tagger": "clef-decisions"}

    assert _spec(**tagged).cluster == PatternHierarchyClusterSpec()
    assert _spec(**tagged).sampling == RepresentativeSamplingSpec()
    assert _spec(**tagged, cluster_with="descriptor").cluster == DescriptorClusterSpec()
    assert _spec(**tagged, sample_with="medoid").sampling == MedoidSamplingSpec()
    assert _spec().cluster == DescriptorClusterSpec()
    assert _spec().sampling == MedoidSamplingSpec()


def test_leaving_a_flag_unnamed_is_persisted_as_a_default_rather_than_a_name() -> None:
    """Freezing today's answer into the settings would make the default impossible to change."""
    settings = mining_preparation_settings(
        source_kind="otlp",
        mine_with="decision-tag",
        cluster_with=None,
        sample_with=None,
        tagger="clef-decisions",
        question_bank=EXAMPLE_BANK_ID,
    )

    assert settings.mining_cluster is None
    assert settings.mining_sampling is None
    assert mining_spec_for(settings).cluster == PatternHierarchyClusterSpec()
