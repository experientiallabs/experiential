"""Planning prices the full schedule without runtime imports, credentials or provider calls."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation import workbook_runtime
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.costs import decision_cost, model_cost, runtime_costs, total_cost
from exp.optimize.evaluation.workbook_runtime_test import workbook_definition


def test_known_rates_price_every_attempt_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing credentials and an unrelated unpriced judge do not erase actor estimates."""
    write_catalog(tmp_path)
    project = ProjectStore(tmp_path, "case")
    catalog = load_model_catalog(project.model_catalog_path)
    row = catalog.models["world"]
    assert row.capabilities is not None
    priced = row.model_copy(
        update={
            "capabilities": row.capabilities.model_copy(
                update={
                    "input_cost_per_million_tokens_usd": 1.0,
                    "output_cost_per_million_tokens_usd": 2.0,
                }
            )
        }
    )
    write_model_catalog(
        project.model_catalog_path,
        catalog.model_copy(update={"models": {**catalog.models, "world": priced}}),
    )
    monkeypatch.delenv("HELD_OUT_TEST_KEY", raising=False)
    monkeypatch.setattr(
        "exp.runtime.models.registry.RuntimeModelCatalog.resolve",
        lambda *args: pytest.fail("cost review must not resolve credentials"),
    )
    actor = model_cost(project, "world", 2, 4096)
    assert actor.usd is not None
    assert actor.attempts == 2 and actor.usd == pytest.approx(0.416384)
    reader = decision_cost(project, "world", 2)
    assert reader.usd is not None and reader.usd > actor.usd
    assert decision_cost(project, "judge", 1).usd is None
    judge = model_cost(project, "judge", 1, 4096)
    assert judge.usd is None
    assert total_cost((actor, judge)) is None
    assert total_cost((actor, actor.scaled(3))) == pytest.approx(4 * actor.usd)


def test_native_factory_is_zero_and_custom_code_is_never_executed(tmp_path: Path) -> None:
    """A known local XLSX backend costs no inference; unknown custom code stays unpriced."""
    project, spec, _ = workbook_definition(tmp_path)
    assert (
        runtime_costs(
            project,
            sha256_bytes(Path(workbook_runtime.__file__).read_bytes()),
            spec.model_dump(mode="json"),
        )
        == ()
    )
    assert total_cost(()) == 0
    arbitrary = tmp_path / "custom.py"
    arbitrary.write_text('raise AssertionError("must not import during costing")')
    rows = runtime_costs(project, sha256_bytes(arbitrary.read_bytes()), {})
    assert total_cost(rows) is None and "undeclared" in rows[0].name
