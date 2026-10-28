"""The backend-declared deriver definition selects the native workbook deriver by kind."""

import pytest
from pydantic import TypeAdapter, ValidationError

from exp.common.tasks.facts import InstanceFactsDeriver
from exp.runtime.environments.facts_derivers import (
    DEFAULT_MAX_FACT_BYTES,
    FactsDeriverDefinition,
    WorkbookFactsDeriver,
)


def test_definition_selects_the_workbook_deriver_and_its_bounds() -> None:
    """A spec's JSON names the deriver kind; bounds default and stay within their limits."""
    adapter = TypeAdapter(FactsDeriverDefinition)
    deriver = adapter.validate_python({"kind": "xlsx-workbook"})
    assert isinstance(deriver, WorkbookFactsDeriver)
    assert deriver.max_bytes == DEFAULT_MAX_FACT_BYTES and not deriver.allow_unrepresented
    contract: InstanceFactsDeriver = deriver
    assert contract.kind == "xlsx-workbook"
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "xlsx-workbook", "max_bytes": 64 * 1024 * 1024})
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "unknown"})
