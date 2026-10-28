"""The instance-facts derivers that environment backends declare, as one typed definition.

A backend that decodes its own starting files declares one deriver model implementing
:class:`exp.common.tasks.facts.InstanceFactsDeriver`, with a unique ``kind`` literal and its
own bounds. :data:`FactsDeriverDefinition` is the union of those models, so a derivation spec
names its deriver and bounds as typed JSON (``{"kind": "xlsx-workbook", ...}``) and pydantic
selects the backend. A new backend adds its model to the union, discriminated on ``kind``.

Selecting a deriver imports no backend: each model's ``derive`` imports its decoder, so specs
validate on a base install and only running a deriver needs the backend's optional extra.
"""

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.tasks.facts import InstanceFacts

DEFAULT_MAX_FACT_BYTES = 1024 * 1024
MAX_FACT_BYTES = 8 * 1024 * 1024


class WorkbookFactsDeriver(ContractModel):
    """The native workbook backend's instance-facts deriver.

    Attributes:
        kind: Deriver name in derivation specs.
        max_bytes: Bound on the encoded value's canonical JSON size; larger workbooks are
            refused. Default 1 MiB, at most 8 MiB.
        allow_unrepresented: Accept archive parts whose content lies outside the encoding,
            listing each in the facts, instead of refusing the workbook.
    """

    kind: Literal["xlsx-workbook"] = "xlsx-workbook"
    max_bytes: int = Field(default=DEFAULT_MAX_FACT_BYTES, ge=1024, le=MAX_FACT_BYTES, strict=True)
    allow_unrepresented: bool = Field(default=False, strict=True)

    def derive(self, path: str, payload: bytes) -> InstanceFacts:
        """Validated facts about one starting workbook; see ``derive_workbook_facts``.

        Raises:
            ValueError: The artifact-edits extra (openpyxl) is not installed.
            WorkbookFactsRefused: The workbook cannot be represented exactly or within bounds.
        """
        try:
            from exp.runtime.environments.workbook_facts import derive_workbook_facts
        except ModuleNotFoundError as error:
            if error.name not in {"openpyxl", "et_xmlfile"}:
                raise
            # Not an InstanceFactsError: a missing extra is no refusal of this input.
            raise ValueError(
                "xlsx-workbook facts need the artifact-edits extra; "
                "install experiential[artifact-edits]"
            ) from error

        return derive_workbook_facts(
            path, payload, max_bytes=self.max_bytes, allow_unrepresented=self.allow_unrepresented
        )


# Every backend-declared deriver. The native workbook backend's is the only one so far; with
# a second, this becomes ``Annotated[A | B, Field(discriminator="kind")]``.
FactsDeriverDefinition = WorkbookFactsDeriver
