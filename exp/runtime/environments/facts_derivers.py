"""The instance-facts derivers that environment backends declare, as one typed definition.

A backend that decodes its own starting files declares one deriver model implementing
:class:`exp.common.tasks.facts.InstanceFactsDeriver`, with a unique ``kind`` literal and its
own bounds. :data:`FactsDeriverDefinition` is the union of those models, so a derivation spec
names its deriver and bounds as typed JSON (``{"kind": "xlsx-workbook", ...}``) and pydantic
selects the backend. A new backend adds its model to the union, discriminated on ``kind``.
"""

from exp.runtime.environments.workbook_facts import WorkbookFactsDeriver

# Every backend-declared deriver. The native workbook backend's is the only one so far; with
# a second, this becomes ``Annotated[A | B, Field(discriminator="kind")]``.
FactsDeriverDefinition = WorkbookFactsDeriver
