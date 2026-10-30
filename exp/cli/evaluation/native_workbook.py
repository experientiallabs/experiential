"""The ``native-workbook`` continuation action, which needs the artifact-edits extra.

Its world types read and project workbooks with openpyxl, so ``operator_inputs`` declares the
action with a JSON payload and imports this module only when a spec selects it. Every other
continuation action runs on a base install.
"""

import json
from pathlib import Path

from exp.cli.evaluation.continuation_context import (
    ContinuationContext,
    require_new_operation_file,
)
from exp.optimize.evaluation.executable.workbook_operator import (
    NativeWorkbook,
    native_workbook_capture,
)


class NativeWorkbookOperation(NativeWorkbook):
    """A native workbook world for one development task, and where to save its capture.

    Attributes:
        operation_file: Optional new JSON file for the ready ``capture`` operation. Existing
            files are never overwritten; dry-run does not create it.
    """

    operation_file: Path | None = None


def handle_native_workbook(context: ContinuationContext, request: NativeWorkbookOperation) -> None:
    """Publish the starting workbooks and print a ready native capture operation.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` publishes nothing and
            reports a placeholder input reference and no projection.
        request: Development task, starting files, views, reference answers and output file.

    Returns:
        None. ``input_artifact``, ``capture``, ``verifier`` and ``initial_projection`` are
        printed; the capture operation is saved to ``operation_file`` unless dry-run.
    """
    require_new_operation_file(request.operation_file)
    prepared = native_workbook_capture(
        context.store, request, revision=context.revision, publish=not context.dry_run
    )
    if request.operation_file is not None and not context.dry_run:
        with request.operation_file.open("x", encoding="utf-8") as stream:
            json.dump(prepared["capture"], stream, indent=2)
            stream.write("\n")
    context.console.print_json(data=prepared)
