"""Public CLI composition for deriving instance facts from declared task input files."""

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.learning_summary import save_details
from exp.optimize.evaluation.fidelity.fact_derivation import (
    FactsDerivation,
    attach_trace_package,
    derive_facts,
    save_derived_facts,
)


def handle_derive_facts(context: ContinuationContext, spec: FactsDerivation) -> None:
    """Derive facts, attach them to the trace package, save them and print where they went.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` writes nothing.
        spec: Deriver, declared inputs, and the optional trace package to annotate.

    Returns:
        None. Counts, refusals and the written paths are printed to the console.
    """
    console, store = context.console, context.store
    result = derive_facts(spec)
    attached = (
        attach_trace_package(spec, result, write=not context.dry_run)
        if spec.traces is not None
        else frozenset()
    )
    derived = [row for row in result.inputs if row.refusal is None]
    largest = max((row.fact_bytes or 0 for row in derived), default=0)
    lines = [
        f"Instance facts ({spec.deriver.kind}): {len(derived)} of {len(result.inputs)} inputs "
        f"validated for {len(result.facts)} traces, {len(result.refused)} refused; largest "
        f"{largest:,} bytes. No provider calls."
    ]
    lines.extend(f"Refused {row.trace} {row.path}: {row.refusal}" for row in result.refused)
    if spec.traces is not None:
        lines.append(f"Trace package: {len(attached)} conversations receive instance_facts.")
    for line in lines:
        console.print(line, markup=False)
    if context.dry_run:
        console.print("Dry run: nothing written.")
        return
    facts_path = save_derived_facts(store, result)
    console.print(f"Facts for runtime definitions: {facts_path}", markup=False)
    if spec.output is not None:
        console.print(
            f"Import it: exp build {context.project} --traces {spec.output} --source chat-json",
            markup=False,
        )
    report = save_details(store, "facts", result.identity, "\n".join(lines) + "\n")
    console.print(f"Report: {report}", markup=False)
