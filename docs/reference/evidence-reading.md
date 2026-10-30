# Objective-driven investigation reading

For a new learning project, `exp build PROJECT --traces PATH --source SOURCE --root ROOT
--evidence-reading` enables bounded evidence reading after a simulator discrepancy. The
investigator asks an explicit question, the configured memory-recall decision model locates
support and contradictions in eligible prior memory records, and the investigator may refine
its question before synthesizing a scoped hypothesis. Source records are returned unchanged;
reader judgments do not establish native truth.

The default allows two reading rounds, three sources per round and at most eight additional
logical calls per discrepancy. Preflight includes the expanded physical-attempt ceiling. Large records stay unread rather than being truncated. Reports in the saved
research records distinguish inspected sources, selected passages, unread sources and budget
limits. All inspected sources contribute provenance, including sources the reader rejects.
Full original passages live in a separate immutable evidence receipt. Recalled notes contain
only its reference and compact coverage metadata, so evidence is not recursively copied into
future memory-recall requests. This profile also pins a 24,000-byte memory-recall request
budget, including a framing reserve. Recall packs whole records in shortlist order, skips
records that do not fit, and records packed and omitted IDs. If none fit, it returns no
selected memory without calling the reader; oversized required current context refuses
before dispatch. The same limit follows the frozen memory into evaluation.

The learning plan pins this policy. Omitted flags reuse it; enabling readers after learning is
refused. Compare policies in separate projects. Each trace uses its prior pinned memory; frozen
evaluation performs no investigations or reading passes.
Saved investigations and reader judgments replay; use `--retry-incomplete` to resume an interrupted reading pass. Saved planner, reader and
synthesis responses are reused; unknown-outcome retries are explicitly recorded and qualified.

This mode reads the caller-admitted prior memory catalog in its supplied order. It does not
search arbitrary raw traces, implement a persistent wiki, or certify a hypothesis by model
agreement. Source interpretation remains a measured model behavior.
