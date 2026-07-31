# Codeforces long-context effort development protocol

Status: frozen before provider execution on 2026-07-31.

## Objective

Correct the execution confound discovered in the first Codeforces source run,
develop a latency-neutral effort router on the original 160-task development
cohort, and evaluate the frozen route once on a separate untouched Codeforces
confirmation cohort before any DeepSWE transfer.

## Development corpus and boundary

The development corpus remains the 160-task `open-r1/codeforces-cots`
`solutions_py_decontaminated` cohort at revision
`39ac85c150806230473c70ad72c31f6232fe3f41`, with task SHA-256
`c99ac2b6637cc3c689f0c105938bc2932a40d7b3e9ed738239e10fa2b3c764c6`.
Published generations remain unloaded. DeepSWE outcomes remain sealed.

The first source run showed that 32,768 output tokens truncated 85 max and 33
xhigh cells. This development protocol raises `max_output_tokens` to 131,072
for every effort. All other prompts, tests, model identity, isolation, reward,
cost provenance, and retry rules remain unchanged.

## Corrected development matrix

- Model: `gpt-5.6-luna`
- Efforts: `low`, `medium`, `high`, `xhigh`, `max`
- Attempts: two
- Tasks: 160
- Cells: 1,600
- Maximum output tokens: 131,072
- Reward: fraction of frozen tests passed

Run one four-cell smoke first on two development tasks whose earlier high
effort calls were truncated, at xhigh and max, attempt 0. The smoke must prove
completed provider status, exact model attestation, gradeable outcomes, raw and
code hashes, and zero-pending resume. If the corrected smoke still truncates,
stop without launching the matrix.

## Development and confirmation

The corrected development outcomes may be used adaptively to choose features,
algorithm family, thresholds, and arm set. They may not authorize DeepSWE.
After development, freeze exactly one lightweight route rule and a separate
160-task confirmation cohort from previously unused eligible Codeforces tasks.
The confirmation cohort must exclude every development task and remain unseen
until the rule and promotion criteria are frozen.

Promotion requires positive reward advantage over the matched task-blind
mixture, a positive contest-cluster bootstrap lower bound, no static-arm
dominance, a failed shuffled-label control, complete grading, and no target
leakage. Only a passing untouched confirmation can authorize the single
DeepSWE transfer.
