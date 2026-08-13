---
name: improve-judge
description: Improve a canonical common-owned judge from reviewed rollout evidence, human labels, and leakage-safe calibration without bypassing the frozen router workflow.
---

# Improve the Judge

Use the common-owned judging contracts and persisted local review state. Every change starts from
a reviewed disagreement in persisted rollout evidence and ends with a regression that would catch
the same failure. Never tune a judge from intuition or from unverified caller data.

## 1. Resolve the canonical evidence

- Start from one local project and its completed rollout, judgment, rubric, label, lineage, and
  calibration artifacts.
- Use `wmo.common.judging.Judge` as the scoring boundary. A judge receives recursively verified
  artifact IDs through `judge_persisted` and returns a structured `Judgment`.
- Use `RubricReview`, `HumanScoreReview`, and `JudgeCalibrationService` to inspect judgments,
  record human score corrections, refresh calibration reports, and explicitly approve
  calibration. CLI and Platform workflows must call these same services rather than create a
  second artifact path.
- Keep raw review and run output under the local project root or `/tmp`. Do not commit customer
  evidence or operator-local outputs.

For a composed router workflow, read the judgments persisted by `wmo.compose_router`. The workflow
injects its approved review supplier, setup supplier, simulator factory, judge, model catalog, and
finite budgets. Do not create a second scoring or evaluation path around that composition seam.

## 2. Classify disagreements

Sample about 20 reviewed cells across the score range and compare every dimension judgment with
the active human score. Classify each actionable miss:

- A false positive scores materially above the human label.
- A false negative scores materially below the human label.
- A protocol failure lacks a valid structured `Judgment` and is infrastructure evidence, not a
  low score.
- A lineage or provenance mismatch is an invalid input and must fail before calibration.

Recheck controls after every change. A new miss on an established control is a regression even if
the target disagreement improves.

## 3. Locate the owning layer

Read the stored prompt identity, model snapshot, raw dimension scores, critique, human history,
and calibration report before proposing a change. Attribute the miss to one owner:

- Rubric or prompt semantics in `wmo/common/judging/`.
- Model capability or request behavior behind the injected common model client.
- Missing or excessive context in the persisted rollout and rubric inputs.
- Calibration behavior in `JudgeCalibrationService` and its leakage-safe grouped metrics.
- Parse, schema, lineage, or provenance validation.

Change one layer per experiment. Preserve immutable IDs and bump the owning prompt or protocol
identity whenever scoring semantics change.

## 4. Encode the expectation first

Add the smallest failing regression beside the common judging owner. Prefer focused tests for the
rubric, LM judge, calibration metrics, provenance, review, or risk-acceptance layer. For workflow
behavior, add coverage beside `wmo/workflow/router_test.py`.

Use a counter-control for every change that could overcorrect. Model comparisons use the same
frozen rollout, rubric, human labels, prompt identity, and lineage split. Never compare runs whose
inputs or failover behavior differ.

## 5. Prove and re-anchor

- Run the focused owner tests twice when model behavior is stochastic.
- Run the common judging and composed-router regressions.
- Confirm controls are stable and that calibration uses fit lineages only. Held-out router
  evidence must remain unopened until the policy is locked.
- Inspect the worst stored disagreements in the new `CalibrationReport`; a green aggregate alone
  is insufficient.
- If semantics changed, report the old and new prompt or protocol identities and the score shift
  on the same frozen evidence.

Stop when a fresh reviewed sample produces no disagreement worth encoding. Keep the canonical
artifact chain intact and do not add compatibility aliases, parallel judge modules, or product
commands outside the approved `wmo config judge` surface.
