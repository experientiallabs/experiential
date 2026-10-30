"""Manual hill climbing on an agent's system prompt, one measured edit per round.

A person, or an analyst reading train failures, proposes one edit; the edit is screened on train
lineages and confirmed on held-out test lineages with fresh repeats, and kept only on a paired
gain whose interval excludes zero and whose estimate exceeds an A/A noise floor. Every edit,
kept or reverted, stays as a saved prompt version. Nothing here searches or optimizes
automatically: each round measures exactly the edit it was given.
"""
