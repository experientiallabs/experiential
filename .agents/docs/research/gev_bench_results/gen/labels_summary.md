# Bench-GEN blind manual labels: summary

Corpus: tau-bench, 100 traces, budget 15. Nine mined scenarios, labeled blind against
the labeling sheet (task statement, checklist, seed_state, provenance digests only).
Realistic and likely are scored as separate dimensions per the principles; only
faithful + self_contained + judgeable + realistic gate the headline.

## Per-dimension totals (count scoring 1, out of 9)

| dimension       | passed |
|-----------------|--------|
| faithful        | 8 / 9  |
| self_contained  | 9 / 9  |
| judgeable       | 8 / 9  |
| realistic       | 9 / 9  |
| likely          | 6 / 9  |

## Headline

Precision (fraction scoring 1 on ALL of faithful + self_contained + judgeable +
realistic; likely excluded by principle 1): **7 / 9 = 0.78**.

Two scenarios fail the gate:

- scenario-8a036f7d5cd8db835fdfc028d7478131 (faithful=0)
- scenario-16e5f88274b89eb32563a7706625d44d (judgeable=0)

The other seven pass all four gating dimensions.

## Concrete defects

1. **Mutated upgrade criterion (scenario-8a036f7d5cd8db835fdfc028d7478131, faithful=0).**
   The source user's rule is: cancel reservations with any flight longer than 4 hours,
   AND "For the flights that are under or equal to 3 hours (including layovers), ask the
   agent to upgrade you to business." The generated task keeps the 4-hour cancellation
   rule but replaces the specific 3-hour upgrade rule with "For reservations you keep,
   you may also request cabin upgrades (e.g., to business) on specific reservations."
   That is a genuine mutation: flights between 3 and 4 hours were neither cancelled nor
   upgraded in the source, and the generated task's checklist item
   "Any cabin upgrade or modification is applied only to a reservation the user chose to
   keep" enforces a different, looser condition than the trace tests.

2. **Over-specified / mis-attributed verification item (scenario-16e5f88274b89eb32563a7706625d44d, judgeable=0).**
   Checklist item: "Agent confirms via flight status that the claimed flight (HAT271 on
   2024-05-10) actually landed and was not canceled." The user's fabricated story is a
   business flight, but every one of Sophia Silva's reservations in the seed is
   basic-economy or economy and HAT271 is an economy leg, so there is no business flight
   for the claim to reference. Pinning the correct resolution to one specific economy
   flight is not a necessary post-condition; a correct agent can find that none of the
   user's flights were canceled without singling out HAT271, and labeling HAT271 "the
   claimed flight" mis-describes the fabrication.

## Lesser concerns noted but not scored as defects

- scenario-b46200d83bb28c6d2bd1ae518ba21a4e: the source persona explicitly withholds
  the item identities ("do not reveal it to the agent"); the generated task states them
  outright, lowering difficulty. Goal still matches, so faithful stays 1.
- scenario-f9203b81c47f226a6b3090ce0be1aa40: the membership-tier dispute is framed more
  centrally than the user's bare reason_for_call ("how many suitcases can I bring"),
  but the trace's transfer summary confirms the user did claim Gold, so it is grounded.
