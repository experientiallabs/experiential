# Bench-VERIFY disagreement attribution

Every case where the majority judge verdict differed from the deterministic bird-sql outcome, hand
read (final SQL vs reference SQL, transcript rows). Categories: `judge_wrong` (the judge reached the
wrong verdict on a checkable case), `recorded_label_wrong` (the deterministic grade is a grader
artifact, the judge was actually right), `assertion_underspecified` (the gold reference does not
faithfully encode the task, so the success spec is self-contradictory), `genuinely_ambiguous` (a
legitimate judgment call).

The bird-sql grader compares result rows as positional tuples (order-insensitive multiset of rows,
but column position within a row is significant); confirmed in
`environment_capture/benchmarks/bird_sql.py` (`_canon_row`, `rows_match`). This matters below.

## Attribution counts

- judge_wrong: 5
- assertion_underspecified: 1
- recorded_label_wrong: 1
- genuinely_ambiguous: 1

Of the 5 judge_wrong and the 1 genuinely_ambiguous, 4 share one mechanism: the judge cannot execute
SQL, so it cannot tell that two structurally different queries return identical rows (false-fails)
or that a plausible query returns wrong rows (false-passes). Feeding the judge the reference RESULT
ROWS (or an execution tool) is the single highest-value fix; see the report.

## False-pass (judge said PASS, recorded FAIL)

### bird-train-188 -> judge_wrong
Question asks for the top nine districts and "the number of female account holders". Agent selects
only `A2` (district) and groups by `A2`; gold selects `A2, COUNT(client_id)` and groups by
`district_id, A2`. The agent output is missing the count column the question requests and merges
same-named districts, so the row tuples cannot match. The judge unanimously (3/3) credited it, so it
overlooked a missing requested column plus a grouping change that alters the counts.

### bird-train-192 -> assertion_underspecified
Question: "list the district ... and the ... percentage unemployment rate increment." Gold selects
ONLY the rate expression and omits the district; the agent selects `A3, rate`. The recorded FAIL is
an arity mismatch against gold, but gold itself does not encode the district the question explicitly
asks for, so the assertion ("answers the question AND equals gold") is internally contradictory. The
judge credited the query that better answers the question; the fault is the reference, not the judge.

### bird-train-141 -> recorded_label_wrong
Agent selects `ID, age, Diagnosis`; gold selects `Diagnosis, ID, age`. Identical data, different
COLUMN ORDER. Because the grader compares positional tuples, this scores 0.0 even though the answer
is semantically correct and the question fixes no column order. The judge (which grades by meaning)
said PASS and is right; the deterministic label is a column-order artifact of the grader.

### bird-train-42 -> judge_wrong
Top 5 schools per county by reading score. Agent uses `ROW_NUMBER()`, gold uses `RANK()`. On tied
scores at the rank-5 boundary these return different school sets, which is exactly why the execution
match fails. The judge (3/3) treated the two window functions as equivalent, missing the tie-handling
difference.

## False-fail (judge said FAIL, recorded PASS)

### bird-train-45 -> judge_wrong
Agent filters `schools` directly on `District = 'San Bernardino City Unified'`, `SOC='62'`,
`DOC='54'`; gold joins `frpm` and filters on `City='San Bernardino'`. The recorded execution match
confirms both return the same admin emails. The judge could not verify that two differently filtered
queries coincide and defaulted to fail (1/3). Root cause: SQL equivalence is not decidable by reading
the queries; the judge needs the reference rows or an execution tool.

### bird-train-37 -> judge_wrong
Agent joins `income` to `member` directly; gold takes a redundant four-way join through
`event`/`attendance` to the same members. On this data both return the same funded students plus
amounts (recorded PASS). The judge (1/3) penalized the missing event/attendance joins that are a
no-op on the result. Same needs-execution mechanism.

### bird-train-32 -> genuinely_ambiguous
Agent omits gold's `WHERE position = 'Member'` filter. On this database every matching row already has
`position='Member'`, so the result is identical (recorded PASS), but the omitted predicate is a real
semantic difference that would diverge on other data. Whether an answer that coincidentally matches
counts as "correct" is a legitimate judgment call; the judge's stricter reading (0/3) is defensible.

### bird-train-3 -> judge_wrong
Percentage of 'Bad' superheroes plus Marvel count. Agent uses `LEFT JOIN` with `SUM(CASE ...)` over
all rows; gold filters to Bad with an `INNER JOIN` and a `(SELECT COUNT(*) FROM superhero)`
denominator. On this data both compute the same two numbers (recorded PASS). The judge (0/3) was
thrown by the restructuring and the numeric-equivalence reasoning it cannot execute.
