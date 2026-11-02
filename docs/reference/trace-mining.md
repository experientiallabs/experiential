# Representative-task mining

`exp build` mines a small set of representative tasks from imported traces and freezes them as an
immutable task set. Two selection geometries decide which traces are representative. The geometry is
recorded in the mining spec, so a stored task set always states how it was produced.

| Geometry | `--mine-with` | Provider calls | Needs a question bank |
|---|---|---|---|
| Decision tag | `decision-tag` (default) | yes, one decision model | yes |
| Hashing | `hashing` | none | no |

## You must supply your own question bank

The decision-tag geometry asks a decision model a fixed bank of propositions about each trace. Every
answer is a probability, and the vector of answers is the embedding that clustering reads. **The
geometry can only tell apart what its propositions ask about, so the bank is the single most
important input to mining.**

Experiential ships exactly one bank, `example-v1`. It is deliberately generic and exists so that
tests and documentation can load a bank with no setup. **It is not a starting point for real work and
it is never used as a fallback.** Tagging a corpus with propositions that do not describe that corpus
still returns a matrix, still clusters, and still reports coverage, so a silent substitution would be
indistinguishable from success while costing real provider spend to discover. Naming a bank that is
not installed therefore fails, and the failure names both locations that were searched.

Author your own bank and install it as `.exp/question_banks/<bank_id>.toml`, beside the local model
catalog and equally untracked. Set `EXP_QUESTION_BANK_ROOT` to read banks from another directory.
Resolution checks that directory first and packaged banks second.

### Bank file format

```toml
bank_id = "my-bank-v1"   # must match the file name
version = 1

[[questions]]
question_id = "goal.waits_on_external_system"
family = "goal"
text = "Must the agent wait on a system it does not control before the work can be finished?"
```

`question_id` values must be unique. **File order is the contract**: a tag vector's columns are the
file's question order, so two matrices built from one bank identity are directly comparable. Families
partition a bank and are the unit of selection, so group them contiguously and keep each family to a
single view.

`text` is hashed into each answer's cache key. Rewording a question invalidates every answer
previously bought for that column, so treat wording as frozen once a corpus has been tagged and bump
`version` when it changes.

### Writing propositions that work

Each question is one proposition answered as a probability, so:

- **Ask about the shape of the work, not its subject.** A bank that asks which subject matter a task
  concerns will group tasks by subject, which is what the hashing geometry already does more cheaply.
  Asking what the task *requires* is what lets two tasks in unrelated domains that demand the same
  capabilities group together.
- **One proposition per question.** A probability on a compound proposition is uninterpretable.
- **Prefer propositions that split the corpus.** A proposition true of almost every trace, or almost
  none, contributes nothing to the geometry and is paid for on every trace.

## Views

A bank is asked about one rendered view of each trace, chosen per mining run.

- `task` (default) renders only what was known before the agent acted: the initial instruction,
  request-visible initial context, the tool schemas available at the start, and caller tags. It is
  the same request-visible descriptor the router reads, so it adds no leakage surface, and it is
  substantially smaller than the alternative.
- `observations` renders everything the agent received and nothing it produced: person turns, the
  tool call and result stream with large results condensed, failures, the final answer and the
  recorded outcome. It carries no agent reasoning.

One mining run asks every question about one view. Propositions that need the trace body therefore
belong in their own families so they can be selected and bought separately.

## Tag geometry and clustering

The decision-tag route defaults to the geometry below. Each stage accepts alternatives, and the
chosen values are recorded in the spec.

**Transform** (`l2`, the default) unit-normalizes the row of probabilities, so distance reflects
which propositions a case triggered rather than how many. Selection measures in this geometry, and
it stays continuous deliberately: medoids, facility location, nearest-representative workload
assignment and the coverage report all read it, and none of them can separate two cases whose
coordinates have been rounded to set membership. `hellinger` and `logit` are the other normalizing
choices. `indicator` rounds each probability to zero or one; it is an explicit opt-in rather than a
default, because its rows are not unit vectors and the core-set metric saturates on them.

**Clustering** (`pattern_hierarchy`, the default) does its own rounding instead of asking the
geometry for it. It thresholds each answer at `threshold`, so a case becomes the set of propositions
that fired, then groups those sets by average-linkage agglomeration over Jaccard distance, computed
over the distinct sets rather than over every trace. It requires a decision-tag selection geometry,
because a hashing vector has no reading as a per-tag probability, and it refuses anything else
rather than coercing it.

`cut` is the linkage height the dendrogram is cut at: `0.0` groups only traces that fired exactly
the same set, and raising it rolls near-identical sets together. `threshold` and `cut` are both
recorded in the spec but they are not equally cheap to change. `cut` re-cuts a tree that already
exists, so one tagged corpus supports many granularities with no new provider calls. `threshold`
decides which patterns exist at all, so changing it rebuilds the tree itself.

The backend also publishes that tree and a coarser partition cut at `strata_cut`, so a sampler can
spend budget across broad groups rather than across the finest clusters.

Traces where no proposition fired share a single cluster. Jaccard distance between two empty sets is
undefined, and these are the least distinctive traces in the corpus, so isolating each one would be
backwards.

**Sampling** (`hierarchical`, the default) walks that tree, apportioning a node's budget among its
children by the workload mass under each, so a family of near-identical groups is weighed as one
before any budget is split inside it. At the bottom it draws by greedy facility location, which
takes the cases most typical of their group. `representative` applies the same picker to the flat
clusters instead; the two differ only in where rounding falls. `balanced` and `stratified` give
equal shares per group, which spreads budget widely but spends it on whichever groups win the
rounding when a corpus has far more clusters than the budget has slots.

## The hashing route

`--mine-with hashing` hashes the request-visible descriptor text into a signed vector and needs no
bank, no decision model and no provider call. It clusters descriptors and selects cluster medoids.
This is the route to use when no bank is installed, when mining must not make provider calls, or when
reproducing an existing task set that was built with it.

## Cost

Each decision call carries at most 32 questions, so a bank is asked in consecutive batches of that
size and the rendered view is re-sent with each batch. Mining cost therefore scales with the number
of batches, the size of the chosen view, and the number of traces. Keeping a bank near a multiple of
32 avoids paying for a batch that is nearly empty.

Answers are bought once and stored per trace, question set and model identity, so repeating a mining
run over an unchanged corpus makes no new provider calls.
