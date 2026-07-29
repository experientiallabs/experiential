"""The swe bench-defaults lens: REAL SWE-bench episodes through the shared corners runner.

Declarative only, like every other lens (charter Amendment: one shared runner, corners hold a
lens spec and findings prose). Two things make this leg different from the world-model cohorts
beside it, and both are set explicitly here rather than inherited:

- PROVENANCE is `real_episode`. Every cell is a real mini-swe-agent episode inside that
  instance's real SWE-bench container, and the runner's labeling rules forbid a computed delta
  from crossing provenance, so these labels are load bearing.
- The JUDGE is not a judge. It is SWE-bench's own test suite (FAIL_TO_PASS plus PASS_TO_PASS),
  a deterministic verifier, so this leg carries none of the LLM-judge caveats the tau and
  terminal legs do. An empty patch is scored 0 by the same verifier's empty-patch bucket, never
  dropped.

Three caveats the findings prose has to repeat, because a figure footnote cannot carry them:

- The step limit is pinned at 75 against mini-swe-agent's shipped 250, uniformly for every
  candidate, because a 217-step thrash probe cost 15.4 minutes of wall clock on this emulated
  arm64 box. Solve rates here are therefore NOT comparable to published SWE-bench numbers.
- The gpt-5.6 candidates ran with reasoning OFF, because /v1/chat/completions refuses function
  tools any other way (see REASONING_OPT_OUT in the runner). A weaker and cheaper configuration
  than their default, and not comparable to a reasoning-on one.
- Anthropic candidates are priced WITH prompt-cache credit, which this program had to implement
  (`AnthropicProvider.complete_chat`) before the anchor could run a tool-calling harness at all.
  Whether each other backend's automatic caching actually engaged is reported per candidate from
  the recorded cache tokens rather than assumed.

No dial_curve figure: that kind draws the D-DIAL anchor set measured on the world-model cohort,
and mixing this corpus onto that axis would blend provenances. The savings frontier and the
cost-per-task chart are dataset-native and carry the routed rungs this leg fits.
"""

from build_corners import REAL_EPISODE, FigureSpec, LensSpec
from data import main_checkout

COHORT = "swe-8bd6c3a11dea"
"""The cohort these figures were rendered from, named so a rerun cannot silently pick another.

A cohort is one harness, one pool, one pin, one step limit. Pointing the lens at the parent
directory would let a later cohort's arm be swept into the same figure, which is the
cells-measured-under-two-harnesses error the runner's cohort discipline exists to prevent.
"""

LENS = LensSpec(
    name="bench-defaults-swe",
    corner_dir="bench-defaults-swe",
    dataset_root=str(main_checkout() / ".wmo" / "jt" / "bench-defaults" / "swe" / COHORT),
    dataset_label="swe-bench-verified-real",
    judge_label="swe-bench test suite (deterministic verifier; FAIL_TO_PASS + PASS_TO_PASS)",
    provenance_label=REAL_EPISODE,
    split_label="pinned-eval-20",
    figures=(
        # Cost savings is this program's primary headline, so the frontier leads.
        FigureSpec(kind="savings_frontier", filename="savings_vs_fable5.png"),
        FigureSpec(kind="cost_per_task", filename="effective_cost_per_task.png"),
    ),
)
