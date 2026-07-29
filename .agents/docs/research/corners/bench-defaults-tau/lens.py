"""The tau bench-defaults lens: REAL tau2 episodes through the shared corners runner.

Declarative only, like every other lens (charter Amendment: one shared runner, corners hold a
lens spec and findings prose). What makes this lens different from the tau lenses beside it is
provenance: every other matrix in this program was measured INSIDE a world model and scored by an
LLM judge, and this one is Sierra's actual benchmark scored by tau2's own reward. The runner's
labeling rules forbid a computed delta from crossing provenance, so the labels here are load
bearing rather than decorative, and they are set explicitly instead of inherited.

Two caveats the findings prose has to repeat, because a figure footnote cannot carry them:

- 7 of the 20 pinned tasks include tau2's NL-assertion judge in their reward basis, so the leg
  is not uniformly deterministic. Rows record which, and the deterministic subset is reported
  separately.
- The gpt-5.6 candidates ran with their reasoning budget OFF, because their API refuses function
  tools any other way (see REASONING_OFF_CANDIDATES in the real-episode runner). Their numbers
  are not comparable to a reasoning-on configuration.

No dial_curve figure: that kind draws the D-DIAL anchor set of the world-model cohort, and mixing
this corpus into it would blend provenances on one axis. The savings frontier and the
cost-per-task chart are dataset-native and carry the routed rungs this leg fits.
"""

from build_corners import REAL_EPISODE, FigureSpec, LensSpec
from data import main_checkout

LENS = LensSpec(
    name="bench-defaults-tau",
    corner_dir="bench-defaults-tau",
    dataset_root=str(main_checkout() / ".wmo" / "jt" / "bench-defaults" / "tau"),
    dataset_label="tau2-real",
    judge_label="tau2 reward (7/20 pinned tasks include tau2's NL-assertion judge)",
    provenance_label=REAL_EPISODE,
    split_label="pinned-eval-20",
    figures=(
        # Cost savings is the primary headline for this program, so the frontier leads.
        FigureSpec(kind="savings_frontier", filename="savings_vs_fable5.png"),
        FigureSpec(kind="cost_per_task", filename="effective_cost_per_task.png"),
    ),
)
