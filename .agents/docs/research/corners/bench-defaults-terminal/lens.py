"""bench-defaults/terminal lens: REAL Terminal-Bench-2 episodes through the shared runner.

Declarative only, per the charter amendment (one shared runner; corners hold a lens spec and
findings prose). This is the third dataset to render through `build_corners.py` and the FIRST
whose rows are `real_episode` rather than `wm_simulated`: every cell is a real TB2 task graded
by the benchmark's own pytest verifier in an E2B container, driven by harbor's terminus-2 with
the scaffold pinned identically across all 16 candidates.

Read the label carefully against its sibling. The `tb2-cost` corner is named for TB2 but its
main corpus is the org's HF terminal-tasks bundle, 280 one-shot bash traces (its own README
files that relabel). THIS lens is the actual Terminal-Bench-2 benchmark: harbor's registry
dataset `terminal-bench` version 2.0, 20 pinned task ids x 2 episodes. The two corpora are
never blended, which is exactly what the dataset label on every record is for.

Judge label names the verifier, not a rubric: there is no LLM judge anywhere on this path, so
no judge-calibration caveat applies to these numbers (a real difference from both sibling
corners, whose rubric judges were un-meta-eval'd on their corpora).
"""

from build_corners import FigureSpec, LensSpec
from data import main_checkout

LENS = LensSpec(
    name="bench-defaults-terminal",
    corner_dir="bench-defaults-terminal",
    dataset_root=str(main_checkout() / ".wmo" / "jt" / "bench-defaults" / "terminal"),
    dataset_label="terminal-bench-2",
    judge_label="TB2 task verifier (pytest via ctrf.json; no LLM judge)",
    # The docstring's central claim, now carried by the artifact too: these are real TB2
    # episodes, not world-model rows. The runner's default (wm_simulated) mislabeled the
    # first render; caught by the master before publication (2026-07-29).
    provenance_label="real_episode",
    split_label="the 20 pinned TB2 tasks (real benchmark episodes)",
    arms=("identity",),
    figures=(
        FigureSpec(kind="savings_frontier", filename="savings_vs_fable5.png"),
        FigureSpec(kind="cost_per_task", filename="effective_cost_per_task.png"),
    ),
)
