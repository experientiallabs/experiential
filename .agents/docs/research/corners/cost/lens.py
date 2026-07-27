"""The COST-MAX corner's lens spec: which figures, rendered by common/build_corners.py.

Declarative only (charter Amendment: corners hold a lens spec and findings prose; the one
shared runner does every computation). Figure kinds live in the runner's registry; a new
need here means extending the runner, never a standalone script.
"""

from build_corners import FigureSpec, LensSpec

LENS = LensSpec(
    name="cost",
    corner_dir="cost",
    figures=(
        FigureSpec(kind="savings_frontier", filename="savings_vs_fable5.png"),
        FigureSpec(kind="cost_per_task", filename="effective_cost_per_task.png"),
        FigureSpec(kind="dial_curve", filename="dial_cost_curve.png"),
        FigureSpec(
            kind="training_stage",
            filename="training_stage_cost_lens.png",
            params={"lens": "cost"},
        ),
    ),
)
