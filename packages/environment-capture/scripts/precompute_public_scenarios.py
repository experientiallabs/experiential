"""Precompute example open-loop reconstructions for the public world models.

The platform's unauthenticated catalog shows, per model, a few hard-coded "reconstruction"
scenarios: a recorded action, the real observation the environment produced, and the world
model's predicted observation. Recomputing these live on every page view would be slow and
costly, so we run them once here and vendor the result into the platform frontend build.

Each step's golden data comes from the model's own indexed corpus (`sample_steps`), and the
prediction from `step_open_loop` (teacher-forced: predict, then advance from ground truth). The
headline fidelity shown in the UI is the model's official held-out score from its card, not
recomputed here, so self-retrieval on indexed steps never inflates the number a visitor sees.

Run once, with the models' serve provider available (Bedrock):

    uv run python packages/environment-capture/scripts/precompute_public_scenarios.py \
        --out ../platform-public-catalog/apps/web/data/public-scenarios.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmh.config.card import load_card
from wmh.core.render import render_action
from wmh.engine.loader import load_world_model

MODELS_GLOB = "*/models/*"
SCENARIOS_PER_MODEL = 2
STEPS_PER_SCENARIO = 3
OBSERVATION_CLIP = 900


def _clip(text: str) -> str:
    flat = text.strip()
    return flat if len(flat) <= OBSERVATION_CLIP else flat[: OBSERVATION_CLIP - 1] + "…"


def _model_dirs(capture_root: Path) -> list[Path]:
    return sorted(p for p in capture_root.glob(MODELS_GLOB) if (p / "config.toml").is_file())


def precompute_model(model_dir: Path, *, scenarios: int, steps_per: int) -> dict | None:
    """Build the vendored scenarios for one model, or None when it has too few indexed steps."""
    card = load_card(model_dir)
    if card is None:
        return None
    wm, _provider = load_world_model(model_dir)
    total = scenarios * steps_per
    sampled = wm.sample_steps(total)
    if len(sampled) < steps_per:
        return None

    replayed: list[dict] = []
    for step in sampled:
        session = wm.new_session(task=step.task, seed_state=step.state_before, enrich=False)
        predicted = wm.step_open_loop(session.id, step.action, step.observation)
        replayed.append(
            {
                "action": _clip(render_action(step.action)),
                "golden": _clip(step.observation.content or ""),
                "predicted": _clip(predicted.content or ""),
                "isErrorGolden": step.observation.is_error,
                "isErrorPredicted": predicted.is_error,
            }
        )

    grouped = [replayed[i : i + steps_per] for i in range(0, len(replayed), steps_per)]
    out_scenarios = [
        {"id": f"s{i}", "label": f"Reconstruction {i + 1}", "steps": group}
        for i, group in enumerate(grouped)
        if group
    ]
    # Suggested "example tool calls" for the live playground: the first distinct rendered actions.
    suggestions: list[str] = []
    for step in sampled:
        label = _clip(render_action(step.action))
        if label and label not in suggestions:
            suggestions.append(label)
        if len(suggestions) >= 4:
            break

    return {
        "slug": card.name,
        "fidelity": card.fidelity.score if card.fidelity else None,
        "scenarios": out_scenarios,
        "suggestions": suggestions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Path to write the vendored JSON.")
    parser.add_argument("--only", default=None, help="Precompute a single model slug (for probes).")
    parser.add_argument("--scenarios", type=int, default=SCENARIOS_PER_MODEL)
    parser.add_argument("--steps", type=int, default=STEPS_PER_SCENARIO)
    args = parser.parse_args()

    capture_root = Path(__file__).resolve().parents[1]  # packages/environment-capture
    out: list[dict] = []
    for model_dir in _model_dirs(capture_root):
        if args.only and model_dir.name != args.only:
            continue
        print(f"precomputing {model_dir.name} ...", flush=True)
        try:
            entry = precompute_model(model_dir, scenarios=args.scenarios, steps_per=args.steps)
        except Exception as exc:  # noqa: BLE001 - one bad model must not sink the whole run
            print(f"  skipped {model_dir.name}: {exc}", flush=True)
            continue
        if entry is None:
            print(f"  skipped {model_dir.name}: no card or too few steps", flush=True)
            continue
        out.append(entry)
        print(f"  {model_dir.name}: {len(entry['scenarios'])} scenarios", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} models to {out_path}", flush=True)


if __name__ == "__main__":
    main()
