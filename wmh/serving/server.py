"""Local FastAPI backend — the live environment agents call over HTTP.

Routes are namespaced by world model name (`/world_models/{name}/...`) so one backend can serve
several named models at once — from one or more store roots (`.wmh`, `examples/<task>`, ...).
Each route is a thin transport over an in-process `WorldModel`; the CLI and the API share the
same code path. `GET /world_models` also returns each model's `card.json` (when present), and
the `/world_models/builds` routes run new builds server-side so the website's build-your-own
flow can watch progress over SSE.

The backend is also the *reward* server for RL training: `POST .../sessions/{id}/score` judges the
session's rollout (task + history) with `EpisodeRewardJudge`, returning the scalar episode reward
(GRPO/PPO/REINFORCE++), per-step rewards, and a critique string (SDPO's teacher feedback) — so a
training scaffold gets environment and reward behind one API.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from wmh.config import ARTIFACT_DIR, WorldModelStore
from wmh.config.card import ModelCard, load_card
from wmh.core.types import Action, EnvState, Observation, Session
from wmh.engine.loader import load_world_model
from wmh.engine.world_model import WorldModel
from wmh.optimize.reward import EpisodeScore
from wmh.serving.builds import BuildManager, BuildRouteRequest, BuildSnapshot
from wmh.tracking import RunRecord

logger = logging.getLogger(__name__)

# Only browser origins matching this may reach the API (see the CORS note in create_app). Kept as
# a module constant so the multipart-upload route can re-check it: multipart POSTs are CORS
# "simple requests" that skip preflight, so CORSMiddleware alone does not stop a foreign page from
# writing to disk — the route validates the Origin header itself.
ALLOWED_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

# Reject an upload whose body exceeds this; keeps a stray/malicious POST from filling the disk.
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class NewSessionRequest(BaseModel):
    task: str | None = None
    seed_state: EnvState | None = None


class NewSessionResponse(BaseModel):
    session_id: str
    state: EnvState


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    # The env state after this step, so clients can render scratchpad/structured state without a
    # follow-up GET of the whole (linearly growing) session on every step.
    state: EnvState


class ModelCardEntry(BaseModel):
    name: str
    card: ModelCard | None = None


class ModelsResponse(BaseModel):
    world_models: list[str]  # names-only shape, kept for existing clients
    models: list[ModelCardEntry]


class NewBuildResponse(BaseModel):
    build_id: str


class UploadResponse(BaseModel):
    path: str


def resolve_model_dirs(artifact_dirs: Sequence[str], names: list[str] | None) -> dict[str, Path]:
    """Map model name -> artifact dir across every root, failing fast on ambiguity.

    A name appearing under two roots is an error (serving would silently pick one); a requested
    `names` entry that no root provides is an error listing what is available.
    """
    resolved: dict[str, Path] = {}
    owners: dict[str, str] = {}
    wanted = set(names) if names is not None else None
    for root in artifact_dirs:
        store = WorldModelStore(root)
        for name in store.list_names():
            if wanted is not None and name not in wanted:
                continue  # only names we'll actually serve can collide
            if name in resolved:
                raise ValueError(
                    f"world model {name!r} exists under both {owners[name]!r} and {root!r}; "
                    "rename one or serve the roots separately"
                )
            resolved[name] = store.model_dir(name)
            owners[name] = str(root)
    if names is not None:
        missing = [name for name in names if name not in resolved]
        if missing:
            available = ", ".join(sorted(resolved)) or "(none)"
            raise FileNotFoundError(
                f"no world model named {', '.join(missing)} under "
                f"{', '.join(map(str, artifact_dirs))}; have: {available}"
            )
        resolved = {name: resolved[name] for name in names}
    if not resolved:
        raise FileNotFoundError(
            f"no world models built under {', '.join(map(str, artifact_dirs))}; "
            "run `wmh build --name <name>` first"
        )
    return resolved


def _load_card_or_none(model_dir: Path) -> ModelCard | None:
    """Read a model's card, degrading a malformed one to None instead of aborting the server.

    A card is additive metadata (see `wmh.config.card`); one corrupt `card.json` — e.g. a
    build killed mid-write — must not stop the healthy models from being served.
    """
    try:
        return load_card(model_dir)
    except ValueError as exc:
        logger.warning("ignoring unreadable card for %s: %s", model_dir.name, exc)
        return None


def _load_models(
    artifact_dirs: Sequence[str], names: list[str] | None
) -> tuple[dict[str, WorldModel], dict[str, ModelCard | None]]:
    """Load the requested world models (or all built ones) plus their cards."""
    telemetry_root = artifact_dirs[0]
    models: dict[str, WorldModel] = {}
    cards: dict[str, ModelCard | None] = {}
    for name, model_dir in resolve_model_dirs(artifact_dirs, names).items():
        world_model, _provider = load_world_model(model_dir, telemetry_root=telemetry_root)
        models[name] = world_model
        cards[name] = _load_card_or_none(model_dir)
    return models, cards


def create_app(
    artifact_dirs: Sequence[str] = (ARTIFACT_DIR,),
    names: list[str] | None = None,
    world_models: dict[str, WorldModel] | None = None,
    cards: dict[str, ModelCard | None] | None = None,
    build_manager: BuildManager | None = None,
) -> FastAPI:
    """Build the FastAPI app serving one or more named WorldModels.

    Models are either injected directly via `world_models` (name -> model, for tests), or loaded
    from every root in `artifact_dirs` with `names` selecting which to serve (default: all built
    ones). When loading from disk, server-side builds are enabled and land in the first root
    (the writable one, `.wmh` by convention); with injected models pass `build_manager`
    explicitly or the build routes return 503.
    """
    app = FastAPI(title="World Model Harness")
    # The website (localhost:3000/6001/...) is a browser client of this API on another port.
    # Localhost origins only: a foreign website must not be able to script the local backend
    # (steps and builds spend real provider tokens).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOWED_ORIGIN_REGEX,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if world_models is not None:
        models = world_models
        model_cards = cards if cards is not None else {}
    else:
        models, model_cards = _load_models(artifact_dirs, names)

    def _register(name: str, model_dir: Path) -> None:
        """A finished serve-side build joins the live serving set immediately.

        The card is published BEFORE the model so a concurrent `GET /world_models` can never list
        the new model with a null card (readers key off `models`); any name they see already has
        its card entry.
        """
        world_model, _provider = load_world_model(model_dir, telemetry_root=artifact_dirs[0])
        model_cards[name] = _load_card_or_none(model_dir)
        models[name] = world_model

    def _name_taken(name: str) -> bool:
        # Taken if it's in the live served set OR built on disk under any root — the latter
        # catches a model built earlier but not currently served (via `--name`), which a build
        # would otherwise silently overwrite.
        if name in models:
            return True
        return any(WorldModelStore(root).exists(name) for root in artifact_dirs)

    builds = build_manager
    if builds is None and world_models is None:
        builds = BuildManager(
            store_root=artifact_dirs[0],
            name_taken=_name_taken,
            register=_register,
        )

    def _model_or_404(name: str) -> WorldModel:
        try:
            return models[name]
        except KeyError:
            available = ", ".join(sorted(models)) or "(none)"
            raise HTTPException(
                status_code=404, detail=f"no world model {name!r}; have: {available}"
            ) from None

    def _session_or_404(wm: WorldModel, session_id: str) -> Session:
        try:
            return wm.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no session {session_id}") from None

    def _builds_or_503() -> BuildManager:
        if builds is None:
            raise HTTPException(
                status_code=503,
                detail="server-side builds are not enabled on this backend; "
                "run `wmh build` locally instead",
            )
        return builds

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/world_models", response_model=ModelsResponse)
    def list_world_models() -> ModelsResponse:
        return ModelsResponse(
            world_models=sorted(models),
            models=[
                ModelCardEntry(name=name, card=model_cards.get(name)) for name in sorted(models)
            ],
        )

    @app.post("/world_models/builds", response_model=NewBuildResponse, status_code=202)
    def new_build(req: BuildRouteRequest) -> NewBuildResponse:
        manager = _builds_or_503()
        try:
            return NewBuildResponse(build_id=manager.start(req))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError as exc:  # bad provider kind / model name / split
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @app.post("/world_models/builds/uploads", response_model=UploadResponse)
    def upload_traces(request: Request, file: UploadFile) -> UploadResponse:
        manager = _builds_or_503()
        # A multipart POST is a CORS "simple request" that skips preflight, so CORSMiddleware
        # can't stop a foreign page from reaching this disk-writing route. Re-check the Origin
        # here: browsers always send it on cross-origin requests, and we require it to be one of
        # our own (or absent, i.e. a same-origin/non-browser caller like curl).
        origin = request.headers.get("origin")
        if origin is not None and not re.match(ALLOWED_ORIGIN_REGEX, origin):
            raise HTTPException(status_code=403, detail=f"origin {origin!r} not allowed")
        manager.uploads_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "traces.jsonl").suffix or ".jsonl"
        target = manager.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
        # Stream to disk in chunks (Starlette already spooled it to a temp file) with a size cap,
        # so a huge/malicious upload can't be slurped whole into memory or fill the disk.
        written = 0
        with target.open("wb") as fh:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                fh.write(chunk)
        return UploadResponse(path=str(target))

    def _snapshot_or_404(manager: BuildManager, build_id: str) -> BuildSnapshot:
        try:
            return manager.snapshot(build_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no build {build_id}") from None

    @app.get("/world_models/builds/{build_id}", response_model=BuildSnapshot)
    def build_snapshot(build_id: str) -> BuildSnapshot:
        return _snapshot_or_404(_builds_or_503(), build_id)

    @app.get("/world_models/builds/{build_id}/events")
    def build_events(build_id: str, request: Request) -> StreamingResponse:
        manager = _builds_or_503()
        _snapshot_or_404(manager, build_id)  # 404 before we hand back a stream
        # Resume from the client's Last-Event-ID on reconnect so events aren't replayed/duplicated.
        last = request.headers.get("last-event-id")
        start = int(last) + 1 if last is not None and last.isdigit() else 0
        return StreamingResponse(
            manager.sse_events(build_id, start),
            media_type="text/event-stream",
            # Keep proxies/compression from buffering the stream (which would freeze progress
            # until the build finishes and flushes everything at once).
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/world_models/{world_model_name}/sessions", response_model=NewSessionResponse)
    def new_session(world_model_name: str, req: NewSessionRequest) -> NewSessionResponse:
        wm = _model_or_404(world_model_name)
        session = wm.new_session(task=req.task, seed_state=req.seed_state)
        return NewSessionResponse(session_id=session.id, state=session.state)

    @app.get("/world_models/{world_model_name}/sessions/{session_id}", response_model=Session)
    def get_session(world_model_name: str, session_id: str) -> Session:
        wm = _model_or_404(world_model_name)
        return _session_or_404(wm, session_id)

    @app.get(
        "/world_models/{world_model_name}/sessions/{session_id}/usage", response_model=RunRecord
    )
    def session_usage(world_model_name: str, session_id: str) -> RunRecord:
        """Per-session token/cost/time so far (serve-time observability)."""
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        return wm.session_usage(session_id)

    @app.post(
        "/world_models/{world_model_name}/sessions/{session_id}/step", response_model=StepResponse
    )
    def step(world_model_name: str, session_id: str, req: StepRequest) -> StepResponse:
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        observation = wm.step(session_id, req.action)
        return StepResponse(observation=observation, state=wm.get_session(session_id).state)

    @app.post(
        "/world_models/{world_model_name}/sessions/{session_id}/score",
        response_model=EpisodeScore,
    )
    def score_session(world_model_name: str, session_id: str) -> EpisodeScore:
        """Judge the session's rollout so far: episode reward + per-step rewards + critique."""
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        return wm.score_session(session_id)

    @app.delete("/world_models/{world_model_name}/sessions/{session_id}", response_model=RunRecord)
    def end_session(world_model_name: str, session_id: str) -> RunRecord:
        """End the session (free its memory + metering) and return its final usage record."""
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        return wm.end_session(session_id)

    return app
