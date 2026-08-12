"""Local FastAPI backend - the live environment agents call over HTTP.

Routes are namespaced by world model name (`/world_models/{name}/...`) so one backend can serve
several named models at once - from one or more store roots (`.wmo`, a downloaded task dir, ...).
Each route is a thin transport over an in-process `WorldModel`; the CLI and the API share the
same code path. `GET /world_models` also returns each model's `card.json` when present. Canonical
trace normalization and task mining belong exclusively to the local `wmo build` command.

The backend is also the *reward* server for RL training: `POST .../sessions/{id}/score` judges the
session's rollout (task + history) with `EpisodeRewardJudge`, returning the scalar episode reward
(GRPO/PPO/REINFORCE++), per-step rewards, and a critique string (SDPO's teacher feedback) - so a
training scaffold gets environment and reward behind one API.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from wmo.common.config import ARTIFACT_DIR, WorldModelStore, validate_name
from wmo.common.config.card import ModelCard, load_card
from wmo.common.core.types import Action, EnvState, Observation, Session
from wmo.common.judging.episode import EpisodeScore
from wmo.common.observability import RunRecord
from wmo.optimize.routing.pareto import PARETO_FILENAME, ParetoCurve
from wmo.optimize.routing.policy import POLICY_FILENAME, RoutingPolicy
from wmo.simulation.model.loader import load_world_model
from wmo.simulation.model.world_model import WorldModel
from wmo.simulation.serving.chat import (
    EndpointRuntime,
    RequestLog,
    create_chat_router,
    install_openai_error_shapes,
)
from wmo.simulation.serving.endpoint_config import ENDPOINT_CONFIG_FILENAME, EndpointConfig
from wmo.simulation.serving.query_embeddings import QUERY_EMBEDDING_FILENAME, QueryEmbeddingStore
from wmo.simulation.serving.traces_source import (
    TRACES_FILENAME,
    TracesDownloader,
    TracesResponse,
    local_traces_path,
    resolve_url,
    trace_summaries_from_otlp,
)

logger = logging.getLogger(__name__)

# Only browser origins matching this may reach the API. The API keeps CORS restricted because
# served sessions and OpenAI-compatible endpoints may invoke configured local providers.
ALLOWED_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


class NewSessionRequest(BaseModel):
    """Session-open body: the task to run, optionally seeded with a starting state."""

    task: str | None = None
    seed_state: EnvState | None = None


class NewSessionResponse(BaseModel):
    """Session-open reply: the new session id and its initial state."""

    session_id: str
    state: EnvState


class StepRequest(BaseModel):
    """Step body: the action to simulate."""

    action: Action


class StepResponse(BaseModel):
    """Step reply: the predicted observation and the state it left behind."""

    observation: Observation
    # The env state after this step, so clients can render scratchpad/structured state without a
    # follow-up GET of the whole (linearly growing) session on every step.
    state: EnvState


class ModelCardEntry(BaseModel):
    """One served world model, with its card when the artifact has one."""

    name: str
    card: ModelCard | None = None


class ModelsResponse(BaseModel):
    """World-model listing reply."""

    world_models: list[str]  # names-only shape, kept for existing clients
    models: list[ModelCardEntry]


class KnowledgeResponse(BaseModel):
    """The model's knowledge base: enabled + every markdown file's content."""

    enabled: bool
    files: dict[str, str]


class KnowledgeFileRequest(BaseModel):
    """Knowledge-file write body: the markdown to store."""

    content: str


def resolve_model_dirs(artifact_dirs: Sequence[str], names: list[str] | None) -> dict[str, Path]:
    """Map model name -> artifact dir across every root, failing fast on ambiguity.

    A name appearing under two roots is an error (serving would silently pick one); a requested
    `names` entry that no root provides is an error listing what is available.
    """
    resolved: dict[str, Path] = {}
    owners: dict[str, str] = {}
    if names is not None:
        for name in names:
            validate_name(name)  # friendly ValueError on an unsafe name, before any disk lookup
    wanted = set(names) if names is not None else None
    built: set[str] = set()  # every name on disk, so a typo can be told what IS there
    for root in artifact_dirs:
        store = WorldModelStore(root)
        for name in store.list_names():
            built.add(name)
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
            available = ", ".join(sorted(built)) or "(none)"
            remedy = "`wmo list` shows what is built" if built else "run `wmo build --name <name>`"
            raise FileNotFoundError(
                f"no world model named {', '.join(missing)} under "
                f"{', '.join(map(str, artifact_dirs))}; have: {available}; {remedy}"
            )
        resolved = {name: resolved[name] for name in names}
    if not resolved:
        raise FileNotFoundError(
            f"no world models built under {', '.join(map(str, artifact_dirs))}; "
            "run `wmo build --name <name>` first"
        )
    return resolved


def _load_card_or_none(model_dir: Path) -> ModelCard | None:
    """Read a model's card, degrading a malformed one to None instead of aborting the server.

    A card is additive metadata (see `wmo.common.config.card`); one corrupt `card.json` - e.g. a
    build killed mid-write - must not stop the healthy models from being served.
    """
    try:
        return load_card(model_dir)
    except ValueError as exc:
        logger.warning("ignoring unreadable card for %s: %s", model_dir.name, exc)
        return None


def _load_models(
    artifact_dirs: Sequence[str], names: list[str] | None, *, max_fidelity: bool = False
) -> tuple[dict[str, WorldModel], dict[str, ModelCard | None], dict[str, Path]]:
    """Load the requested world models (or all built ones) plus their cards and dirs."""
    telemetry_root = artifact_dirs[0]
    models: dict[str, WorldModel] = {}
    cards: dict[str, ModelCard | None] = {}
    dirs: dict[str, Path] = {}
    for name, model_dir in resolve_model_dirs(artifact_dirs, names).items():
        world_model, _provider = load_world_model(
            model_dir, telemetry_root=telemetry_root, max_fidelity=max_fidelity
        )
        models[name] = world_model
        cards[name] = _load_card_or_none(model_dir)
        dirs[name] = model_dir
    return models, cards, dirs


def _load_pareto_or_none(model_dir: Path | None) -> ParetoCurve | None:
    """The measured curve written at optimize time, or None when it never was.

    A malformed file fails the mount loudly (same posture as an invalid policy): serving a
    stale or truncated curve as "the measured frontier" is worse than refusing to start.
    """
    if model_dir is None:
        return None
    path = model_dir / PARETO_FILENAME
    if not path.is_file():
        return None
    try:
        return ParetoCurve.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"invalid pareto curve at {path}: {exc}") from exc


def _endpoint_runtimes(
    policies: Mapping[str, RoutingPolicy],
    model_dirs: Mapping[str, Path],
    log: RequestLog,
    embeddings: QueryEmbeddingStore | None = None,
) -> dict[str, EndpointRuntime]:
    """Turn policies into served endpoints, each on the dial its `endpoint.toml` asks for.

    The dial file lives beside the model's `policy.json`; see
    `wmo.simulation.serving.endpoint_config`. Only disk-loaded models have one. No file means
    "serve the policy exactly as fitted": mounting
    must never silently re-tune an artifact. The path is handed to the runtime as well, so a live
    `PUT /v1/endpoints/{name}/config` lands back in the same file and survives a restart.
    """
    runtimes: dict[str, EndpointRuntime] = {}
    for name, policy in policies.items():
        model_dir = model_dirs.get(name)
        config_path = model_dir / ENDPOINT_CONFIG_FILENAME if model_dir is not None else None
        settings = EndpointConfig()
        try:
            # Reading the file is inside the guard too: a malformed or misspelled endpoint.toml
            # must fail with the endpoint AND the path named, not with a bare parse error that
            # leaves an operator hunting for which model directory it came from.
            if config_path is not None:
                settings = EndpointConfig.load(config_path)
            # A dial setting resolves the novelty floor against the evidence bank, so an endpoint
            # WITH a dial file loads its `.npz` sidecar here at mount instead of lazily on the
            # first request. That is the better trade for a served endpoint (the cost lands before
            # traffic rather than inside someone's first request latency), and it is why a broken
            # sidecar surfaces at startup for a dialed endpoint and on first use for an as-fitted
            # one.
            runtimes[name] = EndpointRuntime(
                name,
                policy,
                log=log,
                cost_quality=settings.cost_quality,
                config_path=config_path,
                embeddings=embeddings,
                log_query_embeddings=settings.log_query_embeddings,
                pareto=_load_pareto_or_none(model_dir),
            )
        except ValueError as exc:
            # Fail fast and name the file: a dial the policy cannot honor (a savings position on
            # a policy fitted without costs) must not degrade silently to the fitted knobs.
            raise ValueError(
                f"{config_path} for endpoint {name!r} cannot be served: {exc}"
            ) from exc
    return runtimes


def create_app(
    artifact_dirs: Sequence[str] = (ARTIFACT_DIR,),
    names: list[str] | None = None,
    world_models: dict[str, WorldModel] | None = None,
    cards: dict[str, ModelCard | None] | None = None,
    max_fidelity: bool = False,
    policies: dict[str, RoutingPolicy] | None = None,
) -> FastAPI:
    """Build the FastAPI app serving one or more named WorldModels.

    Models are either injected directly via `world_models` (name -> model, for tests), or loaded
    from every root in `artifact_dirs` with `names` selecting which to serve (default: all built
    ones). `max_fidelity` serves every disk-loaded model with its online extras.

    A model whose artifact dir carries a `policy.json` also serves as an OpenAI-compatible
    ENDPOINT: `/v1/chat/completions` (streaming included) routes each call through that policy
    (`wmo.simulation.serving.chat`). An `endpoint.toml` beside it sets the endpoint's cost/quality
    dial (`wmo.simulation.serving.endpoint_config`). `policies` injects them directly for tests.
    """
    app = FastAPI(title="World Model Optimizer")
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
        model_dirs: dict[str, Path] = {}
    else:
        models, model_cards, model_dirs = _load_models(
            artifact_dirs, names, max_fidelity=max_fidelity
        )
    downloader = TracesDownloader()

    # OpenAI-compatible endpoints: every served model whose artifact dir carries a policy.json
    # (or every injected `policies` entry) answers /v1/chat/completions through that policy.
    endpoint_policies = policies if policies is not None else {}
    if policies is None:
        for model_name, model_dir in model_dirs.items():
            policy_path = model_dir / POLICY_FILENAME
            if policy_path.is_file():
                try:
                    endpoint_policies[model_name] = RoutingPolicy.load(policy_path)
                except Exception as exc:
                    # Fail fast, but name the file: a bare ValidationError at startup doesn't
                    # say WHICH model's policy.json is broken.
                    raise ValueError(f"invalid routing policy at {policy_path}: {exc}") from exc
    # Mounted even with zero policies so a client wired up before a policy is fitted gets an
    # empty /v1/models list and an OpenAI-shaped "no endpoint" error instead of a bare 404.
    # With no artifact root (injected-models tests) the log keeps its in-memory tail only.
    log_path = Path(artifact_dirs[0]) / "serving" / "requests.jsonl" if artifact_dirs else None
    request_log = RequestLog(log_path)
    # The query-vector sidecar lives beside the request log, so a log row and the vector it was
    # routed on are one directory apart. No artifact root means no store, matching the log.
    embeddings = QueryEmbeddingStore(
        log_path.parent / QUERY_EMBEDDING_FILENAME if log_path is not None else None
    )
    app.include_router(
        create_chat_router(
            _endpoint_runtimes(endpoint_policies, model_dirs, request_log, embeddings)
        )
    )
    install_openai_error_shapes(app)

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

    @app.get("/world_models/{world_model_name}/knowledge", response_model=KnowledgeResponse)
    def get_knowledge(world_model_name: str) -> KnowledgeResponse:
        """Read the model's knowledge base (`enabled=False` for pre-knowledge artifacts)."""
        kb = _model_or_404(world_model_name).knowledge
        if kb is None:
            return KnowledgeResponse(enabled=False, files={})
        return KnowledgeResponse(enabled=True, files=kb.files())

    @app.put(
        "/world_models/{world_model_name}/knowledge/{file_name}",
        response_model=KnowledgeResponse,
    )
    def put_knowledge(
        world_model_name: str, file_name: str, req: KnowledgeFileRequest
    ) -> KnowledgeResponse:
        """Create/replace one knowledge markdown file (the HTTP face of 'edit the folder')."""
        kb = _model_or_404(world_model_name).knowledge
        if kb is None:
            raise HTTPException(
                status_code=409,
                detail=f"world model {world_model_name!r} has no knowledge base; "
                "build it with knowledge enabled (or create a knowledge/ dir in its artifact "
                "and re-serve)",
            )
        try:
            kb.write_file(file_name, req.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return KnowledgeResponse(enabled=True, files=kb.files())

    @app.get("/world_models/{world_model_name}/traces", response_model=TracesResponse)
    def get_traces(world_model_name: str) -> TracesResponse:
        """Recorded traces for the model: local scenarios if present, else a Hub download offer."""
        _model_or_404(world_model_name)
        model_dir = model_dirs.get(world_model_name)
        card = model_cards.get(world_model_name)
        progress = downloader.progress(world_model_name)
        local = local_traces_path(model_dir) if model_dir is not None else None
        if local is not None:
            return TracesResponse(
                source="local",
                downloadable=False,
                scenarios=trace_summaries_from_otlp(local),
                download=progress,
            )
        has_hub = card is not None and card.traces_hf is not None
        return TracesResponse(
            source="hub" if has_hub else "none",
            downloadable=has_hub,
            download=progress,
        )

    @app.post("/world_models/{world_model_name}/traces/download", status_code=202)
    def download_traces(world_model_name: str) -> dict[str, str]:
        """Kick off a background fetch of the model's traces from its declared Hub source."""
        _model_or_404(world_model_name)
        model_dir = model_dirs.get(world_model_name)
        card = model_cards.get(world_model_name)
        if model_dir is None or card is None or card.traces_hf is None:
            raise HTTPException(
                status_code=400,
                detail=f"no Hugging Face traces source declared for {world_model_name!r}",
            )
        downloader.start(world_model_name, resolve_url(card.traces_hf), model_dir / TRACES_FILENAME)
        return {"status": "started"}

    @app.get("/world_models/{world_model_name}/traces/download")
    def download_progress(world_model_name: str) -> dict[str, object]:
        """Poll the current/last trace download's byte progress for this model."""
        _model_or_404(world_model_name)
        progress = downloader.progress(world_model_name)
        return {"download": progress.model_dump() if progress else None}

    return app
