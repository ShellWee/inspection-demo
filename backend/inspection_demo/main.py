from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from .asset_bootstrap import AssetBootstrapError, AssetBootstrapper, PreparedAssets
from .grounding import GroundingAdapter, SubprocessResearchGroundingAdapter
from .jobs import TERMINAL_STATUSES, BusyError, JobManager
from .models import (
    Capabilities,
    CapabilityAsset,
    ExecutionControlRequest,
    ExecutionRequest,
    ExecutionResult,
    ExecutionRunCreated,
    FloorMap,
    GroundingRequest,
    GroundingRun,
    GroundingRunCreated,
    ModelCatalogRequest,
    ModelOption,
    NodeBatchRequest,
    PublicNodeMetadata,
)
from .openai_catalog import MODEL_PROFILES, OpenAIModelCatalog
from .repository import RunRecord, RunStore
from .settings import Settings

OWNER = "huggingface-private-user"


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_owner() -> str:
    return OWNER


OwnerDep = Annotated[str, Depends(get_owner)]


def get_store(request: Request) -> RunStore:
    return request.app.state.store


StoreDep = Annotated[RunStore, Depends(get_store)]


def get_jobs(request: Request) -> JobManager:
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "bootstrap_error", "Assets are not ready"),
        )
    return jobs


JobsDep = Annotated[JobManager, Depends(get_jobs)]


def _authorize(record: RunRecord | None, owner: str) -> RunRecord:
    if record is None or record.owner != owner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return record


def _grounding_public(record: RunRecord, store: RunStore) -> GroundingRun:
    request = store.request(record)
    return GroundingRun(
        id=record.id,
        owner=record.owner,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        model_id=request["model_id"],
        query=request["query"],
        result=store.result(record),
        events=store.events(record.id),
    )


def _load_floor_maps(prepared: PreparedAssets) -> dict[str, FloorMap]:
    floors: dict[str, FloorMap] = {}
    for path in sorted(prepared.floorplan_dir.glob("*.json")):
        floor = FloorMap.model_validate_json(path.read_text(encoding="utf-8"))
        if floor.source_ifc_sha256 != prepared.ifc_sha256:
            raise AssetBootstrapError(f"floor map IFC hash mismatch: {path.name}")
        if floor.id in floors:
            raise AssetBootstrapError(f"duplicate floor map id: {floor.id}")
        floors[floor.id] = floor
    if not floors:
        raise AssetBootstrapError("no validated floor maps were prepared")
    return floors


def _device_label(device: str) -> str:
    if device != "cuda":
        return "CPU (local verification)"
    import torch

    return f"{torch.cuda.get_device_name(0)} / CUDA {torch.version.cuda}"


def _capabilities(
    prepared: PreparedAssets, floors: dict[str, FloorMap], device: str
) -> Capabilities:
    runtime_manifest = json.loads(
        (prepared.runtime_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return Capabilities(
        ready=True,
        device=_device_label(device),
        ifc=CapabilityAsset(id="ecore-fixed", name="Ecore IFC (fixed)", sha256=prepared.ifc_sha256),
        gnn_runtime=CapabilityAsset(
            id="text-gnn-v5-seed43-epoch10",
            name="Text-GNN v5 · seed43 · epoch10",
            sha256=prepared.runtime_manifest_sha256,
        ),
        graph_hash=str(runtime_manifest["inspection_graph"]["sha256"]),
        floor_count=len(floors),
        allowed_models=list(MODEL_PROFILES),
        embedding_model="text-embedding-3-small",
    )


router = APIRouter(prefix="/api/v1", tags=["inspection"])


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "jobs", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "bootstrap_error", "Assets are not ready"),
        )
    return {"status": "ok"}


@router.get("/capabilities", response_model=Capabilities)
def capabilities(request: Request) -> Capabilities:
    result = getattr(request.app.state, "capabilities", None)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "bootstrap_error", "Assets are not ready"),
        )
    return result


@router.post("/openai/models", response_model=list[ModelOption])
async def list_openai_models(body: ModelCatalogRequest, owner: OwnerDep) -> list[ModelOption]:
    del owner
    try:
        return await OpenAIModelCatalog().list(body.api_key.get_secret_value())
    except Exception as error:  # noqa: BLE001 - provider diagnostics must not leak
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to list models: {type(error).__name__}",
        ) from error


@router.post(
    "/grounding-runs",
    response_model=GroundingRunCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_grounding_run(
    body: GroundingRequest, owner: OwnerDep, jobs: JobsDep
) -> GroundingRunCreated:
    try:
        record = jobs.create_grounding(owner=owner, request=body)
    except BusyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return GroundingRunCreated(run_id=record.id, status="queued")


@router.get("/grounding-runs/{run_id}", response_model=GroundingRun)
def get_grounding_run(run_id: str, owner: OwnerDep, store: StoreDep) -> GroundingRun:
    return _grounding_public(_authorize(store.get(run_id), owner), store)


async def _events(
    *, record_id: str, owner: str, store: RunStore, last_event_id: int | None
) -> AsyncIterator[ServerSentEvent]:
    _authorize(store.get(record_id), owner)
    delivered = int(last_event_id or 0)
    while True:
        record = _authorize(store.get(record_id), owner)
        for event in store.events(record_id):
            if event.event_id <= delivered:
                continue
            delivered = event.event_id
            yield ServerSentEvent(
                data=event.model_dump(mode="json"),
                event="progress",
                id=str(event.event_id),
            )
        if record.status in TERMINAL_STATUSES:
            break
        await asyncio.sleep(0.1)


@router.get("/grounding-runs/{run_id}/events", response_class=EventSourceResponse)
async def grounding_events(
    run_id: str,
    owner: OwnerDep,
    store: StoreDep,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> AsyncIterator[ServerSentEvent]:
    async for event in _events(
        record_id=run_id, owner=owner, store=store, last_event_id=last_event_id
    ):
        yield event


@router.get("/subgraphs/{run_id}")
def get_subgraph(run_id: str, owner: OwnerDep, store: StoreDep) -> dict[str, Any]:
    run = _grounding_public(_authorize(store.get(run_id), owner), store)
    if run.result is None or run.result.subgraph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subgraph not available")
    return run.result.subgraph.model_dump(mode="json")


@router.post("/grounding-runs/{run_id}/nodes/batch", response_model=list[PublicNodeMetadata])
def get_nodes_batch(
    run_id: str,
    body: NodeBatchRequest,
    owner: OwnerDep,
    store: StoreDep,
) -> list[PublicNodeMetadata]:
    run = _grounding_public(_authorize(store.get(run_id), owner), store)
    if run.result is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Run is not complete")
    nodes = {node.id: node for node in (run.result.subgraph.nodes if run.result.subgraph else [])}
    tasks = {task.node_id: task for task in run.result.tasks}
    result: list[PublicNodeMetadata] = []
    for node_id in body.node_ids:
        node = nodes.get(node_id)
        task = tasks.get(node_id)
        if node is None and task is None:
            continue
        metadata = {**(node.metadata if node else {}), **(task.metadata if task else {})}
        result.append(
            PublicNodeMetadata(
                node_id=node_id,
                ifc_guid=task.ifc_guid if task else node_id.removeprefix("ifc_"),
                name=task.target_name if task else node.label,
                ifc_class=task.target_kind if task else node.kind,
                floor_id=task.floor_id if task else node.floor_id,
                properties=metadata,
            )
        )
    return result


@router.get("/floors/{floor_map_id}", response_model=FloorMap)
def get_floor(floor_map_id: str, owner: OwnerDep, request: Request) -> FloorMap:
    del owner
    floor = getattr(request.app.state, "floor_maps", {}).get(floor_map_id)
    if floor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Floor map not found")
    return floor


@router.post(
    "/executions",
    response_model=ExecutionRunCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_execution(
    body: ExecutionRequest, owner: OwnerDep, jobs: JobsDep
) -> ExecutionRunCreated:
    try:
        record = jobs.create_execution(owner=owner, request=body)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Grounding run not found"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (BusyError, PermissionError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return ExecutionRunCreated(execution_id=record.id, status="queued")


@router.get("/executions/{execution_id}/replay", response_model=ExecutionResult)
def execution_replay(execution_id: str, owner: OwnerDep, store: StoreDep) -> ExecutionResult:
    record = _authorize(store.get(execution_id), owner)
    if record.kind != "execution":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    payload = store.result(record)
    if payload is None:
        return ExecutionResult(
            status=record.status, frames=[], completed_task_ids=[], terminal_reason=record.status
        )
    return ExecutionResult.model_validate(payload)


@router.post("/executions/{execution_id}/control", response_model=ExecutionResult)
async def control_execution(
    execution_id: str,
    body: ExecutionControlRequest,
    owner: OwnerDep,
    store: StoreDep,
    jobs: JobsDep,
) -> ExecutionResult:
    record = _authorize(store.get(execution_id), owner)
    if record.kind != "execution":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    if body.command != "abort":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Retry and skip require a paused planning failure",
        )
    try:
        return await jobs.abort_execution(execution_id)
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/executions/{execution_id}/events", response_class=EventSourceResponse)
async def execution_events(
    execution_id: str,
    owner: OwnerDep,
    store: StoreDep,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> AsyncIterator[ServerSentEvent]:
    record = _authorize(store.get(execution_id), owner)
    if record.kind != "execution":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    async for event in _events(
        record_id=execution_id,
        owner=owner,
        store=store,
        last_event_id=last_event_id,
    ):
        yield event


def create_app(
    settings: Settings | None = None,
    *,
    prepared_assets: PreparedAssets | None = None,
    grounding_adapter: GroundingAdapter | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    resolved.data_dir.mkdir(parents=True, exist_ok=True)
    store = RunStore(resolved.data_dir / "runs.sqlite3")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        jobs: JobManager | None = None
        try:
            prepared = prepared_assets or await asyncio.to_thread(
                AssetBootstrapper(
                    asset_mount=resolved.asset_mount,
                    expected_manifest_path=resolved.expected_manifest_path,
                    workspace=resolved.runtime_workspace,
                    data_dir=resolved.data_dir,
                    require_cuda=resolved.research_device == "cuda",
                ).prepare
            )
            floors = _load_floor_maps(prepared)
            adapter = grounding_adapter
            if adapter is None:
                project_root = Path(__file__).resolve().parents[2]
                adapter = SubprocessResearchGroundingAdapter(
                    python_executable=sys.executable,
                    bridge_path=Path(__file__).with_name("research_bridge.py"),
                    base_payload={
                        "cobbie_root": str(project_root / "vendor" / "cobbie-ecore"),
                        "tog_root": str(project_root / "vendor" / "tog-ifc-ecore" / "src"),
                        "runtime_dir": str(prepared.runtime_dir),
                        "ifc_path": str(prepared.ifc_path),
                        "cache_dir": str(prepared.cache_dir),
                        "floorplan_dir": str(prepared.floorplan_dir),
                        "manifest_sha256": prepared.runtime_manifest_sha256,
                        "device": resolved.research_device,
                    },
                    timeout_seconds=resolved.grounding_timeout_seconds,
                )
            jobs = JobManager(
                store=store,
                grounding_adapter=adapter,
                floor_loader=floors.get,
                simulation_event_period_seconds=resolved.simulation_event_period_seconds,
            )
            application.state.floor_maps = floors
            application.state.capabilities = _capabilities(
                prepared, floors, resolved.research_device
            )
            application.state.jobs = jobs
            application.state.bootstrap_error = None
        except Exception as error:  # noqa: BLE001 - fail closed while keeping health visible
            application.state.jobs = None
            application.state.floor_maps = {}
            application.state.capabilities = None
            application.state.bootstrap_error = f"Asset readiness failed: {type(error).__name__}"
        yield
        if jobs is not None:
            await jobs.shutdown()

    application = FastAPI(
        title="Ecore Inspection Target Planning",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved
    application.state.store = store
    application.state.jobs = None
    application.state.floor_maps = {}
    application.state.capabilities = None
    application.state.bootstrap_error = "Assets have not been verified"
    application.include_router(router)

    @application.get("/favicon.ico", include_in_schema=False, status_code=204)
    def favicon() -> Response:
        return Response(status_code=204)

    @application.api_route(
        "/api/v1/{unknown_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def unknown_api(unknown_path: str) -> None:
        del unknown_path
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API route not found")

    application.frontend(
        "/", directory=resolved.frontend_dist, fallback="index.html", check_dir=False
    )
    return application


app = create_app()
