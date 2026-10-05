import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from inspection_demo.asset_bootstrap import PreparedAssets
from inspection_demo.main import create_app
from inspection_demo.models import (
    ActionSpec,
    CostEstimate,
    FloorMap,
    GroundedTask,
    GroundingResult,
    ReasoningSummary,
    Subgraph,
    SubgraphNode,
)
from inspection_demo.settings import Settings


def _prepared(tmp_path: Path) -> PreparedAssets:
    source_hash = "a" * 64
    runtime = tmp_path / "runtime"
    cache = tmp_path / "cache"
    floors = tmp_path / "floors"
    runtime.mkdir()
    cache.mkdir()
    floors.mkdir()
    ifc = tmp_path / "ecore.ifc"
    ifc.write_text("ISO-10303-21;", encoding="utf-8")
    (runtime / "manifest.json").write_text(
        json.dumps({"inspection_graph": {"sha256": "b" * 64}}), encoding="utf-8"
    )
    floor = FloorMap(
        id="level-test-floorplan-v1",
        floor_id="LEVEL TEST",
        origin=(0.0, 0.0),
        width_m=10.0,
        height_m=10.0,
        polygons=[],
        occupancy_rows=["." * 100 for _ in range(100)],
        target_positions={"ifc_target": (7.0, 7.0)},
        source_ifc_sha256=source_hash,
    )
    (floors / "level-test.json").write_text(floor.model_dump_json(), encoding="utf-8")
    return PreparedAssets(
        ifc_path=ifc,
        runtime_dir=runtime,
        cache_dir=cache,
        floorplan_dir=floors,
        asset_manifest_sha256="c" * 64,
        ifc_sha256=source_hash,
        runtime_manifest_sha256="d" * 64,
    )


class CapturingAdapter:
    def __init__(self, delay: float = 0.0) -> None:
        self.queries: list[str] = []
        self.delay = delay

    async def ground(self, *, query, model_id, api_key, progress=None):
        self.queries.append(query)
        if progress:
            await progress("query_planning", 25, "Planning the submitted query.")
        if self.delay:
            await asyncio.sleep(self.delay)
        task = GroundedTask(
            task_id="task-0-0",
            action="Inspect",
            binding_index=0,
            node_id="ifc_target",
            ifc_guid="target",
            floor_id="LEVEL TEST",
            target_name="Test terminal",
            target_kind="IfcFlowTerminal",
            metadata={"floor_map_id": "level-test-floorplan-v1", "system": "HVAC"},
            target_xy=(7.0, 7.0),
            retrieval_score=0.8,
            validation_status="certified",
            action_spec=ActionSpec(
                name="Inspect",
                cost=CostEstimate(
                    distance_m=5,
                    duration_s=5,
                    risk_score=0.1,
                    information_gain=0.8,
                    model_score=0.8,
                ),
            ),
        )
        return GroundingResult(
            status="completed",
            answer="One certified inspection target was found.",
            closure_status="pass",
            closure_stop_reason="all_bindings_certified",
            tasks=[task],
            reasoning=[ReasoningSummary(phase="hierarchy", summary="Graph evidence linked it.")],
            subgraph=Subgraph(
                id="run-subgraph",
                nodes=[
                    SubgraphNode(
                        id="ifc_target",
                        label="Test terminal",
                        kind="IfcFlowTerminal",
                        is_target=True,
                        floor_id="LEVEL TEST",
                        metadata={"system": "HVAC"},
                    )
                ],
                edges=[],
            ),
            executable=True,
            graph_hash="b" * 64,
        )


def _app(tmp_path: Path, adapter: CapturingAdapter):
    return create_app(
        Settings(
            data_dir=tmp_path / "data",
            runtime_workspace=tmp_path / "work",
            research_device="cpu",
            frontend_dist=tmp_path / "missing-dist",
        ),
        prepared_assets=_prepared(tmp_path),
        grounding_adapter=adapter,
    )


def _terminal(client: TestClient, run_id: str) -> dict:
    for _ in range(100):
        payload = client.get(f"/api/v1/grounding-runs/{run_id}").json()
        if payload["status"] in {"completed", "abstained", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("grounding did not finish")


def test_capabilities_are_derived_from_verified_assets(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, CapturingAdapter())) as client:
        response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    assert response.json()["floor_count"] == 1
    assert response.json()["allowed_models"] == ["gpt-4.1", "gpt-5", "gpt-5.6-luna"]
    assert response.json()["graph_hash"] == "b" * 64
    assert response.json()["device"] == "CPU (local verification)"


def test_arbitrary_query_is_forwarded_exactly_and_key_is_not_persisted(tmp_path: Path) -> None:
    adapter = CapturingAdapter()
    query = "Which HVAC terminal could explain unusually warm conditions in the east lab?"
    with TestClient(_app(tmp_path, adapter)) as client:
        created = client.post(
            "/api/v1/grounding-runs",
            json={"api_key": "sk-session-secret", "model_id": "gpt-4.1", "query": query},
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        run = _terminal(client, run_id)
        nodes = client.post(
            f"/api/v1/grounding-runs/{run_id}/nodes/batch",
            json={"node_ids": ["ifc_target"]},
        )
    assert adapter.queries == [query]
    assert "sk-session-secret" not in json.dumps(run)
    assert nodes.json()[0]["properties"]["system"] == "HVAC"


def test_second_job_is_rejected_while_single_worker_is_reserved(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, CapturingAdapter(delay=0.2))) as client:
        first = client.post(
            "/api/v1/grounding-runs",
            json={"api_key": "sk-one", "model_id": "gpt-4.1", "query": "Inspect target A"},
        )
        second = client.post(
            "/api/v1/grounding-runs",
            json={"api_key": "sk-two", "model_id": "gpt-4.1", "query": "Inspect target B"},
        )
        _terminal(client, first.json()["run_id"])
    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"].startswith("Another grounding")


def test_upload_routes_and_asset_selectors_do_not_exist(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, CapturingAdapter())) as client:
        assert client.get("/api/v1/assets").status_code == 404
        assert client.post("/api/v1/assets/ifc").status_code == 404


def test_missing_assets_keep_readiness_closed(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_dir=tmp_path / "data",
            asset_mount=tmp_path / "missing-assets",
            expected_manifest_path=tmp_path / "missing-manifest.json",
            runtime_workspace=tmp_path / "runtime",
            research_device="cpu",
            frontend_dist=tmp_path / "missing-dist",
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 503
        assert client.get("/api/v1/capabilities").status_code == 503
