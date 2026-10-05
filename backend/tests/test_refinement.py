import asyncio
import sys
from types import SimpleNamespace

import pytest
from inspection_demo.grounding import SubprocessResearchGroundingAdapter
from inspection_demo.jobs import TERMINAL_STATUSES, JobManager
from inspection_demo.models import ExecutionRequest, GroundingRequest, Pose2DModel
from inspection_demo.repository import RunStore
from inspection_demo.research_bridge import (
    _subgraph,
    _validated_navigation_xy,
    normalize_tog_response,
)
from pydantic import ValidationError
from test_api_hf import CapturingAdapter


def test_answer_run_cannot_execute_even_with_a_stale_executable_flag(tmp_path):
    from inspection_demo.models import GroundingResult

    store = RunStore(tmp_path / "runs.sqlite3")
    record = store.create(kind="grounding", owner="test", request={})
    result = GroundingResult(
        status="completed",
        answer="An explanation",
        query_type="answer",
        closure_status="pass",
        closure_stop_reason="all_bindings_certified",
        executable=True,
    )
    store.set_result(record.id, status="completed", result=result.model_dump(mode="json"))
    jobs = JobManager(
        store=store, grounding_adapter=CapturingAdapter(), floor_loader=lambda _: None
    )
    request = ExecutionRequest(
        grounding_run_id=record.id,
        ordered_tasks=[{"task_id": "stale-task", "order": 0}],
        robot_profile_id="jackal",
        planner_id="dwa",
        initial_pose=Pose2DModel(x=1, y=1),
    )
    with pytest.raises(PermissionError):
        jobs.create_execution(owner="test", request=request)
    assert not jobs.busy


@pytest.mark.asyncio
async def test_terminal_event_always_has_a_persisted_result(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite3")
    jobs = JobManager(
        store=store, grounding_adapter=CapturingAdapter(), floor_loader=lambda _: None
    )
    original = jobs._event
    observed = []

    async def observe(record_id, phase, *args, **kwargs):
        if phase in TERMINAL_STATUSES:
            observed.append(store.result(store.get(record_id)))
        await original(record_id, phase, *args, **kwargs)

    jobs._event = observe
    jobs.create_grounding(
        owner="test", request=GroundingRequest(api_key="secret", query="Inspect equipment")
    )
    await asyncio.gather(*jobs._tasks)
    assert observed and all(item is not None for item in observed)


@pytest.mark.asyncio
async def test_cancelled_grounding_terminates_child(tmp_path, monkeypatch):
    bridge = tmp_path / "bridge.py"
    bridge.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    adapter = SubprocessResearchGroundingAdapter(
        python_executable=sys.executable, bridge_path=bridge, base_payload={}
    )
    processes = []
    original = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    task = asyncio.create_task(
        adapter.ground(query="Inspect equipment", model_id="gpt-4.1", api_key="secret")
    )
    while not processes:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        assert processes[0].returncode is not None
    finally:
        if processes[0].returncode is None:
            processes[0].kill()
            await processes[0].wait()


def test_unverified_metadata_cannot_invent_a_navigation_position():
    assert (
        _validated_navigation_xy(
            "ifc_test", {"navigation_transform_verified": True, "navigation_local_xy": [2, 3]}
        )
        is None
    )


def test_graph_keeps_certified_targets_and_real_retrieval_edges():
    graph = _subgraph(
        {
            "gnn_subgraph": {
                "node_ids": [f"n{i}" for i in range(250)],
                "edges": [{"source": "n0", "target": "n1", "relation": "connected"}],
            }
        },
        {"certified"},
    )
    assert "certified" in {node.id for node in graph.nodes}
    assert any(edge.source == "n0" and edge.target == "n1" for edge in graph.edges)


@pytest.mark.parametrize("query", ["   ", " \n\t "])
def test_blank_queries_are_rejected(query):
    with pytest.raises(ValidationError):
        GroundingRequest(api_key="secret", query=query)


def test_nonfinite_pose_and_duplicate_tasks_are_rejected():
    with pytest.raises(ValidationError):
        Pose2DModel(x=float("nan"), y=2)
    with pytest.raises(ValidationError):
        ExecutionRequest(
            grounding_run_id="run",
            ordered_tasks=[{"task_id": "t", "order": 0}, {"task_id": "t", "order": 1}],
            robot_profile_id="jackal",
            planner_id="dwa",
            initial_pose={"x": 2, "y": 2},
        )


def test_provider_usage_survives_normalization():
    result = normalize_tog_response(
        {
            "llm_calls": 4,
            "input_tokens": 1234,
            "output_tokens": 234,
            "cached_input_tokens": 100,
            "gnn_embedding_input_tokens_total": 18,
        }
    )
    assert result.usage.llm_calls == 4
    assert result.usage.input_tokens == 1234
    assert result.usage.output_tokens == 234
    assert result.usage.embedding_input_tokens == 18


def test_limited_gpt41_preserves_snapshot_and_enforces_output_limit(monkeypatch):
    from inspection_demo.research_bridge import _install_dynamic_baml_client

    captured = []

    class FakeLlm:
        def __init__(self, client, max_calls, **kwargs):
            self.registry = SimpleNamespace(
                add_llm_client=lambda *args: captured.append(args),
                set_primary=lambda _: None,
            )

    integration = SimpleNamespace(BamlToGLlm=FakeLlm)
    monkeypatch.setitem(sys.modules, "src", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "src.integrations", SimpleNamespace(tog=integration))
    name = _install_dynamic_baml_client("gpt-4.1", "test-secret", 2048)
    integration.BamlToGLlm(name, 24)
    _, provider, options = captured[0]
    assert provider == "openai"
    assert options["model"] == "gpt-4.1-2025-04-14"
    assert options["max_tokens"] == 2048
    assert options["temperature"] == 0


def test_sse_rejects_invalid_resume_cursor(tmp_path):
    from fastapi.testclient import TestClient
    from test_api_hf import _app

    with TestClient(_app(tmp_path, CapturingAdapter())) as client:
        response = client.get(
            "/api/v1/grounding-runs/missing/events", headers={"Last-Event-ID": "oops"}
        )
        assert response.status_code == 422
