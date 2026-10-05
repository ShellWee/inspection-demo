from inspection_demo.models import FloorMap, MapPolygon, Point2D
from inspection_demo.research_bridge import (
    _load_floor_maps,
    _weights_only_loader,
    normalize_tog_response,
)


def test_abstention_retains_model_answer_without_certifying_it():
    result = normalize_tog_response(
        {
            "answer": "I cannot determine the answer from the IFC knowledge graph.",
            "query_intent": {"kind": "answer", "reason": "Requests an impact explanation."},
            "debug": {
                "provider_group_selection": {
                    "answer_summary": (
                        "Shared membership suggests possible exposure; flow direction is unknown."
                    )
                }
            },
            "evidence_closure_certificate": {
                "closure_status": "abstain",
                "stop_reason": "unresolved_groups",
            },
        }
    )
    assert (
        result.agent_response
        == "Shared membership suggests possible exposure; flow direction is unknown."
    )
    assert result.response_status == "unverified"
    assert result.query_type == "answer"
    assert result.executable is False


def test_answer_query_never_exposes_internal_inspection_bindings_as_tasks():
    result = normalize_tog_response(
        {
            "query_intent": {"kind": "answer", "reason": "Asks which elements may be affected."},
            "debug": {
                "query_plan": {"action_bindings": [{"action": "Inspect", "binding_index": 0}]}
            },
            "evidence_closure_certificate": {
                "closure_status": "pass",
                "certified_target_ids": {"0": ["ifc_fan"]},
            },
            "selected_entities": [{"node_id": "ifc_fan", "score": 0.024, "label": "Fan"}],
        }
    )
    assert result.tasks == []
    assert result.query_type == "answer"
    assert result.executable is False
    assert result.planning_score.value == 0.024
    assert result.planning_score.kind == "mean_retrieval_rank_score"
    assert result.planning_score.calibrated is False


def test_no_score_is_reported_when_no_selected_evidence_exists():
    result = normalize_tog_response(
        {"query_intent": {"kind": "unknown", "reason": "Classification unavailable"}}
    )
    assert result.planning_score.value is None
    assert result.agent_response
    assert result.executable is False


def test_bridge_builds_binding_from_plan_and_certificate_not_answer_text() -> None:
    """Catches accidental parsing of the display-only PDDL answer string."""
    raw = {
        "answer": "This text is intentionally not parseable.",
        "graph_hash": "graph-sha",
        "validation_status": "pass",
        "errors": [],
        "debug": {"query_plan": {"action_bindings": [{"action": "Navigate", "binding_index": 0}]}},
        "evidence_closure_certificate": {
            "closure_status": "pass",
            "stop_reason": "all_bindings_certified",
            "certified_target_ids": {"0": ["ifc_0aZxGK_jD1LR$XoST2vRo$"]},
        },
        "selected_entities": [
            {
                "node_id": "ifc_0aZxGK_jD1LR$XoST2vRo$",
                "label": "CYBER-PHYSICAL SYSTEMS LAB",
                "global_id": "0aZxGK_jD1LR$XoST2vRo$",
                "ifc_class": "IfcSpace",
                "score": 0.96,
                "metadata": {"storey": "LEVEL 4", "name": "404"},
            }
        ],
        "evidence": [],
        "gnn_subgraph": {"node_ids": [], "edges": [], "node_scores": {}},
    }

    result = normalize_tog_response(raw)

    assert result.closure_status == "pass"
    assert result.tasks[0].action == "Navigate"
    assert result.tasks[0].node_id == "ifc_0aZxGK_jD1LR$XoST2vRo$"
    assert result.tasks[0].validation_status == "unlocalizable"
    assert result.tasks[0].target_xy is None
    assert result.executable is False


def test_bridge_reads_v3_query_plan_from_retrieval_packet() -> None:
    """Catches dropping certified v3 targets when debug omits query_plan."""
    node_id = "ifc_0aZxGK_jD1LR$XoST2vRo$"
    raw = {
        "answer": "Navigate(0aZxGK_jD1LR$XoST2vRo$)",
        "graph_hash": "graph-sha",
        "errors": [],
        "debug": {"closure_profile": "closure-adjudication-v3"},
        "retrieval_evidence_packet": {
            "query_plan": {"action_bindings": [{"action": "Navigate", "binding_index": 0}]}
        },
        "evidence_closure_certificate": {
            "closure_status": "pass",
            "stop_reason": "all_bindings_certified",
            "certified_target_ids": {"0": [node_id]},
        },
        "selected_entities": [
            {
                "node_id": node_id,
                "label": "CYBER-PHYSICAL SYSTEMS LAB",
                "global_id": "0aZxGK_jD1LR$XoST2vRo$",
                "ifc_class": "IfcSpace",
                "score": 0.96,
                "metadata": {"storey": "LEVEL 4", "name": "404"},
            }
        ],
        "evidence": [],
        "gnn_subgraph": {"node_ids": [], "edges": [], "node_scores": {}},
    }

    result = normalize_tog_response(raw)

    assert [task.node_id for task in result.tasks] == [node_id]
    assert result.tasks[0].validation_status == "unlocalizable"
    assert result.executable is False


def test_bridge_localizes_v3_target_from_verified_floor_bundle() -> None:
    """Catches certified arbitrary targets remaining blocked outside the Room 404 fixture."""
    node_id = "ifc_1X_G87E4f96Q$DWxa9TuYT"
    source_hash = "8" * 64
    floor = FloorMap(
        id="ecore-level-4-floorplan-v1",
        floor_id="LEVEL 4",
        origin=(100.0, 200.0),
        width_m=20.0,
        height_m=10.0,
        polygons=[
            MapPolygon(
                id="slab",
                kind="slab",
                points=[
                    Point2D(x=0.0, y=0.0),
                    Point2D(x=20.0, y=0.0),
                    Point2D(x=20.0, y=10.0),
                ],
            )
        ],
        occupancy_rows=["." * 200 for _ in range(100)],
        target_positions={node_id: (12.25, 4.75)},
        source_ifc_sha256=source_hash,
    )
    raw = {
        "answer": "Inspect(1X_G87E4f96Q$DWxa9TuYT)",
        "graph_hash": "graph-sha",
        "errors": [],
        "debug": {"closure_profile": "closure-adjudication-v3"},
        "retrieval_evidence_packet": {
            "query_plan": {"action_bindings": [{"action": "Inspect", "binding_index": 0}]}
        },
        "evidence_closure_certificate": {
            "closure_status": "pass",
            "stop_reason": "all_bindings_certified",
            "certified_target_ids": {"0": [node_id]},
        },
        "selected_entities": [
            {
                "node_id": node_id,
                "label": "EF_Outlet-Switch",
                "global_id": "1X_G87E4f96Q$DWxa9TuYT",
                "ifc_class": "IfcBuildingElementProxy",
                "score": 0.91,
                "metadata": {"storey": "LEVEL 4", "role": "switch"},
            }
        ],
        "evidence": [],
        "gnn_subgraph": {"node_ids": [], "edges": [], "node_scores": {}},
    }

    result = normalize_tog_response(
        raw,
        floor_maps={"LEVEL 4": floor},
        expected_ifc_sha256=source_hash,
    )

    assert result.tasks[0].target_xy == (12.25, 4.75)
    assert result.tasks[0].metadata["floor_map_id"] == floor.id
    assert result.tasks[0].validation_status == "certified"
    assert result.executable is True


def test_bridge_rejects_floor_bundle_from_a_different_ifc() -> None:
    """Catches navigation coordinates crossing IFC asset boundaries."""
    node_id = "ifc_target"
    floor = FloorMap(
        id="foreign-level-4-floorplan-v1",
        floor_id="LEVEL 4",
        origin=(0.0, 0.0),
        width_m=5.0,
        height_m=5.0,
        polygons=[],
        occupancy_rows=["." * 50 for _ in range(50)],
        target_positions={node_id: (2.0, 2.0)},
        source_ifc_sha256="f" * 64,
    )
    raw = {
        "answer": "Navigate(target)",
        "debug": {},
        "retrieval_evidence_packet": {
            "query_plan": {"action_bindings": [{"action": "Navigate", "binding_index": 0}]}
        },
        "evidence_closure_certificate": {
            "closure_status": "pass",
            "stop_reason": "all_bindings_certified",
            "certified_target_ids": {"0": [node_id]},
        },
        "selected_entities": [
            {
                "node_id": node_id,
                "label": "Target",
                "global_id": "target",
                "ifc_class": "IfcSpace",
                "metadata": {"storey": "LEVEL 4"},
            }
        ],
    }

    result = normalize_tog_response(
        raw,
        floor_maps={"LEVEL 4": floor},
        expected_ifc_sha256="e" * 64,
    )

    assert result.tasks[0].validation_status == "unlocalizable"
    assert "floor_map_id" not in result.tasks[0].metadata
    assert result.executable is False


def test_bridge_loads_only_floor_maps_for_the_runtime_ifc(tmp_path) -> None:
    """Catches a run consuming a stale or unrelated prebuilt floor bundle."""
    expected_hash = "a" * 64
    valid = FloorMap(
        id="ecore-level-4-floorplan-v1",
        floor_id="LEVEL 4",
        origin=(0.0, 0.0),
        width_m=1.0,
        height_m=1.0,
        polygons=[],
        occupancy_rows=["." * 10 for _ in range(10)],
        source_ifc_sha256=expected_hash,
    )
    foreign = valid.model_copy(
        update={"id": "foreign", "floor_id": "LEVEL 5", "source_ifc_sha256": "b" * 64}
    )
    (tmp_path / "valid.json").write_text(valid.model_dump_json(), encoding="utf-8")
    (tmp_path / "foreign.json").write_text(foreign.model_dump_json(), encoding="utf-8")
    (tmp_path / "broken.json").write_text("not-json", encoding="utf-8")

    floors = _load_floor_maps(tmp_path, expected_hash)

    assert list(floors) == ["LEVEL 4"]
    assert floors["LEVEL 4"].id == valid.id


def test_research_checkpoint_loader_forces_weights_only() -> None:
    """Catches the bundled runtime's legacy weights_only=False checkpoint call."""
    received: dict[str, object] = {}

    def loader(*args: object, **kwargs: object) -> str:
        received.update(kwargs)
        return "loaded"

    result = _weights_only_loader(loader)("checkpoint.bin", weights_only=False)

    assert result == "loaded"
    assert received["weights_only"] is True


def test_bridge_blocks_execution_when_certified_targets_span_floors() -> None:
    source_hash = "c" * 64
    floors = {
        name: FloorMap(
            id=f"{name.lower()}-map",
            floor_id=name,
            origin=(0.0, 0.0),
            width_m=5.0,
            height_m=5.0,
            polygons=[],
            occupancy_rows=["." * 50 for _ in range(50)],
            target_positions={f"ifc_{name}": (2.0, 2.0)},
            source_ifc_sha256=source_hash,
        )
        for name in ("LEVEL A", "LEVEL B")
    }
    raw = {
        "answer": "Two targets",
        "debug": {
            "query_plan": {
                "action_bindings": [
                    {"action": "Inspect", "binding_index": 0},
                    {"action": "Inspect", "binding_index": 1},
                ]
            }
        },
        "evidence_closure_certificate": {
            "closure_status": "pass",
            "stop_reason": "all_bindings_certified",
            "certified_target_ids": {"0": ["ifc_LEVEL A"], "1": ["ifc_LEVEL B"]},
        },
        "selected_entities": [
            {"node_id": "ifc_LEVEL A", "label": "A", "metadata": {"storey": "LEVEL A"}},
            {"node_id": "ifc_LEVEL B", "label": "B", "metadata": {"storey": "LEVEL B"}},
        ],
    }
    result = normalize_tog_response(raw, floor_maps=floors, expected_ifc_sha256=source_hash)
    assert all(task.validation_status == "certified" for task in result.tasks)
    assert result.executable is False
