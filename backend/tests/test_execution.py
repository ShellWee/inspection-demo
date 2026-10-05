import pytest
from inspection_demo.execution import (
    ROBOT_PROFILES,
    CrossFloorExecutionError,
    grid_from_floor,
    simulate_execution,
    suggested_order,
)
from inspection_demo.models import ActionSpec, CostEstimate, FloorMap, GroundedTask
from inspection_demo.navigation import Pose2D


def _floor() -> FloorMap:
    return FloorMap(
        id="floor",
        floor_id="LEVEL TEST",
        origin=(0.0, 0.0),
        width_m=10.0,
        height_m=10.0,
        polygons=[],
        occupancy_rows=["." * 100 for _ in range(100)],
        source_ifc_sha256="a" * 64,
    )


def _task(task_id: str, risk: float, distance: float) -> GroundedTask:
    return GroundedTask(
        task_id=task_id,
        action="Navigate",
        binding_index=0,
        node_id=f"ifc_{task_id}",
        ifc_guid=task_id,
        floor_id="LEVEL TEST",
        target_name=task_id,
        target_kind="IfcSpace",
        target_xy=(7.0, 7.0),
        retrieval_score=0.8,
        validation_status="certified",
        action_spec=ActionSpec(
            name="Navigate",
            cost=CostEstimate(
                distance_m=distance,
                duration_s=distance,
                risk_score=risk,
                information_gain=0.5,
                model_score=0.8,
            ),
        ),
    )


def test_suggested_order_uses_safety_then_distance() -> None:
    assert [
        task.task_id for task in suggested_order([_task("risky", 0.4, 1), _task("safe", 0.1, 3)])
    ] == ["safe", "risky"]


def test_dwa_execution_reaches_verified_target_without_collision() -> None:
    result = simulate_execution(
        floor=_floor(),
        tasks=[_task("target", 0.1, 5)],
        initial_pose=Pose2D(2.0, 2.0, 0.0),
        robot=ROBOT_PROFILES["jackal"],
        planner="dwa",
    )
    assert result.status == "completed"
    assert result.completed_task_ids == ["target"]
    assert result.frames[-1].status == "completed"
    assert all(frame.t == round(frame.t, 6) for frame in result.frames)


def test_execution_rejects_targets_on_multiple_floors() -> None:
    other = _task("other", 0.1, 2).model_copy(update={"floor_id": "LEVEL OTHER"})
    with pytest.raises(CrossFloorExecutionError, match="one floor"):
        simulate_execution(
            floor=_floor(),
            tasks=[_task("first", 0.1, 1), other],
            initial_pose=Pose2D(2.0, 2.0, 0.0),
            robot=ROBOT_PROFILES["jackal"],
            planner="astar",
        )


def test_unlocalizable_target_has_no_zero_coordinate_fallback() -> None:
    task = _task("missing", 0.1, 1).model_copy(
        update={"target_xy": None, "validation_status": "unlocalizable"}
    )
    with pytest.raises(ValueError, match="certified"):
        simulate_execution(
            floor=_floor(),
            tasks=[task],
            initial_pose=Pose2D(2.0, 2.0, 0.0),
            robot=ROBOT_PROFILES["jackal"],
            planner="astar",
        )


def test_navigation_grid_stays_in_floor_local_coordinates() -> None:
    floor = _floor().model_copy(update={"origin": (589_892.2, 69_481.0)})
    grid = grid_from_floor(floor)
    assert grid.origin_x == 0.0
    assert grid.origin_y == 0.0
    assert grid.is_occupied_world(0.5, 0.5) is False
