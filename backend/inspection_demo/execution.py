from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Event

from .models import ExecutionFrame, ExecutionResult, FloorMap, GroundedTask, Pose2DModel
from .navigation import GridMap, NavigationError, PlannerName, Pose2D, plan_path, sample_target_pose


class CrossFloorExecutionError(ValueError):
    """Raised when a v1 execution contains tasks from multiple floors."""


@dataclass(frozen=True, slots=True)
class RobotSimulationProfile:
    id: str
    label: str
    radius_m: float
    clearance_m: float
    max_linear_mps: float
    max_angular_rps: float


ROBOT_PROFILES: dict[str, RobotSimulationProfile] = {
    "jackal": RobotSimulationProfile("jackal", "Clearpath Jackal", 0.31, 0.10, 1.0, 1.8),
    "husky": RobotSimulationProfile("husky", "Clearpath Husky", 0.48, 0.12, 0.9, 1.4),
    "robot-dog": RobotSimulationProfile("robot-dog", "Robot dog", 0.34, 0.10, 1.2, 2.0),
}


def suggested_order(tasks: list[GroundedTask]) -> list[GroundedTask]:
    feasible = [task for task in tasks if task.enabled and task.validation_status == "certified"]
    return sorted(
        feasible,
        key=lambda task: (
            task.action_spec.cost.risk_score,
            task.action_spec.cost.distance_m,
            -task.action_spec.cost.information_gain,
            task.action_spec.cost.duration_s,
            -task.action_spec.cost.model_score,
            task.binding_index,
        ),
    )


def grid_from_floor(floor: FloorMap, cancel_event: Event | None = None) -> GridMap:
    base = GridMap.from_rows(floor.occupancy_rows, floor.resolution_m)
    return GridMap(
        occupied=base.occupied,
        resolution_m=base.resolution_m,
        origin_x=0.0,
        origin_y=0.0,
        cancelled=cancel_event.is_set if cancel_event else None,
    )


def simulate_execution(
    *,
    floor: FloorMap,
    tasks: list[GroundedTask],
    initial_pose: Pose2D,
    robot: RobotSimulationProfile,
    planner: PlannerName,
    cancel_event: Event | None = None,
) -> ExecutionResult:
    enabled_tasks = [task for task in tasks if task.enabled]
    if not enabled_tasks:
        raise ValueError("at least one enabled task is required")
    floors = {task.floor_id for task in enabled_tasks}
    if len(floors) != 1 or floor.floor_id not in floors:
        raise CrossFloorExecutionError("v1 execution supports exactly one floor")
    if any(task.validation_status != "certified" for task in enabled_tasks):
        raise ValueError("every enabled task must have a certified target")
    if any(task.target_xy is None for task in enabled_tasks):
        raise ValueError("every enabled task must have a verified navigation position")

    grid = grid_from_floor(floor, cancel_event)
    footprint_radius = robot.radius_m + robot.clearance_m
    if grid.is_occupied_world(initial_pose.x, initial_pose.y, footprint_radius):
        raise NavigationError("initial pose is occupied")

    frames: list[ExecutionFrame] = [
        ExecutionFrame(
            t=0,
            pose=Pose2DModel(x=initial_pose.x, y=initial_pose.y, yaw=initial_pose.yaw),
            task_id=enabled_tasks[0].task_id,
            status="navigating",
        )
    ]
    completed: list[str] = []
    current = initial_pose
    elapsed = 0.0
    frame_period = 0.2
    for task in enabled_tasks:
        target_pose = sample_target_pose(
            grid=grid,
            start=current,
            target_xy=task.target_xy,
            action=task.action,
            robot_radius_m=footprint_radius,
        )
        path = plan_path(
            grid,
            current,
            target_pose,
            planner=planner,
            robot_radius_m=footprint_radius,
        )
        for start, end in zip(path, path[1:], strict=False):
            segment_length = math.hypot(end.x - start.x, end.y - start.y)
            steps = max(1, math.ceil(segment_length / (robot.max_linear_mps * frame_period)))
            for step in range(1, steps + 1):
                if cancel_event and cancel_event.is_set():
                    raise NavigationError("simulation cancelled")
                ratio = step / steps
                x = start.x + (end.x - start.x) * ratio
                y = start.y + (end.y - start.y) * ratio
                yaw = math.atan2(end.y - start.y, end.x - start.x)
                elapsed = round(elapsed + frame_period, 6)
                frames.append(
                    ExecutionFrame(
                        t=elapsed,
                        pose=Pose2DModel(x=x, y=y, yaw=yaw),
                        task_id=task.task_id,
                        status="navigating",
                    )
                )
        current = target_pose
        dwell_seconds = 3.0 if task.action == "Scan" else 5.0 if task.action == "Inspect" else 0.0
        for _ in range(round(dwell_seconds / frame_period)):
            elapsed = round(elapsed + frame_period, 6)
            frames.append(
                ExecutionFrame(
                    t=elapsed,
                    pose=Pose2DModel(x=current.x, y=current.y, yaw=current.yaw),
                    task_id=task.task_id,
                    status="dwelling",
                )
            )
        completed.append(task.task_id)
        elapsed = round(elapsed + frame_period, 6)
        frames.append(
            ExecutionFrame(
                t=elapsed,
                pose=Pose2DModel(x=current.x, y=current.y, yaw=current.yaw),
                task_id=task.task_id,
                status="completed",
            )
        )

    return ExecutionResult(
        status="completed",
        frames=frames,
        completed_task_ids=completed,
        terminal_reason="all_tasks_completed",
    )
