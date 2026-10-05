from __future__ import annotations

import heapq
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

PlannerName = Literal["astar", "dwa", "rrt_star"]


class NavigationError(RuntimeError):
    """Raised when no collision-free route can be produced."""


class UnlocalizableTarget(NavigationError):
    """Raised when the target has no reachable navigation pose."""


@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True, slots=True)
class GridMap:
    occupied: tuple[tuple[bool, ...], ...]
    resolution_m: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    cancelled: Callable[[], bool] | None = None

    @classmethod
    def from_rows(cls, rows: list[str], resolution_m: float) -> GridMap:
        if not rows or not rows[0]:
            raise ValueError("grid must contain at least one cell")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("grid rows must have equal width")
        if resolution_m <= 0:
            raise ValueError("resolution must be positive")
        return cls(
            occupied=tuple(tuple(cell == "#" for cell in row) for row in rows),
            resolution_m=resolution_m,
        )

    @property
    def width(self) -> int:
        return len(self.occupied[0])

    @property
    def height(self) -> int:
        return len(self.occupied)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor((x - self.origin_x) / self.resolution_m),
            math.floor((y - self.origin_y) / self.resolution_m),
        )

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        col, row = cell
        return (
            self.origin_x + (col + 0.5) * self.resolution_m,
            self.origin_y + (row + 0.5) * self.resolution_m,
        )

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        col, row = cell
        return 0 <= col < self.width and 0 <= row < self.height

    def is_occupied_cell(self, cell: tuple[int, int]) -> bool:
        if self.cancelled and self.cancelled():
            raise NavigationError("simulation cancelled")
        if not self.in_bounds(cell):
            return True
        col, row = cell
        return self.occupied[row][col]

    def is_occupied_world(self, x: float, y: float, radius_m: float = 0.0) -> bool:
        center = self.world_to_cell(x, y)
        if radius_m <= 0:
            return self.is_occupied_cell(center)
        max_x = self.origin_x + self.width * self.resolution_m
        max_y = self.origin_y + self.height * self.resolution_m
        if (
            x - radius_m < self.origin_x
            or y - radius_m < self.origin_y
            or x + radius_m > max_x
            or y + radius_m > max_y
        ):
            return True
        min_col = math.floor((x - radius_m - self.origin_x) / self.resolution_m)
        max_col = math.floor((x + radius_m - self.origin_x) / self.resolution_m)
        min_row = math.floor((y - radius_m - self.origin_y) / self.resolution_m)
        max_row = math.floor((y + radius_m - self.origin_y) / self.resolution_m)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if not self.is_occupied_cell((col, row)):
                    continue
                cell_min_x = self.origin_x + col * self.resolution_m
                cell_max_x = cell_min_x + self.resolution_m
                cell_min_y = self.origin_y + row * self.resolution_m
                cell_max_y = cell_min_y + self.resolution_m
                nearest_x = min(max(x, cell_min_x), cell_max_x)
                nearest_y = min(max(y, cell_min_y), cell_max_y)
                if math.hypot(nearest_x - x, nearest_y - y) <= radius_m:
                    return True
        return False


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reachable_costs(
    grid: GridMap,
    start: tuple[int, int],
    robot_radius_m: float,
) -> dict[tuple[int, int], float]:
    if grid.is_occupied_world(*grid.cell_to_world(start), radius_m=robot_radius_m):
        return {}
    costs = {start: 0.0}
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    )
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current_cost > costs[current]:
            continue
        for dx, dy, step_cost in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            wx, wy = grid.cell_to_world(nxt)
            if grid.is_occupied_world(wx, wy, radius_m=robot_radius_m):
                continue
            if dx and dy:
                side_a = (current[0] + dx, current[1])
                side_b = (current[0], current[1] + dy)
                if grid.is_occupied_cell(side_a) or grid.is_occupied_cell(side_b):
                    continue
            new_cost = current_cost + step_cost
            if new_cost < costs.get(nxt, math.inf):
                costs[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, nxt))
    return costs


def _astar_cells(
    grid: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    robot_radius_m: float = 0.0,
) -> list[tuple[int, int]]:
    if grid.is_occupied_world(*grid.cell_to_world(start), radius_m=robot_radius_m):
        raise NavigationError("initial pose is occupied")
    if grid.is_occupied_world(*grid.cell_to_world(goal), radius_m=robot_radius_m):
        raise NavigationError("goal pose is occupied")

    frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far = {start: 0.0}
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    )

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        if current_cost > cost_so_far[current]:
            continue
        for dx, dy, step_cost in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            wx, wy = grid.cell_to_world(nxt)
            if grid.is_occupied_world(wx, wy, radius_m=robot_radius_m):
                continue
            if dx and dy:
                side_a = (current[0] + dx, current[1])
                side_b = (current[0], current[1] + dy)
                if grid.is_occupied_cell(side_a) or grid.is_occupied_cell(side_b):
                    continue
            new_cost = current_cost + step_cost
            if new_cost < cost_so_far.get(nxt, math.inf):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                priority = new_cost + _heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, new_cost, nxt))

    if goal not in came_from:
        raise NavigationError("no collision-free path")
    cells: list[tuple[int, int]] = []
    current: tuple[int, int] | None = goal
    while current is not None:
        cells.append(current)
        current = came_from[current]
    return list(reversed(cells))


def _segment_is_free(
    grid: GridMap,
    start: Pose2D,
    end: Pose2D,
    robot_radius_m: float,
) -> bool:
    length = math.hypot(end.x - start.x, end.y - start.y)
    samples = max(1, math.ceil(length / (grid.resolution_m * 0.25)))
    for index in range(samples + 1):
        ratio = index / samples
        x = start.x + (end.x - start.x) * ratio
        y = start.y + (end.y - start.y) * ratio
        if grid.is_occupied_world(x, y, radius_m=robot_radius_m):
            return False
    return True


def _poses_from_cells(
    grid: GridMap,
    cells: list[tuple[int, int]],
    start: Pose2D,
    goal: Pose2D,
) -> list[Pose2D]:
    poses: list[Pose2D] = []
    for index, cell in enumerate(cells):
        x, y = grid.cell_to_world(cell)
        if index + 1 < len(cells):
            next_x, next_y = grid.cell_to_world(cells[index + 1])
            yaw = math.atan2(next_y - y, next_x - x)
        else:
            yaw = goal.yaw
        poses.append(Pose2D(x, y, yaw))
    poses[0] = start
    poses[-1] = goal
    return poses


def _angle_delta(target: float, source: float) -> float:
    return math.atan2(math.sin(target - source), math.cos(target - source))


def _dwa_track_reference(
    grid: GridMap,
    reference: list[Pose2D],
    start: Pose2D,
    goal: Pose2D,
    robot_radius_m: float,
) -> list[Pose2D]:
    """Track an A* reference with a deterministic local velocity search."""
    current = start
    poses = [start]
    dt = 0.2
    arrival_tolerance = max(0.025, grid.resolution_m * 0.25)
    reference_stride = max(1, round(0.5 / grid.resolution_m))
    tracking_reference = reference[reference_stride::reference_stride]
    if not tracking_reference or tracking_reference[-1] != goal:
        tracking_reference.append(goal)

    for waypoint_index, waypoint in enumerate(tracking_reference, start=1):
        for _ in range(120):
            waypoint_distance = math.hypot(waypoint.x - current.x, waypoint.y - current.y)
            if waypoint_distance <= arrival_tolerance:
                if _segment_is_free(grid, current, waypoint, robot_radius_m):
                    current = Pose2D(waypoint.x, waypoint.y, current.yaw)
                    poses.append(current)
                    break
                raise NavigationError("DWA cannot connect to its global reference")

            desired_heading = math.atan2(waypoint.y - current.y, waypoint.x - current.x)
            heading_delta = _angle_delta(desired_heading, current.yaw)
            preferred_angular = max(-1.5, min(1.5, heading_delta / dt))
            angular_samples = {
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                preferred_angular,
            }
            preferred_linear = min(0.75, waypoint_distance / dt)
            linear_samples = {
                0.0,
                preferred_linear * 0.35,
                preferred_linear * 0.7,
                preferred_linear,
            }
            if abs(heading_delta) > 0.65:
                linear_samples = {0.0, preferred_linear * 0.2}

            candidates: list[tuple[float, Pose2D]] = []
            for linear in linear_samples:
                for angular in angular_samples:
                    yaw = current.yaw + angular * dt
                    midpoint_yaw = current.yaw + angular * dt * 0.5
                    candidate = Pose2D(
                        current.x + linear * math.cos(midpoint_yaw) * dt,
                        current.y + linear * math.sin(midpoint_yaw) * dt,
                        yaw,
                    )
                    if not _segment_is_free(grid, current, candidate, robot_radius_m):
                        continue
                    candidate_distance = math.hypot(
                        waypoint.x - candidate.x, waypoint.y - candidate.y
                    )
                    heading_error = abs(_angle_delta(desired_heading, candidate.yaw))
                    score = candidate_distance + 0.12 * heading_error - 0.02 * linear
                    candidates.append((score, candidate))

            if not candidates:
                raise NavigationError("DWA found no collision-free local velocity")
            _, current = min(candidates, key=lambda item: item[0])
            poses.append(current)
        else:
            raise NavigationError(f"DWA stalled before global-reference waypoint {waypoint_index}")

    if not _segment_is_free(grid, current, goal, robot_radius_m):
        raise NavigationError("DWA cannot connect to the goal pose")
    poses.append(goal)
    return poses


@dataclass(slots=True)
class _RrtNode:
    pose: Pose2D
    parent: int | None
    cost: float


def _rrt_star_path(
    grid: GridMap,
    start: Pose2D,
    goal: Pose2D,
    robot_radius_m: float,
) -> list[Pose2D]:
    """Plan a deterministic continuous-space route with RRT* rewiring."""
    if grid.is_occupied_world(start.x, start.y, radius_m=robot_radius_m):
        raise NavigationError("initial pose is occupied")
    if grid.is_occupied_world(goal.x, goal.y, radius_m=robot_radius_m):
        raise NavigationError("goal pose is occupied")

    random_source = random.Random(43)
    step_size = max(0.35, grid.resolution_m * 1.5)
    near_radius = step_size * 2.5
    min_x = grid.origin_x + robot_radius_m
    max_x = grid.origin_x + grid.width * grid.resolution_m - robot_radius_m
    min_y = grid.origin_y + robot_radius_m
    max_y = grid.origin_y + grid.height * grid.resolution_m - robot_radius_m
    nodes = [_RrtNode(start, None, 0.0)]
    goal_index: int | None = None

    for iteration in range(5000):
        sample = (
            goal
            if iteration % 8 == 0
            else Pose2D(
                random_source.uniform(min_x, max_x),
                random_source.uniform(min_y, max_y),
            )
        )
        nearest_index = min(
            range(len(nodes)),
            key=lambda index: math.hypot(
                sample.x - nodes[index].pose.x,
                sample.y - nodes[index].pose.y,
            ),
        )
        nearest = nodes[nearest_index].pose
        distance = math.hypot(sample.x - nearest.x, sample.y - nearest.y)
        if distance <= 1e-9:
            continue
        ratio = min(1.0, step_size / distance)
        candidate = Pose2D(
            nearest.x + (sample.x - nearest.x) * ratio,
            nearest.y + (sample.y - nearest.y) * ratio,
        )
        if not _segment_is_free(grid, nearest, candidate, robot_radius_m):
            continue

        nearby = [
            index
            for index, node in enumerate(nodes)
            if math.hypot(candidate.x - node.pose.x, candidate.y - node.pose.y) <= near_radius
            and _segment_is_free(grid, node.pose, candidate, robot_radius_m)
        ]
        parent_index = min(
            nearby or [nearest_index],
            key=lambda index: (
                nodes[index].cost
                + math.hypot(candidate.x - nodes[index].pose.x, candidate.y - nodes[index].pose.y)
            ),
        )
        parent = nodes[parent_index]
        candidate_cost = parent.cost + math.hypot(
            candidate.x - parent.pose.x, candidate.y - parent.pose.y
        )
        nodes.append(_RrtNode(candidate, parent_index, candidate_cost))
        candidate_index = len(nodes) - 1

        for nearby_index in nearby:
            if nearby_index == parent_index:
                continue
            nearby_node = nodes[nearby_index]
            rewired_cost = candidate_cost + math.hypot(
                nearby_node.pose.x - candidate.x,
                nearby_node.pose.y - candidate.y,
            )
            if rewired_cost < nearby_node.cost:
                nearby_node.parent = candidate_index
                nearby_node.cost = rewired_cost

        goal_distance = math.hypot(goal.x - candidate.x, goal.y - candidate.y)
        if goal_distance <= step_size and _segment_is_free(grid, candidate, goal, robot_radius_m):
            nodes.append(_RrtNode(goal, candidate_index, candidate_cost + goal_distance))
            goal_index = len(nodes) - 1
            break

    if goal_index is None:
        raise NavigationError("RRT* found no collision-free path")

    reversed_path: list[Pose2D] = []
    current_index: int | None = goal_index
    while current_index is not None:
        reversed_path.append(nodes[current_index].pose)
        current_index = nodes[current_index].parent
    positions = list(reversed(reversed_path))
    poses: list[Pose2D] = []
    for index, pose in enumerate(positions):
        if index + 1 < len(positions):
            nxt = positions[index + 1]
            yaw = math.atan2(nxt.y - pose.y, nxt.x - pose.x)
        else:
            yaw = goal.yaw
        poses.append(Pose2D(pose.x, pose.y, yaw))
    poses[0] = start
    poses[-1] = goal
    return poses


def plan_path(
    grid: GridMap,
    start: Pose2D,
    goal: Pose2D,
    *,
    planner: PlannerName,
    robot_radius_m: float = 0.0,
) -> list[Pose2D]:
    if planner not in {"astar", "dwa", "rrt_star"}:
        raise ValueError(f"unsupported planner: {planner}")
    cells = _astar_cells(
        grid,
        grid.world_to_cell(start.x, start.y),
        grid.world_to_cell(goal.x, goal.y),
        robot_radius_m,
    )
    astar_reference = _poses_from_cells(grid, cells, start, goal)
    if planner == "astar":
        return astar_reference
    if planner == "dwa":
        return _dwa_track_reference(
            grid,
            astar_reference,
            start,
            goal,
            robot_radius_m,
        )
    return _rrt_star_path(grid, start, goal, robot_radius_m)


def sample_target_pose(
    *,
    grid: GridMap,
    start: Pose2D,
    target_xy: tuple[float, float],
    action: str,
    robot_radius_m: float,
) -> Pose2D:
    target_x, target_y = target_xy
    distances = (0.0, 0.4, 0.8) if action.lower() == "navigate" else (0.8, 1.2, 1.6)
    reachable_costs = _reachable_costs(
        grid,
        grid.world_to_cell(start.x, start.y),
        robot_radius_m,
    )
    candidates: list[tuple[float, Pose2D]] = []
    for distance in distances:
        sample_count = 1 if distance == 0 else 32
        for index in range(sample_count):
            angle = 2 * math.pi * index / sample_count
            x = target_x + distance * math.cos(angle)
            y = target_y + distance * math.sin(angle)
            if grid.is_occupied_world(x, y, radius_m=robot_radius_m):
                continue
            pose = Pose2D(x, y, math.atan2(target_y - y, target_x - x))
            path_cost = reachable_costs.get(grid.world_to_cell(x, y))
            if path_cost is None:
                continue
            path_length = path_cost * grid.resolution_m
            turn_cost = abs(
                math.atan2(math.sin(pose.yaw - start.yaw), math.cos(pose.yaw - start.yaw))
            )
            candidates.append((path_length + turn_cost * 0.05, pose))
    if not candidates:
        raise UnlocalizableTarget("no collision-free navigation pose")
    return min(candidates, key=lambda item: item[0])[1]
