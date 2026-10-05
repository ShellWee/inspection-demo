from math import isclose

import pytest
from inspection_demo.navigation import (
    GridMap,
    Pose2D,
    UnlocalizableTarget,
    plan_path,
    sample_target_pose,
)


def test_astar_routes_around_an_obstacle_wall() -> None:
    """Catches planners that walk through occupied cells or fail to reach the goal."""
    grid = GridMap.from_rows(
        [
            "..........",
            "....#.....",
            "....#.....",
            "....#.....",
            "..........",
        ],
        resolution_m=1.0,
    )

    path = plan_path(grid, Pose2D(1, 2, 0), Pose2D(8, 2, 0), planner="astar")

    assert path[0].x == 1 and path[0].y == 2
    assert path[-1].x == 8 and path[-1].y == 2
    assert all(not grid.is_occupied_world(p.x, p.y) for p in path)
    assert any(p.y <= 0.5 or p.y >= 4.5 for p in path)


@pytest.mark.parametrize("planner", ["astar", "dwa", "rrt_star"])
def test_each_planner_returns_a_collision_free_path(planner: str) -> None:
    """Catches algorithm adapters that return empty or colliding paths."""
    grid = GridMap.from_rows(
        [
            "............",
            "...###......",
            "............",
            "......###...",
            "............",
        ],
        resolution_m=0.5,
    )

    path = plan_path(grid, Pose2D(0.25, 1.25, 0), Pose2D(5.25, 1.25, 0), planner=planner)

    assert len(path) >= 2
    assert all(not grid.is_occupied_world(p.x, p.y) for p in path)
    assert isclose(path[-1].x, 5.25, abs_tol=0.51)
    assert isclose(path[-1].y, 1.25, abs_tol=0.51)


def test_dwa_tracks_an_astar_reference_with_local_velocity_samples() -> None:
    """Catches a DWA selector that is only an alias for the A* polyline."""
    grid = GridMap.from_rows(
        [
            "............",
            "...###......",
            "............",
            "......###...",
            "............",
        ],
        resolution_m=0.5,
    )
    start = Pose2D(0.25, 1.25, 0)
    goal = Pose2D(5.25, 1.25, 0)

    astar = plan_path(grid, start, goal, planner="astar")
    dwa = plan_path(grid, start, goal, planner="dwa")

    assert len(dwa) > len(astar)
    assert [(pose.x, pose.y) for pose in dwa] != [(pose.x, pose.y) for pose in astar]


def test_rrt_star_produces_a_continuous_space_route() -> None:
    """Catches an RRT* selector that silently returns occupancy-grid centres."""
    grid = GridMap.from_rows(
        [
            "............",
            "...###......",
            "............",
            "......###...",
            "............",
        ],
        resolution_m=0.5,
    )

    path = plan_path(
        grid,
        Pose2D(0.25, 1.25, 0),
        Pose2D(5.25, 1.25, 0),
        planner="rrt_star",
    )

    assert any(
        not isclose((pose.x / grid.resolution_m) % 1, 0.5, abs_tol=1e-6)
        or not isclose((pose.y / grid.resolution_m) % 1, 0.5, abs_tol=1e-6)
        for pose in path[1:-1]
    )


def test_sampler_picks_a_reachable_pose_facing_the_target() -> None:
    """Catches pose sampling that ignores reachability or target-facing yaw."""
    grid = GridMap.from_rows(["........", "........", "........", "........"], 0.5)

    pose = sample_target_pose(
        grid=grid,
        start=Pose2D(0.25, 0.75, 0),
        target_xy=(2.25, 0.75),
        action="Inspect",
        robot_radius_m=0.2,
    )

    assert not grid.is_occupied_world(pose.x, pose.y)
    assert 0.7 <= ((pose.x - 2.25) ** 2 + (pose.y - 0.75) ** 2) ** 0.5 <= 1.7
    assert abs(pose.yaw) <= 3.1416


def test_sampler_reports_unlocalizable_target_when_no_candidate_is_free() -> None:
    """Catches silent fallback to an invalid target centroid."""
    grid = GridMap.from_rows(["#####", "#####", "#####", "#####", "#####"], 0.5)

    with pytest.raises(UnlocalizableTarget, match="no collision-free navigation pose"):
        sample_target_pose(
            grid=grid,
            start=Pose2D(0.25, 0.25, 0),
            target_xy=(1.25, 1.25),
            action="Scan",
            robot_radius_m=0.2,
        )
