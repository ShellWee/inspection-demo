from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Event
from typing import Any

from .execution import ROBOT_PROFILES, simulate_execution
from .grounding import GroundingAdapter
from .models import (
    ExecutionRequest,
    ExecutionResult,
    FloorMap,
    GroundingRequest,
    GroundingResult,
    RunEvent,
)
from .navigation import NavigationError, Pose2D
from .repository import RunRecord, RunStore

TERMINAL_STATUSES = frozenset({"completed", "abstained", "failed", "cancelled"})


class BusyError(RuntimeError):
    """The single live worker is already reserved by another job."""


class JobManager:
    def __init__(
        self,
        *,
        store: RunStore,
        grounding_adapter: GroundingAdapter,
        floor_loader: Callable[[str], FloorMap | None],
        simulation_event_period_seconds: float = 0.0,
    ) -> None:
        self.store = store
        self.grounding_adapter = grounding_adapter
        self.floor_loader = floor_loader
        self.simulation_event_period_seconds = simulation_event_period_seconds
        self._worker = asyncio.Semaphore(1)
        self._tasks: set[asyncio.Task[None]] = set()
        self._tasks_by_record: dict[str, asyncio.Task[None]] = {}
        self._busy_record_id: str | None = None

    @property
    def busy(self) -> bool:
        return self._busy_record_id is not None

    def _reserve(self) -> None:
        if self.busy:
            raise BusyError("Another grounding or execution is already running")
        self._busy_record_id = "reserved"

    def create_grounding(self, *, owner: str, request: GroundingRequest) -> RunRecord:
        self._reserve()
        try:
            sanitized = request.model_dump(mode="json", exclude={"api_key"})
            record = self.store.create(kind="grounding", owner=owner, request=sanitized)
            self._busy_record_id = record.id
            api_key = request.api_key.get_secret_value()
            task = asyncio.create_task(self._run_grounding(record.id, request, api_key))
            self._track(task, record.id)
            return record
        except Exception:
            self._busy_record_id = None
            raise

    async def _run_grounding(self, record_id: str, request: GroundingRequest, api_key: str) -> None:
        async with self._worker:
            try:
                self.store.set_status(record_id, "running")
                await self._event(record_id, "queued", 1, "Grounding worker acquired.")

                async def progress(phase: str, percent: int, message: str) -> None:
                    await self._event(record_id, phase, percent, message)

                result = await self.grounding_adapter.ground(
                    query=request.query,
                    model_id=request.model_id,
                    api_key=api_key,
                    progress=progress,
                    **({"limits": request.limits} if request.limits else {}),
                )
                self.store.set_result(
                    record_id,
                    status=result.status,
                    result=result.model_dump(mode="json"),
                )
                await self._event(record_id, result.status, 100, "Grounding finished.")
            except Exception as error:  # noqa: BLE001 - persisted as a sanitized job failure
                result = GroundingResult(
                    status="failed",
                    answer="Grounding failed.",
                    closure_status="failed",
                    closure_stop_reason="runtime_error",
                    errors=[type(error).__name__],
                    executable=False,
                )
                self.store.set_result(
                    record_id, status="failed", result=result.model_dump(mode="json")
                )
                await self._event(record_id, "failed", 100, "Grounding failed.")
            finally:
                api_key = ""  # noqa: F841 - drop the only long-lived reference promptly

    def create_execution(self, *, owner: str, request: ExecutionRequest) -> RunRecord:
        self._reserve()
        try:
            return self._create_execution_reserved(owner=owner, request=request)
        except Exception:
            self._busy_record_id = None
            raise

    def _create_execution_reserved(self, *, owner: str, request: ExecutionRequest) -> RunRecord:
        grounding_record = self.store.get(request.grounding_run_id)
        if grounding_record is None or grounding_record.owner != owner:
            raise KeyError(request.grounding_run_id)
        grounding_payload = self.store.result(grounding_record)
        if grounding_payload is None:
            raise RuntimeError("Grounding run is not complete")
        grounding = GroundingResult.model_validate(grounding_payload)
        if not grounding.executable or grounding.query_type != "planning":
            raise PermissionError("Grounding run is not executable")
        known = {task.task_id for task in grounding.tasks}
        if any(item.task_id not in known for item in request.ordered_tasks):
            raise ValueError("Execution contains an unknown task")
        record = self.store.create(
            kind="execution",
            owner=owner,
            request=request.model_dump(mode="json"),
        )
        self._busy_record_id = record.id
        task = asyncio.create_task(self._run_execution(record.id, request, grounding))
        self._track(task, record.id)
        return record

    async def abort_execution(self, record_id: str) -> ExecutionResult:
        record = self.store.get(record_id)
        if record is None or record.kind != "execution":
            raise KeyError(record_id)
        existing = self.store.result(record)
        if record.status in TERMINAL_STATUSES:
            if existing is None:
                raise RuntimeError("Execution has already terminated")
            return ExecutionResult.model_validate(existing)
        task = self._tasks_by_record.get(record_id)
        if task is None:
            raise RuntimeError("Execution worker is not available")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        result = ExecutionResult(
            status="cancelled",
            frames=[],
            completed_task_ids=[],
            terminal_reason="user_abort",
        )
        self.store.set_result(
            record_id,
            status="cancelled",
            result=result.model_dump(mode="json"),
        )
        await self._event(record_id, "cancelled", 100, "Execution aborted by the user.")
        return result

    async def _run_execution(
        self,
        record_id: str,
        request: ExecutionRequest,
        grounding: GroundingResult,
    ) -> None:
        async with self._worker:
            try:
                self.store.set_status(record_id, "running")
                await self._event(record_id, "planning", 5, "Sampling navigation poses.")
                by_id = {task.task_id: task for task in grounding.tasks}
                ordered = sorted(request.ordered_tasks, key=lambda item: item.order)
                tasks = [
                    by_id[item.task_id].model_copy(update={"enabled": item.enabled})
                    for item in ordered
                    if item.task_id in by_id
                ]
                if len(tasks) != len(ordered):
                    raise ValueError("Execution contains an unknown task")
                floor_map_ids = {
                    str(task.metadata.get("floor_map_id") or "") for task in tasks if task.enabled
                }
                if len(floor_map_ids) != 1 or "" in floor_map_ids:
                    raise ValueError("Enabled tasks do not resolve to exactly one floor map")
                floor = self.floor_loader(next(iter(floor_map_ids)))
                if floor is None:
                    raise ValueError("Grounded floor map is unavailable")
                cancel_event = Event()
                simulation = asyncio.create_task(
                    asyncio.to_thread(
                        simulate_execution,
                        floor=floor,
                        tasks=tasks,
                        initial_pose=Pose2D(
                            request.initial_pose.x,
                            request.initial_pose.y,
                            request.initial_pose.yaw,
                        ),
                        robot=ROBOT_PROFILES[request.robot_profile_id],
                        planner=request.planner_id,
                        cancel_event=cancel_event,
                    )
                )
                try:
                    result = await asyncio.wait_for(asyncio.shield(simulation), timeout=600)
                except (asyncio.CancelledError, TimeoutError):
                    cancel_event.set()
                    # Keep the worker reservation until the CPU planner really stops.
                    await asyncio.gather(simulation, return_exceptions=True)
                    raise
                total_frames = max(1, len(result.frames))
                for index, frame in enumerate(result.frames):
                    await self._event(
                        record_id,
                        "simulation",
                        min(99, 10 + round(89 * index / total_frames)),
                        "Robot is moving.",
                        {"frame": frame.model_dump(mode="json")},
                    )
                    if self.simulation_event_period_seconds:
                        await asyncio.sleep(self.simulation_event_period_seconds)
                self.store.set_result(
                    record_id,
                    status=result.status,
                    result=result.model_dump(mode="json"),
                )
                await self._event(record_id, result.status, 100, "Simulation finished.")
            except Exception as error:  # noqa: BLE001 - convert worker failures into public status
                failed = ExecutionResult(
                    status="failed",
                    frames=[],
                    completed_task_ids=[],
                    terminal_reason=str(error)[:200]
                    if isinstance(error, (ValueError, NavigationError))
                    else type(error).__name__,
                )
                self.store.set_result(
                    record_id, status="failed", result=failed.model_dump(mode="json")
                )
                await self._event(record_id, "failed", 100, "Execution failed.")

    async def _event(
        self,
        record_id: str,
        phase: str,
        progress: int,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = RunEvent(
            event_id=len(self.store.events(record_id)) + 1,
            timestamp=datetime.now(UTC),
            phase=phase,
            progress=progress,
            message=message,
            payload=payload or {},
        )
        self.store.append_event(record_id, event)
        await asyncio.sleep(0)

    def _track(self, task: asyncio.Task[None], record_id: str) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._tasks_by_record[record_id] = task

        def remove_record(completed: asyncio.Task[None]) -> None:
            if self._tasks_by_record.get(record_id) is completed:
                self._tasks_by_record.pop(record_id, None)
            if self._busy_record_id == record_id:
                self._busy_record_id = None

        task.add_done_callback(remove_record)

    async def shutdown(self) -> None:
        if self._tasks:
            for task in tuple(self._tasks):
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
