from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

ActionName = Literal["Navigate", "Scan", "Inspect"]
RunStatus = Literal["queued", "running", "completed", "abstained", "failed", "cancelled"]
ReasoningModel = Literal["gpt-4.1", "gpt-5", "gpt-5.6-luna"]


class Pose2DModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    x: float
    y: float
    yaw: float = 0.0


class CostEstimate(BaseModel):
    distance_m: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=1)
    information_gain: float = Field(ge=0, le=1)
    model_score: float = Field(ge=0, le=1)


class FailurePolicy(BaseModel):
    pose_resamples: int = Field(default=1, ge=0, le=3)
    replans: int = Field(default=2, ge=0, le=5)
    on_exhausted: Literal["pause", "skip", "abort"] = "pause"


class ActionSpec(BaseModel):
    name: ActionName
    parameters: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    cost: CostEstimate
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)


class GroundedTask(BaseModel):
    task_id: str
    action: ActionName
    binding_index: int = Field(ge=0)
    node_id: str
    ifc_guid: str
    floor_id: str
    target_name: str
    target_kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    target_xy: tuple[float, float] | None = None
    retrieval_score: float = Field(ge=0, le=1)
    enabled: bool = True
    validation_status: Literal["certified", "rejected", "unlocalizable"]
    action_spec: ActionSpec


class ReasoningSummary(BaseModel):
    phase: str
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    certificate_status: str | None = None


class SubgraphNode(BaseModel):
    id: str
    label: str
    kind: str
    score: float = Field(default=0, ge=0, le=1)
    is_target: bool = False
    floor_id: str | None = None
    x: float | None = None
    y: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubgraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    on_evidence_path: bool = False


class Subgraph(BaseModel):
    id: str
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


class UsageSummary(BaseModel):
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    embedding_input_tokens: int = Field(default=0, ge=0)


class GroundingLimits(BaseModel):
    """Optional tighter limits; never expands the normal research call allowance."""

    model_config = ConfigDict(extra="forbid")
    max_llm_calls: int = Field(default=24, ge=4, le=48)
    input_token_budget: int = Field(default=60_000, ge=1_000, le=200_000)
    max_output_tokens: int = Field(default=2_048, ge=256, le=8_192)


class QueryIntent(BaseModel):
    kind: Literal["answer", "planning", "unknown"]
    reason: str


class PlanningScore(BaseModel):
    value: float | None = None
    kind: Literal["mean_retrieval_rank_score"] = "mean_retrieval_rank_score"
    calibrated: Literal[False] = False
    description: str = (
        "Mean selected-target retrieval rank score; not a probability of correctness or failure."
    )


class GroundingResult(BaseModel):
    status: RunStatus
    answer: str
    agent_response: str = ""
    response_status: Literal["evidence_supported", "unverified", "unavailable"] = "unavailable"
    query_type: Literal["answer", "planning", "unknown"] = "planning"
    query_type_reason: str = ""
    planning_score: PlanningScore = Field(default_factory=PlanningScore)
    closure_status: Literal["pass", "abstain", "failed"]
    closure_stop_reason: str
    tasks: list[GroundedTask] = Field(default_factory=list)
    reasoning: list[ReasoningSummary] = Field(default_factory=list)
    subgraph: Subgraph | None = None
    errors: list[str] = Field(default_factory=list)
    executable: bool = False
    graph_hash: str = ""
    usage: UsageSummary = Field(default_factory=UsageSummary)


class GroundingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: ReasoningModel = "gpt-4.1"
    api_key: SecretStr
    query: str = Field(min_length=3, max_length=2_000)
    limits: GroundingLimits | None = None

    @field_validator("api_key")
    @classmethod
    def nonblank_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("An OpenAI API key is required")
        return value

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        if len(value.strip()) < 3:
            raise ValueError("Query must contain at least three non-whitespace characters")
        return value  # Preserve the exact submitted wording for system.ask.


class RunEvent(BaseModel):
    event_id: int = Field(ge=1)
    timestamp: datetime
    phase: str
    progress: int = Field(ge=0, le=100)
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class GroundingRun(BaseModel):
    id: str
    owner: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    model_id: str
    ifc_asset_id: str = "ecore-fixed"
    gnn_runtime_id: str = "text-gnn-v5-seed43-epoch10"
    query: str
    result: GroundingResult | None = None
    events: list[RunEvent] = Field(default_factory=list)


class GroundingRunCreated(BaseModel):
    run_id: str
    status: RunStatus


class Point2D(BaseModel):
    x: float
    y: float


class MapPolygon(BaseModel):
    id: str
    kind: Literal["slab", "wall", "column", "space", "target"]
    points: list[Point2D] = Field(min_length=3)
    ifc_guid: str | None = None
    label: str | None = None


class FloorMap(BaseModel):
    id: str
    floor_id: str
    units: Literal["m"] = "m"
    origin: tuple[float, float]
    resolution_m: float = Field(default=0.1, gt=0)
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    polygons: list[MapPolygon]
    occupancy_rows: list[str]
    target_positions: dict[str, tuple[float, float]] = Field(default_factory=dict)
    target_position_sources: dict[str, str] = Field(default_factory=dict)
    transform: dict[str, float] = Field(default_factory=dict)
    subgraph_projection: list[MapPolygon] = Field(default_factory=list)
    source_ifc_sha256: str
    artifact_version: str = "floorplan-v1"


class RobotProfile(BaseModel):
    id: Literal["jackal", "husky", "robot-dog"]
    label: str
    radius_m: float = Field(gt=0)
    max_linear_mps: float = Field(gt=0)
    max_angular_rps: float = Field(gt=0)
    clearance_m: float = Field(ge=0)


class ExecutionTaskInput(BaseModel):
    task_id: str
    enabled: bool = True
    order: int = Field(ge=0)


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grounding_run_id: str
    ordered_tasks: list[ExecutionTaskInput] = Field(min_length=1)
    robot_profile_id: Literal["jackal", "husky", "robot-dog"]
    planner_id: Literal["astar", "dwa", "rrt_star"]
    initial_pose: Pose2DModel

    @model_validator(mode="after")
    def unique_tasks(self) -> ExecutionRequest:
        if len({item.task_id for item in self.ordered_tasks}) != len(self.ordered_tasks):
            raise ValueError("Each task may appear only once")
        if len({item.order for item in self.ordered_tasks}) != len(self.ordered_tasks):
            raise ValueError("Task order values must be unique")
        if not any(item.enabled for item in self.ordered_tasks):
            raise ValueError("Select at least one task")
        return self


class ExecutionFrame(BaseModel):
    t: float = Field(ge=0)
    pose: Pose2DModel
    task_id: str
    status: Literal["navigating", "dwelling", "completed", "failed", "paused"]


class ExecutionResult(BaseModel):
    status: RunStatus
    frames: list[ExecutionFrame]
    completed_task_ids: list[str]
    failed_task_id: str | None = None
    terminal_reason: str


class ExecutionRunCreated(BaseModel):
    execution_id: str
    status: RunStatus


class ExecutionControlRequest(BaseModel):
    command: Literal["retry", "skip", "abort"]


class ModelOption(BaseModel):
    id: str
    label: str
    compatibility: Literal["validated", "compatible", "unsupported"]
    reason: str
    recommended: bool = False


class ModelCatalogRequest(BaseModel):
    api_key: SecretStr


class AssetOption(BaseModel):
    id: str
    name: str
    kind: Literal["ifc", "gnn-runtime"]
    status: Literal["ready", "preprocessing", "quarantined", "incompatible"]
    default: bool = False
    grounding_available: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeBatchRequest(BaseModel):
    node_ids: list[str] = Field(min_length=1, max_length=200)


class PublicNodeMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node_id: str
    ifc_guid: str
    name: str
    ifc_class: str
    floor_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class CapabilityAsset(BaseModel):
    id: str
    name: str
    sha256: str


class Capabilities(BaseModel):
    ready: bool
    readiness_reason: str | None = None
    device: str
    ifc: CapabilityAsset
    gnn_runtime: CapabilityAsset
    graph_hash: str
    floor_count: int = Field(ge=0)
    allowed_models: list[str]
    embedding_model: str
