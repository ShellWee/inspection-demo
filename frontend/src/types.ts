export type RunStatus = "queued" | "running" | "completed" | "abstained" | "failed" | "cancelled";
export type ActionName = "Navigate" | "Scan" | "Inspect";

export interface Capabilities {
  ready: boolean;
  readiness_reason: string | null;
  device: string;
  ifc: { id: string; name: string; sha256: string };
  gnn_runtime: { id: string; name: string; sha256: string };
  graph_hash: string;
  floor_count: number;
  allowed_models: string[];
  embedding_model: string;
}

export interface ModelOption {
  id: string;
  label: string;
  compatibility: "validated" | "compatible" | "unsupported";
  reason: string;
  recommended: boolean;
}

export interface RunEvent {
  event_id: number;
  timestamp: string;
  phase: string;
  progress: number;
  message: string;
  payload: Record<string, unknown>;
}

export interface ReasoningSummary {
  phase: string;
  summary: string;
  evidence_refs: string[];
  certificate_status: string | null;
}

export interface ActionSpec {
  name: ActionName;
  parameters: Record<string, unknown>;
  preconditions: string[];
  cost: {
    distance_m: number;
    duration_s: number;
    risk_score: number;
    information_gain: number;
    model_score: number;
  };
  failure_policy: { pose_resamples: number; replans: number; on_exhausted: string };
}

export interface GroundedTask {
  task_id: string;
  action: ActionName;
  binding_index: number;
  node_id: string;
  ifc_guid: string;
  floor_id: string;
  target_name: string;
  target_kind: string;
  metadata: Record<string, unknown>;
  target_xy: [number, number] | null;
  retrieval_score: number;
  enabled: boolean;
  validation_status: "certified" | "rejected" | "unlocalizable";
  action_spec: ActionSpec;
}

export interface SubgraphNode {
  id: string;
  label: string;
  kind: string;
  score: number;
  is_target: boolean;
  floor_id: string | null;
  x: number | null;
  y: number | null;
  metadata: Record<string, unknown>;
}

export interface SubgraphEdge {
  source: string;
  target: string;
  relation: string;
  on_evidence_path: boolean;
}

export interface GroundingResult {
  agent_response?: string;
  response_status?: "evidence_supported" | "unverified" | "unavailable";
  query_type?: "answer" | "planning" | "unknown";
  query_type_reason?: string;
  planning_score?: { value: number | null; kind: string; calibrated: false; description: string };
  usage?: { llm_calls: number; input_tokens: number; output_tokens: number; cached_input_tokens: number; embedding_input_tokens: number };
  status: RunStatus;
  answer: string;
  closure_status: "pass" | "abstain" | "failed";
  closure_stop_reason: string;
  tasks: GroundedTask[];
  reasoning: ReasoningSummary[];
  subgraph: { id: string; nodes: SubgraphNode[]; edges: SubgraphEdge[] } | null;
  errors: string[];
  executable: boolean;
  graph_hash: string;
}

export interface MapPolygon {
  id: string;
  kind: "slab" | "wall" | "column" | "space" | "target";
  points: { x: number; y: number }[];
  ifc_guid: string | null;
  label: string | null;
}

export interface FloorMap {
  id: string;
  floor_id: string;
  units: "m";
  origin: [number, number];
  resolution_m: number;
  width_m: number;
  height_m: number;
  polygons: MapPolygon[];
  occupancy_rows: string[];
  target_positions: Record<string, [number, number]>;
  transform: Record<string, number>;
  subgraph_projection: MapPolygon[];
  source_ifc_sha256: string;
  artifact_version: string;
}

export interface Pose2D { x: number; y: number; yaw: number }

export interface ExecutionFrame {
  t: number;
  pose: Pose2D;
  task_id: string;
  status: "navigating" | "dwelling" | "completed" | "failed" | "paused";
}

export interface ExecutionResult {
  status: RunStatus;
  frames: ExecutionFrame[];
  completed_task_ids: string[];
  failed_task_id: string | null;
  terminal_reason: string;
}

export interface GroundingInput {
  api_key: string;
  query: string;
  model_id: string;
}

export interface PublicNodeMetadata {
  node_id: string;
  ifc_guid: string;
  name: string;
  ifc_class: string;
  floor_id: string | null;
  properties: Record<string, unknown>;
}

export interface ExecutionInput {
  grounding_run_id: string;
  ordered_tasks: { task_id: string; enabled: boolean; order: number }[];
  robot_profile_id: "jackal" | "husky" | "robot-dog";
  planner_id: "astar" | "dwa" | "rrt_star";
  initial_pose: Pose2D;
}
