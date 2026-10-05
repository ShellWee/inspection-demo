import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import type { InspectionApi } from "./api";
import App from "./App";
import type { Capabilities, FloorMap, GroundedTask, GroundingInput, GroundingResult } from "./types";

const capabilities: Capabilities = {
  ready: true,
  readiness_reason: null,
  device: "NVIDIA L4 / CUDA 12.8",
  ifc: { id: "ecore-fixed", name: "Ecore IFC (fixed)", sha256: "a".repeat(64) },
  gnn_runtime: { id: "gnn-fixed", name: "Text-GNN v5 · seed43 · epoch10", sha256: "b".repeat(64) },
  graph_hash: "c".repeat(64),
  floor_count: 12,
  allowed_models: ["gpt-4.1", "gpt-5", "gpt-5.6-luna"],
  embedding_model: "text-embedding-3-small",
};

const task: GroundedTask = {
  task_id: "task-0-0",
  action: "Inspect",
  binding_index: 0,
  node_id: "ifc_hvac_terminal",
  ifc_guid: "hvac-terminal",
  floor_id: "LEVEL 3",
  target_name: "Supply air terminal",
  target_kind: "IfcFlowTerminal",
  metadata: { floor_map_id: "ecore-level-3-floorplan-v1", system: "HVAC" },
  target_xy: [7, 5],
  retrieval_score: 0.91,
  enabled: true,
  validation_status: "certified",
  action_spec: {
    name: "Inspect",
    parameters: { dwell_s: 5 },
    preconditions: ["closure certificate passed"],
    cost: { distance_m: 7, duration_s: 12, risk_score: 0.1, information_gain: 0.8, model_score: 0.91 },
    failure_policy: { pose_resamples: 1, replans: 2, on_exhausted: "pause" },
  },
};

const grounding: GroundingResult = {
  status: "completed",
  answer: "A certified target was found.",
  closure_status: "pass",
  closure_stop_reason: "all_bindings_certified",
  tasks: [task],
  reasoning: [{ phase: "hierarchy", summary: "The terminal is connected by typed HVAC evidence.", evidence_refs: [task.node_id], certificate_status: null }],
  subgraph: {
    id: "run-subgraph",
    nodes: [{ id: task.node_id, label: task.target_name, kind: task.target_kind, score: 0.91, is_target: true, floor_id: task.floor_id, x: 4, y: 4, metadata: task.metadata }],
    edges: [],
  },
  errors: [],
  executable: true,
  graph_hash: capabilities.graph_hash,
};

const floor: FloorMap = {
  id: "ecore-level-3-floorplan-v1",
  floor_id: "LEVEL 3",
  units: "m",
  origin: [0, 0],
  resolution_m: 0.1,
  width_m: 10,
  height_m: 8,
  polygons: [{ id: "slab", kind: "slab", points: [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 8 }], ifc_guid: null, label: null }],
  occupancy_rows: Array.from({ length: 80 }, () => ".".repeat(100)),
  target_positions: { [task.node_id]: task.target_xy! },
  transform: {},
  subgraph_projection: [],
  source_ifc_sha256: capabilities.ifc.sha256,
  artifact_version: "floorplan-v1",
};

class FakeApi implements InspectionApi {
  submitted: GroundingInput[] = [];
  async getCapabilities() { return capabilities; }
  async listModels() {
    return [
      { id: "gpt-4.1", label: "GPT-4.1", compatibility: "validated" as const, reason: "Validated", recommended: true },
      { id: "gpt-5", label: "GPT-5", compatibility: "compatible" as const, reason: "Compatible", recommended: false },
      { id: "gpt-5.6-luna", label: "GPT-5.6 Luna", compatibility: "compatible" as const, reason: "Compatible", recommended: false },
    ];
  }
  async runGrounding(input: GroundingInput, onEvent: (event: never) => void) {
    this.submitted.push(input);
    void onEvent;
    return { runId: "run-one", result: grounding };
  }
  async getNodes() { return [{ node_id: task.node_id, ifc_guid: task.ifc_guid, name: task.target_name, ifc_class: task.target_kind, floor_id: task.floor_id, properties: task.metadata }]; }
  async getFloor() { return floor; }
  async runExecution() {
    return { executionId: "execution-one", result: { status: "completed" as const, frames: [{ t: 0, pose: { x: 2, y: 2, yaw: 0 }, task_id: task.task_id, status: "navigating" as const }, { t: 0.1, pose: { x: 6.5, y: 5, yaw: 0 }, task_id: task.task_id, status: "completed" as const }], completed_task_ids: [task.task_id], failed_task_id: null, terminal_reason: "all_tasks_completed" } };
  }
}

describe("Inspection control desk", () => {
  test("keeps an abstained agent response visible and skips execution for answer queries", async () => {
    const user = userEvent.setup();
    const api = new FakeApi();
    vi.spyOn(api, "runGrounding").mockResolvedValue({ runId: "answer-one", result: {
      ...grounding, status: "abstained", closure_status: "abstain", executable: false, tasks: [],
      query_type: "answer", query_type_reason: "The user asks about hypothetical impact.",
      agent_response: "Supply paths are incomplete, so affected rooms cannot be confirmed.",
      response_status: "unverified",
      planning_score: { value: null, kind: "mean_retrieval_rank_score", calibrated: false, description: "Not a probability." },
    } });
    render(<App api={api} />);
    await screen.findByText("Workspace ready");
    await user.type(screen.getByLabelText("OpenAI API key"), "sk-test-ui");
    await user.click(screen.getByRole("button", { name: "Connect models" }));
    await user.type(screen.getByLabelText("Natural-language query"), "Which rooms might lose cooling?");
    await user.click(screen.getByRole("button", { name: "Run target grounding" }));
    expect(await screen.findByText("Supply paths are incomplete, so affected rooms cannot be confirmed.")).toBeInTheDocument();
    expect(screen.getByText("Answer only · no simulation required")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Configure execution" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /03.*Execute/ })).toBeDisabled();
    expect(screen.getByText(/Unverified model response/)).toBeInTheDocument();
  });

  test("shows a retrieval score without presenting it as probability", async () => {
    const user = userEvent.setup();
    const api = new FakeApi();
    vi.spyOn(api, "runGrounding").mockResolvedValue({ runId: "plan-one", result: {
      ...grounding, query_type: "planning", agent_response: "Inspect the supported terminal.",
      planning_score: { value: .025, kind: "mean_retrieval_rank_score", calibrated: false, description: "Not a probability of correctness." },
    } });
    render(<App api={api} />);
    await screen.findByText("Workspace ready");
    await user.type(screen.getByLabelText("OpenAI API key"), "sk-test-ui");
    await user.click(screen.getByRole("button", { name: "Connect models" }));
    await user.type(screen.getByLabelText("Natural-language query"), "Inspect a terminal.");
    await user.click(screen.getByRole("button", { name: "Run target grounding" }));
    expect(await screen.findByText(/Retrieval rank score: 0.0250/)).toBeInTheDocument();
    expect(screen.getByText(/Not a probability of correctness/)).toBeInTheDocument();
  });

  test("does not allow opening execution before a verified map exists", async () => {
    render(<App api={new FakeApi()} />);
    expect(screen.getByRole("button", { name: /03.*Execute/ })).toBeDisabled();
  });

  test("changing a connected API key invalidates its model catalog", async () => {
    const user = userEvent.setup();
    render(<App api={new FakeApi()} />);
    await user.type(screen.getByLabelText("OpenAI API key"), "sk-test-ui");
    await user.click(screen.getByRole("button", { name: "Connect models" }));
    await waitFor(() => expect(screen.getByLabelText("Reasoning model")).toBeEnabled());
    await user.type(screen.getByLabelText("OpenAI API key"), "-changed");
    expect(screen.getByLabelText("Reasoning model")).toBeDisabled();
  });
  test("shows immutable server-derived assets and an empty query", async () => {
    render(<App api={new FakeApi()} />);
    expect(screen.getByRole("heading", { name: "Inspection workspace" })).toBeInTheDocument();
    expect(await screen.findByText("12 floor plans available")).toBeInTheDocument();
    expect(screen.queryByText("Text-GNN v5 · seed43 · epoch10")).not.toBeInTheDocument();
    expect(screen.getByText("Workspace ready")).toBeInTheDocument();
    expect(screen.getByLabelText("Natural-language query")).toHaveValue("");
    expect(screen.queryByLabelText("Upload IFC model")).not.toBeInTheDocument();
  });

  test("forwards an arbitrary query and exposes all three models", async () => {
    const user = userEvent.setup();
    const api = new FakeApi();
    render(<App api={api} />);
    await screen.findByText("12 floor plans available");
    await user.type(screen.getByLabelText("OpenAI API key"), "sk-test-ui");
    await user.click(screen.getByRole("button", { name: "Connect models" }));
    const modelSelect = screen.getByLabelText("Reasoning model") as HTMLSelectElement;
    await waitFor(() => expect(modelSelect).toHaveValue("gpt-4.1"));
    expect(within(modelSelect).getAllByRole("option").map((item) => item.textContent)).toEqual(["GPT-4.1", "GPT-5", "GPT-5.6 Luna"]);
    const query = "Which air terminal should be inspected after an unexpected temperature rise?";
    await user.type(screen.getByLabelText("Natural-language query"), query);
    await user.click(screen.getByRole("button", { name: "Run target grounding" }));
    expect(await screen.findByText("Supply air terminal")).toBeInTheDocument();
    expect(api.submitted[0].query).toBe(query);
    expect(Object.keys(api.submitted[0]).sort()).toEqual(["api_key", "model_id", "query"]);
  });

  test("loads the grounded floor and completes the browser replay", async () => {
    const user = userEvent.setup();
    const api = new FakeApi();
    const getFloor = vi.spyOn(api, "getFloor");
    render(<App api={api} />);
    await screen.findByText("12 floor plans available");
    await user.type(screen.getByLabelText("OpenAI API key"), "sk-test-ui");
    await user.click(screen.getByRole("button", { name: "Connect models" }));
    await user.type(screen.getByLabelText("Natural-language query"), "Inspect a supply terminal.");
    await user.click(screen.getByRole("button", { name: "Run target grounding" }));
    await screen.findByText("Supply air terminal");
    await user.click(screen.getByRole("button", { name: "Configure execution" }));
    expect(getFloor).toHaveBeenCalledWith("ecore-level-3-floorplan-v1");
    expect(screen.getByRole("button", { name: "Start simulation" })).toBeDisabled();
    await user.clear(screen.getByLabelText("Initial x"));
    await user.type(screen.getByLabelText("Initial x"), "2");
    await user.clear(screen.getByLabelText("Initial y"));
    await user.type(screen.getByLabelText("Initial y"), "2");
    await user.click(screen.getByRole("button", { name: "Start simulation" }));
    await waitFor(() => expect(screen.getByText("Mission complete")).toBeInTheDocument());
  });
});
