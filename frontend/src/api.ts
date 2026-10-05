import type {
  Capabilities,
  ExecutionInput,
  ExecutionResult,
  FloorMap,
  GroundingInput,
  GroundingResult,
  ModelOption,
  PublicNodeMetadata,
  RunEvent,
} from "./types";

export interface GroundingRunResponse { runId: string; result: GroundingResult }
export interface ExecutionRunResponse { executionId: string; result: ExecutionResult }

export interface InspectionApi {
  getCapabilities(): Promise<Capabilities>;
  listModels(apiKey: string): Promise<ModelOption[]>;
  runGrounding(input: GroundingInput, onEvent: (event: RunEvent) => void): Promise<GroundingRunResponse>;
  getNodes(runId: string, nodeIds: string[]): Promise<PublicNodeMetadata[]>;
  getFloor(floorMapId: string): Promise<FloorMap>;
  runExecution(input: ExecutionInput, onEvent: (event: RunEvent) => void): Promise<ExecutionRunResponse>;
}

const terminal = new Set(["completed", "abstained", "failed", "cancelled"]);

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  const response = await fetch(url, { credentials: "same-origin", ...init, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = Array.isArray(body.detail) ? body.detail.map((item: { msg?: string }) => item.msg ?? "Invalid input").join("; ") : body.detail;
    throw new Error(response.status === 409 ? `Workspace busy or not ready: ${detail}` : detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function waitForTerminalEvent(
  url: string,
  onEvent: (event: RunEvent) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const source = new EventSource(url, { withCredentials: true });
    let lastEventId = 0;
    const timeout = window.setTimeout(() => {
      source.close();
      reject(new Error("The run exceeded the 10 minute hard timeout."));
    }, 610_000);
    source.addEventListener("progress", (message) => {
      let event: RunEvent;
      try { event = JSON.parse((message as MessageEvent).data) as RunEvent; }
      catch { window.clearTimeout(timeout); source.close(); reject(new Error("Received an invalid progress update. Please reload the workspace.")); return; }
      if (event.event_id <= lastEventId) return;
      lastEventId = event.event_id;
      onEvent(event);
      if (terminal.has(event.phase)) {
        window.clearTimeout(timeout);
        source.close();
        resolve();
      }
    });
    source.onerror = () => {
      // EventSource reconnects automatically and sends Last-Event-ID.
      // The hard timeout handles a permanently unavailable stream.
    };
  });
}

export class HttpApiClient implements InspectionApi {
  async getCapabilities(): Promise<Capabilities> {
    return json("/api/v1/capabilities");
  }

  async listModels(apiKey: string): Promise<ModelOption[]> {
    return json("/api/v1/openai/models", {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey }),
    });
  }

  async runGrounding(
    input: GroundingInput,
    onEvent: (event: RunEvent) => void,
  ): Promise<GroundingRunResponse> {
    const created = await json<{ run_id: string }>("/api/v1/grounding-runs", {
      method: "POST",
      body: JSON.stringify(input),
    });
    await waitForTerminalEvent(`/api/v1/grounding-runs/${created.run_id}/events`, onEvent);
    const run = await json<{ status: string; result: GroundingResult | null }>(
      `/api/v1/grounding-runs/${created.run_id}`,
    );
    if (!run.result) throw new Error("Grounding ended without a result.");
    return { runId: created.run_id, result: run.result };
  }

  async getNodes(runId: string, nodeIds: string[]): Promise<PublicNodeMetadata[]> {
    return json(`/api/v1/grounding-runs/${encodeURIComponent(runId)}/nodes/batch`, {
      method: "POST",
      body: JSON.stringify({ node_ids: nodeIds }),
    });
  }

  async getFloor(floorMapId: string): Promise<FloorMap> {
    return json(`/api/v1/floors/${encodeURIComponent(floorMapId)}`);
  }

  async runExecution(
    input: ExecutionInput,
    onEvent: (event: RunEvent) => void,
  ): Promise<ExecutionRunResponse> {
    const created = await json<{ execution_id: string }>("/api/v1/executions", {
      method: "POST",
      body: JSON.stringify(input),
    });
    await waitForTerminalEvent(`/api/v1/executions/${created.execution_id}/events`, onEvent);
    const result = await json<ExecutionResult>(
      `/api/v1/executions/${created.execution_id}/replay`,
    );
    return { executionId: created.execution_id, result };
  }
}
