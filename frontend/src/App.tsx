import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HttpApiClient, type InspectionApi } from "./api";
import FloorPlanCanvas from "./FloorPlanCanvas";
import SubgraphView from "./SubgraphView";
import GroundingResponse from "./GroundingResponse";
import { moveTask, suggestedTaskOrder } from "./taskOrdering";
import { interpolateYaw, isPoseFree, robotClearance } from "./navigationPreview";
import type { Capabilities, ExecutionResult, FloorMap, GroundedTask, GroundingResult, ModelOption, Pose2D, PublicNodeMetadata, RunEvent } from "./types";
import "./styles.css";

interface Props { api?: InspectionApi }
type Step = "setup" | "grounding" | "execution";
const defaultApi = new HttpApiClient();
const messageOf = (error: unknown) => error instanceof Error ? error.message : "Something went wrong. Please try again.";
const phaseLabels: Record<string, string> = { queued: "Starting your request", asset_sync: "Preparing building data", query_planning: "Interpreting your request", gnn_retrieval: "Finding related elements", hierarchy: "Reviewing building relationships", closure_validation: "Checking evidence", completed: "Review complete", abstained: "Insufficient evidence", failed: "Request stopped", cancelled: "Cancelled" };
const steps = [{ id: "setup", label: "Setup", description: "Describe your task" }, { id: "grounding", label: "Ground & verify", description: "Review the evidence" }, { id: "execution", label: "Execute", description: "Simulate the mission" }] as const;

function blockedReason(result: GroundingResult): string {
  if (result.query_type === "answer") return "This is an informational answer. No robot mission was requested.";
  if (result.query_type === "unknown") return "The request type could not be determined. Clarify whether you want an answer or an inspection mission.";
  if (result.status === "failed") return "The request could not finish. Review the diagnostics or try a simpler request.";
  if (result.closure_status !== "pass") return "The building evidence was not sufficient to certify this request. Try specifying a location, system, or inspection action.";
  if (result.tasks.some(task => task.validation_status === "unlocalizable")) return "Some targets have no verified position on the floor plan. They can be reviewed, but cannot be simulated.";
  if (new Set(result.tasks.map(task => task.floor_id)).size > 1) return "These targets span multiple floors. Split your request into one floor at a time to simulate a mission.";
  return "No executable inspection targets were found for this request.";
}

export default function App({ api = defaultApi }: Props) {
  const [step, setStep] = useState<Step>("setup");
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [apiKey, setApiKey] = useState("");
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [modelId, setModelId] = useState("gpt-4.1");
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [groundingRunId, setGroundingRunId] = useState<string | null>(null);
  const [grounding, setGrounding] = useState<GroundingResult | null>(null);
  const [tasks, setTasks] = useState<GroundedTask[]>([]);
  const [highlightedNode, setHighlightedNode] = useState<string | null>(null);
  const [nodeMetadata, setNodeMetadata] = useState<PublicNodeMetadata | null>(null);
  const highlightVersion = useRef(0);
  const [floor, setFloor] = useState<FloorMap | null>(null);
  const [pose, setPose] = useState<Pose2D>({ x: 0, y: 0, yaw: 0 });
  const [poseChosen, setPoseChosen] = useState(false);
  const [robot, setRobot] = useState<"jackal" | "husky" | "robot-dog">("jackal");
  const [planner, setPlanner] = useState<"astar" | "dwa" | "rrt_star">("dwa");
  const [execution, setExecution] = useState<ExecutionResult | null>(null);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [draggedTask, setDraggedTask] = useState<number | null>(null);
  const [showAllTargets, setShowAllTargets] = useState(false);
  const [busy, setBusy] = useState<"connect" | "grounding" | "floor" | "execution" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [readinessError, setReadinessError] = useState(false);
  // Historical replays may contain accumulated 0.2-second floating-point drift.
  const lastTime = Math.round((execution?.frames.at(-1)?.t ?? 0) * 1_000_000) / 1_000_000;

  const checkReadiness = useCallback(() => {
    setReadinessError(false);
    void api.getCapabilities().then(setCapabilities).catch(() => setReadinessError(true));
  }, [api]);
  useEffect(checkReadiness, [checkReadiness]);
  useEffect(() => {
    if (!playing || !execution?.frames.length) return;
    let request: number;
    let previous: number | undefined;
    const tick = (now: number) => {
      if (previous !== undefined) {
        const elapsed = Math.min(0.1, (now - previous) / 1000);
        setPlaybackTime(time => Math.min(lastTime, time + elapsed * speed));
      }
      previous = now;
      request = requestAnimationFrame(tick);
    };
    request = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(request);
  }, [playing, execution, lastTime, speed]);
  useEffect(() => { if (playbackTime >= lastTime) setPlaying(false); }, [playbackTime, lastTime]);

  const connectModels = async () => {
    setBusy("connect"); setError(null); setModels([]);
    try {
      const available = (await api.listModels(apiKey)).filter(item => item.compatibility !== "unsupported");
      if (!available.length) throw new Error("This key cannot access any supported model. Check your OpenAI project permissions.");
      setModels(available); setModelId((available.find(item => item.recommended) ?? available[0]).id);
    } catch (reason) { setError(messageOf(reason)); }
    finally { setBusy(null); }
  };
  const runGrounding = async () => {
    setBusy("grounding"); setError(null); setEvents([]); setGrounding(null); setGroundingRunId(null);
    setTasks([]); setFloor(null); setExecution(null); setPlaying(false); setPoseChosen(false);
    setShowAllTargets(false);
    highlightVersion.current++; setHighlightedNode(null); setNodeMetadata(null);
    setSubmittedQuery(query); setStep("grounding");
    try {
      const submittedKey = apiKey;
      setApiKey(""); setModels([]);
      const response = await api.runGrounding({ api_key: submittedKey, query, model_id: modelId }, event => setEvents(current => [...current, event]));
      setGroundingRunId(response.runId); setGrounding(response.result); setTasks(response.result.tasks);
    } catch (reason) { setError(messageOf(reason)); }
    finally { setBusy(null); }
  };
  const configureExecution = async () => {
    if (!grounding?.executable || grounding.query_type === "answer" || grounding.query_type === "unknown" || busy) return;
    if (floor) { setStep("execution"); return; }
    setBusy("floor"); setError(null);
    try {
      const floorMapId = tasks.find(task => task.enabled)?.metadata.floor_map_id;
      if (typeof floorMapId !== "string") throw new Error("This target has no verified floor plan.");
      setFloor(await api.getFloor(floorMapId)); setStep("execution");
    } catch (reason) { setError(messageOf(reason)); }
    finally { setBusy(null); }
  };
  const poseValid = !!floor && poseChosen && isPoseFree(floor, pose, robotClearance[robot]);
  const startExecution = async () => {
    if (!groundingRunId || !poseValid) return;
    setBusy("execution"); setError(null); setExecution(null); setPlaying(false); setPlaybackTime(0);
    try {
      const response = await api.runExecution({ grounding_run_id: groundingRunId, ordered_tasks: tasks.map((task, order) => ({ task_id: task.task_id, enabled: task.enabled, order })), robot_profile_id: robot, planner_id: planner, initial_pose: pose }, () => {});
      setExecution(response.result);
      if (response.result.status === "completed") setPlaying(true);
      else setError(`Simulation stopped: ${response.result.terminal_reason.replaceAll("_", " ")}. Choose another start position or planner and try again.`);
    } catch (reason) { setError(messageOf(reason)); }
    finally { setBusy(null); }
  };
  const onHighlight = useCallback((nodeId: string | null) => {
    const version = ++highlightVersion.current;
    setHighlightedNode(nodeId); setNodeMetadata(null);
    if (nodeId && groundingRunId) void api.getNodes(groundingRunId, [nodeId]).then(items => {
      if (highlightVersion.current === version) setNodeMetadata(items[0] ?? null);
    }).catch(() => { /* Optional metadata must not interrupt the run. */ });
  }, [api, groundingRunId]);
  const choosePose = (value: Pose2D) => { setPose(value); setPoseChosen(true); setExecution(null); setPlaying(false); };
  const resetReplay = () => { setExecution(null); setPlaying(false); };
  const activeFrame = useMemo(() => {
    if (!execution?.frames.length) return -1;
    let low = 0, high = execution.frames.length - 1;
    while (low < high) { const mid = Math.ceil((low + high) / 2); if (execution.frames[mid].t <= playbackTime) low = mid; else high = mid - 1; }
    return low;
  }, [execution, playbackTime]);
  const currentFrame = useMemo(() => {
    if (!execution?.frames.length) return undefined;
    const start = execution.frames[Math.max(0, activeFrame)];
    const end = execution.frames[Math.min(activeFrame + 1, execution.frames.length - 1)];
    const ratio = Math.max(0, Math.min(1, (playbackTime - start.t) / Math.max(0.0001, end.t - start.t)));
    return { ...start, pose: { x: start.pose.x + (end.pose.x - start.pose.x) * ratio, y: start.pose.y + (end.pose.y - start.pose.y) * ratio, yaw: interpolateYaw(start.pose.yaw, end.pose.yaw, ratio) } };
  }, [activeFrame, execution, playbackTime]);
  const trajectory = useMemo(() => execution?.frames.slice(0, activeFrame + 1), [execution, activeFrame]);
  const missionComplete = execution?.status === "completed" && playbackTime >= lastTime;
  const progress = grounding ? 100 : events.at(-1)?.progress ?? 0;
  const visiblePhases = events.filter((event, index) => events.findIndex(other => other.phase === event.phase) === index);

  return <main className="app-shell">
    <header className="topbar"><a className="brand" href="#" onClick={event => { event.preventDefault(); if (!busy) setStep("setup"); }} aria-label="Inspection workspace home"><span className="brand-mark" aria-hidden="true">i<span /></span><div><h1>Inspection workspace</h1><p>Ecore building · Robot task planning</p></div></a><span className={`system-state ${capabilities?.ready ? "ready" : "pending"}`}><i />{capabilities?.ready ? "Workspace ready" : readinessError ? "Workspace unavailable" : "Preparing workspace"}</span></header>
    <nav className="step-rail" aria-label="Workflow stages">{steps.map((item, index) => <button key={item.id} aria-current={step === item.id ? "step" : undefined} className={step === item.id ? "active" : ""} disabled={!!busy || (item.id === "grounding" && !grounding) || (item.id === "execution" && (!grounding?.executable || grounding.query_type === "answer" || grounding.query_type === "unknown"))} onClick={() => item.id === "execution" ? void configureExecution() : setStep(item.id)}><span className="step-number">{String(index + 1).padStart(2, "0")}</span><span><strong>{item.label}</strong><small>{item.description}</small></span><span className="step-arrow" aria-hidden="true">→</span></button>)}</nav>
    {error && <div className="error-banner" role="alert"><div><strong>Unable to complete this step</strong><p>{error}</p></div><button className="text-button" onClick={() => setError(null)} aria-label="Dismiss error">×</button></div>}
    {step === "setup" && <section className="setup-grid">
      <aside className="brief-panel"><p className="eyebrow">BUILDING INTELLIGENCE</p><h2>From a question<br />to an inspection.</h2><p className="intro">Find the right building elements, review the supporting evidence, and plan a robot’s next move.</p><div className="building-diagram" aria-hidden="true"><div className="building-isometric"><span /><span /><span /><span /><span /></div><div className="diagram-caption"><span>CONNECTED BUILDING</span><strong>Ecore</strong></div></div><div className="scope-note"><span className="scope-icon" aria-hidden="true">↗</span><p>Explore inspection and building-system questions. Simulation is available for verified targets on a single floor.</p></div></aside>
      <div className="input-panel"><div className="section-heading"><div><p className="eyebrow">NEW REQUEST</p><h2>What would you like to investigate?</h2></div></div><div className="building-context"><span className="building-icon" aria-hidden="true">▥</span><div><strong>Ecore building</strong><span>{capabilities ? `${capabilities.floor_count} floor plans available` : "Checking building data…"}</span></div><span className="context-tag">Fixed workspace</span></div>
        {!capabilities?.ready && <div className="notice" role="status">{readinessError ? "The workspace is not ready yet. It may still be starting." : "Verifying the building data. This can take a few minutes after a restart."}<button className="text-button" onClick={checkReadiness}>Check again</button></div>}
        <div className="key-line"><label>OpenAI API key<input type="password" autoComplete="off" spellCheck={false} value={apiKey} onChange={event => { setApiKey(event.target.value); setModels([]); }} disabled={!!busy} placeholder="Enter your API key" /></label><button className="secondary" disabled={!apiKey.trim() || !!busy} onClick={connectModels}>{busy === "connect" ? "Connecting…" : models.length ? "Reconnect" : "Connect models"}</button></div><p className="field-hint">Your key is used for this request only and is never saved. OpenAI API usage is billed to your account.</p>
        <label>Reasoning model<select value={modelId} onChange={event => setModelId(event.target.value)} disabled={!models.length || !!busy}>{!models.length && <option value={modelId}>Connect your key to choose a model</option>}{models.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>{models.length > 0 && <p className="connected-message" role="status">✓ Connected · {models.length} available models</p>}
        <label className="query-label">Natural-language query<textarea value={query} onChange={event => setQuery(event.target.value)} rows={5} minLength={3} maxLength={2000} placeholder="Describe a location, a system, or a problem you want to investigate…" /></label><div className="query-caption"><span>Write your request in English.</span><span>{query.length.toLocaleString()} / 2,000</span></div><div className="form-footer"><span>Results depend on available building evidence.</span><button className="primary" aria-label="Run target grounding" disabled={!apiKey.trim() || !models.length || query.trim().length < 3 || !!busy || !capabilities?.ready} onClick={runGrounding}>Find inspection targets <span aria-hidden="true">→</span></button></div>
      </div>
    </section>}
    {step === "grounding" && <section className="grounding-workspace">
      {grounding && <GroundingResponse result={grounding} />}
      <div className="request-heading"><div><p className="eyebrow">YOUR REQUEST</p><h2>{submittedQuery}</h2></div><span className={`status-pill ${grounding?.closure_status === "pass" ? "success" : grounding ? "warning" : "pending"}`}>{grounding?.closure_status === "pass" ? "Evidence verified" : grounding ? "Needs review" : "Reviewing evidence…"}</span></div>
      <div className="grounding-layout"><div className="evidence-stage"><div className="panel-heading"><div><h3>Building evidence</h3><p>Explore the retrieved elements and their connections.</p></div><div className="graph-legend"><i />Target<span /><i />Related element</div></div>{grounding?.subgraph ? <SubgraphView subgraph={grounding.subgraph} highlightedNodeId={highlightedNode} onHighlight={onHighlight} /> : <div className="graph-empty">{busy === "grounding" ? <><span className="loader" /><h3>Following the building evidence</h3><p>This may take a few minutes. You can review the progress below.</p></> : <><span className="empty-symbol">⌕</span><h3>No evidence graph available</h3><p>Try a more specific building or inspection request.</p></>}</div>}<div className="summary-panel"><p className="eyebrow">INSPECTION SUMMARY</p>{grounding ? <p>{grounding.reasoning.find(item => item.phase === "hierarchy")?.summary || (grounding.executable ? "The listed targets are supported by building evidence and ready for simulation." : blockedReason(grounding))}</p> : <p>The evidence summary will appear when the review is complete.</p>}</div></div>
      <aside className="binding-panel"><div className="panel-heading"><h3>Inspection targets</h3><span className="count-badge">{tasks.length}</span></div><div className="target-list">{tasks.slice(0, showAllTargets ? tasks.length : 10).map((task, index) => <article key={task.task_id} tabIndex={0} className={`target-card ${highlightedNode === task.node_id ? "highlighted" : ""}`} onFocus={() => onHighlight(task.node_id)} onBlur={() => onHighlight(null)} onMouseEnter={() => onHighlight(task.node_id)} onMouseLeave={() => onHighlight(null)}><div className="target-topline"><span className="action-tag">{task.action}</span><span className="target-index">{String(index + 1).padStart(2, "0")}</span></div><h3>{task.target_name}</h3><p className="target-score">Retrieval score {task.retrieval_score.toFixed(4)}</p><p className="target-floor">{task.floor_id === "unlocalized" ? "Floor not located" : task.floor_id}</p><span className={`target-state ${task.validation_status === "certified" ? "success" : "warning"}`}>{task.validation_status === "certified" ? "✓ Located on floor plan" : "Position unavailable"}</span></article>)}{!tasks.length && <p className="empty-targets">{busy === "grounding" ? "Targets will appear here after verification." : "No certified targets to display."}</p>}{tasks.length > 10 && <button className="text-button full" onClick={() => setShowAllTargets(value => !value)}>{showAllTargets ? "Show first 10 targets" : `Show all ${tasks.length} targets`}</button>}</div>
        {nodeMetadata && <div className="node-metadata" role="status"><strong>{nodeMetadata.name}</strong><span>{nodeMetadata.ifc_class} · {nodeMetadata.floor_id ?? "Floor unknown"}</span><dl>{Object.entries(nodeMetadata.properties).filter(([key, value]) => !/hash|sha|score|navigation|floor_map/.test(key) && typeof value !== "object").slice(0, 4).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(value)}</dd></div>)}</dl></div>}{grounding && grounding.query_type !== "answer" && !grounding.executable && <div className="notice warning"><strong>Simulation unavailable</strong><p>{blockedReason(grounding)}</p></div>}{grounding?.query_type !== "answer" && <button aria-label="Configure execution" className="primary full" disabled={!grounding?.executable || grounding.query_type === "unknown" || !!busy} onClick={configureExecution}>{busy === "floor" ? "Loading floor plan…" : "Configure simulation"}<span aria-hidden="true">→</span></button>}{!busy && <button className="text-button full" onClick={() => setStep("setup")}>Edit request</button>}
      </aside></div>
      <details className="details-panel" open={busy === "grounding" ? true : undefined}><summary><span>Run progress & technical details</span><span>{progress}%</span></summary><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><ol className="phase-list">{visiblePhases.map(event => <li key={event.event_id}><span className="phase-dot" /><strong>{phaseLabels[event.phase] ?? event.phase}</strong></li>)}</ol>{grounding && <div className="technical-details"><p>Validation: {grounding.closure_status} · {grounding.closure_stop_reason}</p>{grounding.errors.map((item, i) => <p key={i}>{item}</p>)}{grounding.reasoning.filter(item => item.phase !== "hierarchy").map(item => <p key={item.phase}>{item.summary}</p>)}{grounding.usage && <p>API usage: {grounding.usage.llm_calls} calls · {grounding.usage.input_tokens.toLocaleString()} input tokens · {grounding.usage.output_tokens.toLocaleString()} output tokens</p>}</div>}</details>
    </section>}
    {step === "execution" && floor && <section className="execution-layout"><aside className="mission-panel"><div className="panel-heading"><div><p className="eyebrow">MISSION SETUP</p><h2>Plan your inspection</h2></div></div><div className="queue-heading"><span>{tasks.filter(task => task.enabled).length} task{tasks.filter(task => task.enabled).length === 1 ? "" : "s"} selected</span><button className="text-button" disabled={!!busy} onClick={() => { setTasks(current => suggestedTaskOrder(current)); resetReplay(); }}>Suggest order</button></div><div className="task-list">{tasks.map((task, index) => <article key={task.task_id} className={!task.enabled ? "disabled-task" : ""} draggable={!busy} onDragStart={() => setDraggedTask(index)} onDragOver={event => event.preventDefault()} onDragEnd={() => setDraggedTask(null)} onDrop={() => { if (draggedTask !== null && !busy) { setTasks(current => moveTask(current, draggedTask, index)); resetReplay(); } setDraggedTask(null); }}><span className="drag-handle" aria-hidden="true">⠿</span><div className="task-text"><strong>{task.action}</strong><p>{task.target_name}</p><div className="move-buttons"><button disabled={!!busy || index === 0} aria-label={`Move ${task.target_name} up`} onClick={() => { setTasks(current => moveTask(current, index, index - 1)); resetReplay(); }}>↑</button><button disabled={!!busy || index === tasks.length - 1} aria-label={`Move ${task.target_name} down`} onClick={() => { setTasks(current => moveTask(current, index, index + 1)); resetReplay(); }}>↓</button></div></div><input aria-label={`Enable ${task.target_name}`} type="checkbox" disabled={!!busy} checked={task.enabled} onChange={() => { setTasks(current => current.map(item => item.task_id === task.task_id ? { ...item, enabled: !item.enabled } : item)); resetReplay(); }} /></article>)}</div>
      <label>Robot profile<select value={robot} disabled={!!busy} onChange={event => { setRobot(event.target.value as typeof robot); resetReplay(); }}><option value="jackal">Clearpath Jackal</option><option value="husky">Clearpath Husky</option><option value="robot-dog">Robot dog</option></select></label><label>Navigation planner<select value={planner} disabled={!!busy} onChange={event => { setPlanner(event.target.value as typeof planner); resetReplay(); }}><option value="dwa">DWA</option><option value="astar">A*</option><option value="rrt_star">RRT*</option></select></label>
      <fieldset className="pose-fields" disabled={!!busy}><legend>Starting pose</legend><p>Click and drag on the map, or enter coordinates.</p><div>{(["x", "y", "yaw"] as const).map(axis => <label key={axis}>{axis === "yaw" ? "Heading (°)" : `${axis.toUpperCase()} (m)`}<input aria-label={`Initial ${axis}`} type="number" step={axis === "yaw" ? 1 : 0.1} value={Number((axis === "yaw" ? pose.yaw * 180 / Math.PI : pose[axis]).toFixed(2))} onChange={event => choosePose({ ...pose, [axis]: axis === "yaw" ? Number(event.target.value) * Math.PI / 180 : Number(event.target.value) })} /></label>)}</div></fieldset><p className={`pose-validation ${poseValid ? "success" : poseChosen ? "warning" : ""}`} role="status">{poseValid ? "✓ Start position is clear" : poseChosen ? "Choose free space with enough room for the robot." : "Set a start position to continue."}</p><button className="primary full" aria-label="Start simulation" disabled={!!busy || !poseValid || !tasks.some(task => task.enabled)} onClick={startExecution}>{busy === "execution" ? "Planning simulation…" : "Start simulation"}<span aria-hidden="true">▷</span></button><p className="simulation-disclaimer">2D simulation only. No physical robot is connected.</p>
    </aside><div className="map-stage"><div className="panel-heading"><div><p className="eyebrow">FLOOR PLAN</p><h2>{floor.floor_id}</h2></div><span className={`status-pill ${missionComplete ? "success" : "pending"}`}>{missionComplete ? "Mission complete" : busy === "execution" ? "Planning route…" : execution ? "Simulation replay" : "Set starting pose"}</span></div><FloorPlanCanvas floor={floor} tasks={tasks.filter(task => task.enabled)} pose={poseChosen ? pose : undefined} onPoseChange={!busy ? choosePose : undefined} frame={currentFrame} trajectory={trajectory} highlightedNodeId={highlightedNode} robotRadius={robotClearance[robot]} poseValid={poseValid} /><div className="map-legend"><span><i className="legend-wall" />Structure</span><span><i className="legend-target" />Inspection target</span><span><i className="legend-robot" />Robot</span><span><i className="legend-route" />Route</span></div>{execution?.frames.length ? <div className="replay-controls"><button className="secondary" aria-label={playing ? "Pause replay" : "Play replay"} onClick={() => { if (playbackTime >= lastTime) setPlaybackTime(0); setPlaying(!playing); }}>{playing ? "Pause" : "Play"}</button><button className="text-button" onClick={() => { setPlaybackTime(0); setPlaying(true); }}>Restart</button><input aria-label="Replay position" type="range" min={0} max={lastTime} step={0.1} value={playbackTime} onChange={event => { setPlaying(false); setPlaybackTime(Number(event.target.value)); }} /><span>{playbackTime.toFixed(0)} / {lastTime.toFixed(0)} s</span><select aria-label="Replay speed" value={speed} onChange={event => setSpeed(Number(event.target.value))}><option value={1}>1×</option><option value={2}>2×</option><option value={5}>5×</option></select></div> : <div className="map-help"><strong>Place the robot in open space.</strong><span>Drag in the direction it should face. Walls and columns are treated as obstacles.</span></div>}</div></section>}
    <footer className="workspace-footer"><span>Ecore inspection workspace</span><span>Research demonstration · Verify results before real-world use</span></footer>
  </main>;
}
