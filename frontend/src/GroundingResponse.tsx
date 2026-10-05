import type { GroundingResult } from "./types";

export default function GroundingResponse({ result }: { result: GroundingResult }) {
  const score = result.planning_score;
  return <section className="agent-response" aria-label="Agent response">
    <div className="panel-heading"><div><p className="eyebrow">AGENT RESPONSE</p><h3>{result.query_type === "answer" ? "Answer only · no simulation required" : result.query_type === "unknown" ? "Request type unresolved · simulation disabled" : "Inspection task planning"}</h3></div><span className={`status-pill ${result.response_status === "evidence_supported" ? "success" : "warning"}`}>{result.response_status === "evidence_supported" ? "Target evidence verified" : "Unverified model response"}</span></div>
    <p className="intent-explanation">{result.query_type_reason}</p>
    <p className="agent-answer">{result.agent_response || result.answer || "No model response was produced."}</p>
    {result.closure_status !== "pass" && <p className="response-caveat">Evidence is incomplete. This response is retained for review; it does not authorize execution.</p>}
    <div className="response-score"><strong>Retrieval rank score: {score?.value == null ? "Unavailable" : score.value.toFixed(4)}</strong><p>{score?.description || "Not a probability of correctness. No calibrated planning confidence is available."}</p>{score?.value == null && <p>No selected targets with retrieval scores were returned.</p>}</div>
  </section>;
}
