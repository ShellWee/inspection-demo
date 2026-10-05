# What-if answering and planning — 2026-09-05

Changes are confined to `inspection-demo-hf`. No model weights, IFC, frozen graph,
floor artifacts or original research working directories were changed. This is
a local update; the running Hugging Face Space has not been redeployed.

## Behavior

- A small request-scoped model call classifies the original query as `answer`,
  `planning`, or `unknown`, with a short reason. Hypothetical shutdowns and
  recommendations about what should be inspected are answers; explicit generation
  of inspection tasks is planning. No asset IDs or acceptance questions occur in
  production classification rules. The original question still reaches `system.ask`
  unchanged. The classifier uses the selected model and its usage is included in
  the run total. Unknown intent cannot execute.
- Internal retrieval action bindings are not execution authorization. Answer-only
  results contain no robot tasks and cannot enter stage 3, including through the API.
- `agent_response` preserves the reviewer's answer summary, falling back to the
  initial summary, reasoning conclusion, or backend answer. `answer` still retains
  the original backend result. Abstained explanations are visibly unverified.
  This exposes conclusions, not hidden chain-of-thought.
- `planning_score` is the mean selected-entity retrieval rank score (RRF in the
  current closure runtime). It is **not** calibrated confidence, a percentage,
  likelihood of a fault, or probability of correctness. No selected scored entities
  means null/Unavailable, not a fabricated zero. Target cards show individual
  retrieval scores. The first ten cards are shown with an explicit expand control;
  this does not remove targets or change mission ordering.

## Root causes and corrections

1. The UI did not render the full answer, and the bridge ignored the v3 provider's
   `answer_summary`. A valid explanation could disappear behind generic abstention.
2. The parser synthesizes internal Inspect bindings for impact retrieval. These
   were previously exposed as a mission even for informational questions. The
   classification/gating layer now separates these concepts.
3. Semantic plan merging reintroduced a source equipment's room as a downstream
   target constraint. `room_names`, like room and storey, is now owned by the
   graph-grounded fallback plan.
4. A GUID explicitly included in the pump question was echoed into `scope_terms`,
   where identifiers are prohibited. The closure contract compiler now receives
   an identifier-masked question view with stable references. Exact IDs remain in
   the original query and sealed graph references. The identifier validator is not
   weakened and no generated target IDs are trusted.
5. All-matching reconciliation retained initial groups even when the reviewer
   rejected them all. This yielded a pass with 39 targets while the answer said no
   impact was established. Retained groups that the reviewer removed now create
   `provider_selection_disagreement` and abstain. Both proposals remain available
   in diagnostics. This is a conservative change to the HF source snapshot's
   closure reconciliation, so it may increase abstentions on disputed sets.

## Real GPT-4.1 checks

Six journaled attempts, including baseline/debug runs, used at most 24 reported
LLM calls each, a 60,000 input-token phase budget, and a 2,048-token output ceiling
per research call. Intent classification has a smaller 384-token output ceiling.
Input budgeting is checked at phase boundaries, not an account-level dollar cap.
No paid retries were used by the test runner. An early test loaded the installed
ToG copy during plugin validation; final runs explicitly assert the vendored
source import path, matching the research child process.

After the final reconciliation correction, the three saved real provider responses
were re-adjudicated locally without new model calls. This keeps the inputs identical
and directly verifies the conflict fix. All displayed target IDs exist in the
frozen display graph.

| User case | Type | Final result on saved real responses |
| --- | --- | --- |
| Room 319 overheating, generate tasks | planning | pass; 15 inspection targets; executable; mean rank score 0.01266385 |
| Blower coil shutdown impact | answer | abstain: no supported same-system endpoint evidence in the retrieved ledger; explanation retained |
| Pump shutdown impact | answer | abstain: initial selection and reviewer disagree; no confirmed impact claim or robot mission |

Passing target selection is not evidence of an actual fault. The blower result
describes missing evidence in the retrieved ledger, not proof that the full IFC
contains no relevant relationship. Neither what-if case establishes a complete,
causally verified set of impacted rooms. System membership alone does not establish
operational flow direction or shutdown consequences.

Total estimated OpenAI cost for all six attempts: **US$0.314161**, using reported
input, cached-input, output and embedding tokens. Pricing used: GPT-4.1
$2/$0.50/$8 per million input/cached/output tokens, embeddings $0.02 per million.
This is a usage estimate, not an invoice. Source:
[GPT-4.1 pricing](https://developers.openai.com/api/docs/models/gpt-4.1).

Reports and screenshots: `.runtime-host/whatif-tests/summary.json`,
`final-*-result.json`, and `*-ui.png`. The scripts/results are not copied by the
production Dockerfile. No keys are saved in them.

## Verification and usage

- Backend: 64 tests passed, including response retention, query scope, identifier
  masking, disagreement abstention, and API execution gating.
- Frontend: 10 tests passed; TypeScript and Vite production build passed.
- Release leakage checks passed.
- Headless Chromium rendered all three recorded real results: response visibility,
  answer-only stage gating, scores, 10/15-card expansion, no console errors or
  horizontal overflow. Browser replay does not make paid requests.
- The frozen GNN runtime emits an existing NumPy invalid-cast warning during ranking.
  This task does not modify that hash-pinned runtime; GPU parity was not tested.

Restart an existing local backend to load the new response schema, then refresh:

```powershell
cd C:\Users\Justin\Documents\inspection\inspection-demo-hf
.\.venv\Scripts\python.exe scripts/run_local_server.py --state-dir .runtime-host
```

Open `http://127.0.0.1:7860/`. Do not start a second server on the occupied port.
