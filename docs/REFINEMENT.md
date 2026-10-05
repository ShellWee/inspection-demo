# Inspection workspace refinement — 2026-09-04

## Scope and sources

Only `inspection-demo-hf` is changed. The original demo and research working
directories, pinned weights, asset dataset, graph and floor artifacts are unchanged.

Reviewed the requested HF implementation specification, this project's README,
SOURCE_MANIFEST and deployment scripts; the original demo README and deployment
notes; vendored ToG documentation and current pipeline configuration; the React
frontend, API contracts, asset bootstrap, job manager, research bridge, model
catalog, navigation/simulation, repository and existing tests.

The research pipeline remains paper-v14-clean-v1 / Text-GNN v5 / typed hierarchy /
closure-adjudication-v3. There are no query-specific branches, gold answers or
synthetic production targets. Presentation metadata is enriched from the
hash-verified graph **after** reasoning and cannot influence target selection.

## Interface

- Light engineering workspace: navy primary actions, restrained teal statuses,
  IBM Plex Sans typography, consistent spacing, borders and keyboard focus states.
- Setup presents Ecore, API key, model and an empty query field. Runtime names,
  checkpoints, hashes and GPU labels no longer occupy the main UI.
- Ground & Verify prioritizes the request, targets, relationship graph and an
  evidence summary. Raw closure codes and API usage are expandable. Abstention,
  runtime failure, unlocalizable and cross-floor results block simulation clearly.
- Execute supports toggles, drag sorting plus keyboard up/down controls, robot
  and planner choice, pointer or numeric starting pose, footprint clearance checks,
  all selected targets, and play/pause/restart/scrub/speed controls for replay.
- Responsive panel and canvas sizing; still a desktop-oriented demonstration.

## Defects addressed

1. Workflow tabs could enter Execute without loading the floor map.
2. New queries retained old tasks, map and run-scoped metadata.
3. Changed or failed API key connections retained a stale model catalog.
4. Hover races could overwrite newer selections; metadata failures were unhandled.
5. Cytoscape `node[target]` marked false target flags as targets too.
6. Displayed subgraphs dropped retrieval edges and could truncate certified
   targets. Related-node labels now use frozen graph metadata where available.
7. Navigation trusted a self-declared metadata flag; only matching floor artifacts
   now provide target coordinates.
8. Terminal SSE events preceded persisted results, racing result/replay fetches.
9. Cancellation left grounding children alive. They are now killed and awaited.
   Simulation cancellation uses cooperative planner checks and waits for its
   thread before releasing the worker. Simulation has a ten-minute timeout.
10. Blank queries, nonfinite poses, duplicate tasks/orders, empty enabled queues
    and unknown task IDs now fail validation. Invalid SSE cursors return 422.
11. Model connection has a bounded timeout and no retries; API validation errors
    are readable and duplicate SSE events ignored.
12. Replay replaces unlabeled fixed 5× playback with explicit controls and
    interpolates the shortest rotation across ±180°.
13. Floating-point drift made the replay slider's exact endpoint unreachable.
    Simulation timestamps are now normalized, and older replays are tolerated.

The static floor geometry is memoized on a separate canvas layer; robot updates
do not reconstruct the whole building on each animation frame.

## Limited real testing

The grounding API accepts optional tighter `limits`: `max_llm_calls`,
`input_token_budget`, `max_output_tokens`. Omission preserves normal research
settings. Limited GPT-4.1 uses the same model snapshot, temperature, prompts and
structured-output parser, with an output ceiling.

`scripts/refinement_live_check.py` allows one attempt for each of cases 1 and 2,
journaling before submission to prevent duplicate paid runs after network errors.
Limits: 24 calls, 60,000 input-token budget and 2,048 output tokens per call. The
input budget uses the existing research engine's phase-boundary checks, **not a
provider-enforced dollar cap**; an in-flight call may exceed that threshold.
No paid model probes or automatic paid test retries are used.

`usage` returns provider-reported input/output/cached tokens and embedding tokens.
Estimated GPT-4.1 costs use $2/$0.50/$8 per million input/cached/output tokens and
$0.02 per million embedding tokens; HF hardware, taxes and billing adjustments
are excluded. Sources: [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1)
and [API pricing](https://developers.openai.com/api/docs/pricing).

Reports and screenshots are under `.webapp/refinement/`, excluded from the image
and upload. Keys are never written there. Deterministic browser fixtures exist
only in excluded test scripts, not the production application.

## Verification and limitations

Commands: `python -m pytest backend/tests -q`, `pnpm --dir frontend test`,
`pnpm --dir frontend build`, `python scripts/verify_release.py`,
`python scripts/browser_refinement.py` (local frontend preview on port 5174).
The browser script exercises all three stages, ordering/toggles, pose validation,
replay and a second abstained request, with no paid calls.

This does not guarantee every arbitrary question succeeds or establish real HVAC
causality. No ROS, physical robot, multi-floor navigation or physical-dynamics
integration is added. A*, DWA and RRT* remain simplified simulation implementations.
The server plans the simulation; the browser replays it, not live robot telemetry.
Retry/skip on paused planning failures remains unsupported; choose another pose
or planner and start again. Chrome/Edge/WebKit coverage must be reported only for
engines actually tested.

Deployment and real-test outcomes are recorded after verification. Do not silently
resume a paused Space for paid L4 testing.

### Confirmed local real test

On 2026-09-04, GPT-4.1 processed `Navigate to room 404.` through the real HF-copy
backend using CPU for local GNN inference. Closure passed, one target resolved to
the hash-verified Level 4 artifact, and Jackal/DWA completed a collision-checked
428-frame simulation and persisted replay. Grounding took 33.7 seconds in this
single run; it is not an L4 or p95 performance claim.

Provider usage: 14 calls, 44,273 input tokens, 960 output tokens, no reported cached
input or new embedding tokens. Estimated OpenAI cost: **US$0.096226**. No second
paid query has been submitted. The source run and usage report are saved locally
under `.webapp/refinement/live-1-*`. Cloud deployment is pending permission to
temporarily resume the paused Space; no L4 was resumed for this test.

Final checks: **55 backend tests, 8 frontend tests**, TypeScript/production build,
Ruff and production leakage checks passed. All 61 vendored file hashes match
SOURCE_MANIFEST. Chromium verified synthetic multi-action and abstention UI paths,
plus the real saved grounding result, metadata, floor, new navigation execution
and replay via actual local APIs. No browser console/HTTP errors were observed.
At 1280 and 1440 px the tested views had no horizontal overflow. During one
two-second real-floor replay sample, the requestAnimationFrame callback p95 gap
was 16.7 ms; this is an observed callback cadence, not a cross-browser FPS guarantee.

Local preview: `http://127.0.0.1:7860/`. To restart using the existing host-owned
cache: `.\.venv\Scripts\python.exe scripts/run_local_server.py --state-dir .runtime-host`.
The default `.webapp` cache may have Windows ACLs belonging to a different execution
identity; selecting a readable cache avoids changing those permissions.
