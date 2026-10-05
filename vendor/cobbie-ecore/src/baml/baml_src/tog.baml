enum ToGOperator {
  Lookup @alias("lookup")
  List @alias("list")
  Distinct @alias("distinct")
  Count @alias("count")
  GroupCount @alias("group_count")
  Argmax @alias("argmax")
  Nearest @alias("nearest")
  AllMatching @alias("all_matching")
  Unconnected @alias("unconnected")
  Path @alias("path")
}

class ToGQueryPlanResult {
  operator ToGOperator
  mentions string[]
  storey string?
  room string?
  room_names string[]
  target_kind string?
  target_ifc_class string?
  target_role string?
  target_roles string[]
  target_domain string?
  target_space_type string?
  target_name string?
  target_names string[]
  target_family_terms string[]
  target_type_terms string[]
  target_keywords string[]
  property_terms string[]
  action_sequence string[]
  action_bindings ToGActionTargetBindingResult[]
  requires_exhaustive bool
  gnn_hops int @description("GNN subgraph expansion hops within the supplied bounds")
  tog_depth int @description("Maximum ToG traversal depth within the supplied bounds")
  tog_width int @description("ToG relation/entity beam width within the supplied bounds")
  gnn_retrieval_levels string[] @description("One or more of object, space, system, function")
  rationale string
}

class ToGActionTargetBindingResult {
  action string
  binding_index int @description("Zero-based occurrence index in the requested action order")
  target_kind string?
  target_roles string[]
  target_names string[]
  target_domains string[]
  function_types string[]
  system_categories string[]
  cardinality_policy string @description("One of single, all, count, distinct, group_count, argmax")
}

class ToGRelationScore {
  name string
  direction string @description("One of out, in, value")
  score float @description("Relevance score from 0 to 1")
}

class ToGRelationSelection {
  relations ToGRelationScore[]
}

class ToGEntityScore {
  node_id string
  score float @description("Relevance score from 0 to 1")
}

class ToGEntitySelection {
  entities ToGEntityScore[]
}

enum ToGTargetResolutionStatus {
  Resolved @alias("resolved")
  Ambiguous @alias("ambiguous")
  Unsupported @alias("unsupported")
  Contradictory @alias("contradictory")
}

class ToGTargetResolution {
  status ToGTargetResolutionStatus
  selected_ids string[] @description("Only node IDs supplied in candidate_constraints_json")
  rejected_ids string[] @description("Only node IDs supplied in candidate_constraints_json")
  reason string
}

enum ToGQueryHypothesisSelectionStatus {
  Selected @alias("selected")
  Ambiguous @alias("ambiguous")
  Unsupported @alias("unsupported")
  Contradictory @alias("contradictory")
}

class ToGQueryHypothesisSelection {
  status ToGQueryHypothesisSelectionStatus
  selected_hypothesis_id string? @description("Only a call-local H1, H2, ... ID supplied in hypotheses_json")
  rejected_hypothesis_ids string[] @description("Only call-local H1, H2, ... IDs supplied in hypotheses_json")
  reason string
}

class ToGSufficiencyResult {
  sufficient bool
  reason string
}

class ToGAnswerResult {
  answer string
}

class ToGReasoningRequirementResult {
  requirement string
  status string @description("Exactly one of satisfied, missing, conflicting")
  evidence_ids string[] @description("Only evidence:N or hierarchy:N IDs supplied in the prompt")
  conclusion string
}

class ToGHierarchyReasoningResult {
  requirements ToGReasoningRequirementResult[]
  hierarchy_claims string[] @description("Concise claims, each grounded by supplied path/evidence IDs")
  conclusion string
  sufficient bool
  missing_evidence string[]
  conflicting_evidence string[]
}

enum ToGEvidenceDecision {
  Pass @alias("pass")
  Repair @alias("repair")
  Abstain @alias("abstain")
}

class ToGEvidenceCoverageResult {
  requirement string
  covered bool
  evidence_ids string[] @description("Only evidence:N or hierarchy:N IDs supplied in the prompt")
  note string
}

class ToGEvidenceReviewResult {
  decision ToGEvidenceDecision
  evidence_summary string
  intent_aligned bool
  reasoning_supported bool
  hierarchy_consistent bool
  deterministic_complete bool
  hierarchy_required bool
  coverage ToGEvidenceCoverageResult[]
  action_bindings_covered bool
  conflicting_extras bool
  missing_requirements string[]
  needed_relations string[] @description("Exact graph relations needed for repair, normally contains or part_of")
  reason string
}

enum ToGSelectionModeV3 {
  OneGraphNode @alias("one_graph_node")
  BestEquivalenceClass @alias("best_equivalence_class")
  AllMatching @alias("all_matching")
  BoundedOperator @alias("bounded_operator")
}

class ToGBindingIntentV3 {
  binding_index int
  action string @description("Exactly Navigate, Inspect, or Scan, preserving occurrence order")
  selection_mode ToGSelectionModeV3
  target_phrase string @description("Only the action target noun phrase; exclude scope/reference phrases")
  target_terms string[] @description("Concise target family/type/role/name terms copied or paraphrased from the question")
  target_kind string @description("space, object, system, function, or empty when unresolved")
  target_ifc_class string @description("IFC class only when explicit or unambiguous; otherwise empty")
  scope_phrase string
  scope_terms string[]
  relation string @description("Canonical relation or empty when not required")
  direction string @description("target_to_scope, scope_to_target, symmetric, unspecified, or empty")
  requires_exhaustive_set bool
}

class ToGRetrievalIntentContractV3 {
  bindings ToGBindingIntentV3[]
  summary string
}

class ToGBindingGroupSelectionV3 {
  binding_index int
  selected_group_ids string[] @description("Only call-local BxxGxxx group IDs from the sealed ledger")
  uncertain_group_ids string[] @description("Only unresolved groups that could change the answer")
  reason string @description("One concise evidence-grounded explanation")
}

class ToGGroupAdjudicationResultV3 {
  binding_selections ToGBindingGroupSelectionV3[]
  closure_supported bool
  issues string[]
  answer_summary string @description("Concise grounded answer conclusion without GUIDs or invented identifiers")
}

function PlanIfcToGQuery(
  question: string,
  fallback_plan_json: string,
  max_gnn_hops: int,
  max_tog_depth: int,
  max_tog_width: int,
) -> ToGQueryPlanResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Compile the instruction into the typed building-query schema. Use only the
    instruction and deterministic graph links; never use evaluation labels,
    reference answers, benchmark assumptions, examples, or outside facts.
    Preserve stated constraints and high-confidence links. Keep unresolved text
    unresolved unless its linked candidate explicitly supplies compatible role,
    domain, or system vocabularies; in that case you may fill the slot using
    only values copied from those vocabularies. Never invent an entity, type,
    role, system, or function.

    Operators: lookup, list, distinct, count, group_count, argmax, nearest,
    all_matching, unconnected, path. Actions: Navigate, Inspect, Scan. Preserve
    action order and keep every action's scope, target, function/system, and
    cardinality separate. Graph kinds are space, object, system, function;
    relations are contains, adjacent_to, serves, assigned_to_system, nearest,
    and unconnected. Choose the smallest retrieval bounds sufficient for the
    explicit relation stages, within the supplied maxima.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Deterministic typed links: {{ fallback_plan_json }}
    Retrieval maxima: hops={{ max_gnn_hops }}, depth={{ max_tog_depth }}, width={{ max_tog_width }}
  "#
}

function ResolveIfcToGTargets(
  question: string,
  plan_json: string,
  candidate_constraints_json: string,
) -> ToGTargetResolution {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Resolve typed action-binding slots from a bounded IFC/BIM candidate list.
    Use only node IDs present in candidate_constraints_json. Never add an ID,
    infer a missing spatial relation, or select a support-only node.

    A hard fail excludes a candidate. An unknown constraint is missing evidence,
    not positive evidence. Select every candidate required by an explicit plural
    or all-cardinality binding, and exactly one only when the plan requests a
    singular target and the supplied evidence distinguishes it. If required
    geometry, adjacency, containment, or system evidence is unavailable, return
    unsupported or ambiguous instead of guessing.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Typed plan: {{ plan_json }}
    Candidate constraint matrix: {{ candidate_constraints_json }}
  "#
}

function SelectIfcQueryHypothesis(
  question: string,
  hypotheses_json: string,
) -> ToGQueryHypothesisSelection {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Select a typed interpretation of the building instruction from the bounded
    hypotheses supplied by the caller. Compare only their typed plan semantics
    and offline diagnostics. A contradiction is negative evidence; a missing
    metric or incomplete binding is missing evidence, not permission to guess.

    If exactly one interpretation is supported, return selected and copy its
    call-local ID exactly. Otherwise return ambiguous, unsupported, or
    contradictory and leave selected_hypothesis_id null. You may copy only H1,
    H2, ... IDs present in hypotheses_json. Never output, infer, request, or
    construct a graph node ID, IFC GUID, action target, or answer entity.

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Instruction: {{ question }}
    Bounded typed hypotheses: {{ hypotheses_json }}
  "#
}

function SelectIfcToGRelations(
  question: string,
  entity_label: string,
  relation_candidates: string[],
  width: int,
) -> ToGRelationSelection {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Select at most {{ width }} relations that best advance the IFC question.
    Copy name and direction exactly from the candidates. Score relevance only;
    do not treat inferred/candidate evidence confidence as relevance.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Current entity: {{ entity_label }}
    Candidate format is direction|name:
    {% for relation in relation_candidates %}- {{ relation }}
    {% endfor %}
  "#
}

function ScoreIfcToGEntities(
  question: string,
  relation: string,
  entity_candidates: string[],
  width: int,
) -> ToGEntitySelection {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Rank at most {{ width }} IFC graph candidates for the question. Copy each
    node_id exactly. Prefer exact room/storey/name/role constraints and retain
    literal property values when the question asks for a property.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Traversed relation: {{ relation }}
    Candidate format is node_id|label|IFC class|metadata:
    {% for entity in entity_candidates %}- {{ entity }}
    {% endfor %}
  "#
}

function AssessIfcToGEvidence(
  question: string,
  evidence: string[],
) -> ToGSufficiencyResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Decide whether the retrieved IFC graph evidence is sufficient to ground the
    complete answer. For all/list/count/argmax/nearest questions, a few samples
    are not sufficient; deterministic operator output is still required. For
    PDDL tasks, the required raw IFC GlobalIds and every requested target/action
    must be present. Do not use outside knowledge.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Evidence:
    {% for item in evidence %}- {{ item }}
    {% endfor %}
  "#
}

function GenerateIfcToGAnswer(
  question: string,
  output_contract_json: string,
  plan_json: string,
  evidence: string[],
  deterministic_answer: string,
) -> ToGAnswerResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Answer only from the supplied IFC graph evidence and deterministic operator
    result. Never add facts from general knowledge. If evidence is insufficient,
    explicitly abstain.

    Follow the supplied QueryPlan-derived output contract exactly. A direct
    operator result uses concise auditable prose. An ordered action contract
    uses only its declared actions as Action(raw_IFC_GUID), preserves binding
    order and exhaustive targets, and never adds an ifc_ prefix.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Output contract: {{ output_contract_json }}
    Question: {{ question }}
    Query plan: {{ plan_json }}
    Deterministic result: {{ deterministic_answer }}
    Evidence:
    {% for item in evidence %}- {{ item }}
    {% endfor %}
  "#
}

function ReasonIfcToGHierarchy(
  question: string,
  output_contract_json: string,
  plan_json: string,
  reasoning_mode: string,
  hierarchy_summary_json: string,
  hierarchy_paths: string[],
  evidence: string[],
  deterministic_answer: string,
) -> ToGHierarchyReasoningResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Build a concise, auditable IFC hierarchy reasoning trace. Map every explicit
    requirement in the original question to supplied evidence IDs. Hierarchy
    claims must follow only supplied typed inspection paths: contains, part_of,
    assigned_to_system, serves, related_to_system, and requires_inspection_of.
    Respect their stated direction, provenance, and confidence. Actionability is
    determined by the supplied action_target_kind, not by hierarchy level: L0
    spaces and L1 objects may be targets, and an explicit GUID-bearing L2 system
    may be a system target. Inferred/non-actionable L2 nodes and all current L3
    function nodes are support only. Do not invent intermediate nodes,
    facts, GUIDs, or hidden chain-of-thought. Mark unsupported requirements as
    missing and contradictory paths as conflicting.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Original question: {{ question }}
    Output contract: {{ output_contract_json }}
    Reasoning mode: {{ reasoning_mode }}
    Query plan: {{ plan_json }}
    Deterministic result: {{ deterministic_answer }}
    Hierarchy summary: {{ hierarchy_summary_json }}
    Hierarchy paths:
    {% for item in hierarchy_paths %}- hierarchy:{{ loop.index }} {{ item }}
    {% endfor %}
    Graph evidence:
    {% for item in evidence %}- evidence:{{ loop.index }} {{ item }}
    {% endfor %}
  "#
}

function ReviewIfcToGEvidence(
  question: string,
  output_contract_json: string,
  plan_json: string,
  reasoning_mode: string,
  hierarchy_summary_json: string,
  hierarchy_paths: string[],
  evidence: string[],
  deterministic_answer: string,
  reasoning_trace_json: string,
  target_validation_summary_json: string,
  invalid_targets_json: string,
  excluded_candidates_json: string,
  action_bindings_json: string,
) -> ToGEvidenceReviewResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Independently review whether final IFC evidence and the explicit reasoning
    trace answer the original question intent. Check entity, room, storey,
    relation, functional/system path, nested scope, operator, requested action
    order, exhaustive scope, and raw GUID
    requirements. Per-target room, class, role, family/type, relation, and action
    checks are authoritative deterministic backend results; review their summary
    and invalid targets rather than trying to enumerate every included target.
    Every action binding must be covered and conflicting extras must fail.
    When a binding is missing, name the concrete slot in missing_requirements,
    for example `missing Scan target: role=panel, room=444`. When a typed
    functional route is missing, use `missing functional path:
    function=thermal_ir_scan -> target=panel` and request only the relevant
    relation template (`requires_inspection_of`, or
    `related_to_system` + `assigned_to_system`).
    Treat declarative conditions supplied by the question (for example water
    stain, poor airflow, leakage, or malfunction) as premises. Do not require
    those premises to appear in IFC evidence unless the question asks you to
    discover which target has that condition. Inspect excluded candidates as
    negative evidence. A deterministic lookup/aggregation may pass without hierarchy
    when no relation or hop is needed. Return repair only when bounded graph
    retrieval could supply missing evidence; return abstain for contradictions
    or ungroundable intent. Never use outside knowledge or a reference answer.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Original question: {{ question }}
    Output contract: {{ output_contract_json }}
    Reasoning mode: {{ reasoning_mode }}
    Query plan: {{ plan_json }}
    Deterministic result: {{ deterministic_answer }}
    Hierarchy summary: {{ hierarchy_summary_json }}
    Hierarchy paths:
    {% for item in hierarchy_paths %}- hierarchy:{{ loop.index }} {{ item }}
    {% endfor %}
    Graph evidence:
    {% for item in evidence %}- evidence:{{ loop.index }} {{ item }}
    {% endfor %}
    Reasoning trace: {{ reasoning_trace_json }}
    Backend target validation summary: {{ target_validation_summary_json }}
    Backend invalid targets: {{ invalid_targets_json }}
    Excluded candidates (bounded): {{ excluded_candidates_json }}
    Action bindings: {{ action_bindings_json }}
  "#
}

function FinalizeIfcToGAnswer(
  question: string,
  output_contract_json: string,
  plan_json: string,
  deterministic_answer: string,
  hierarchy_paths: string[],
  evidence: string[],
  reasoning_trace_json: string,
  evidence_review_json: string,
) -> ToGAnswerResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    Produce the most strongly supported final answer from the supplied evidence,
    including when the evidence review did not pass after bounded repair.
    Use only the deterministic result, hierarchy paths, and graph evidence.
    Follow the supplied QueryPlan-derived output contract. Direct operator
    results use concise auditable prose; ordered action contracts use only the
    declared Action(raw_IFC_GUID) calls. Preserve binding order and exhaustive
    targets and never invent a GUID.
    Never prefix the answer with "Abstain:". If evidence is incomplete, retain
    only supported targets and do not fill gaps with outside knowledge.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Original question: {{ question }}
    Output contract: {{ output_contract_json }}
    Query plan: {{ plan_json }}
    Deterministic result: {{ deterministic_answer }}
    Hierarchy paths:
    {% for item in hierarchy_paths %}- hierarchy:{{ loop.index }} {{ item }}
    {% endfor %}
    Graph evidence:
    {% for item in evidence %}- evidence:{{ loop.index }} {{ item }}
    {% endfor %}
    Reasoning trace: {{ reasoning_trace_json }}
    Evidence review: {{ evidence_review_json }}
  "#
}

function CompileIfcRetrievalIntentContractV3(
  question: string,
  compact_plan_json: string,
) -> ToGRetrievalIntentContractV3 {
  client OpenAI_GPT_4_1_2025_04_14_NoCap
  prompt #"
    {{ _.role("system") }}
    Independently compile an IFC retrieval answer contract from the original
    instruction and a compact, fallible QueryPlan. This call cannot see graph
    candidates, node catalogues, answer IDs, gold labels, or benchmark fields.

    Separate each action target from reference/scope entities. Preserve every
    action occurrence and binding_index in order; never add or remove a binding.
    Correct a target phrase, role, class, scope, or cardinality only when the
    original instruction itself supports the correction.

    A natural-language singular denotes one semantic referent, but one referent
    may be represented by several equivalent IFC leaves/components. Use:
    - one_graph_node only for an explicit unique tag/identifier;
    - bounded_operator for nearest/closest/argmax semantics;
    - all_matching for explicit all/list/set/compound-target requests;
    - best_equivalence_class for an ordinary definite referent that may contain
      repeated, paired, tiled, or component IFC nodes.

    target_phrase and target_terms must exclude rooms, storeys, reference
    objects, faults, and reasons for the action. scope fields contain those
    constraints instead. Do not output an IFC GUID, node ID, candidate ID,
    benchmark label, hidden answer, or outside fact. Use only generic IFC
    semantics and concise auditable text.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Original instruction: {{ question }}
    Compact QueryPlan: {{ compact_plan_json }}
  "#
}

function AdjudicateAndAnswerIfcEvidenceGroupsV3(
  question: string,
  intent_contract_json: string,
  candidate_group_ledger_json: string,
) -> ToGGroupAdjudicationResultV3 {
  client OpenAI_GPT_4_1_2025_04_14_NoCap
  prompt #"
    {{ _.role("system") }}
    Propose a bounded IFC evidence closure from a sealed candidate-group
    ledger. Select only supplied call-local
    BxxGxxx group IDs; never emit or invent a node, GUID, relation, path, graph
    fact, benchmark label, or outside fact.

    Each group is an evidence-equivalence class: selecting it deterministically
    selects every member node. Use target family/type/name/role before broad
    domain similarity. A more specific target phrase must reject merely related
    equipment. For scoped or relational requests, require an authoritative typed
    path or bounded-operator scope witness with the correct endpoint semantics.
    Missing metadata is unknown rather than negative evidence, but an explicit
    class/kind contradiction cannot be overridden.

    For one_graph_node select exactly one one-member group. For
    best_equivalence_class select the best semantically equivalent group(s),
    including multiple groups only when together they express a coordinated or
    component target phrase. For all_matching select every and only matching
    group inside the bounded scope. For bounded_operator select a winning group
    that contains at least one bounded-operator member; equivalent leaves in that
    same group remain part of the referent. Put only genuinely answer-changing
    ambiguity in uncertain_group_ids. Omitted groups are rejected.

    closure_supported means every action binding is supportable from this sealed
    bounded universe. answer_summary is explanatory only; deterministic code
    owns GUID expansion, cardinality, action names, and action order. Keep the
    response concise and evidence-grounded.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Original instruction: {{ question }}
    Normalized intent contract: {{ intent_contract_json }}
    Sealed candidate-group ledger: {{ candidate_group_ledger_json }}
  "#
}

function ReviewAndAnswerIfcEvidenceGroupsV3(
  question: string,
  intent_contract_json: string,
  candidate_group_ledger_json: string,
  initial_selection_json: string,
) -> ToGGroupAdjudicationResultV3 {
  client OpenAI_GPT_4_1_2025_04_14_NoCap
  prompt #"
    {{ _.role("system") }}
    Independently review and, when evidence requires it, repair an IFC group
    selection inside one sealed candidate-group ledger. This is the required
    answering call. You may only add or remove supplied call-local BxxGxxx
    group IDs. Never emit or invent a node, GUID, relation, path, graph fact,
    benchmark label, or outside fact.

    Re-evaluate the initial proposal against the original instruction and the
    normalized contract. Check target specificity before broad domain
    similarity; scope endpoint, relation, and direction when requested; and
    action-binding order. A selected scoped target needs an authoritative typed
    path or bounded-operator scope witness. Missing metadata is unknown, while
    an explicit class or kind contradiction is disqualifying.

    For one_graph_node return exactly one one-member group. For
    best_equivalence_class return the smallest group set that expresses the
    requested semantic referent. For all_matching include every and only group
    proven to match inside the bounded scope; do not treat related equipment as
    the requested target. For bounded_operator select only a winning group with
    a bounded-operator witness. Put an alias in uncertain_group_ids only when
    resolving it would change the answer and the sealed evidence cannot do so.
    Omitted groups are rejected.

    closure_supported means every binding is closed after this one repair.
    answer_summary is explanatory only: deterministic code owns GUID expansion,
    action names, cardinality, and action order. Keep the response concise and
    auditable; do not expose hidden chain-of-thought.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Original instruction: {{ question }}
    Normalized intent contract: {{ intent_contract_json }}
    Sealed candidate-group ledger: {{ candidate_group_ledger_json }}
    Initial group selection: {{ initial_selection_json }}
  "#
}
