enum RoGDirection {
  Out @alias("out")
  In @alias("in")
  Value @alias("value")
}

class RoGRelationStepResult {
  relation string @description("Exact relation name copied from the graph schema")
  direction RoGDirection
  target_ifc_class string?
  target_category string?
  target_role string?
  target_domain string?
  target_space_type string?
  target_name_contains string?
}

class RoGRelationPathResult {
  path_id string
  purpose string
  steps RoGRelationStepResult[]
}

class RoGPlanResult {
  seed_node_ids string[] @description("Only node IDs from the supplied seed candidates")
  relation_paths RoGRelationPathResult[]
  requires_exhaustive bool
  action_sequence string[]
  rationale string
}

class RoGAdaptivePlanResult {
  seed_node_ids string[] @description("Only node IDs from the supplied seed candidates")
  relation_paths RoGRelationPathResult[]
  requires_exhaustive bool
  action_sequence string[]
  rationale string
}

class RoGAnswerResult {
  answer string
}

function PlanIfcRoGStatic(
  question: string,
  category: int,
  schema_json: string,
  seed_candidates_json: string,
  constraints_json: string,
) -> RoGPlanResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    You are the planning stage of a RoG-style IFC knowledge-graph baseline.
    Produce at most three faithful relation paths, each with at most three hops.
    Copy relation names and directions exactly from schema_json. Unknown aliases
    are invalid and must never be invented. Copy seed node IDs only from the
    supplied candidates. A zero-hop path is allowed only when the seed itself is
    the requested answer entity.

    This is pure relation-path retrieval: do not request SQL, aggregation,
    argmax, nearest-neighbor computation, raw IFC access, or outside knowledge.
    Preserve exhaustive and action constraints. Categories are 1=QA,
    2=explicit PDDL, 3=spatial PDDL, 4=functional PDDL.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Category: {{ category }}
    Constraints: {{ constraints_json }}
    Resolved seed candidates: {{ seed_candidates_json }}
    Compact graph schema: {{ schema_json }}
  "#
}

function PlanIfcRoGAdaptive(
  question: string,
  category: int,
  schema_json: string,
  seed_candidates_json: string,
  constraints_json: string,
  exploration_history_json: string,
  tool_docs: string,
) -> CodeAction | RoGAdaptivePlanResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    You are the adaptive planning stage of a RoG-style IFC graph baseline.
    Either return CodeAction to inspect the graph through the listed read-only
    functions, or return a final structured relation plan. Python persists
    between iterations. Imports, files, network, SQL, raw IFC, eval/exec/open,
    Cobbie helper tools, and ground truth are unavailable.

    A final plan may contain at most three paths and three hops per path. Copy
    exact relation/direction pairs from the schema or observations and only use
    supplied seed IDs. Preview output helps planning only; the engine will
    independently execute the final plan for evidence. This is pure path
    retrieval, not aggregation/distance/operator execution.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Category: {{ category }}
    Constraints: {{ constraints_json }}
    Seed candidates: {{ seed_candidates_json }}
    Initial compact schema: {{ schema_json }}
    Tools:
    {{ tool_docs }}
    Exploration history: {{ exploration_history_json }}
  "#
}

function GenerateIfcRoGAnswer(
  question: string,
  category: int,
  plan_json: string,
  reasoning_paths: string[],
) -> RoGAnswerResult {
  client GLM_4_7
  prompt #"
    {{ _.role("system") }}
    You are the shared reasoning stage of a RoG-style IFC graph baseline.
    Answer strictly and only from the validated plan and retrieved reasoning
    paths. Do not use exploration history, raw IFC, external facts, or general
    knowledge. If the paths do not ground the complete answer, answer with an
    explicit abstention.

    Category 1: give the direct answer and concise path evidence. Categories
    2-4: output only ordered Navigate(raw_IFC_GUID), Inspect(raw_IFC_GUID), or
    Scan(raw_IFC_GUID) calls. Use only terminal entity GUIDs visible in the
    paths; never invent a GUID or add an ifc_ prefix.
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
    Category: {{ category }}
    Validated plan: {{ plan_json }}
    Retrieved reasoning paths:
    {% for path in reasoning_paths %}- {{ path }}
    {% endfor %}
  "#
}
