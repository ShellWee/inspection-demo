from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Operator = Literal[
    "lookup",
    "list",
    "distinct",
    "count",
    "group_count",
    "argmax",
    "nearest",
    "all_matching",
    "unconnected",
    "path",
]

ExecutionProfile = Literal["legacy", "lean-grounding"]
RetrievalFlowProfile = Literal[
    "legacy",
    "closure-adjudication-v3",
]
RetrievalClosureArm = Literal[
    "no_hierarchy", "typed_hierarchy", "relation_shuffled_control"
]
TargetBindingMode = Literal["auto", "system_entity", "system_members"]
TargetUnit = Literal["entity", "logical_group"]
LogicalTargetGroupKind = Literal[
    "continuous_surface",
    "asset_system",
    "authored_asset",
]
CardinalityPolicy = Literal[
    "single", "all", "count", "distinct", "group_count", "argmax"
]
ScopeCardinality = Literal["single", "all", "aggregation"]
QueryHypothesisSelectionStatus = Literal[
    "selected", "ambiguous", "unsupported", "contradictory"
]
GnnRetrievalProfile = Literal["query-conditioned-plan-v5.0"]
GnnRepresentation = Literal["four-level", "flat"]


@dataclass(slots=True)
class ToGConfig:
    # Library users keep the historical behaviour unless they explicitly opt
    # into the bounded evaluation profile.  The lean profile applies hard
    # ceilings through the ``effective_*`` properties below, even when an old
    # runner still passes the former (larger) defaults.
    profile: ExecutionProfile = "legacy"
    # The retrieval-closure coordinator is an opt-in post-traversal branch.
    # ``legacy`` remains the default and therefore preserves ToG-BIM behavior.
    retrieval_flow_profile: RetrievalFlowProfile = "legacy"
    variant: Literal["canonical", "bim", "bim-gnn"] = "bim"
    width: int = 3
    depth: int = 3
    candidate_cap: int = 20
    retained_entities: int = 5
    random_seed: int = 42
    max_llm_calls: int = 48
    include_evidence: tuple[str, ...] = ("explicit", "inferred", "candidate")
    gnn_max_hops: int = 3
    gnn_max_depth: int = 4
    gnn_max_width: int = 5
    gnn_anchor_k: int = 20
    gnn_max_subgraph_nodes: int = 200
    gnn_max_subgraph_edges: int = 2000
    # ``variant`` identifies the graph reasoning family.  The GNN prior is an
    # independent intervention so R0--R3 can share every other setting.
    use_gnn_prior: bool | None = None
    gnn_retrieval_profile: GnnRetrievalProfile = "query-conditioned-plan-v5.0"
    gnn_representation: GnnRepresentation = "four-level"
    # Routing is derived from the raw question. Evaluator-side categories are
    # deliberately outside the public runtime contract.
    query_routing_profile: Literal["query-only-v1"] = "query-only-v1"
    gnn_use_anchors: bool = True
    gnn_use_expansion: bool = True
    hierarchy_reasoning: bool = False
    hierarchy_max_repairs: int = 3
    hierarchy_precision_gate: bool = False
    hierarchy_max_paths: int = 100
    hierarchy_max_depth: int = 6
    hierarchy_prompt_max_chars: int = 50_000
    prompt_evidence_limit: int = 500
    prompt_evidence_max_chars: int = 50_000
    audit_excluded_limit: int = 200
    input_token_budget: int | None = None
    traversal_max_llm_calls: int | None = None
    llm_hierarchy_review: bool | None = None
    deterministic_finalization: bool | None = None
    # Formal answering experiments may require one model-generated answer even
    # when the deterministic binding is already complete.  The category label
    # is never passed to that model; output shape comes from QueryPlan.
    llm_answering_required: bool = False
    hierarchy_path_control: Literal["full", "none", "shuffled"] = "full"
    hybrid_target_selection: bool = False
    target_selection_candidate_cap: int = 20
    target_selection_max_chars: int = 8_000
    bounded_query_hypotheses: bool = False
    semantic_enabled: bool = True
    semantic_llm_calls_per_question: int = 1
    semantic_input_token_budget: int = 2_000
    query_hypothesis_cap: int = 4
    deterministic_planner: bool = False
    debug: bool = False

    @property
    def effective_max_llm_calls(self) -> int:
        return min(self.max_llm_calls, 12) if self.profile == "lean-grounding" else self.max_llm_calls

    @property
    def effective_hierarchy_max_repairs(self) -> int:
        return min(self.hierarchy_max_repairs, 1) if self.profile == "lean-grounding" else self.hierarchy_max_repairs

    @property
    def effective_hierarchy_max_paths(self) -> int:
        return min(self.hierarchy_max_paths, 12) if self.profile == "lean-grounding" else self.hierarchy_max_paths

    @property
    def effective_hierarchy_prompt_max_chars(self) -> int:
        return min(self.hierarchy_prompt_max_chars, 8_000) if self.profile == "lean-grounding" else self.hierarchy_prompt_max_chars

    @property
    def effective_prompt_evidence_limit(self) -> int:
        return min(self.prompt_evidence_limit, 100) if self.profile == "lean-grounding" else self.prompt_evidence_limit

    @property
    def effective_prompt_evidence_max_chars(self) -> int:
        return min(self.prompt_evidence_max_chars, 8_000) if self.profile == "lean-grounding" else self.prompt_evidence_max_chars

    @property
    def effective_audit_excluded_limit(self) -> int:
        return min(self.audit_excluded_limit, 8) if self.profile == "lean-grounding" else self.audit_excluded_limit

    @property
    def effective_input_token_budget(self) -> int | None:
        if self.profile != "lean-grounding":
            return self.input_token_budget
        # Formal answering experiments opt in to one observable model answer
        # call per question.  Permit their explicitly supplied per-question
        # ceiling to exceed the ordinary lean-profile cap; all other lean runs
        # retain the historical 30k guardrail.
        if self.llm_answering_required and self.input_token_budget is not None:
            return self.input_token_budget
        return min(self.input_token_budget or 30_000, 30_000)

    @property
    def effective_traversal_max_llm_calls(self) -> int | None:
        if self.profile != "lean-grounding":
            return self.traversal_max_llm_calls
        return min(self.traversal_max_llm_calls or 4, 4)

    @property
    def effective_llm_hierarchy_review(self) -> bool:
        if self.llm_hierarchy_review is not None:
            return bool(self.llm_hierarchy_review) and self.profile != "lean-grounding"
        return self.profile != "lean-grounding"

    @property
    def effective_deterministic_finalization(self) -> bool:
        if self.deterministic_finalization is not None:
            return bool(self.deterministic_finalization) or self.profile == "lean-grounding"
        return self.profile == "lean-grounding"

    @property
    def effective_hybrid_target_selection(self) -> bool:
        return self.hybrid_target_selection or self.profile == "lean-grounding"

    @property
    def effective_semantic_llm_calls_per_question(self) -> int:
        if self.profile == "lean-grounding":
            return min(self.semantic_llm_calls_per_question, 1)
        return self.semantic_llm_calls_per_question

    @property
    def effective_semantic_input_token_budget(self) -> int:
        if self.profile == "lean-grounding":
            return min(self.semantic_input_token_budget, 2_000)
        return self.semantic_input_token_budget

    @property
    def effective_query_hypothesis_cap(self) -> int:
        if self.profile == "lean-grounding":
            return min(self.query_hypothesis_cap, 4)
        return self.query_hypothesis_cap


@dataclass(slots=True)
class EntityRef:
    node_id: str
    label: str
    global_id: str | None = None
    ifc_class: str | None = None
    kind: str = "entity"
    score: float = 0.0
    match_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RelationRef:
    name: str
    direction: Literal["out", "in", "value"]
    score: float = 0.0
    gnn_score: float = 0.0


@dataclass(slots=True)
class TripleEvidence:
    source_id: str
    source_label: str
    relation: str
    target_id: str
    target_label: str
    direction: str = "out"
    provenance: str = "explicit"
    confidence: float = 1.0
    relevance_score: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)

    def prompt_line(self) -> str:
        visible = {
            key: self.features[key]
            for key in (
                "global_id", "name", "long_name", "family", "type_name",
                "object_type", "tag", "storey", "role", "space_type", "distance",
            )
            if self.features.get(key) not in (None, "")
        }
        suffix = ""
        if visible:
            rendered = ", ".join(f"{key}={value}" for key, value in visible.items())
            suffix = f" [{rendered}]"
        return f"({self.source_label}, {self.relation}, {self.target_label}){suffix}"


ReasoningMode = Literal["deterministic", "hierarchy", "hybrid"]
ReviewDecision = Literal["pass", "repair", "abstain"]


@dataclass(slots=True)
class HierarchyEdge:
    source_id: str
    source_label: str
    source_ifc_class: str | None
    relation: str
    target_id: str
    target_label: str
    target_ifc_class: str | None
    provenance: str
    confidence: float

    def prompt_line(self) -> str:
        return (
            f"({self.source_label} [{self.source_ifc_class or 'unknown'}], "
            f"{self.relation}, {self.target_label} "
            f"[{self.target_ifc_class or 'unknown'}]) "
            f"{{provenance={self.provenance}, confidence={self.confidence:.3f}}}"
        )


@dataclass(slots=True)
class HierarchyPath:
    path_id: str
    target_id: str
    node_ids: list[str] = field(default_factory=list)
    node_labels: list[str] = field(default_factory=list)
    node_ifc_classes: list[str | None] = field(default_factory=list)
    edges: list[HierarchyEdge] = field(default_factory=list)

    def prompt_line(self) -> str:
        scope = " > ".join(
            f"{label} [{ifc_class or 'unknown'}]"
            for label, ifc_class in zip(self.node_labels, self.node_ifc_classes)
        )
        triples = "; ".join(edge.prompt_line() for edge in self.edges)
        return f"{self.path_id}: scope={scope}; triples={triples or 'none'}"


@dataclass(slots=True)
class HierarchyContext:
    summary: dict[str, Any] = field(default_factory=dict)
    paths: list[HierarchyPath] = field(default_factory=list)
    relevant_node_ids: list[str] = field(default_factory=list)
    truncated: bool = False
    truncation_reasons: list[str] = field(default_factory=list)
    prompt_max_paths: int = 100
    prompt_max_chars: int = 50_000

    def prompt_lines(self, limit: int = 100, max_chars: int = 50_000) -> list[str]:
        limit = min(limit, self.prompt_max_paths)
        max_chars = min(max_chars, self.prompt_max_chars)
        result: list[str] = []
        used = 0
        for path in self.paths[:limit]:
            line = path.prompt_line()
            if result and used + len(line) > max_chars:
                break
            result.append(line[:max_chars] if not result else line)
            used += len(result[-1])
        return result


@dataclass(slots=True)
class ReasoningRequirement:
    requirement: str
    status: Literal["satisfied", "missing", "conflicting"]
    evidence_ids: list[str] = field(default_factory=list)
    conclusion: str = ""


@dataclass(slots=True)
class HierarchyReasoningTrace:
    requirements: list[ReasoningRequirement] = field(default_factory=list)
    hierarchy_claims: list[str] = field(default_factory=list)
    conclusion: str = ""
    sufficient: bool = False
    missing_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceCoverage:
    requirement: str
    covered: bool
    evidence_ids: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class TargetValidation:
    target_id: str
    valid: bool
    checks: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TargetAudit:
    included_targets: list[dict[str, Any]] = field(default_factory=list)
    excluded_candidates: list[dict[str, Any]] = field(default_factory=list)
    target_validations: list[TargetValidation] = field(default_factory=list)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    action_bindings_covered: bool = True
    conflicting_extras: bool = False


@dataclass(slots=True)
class EvidenceReview:
    decision: ReviewDecision = "repair"
    evidence_summary: str = ""
    intent_aligned: bool = False
    reasoning_supported: bool = False
    hierarchy_consistent: bool = False
    deterministic_complete: bool = False
    hierarchy_required: bool = False
    coverage: list[EvidenceCoverage] = field(default_factory=list)
    target_validations: list[TargetValidation] = field(default_factory=list)
    audited_target_count: int = 0
    action_bindings_covered: bool = True
    conflicting_extras: bool = False
    missing_requirements: list[str] = field(default_factory=list)
    needed_relations: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class ActionTargetBinding:
    action: str
    # ``binding_index`` identifies an action occurrence rather than an action
    # name.  This matters for plans such as Inspect(A), Inspect(B), where
    # deduplicating the verb would silently lose the second binding.
    binding_index: int = 0
    source_stage: str = "targets"
    target_kind: str | None = None
    target_roles: list[str] = field(default_factory=list)
    target_names: list[str] = field(default_factory=list)
    target_domains: list[str] = field(default_factory=list)
    # Coordinated noun phrases retain their conjunct-local semantics.  Each
    # branch is an AND of role/name/domain constraints; branches are ORed.
    # Flat fields remain a broad retrieval projection for compatibility.
    constraint_branches: list[dict[str, Any]] = field(default_factory=list)
    result_stage: str = "targets"
    function_types: list[str] = field(default_factory=list)
    system_categories: list[str] = field(default_factory=list)
    target_mode: TargetBindingMode = "auto"
    cardinality_policy: CardinalityPolicy = "single"
    # Natural-language singularity does not always imply one IFC product.
    # A logical group remains one action-binding unit while deterministic
    # finalization expands its verified member entities into executable PDDL
    # actions.  The default preserves the historical one-entity contract.
    target_unit: TargetUnit = "entity"
    allowed_logical_group_kinds: list[LogicalTargetGroupKind] = field(
        default_factory=list
    )


@dataclass(slots=True)
class MentionLink:
    """A question mention linked to graph/schema vocabulary.

    The compiler never needs a graph backend.  Callers may supply lightweight
    node records and the linker records only the typed information needed by
    downstream planning.  ``node_ids`` are candidate identities, not selected
    answers.
    """

    text: str
    canonical: str
    kind: Literal["space", "object", "system", "function", "property", "unknown"]
    char_start: int = -1
    char_end: int = -1
    action_index: int | None = None
    query_focus: bool = False
    node_ids: list[str] = field(default_factory=list)
    candidate_count: int = 0
    ifc_class: str | None = None
    graph_level: str | None = None
    role: str | None = None
    domain: str | None = None
    space_type: str | None = None
    function_type: str | None = None
    function_kind: str | None = None
    system_category: str | None = None
    compatible_roles: list[str] = field(default_factory=list)
    compatible_names: list[str] = field(default_factory=list)
    compatible_domains: list[str] = field(default_factory=list)
    compatible_space_types: list[str] = field(default_factory=list)
    related_system_categories: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: Literal["graph", "schema", "syntax"] = "graph"
    source_field: str = ""


@dataclass(slots=True)
class RelationReference:
    """A typed reference entity for a relational operator.

    References are deliberately separate from hard containment scope.  For
    example, in "the device nearest a named room", the room is the metric reference,
    not a room that must contain the target device.
    """

    stage_id: str
    relation: Literal[
        "nearest",
        "adjacent_to",
        "contains",
        "serves",
        "assigned_to_system",
        "unconnected",
    ]
    mentions: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    reference_kind: str | None = None
    source_stage: str = "scope"


@dataclass(slots=True)
class ScopePredicate:
    """A typed constraint evaluated while constructing the spatial scope."""

    stage_id: str
    predicate: Literal[
        "storey",
        "space_number",
        "space_name",
        "space_type",
        "space_function",
        "contains_role",
        "contains_domain",
        "contains_name",
        "argmax_area",
    ]
    values: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TargetPredicate:
    """A typed target constraint evaluated after the scope stage."""

    stage_id: str = "targets"
    predicate: Literal[
        "kind",
        "ifc_class",
        "role",
        "name",
        "domain",
        "function",
        "system",
    ] = "kind"
    values: list[str] = field(default_factory=list)
    source_stage: str = "scope"
    required: bool = True


@dataclass(slots=True)
class QueryPlan:
    operator: Operator = "lookup"
    mentions: list[str] = field(default_factory=list)
    storey: str | None = None
    room: str | None = None
    room_names: list[str] = field(default_factory=list)
    target_kind: str | None = None
    target_ifc_class: str | None = None
    target_role: str | None = None
    target_roles: list[str] = field(default_factory=list)
    target_domain: str | None = None
    target_space_type: str | None = None
    target_name: str | None = None
    target_names: list[str] = field(default_factory=list)
    # Surface noun phrases for a coordinated target.  These retain syntax
    # that a broad retrieval projection (target_roles/target_names) cannot,
    # including per-conjunct cardinality.  They are graph linked before being
    # used and never contain benchmark labels or expected answers.
    target_phrases: list[str] = field(default_factory=list)
    target_family_terms: list[str] = field(default_factory=list)
    target_type_terms: list[str] = field(default_factory=list)
    target_keywords: list[str] = field(default_factory=list)
    property_terms: list[str] = field(default_factory=list)
    action_sequence: list[str] = field(default_factory=list)
    action_bindings: list[ActionTargetBinding] = field(default_factory=list)
    mention_links: list[MentionLink] = field(default_factory=list)
    scope_predicates: list[ScopePredicate] = field(default_factory=list)
    relation_references: list[RelationReference] = field(default_factory=list)
    target_predicates: list[TargetPredicate] = field(default_factory=list)
    function_intents: list[str] = field(default_factory=list)
    target_binding_mode: TargetBindingMode = "auto"
    cardinality_policy: CardinalityPolicy = "single"
    scope_cardinality: ScopeCardinality = "single"
    search_exhaustive: bool = False
    target_grouping: list[str] = field(default_factory=list)
    unresolved_slots: list[str] = field(default_factory=list)
    requires_exhaustive: bool = False
    gnn_hops: int = 1
    tog_depth: int = 3
    tog_width: int = 3
    gnn_retrieval_levels: list[str] = field(default_factory=lambda: ["object", "space"])
    rationale: str = ""


@dataclass(slots=True)
class QueryHypothesis:
    """One bounded, typed interpretation of a natural-language instruction.

    ``hypothesis_id`` is an opaque caller-owned identity.  It is never an IFC
    node identity and adapters should expose only short call-local aliases to
    an LLM.  The remaining fields are offline diagnostics: they summarize
    whether graph execution can support the interpretation without exposing
    candidate node IDs or permitting the model to generate answer entities.
    """

    hypothesis_id: str
    plan: QueryPlan
    hard_valid_count: int = 0
    contradiction: bool = False
    missing_metric: bool = False
    binding_complete: bool = False
    evidence_summary: list[str] = field(default_factory=list)
    path_summary: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryHypothesisSelectionResult:
    """Whitelisted result of selecting among supplied query hypotheses."""

    status: QueryHypothesisSelectionStatus = "unsupported"
    selected_hypothesis_id: str | None = None
    rejected_hypothesis_ids: list[str] = field(default_factory=list)
    hypothesis_count: int = 0
    llm_used: bool = False
    fallback_used: bool = False
    reason: str = ""


@dataclass(slots=True)
class OperatorResult:
    answer: str = ""
    candidates: list[EntityRef] = field(default_factory=list)
    evidence: list[TripleEvidence] = field(default_factory=list)
    matched_predicates: list[str] = field(default_factory=list)
    failed_predicates: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    complete: bool = False
    unresolved_slots: list[str] = field(default_factory=list)
    excluded_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class LogicalTargetGroup:
    """A typed, auditable logical target composed of executable IFC members.

    ``group_id`` is a stable derived identity, not an IFC GUID.  Only
    ``member_ids`` may be expanded into PDDL action arguments.  Group
    construction is deterministic and records enough evidence for target
    audit to verify the exact member set independently of candidate order.
    """

    group_id: str
    group_kind: LogicalTargetGroupKind
    binding_index: int
    member_ids: list[str] = field(default_factory=list)
    source: Literal["explicit_graph", "virtual_authored"] = "virtual_authored"
    identity: str = ""
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TargetSelectionResult:
    selected_ids: list[str] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)
    candidate_count: int = 0
    llm_used: bool = False
    fallback_used: bool = False
    reason: str = ""
    logical_groups: list[LogicalTargetGroup] = field(default_factory=list)


@dataclass(slots=True)
class HierarchyValidation:
    binding_valid_paths: list[str] = field(default_factory=list)
    violated_constraints: list[str] = field(default_factory=list)
    unresolved_slots: list[str] = field(default_factory=list)
    target_node_ids: list[str] = field(default_factory=list)
    support_node_ids: list[str] = field(default_factory=list)
    per_binding_path_coverage: dict[int, list[str]] = field(default_factory=dict)
    targets_without_valid_path: list[str] = field(default_factory=list)
    extra_path_valid_targets: list[str] = field(default_factory=list)
    closure_complete: bool = False
    complete: bool = False


@dataclass(slots=True)
class GnnAnchor:
    node_id: str
    level: str
    similarity: float
    rank: int
    label: str = ""


@dataclass(slots=True)
class GnnSubgraphEdge:
    source_id: str
    relation: str
    target_id: str
    direction: str
    provenance: str
    confidence: float


@dataclass(slots=True)
class GnnSubgraphPath:
    target_id: str
    node_ids: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    score: float = 0.0
    binding_index: int | None = None
    scope_id: str | None = None
    scope_match_mode: str = "unverified"
    complete_typed_path: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalCandidateUniverse:
    """Bounded target/support identities admitted to C2--C4 finalization.

    The universe is produced only from retrieval, exact query seeds, and
    bounded typed expansion.  It prevents deterministic graph operators from
    reintroducing arbitrary full-graph nodes after the retrieval intervention.
    """

    target_node_ids: tuple[str, ...]
    support_node_ids: tuple[str, ...] = ()
    exact_seed_node_ids: tuple[str, ...] = ()
    max_target_nodes: int = 200
    max_support_nodes: int = 200
    max_total_nodes: int = 200
    schema_version: str = "retrieval-candidate-universe-v2"

    def __post_init__(self) -> None:
        targets = tuple(dict.fromkeys(str(value) for value in self.target_node_ids))
        supports = tuple(dict.fromkeys(str(value) for value in self.support_node_ids))
        exact = tuple(dict.fromkeys(str(value) for value in self.exact_seed_node_ids))
        if not all(targets) or not all(supports) or not all(exact):
            raise ValueError("Candidate-universe node IDs must be non-empty strings")
        if len(targets) > self.max_target_nodes:
            raise ValueError("Candidate-universe target budget exceeded")
        if len(supports) > self.max_support_nodes:
            raise ValueError("Candidate-universe support budget exceeded")
        if len(targets) + len(supports) > self.max_total_nodes:
            raise ValueError("Candidate-universe total node budget exceeded")
        if set(targets).intersection(supports):
            raise ValueError("Target and support candidate identities must be disjoint")
        if not set(exact).issubset(targets):
            raise ValueError("Exact actionable seeds must remain in the target universe")
        object.__setattr__(self, "target_node_ids", targets)
        object.__setattr__(self, "support_node_ids", supports)
        object.__setattr__(self, "exact_seed_node_ids", exact)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidatePathEvidence:
    """Binding-specific retrieval and hierarchy evidence for one candidate."""

    binding_index: int = 0
    node_id: str = ""
    path_status: Literal[
        "complete", "partial", "unsupported", "contradictory"
    ] = "unsupported"
    node_ids: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    constraint_states: dict[str, str] = field(default_factory=dict)
    positive_constraints: list[str] = field(default_factory=list)
    unknown_constraints: list[str] = field(default_factory=list)
    contradiction: bool = False
    scope_connected: bool = False
    target_anchor_rank: int | None = None
    support_anchor_rank: int | None = None
    gnn_rank_score: float = 0.0
    path_confidence: float = 0.0
    hub_exposure: float = 0.0
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GnnSubgraph:
    anchors: list[GnnAnchor] = field(default_factory=list)
    target_anchors: list[GnnAnchor] = field(default_factory=list)
    support_anchors: list[GnnAnchor] = field(default_factory=list)
    evaluation_anchors: list[GnnAnchor] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    edges: list[GnnSubgraphEdge] = field(default_factory=list)
    paths: list[GnnSubgraphPath] = field(default_factory=list)
    support_to_target_paths: list[GnnSubgraphPath] = field(default_factory=list)
    binding_candidate_ids: dict[int, list[str]] = field(default_factory=dict)
    candidate_evidence: dict[str, list[CandidatePathEvidence]] = field(
        default_factory=dict
    )
    path_valid_target_ids: list[str] = field(default_factory=list)
    node_scores: dict[str, float] = field(default_factory=dict)
    relation_support: dict[str, float] = field(default_factory=dict)
    hops: int = 0
    truncated: bool = False
    truncation_reasons: list[str] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    prior_used: bool = False
    query_embedding_calls: int = 0
    query_embedding_cache_hits: int = 0
    query_embedding_tokens: int = 0
    rrf_trace: dict[str, Any] = field(default_factory=dict)
    target_descriptor: str = ""
    scope_descriptor: str = ""
    prefilter_pool_counts: dict[str, int] = field(default_factory=dict)
    retrieval_stage: str = "initial"
    max_nodes: int = 0
    max_edges: int = 0
    candidate_universe: RetrievalCandidateUniverse | None = None


@dataclass(slots=True)
class GnnArtifactReport:
    artifact_dir: Path
    artifact_hash: str
    source_hash: str
    embedding_dim: int
    node_count: int
    overlap_count: int
    validated: bool
    manifest_schema: str = ""
    representation_graph_hash: str = ""
    checkpoint_hash: str = ""
    embedding_space: str = ""
    profile: str = ""
    query_input: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["artifact_dir"] = str(self.artifact_dir)
        return result


@dataclass(slots=True)
class IndexBuildReport:
    index_path: Path
    source_path: Path
    source_hash: str
    schema_version: str
    cache_hit: bool
    build_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["index_path"] = str(self.index_path)
        result["source_path"] = str(self.source_path)
        return result


@dataclass(slots=True)
class ToGResponse:
    answer: str
    variant: str
    operator: str
    execution_profile: str = "legacy"
    retrieval_flow_profile: str = "legacy"
    reasoning_chains: list[list[TripleEvidence]] = field(default_factory=list)
    seed_entities: list[EntityRef] = field(default_factory=list)
    evidence: list[TripleEvidence] = field(default_factory=list)
    visited_nodes: int = 0
    visited_edges: int = 0
    depth_reached: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Provider-reported subsets of the historical totals above. Cached input
    # remains part of ``input_tokens`` and reasoning remains part of
    # ``output_tokens``; these fields expose the breakdown without double
    # counting either total.
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    # Run-level accounting fields are additive and keep the historical
    # ``input_tokens`` value intact for older integrations.
    gnn_embedding_calls_total: int = 0
    gnn_embedding_input_tokens_total: int = 0
    reasoning_input_tokens_total: int = 0
    system_input_tokens_total: int = 0
    graph_hash: str = ""
    graph_schema: str = ""
    planned_hops: int = 0
    planned_depth: int = 0
    planned_width: int = 0
    gnn_retrieval_levels: list[str] = field(default_factory=list)
    gnn_artifact_hash: str = ""
    gnn_subgraph: GnnSubgraph | None = None
    initial_gnn_subgraph: GnnSubgraph | None = None
    hierarchy_context: HierarchyContext | None = None
    reasoning_mode: ReasoningMode | None = None
    reasoning_trace: HierarchyReasoningTrace | None = None
    evidence_review: EvidenceReview | None = None
    target_audit: TargetAudit | None = None
    operator_result: OperatorResult | None = None
    query_hypotheses: list[QueryHypothesis] = field(default_factory=list)
    query_hypothesis_selection: QueryHypothesisSelectionResult | None = None
    target_selection: TargetSelectionResult | None = None
    hierarchy_validation: HierarchyValidation | None = None
    retrieval_evidence_packet: dict[str, Any] | None = None
    evidence_closure_certificate: dict[str, Any] | None = None
    closure_repair_trace: dict[str, Any] | None = None
    selected_entities: list[EntityRef] = field(default_factory=list)
    constraint_valid_node_ids: list[str] = field(default_factory=list)
    binding_complete: bool = False
    phase_call_usage: dict[str, int] = field(default_factory=dict)
    phase_token_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    repair_count: int = 0
    finalization_mode: str = "legacy"
    validation_status: str = "not_run"
    budget_exhausted: bool = False
    budget_reason: str = ""
    budget_status: str = "ok"
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
