from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar

from ._closure_ledger import (
    compact_query_plan_v2,
)
from ._closure_packet import (
    RetrievalEvidencePacketV1,
    canonical_json,
    packet_arm_invariants,
    sha256_json,
)
from .backend import IfcGraphBackend
from .index import GraphIndexManager
from .llm import CallBudgetExceeded, ToGLlm
from .models import (
    ActionTargetBinding,
    EntityRef,
    EvidenceCoverage,
    EvidenceReview,
    GnnArtifactReport,
    GnnSubgraph,
    HierarchyContext,
    HierarchyReasoningTrace,
    HierarchyValidation,
    IndexBuildReport,
    LogicalTargetGroup,
    OperatorResult,
    QueryHypothesis,
    QueryHypothesisSelectionResult,
    QueryPlan,
    ReasoningMode,
    ReasoningRequirement,
    RelationRef,
    RetrievalCandidateUniverse,
    RetrievalClosureArm,
    TargetAudit,
    TargetSelectionResult,
    ToGConfig,
    ToGResponse,
    TripleEvidence,
)
from .planning import (
    PlanningSchema,
    build_action_bindings,
    infer_query_plan,
    merge_query_plan,
)
from .retrieval_closure import (
    adjudicate_group_selection_v3,
    build_candidate_group_ledger_v3,
    certified_answer_v3,
    contract_question_view,
    normalize_group_selection_v3,
    normalize_intent_contract_v3,
    reconcile_group_selections_v3,
)
from .retriever_plugin import (
    RetrievalRuntimeContext,
    SubgraphRetrievalRequest,
    SubgraphRetrieverPlugin,
    retrieve_subgraph,
    validate_retriever_runtime,
)
from .routing import RetrievalRouteDecision, classify_query_intent

T = TypeVar("T")


class RetrievalQueryPlanProvider(Protocol):
    """Return a query plan used only by an external retrieval provider.

    This seam deliberately does not replace ToG's own planner output.  The
    latter continues to drive relation selection, graph traversal, entity
    pruning, sufficiency reasoning, and final answering.
    """

    def plan(self, question: str, fallback: QueryPlan) -> QueryPlan: ...

_PHASE_NAMES = (
    "planner",
    "query_hypothesis",
    "gnn_retrieval",
    "traversal",
    "candidate_selector",
    "hierarchy",
    "repair",
    "final_explanation",
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "from"}
    }


def _partition_closure_upstream_errors(
    errors: Sequence[str],
    *,
    llm_calls: int,
    max_llm_calls: int,
    reserved_downstream_calls: int = 3,
) -> tuple[list[str], list[str]]:
    """Separate the expected closure call-cap stop from actual upstream errors.

    Retrieval Closure reserves exactly three calls for reason, review, and answer.
    The legacy traversal can therefore end by attempting call ``max_calls - 3``;
    that bounded stop is part of the preregistered runtime contract, not a
    provider/schema fallback.  Any differently worded budget event, an early
    occurrence, or any other error remains fail-closed.
    """

    upstream_cap = max(0, int(max_llm_calls) - int(reserved_downstream_calls))
    expected = f"ToG LLM call budget exhausted ({upstream_cap})"
    actual_errors: list[str] = []
    bounded_stops: list[str] = []
    for raw in errors:
        message = str(raw)
        if message == expected and int(llm_calls) == upstream_cap:
            if message not in bounded_stops:
                bounded_stops.append(message)
        elif message not in actual_errors:
            actual_errors.append(message)
    return actual_errors, bounded_stops


class HeuristicToGLlm:
    """No-network fallback and deterministic test implementation."""

    calls = 0
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    reasoning_output_tokens = 0

    def plan_query(
        self,
        question: str,
        fallback: QueryPlan,
        max_gnn_hops: int,
        max_tog_depth: int,
        max_tog_width: int,
    ) -> QueryPlan:
        return fallback

    def select_relations(
        self,
        question: str,
        entity: EntityRef,
        relations: Sequence[RelationRef],
        width: int,
    ) -> list[RelationRef]:
        query_tokens = _tokens(question)
        aliases = {
            "contains": {"room", "space", "in", "inside", "all", "fixture", "element"},
            "part_of": {"level", "building", "part"},
            "adjacent_to": {"adjacent", "next", "nearby"},
            "connects_to": {"connect", "system", "duct", "pipe"},
            "assigned_to_system": {"system", "hvac", "plumbing", "electrical", "fire"},
            "requires_inspection_of": {"inspect", "scan", "safety", "issue"},
            "serves": {"serves", "room", "hvac"},
        }
        scored: list[RelationRef] = []
        for relation in relations:
            relation_tokens = _tokens(relation.name.replace("_", " ")) | aliases.get(relation.name, set())
            score = float(len(query_tokens & relation_tokens))
            if relation.direction == "value" and any(
                token in relation.name.lower() for token in query_tokens
            ):
                score += 2.0
            scored.append(RelationRef(relation.name, relation.direction, score))
        scored.sort(key=lambda item: (-item.score, item.name, item.direction))
        selected = scored[:width]
        if selected and all(item.score == 0 for item in selected):
            for index, relation in enumerate(selected):
                relation.score = 1.0 / (index + 1)
        return selected

    def score_entities(
        self,
        question: str,
        relation: RelationRef,
        entities: Sequence[EntityRef],
        width: int,
    ) -> list[EntityRef]:
        query_tokens = _tokens(question)
        result: list[EntityRef] = []
        for entity in entities:
            metadata_text = " ".join(str(value) for value in entity.metadata.values())
            overlap = len(query_tokens & _tokens(f"{entity.label} {metadata_text}"))
            entity.score = relation.score * (1.0 + overlap)
            result.append(entity)
        result.sort(key=lambda item: (-item.score, item.node_id))
        return result[:width]

    def resolve_targets(
        self,
        question: str,
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
    ) -> TargetSelectionResult:
        hard_valid = []
        for entity in candidates:
            matrix = entity.metadata.get("_constraint_status", {})
            statuses: set[str] = set()

            def collect(value: object) -> None:
                if isinstance(value, dict):
                    for key, nested in value.items():
                        if key == "overall" and isinstance(nested, str):
                            statuses.add(nested.lower())
                        elif key == "constraints" and isinstance(nested, dict):
                            statuses.update(
                                str(item).lower() for item in nested.values()
                            )
                        else:
                            collect(nested)
                elif isinstance(value, list):
                    for nested in value:
                        collect(nested)

            collect(matrix)
            if "fail" not in statuses:
                hard_valid.append(entity)
        if len(hard_valid) == 1 and plan.cardinality_policy == "single":
            return TargetSelectionResult(
                selected_ids=[hard_valid[0].node_id],
                rejected_ids=[
                    item.node_id for item in candidates
                    if item.node_id != hard_valid[0].node_id
                ],
                candidate_count=len(candidates),
                reason="deterministic_unique_constraint_resolution",
            )
        return TargetSelectionResult(
            selected_ids=[],
            rejected_ids=[],
            candidate_count=len(candidates),
            reason="deterministic_ambiguous_or_unsupported",
        )

    def select_query_hypothesis(
        self,
        question: str,
        hypotheses: Sequence[QueryHypothesis],
    ) -> QueryHypothesisSelectionResult:
        del question
        viable = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.hard_valid_count > 0
            and not hypothesis.contradiction
            and not hypothesis.missing_metric
        ]
        if len(viable) == 1:
            selected = viable[0]
            return QueryHypothesisSelectionResult(
                status="selected",
                selected_hypothesis_id=selected.hypothesis_id,
                rejected_hypothesis_ids=[
                    item.hypothesis_id
                    for item in hypotheses
                    if item.hypothesis_id != selected.hypothesis_id
                ],
                hypothesis_count=len(hypotheses),
                reason="deterministic_unique_supported_hypothesis",
            )
        return QueryHypothesisSelectionResult(
            status="ambiguous" if viable else "unsupported",
            rejected_hypothesis_ids=[],
            hypothesis_count=len(hypotheses),
            reason="deterministic_hypothesis_ambiguity",
        )

    def is_sufficient(self, question: str, evidence: Sequence[TripleEvidence]) -> bool:
        return bool(evidence)

    def generate_answer(
        self,
        question: str,
        plan: QueryPlan,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
    ) -> str:
        return deterministic_answer

    def reason_hierarchy(
        self,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_mode: ReasoningMode,
    ) -> HierarchyReasoningTrace:
        evidence_ids = [
            f"evidence:{index + 1}" for index, _item in enumerate(evidence[:100])
        ]
        path_ids = [
            f"hierarchy:{index + 1}"
            for index, _path in enumerate(hierarchy_context.paths)
        ]
        requirements = [
            ReasoningRequirement(
                requirement=f"operator={plan.operator}",
                status="satisfied" if deterministic_answer else "missing",
                evidence_ids=evidence_ids[:5],
                conclusion=deterministic_answer,
            )
        ]
        if plan.storey:
            requirements.append(
                ReasoningRequirement(
                    requirement=f"storey={plan.storey}",
                    status="satisfied" if hierarchy_context.paths else "missing",
                    evidence_ids=path_ids[:5],
                    conclusion="Hierarchy paths preserve the requested storey scope.",
                )
            )
        if plan.room:
            requirements.append(
                ReasoningRequirement(
                    requirement=f"room={plan.room}",
                    status="satisfied" if hierarchy_context.paths else "missing",
                    evidence_ids=path_ids[:5],
                    conclusion="Hierarchy paths preserve the requested room scope.",
                )
            )
        missing = [item.requirement for item in requirements if item.status == "missing"]
        return HierarchyReasoningTrace(
            requirements=requirements,
            hierarchy_claims=[path.prompt_line() for path in hierarchy_context.paths[:10]],
            conclusion=deterministic_answer,
            sufficient=not missing and bool(deterministic_answer),
            missing_evidence=missing,
        )

    def review_evidence(
        self,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_mode: ReasoningMode,
        reasoning_trace: HierarchyReasoningTrace,
        target_audit: TargetAudit,
    ) -> EvidenceReview:
        abstention = deterministic_answer.startswith(
            "I cannot determine the answer from the IFC knowledge graph."
        )
        deterministic_complete = bool(deterministic_answer) and not abstention
        hierarchy_required = reasoning_mode in {"hierarchy", "hybrid"}
        hierarchy_consistent = bool(hierarchy_context.paths) or not hierarchy_required
        reasoning_supported = reasoning_trace.sufficient or (
            reasoning_mode == "deterministic" and deterministic_complete
        )
        passed = deterministic_complete and hierarchy_consistent and reasoning_supported
        decision = "pass" if passed else "repair"
        coverage = [
            EvidenceCoverage(
                requirement=item.requirement,
                covered=item.status == "satisfied",
                evidence_ids=list(item.evidence_ids),
                note=item.conclusion,
            )
            for item in reasoning_trace.requirements
        ]
        missing = [item.requirement for item in coverage if not item.covered]
        target_validations = list(target_audit.target_validations)
        if any(not item.valid for item in target_validations):
            decision = "repair"
            passed = False
        action_bindings_covered = target_audit.action_bindings_covered
        if not action_bindings_covered:
            decision = "repair"
        binding_coverage = target_audit.validation_summary.get(
            "action_binding_coverage", {}
        )
        for index, binding in enumerate(plan.action_bindings):
            if not bool(binding_coverage.get(str(index), False)):
                slots = []
                if binding.target_roles:
                    slots.append(f"role={'|'.join(binding.target_roles)}")
                if binding.target_names:
                    slots.append(f"name={'|'.join(binding.target_names)}")
                missing.append(
                    f"missing {binding.action} target: "
                    + (", ".join(slots) if slots else f"kind={binding.target_kind}")
                )
        needed_relations: list[str] = []
        for validation in target_validations:
            for issue in validation.issues:
                if issue.startswith("missing_functional_path"):
                    function_type = issue.partition(":")[2]
                    missing.append(
                        f"missing functional path: function={function_type} -> target={validation.target_id}"
                    )
                    needed_relations.extend(
                        ["requires_inspection_of", "related_to_system", "assigned_to_system"]
                    )
                elif issue in {"missing_authoritative_room_path", "required_relation_missing"}:
                    needed_relations.append("contains")
        if hierarchy_required and not hierarchy_consistent:
            needed_relations.extend(["contains", "part_of"])
        return EvidenceReview(
            decision=decision,
            evidence_summary=deterministic_answer,
            intent_aligned=deterministic_complete,
            reasoning_supported=reasoning_supported,
            hierarchy_consistent=hierarchy_consistent,
            deterministic_complete=deterministic_complete,
            hierarchy_required=hierarchy_required,
            coverage=coverage,
            target_validations=target_validations,
            audited_target_count=len(target_audit.included_targets),
            action_bindings_covered=action_bindings_covered,
            conflicting_extras=target_audit.conflicting_extras,
            missing_requirements=list(dict.fromkeys(missing)),
            needed_relations=list(dict.fromkeys(needed_relations)),
            reason="heuristic grounded review",
        )

    def finalize_answer(
        self,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_trace: HierarchyReasoningTrace,
        evidence_review: EvidenceReview,
    ) -> str:
        return deterministic_answer

class ToGSystem:
    def __init__(
        self,
        index_manager: GraphIndexManager,
        llm_factory: Callable[[], ToGLlm] | None = None,
        config: ToGConfig | None = None,
        rebuild_index: bool = False,
        gnn_retriever: SubgraphRetrieverPlugin | None = None,
        retrieval_query_plan_provider: RetrievalQueryPlanProvider | None = None,
    ) -> None:
        self.index_manager = index_manager
        self.llm_factory = llm_factory or HeuristicToGLlm
        self.config = config or ToGConfig()
        self.rebuild_index = rebuild_index
        self.gnn_retriever = gnn_retriever
        self.retrieval_query_plan_provider = retrieval_query_plan_provider
        self.use_gnn_prior = (
            self.config.variant == "bim-gnn"
            if self.config.use_gnn_prior is None
            else bool(self.config.use_gnn_prior)
        )
        self.use_query_conditioned_v50 = (
            self.config.gnn_retrieval_profile == "query-conditioned-plan-v5.0"
        )
        self.use_retrieval_prior = self.use_gnn_prior
        self._backends: dict[Path, IfcGraphBackend] = {}
        self.index_reports: dict[Path, IndexBuildReport] = {}
        self.gnn_artifact_reports: dict[Path, GnnArtifactReport] = {}
        self._planning_schemas: dict[str, PlanningSchema] = {}
        self._retrieval_packet_runtime: dict[str, dict[str, object]] = {}
        if self.config.profile not in {"legacy", "lean-grounding"}:
            raise ValueError("profile must be 'legacy' or 'lean-grounding'")
        if self.config.retrieval_flow_profile not in {
            "legacy",
            "closure-adjudication-v3",
        }:
            raise ValueError("invalid retrieval_flow_profile")
        if self.config.retrieval_flow_profile == "closure-adjudication-v3":
            if not self.use_gnn_prior or not self.use_query_conditioned_v50:
                raise ValueError(
                    "retrieval closure requires the frozen v5 GNN retrieval profile"
                )
            if self.config.variant != "bim-gnn":
                raise ValueError("retrieval closure requires variant='bim-gnn'")
            if not self.config.hierarchy_reasoning:
                raise ValueError(
                    "retrieval closure requires hierarchy_reasoning for sealed path inventory"
                )
        if self.config.hierarchy_path_control not in {"full", "none", "shuffled"}:
            raise ValueError("hierarchy_path_control must be full, none, or shuffled")
        if self.config.gnn_representation not in {"four-level", "flat"}:
            raise ValueError("gnn_representation must be 'four-level' or 'flat'")
        if self.config.query_routing_profile != "query-only-v1":
            raise ValueError("invalid query_routing_profile")
        if self.use_gnn_prior and not self.use_query_conditioned_v50:
            raise ValueError(
                "the current ToG package only accepts query-conditioned-plan-v5.0"
            )
        if self.use_retrieval_prior and self.gnn_retriever is None:
            raise ValueError("configured retrieval profile requires a SubgraphRetrieverPlugin")
        if self.use_retrieval_prior and self.gnn_retriever is not None:
            retriever_profile = getattr(
                self.gnn_retriever, "retrieval_profile", None
            )
            compatible_retriever_profiles = {self.config.gnn_retrieval_profile}
            if self.config.gnn_retrieval_profile == "query-conditioned-plan-v5.0":
                compatible_retriever_profiles.add(
                    "query-conditioned-plan-v5.0-generic"
                )
            if retriever_profile not in compatible_retriever_profiles:
                raise ValueError(
                    "ToG/retriever query-aware hybrid profile mismatch: "
                    f"config={self.config.gnn_retrieval_profile} "
                    f"retriever={retriever_profile or 'unbound'}"
                )
        if (
            self.retrieval_query_plan_provider is not None
            and not self.use_query_conditioned_v50
        ):
            raise ValueError(
                "Retrieval-only QueryPlan providers require query-conditioned-plan-v5.0"
            )
        if self.use_retrieval_prior:
            if not 0 <= self.config.gnn_max_hops <= 3:
                raise ValueError("gnn_max_hops must be between 0 and 3")
            if not 1 <= self.config.gnn_max_depth <= 4:
                raise ValueError("gnn_max_depth must be between 1 and 4")
            if not 1 <= self.config.gnn_max_width <= 5:
                raise ValueError("gnn_max_width must be between 1 and 5")
            if self.config.gnn_anchor_k <= 0:
                raise ValueError("gnn_anchor_k must be positive")
            if (
                self.config.gnn_max_subgraph_nodes <= 0
                or self.config.gnn_max_subgraph_edges <= 0
            ):
                raise ValueError("GNN subgraph caps must be positive")
        if not 0 <= self.config.hierarchy_max_repairs <= 3:
            raise ValueError("hierarchy_max_repairs must be between 0 and 3")
        if self.config.hierarchy_max_paths <= 0:
            raise ValueError("hierarchy_max_paths must be positive")
        if self.config.hierarchy_max_depth <= 0:
            raise ValueError("hierarchy_max_depth must be positive")
        if self.config.effective_max_llm_calls <= 0:
            raise ValueError("max_llm_calls must be positive")
        if self.config.target_selection_candidate_cap <= 0:
            raise ValueError("target_selection_candidate_cap must be positive")
        if self.config.target_selection_max_chars <= 0:
            raise ValueError("target_selection_max_chars must be positive")
        if self.config.effective_semantic_llm_calls_per_question <= 0:
            raise ValueError("semantic_llm_calls_per_question must be positive")
        if self.config.effective_semantic_input_token_budget <= 0:
            raise ValueError("semantic_input_token_budget must be positive")
        if not 2 <= self.config.effective_query_hypothesis_cap <= 4:
            raise ValueError("query_hypothesis_cap must be between 2 and 4")
        if (
            self.config.effective_input_token_budget is not None
            and self.config.effective_input_token_budget <= 0
        ):
            raise ValueError("input_token_budget must be positive")

    def _retrieval_candidate_universe(
        self,
        backend: IfcGraphBackend,
        seeds: Sequence[EntityRef],
        subgraph: GnnSubgraph,
        legal_expansion: Sequence[EntityRef] = (),
    ) -> RetrievalCandidateUniverse:
        """Build the frozen-v5 binding-scoped target/support boundary.

        ``legal_expansion`` remains in the private call signature so the core
        traversal can pass its frontier without changing phase ordering.  It
        cannot promote targets: only frozen per-binding candidates or complete
        typed paths may do so.
        """

        del legal_expansion
        exact_ids = {
            seed.node_id
            for seed in seeds
            if seed.kind == "entity"
            and seed.score >= 0.9
            and backend.action_target_kind(seed) != "none"
        }
        binding_ids = {
            node_id
            for values in subgraph.binding_candidate_ids.values()
            for node_id in values
        }
        target_ids = binding_ids | set(subgraph.path_valid_target_ids)
        support_ids = {anchor.node_id for anchor in subgraph.support_anchors}
        support_ids.update(subgraph.node_ids)
        support_ids.difference_update(target_ids)
        ranked_targets = sorted(
            target_ids,
            key=lambda node_id: (
                -float(subgraph.node_scores.get(node_id, -1.0)),
                node_id,
            ),
        )[: self.config.gnn_max_subgraph_nodes]
        remaining_budget = max(
            0, self.config.gnn_max_subgraph_nodes - len(ranked_targets)
        )
        ranked_supports = sorted(
            support_ids,
            key=lambda node_id: (
                -float(subgraph.node_scores.get(node_id, -1.0)),
                node_id,
            ),
        )[:remaining_budget]
        admitted_exact = tuple(sorted(exact_ids.intersection(ranked_targets)))
        return RetrievalCandidateUniverse(
            target_node_ids=tuple(ranked_targets),
            support_node_ids=tuple(ranked_supports),
            exact_seed_node_ids=admitted_exact,
            max_target_nodes=self.config.gnn_max_subgraph_nodes,
            max_support_nodes=self.config.gnn_max_subgraph_nodes,
            max_total_nodes=self.config.gnn_max_subgraph_nodes,
        )

    def _planning_schema(self, backend: IfcGraphBackend) -> PlanningSchema:
        """Build a reusable language vocabulary from the active graph.

        Evaluation categories and answers never enter this projection.  The
        same compiler therefore sees the same schema for every query and every
        reporting category over a graph artifact.
        """
        cached = self._planning_schemas.get(backend.graph_hash)
        if cached is not None:
            return cached
        rows = backend.connection.execute(
            """
            SELECT node_id, global_id, ifc_class, name, long_name,
                   family, type_name, object_type, tag, level, category,
                   domain, role, space_type, system_category, function_type
            FROM nodes
            ORDER BY node_id
            """
        )
        records = [dict(row) for row in rows]
        records_by_id = {str(row["node_id"]): row for row in records}
        for value_row in backend.connection.execute(
            """
            SELECT node_id, predicate, value_text
            FROM node_values
            WHERE predicate IN (
                'property.synonyms',
                'property.function_kind',
                'property.target_roles',
                'property.target_name_terms',
                'property.target_domains',
                'property.target_space_types',
                'property.related_system_categories'
            )
            ORDER BY node_id, predicate, value_text
            """
        ):
            record = records_by_id.get(str(value_row["node_id"]))
            if record is None:
                continue
            field_name = {
                "property.synonyms": "synonyms",
                "property.function_kind": "function_kind",
                "property.target_roles": "target_roles",
                "property.target_name_terms": "target_name_terms",
                "property.target_domains": "target_domains",
                "property.target_space_types": "target_space_types",
                "property.related_system_categories": "related_system_categories",
            }[str(value_row["predicate"])]
            if field_name == "function_kind":
                record.setdefault(field_name, str(value_row["value_text"]))
            else:
                record.setdefault(field_name, []).append(
                    str(value_row["value_text"])
                )
        schema = PlanningSchema.from_records(records)
        self._planning_schemas[backend.graph_hash] = schema
        return schema

    @staticmethod
    def _plan_requires_fallback(plan: QueryPlan) -> bool:
        if plan.unresolved_slots:
            return True
        if any(
            getattr(link, "kind", "unknown") == "unknown"
            and getattr(link, "action_index", None) is not None
            for link in getattr(plan, "mention_links", [])
        ):
            return True
        return any(
            getattr(reference, "mentions", [])
            and not getattr(reference, "node_ids", [])
            for reference in getattr(plan, "relation_references", [])
        )

    @staticmethod
    def _plan_interpretation_key(plan: QueryPlan) -> str:
        """Return the typed fields that materially change graph execution."""

        return json.dumps(
            {
                "operator": plan.operator,
                "target_kind": plan.target_kind,
                "target_binding_mode": plan.target_binding_mode,
                "cardinality": plan.cardinality_policy,
                "scope_cardinality": plan.scope_cardinality,
                "bindings": [
                    {
                        "index": item.binding_index,
                        "action": item.action,
                        "kind": item.target_kind,
                        "mode": item.target_mode,
                        "cardinality": item.cardinality_policy,
                        "roles": list(item.target_roles),
                        "names": list(item.target_names),
                        "domains": list(item.target_domains),
                        "functions": list(item.function_types),
                        "systems": list(item.system_categories),
                    }
                    for item in plan.action_bindings
                ],
            },
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _system_mode_plan(plan: QueryPlan, mode: str) -> QueryPlan:
        """Project one graph-linked system mention into a typed binding mode."""

        result = copy.deepcopy(plan)
        if mode == "system_members":
            result.target_kind = "object"
            result.target_binding_mode = "system_members"
            result.cardinality_policy = "all"
            result.requires_exhaustive = True
            result.search_exhaustive = True
            result.action_bindings = [
                replace(
                    item,
                    target_kind="object",
                    target_mode="system_members",
                    cardinality_policy="all",
                )
                for item in result.action_bindings
            ]
        elif mode == "system_entity":
            result.target_kind = "system"
            result.target_binding_mode = "system_entity"
            result.cardinality_policy = "single"
            result.requires_exhaustive = False
            result.action_bindings = [
                replace(
                    item,
                    target_kind="system",
                    target_mode="system_entity",
                    cardinality_policy="single",
                )
                for item in result.action_bindings
            ]
        else:
            raise ValueError(f"unsupported system binding mode: {mode}")
        result.rationale = (
            f"{plan.rationale}; graph-backed hypothesis target_mode={mode}"
        ).strip("; ")
        return result

    def _compile_query_hypotheses(
        self,
        backend: IfcGraphBackend,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
    ) -> list[QueryHypothesis]:
        """Enumerate and execute a bounded set of typed interpretations.

        No evaluation label or expected answer is available here.  Alternatives
        are generated only from an explicit graph-linked system mention, and
        the LLM-facing diagnostics contain aggregate evidence rather than IFC
        identities.
        """

        if not plan.action_bindings:
            return []
        system_linked = any(
            getattr(link, "kind", None) == "system"
            for link in plan.mention_links
        ) or any(
            binding.system_categories
            or binding.target_mode in {"system_entity", "system_members"}
            for binding in plan.action_bindings
        )
        if not system_linked:
            return []

        interpretations: list[tuple[str, QueryPlan]] = [("base", plan)]
        current_modes = {
            binding.target_mode for binding in plan.action_bindings
        } | {plan.target_binding_mode}
        if "system_entity" in current_modes or plan.target_kind == "system":
            interpretations.append(
                ("system_members", self._system_mode_plan(plan, "system_members"))
            )
        if "system_members" in current_modes or plan.target_kind == "object":
            interpretations.append(
                ("system_entity", self._system_mode_plan(plan, "system_entity"))
            )

        unique: list[tuple[str, QueryPlan]] = []
        seen: set[str] = set()
        for name, candidate_plan in interpretations:
            key = self._plan_interpretation_key(candidate_plan)
            if key in seen:
                continue
            seen.add(key)
            unique.append((name, candidate_plan))
        unique = unique[: self.config.effective_query_hypothesis_cap]
        if len(unique) < 2:
            return []

        hypotheses: list[QueryHypothesis] = []
        for index, (name, candidate_plan) in enumerate(unique, 1):
            result = backend.execute_operator_result(candidate_plan, seeds)
            # The hypothesis stage is a rescue path.  A complete base
            # interpretation must remain untouched even when another typed
            # system projection could also retrieve graph entities.
            if name == "base" and result.complete:
                return []
            relation_counts = Counter(
                item.relation for item in result.evidence
            )
            provenance_counts = Counter(
                item.provenance for item in result.evidence
            )
            hard_valid_count = len(result.candidates)
            missing_metric = bool(
                candidate_plan.operator == "nearest"
                and not result.candidates
            )
            contradiction = bool(
                candidate_plan.action_bindings
                and not result.candidates
                and not missing_metric
            )
            hypotheses.append(
                QueryHypothesis(
                    hypothesis_id=f"query_mode_{index}_{name}",
                    plan=candidate_plan,
                    hard_valid_count=hard_valid_count,
                    contradiction=contradiction,
                    missing_metric=missing_metric,
                    binding_complete=bool(result.complete),
                    evidence_summary=[
                        f"target_mode={candidate_plan.target_binding_mode}",
                        f"target_kind={candidate_plan.target_kind or 'unknown'}",
                        f"candidate_count={hard_valid_count}",
                        "relations="
                        + ",".join(
                            f"{relation}:{count}"
                            for relation, count in sorted(relation_counts.items())
                        ),
                    ],
                    path_summary=[
                        "provenance="
                        + ",".join(
                            f"{source}:{count}"
                            for source, count in sorted(provenance_counts.items())
                        ),
                        f"binding_complete={bool(result.complete)}",
                    ],
                )
            )
        return hypotheses

    @staticmethod
    def _deterministic_hypothesis_selection(
        hypotheses: Sequence[QueryHypothesis],
    ) -> QueryHypothesisSelectionResult | None:
        viable = [
            item
            for item in hypotheses
            if item.hard_valid_count > 0
            and not item.contradiction
            and not item.missing_metric
        ]
        if len(viable) == 1:
            chosen = viable[0]
        else:
            complete = [item for item in viable if item.binding_complete]
            if len(complete) != 1:
                return None
            chosen = complete[0]
        return QueryHypothesisSelectionResult(
            status="selected",
            selected_hypothesis_id=chosen.hypothesis_id,
            rejected_hypothesis_ids=[
                item.hypothesis_id
                for item in hypotheses
                if item.hypothesis_id != chosen.hypothesis_id
            ],
            hypothesis_count=len(hypotheses),
            reason="deterministic_unique_supported_query_hypothesis",
        )

    def preflight(self, model_paths: Sequence[str | Path]) -> list[IndexBuildReport]:
        reports: list[IndexBuildReport] = []
        for model_path in sorted({Path(path).resolve() for path in model_paths}):
            report = self.index_manager.ensure_index(model_path, rebuild=self.rebuild_index)
            self.index_reports[model_path] = report
            if self.use_retrieval_prior and self.gnn_retriever is not None:
                self.gnn_artifact_reports[model_path] = (
                    validate_retriever_runtime(
                        self.gnn_retriever,
                        RetrievalRuntimeContext(report, self._backend(report))
                    ).artifact_report
                )
            reports.append(report)
        return reports

    def close(self) -> None:
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()

    def _backend(self, report: IndexBuildReport) -> IfcGraphBackend:
        backend = self._backends.get(report.index_path)
        if backend is None:
            backend = IfcGraphBackend(report.index_path, self.config.include_evidence)
            self._backends[report.index_path] = backend
        return backend

    def _budgeted(
        self,
        llm: ToGLlm,
        method: str,
        *args: object,
        call_limit: int | None = None,
        input_token_offset: int = 0,
        estimated_input_tokens: int | None = None,
    ) -> Any:
        max_calls = self.config.effective_max_llm_calls
        limit = max_calls if call_limit is None else min(max_calls, call_limit)
        if llm.calls >= limit:
            raise CallBudgetExceeded(
                f"ToG LLM call budget exhausted ({limit})"
            )
        input_budget = self.config.effective_input_token_budget
        if input_budget is not None:
            # The BAML collector exposes actual cumulative usage only after a
            # request.  Reserve a small prompt overhead and use the concrete
            # method arguments to avoid starting a request that plainly cannot
            # fit in the remaining per-question budget.
            estimated = (
                max(1, int(estimated_input_tokens))
                if estimated_input_tokens is not None
                else 256 + max(1, sum(len(repr(arg)) for arg in args) // 4)
            )
            consumed = llm.input_tokens + max(0, int(input_token_offset))
            if consumed + estimated > input_budget:
                raise CallBudgetExceeded(
                    "ToG input token budget exhausted "
                    f"({consumed}+~{estimated}>{input_budget})"
                )
        set_call_limit = getattr(llm, "set_call_limit", None)
        if callable(set_call_limit):
            set_call_limit(limit)
        try:
            result = getattr(llm, method)(*args)
            consumed = llm.input_tokens + max(0, int(input_token_offset))
            if input_budget is not None and consumed > input_budget:
                raise CallBudgetExceeded(
                    "ToG input token budget exhausted after call "
                    f"({consumed}>{input_budget})"
                )
            return result
        finally:
            if callable(set_call_limit):
                set_call_limit(max_calls)

    def _estimated_finalizer_input_tokens(
        self,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_trace: HierarchyReasoningTrace,
        evidence_review: EvidenceReview,
    ) -> int:
        """Estimate the serialized, capped finalizer prompt rather than raw objects.

        ``repr(hierarchy_context)`` includes every internal path, although the
        model adapter emits only the configured bounded prompt view.  Using the
        raw object caused a false 300k-token preflight rejection for a prompt
        that is capped to the lean-profile path/evidence budgets.
        """

        path_text = "\n".join(hierarchy_context.prompt_lines(100))[
            : self.config.effective_hierarchy_prompt_max_chars
        ]
        evidence_text = ""
        remaining = self.config.effective_prompt_evidence_max_chars
        for item in evidence[: self.config.effective_prompt_evidence_limit]:
            line = item.prompt_line()
            if remaining <= 0:
                break
            evidence_text += line[:remaining]
            remaining -= min(len(line), remaining)
        serialized_chars = sum(
            len(value)
            for value in (
                question,
                repr(plan),
                deterministic_answer,
                path_text,
                evidence_text,
                repr(reasoning_trace),
                repr(evidence_review),
            )
        )
        # Reserve room for the fixed system prompt, output contract, JSON
        # punctuation, and the approximation error of four characters/token.
        return 2_048 + max(1, serialized_chars // 4)

    @staticmethod
    def _usage_snapshot(llm: ToGLlm) -> tuple[int, int, int, int, int]:
        # Keep calls/input/output in the historical tuple positions because
        # several budget paths consume indexes 0–2 directly.
        return (
            int(getattr(llm, "calls", 0) or 0),
            int(getattr(llm, "input_tokens", 0) or 0),
            int(getattr(llm, "output_tokens", 0) or 0),
            int(getattr(llm, "cached_input_tokens", 0) or 0),
            int(getattr(llm, "reasoning_output_tokens", 0) or 0),
        )

    @staticmethod
    def _record_phase_usage(
        phase_token_usage: dict[str, dict[str, int]],
        phase: str,
        before: tuple[int, int, int, int, int],
        llm: ToGLlm,
    ) -> None:
        after = ToGSystem._usage_snapshot(llm)
        row = phase_token_usage.setdefault(
            phase,
            {
                "calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "total_tokens": 0,
            },
        )
        row.setdefault("cached_input_tokens", 0)
        row.setdefault("reasoning_output_tokens", 0)
        row["calls"] += max(0, after[0] - before[0])
        row["input_tokens"] += max(0, after[1] - before[1])
        row["output_tokens"] += max(0, after[2] - before[2])
        row["cached_input_tokens"] += max(0, after[3] - before[3])
        row["reasoning_output_tokens"] += max(0, after[4] - before[4])
        row["total_tokens"] = row["input_tokens"] + row["output_tokens"]

    @staticmethod
    def _remember_budget_error(errors: list[str], exc: CallBudgetExceeded) -> None:
        message = str(exc)
        if message not in errors:
            errors.append(message)

    def _explore(
        self,
        question: str,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        backend: IfcGraphBackend,
        llm: ToGLlm,
        errors: list[str],
        gnn_subgraph: GnnSubgraph | None = None,
        llm_call_limit: int | None = None,
        input_token_offset: int = 0,
    ) -> tuple[list[list[TripleEvidence]], list[TripleEvidence], list[EntityRef], int, int, int]:
        width = plan.tog_width if self.config.hierarchy_reasoning else self.config.width
        depth_limit = plan.tog_depth if self.config.hierarchy_reasoning else self.config.depth
        frontier = [seed for seed in seeds if seed.kind == "entity"][: self.config.retained_entities]
        encountered = list(frontier)
        visited = {entity.node_id for entity in frontier}
        visited_edges: set[tuple[str, str, str, str]] = set()
        chains: list[list[TripleEvidence]] = []
        all_evidence: list[TripleEvidence] = []
        depth_reached = 0

        for depth in range(1, depth_limit + 1):
            depth_reached = depth
            candidate_records: list[tuple[EntityRef, TripleEvidence]] = []
            for entity in frontier:
                relations = backend.relations(entity.node_id)
                if not relations:
                    continue
                try:
                    selected_relations = self._budgeted(
                        llm, "select_relations", question, entity, relations, width,
                        call_limit=llm_call_limit,
                        input_token_offset=input_token_offset,
                    )
                    if not selected_relations:
                        selected_relations = HeuristicToGLlm().select_relations(
                            question, entity, relations, width
                        )
                except CallBudgetExceeded as exc:
                    self._remember_budget_error(errors, exc)
                    selected_relations = HeuristicToGLlm().select_relations(
                        question, entity, relations, width
                    )
                except TimeoutError:
                    raise
                except Exception as exc:
                    errors.append(f"relation_selection:{type(exc).__name__}:{exc}")
                    selected_relations = HeuristicToGLlm().select_relations(
                        question, entity, relations, width
                    )

                if gnn_subgraph is not None and self.config.gnn_use_expansion:
                    selected_relations = self._rrf_relations(
                        entity.node_id,
                        selected_relations,
                        relations,
                        gnn_subgraph,
                        width,
                    )

                for relation in selected_relations:
                    preferred_ids: set[str] = set()
                    if gnn_subgraph is not None and self.config.gnn_use_expansion:
                        preferred_ids = {
                            edge.target_id
                            for edge in gnn_subgraph.edges
                            if edge.source_id == entity.node_id
                            and edge.relation == relation.name
                            and edge.direction == relation.direction
                        }
                    expanded = backend.expand(
                        entity,
                        relation,
                        self.config.candidate_cap,
                        preferred_node_ids=preferred_ids,
                    )
                    if not expanded:
                        continue
                    entities = [item[0] for item in expanded]
                    evidence_by_id = {item[0].node_id: item[1] for item in expanded}
                    try:
                        selected_entities = self._budgeted(
                            llm, "score_entities", question, relation, entities, width,
                            call_limit=llm_call_limit,
                            input_token_offset=input_token_offset,
                        )
                        if not selected_entities:
                            selected_entities = HeuristicToGLlm().score_entities(
                                question, relation, entities, width
                            )
                    except CallBudgetExceeded as exc:
                        self._remember_budget_error(errors, exc)
                        selected_entities = HeuristicToGLlm().score_entities(
                            question, relation, entities, width
                        )
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        errors.append(f"entity_scoring:{type(exc).__name__}:{exc}")
                        selected_entities = HeuristicToGLlm().score_entities(
                            question, relation, entities, width
                        )
                    if gnn_subgraph is not None and self.config.gnn_use_expansion:
                        selected_entities = self._rrf_entities(
                            entity.node_id,
                            relation,
                            selected_entities,
                            entities,
                            gnn_subgraph,
                            width,
                        )
                    for selected in selected_entities:
                        evidence = evidence_by_id.get(selected.node_id)
                        if evidence is None:
                            continue
                        evidence.relevance_score = selected.score
                        edge_key = (
                            evidence.source_id,
                            evidence.relation,
                            evidence.target_id,
                            evidence.direction,
                        )
                        if edge_key in visited_edges:
                            continue
                        visited_edges.add(edge_key)
                        candidate_records.append((selected, evidence))

            candidate_records.sort(key=lambda item: (-item[0].score, item[0].node_id))
            next_records: list[tuple[EntityRef, TripleEvidence]] = []
            for entity, evidence in candidate_records:
                if entity.kind == "entity" and entity.node_id in visited:
                    continue
                next_records.append((entity, evidence))
                if entity.kind == "entity":
                    visited.add(entity.node_id)
                if len(next_records) >= width:
                    break
            if not next_records:
                break
            depth_evidence = [record[1] for record in next_records]
            chains.append(depth_evidence)
            all_evidence.extend(depth_evidence)
            frontier = [record[0] for record in next_records if record[0].kind == "entity"]
            encountered.extend(frontier)
            if not frontier:
                break
            if plan.action_bindings:
                sufficient = backend.binding_complete(plan, seeds, encountered)
            else:
                try:
                    sufficient = bool(
                        self._budgeted(
                            llm, "is_sufficient", question, all_evidence,
                            call_limit=llm_call_limit,
                            input_token_offset=input_token_offset,
                        )
                    )
                except CallBudgetExceeded as exc:
                    self._remember_budget_error(errors, exc)
                    sufficient = HeuristicToGLlm().is_sufficient(question, all_evidence)
                except TimeoutError:
                    raise
                except Exception as exc:
                    errors.append(f"sufficiency:{type(exc).__name__}:{exc}")
                    sufficient = False
            if sufficient:
                break

        return chains, all_evidence, frontier, len(visited), len(visited_edges), depth_reached

    @staticmethod
    def _rrf_relations(
        node_id: str,
        llm_selected: Sequence[RelationRef],
        available: Sequence[RelationRef],
        subgraph: GnnSubgraph,
        width: int,
        k: int = 60,
    ) -> list[RelationRef]:
        by_key = {(item.name, item.direction): item for item in available}
        llm_rank = {(item.name, item.direction): rank for rank, item in enumerate(llm_selected, 1)}
        gnn_scored = []
        for item in available:
            key = f"{node_id}|{item.direction}|{item.name}"
            if key in subgraph.relation_support:
                gnn_scored.append((subgraph.relation_support[key], item.name, item.direction))
        gnn_scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        gnn_rank = {(name, direction): rank for rank, (_, name, direction) in enumerate(gnn_scored, 1)}
        candidates = set(llm_rank) | set(gnn_rank)
        if not candidates:
            return list(llm_selected)[:width]
        ranked: list[tuple[float, str, str]] = []
        trace: list[dict[str, object]] = []
        for name, direction in candidates:
            score = 0.0
            if (name, direction) in llm_rank:
                score += 1.0 / (k + llm_rank[(name, direction)])
            if (name, direction) in gnn_rank:
                score += 1.0 / (k + gnn_rank[(name, direction)])
            ranked.append((score, name, direction))
            trace.append(
                {
                    "node_id": node_id,
                    "relation": name,
                    "direction": direction,
                    "llm_rank": llm_rank.get((name, direction)),
                    "gnn_rank": gnn_rank.get((name, direction)),
                    "rrf_score": score,
                }
            )
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        subgraph.rrf_trace.setdefault("relations", []).extend(trace)
        result: list[RelationRef] = []
        for score, name, direction in ranked[:width]:
            item = by_key[(name, direction)]
            result.append(RelationRef(name, direction, score, item.gnn_score))
        return result

    @staticmethod
    def _rrf_entities(
        source_id: str,
        relation: RelationRef,
        llm_selected: Sequence[EntityRef],
        available: Sequence[EntityRef],
        subgraph: GnnSubgraph,
        width: int,
        k: int = 60,
    ) -> list[EntityRef]:
        by_id = {item.node_id: item for item in available}
        llm_rank = {item.node_id: rank for rank, item in enumerate(llm_selected, 1)}
        gnn_candidates = [
            item for item in available if item.node_id in subgraph.node_scores
        ]
        gnn_candidates.sort(
            key=lambda item: (-subgraph.node_scores[item.node_id], item.node_id)
        )
        gnn_rank = {item.node_id: rank for rank, item in enumerate(gnn_candidates, 1)}
        candidates = set(llm_rank) | set(gnn_rank)
        if not candidates:
            return list(llm_selected)[:width]
        ranked: list[tuple[float, str]] = []
        trace: list[dict[str, object]] = []
        for node_id in candidates:
            score = 0.0
            if node_id in llm_rank:
                score += 1.0 / (k + llm_rank[node_id])
            if node_id in gnn_rank:
                score += 1.0 / (k + gnn_rank[node_id])
            ranked.append((score, node_id))
            trace.append(
                {
                    "source_id": source_id,
                    "relation": relation.name,
                    "node_id": node_id,
                    "llm_rank": llm_rank.get(node_id),
                    "gnn_rank": gnn_rank.get(node_id),
                    "rrf_score": score,
                }
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        subgraph.rrf_trace.setdefault("entities", []).extend(trace)
        result: list[EntityRef] = []
        for score, node_id in ranked[:width]:
            entity = by_id[node_id]
            entity.score = score
            result.append(entity)
        return result

    @staticmethod
    def _dedupe_evidence(evidence: Sequence[TripleEvidence]) -> list[TripleEvidence]:
        result: list[TripleEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in evidence:
            key = (item.source_id, item.relation, item.target_id, item.direction)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _action_answer(
        plan: QueryPlan,
        entities: Sequence[EntityRef],
        seeds: Sequence[EntityRef],
        selection: TargetSelectionResult | None = None,
    ) -> str:
        bindings = list(plan.action_bindings or build_action_bindings(plan))
        if not bindings:
            return ""
        if (
            selection is not None
            and selection.reason == "ambiguous:multiple_single_scope_spaces"
        ):
            return ""
        selected_ids = set(selection.selected_ids) if selection else set()
        if selected_ids:
            # The constrained selector is authoritative.  Retrieval context
            # may contain useful support nodes or rejected alternatives, but
            # deterministic finalization must not silently re-introduce them.
            entities = [
                entity for entity in entities if entity.node_id in selected_ids
            ]

        spaces = [
            entity for entity in entities
            if entity.global_id
            and IfcGraphBackend.action_target_kind(entity) == "space"
        ]
        objects = [
            entity for entity in entities
            if entity.global_id
            and IfcGraphBackend.action_target_kind(entity) == "object"
        ]
        systems = [
            entity for entity in entities
            if entity.global_id
            and IfcGraphBackend.action_target_kind(entity) == "system"
        ]
        exact_seed_spaces = [
            seed for seed in seeds
            if seed.global_id and seed.score >= 0.9 and seed.ifc_class == "IfcSpace"
        ]
        exact_seed_objects = [
            seed for seed in seeds
            if seed.global_id
            and seed.score >= 0.9
            and IfcGraphBackend.action_target_kind(seed) == "object"
        ]
        exact_seed_systems = [
            seed for seed in seeds
            if seed.global_id
            and seed.score >= 0.9
            and IfcGraphBackend.action_target_kind(seed) == "system"
        ]
        allow_exact_seed_fallback = not (
            plan.operator in {"nearest", "unconnected"}
            or bool(plan.relation_references)
            or (
                selection is not None
                and not selection.selected_ids
                and selection.reason not in {"", "not_applicable"}
            )
        )

        output: list[str] = []
        for binding in bindings:
            action = binding.action
            candidates = {
                "space": spaces,
                "system": systems,
                "object": objects,
            }.get(binding.target_kind or "object", objects)
            if (
                allow_exact_seed_fallback
                and binding.target_kind == "space"
                and not candidates
            ):
                candidates = exact_seed_spaces
            elif (
                allow_exact_seed_fallback
                and binding.target_kind == "object"
                and not candidates
            ):
                candidates = exact_seed_objects
            elif (
                allow_exact_seed_fallback
                and binding.target_kind == "system"
                and not candidates
            ):
                candidates = exact_seed_systems
            targets = list(candidates)
            logical_group: LogicalTargetGroup | None = None
            if binding.target_unit == "logical_group":
                groups = [
                    group
                    for group in (selection.logical_groups if selection else [])
                    if group.binding_index == binding.binding_index
                ]
                if len(groups) != 1:
                    return ""
                logical_group = groups[0]
                expected_member_ids = set(logical_group.member_ids)
                targets = [
                    entity
                    for entity in targets
                    if entity.node_id in expected_member_ids
                ]
                if {
                    entity.node_id for entity in targets
                } != expected_member_ids:
                    return ""

            def binding_status(entity: EntityRef) -> str | None:
                matrix = entity.metadata.get("_constraint_status", {})
                if not isinstance(matrix, dict):
                    return None
                rows = matrix.get("bindings", [])
                if not isinstance(rows, list):
                    return None
                for row in rows:
                    if (
                        isinstance(row, dict)
                        and row.get("binding_index") == binding.binding_index
                    ):
                        return str(row.get("overall", "unknown")).lower()
                return None

            matrix_present = any(
                binding_status(entity) is not None for entity in targets
            )
            matrix_targets = [
                entity for entity in targets
                if binding_status(entity) == "pass"
                or binding.binding_index
                in (
                    entity.metadata.get(
                        "_positive_evidence_binding_indices", []
                    )
                    or []
                )
            ]
            if matrix_present:
                # Reuse the ontology-aware, morphology-aware constraint
                # decision made by the backend.  Re-checking raw substrings at
                # this point can turn a valid plural instruction into an empty
                # action slot (for example, plural query vs singular family).
                targets = matrix_targets
            else:
                # Compatibility path for callers that construct EntityRefs
                # without the per-binding constraint matrix.
                if binding.target_roles:
                    targets = [
                        entity for entity in targets
                        if IfcGraphBackend._semantic_role_match(
                            entity, binding.target_roles
                        )
                    ]
                if binding.target_names:
                    targets = [
                        entity for entity in targets
                        if any(
                            all(
                                token
                                in (
                                    f"{entity.label} "
                                    f"{' '.join(str(v) for v in entity.metadata.values())}"
                                ).lower()
                                for token in _tokens(name)
                            )
                            for name in binding.target_names
                        )
                    ]
                binding_domains = list(
                    getattr(binding, "target_domains", []) or []
                )
                if binding_domains:
                    targets = [
                        entity for entity in targets
                        if any(
                            IfcGraphBackend._semantic_domain_match(entity, domain)
                            for domain in binding_domains
                        )
                    ]
            if not targets:
                # One missing binding invalidates the whole ordered PDDL plan;
                # never let another action with the same verb mask it.
                return ""
            binding_policy = str(
                getattr(binding, "cardinality_policy", None)
                or getattr(plan, "cardinality_policy", "single")
            )
            if (
                binding.target_unit != "logical_group"
                and binding_policy
                not in {"all", "count", "distinct", "group_count"}
            ):
                # Roles and names are alternative constraints inside one
                # binding, not implicit requests for one action per synonym.
                # Multiple executable targets require ``all`` or independent
                # action bindings.
                targets = targets[:1]
            for entity in targets:
                rendered = f"{action}({entity.global_id})"
                if rendered not in output:
                    output.append(rendered)
        return ", ".join(output)

    def _merge_target_context(
        self,
        backend: IfcGraphBackend,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        operator_targets: Sequence[EntityRef],
        frontier: Sequence[EntityRef],
        subgraph: GnnSubgraph | None,
        *,
        require_bounded_retrieval_universe: bool = True,
    ) -> list[EntityRef]:
        candidate_universe: RetrievalCandidateUniverse | None = None
        if require_bounded_retrieval_universe and self.use_retrieval_prior:
            if subgraph is None:
                raise RuntimeError(
                    "v5 retrieval requires a bounded candidate universe"
                )
            candidate_universe = self._retrieval_candidate_universe(
                backend, seeds, subgraph, legal_expansion=frontier
            )
            subgraph.candidate_universe = candidate_universe
            subgraph.rrf_trace["candidate_universe"] = [
                candidate_universe.to_dict()
            ]
        allowed_target_ids = (
            set(candidate_universe.target_node_ids)
            if candidate_universe is not None
            else None
        )
        gnn_target_ids = {
            anchor.node_id for anchor in (subgraph.target_anchors if subgraph else [])
        }
        operator_pool: list[EntityRef] = []
        def annotate_gnn(entity: EntityRef) -> EntityRef:
            if subgraph is None:
                return entity
            rows = subgraph.candidate_evidence.get(entity.node_id, [])
            if rows:
                entity.metadata["_gnn_path_evidence"] = [
                    {
                        "binding_index": row.binding_index,
                        "path_status": row.path_status,
                        "node_ids": list(row.node_ids),
                        "relations": list(row.relations),
                        "constraint_states": dict(row.constraint_states),
                        "positive_constraints": list(row.positive_constraints),
                        "unknown_constraints": list(row.unknown_constraints),
                        "contradiction": row.contradiction,
                        "scope_connected": row.scope_connected,
                        "target_anchor_rank": row.target_anchor_rank,
                        "support_anchor_rank": row.support_anchor_rank,
                        "gnn_rank_score": row.gnn_rank_score,
                        "path_confidence": row.path_confidence,
                        "hub_exposure": row.hub_exposure,
                        "provenance": list(row.provenance),
                    }
                    for row in rows
                ]
            return entity

        for entity in operator_targets:
            if allowed_target_ids is not None and entity.node_id not in allowed_target_ids:
                continue
            # ``operator_targets`` is also used for the already-selected pool
            # during the later evidence merge.  Only candidates produced by
            # the deterministic operator may claim operator provenance;
            # otherwise a scope/GNN singleton could masquerade as a resolved
            # nearest/adjacency result.
            if entity.metadata.get("_deterministic_operator_candidate"):
                entity.match_reason = entity.match_reason or "operator"
                room = backend._authoritative_room(entity.node_id)
                room_id = str((room or {}).get("room_id") or "")
                if room_id and room_id in gnn_target_ids:
                    entity.metadata["_query_scope_support"] = (
                        "gnn_target_anchor"
                    )
            operator_pool.append(annotate_gnn(entity))
        def actionable_pool(entities: Sequence[EntityRef]) -> list[EntityRef]:
            # Missing topology is deliberately not filtered here.  It is an
            # ``unknown`` constraint for the selector, not proof that a GNN
            # candidate is wrong.  Only support-only graph nodes are excluded
            # before per-binding constraint evaluation.
            result: list[EntityRef] = []
            by_id: dict[str, EntityRef] = {}
            for entity in entities:
                if (
                    entity.kind != "entity"
                    or backend.action_target_kind(entity) == "none"
                ):
                    continue
                annotate_gnn(entity)
                provenance = entity.metadata.setdefault(
                    "_candidate_provenance", []
                )
                reason = entity.match_reason or "retrieval"
                if reason not in provenance:
                    provenance.append(reason)
                previous = by_id.get(entity.node_id)
                if previous is not None:
                    previous.score = max(previous.score, entity.score)
                    previous_provenance = previous.metadata.setdefault(
                        "_candidate_provenance", []
                    )
                    for item in provenance:
                        if item not in previous_provenance:
                            previous_provenance.append(item)
                    if (
                        "_gnn_path_evidence" in entity.metadata
                        and "_gnn_path_evidence" not in previous.metadata
                    ):
                        previous.metadata["_gnn_path_evidence"] = (
                            entity.metadata["_gnn_path_evidence"]
                        )
                    continue
                by_id[entity.node_id] = entity
                result.append(entity)
            return result

        # A constraint-complete operator result is already a closed target set
        # for the hierarchy-off arm.  GNN retrieval is a rescue/augmentation
        # path, not permission to replace a complete symbolic result with
        # semantically similar extras.  The hierarchy arm deliberately keeps
        # the larger pool because typed paths can independently validate and
        # close functional/system constraints before selection.
        if (
            (
                subgraph is None
                or not self.config.hierarchy_reasoning
                or not self._has_cross_level_constraints(plan)
            )
            and operator_pool
            and backend.binding_complete(plan, seeds, operator_pool)
        ):
            return actionable_pool(operator_pool)

        pool: list[EntityRef] = list(operator_pool)
        exact_ids = {
            seed.node_id for seed in seeds
            if seed.kind == "entity" and seed.score >= 0.9
        }
        for entity in frontier:
            if allowed_target_ids is not None and entity.node_id not in allowed_target_ids:
                continue
            if entity.node_id in exact_ids:
                entity.match_reason = entity.match_reason or "exact"
            else:
                entity.match_reason = entity.match_reason or "traversal"
            pool.append(entity)
        if any(binding.target_kind == "space" for binding in plan.action_bindings):
            pool.extend(
                entity
                for node_id in backend._room_scope_ids(plan, seeds)
                if (
                    (allowed_target_ids is None or node_id in allowed_target_ids)
                    and (entity := backend.get_node(node_id)) is not None
                )
            )
        if subgraph is not None and self.config.gnn_use_anchors:
            for anchor in subgraph.target_anchors:
                entity = backend.get_node(anchor.node_id)
                if entity is not None and (
                    allowed_target_ids is None or anchor.node_id in allowed_target_ids
                ):
                    entity.score = anchor.similarity
                    entity.match_reason = f"gnn_target_{anchor.level}"
                    pool.append(entity)
        if subgraph is not None and self.config.gnn_use_expansion:
            direct_anchor_ids = {anchor.node_id for anchor in subgraph.anchors}
            for node_id in sorted(
                subgraph.node_ids,
                key=lambda value: (-subgraph.node_scores.get(value, -1.0), value),
            ):
                if node_id in direct_anchor_ids:
                    continue
                if allowed_target_ids is not None and node_id not in allowed_target_ids:
                    continue
                entity = backend.get_node(node_id)
                if entity is not None:
                    entity.score = subgraph.node_scores.get(node_id, entity.score)
                    entity.match_reason = "gnn_support_expansion"
                    pool.append(entity)
        return actionable_pool(pool)

    def _select_target_candidates(
        self,
        backend: IfcGraphBackend,
        question: str,
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
        seeds: Sequence[EntityRef],
        llm: ToGLlm,
        errors: list[str],
        phase_token_usage: dict[str, dict[str, int]],
        phase_call_usage: dict[str, int],
        embedding_input_tokens: int = 0,
        allow_semantic_llm: bool = True,
    ) -> tuple[list[EntityRef], TargetSelectionResult]:
        exact_ids = {
            seed.node_id for seed in seeds
            if seed.kind == "entity" and seed.score >= 0.9
        }
        provenance_priority = {
            "exact": 0,
            "operator": 1,
            "gnn_target": 2,
            "traversal": 3,
            "hierarchy": 4,
            "gnn_support_expansion": 5,
            "repair": 6,
        }

        def raw_path_rows(entity: EntityRef) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            for key in ("_hierarchy_path_evidence", "_gnn_path_evidence"):
                value = entity.metadata.get(key, [])
                if isinstance(value, list):
                    rows.extend(
                        dict(item) for item in value if isinstance(item, dict)
                    )
            return rows

        path_priority = {
            "complete": 0,
            "partial": 1,
            "unsupported": 2,
            "contradictory": 3,
        }

        def priority(entity: EntityRef) -> tuple[int, int, int, float, str]:
            reason = entity.match_reason or ""
            source = next(
                (name for name in provenance_priority if reason.startswith(name)),
                "traversal",
            )
            if entity.node_id in exact_ids:
                source = "exact"
            path_rows = raw_path_rows(entity)
            best_path = min(
                (
                    path_priority.get(
                        str(row.get("path_status", "unsupported")), 2
                    )
                    for row in path_rows
                ),
                default=2,
            )
            contradiction = int(
                bool(path_rows)
                and all(bool(row.get("contradiction")) for row in path_rows)
            )
            return (
                contradiction,
                best_path if self.use_retrieval_prior else 2,
                provenance_priority[source],
                -float(entity.score),
                entity.node_id,
            )

        deduped: dict[str, EntityRef] = {}
        for entity in sorted(candidates, key=priority):
            previous = deduped.get(entity.node_id)
            if previous is None:
                deduped[entity.node_id] = entity
                continue
            previous.score = max(previous.score, entity.score)
            previous_provenance = previous.metadata.setdefault(
                "_candidate_provenance", []
            )
            for item in entity.metadata.get("_candidate_provenance", []):
                if item not in previous_provenance:
                    previous_provenance.append(item)
            for key in ("_hierarchy_path_evidence", "_gnn_path_evidence"):
                if key not in entity.metadata:
                    continue
                existing = previous.metadata.setdefault(key, [])
                for row in entity.metadata[key]:
                    if row not in existing:
                        existing.append(row)
        ranked = list(deduped.values())
        if not ranked:
            return [], TargetSelectionResult(
                candidate_count=0,
                reason="unsupported:no_hard_valid_candidates",
            )

        relational_request = plan.operator in {"nearest", "unconnected"} or any(
            getattr(reference, "relation", "")
            in {"nearest", "adjacent_to", "unconnected"}
            for reference in getattr(plan, "relation_references", [])
        )
        if relational_request:
            # A relation-constrained singleton is not closed merely because it
            # survived semantic filtering.  Require provenance from the graph
            # operator that actually proved distance, adjacency, or absence of
            # connectivity; GNN similarity and exact scope are context only.
            relation_closed = [
                entity
                for entity in ranked
                if entity.metadata.get("_deterministic_operator_candidate")
            ]
            if not relation_closed:
                return [], TargetSelectionResult(
                    rejected_ids=[entity.node_id for entity in ranked],
                    candidate_count=len(ranked),
                    fallback_used=True,
                    reason="unsupported:missing_metric_or_unique_adjacency",
                )
            ranked = relation_closed

        bindings = list(plan.action_bindings or build_action_bindings(plan))
        constraint_fn = getattr(backend, "candidate_constraint_status", None)
        for entity in ranked:
            rows: list[dict[str, object]] = []
            if callable(constraint_fn):
                for binding in bindings or [None]:
                    try:
                        row = constraint_fn(
                            plan, seeds, entity, binding=binding
                        )
                    except TypeError:
                        # Compatibility with experimental backends that have
                        # not yet added the optional per-binding argument.
                        try:
                            row = constraint_fn(plan, seeds, entity)
                        except TypeError:
                            row = constraint_fn(plan, entity, seeds)
                    if isinstance(row, dict):
                        repaired = dict(row)
                        constraints = dict(repaired.get("constraints") or {})
                        repaired["constraints"] = constraints
                        if constraints and all(
                            str(value).lower() == "pass"
                            for value in constraints.values()
                        ):
                            repaired["overall"] = "pass"
                        rows.append(repaired)
            overall_values = {
                str(row.get("overall", "unknown")).lower() for row in rows
            }
            overall = (
                "pass" if "pass" in overall_values
                else "unknown" if "unknown" in overall_values
                else "fail" if overall_values
                else "unknown"
            )
            matrix: dict[str, object] = {
                "overall": overall,
                "bindings": rows,
            }
            entity.metadata["_constraint_status"] = matrix

        def constraint_row(
            entity: EntityRef, binding: ActionTargetBinding | None = None
        ) -> dict[str, object]:
            matrix = entity.metadata.get("_constraint_status", {})
            if not isinstance(matrix, dict):
                return {}
            rows = matrix.get("bindings", [])
            if not isinstance(rows, list):
                return matrix
            if binding is None:
                return matrix
            binding_index = getattr(binding, "binding_index", None)
            for row in rows:
                if (
                    isinstance(row, dict)
                    and row.get("binding_index") == binding_index
                ):
                    return row
            return {}

        def statuses(
            entity: EntityRef, binding: ActionTargetBinding | None = None
        ) -> set[str]:
            matrix = constraint_row(entity, binding)
            result: set[str] = set()
            overall = matrix.get("overall")
            if overall is not None:
                result.add(str(overall).lower())
            constraints = matrix.get("constraints", {})
            if isinstance(constraints, dict):
                result.update(str(value).lower() for value in constraints.values())
            return result

        def path_rows(
            entity: EntityRef, binding: ActionTargetBinding | None = None
        ) -> list[dict[str, object]]:
            binding_index = getattr(binding, "binding_index", None)
            rows = raw_path_rows(entity)
            if binding is None:
                return rows
            return [
                row
                for row in rows
                if row.get("binding_index") == binding_index
            ]

        def best_path_status(
            entity: EntityRef, binding: ActionTargetBinding | None = None
        ) -> str:
            rows = path_rows(entity, binding)
            return min(
                (
                    str(row.get("path_status", "unsupported"))
                    for row in rows
                ),
                key=lambda value: path_priority.get(value, 2),
                default="unsupported",
            )

        def binding_matches(entity: EntityRef, binding: ActionTargetBinding) -> bool:
            # The backend owns ontology-aware role/domain/name normalization.
            # Re-applying raw-string equality here would discard its
            # low-confidence overrides and turn missing relations into false
            # contradictions.
            if "fail" in statuses(entity, binding):
                return False
            if entity.metadata.get("_hierarchy_path_required"):
                # A partial path is useful retrieval evidence, but it does not
                # prove an executable binding.  Keeping partial candidates in
                # the answer pool lets high-degree support neighborhoods crowd
                # out the few targets that actually connect the requested
                # scope/function/system to an actionable node.
                return best_path_status(entity, binding) == "complete"
            return best_path_status(entity, binding) != "contradictory"

        def requested_signal_keys(binding: ActionTargetBinding) -> set[str]:
            result = {"kind"}
            single_binding = len(bindings) <= 1
            if getattr(binding, "constraint_branches", None):
                result.add("semantic_branch")
            else:
                if getattr(binding, "target_roles", None) or (
                    single_binding and (plan.target_roles or plan.target_role)
                ):
                    result.add("role")
                if getattr(binding, "target_names", None) or (
                    single_binding
                    and any(
                        (
                            plan.target_names,
                            plan.target_name,
                            plan.target_family_terms,
                            plan.target_type_terms,
                            plan.target_keywords,
                        )
                    )
                ):
                    result.add("name")
                if getattr(binding, "target_domains", None) or (
                    single_binding and plan.target_domain
                ):
                    result.add("domain")
            if getattr(binding, "function_types", None) or (
                single_binding and plan.function_intents
            ):
                result.add("function")
            if getattr(binding, "system_categories", None):
                result.add("system")
            if plan.storey or plan.room or plan.room_names or plan.scope_predicates:
                result.add("scope")
            return result

        def positive_operator_unknown(
            entity: EntityRef, binding: ActionTargetBinding
        ) -> bool:
            """Recover operator-scoped unknowns without relabeling them pass."""
            row = constraint_row(entity, binding)
            constraints = row.get("constraints", {})
            if not isinstance(constraints, dict):
                return False
            requested = requested_signal_keys(binding)
            requested_statuses = {
                key: str(constraints.get(key, "unknown")).lower()
                for key in requested
            }
            if "fail" in requested_statuses.values():
                return False
            unknown_keys = {
                key for key, value in requested_statuses.items()
                if value == "unknown"
            }
            if "scope" not in unknown_keys:
                return False
            if not unknown_keys.issubset({"scope", "function", "system"}):
                return False
            positive_keys = {"name", "role", "domain", "function", "system"}
            if not any(
                requested_statuses.get(key) == "pass" for key in positive_keys
            ):
                return False
            # Unknown space typing may enter an exhaustive closure only when
            # the query independently identifies the scope.  Exact named
            # scopes, nested graph predicates, or a query-ranked space anchor
            # provide such support; an object-level semantic match alone does
            # not prove that its containing room has the requested type.
            exact_scope = bool(
                plan.room
                or plan.room_names
                or any(
                    predicate.predicate in {"space_name", "space_number"}
                    for predicate in plan.scope_predicates
                )
            )
            nested_scope = any(
                predicate.predicate
                in {
                    "contains_role",
                    "contains_domain",
                    "contains_name",
                    "space_function",
                }
                for predicate in plan.scope_predicates
            )
            if not (
                exact_scope
                or nested_scope
                or entity.metadata.get("_query_scope_support")
            ):
                return False
            provenance = entity.metadata.get("_operator_scope_provenance", {})
            return bool(
                isinstance(provenance, dict)
                and provenance.get("source") == "deterministic_scope_semijoin"
                and (entity.match_reason or "").startswith("operator")
            )

        def path_supported_unknown(
            entity: EntityRef, binding: ActionTargetBinding
        ) -> bool:
            """Promote missing BIM fields only with independent path evidence."""

            if "fail" in statuses(entity, binding):
                return False
            rows = [
                row
                for row in path_rows(entity, binding)
                if str(row.get("path_status")) == "complete"
                and bool(row.get("scope_connected", False))
                and not bool(row.get("contradiction", False))
            ]
            if not rows:
                return False
            requested = requested_signal_keys(binding)
            constraint = constraint_row(entity, binding).get("constraints", {})
            if not isinstance(constraint, dict):
                return False
            requested_states = {
                key: str(constraint.get(key, "unknown")).lower()
                for key in requested
            }
            if "fail" in requested_states.values():
                return False
            if (
                requested_states.get("scope") == "unknown"
                and not entity.metadata.get("_query_scope_support")
            ):
                # Containment in an arbitrary, unclassified space proves only
                # where the object is, not that the space has the requested
                # type/function.  Missing scope metadata therefore needs an
                # independent query-ranked scope anchor; otherwise hierarchy
                # expansion turns every contained semantic match into an
                # exhaustive extra target.
                return False
            positive_semantics = {
                "name", "role", "domain", "function", "system"
            }
            def has_positive_constraint(row: Mapping[str, Any]) -> bool:
                values = row.get("positive_constraints")
                return isinstance(values, (list, tuple, set)) and bool(
                    positive_semantics.intersection(str(value) for value in values)
                )

            return any(
                requested_states.get(key) == "pass"
                for key in positive_semantics
            ) or any(has_positive_constraint(row) for row in rows)

        # Close each action slot independently. Search breadth never changes
        # the number of targets required by a binding.
        hard_valid = [entity for entity in ranked if "fail" not in statuses(entity)]
        closed: list[EntityRef] = []
        logical_groups: list[LogicalTargetGroup] = []
        unresolved_bindings: list[tuple[ActionTargetBinding, list[EntityRef]]] = []
        relevant_ids: set[str] = set()
        positive_unknown_closed = False
        existential_closed = False

        indefinite_space_request = bool(
            re.search(
                r"\b(?:a|an|any|one)\s+(?:[a-z0-9_-]+\s+){0,3}"
                r"(?:room|space|area|zone)\b",
                question.casefold(),
            )
        )
        nested_scope_predicates = {
            "contains_role", "contains_domain", "contains_name", "space_function"
        }
        singular_nested_scope = bool(
            str(getattr(plan, "scope_cardinality", "single")) == "single"
            and any(
                predicate.predicate in nested_scope_predicates
                for predicate in plan.scope_predicates
            )
        )

        def candidate_scope_ids(entities: Sequence[EntityRef]) -> set[str]:
            result: set[str] = set()
            for entity in entities:
                if backend.action_target_kind(entity) == "space":
                    result.add(entity.node_id)
                    continue
                room = backend._authoritative_room(entity.node_id)
                if room and room.get("room_id"):
                    result.add(str(room["room_id"]))
            return result

        def branch_outcome(
            entity: EntityRef,
            binding: ActionTargetBinding,
            branch_index: int,
        ) -> str:
            """Read one conjunct's tri-state result from backend evidence."""

            row = constraint_row(entity, binding)
            evidence = row.get("evidence", {})
            if isinstance(evidence, dict):
                for value in evidence.get("semantic_branch", []) or []:
                    parts = str(value).split(":", 2)
                    if (
                        len(parts) == 3
                        and parts[0] == "branch"
                        and parts[1] == str(branch_index)
                    ):
                        return parts[2].lower()
            # Lightweight test/custom backends may omit detailed evidence.
            # Re-evaluate only this graph-backed branch with the same generic
            # ontology matcher; never infer a pass from candidate ordering.
            branches = list(getattr(binding, "constraint_branches", []) or [])
            if branch_index >= len(branches):
                return "fail"
            branch = branches[branch_index]
            branch_binding = replace(
                binding,
                target_roles=list(branch.get("roles", []) or []),
                target_names=list(branch.get("names", []) or []),
                target_domains=list(branch.get("domains", []) or []),
                constraint_branches=[],
                cardinality_policy=str(branch.get("cardinality", "single")),
            )
            return (
                "pass"
                if IfcGraphBackend._binding_matches(branch_binding, entity)
                else "fail"
            )
        if not bindings:
            if not hard_valid:
                return [], TargetSelectionResult(
                    rejected_ids=[entity.node_id for entity in ranked],
                    candidate_count=0,
                    reason="contradictory:no_hard_valid_candidates",
                )
            return hard_valid, TargetSelectionResult(
                selected_ids=[entity.node_id for entity in hard_valid],
                rejected_ids=[
                    entity.node_id for entity in ranked if entity not in hard_valid
                ],
                candidate_count=len(hard_valid),
                reason="deterministic_constraint_ranking",
            )
        for binding in bindings:
            matches = [
                entity for entity in ranked if binding_matches(entity, binding)
            ]
            relevant_ids.update(entity.node_id for entity in matches)
            policy = str(
                getattr(binding, "cardinality_policy", None)
                or getattr(plan, "cardinality_policy", "single")
            )
            passed = [
                entity
                for entity in matches
                if (
                    (
                        "unknown" not in statuses(entity, binding)
                        and "pass" in statuses(entity, binding)
                    )
                    or path_supported_unknown(entity, binding)
                )
            ]
            # A complete typed hierarchy path is independent positive
            # evidence for a candidate whose IFC metadata is incomplete.  The
            # final audit intentionally does not infer this waiver on its own,
            # so persist the binding-level decision here exactly as is done
            # for deterministic scope-semijoin recovery below.  Without this
            # contract selection could accept an unknown candidate and the
            # audit would subsequently relabel the same target as an extra.
            for entity in passed:
                row = constraint_row(entity, binding)
                if (
                    str(row.get("overall", "unknown")).lower() == "unknown"
                    and path_supported_unknown(entity, binding)
                ):
                    indices = entity.metadata.setdefault(
                        "_positive_evidence_binding_indices", []
                    )
                    binding_index = getattr(binding, "binding_index", None)
                    if binding_index not in indices:
                        indices.append(binding_index)
            unknown = len(passed) != len(matches)
            if (
                policy == "single"
                and len(passed) > 1
            ):
                operator_passed = [
                    entity
                    for entity in passed
                    if entity.metadata.get(
                        "_deterministic_operator_candidate"
                    )
                    and str(
                        constraint_row(entity, binding).get(
                            "overall", "unknown"
                        )
                    ).lower()
                    == "pass"
                ]
                if len(operator_passed) == 1:
                    # A unique, fully constraint-valid graph operator result
                    # is stronger than additional similarity/path candidates.
                    # Hierarchy and GNN evidence may corroborate it, but must
                    # not manufacture ambiguity for a single-cardinality slot.
                    passed = operator_passed
                    matches = operator_passed
                    unknown = False
            if (
                singular_nested_scope
                and not indefinite_space_request
                and len(candidate_scope_ids(passed or matches)) > 1
            ):
                # Evaluate ambiguity after the scope-to-target semi-join.  A
                # unique target can disambiguate several candidate spaces, but
                # targets spanning several singular scopes cannot be unioned
                # merely because retrieval found them all.
                return [], TargetSelectionResult(
                    rejected_ids=[entity.node_id for entity in ranked],
                    candidate_count=len(matches),
                    fallback_used=True,
                    reason="ambiguous:multiple_single_scope_spaces",
                )
            if not matches:
                return [], TargetSelectionResult(
                    rejected_ids=[entity.node_id for entity in ranked],
                    candidate_count=0,
                    reason="contradictory:required_binding_without_hard_valid_candidate",
                )
            branches = list(getattr(binding, "constraint_branches", []) or [])
            if branches:
                branch_closed: list[EntityRef] = []
                ambiguous_candidates: list[EntityRef] = []
                for branch_index, branch in enumerate(branches):
                    branch_passed = [
                        entity
                        for entity in matches
                        if branch_outcome(entity, binding, branch_index) == "pass"
                    ]
                    branch_unknown = [
                        entity
                        for entity in matches
                        if branch_outcome(entity, binding, branch_index) == "unknown"
                    ]
                    branch_policy = str(branch.get("cardinality", "single"))
                    if not branch_passed:
                        return [], TargetSelectionResult(
                            rejected_ids=[entity.node_id for entity in ranked],
                            candidate_count=len(matches),
                            reason=(
                                "unsupported:incomplete_coordinated_branch"
                                if branch_unknown
                                else "contradictory:missing_coordinated_branch"
                            ),
                        )
                    if branch_policy == "all":
                        if branch_unknown:
                            return [], TargetSelectionResult(
                                rejected_ids=[entity.node_id for entity in ranked],
                                candidate_count=len(matches),
                                reason="unsupported:incomplete_exhaustive_branch",
                            )
                        branch_closed.extend(branch_passed)
                    elif len(branch_passed) == 1:
                        branch_closed.extend(branch_passed)
                    else:
                        ambiguous_candidates.extend(branch_passed)
                closed.extend(branch_closed)
                if ambiguous_candidates:
                    unresolved_bindings.append(
                        (
                            binding,
                            list(
                                {
                                    entity.node_id: entity
                                    for entity in ambiguous_candidates
                                }.values()
                            ),
                        )
                    )
                continue
            if policy == "single" and len(passed) > 1:
                group_builder = getattr(
                    backend, "logical_target_groups", None
                )
                group_candidates = (
                    group_builder(plan, seeds, passed, binding)
                    if callable(group_builder)
                    else []
                )
                if self.use_retrieval_prior:
                    # ``authored_asset`` was introduced while inspecting Ecore
                    # failure cases and can silently reinterpret one requested
                    # entity as a multi-GUID answer.  The clean profile admits
                    # only schema-derived physical-unit concepts covered by
                    # dataset-independent synthetic fixtures.
                    group_candidates = [
                        group
                        for group in group_candidates
                        if group.group_kind
                        in {"continuous_surface", "asset_system"}
                    ]
                if len(group_candidates) == 1:
                    group = group_candidates[0]
                    member_ids = set(group.member_ids)
                    members = [
                        entity
                        for entity in passed
                        if entity.node_id in member_ids
                    ]
                    if {
                        entity.node_id for entity in members
                    } == member_ids:
                        binding.target_unit = "logical_group"
                        binding.allowed_logical_group_kinds = [
                            group.group_kind
                        ]
                        group_marker = getattr(
                            backend, "mark_logical_target_group", None
                        )
                        if callable(group_marker):
                            group_marker(group, members)
                        logical_groups.append(group)
                        closed.extend(members)
                        continue
            if policy in {"all", "count", "distinct", "group_count"}:
                if unknown:
                    unknown_matches = [
                        entity for entity in matches if entity not in passed
                    ]
                    recovered = [
                        entity for entity in unknown_matches
                        if positive_operator_unknown(entity, binding)
                    ]
                    if not passed and not recovered:
                        # With no constraint-complete or independently
                        # supported target, dropping unknowns would turn
                        # missing BIM information into a fabricated empty set.
                        return [], TargetSelectionResult(
                            rejected_ids=[entity.node_id for entity in ranked],
                            candidate_count=len(matches),
                            reason="unsupported:incomplete_exhaustive_binding",
                        )
                    for entity in recovered:
                        indices = entity.metadata.setdefault(
                            "_positive_evidence_binding_indices", []
                        )
                        binding_index = getattr(binding, "binding_index", None)
                        if binding_index not in indices:
                            indices.append(binding_index)
                    positive_unknown_closed = positive_unknown_closed or bool(recovered)
                    # Unsupported unknowns are excluded from the answer rather
                    # than promoted by role similarity.  They remain visible
                    # in diagnostics as rejected candidates.
                    closed.extend([*passed, *recovered])
                else:
                    closed.extend(passed)
            elif len(passed) == 1:
                # A single fully evidenced candidate outranks candidates whose
                # missing topology is merely recoverable.  Unknown evidence
                # cannot displace the one constraint-complete binding.
                closed.extend(passed)
            elif (
                len(passed) > 1
                and indefinite_space_request
                and getattr(binding, "target_kind", None) == "space"
                and str(getattr(plan, "scope_cardinality", "single")) == "single"
                and plan.operator not in {"nearest", "unconnected"}
            ):
                # "a/any room used for X" is existential, unlike "the room".
                # Choose a stable graph identity only among fully evidenced
                # candidates; candidate arrival order cannot affect the result.
                closed.append(min(passed, key=lambda entity: entity.node_id))
                existential_closed = True
            else:
                unresolved_bindings.append((binding, matches))

        if not unresolved_bindings:
            selected_by_id = {entity.node_id: entity for entity in closed}
            # Constraint validity is binding-scoped.  A target for one action
            # is expected to fail the role/name constraints of another action
            # in the same plan, so the union must not be filtered by the
            # entity-wide matrix status here.
            selected = [
                entity for entity in ranked
                if entity.node_id in selected_by_id
            ]
            return selected, TargetSelectionResult(
                selected_ids=[entity.node_id for entity in selected],
                rejected_ids=[
                    entity.node_id for entity in ranked
                    if entity.node_id not in selected_by_id
                ],
                candidate_count=len(relevant_ids),
                reason=(
                    "deterministic_logical_group_closure"
                    if logical_groups
                    else "deterministic_existential_closure"
                    if existential_closed
                    else "deterministic_positive_evidence_closure"
                    if positive_unknown_closed
                    else "deterministic_binding_closure"
                ),
                logical_groups=list(logical_groups),
            )

        if not self.config.effective_hybrid_target_selection:
            return [], TargetSelectionResult(
                candidate_count=len(relevant_ids),
                reason="ambiguous:resolver_disabled",
            )
        if not allow_semantic_llm:
            return [], TargetSelectionResult(
                candidate_count=len(relevant_ids),
                fallback_used=True,
                reason="budget:semantic_call_already_used",
            )

        resolver_ids = {
            entity.node_id
            for _binding, matches in unresolved_bindings
            for entity in matches
        }
        resolver_pool = [
            entity for entity in ranked if entity.node_id in resolver_ids
        ]
        resolver_count = len(resolver_pool)
        if resolver_count < 2:
            return [], TargetSelectionResult(
                rejected_ids=[entity.node_id for entity in ranked],
                candidate_count=resolver_count,
                fallback_used=True,
                reason="unsupported:insufficient_resolver_candidates",
            )
        if resolver_count > 20:
            return [], TargetSelectionResult(
                rejected_ids=[entity.node_id for entity in ranked],
                candidate_count=resolver_count,
                fallback_used=True,
                reason="unsupported:resolver_candidate_cap",
            )

        relational_ambiguity = plan.operator == "nearest" or any(
            getattr(reference, "relation", "") in {"nearest", "adjacent_to"}
            for reference in getattr(plan, "relation_references", [])
        )
        if relational_ambiguity:
            # Metric/adjacency closure belongs to deterministic graph
            # operators.  If they did not close the binding, language-model
            # preference is not substitute geometry evidence.
            return [], TargetSelectionResult(
                rejected_ids=[entity.node_id for entity in ranked],
                candidate_count=resolver_count,
                fallback_used=True,
                reason="unsupported:missing_metric_or_unique_adjacency",
            )

        def distinguishing_signature(
            entity: EntityRef, binding: ActionTargetBinding
        ) -> tuple[str, float, str]:
            """Return only graph evidence that could distinguish a candidate.

            Node identity, label spelling, and arrival order are deliberately
            excluded.  If two candidates have the same typed constraint and
            evidence rows, the resolver has no grounded basis for choosing one;
            another language-model call can only restate the ambiguity.
            """

            row = constraint_row(entity, binding)
            evidence = row.get("evidence", {})
            normalized_evidence: dict[str, list[str]] = {}
            if isinstance(evidence, dict):
                for key, values in sorted(evidence.items()):
                    raw_values = (
                        list(values)
                        if isinstance(values, (list, tuple, set))
                        else [values]
                    )
                    normalized_evidence[str(key)] = sorted(
                        str(value) for value in raw_values
                    )
            signature = json.dumps(
                {
                    "overall": row.get("overall", "unknown"),
                    "constraints": row.get("constraints", {}),
                    "evidence": normalized_evidence,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            provenance = (entity.match_reason or "").split(":", 1)[0]
            return signature, round(float(entity.score), 6), provenance

        has_distinguishing_evidence = any(
            len(
                {
                    distinguishing_signature(entity, binding)
                    for entity in matches
                }
            )
            > 1
            for binding, matches in unresolved_bindings
        )
        if not has_distinguishing_evidence:
            return [], TargetSelectionResult(
                rejected_ids=[entity.node_id for entity in ranked],
                candidate_count=resolver_count,
                fallback_used=True,
                reason="ambiguous:indistinguishable_constraint_evidence",
            )

        cap = min(self.config.target_selection_candidate_cap, 20, resolver_count)
        compact: list[EntityRef] = []
        payload_records: list[dict[str, object]] = []
        payload_cap = min(
            self.config.target_selection_max_chars,
            8_000,
            max(
                1_000,
                (
                    self.config.effective_semantic_input_token_budget
                    - 500
                )
                * 4,
            ),
        )
        unresolved_indices = {
            getattr(binding, "binding_index", None)
            for binding, _matches in unresolved_bindings
        }
        allowed_metadata = {
            # Label already carries the authored instance name.  Repeating
            # full name/long-name strings and raw evidence rows made a dozen
            # otherwise valid candidates exceed the resolver payload cap.
            # Retain only graph-backed discriminators needed by the bounded
            # selector; GUIDs and graph paths remain outside LLM control.
            "family", "type_name", "role", "domain",
            "storey", "space_type", "system_category", "action_target_kind",
            "function_type", "_constraint_status",
        }
        for entity in resolver_pool[:cap]:
            metadata: dict[str, object] = {}
            for key, value in entity.metadata.items():
                if key not in allowed_metadata or value in (None, ""):
                    continue
                if key == "_constraint_status" and isinstance(value, dict):
                    minimal_rows: list[dict[str, object]] = []
                    for row in value.get("bindings", []):
                        if not isinstance(row, dict) or row.get("binding_index") not in unresolved_indices:
                            continue
                        evidence = row.get("evidence", {})
                        minimal_rows.append(
                            {
                                "binding_index": row.get("binding_index"),
                                "overall": row.get("overall", "unknown"),
                                "constraints": row.get("constraints", {}),
                                # Evidence values are often long graph path
                                # identifiers.  The selector only needs to
                                # know which independent evidence channels
                                # exist; deterministic validation keeps the
                                # full values and remains authoritative.
                                "evidence_keys": sorted(
                                    str(key)
                                    for key, evidence_values in (
                                        evidence.items()
                                        if isinstance(evidence, dict)
                                        else []
                                    )
                                    if evidence_values
                                ),
                            }
                        )
                    metadata[key] = {
                        "overall": value.get("overall", "unknown"),
                        "bindings": minimal_rows,
                    }
                elif isinstance(value, str):
                    metadata[key] = value[:120]
                elif isinstance(value, (int, float, bool)):
                    metadata[key] = value
            clone = EntityRef(
                node_id=entity.node_id,
                label=entity.label[:120],
                global_id=entity.global_id,
                ifc_class=entity.ifc_class,
                kind=entity.kind,
                score=round(float(entity.score), 6),
                match_reason=entity.match_reason[:80],
                metadata=metadata,
            )
            record = {
                "node_id": clone.node_id,
                "label": clone.label,
                "global_id": clone.global_id,
                "ifc_class": clone.ifc_class,
                "score": clone.score,
                "provenance": clone.match_reason,
                "constraints": metadata.get("_constraint_status", {}),
                "features": {
                    key: value for key, value in metadata.items()
                    if key != "_constraint_status"
                },
            }
            proposed = [*payload_records, record]
            if len(json.dumps(proposed, ensure_ascii=False, default=str)) > payload_cap:
                break
            compact.append(clone)
            payload_records.append(record)
        if len(compact) != resolver_count:
            return [], TargetSelectionResult(
                rejected_ids=[entity.node_id for entity in ranked],
                candidate_count=resolver_count,
                fallback_used=True,
                reason="unsupported:resolver_payload_cap",
            )

        usage_start = self._usage_snapshot(llm)
        call_start = llm.calls
        try:
            resolver = getattr(llm, "resolve_targets", None)
            if not callable(resolver):
                raise RuntimeError("constrained target resolver is unavailable")
            resolution = self._budgeted(
                llm,
                "resolve_targets",
                question,
                plan,
                compact,
                call_limit=min(self.config.effective_max_llm_calls, llm.calls + 1),
                input_token_offset=embedding_input_tokens,
                estimated_input_tokens=(
                    self.config.effective_semantic_input_token_budget
                ),
            )
            semantic_delta = llm.input_tokens - usage_start[1]
            if semantic_delta > self.config.effective_semantic_input_token_budget:
                errors.append(
                    "candidate_selector:semantic input budget exceeded "
                    f"({semantic_delta}>"
                    f"{self.config.effective_semantic_input_token_budget})"
                )
                return [], TargetSelectionResult(
                    rejected_ids=[entity.node_id for entity in ranked],
                    candidate_count=resolver_count,
                    llm_used=True,
                    fallback_used=True,
                    reason="budget:semantic_resolver_input",
                )
            # The resolver is constrained to the compact payload it received;
            # it cannot resurrect a candidate hidden by the character cap.
            allowed = {entity.node_id: entity for entity in compact}
            chosen = [
                allowed[node_id]
                for node_id in resolution.selected_ids
                if node_id in allowed
            ]
            if chosen:
                resolved_selection = [*closed, *chosen]

                def resolver_completed_binding(binding: ActionTargetBinding) -> bool:
                    branches = list(
                        getattr(binding, "constraint_branches", []) or []
                    )
                    if not branches:
                        return (
                            sum(
                                1
                                for entity in chosen
                                if binding_matches(entity, binding)
                            )
                            == 1
                        )
                    for branch_index, branch in enumerate(branches):
                        branch_selected = [
                            entity
                            for entity in resolved_selection
                            if branch_outcome(entity, binding, branch_index) == "pass"
                        ]
                        policy = str(branch.get("cardinality", "single"))
                        if policy == "all":
                            expected = {
                                entity.node_id
                                for entity in ranked
                                if branch_outcome(entity, binding, branch_index)
                                == "pass"
                            }
                            if not expected or {
                                entity.node_id for entity in branch_selected
                            } != expected:
                                return False
                        elif len(branch_selected) != 1:
                            return False
                    return True

                if any(
                    not resolver_completed_binding(binding)
                    for binding, _matches in unresolved_bindings
                ):
                    return [], TargetSelectionResult(
                        selected_ids=[],
                        rejected_ids=[entity.node_id for entity in ranked],
                        candidate_count=resolver_count,
                        llm_used=True,
                        reason="ambiguous:resolver_incomplete_binding",
                    )
                chosen_ids = {
                    entity.node_id for entity in [*closed, *chosen]
                }
                selected = [
                    entity for entity in ranked
                    if entity.node_id in chosen_ids
                ]
                rejected = [
                    entity.node_id for entity in ranked
                    if entity.node_id not in chosen_ids
                ]
                return selected, TargetSelectionResult(
                    selected_ids=[entity.node_id for entity in selected],
                    rejected_ids=rejected,
                    candidate_count=resolver_count,
                    llm_used=True,
                    reason=resolution.reason,
                    logical_groups=list(logical_groups),
                )
            return [], TargetSelectionResult(
                selected_ids=[],
                rejected_ids=list(resolution.rejected_ids),
                candidate_count=resolver_count,
                llm_used=True,
                reason=resolution.reason,
            )
        except CallBudgetExceeded as exc:
            self._remember_budget_error(errors, exc)
            return [], TargetSelectionResult(
                selected_ids=[],
                rejected_ids=[],
                candidate_count=resolver_count,
                fallback_used=True,
                reason="budget:resolver_input",
            )
        except TimeoutError:
            raise
        except Exception as exc:
            errors.append(f"candidate_selector:{type(exc).__name__}:{exc}")
            # A failed resolver must not silently turn arbitrary graph order
            # into a valid PDDL target.
            return [], TargetSelectionResult(
                selected_ids=[],
                rejected_ids=[],
                candidate_count=resolver_count,
                fallback_used=True,
                reason="unsupported:resolver_failure",
            )
        finally:
            phase_call_usage["candidate_selector"] += llm.calls - call_start
            self._record_phase_usage(
                phase_token_usage, "candidate_selector", usage_start, llm
            )

    @staticmethod
    def _validate_pddl_answer(
        answer: str,
        deterministic_answer: str,
        plan: QueryPlan,
        evidence: Sequence[TripleEvidence],
        seeds: Sequence[EntityRef],
    ) -> tuple[str, str | None]:
        guid_pattern = re.compile(r"[0-9A-Za-z_$]{22}")
        answer_ids = set(guid_pattern.findall(answer))
        deterministic_ids = set(guid_pattern.findall(deterministic_answer))
        allowed_ids = {
            str(item.features["global_id"])
            for item in evidence
            if item.features.get("global_id")
        }
        allowed_ids.update(seed.global_id for seed in seeds if seed.global_id)
        if deterministic_ids:
            # The structured executor defines the in-scope action targets.  A
            # traversed seed can be valid graph evidence without itself being
            # an action target (for example, the reference room in a nearest
            # query).
            allowed_ids = set(deterministic_ids)
        exhaustive = any(
            str(
                getattr(binding, "cardinality_policy", None)
                or getattr(plan, "cardinality_policy", "single")
            ) in {"all", "count", "distinct", "group_count"}
            for binding in plan.action_bindings
        )
        if not answer_ids:
            return deterministic_answer, "pddl_answer_missing_guid"
        unknown = answer_ids - allowed_ids
        if unknown:
            return deterministic_answer, f"pddl_answer_hallucinated_guid:{sorted(unknown)}"
        if exhaustive and deterministic_ids - answer_ids:
            return deterministic_answer, "pddl_answer_omitted_exhaustive_targets"
        if not re.search(r"\b(?:Navigate|Nav|Inspect|Scan)\s*\(", answer, re.IGNORECASE):
            return deterministic_answer, "pddl_answer_missing_action"
        pair_pattern = re.compile(
            r"\b(Navigate|Nav|Inspect|Scan)\s*\(\s*([0-9A-Za-z_$]{22})\s*\)",
            re.IGNORECASE,
        )
        def normalize(value: str) -> str:
            return "navigate" if value.lower() == "nav" else value.lower()

        answer_pairs = [(normalize(action), guid) for action, guid in pair_pattern.findall(answer)]
        deterministic_pairs = [
            (normalize(action), guid)
            for action, guid in pair_pattern.findall(deterministic_answer)
        ]
        if any(pair not in deterministic_pairs for pair in answer_pairs):
            return deterministic_answer, "pddl_answer_action_target_binding_mismatch"
        if exhaustive and answer_pairs != deterministic_pairs:
            return deterministic_answer, "pddl_answer_action_order_or_binding_mismatch"
        deterministic_by_action: dict[str, set[str]] = defaultdict(set)
        for action, guid in pair_pattern.findall(deterministic_answer):
            key = normalize(action)
            deterministic_by_action[key].add(guid)
        for binding in plan.action_bindings:
            if not deterministic_by_action.get(binding.action.lower()):
                return deterministic_answer, f"pddl_missing_binding_target:{binding.action}"
        return answer, None

    @staticmethod
    def _reasoning_mode(
        question: str,
        plan: QueryPlan,
        deterministic_answer: str,
    ) -> ReasoningMode:
        lower = question.lower()
        relation_intent = bool(
            re.search(
                r"\b(?:adjacent|inside|within|contained|contains|part of|belongs|"
                r"connected|unconnected|serves|serving|near|nearest|closest)\b",
                lower,
            )
        )
        scoped_object = bool(
            plan.target_kind == "object"
            and (plan.room or plan.target_space_type)
        )
        hierarchy_required = bool(
            bool(plan.action_bindings)
            or relation_intent
            or scoped_object
            or plan.operator in {"path", "nearest", "unconnected", "argmax"}
        )
        has_deterministic = bool(deterministic_answer) and not deterministic_answer.startswith(
            "I cannot determine the answer from the IFC knowledge graph."
        )
        if hierarchy_required and has_deterministic:
            return "hybrid"
        if hierarchy_required:
            return "hierarchy"
        return "deterministic"

    @staticmethod
    def _has_cross_level_constraints(plan: QueryPlan) -> bool:
        """Whether a plan explicitly asks for evidence spanning graph levels."""
        nested_scope = any(
            predicate.predicate
            in {
                "space_function",
                "contains_role",
                "contains_domain",
                "contains_name",
                "argmax_area",
            }
            for predicate in plan.scope_predicates
        )
        functional_target = bool(plan.function_intents) or any(
            predicate.predicate in {"function", "system"}
            for predicate in plan.target_predicates
        )
        functional_binding = any(
            binding.function_types
            or binding.system_categories
            or binding.target_mode in {"system_entity", "system_members"}
            for binding in plan.action_bindings
        )
        scoped_nonspace_binding = bool(
            any(
                binding.target_kind != "space"
                for binding in plan.action_bindings
            )
            and (
                plan.room
                or plan.room_names
                or plan.target_space_type
                or any(
                    predicate.predicate
                    in {
                        "space_name",
                        "space_number",
                        "space_type",
                        "storey",
                    }
                    for predicate in plan.scope_predicates
                )
            )
        )
        relational = bool(plan.relation_references) or plan.operator in {
            "nearest",
            "unconnected",
            "argmax",
        }
        return bool(
            nested_scope
            or functional_target
            or functional_binding
            or scoped_nonspace_binding
            or relational
        )

    @classmethod
    def _hierarchy_binding_needed(
        cls,
        plan: QueryPlan,
        operator_result: OperatorResult | None,
    ) -> bool:
        """Route only bindings whose graph constraints need typed paths.

        A complete exact lookup already proves a simple action binding.  The
        hierarchy is useful when the symbolic operator is incomplete or when
        the instruction contains a functional, system, relational, or nested
        scope constraint whose evidence spans graph levels.
        """
        if not plan.action_bindings:
            return False
        if cls._has_cross_level_constraints(plan):
            return True

        # A direct space navigation/inspection is already executable once the
        # space entity is linked; walking through contained objects or
        # systems cannot add evidence for that binding.
        if all(
            binding.target_kind == "space"
            for binding in plan.action_bindings
        ):
            return False

        # Likewise, a unique graph-backed target mention (for example an IFC
        # tag) is stronger evidence than an auxiliary hierarchy path.
        if any(
            link.query_focus
            and link.kind in {"object", "system"}
            and len(link.node_ids) == 1
            and link.candidate_count == 1
            for link in plan.mention_links
        ):
            return False

        # Incomplete scoped object/system bindings are the principal rescue
        # case: hierarchy paths may bridge missing containment or membership.
        return operator_result is None or not operator_result.complete

    @staticmethod
    def _hierarchy_path_bindings_covered(
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
    ) -> bool:
        """Whether the current merged pool already completes every slot."""

        covered: set[int] = set()
        for entity in candidates:
            rows = entity.metadata.get("_hierarchy_path_evidence", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if (
                    isinstance(row, dict)
                    and row.get("path_status") == "complete"
                    and not row.get("contradiction")
                ):
                    covered.add(int(row.get("binding_index", 0)))
        return bool(plan.action_bindings) and all(
            int(binding.binding_index) in covered
            for binding in plan.action_bindings
        )

    def _hierarchy_context(
        self,
        backend: IfcGraphBackend,
        seeds: Sequence[EntityRef],
        selected_entities: Sequence[EntityRef],
        evidence: Sequence[TripleEvidence],
    ) -> HierarchyContext:
        node_ids: list[str] = []
        for entity in [*seeds, *selected_entities]:
            if entity.kind == "entity" and entity.node_id not in node_ids:
                node_ids.append(entity.node_id)
        for item in evidence:
            for node_id in (item.source_id, item.target_id):
                if (
                    node_id not in {"query", "literal"}
                    and not node_id.startswith("literal_")
                    and node_id not in node_ids
                ):
                    node_ids.append(node_id)
        context = backend.build_hierarchy_context(
            node_ids,
            max_paths=self.config.effective_hierarchy_max_paths,
            max_depth=self.config.hierarchy_max_depth,
        )
        control = self.config.hierarchy_path_control
        context.summary["path_control"] = control
        if control == "none":
            context.paths = []
            context.summary["control_note"] = (
                "Hierarchy paths withheld; target anchors and non-hierarchy "
                "evidence remain unchanged."
            )
        elif control == "shuffled":
            # Negative-control paths preserve node identities, path lengths,
            # prompt size, and ordering while replacing each typed relation by
            # a deterministic incompatible relation.  They are explicitly
            # marked non-authoritative and never enter backend validation.
            shuffled_relation = {
                "contains": "assigned_to_system",
                "part_of": "serves",
                "assigned_to_system": "contains",
                "requires_inspection_of": "part_of",
                "related_to_system": "contains",
                "serves": "requires_inspection_of",
            }
            controlled = copy.deepcopy(context.paths)
            for path in controlled:
                path.path_id = f"shuffled_control_{path.path_id}"
                for edge in path.edges:
                    edge.relation = shuffled_relation.get(
                        edge.relation, "contains" if edge.relation != "contains" else "serves"
                    )
                    edge.provenance = "shuffled_control"
                    edge.confidence = 0.0
            context.paths = controlled
            context.summary["control_note"] = (
                "Relation labels are deterministic shuffled-path controls; "
                "they are not graph facts and cannot validate targets."
            )
        context.prompt_max_paths = self.config.effective_hierarchy_max_paths
        context.prompt_max_chars = self.config.effective_hierarchy_prompt_max_chars
        if (
            len(context.prompt_lines()) < len(context.paths)
            or any(
                len(path.prompt_line()) > context.prompt_max_chars
                for path in context.paths[: context.prompt_max_paths]
            )
        ):
            context.truncated = True
            if "prompt_char_cap" not in context.truncation_reasons:
                context.truncation_reasons.append("prompt_char_cap")
        return context

    def _reason_and_review(
        self,
        llm: ToGLlm,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_mode: ReasoningMode,
        target_audit: TargetAudit,
        errors: list[str],
        phase_call_usage: dict[str, int] | None = None,
        phase_token_usage: dict[str, dict[str, int]] | None = None,
        phase_prefix: str = "",
        input_token_offset: int = 0,
    ) -> tuple[HierarchyReasoningTrace, EvidenceReview]:
        fallback = HeuristicToGLlm()
        if not self.config.effective_llm_hierarchy_review:
            trace = fallback.reason_hierarchy(
                question,
                plan,
                hierarchy_context,
                evidence,
                deterministic_answer,
                reasoning_mode,
            )
            review = fallback.review_evidence(
                question,
                plan,
                hierarchy_context,
                evidence,
                deterministic_answer,
                reasoning_mode,
                trace,
                target_audit,
            )
            review.target_validations = list(target_audit.target_validations)
            review.audited_target_count = len(target_audit.included_targets)
            review.action_bindings_covered = target_audit.action_bindings_covered
            review.conflicting_extras = target_audit.conflicting_extras
            return trace, review

        phase_start = llm.calls
        usage_start = self._usage_snapshot(llm)
        try:
            trace = self._budgeted(
                llm,
                "reason_hierarchy",
                question,
                plan,
                hierarchy_context,
                evidence,
                deterministic_answer,
                reasoning_mode,
                call_limit=min(self.config.effective_max_llm_calls, llm.calls + 1),
                input_token_offset=input_token_offset,
            )
        except TimeoutError:
            raise
        except Exception as exc:
            errors.append(f"hierarchy_reasoner:{type(exc).__name__}:{exc}")
            trace = fallback.reason_hierarchy(
                question,
                plan,
                hierarchy_context,
                evidence,
                deterministic_answer,
                reasoning_mode,
            )
        if phase_call_usage is not None:
            phase_call_usage[f"{phase_prefix}reasoner"] += llm.calls - phase_start
        if phase_token_usage is not None:
            self._record_phase_usage(
                phase_token_usage,
                "repair" if phase_prefix else "hierarchy",
                usage_start,
                llm,
            )
        phase_start = llm.calls
        usage_start = self._usage_snapshot(llm)
        try:
            review = self._budgeted(
                llm,
                "review_evidence",
                question,
                plan,
                hierarchy_context,
                evidence,
                deterministic_answer,
                reasoning_mode,
                trace,
                target_audit,
                call_limit=min(self.config.effective_max_llm_calls, llm.calls + 1),
                input_token_offset=input_token_offset,
            )
        except TimeoutError:
            raise
        except Exception as exc:
            errors.append(f"evidence_reviewer:{type(exc).__name__}:{exc}")
            review = fallback.review_evidence(
                question,
                plan,
                hierarchy_context,
                evidence,
                deterministic_answer,
                reasoning_mode,
                trace,
                target_audit,
            )
        if phase_call_usage is not None:
            phase_call_usage[f"{phase_prefix}reviewer"] += llm.calls - phase_start
        if phase_token_usage is not None:
            self._record_phase_usage(
                phase_token_usage,
                "repair" if phase_prefix else "hierarchy",
                usage_start,
                llm,
            )
        # Per-target validation is authoritative backend output.  The LLM
        # reviews intent and aggregate evidence but never controls list length.
        review.target_validations = list(target_audit.target_validations)
        review.audited_target_count = len(target_audit.included_targets)
        review.action_bindings_covered = target_audit.action_bindings_covered
        review.conflicting_extras = target_audit.conflicting_extras
        return trace, review

    @staticmethod
    def _review_allows_finalization(review: EvidenceReview) -> bool:
        backend_valid = bool(
            all(item.valid for item in review.target_validations)
            and review.action_bindings_covered
            and not review.conflicting_extras
        )
        intent_valid = bool(
            review.decision == "pass"
            and review.intent_aligned
            and review.reasoning_supported
            and review.deterministic_complete
            and not review.missing_requirements
            and all(item.covered for item in review.coverage)
        )
        hierarchy_valid = review.hierarchy_consistent or not review.hierarchy_required
        return backend_valid and intent_valid and hierarchy_valid

    @staticmethod
    def _target_audit_is_valid(audit: TargetAudit) -> bool:
        return bool(
            all(item.valid for item in audit.target_validations)
            and audit.action_bindings_covered
            and not audit.conflicting_extras
        )

    @staticmethod
    def _has_delta_repair_target(review: EvidenceReview) -> bool:
        """Only retrieve again when the audit names a missing binding/path.

        Extra-target closure and generic low-confidence reviews are selection
        problems, not evidence-retrieval problems; another broad graph walk is
        both expensive and unlikely to help them.
        """

        invalid_path_issue = any(
            issue.startswith("missing_") or issue == "required_relation_missing"
            for validation in review.target_validations
            for issue in validation.issues
        )
        return bool(
            review.missing_requirements
            or review.needed_relations
            or not review.action_bindings_covered
            or invalid_path_issue
        ) and not review.conflicting_extras

    @staticmethod
    def _closure_path_inventory(subgraph: GnnSubgraph) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source, paths in (
            ("support_to_target", subgraph.support_to_target_paths),
            ("subgraph", subgraph.paths),
        ):
            for path in paths:
                payload = asdict(path)
                basis = {
                    "source": source,
                    "target_id": payload.get("target_id"),
                    "binding_index": payload.get("binding_index"),
                    "scope_id": payload.get("scope_id"),
                    "node_ids": payload.get("node_ids", []),
                    "relations": payload.get("relations", []),
                }
                path_id = f"v5_path_{sha256_json(basis)[:20]}"
                if path_id in seen:
                    continue
                seen.add(path_id)
                payload.update(
                    {
                        "path_id": path_id,
                        "source": source,
                        "authoritative": True,
                        "provenance": "frozen_v5_real_graph",
                    }
                )
                rows.append(payload)
        # Some v5 artifacts retain complete binding paths only as candidate
        # evidence.  Preserve those rows without manufacturing new endpoints.
        for node_id, evidence_rows in sorted(subgraph.candidate_evidence.items()):
            for evidence in evidence_rows:
                if evidence.path_status != "complete" or evidence.contradiction:
                    continue
                basis = {
                    "source": "candidate_evidence",
                    "target_id": node_id,
                    "binding_index": evidence.binding_index,
                    "node_ids": list(evidence.node_ids),
                    "relations": list(evidence.relations),
                }
                path_id = f"v5_path_{sha256_json(basis)[:20]}"
                if path_id in seen:
                    continue
                seen.add(path_id)
                rows.append(
                    {
                        "path_id": path_id,
                        "target_id": node_id,
                        "binding_index": evidence.binding_index,
                        "scope_id": (
                            evidence.node_ids[0] if evidence.node_ids else None
                        ),
                        "node_ids": list(evidence.node_ids),
                        "relations": list(evidence.relations),
                        "score": float(evidence.path_confidence),
                        "complete_typed_path": True,
                        "scope_match_mode": (
                            "connected" if evidence.scope_connected else "unverified"
                        ),
                        "source": "candidate_evidence",
                        "authoritative": True,
                        "provenance": "frozen_v5_real_graph",
                    }
                )
        return rows

    def _build_retrieval_packet(
        self,
        *,
        question: str,
        plan: QueryPlan,
        retrieval_plan: QueryPlan,
        backend: IfcGraphBackend,
        report: IndexBuildReport,
        source_path: Path,
        seeds: Sequence[EntityRef],
        selected_entities: Sequence[EntityRef],
        traversal_evidence: Sequence[TripleEvidence],
        evidence: Sequence[TripleEvidence],
        gnn_subgraph: GnnSubgraph,
        target_audit: TargetAudit,
        target_selection: TargetSelectionResult,
        operator_result: OperatorResult | None,
        deterministic_answer: str,
        llm: ToGLlm,
        phase_call_usage: dict[str, int],
        phase_token_usage: dict[str, dict[str, int]],
        embedding_calls_total: int,
        embedding_tokens_total: int,
        embedding_cache_hits_total: int,
        chains: Sequence[Sequence[TripleEvidence]],
        visited_nodes: int,
        visited_edges: int,
        depth_reached: int,
        errors: Sequence[str],
    ) -> RetrievalEvidencePacketV1:
        if gnn_subgraph.candidate_universe is None:
            raise RuntimeError(
                "closure-adjudication-v3 requires a sealed retrieval candidate universe"
            )
        candidate_universe = gnn_subgraph.candidate_universe.to_dict()
        candidate_ids = list(candidate_universe.get("target_node_ids", []))
        candidate_entities: list[dict[str, Any]] = []
        for node_id in candidate_ids:
            entity = backend.get_node(str(node_id))
            if entity is not None:
                candidate_entities.append(asdict(entity))
        subgraph_payload = asdict(gnn_subgraph)
        subgraph_payload["candidate_entities"] = candidate_entities
        artifact_report = self.gnn_artifact_reports.get(source_path)
        gnn_artifact_hash = artifact_report.artifact_hash if artifact_report else ""
        lineage_hashes = {
            "ifc_source_sha256": report.source_hash,
            "graph_sha256": backend.graph_hash,
            "gnn_artifact_sha256": gnn_artifact_hash,
            "gnn_checkpoint_sha256": (
                artifact_report.checkpoint_hash if artifact_report else ""
            ),
            "query_plan_sha256": sha256_json(asdict(plan)),
            "retrieval_query_plan_sha256": sha256_json(asdict(retrieval_plan)),
        }
        upstream_errors, bounded_stop_reasons = _partition_closure_upstream_errors(
            errors,
            llm_calls=int(llm.calls),
            max_llm_calls=int(self.config.effective_max_llm_calls),
        )
        packet = RetrievalEvidencePacketV1(
            question=question,
            query_plan=asdict(plan),
            retrieval_query_plan=asdict(retrieval_plan),
            query_plan_sha256=lineage_hashes["query_plan_sha256"],
            retrieval_query_plan_sha256=lineage_hashes[
                "retrieval_query_plan_sha256"
            ],
            target_anchors=[asdict(item) for item in gnn_subgraph.target_anchors[:20]],
            support_anchors=[asdict(item) for item in gnn_subgraph.support_anchors[:20]],
            evaluation_anchors=[
                asdict(item) for item in gnn_subgraph.evaluation_anchors[:50]
            ],
            binding_candidate_ids={
                str(index): list(values)
                for index, values in gnn_subgraph.binding_candidate_ids.items()
            },
            traversal_evidence=[asdict(item) for item in traversal_evidence],
            evidence=[asdict(item) for item in evidence],
            typed_path_inventory=self._closure_path_inventory(gnn_subgraph),
            candidate_universe=candidate_universe,
            selected_entities=[asdict(item) for item in selected_entities],
            seed_entities=[asdict(item) for item in seeds],
            target_audit=asdict(target_audit),
            target_selection=asdict(target_selection),
            operator_result=asdict(operator_result) if operator_result else None,
            deterministic_answer=deterministic_answer,
            graph_hash=backend.graph_hash,
            graph_schema=backend.graph_schema,
            gnn_artifact_hash=gnn_artifact_hash,
            lineage_hashes=lineage_hashes,
            upstream_usage={
                "llm_calls": int(llm.calls),
                "input_tokens": int(llm.input_tokens),
                "output_tokens": int(llm.output_tokens),
                "cached_input_tokens": int(
                    getattr(llm, "cached_input_tokens", 0) or 0
                ),
                "reasoning_output_tokens": int(
                    getattr(llm, "reasoning_output_tokens", 0) or 0
                ),
                "embedding_calls": int(embedding_calls_total),
                "embedding_tokens": int(embedding_tokens_total),
                "embedding_cache_hits": int(embedding_cache_hits_total),
                "phase_call_usage": dict(phase_call_usage),
                "phase_token_usage": copy.deepcopy(phase_token_usage),
                "errors": upstream_errors,
                "bounded_stop_reasons": bounded_stop_reasons,
            },
            traversal_metrics={
                "visited_nodes": int(visited_nodes),
                "visited_edges": int(visited_edges),
                "depth_reached": int(depth_reached),
                "reasoning_chains": [
                    [asdict(item) for item in chain] for chain in chains
                ],
            },
            gnn_subgraph=subgraph_payload,
        ).seal()
        invariants = packet_arm_invariants(packet)
        if not all(
            bool(invariants[key])
            for key in (
                "packet_hash_valid",
                "same_path_count_typed_shuffled",
                "same_path_node_ids_typed_shuffled",
                "same_path_lengths_typed_shuffled",
                "shuffled_non_authoritative",
                "no_hierarchy_paths_withheld",
                "top20_top50_separated",
            )
        ):
            raise RuntimeError(f"retrieval packet invariant failed: {invariants}")
        self._retrieval_packet_runtime[packet.packet_sha256] = {
            "backend": backend,
            "report": report,
            "source_path": source_path,
            "gnn_subgraph": copy.deepcopy(gnn_subgraph),
            "chains": [list(chain) for chain in chains],
        }
        return packet

    def prepare_retrieval_packet(
        self, question: str, *, model_path: str | Path
    ) -> RetrievalEvidencePacketV1:
        """Prepare a sealed v5 retrieval packet from the raw question only."""

        if self.config.retrieval_flow_profile != "closure-adjudication-v3":
            raise ValueError(
                "prepare_retrieval_packet requires closure-adjudication-v3"
            )
        result = self._ask_impl(question, model_path, prepare_only=True)
        if not isinstance(result, RetrievalEvidencePacketV1):
            raise RuntimeError("retrieval packet preparation unexpectedly used legacy flow")
        return result

    def answer_retrieval_packet_v3(
        self,
        packet: RetrievalEvidencePacketV1,
        *,
        node_identities: dict[str, dict[str, Any]] | None = None,
        arm: RetrievalClosureArm = "typed_hierarchy",
    ) -> ToGResponse:
        """Replay one sealed packet through reviewed evidence-algebra v3.

        The first two calls compile intent and propose a closed group selection.
        A third, independent answering call may repair that proposal inside the
        same sealed group ledger.  Deterministic code remains the sole renderer
        of action names and IFC GUIDs.
        """

        if self.config.retrieval_flow_profile != "closure-adjudication-v3":
            raise ValueError(
                "answer_retrieval_packet_v3 requires closure-adjudication-v3"
            )
        if not packet.verify():
            raise ValueError("cannot replay an unsealed or modified retrieval packet")
        if arm not in {
            "no_hierarchy",
            "typed_hierarchy",
            "relation_shuffled_control",
        }:
            raise ValueError(f"unsupported closure-v3 hierarchy arm: {arm}")

        if node_identities is None:
            runtime = self._retrieval_packet_runtime.get(packet.packet_sha256, {})
            backend = runtime.get("backend")
            if isinstance(backend, IfcGraphBackend):
                ids = {
                    str(node_id)
                    for path in packet.typed_path_inventory
                    for node_id in list(path.get("node_ids") or [])
                }
                ids.update(
                    str(item.get("node_id") or "")
                    for item in list(packet.gnn_subgraph.get("candidate_entities") or [])
                )
                ids.update(
                    str(item.get("node_id") or "")
                    for item in list((packet.operator_result or {}).get("candidates") or [])
                )
                node_identities = {}
                for node_id in sorted(ids - {""}):
                    entity = backend.get_node(node_id)
                    if entity is not None:
                        node_identities[node_id] = asdict(entity)

        compact_plan = compact_query_plan_v2(packet.query_plan)
        compact_plan_json = canonical_json(compact_plan)
        llm = self.llm_factory()
        downstream_phase_tokens: dict[str, dict[str, int]] = {}
        errors = list(packet.upstream_usage.get("errors") or [])

        contract_start = llm.calls
        contract_usage_start = self._usage_snapshot(llm)
        contract_raw = self._budgeted(
            llm,
            "compile_retrieval_intent_contract_v3",
            contract_question_view(packet.question),
            compact_plan_json,
            call_limit=min(2, llm.calls + 1),
        )
        contract_calls = llm.calls - contract_start
        self._record_phase_usage(
            downstream_phase_tokens,
            "closure_v3_intent_contract",
            contract_usage_start,
            llm,
        )
        if not isinstance(contract_raw, dict):
            raise RuntimeError("v3 intent compiler returned a non-mapping result")
        contract = normalize_intent_contract_v3(contract_raw, compact_plan)
        ledger = build_candidate_group_ledger_v3(
            packet,
            contract,
            node_identities=node_identities or {},
            arm=arm,
        )
        contract_prompt_json = canonical_json(contract.prompt_payload())
        ledger_prompt_json = canonical_json(ledger.prompt_payload())

        adjudicator_start = llm.calls
        adjudicator_usage_start = self._usage_snapshot(llm)
        selection_raw = self._budgeted(
            llm,
            "adjudicate_and_answer_evidence_groups_v3",
            packet.question,
            contract_prompt_json,
            ledger_prompt_json,
            call_limit=min(2, llm.calls + 1),
        )
        adjudicator_calls = llm.calls - adjudicator_start
        self._record_phase_usage(
            downstream_phase_tokens,
            "closure_v3_group_adjudicator",
            adjudicator_usage_start,
            llm,
        )
        if not isinstance(selection_raw, dict):
            raise RuntimeError("v3 group adjudicator returned a non-mapping result")
        initial_selection = normalize_group_selection_v3(selection_raw, ledger)
        initial_selection_json = canonical_json(initial_selection.to_dict())

        reviewer_start = llm.calls
        reviewer_usage_start = self._usage_snapshot(llm)
        review_raw = self._budgeted(
            llm,
            "review_and_answer_evidence_groups_v3",
            packet.question,
            contract_prompt_json,
            ledger_prompt_json,
            initial_selection_json,
            call_limit=min(3, llm.calls + 1),
        )
        reviewer_calls = llm.calls - reviewer_start
        self._record_phase_usage(
            downstream_phase_tokens,
            "closure_v3_group_reviewer_answer",
            reviewer_usage_start,
            llm,
        )
        if not isinstance(review_raw, dict):
            raise RuntimeError("v3 group reviewer returned a non-mapping result")
        provider_review_selection = normalize_group_selection_v3(review_raw, ledger)
        selection = reconcile_group_selections_v3(
            packet,
            contract,
            ledger,
            initial_selection,
            provider_review_selection,
        )
        certificate = adjudicate_group_selection_v3(
            packet, contract, ledger, selection
        )
        answer = certified_answer_v3(contract, ledger, certificate)
        if certificate.validation_errors:
            errors.extend(
                f"closure_v3_validation:{item}"
                for item in certificate.validation_errors
            )

        candidate_by_id = {
            str(row["node_id"]): row for row in ledger.candidates
        }
        certified_order = list(
            dict.fromkeys(
                node_id
                for binding in contract.bindings
                for node_id in certificate.certified_target_ids.get(
                    str(int(binding["binding_index"])), []
                )
            )
        )
        selected_entities = [
            EntityRef(
                node_id=node_id,
                label=str(candidate_by_id[node_id]["facts"].get("label") or node_id),
                global_id=str(candidate_by_id[node_id].get("global_id") or "") or None,
                ifc_class=str(candidate_by_id[node_id]["facts"].get("ifc_class") or "") or None,
                kind=str(candidate_by_id[node_id]["facts"].get("kind") or "entity"),
                score=float(candidate_by_id[node_id].get("rrf_score") or 0.0),
                match_reason="closure-adjudication-v3-certified-group",
                metadata=copy.deepcopy(candidate_by_id[node_id]["facts"]),
            )
            for node_id in certified_order
            if node_id in candidate_by_id
        ]

        upstream_calls = int(packet.upstream_usage.get("llm_calls") or 0)
        upstream_input = int(packet.upstream_usage.get("input_tokens") or 0)
        upstream_output = int(packet.upstream_usage.get("output_tokens") or 0)
        upstream_cached = int(packet.upstream_usage.get("cached_input_tokens") or 0)
        upstream_reasoning = int(
            packet.upstream_usage.get("reasoning_output_tokens") or 0
        )
        phase_calls = dict(packet.upstream_usage.get("phase_call_usage") or {})
        phase_calls.update(
            {
                "closure_v3_intent_contract": contract_calls,
                "closure_v3_group_adjudicator": adjudicator_calls,
                "closure_v3_group_reviewer_answer": reviewer_calls,
                "closure_v3_downstream": (
                    contract_calls + adjudicator_calls + reviewer_calls
                ),
                "shared_upstream": upstream_calls,
                "total": upstream_calls + llm.calls,
            }
        )
        runtime = self._retrieval_packet_runtime.get(packet.packet_sha256, {})
        gnn_subgraph = runtime.get("gnn_subgraph")
        if not isinstance(gnn_subgraph, GnnSubgraph):
            gnn_subgraph = None
        return ToGResponse(
            answer=answer,
            variant=self.config.variant,
            operator=str(packet.query_plan.get("operator") or "lookup"),
            execution_profile=self.config.profile,
            retrieval_flow_profile="closure-adjudication-v3",
            reasoning_chains=[
                [TripleEvidence(**item) for item in chain]
                for chain in packet.traversal_metrics.get("reasoning_chains", [])
            ],
            seed_entities=[EntityRef(**item) for item in packet.seed_entities],
            evidence=[TripleEvidence(**item) for item in packet.evidence],
            visited_nodes=int(packet.traversal_metrics.get("visited_nodes") or 0),
            visited_edges=int(packet.traversal_metrics.get("visited_edges") or 0),
            depth_reached=int(packet.traversal_metrics.get("depth_reached") or 0),
            llm_calls=upstream_calls + llm.calls,
            input_tokens=upstream_input + int(llm.input_tokens),
            output_tokens=upstream_output + int(llm.output_tokens),
            cached_input_tokens=upstream_cached
            + int(getattr(llm, "cached_input_tokens", 0) or 0),
            reasoning_output_tokens=upstream_reasoning
            + int(getattr(llm, "reasoning_output_tokens", 0) or 0),
            graph_hash=packet.graph_hash,
            graph_schema=packet.graph_schema,
            gnn_artifact_hash=packet.gnn_artifact_hash,
            gnn_subgraph=gnn_subgraph,
            selected_entities=selected_entities,
            constraint_valid_node_ids=certified_order,
            binding_complete=certificate.closure_status == "pass",
            phase_call_usage=phase_calls,
            phase_token_usage={
                **copy.deepcopy(packet.upstream_usage.get("phase_token_usage") or {}),
                **downstream_phase_tokens,
            },
            gnn_embedding_calls_total=int(
                packet.upstream_usage.get("embedding_calls") or 0
            ),
            gnn_embedding_input_tokens_total=int(
                packet.upstream_usage.get("embedding_tokens") or 0
            ),
            reasoning_input_tokens_total=upstream_input + int(llm.input_tokens),
            system_input_tokens_total=upstream_input
            + int(packet.upstream_usage.get("embedding_tokens") or 0)
            + int(llm.input_tokens),
            repair_count=0,
            finalization_mode=(
                "closure_adjudication_v3_reviewed_certified_groups"
                if certificate.closure_status == "pass"
                else "closure_adjudication_v3_reviewed_grounded_abstention"
            ),
            validation_status=(
                "passed" if certificate.closure_status == "pass" else "failed"
            ),
            budget_exhausted=(upstream_calls + llm.calls) > self.config.max_llm_calls,
            budget_reason=(
                "deployment-equivalent call budget exceeded"
                if (upstream_calls + llm.calls) > self.config.max_llm_calls
                else ""
            ),
            budget_status=(
                "global_budget_exhausted"
                if (upstream_calls + llm.calls) > self.config.max_llm_calls
                else "ok"
                if certificate.closure_status == "pass"
                else "insufficient_graph_evidence"
            ),
            errors=errors,
            retrieval_evidence_packet=packet.to_dict(),
            evidence_closure_certificate=certificate.to_dict(),
            closure_repair_trace={
                "schema_version": "closure-selection-trace-v3",
                "contract_sha256": contract.contract_sha256,
                "ledger_sha256": ledger.ledger_sha256,
                "initial_selection_sha256": initial_selection.selection_sha256,
                "provider_review_selection_sha256": (
                    provider_review_selection.selection_sha256
                ),
                "selection_sha256": selection.selection_sha256,
                "repair_count": int(
                    initial_selection.binding_selections
                    != selection.binding_selections
                    or initial_selection.closure_supported
                    != selection.closure_supported
                ),
            },
            debug={
                "closure_profile": "closure-adjudication-v3",
                "closure_arm": arm,
                "intent_contract": contract.to_dict(),
                "candidate_group_ledger": ledger.to_dict(),
                "provider_contract": contract_raw,
                "provider_initial_group_selection": selection_raw,
                "normalized_initial_group_selection": initial_selection.to_dict(),
                "provider_group_selection": review_raw,
                "normalized_provider_review_selection": (
                    provider_review_selection.to_dict()
                ),
                "normalized_group_selection": selection.to_dict(),
                "answer_call_count": reviewer_calls,
                "compact_plan_chars": len(compact_plan_json),
                "contract_prompt_chars": len(contract_prompt_json),
                "candidate_group_prompt_chars": len(ledger_prompt_json),
                "deployment_equivalent_llm_calls": upstream_calls + llm.calls,
            },
        )

    def ask(self, question: str, *, model_path: str | Path) -> ToGResponse:
        """Answer a raw question without evaluator category or gold metadata."""

        result = self._ask_impl(question, model_path)
        if isinstance(result, RetrievalEvidencePacketV1):
            raise RuntimeError("public ask unexpectedly returned a preparation packet")
        return result

    def _ask_impl(
        self,
        question: str,
        model_path: str | Path,
        *,
        prepare_only: bool = False,
    ) -> ToGResponse | RetrievalEvidencePacketV1:
        # C1 is a direct IFC query/aggregation task.  It deliberately bypasses
        # query alignment, descriptor embedding, GNN retrieval, and expansion;
        # the deterministic backend below still compiles and executes its
        # exact operator.  C2--C4 retain the full retrieval/action-binding path.
        route_decision: RetrievalRouteDecision | None = None
        route_provider = self.retrieval_query_plan_provider
        frozen_route = getattr(route_provider, "route", None)
        route_decision = (
            frozen_route(question)
            if callable(frozen_route)
            else classify_query_intent(question)
        )
        requires_retrieval = route_decision.requires_retrieval
        if prepare_only and not requires_retrieval:
            raise ValueError("Retrieval packets are only available for retrieval-routed queries")
        # Query-only routing and the compiled QueryPlan own all model-visible
        # behavior. Evaluator category is not accepted by this public API.
        source_path = Path(model_path).resolve()
        report = self.index_reports.get(source_path)
        if report is None:
            report = self.index_manager.ensure_index(source_path, rebuild=self.rebuild_index)
            self.index_reports[source_path] = report
        backend = self._backend(report)
        if (
            requires_retrieval
            and
            self.use_retrieval_prior
            and source_path not in self.gnn_artifact_reports
            and self.gnn_retriever is not None
        ):
            self.gnn_artifact_reports[source_path] = (
                validate_retriever_runtime(
                    self.gnn_retriever,
                    RetrievalRuntimeContext(report, backend)
                ).artifact_report
            )
        llm = self.llm_factory()
        errors: list[str] = []
        phase_token_usage: dict[str, dict[str, int]] = {
            name: {
                "calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "total_tokens": 0,
            }
            for name in _PHASE_NAMES
        }
        phase_call_usage: dict[str, int] = defaultdict(
            int,
            {
                "planner": 0,
                "query_hypothesis": 0,
                "traversal": 0,
                "candidate_selector": 0,
                "reasoner": 0,
                "reviewer": 0,
                "repair_traversal": 0,
                "repair_reasoner": 0,
                "repair_reviewer": 0,
                "finalizer": 0,
            },
        )
        max_llm_calls = self.config.effective_max_llm_calls
        # The lean profile resolves one bounded candidate set and never enters
        # iterative graph repair. Missing topology is reported as unsupported
        # instead of spending more LLM calls on the same evidence.
        closure_flow = (
            requires_retrieval
            and self.config.retrieval_flow_profile == "closure-adjudication-v3"
        )
        repair_limit = (
            0
            if self.config.profile == "lean-grounding" or closure_flow
            else self.config.effective_hierarchy_max_repairs
        )
        reserved_calls = (
            3
            if closure_flow
            else 3 + 2 * repair_limit
            if self.config.hierarchy_reasoning
            and self.config.effective_llm_hierarchy_review
            and not self.config.effective_deterministic_finalization
            else 0
        )
        traversal_call_limit = (
            0
            if self.config.profile == "lean-grounding"
            else max(0, max_llm_calls - reserved_calls)
        )

        resolved_seeds = backend.resolve(question, self.config.retained_entities)
        fallback_plan = infer_query_plan(
            question,
            schema_context=self._planning_schema(backend),
        )
        phase_start = llm.calls
        usage_start = self._usage_snapshot(llm)
        use_planner_fallback = (
            requires_retrieval
            and self.config.semantic_enabled
            and self._plan_requires_fallback(fallback_plan)
        )
        if self.config.deterministic_planner and not use_planner_fallback:
            plan = fallback_plan
        else:
            try:
                proposed_plan = self._budgeted(
                    llm,
                    "plan_query",
                    question,
                    fallback_plan,
                    self.config.gnn_max_hops,
                    self.config.gnn_max_depth,
                    self.config.gnn_max_width,
                    call_limit=min(max_llm_calls, llm.calls + 1),
                )
                plan = merge_query_plan(
                    fallback_plan,
                    proposed_plan,
                    max_gnn_hops=self.config.gnn_max_hops,
                    max_tog_depth=self.config.gnn_max_depth,
                    max_tog_width=self.config.gnn_max_width,
                )
            except CallBudgetExceeded as exc:
                errors.append(str(exc))
                plan = fallback_plan
            except TimeoutError:
                raise
            except Exception as exc:
                errors.append(f"query_planning:{type(exc).__name__}:{exc}")
                plan = fallback_plan
        phase_call_usage["planner"] += llm.calls - phase_start
        self._record_phase_usage(phase_token_usage, "planner", usage_start, llm)

        query_hypotheses: list[QueryHypothesis] = []
        query_hypothesis_selection: QueryHypothesisSelectionResult | None = None
        if (
            requires_retrieval
            and self.config.bounded_query_hypotheses
            and self.config.variant in {"bim", "bim-gnn"}
        ):
            try:
                query_hypotheses = self._compile_query_hypotheses(
                    backend, plan, resolved_seeds
                )
                query_hypothesis_selection = (
                    self._deterministic_hypothesis_selection(query_hypotheses)
                )
                viable_hypotheses = [
                    item
                    for item in query_hypotheses
                    if item.hard_valid_count > 0
                    and not item.contradiction
                    and not item.missing_metric
                ]
                distinguishing_signatures = {
                    (
                        self._plan_interpretation_key(item.plan),
                        item.hard_valid_count,
                        item.binding_complete,
                        tuple(item.evidence_summary),
                        tuple(item.path_summary),
                    )
                    for item in viable_hypotheses
                }
                if (
                    query_hypothesis_selection is None
                    and self.config.semantic_enabled
                    and 2 <= len(viable_hypotheses)
                    <= self.config.effective_query_hypothesis_cap
                    and len(distinguishing_signatures) > 1
                    and llm.calls
                    < self.config.effective_semantic_llm_calls_per_question
                ):
                    phase_start = llm.calls
                    usage_start = self._usage_snapshot(llm)
                    try:
                        selected = self._budgeted(
                            llm,
                            "select_query_hypothesis",
                            question,
                            viable_hypotheses,
                            call_limit=min(
                                max_llm_calls,
                                llm.calls + 1,
                                self.config.effective_semantic_llm_calls_per_question,
                            ),
                            input_token_offset=0,
                            estimated_input_tokens=(
                                self.config.effective_semantic_input_token_budget
                            ),
                        )
                        query_hypothesis_selection = selected
                        semantic_delta = (
                            llm.input_tokens - usage_start[1]
                        )
                        if (
                            semantic_delta
                            > self.config.effective_semantic_input_token_budget
                        ):
                            errors.append(
                                "query_hypothesis:semantic input budget exceeded "
                                f"({semantic_delta}>"
                                f"{self.config.effective_semantic_input_token_budget})"
                            )
                            query_hypothesis_selection = (
                                QueryHypothesisSelectionResult(
                                    status="unsupported",
                                    hypothesis_count=len(viable_hypotheses),
                                    llm_used=True,
                                    fallback_used=True,
                                    reason="budget:semantic_hypothesis_input",
                                )
                            )
                    except CallBudgetExceeded as exc:
                        self._remember_budget_error(errors, exc)
                        query_hypothesis_selection = (
                            QueryHypothesisSelectionResult(
                                status="unsupported",
                                hypothesis_count=len(viable_hypotheses),
                                fallback_used=True,
                                reason="budget:semantic_hypothesis",
                            )
                        )
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        errors.append(
                            "query_hypothesis:"
                            f"{type(exc).__name__}:{exc}"
                        )
                        query_hypothesis_selection = (
                            QueryHypothesisSelectionResult(
                                status="unsupported",
                                hypothesis_count=len(viable_hypotheses),
                                fallback_used=True,
                                reason="unsupported:hypothesis_selector_failure",
                            )
                        )
                    finally:
                        phase_call_usage["query_hypothesis"] += (
                            llm.calls - phase_start
                        )
                        self._record_phase_usage(
                            phase_token_usage,
                            "query_hypothesis",
                            usage_start,
                            llm,
                        )
                if (
                    query_hypothesis_selection is not None
                    and query_hypothesis_selection.status == "selected"
                    and query_hypothesis_selection.selected_hypothesis_id
                ):
                    selected_plan = next(
                        (
                            item.plan
                            for item in query_hypotheses
                            if item.hypothesis_id
                            == query_hypothesis_selection.selected_hypothesis_id
                        ),
                        None,
                    )
                    if selected_plan is not None:
                        plan = selected_plan
            except TimeoutError:
                raise
            except Exception as exc:
                errors.append(
                    f"query_hypothesis_compile:{type(exc).__name__}:{exc}"
                )
                query_hypotheses = []
                query_hypothesis_selection = (
                    QueryHypothesisSelectionResult(
                        status="unsupported",
                        fallback_used=True,
                        reason="unsupported:hypothesis_compilation_failure",
                    )
                )
        if self.config.effective_traversal_max_llm_calls is not None:
            traversal_call_limit = min(
                traversal_call_limit,
                llm.calls + self.config.effective_traversal_max_llm_calls,
            )

        # A frozen v5 plan is a retrieval input, not a replacement for ToG's
        # planner.  Keep the core ``plan`` above for traversal and answering,
        # and materialize a separate immutable plan for the standalone v5
        # scorer.  Retrieval repairs intentionally reuse this exact plan so
        # the frozen ranking cannot drift with model-generated repair hints.
        retrieval_plan = plan
        if requires_retrieval and self.retrieval_query_plan_provider is not None:
            retrieval_plan = self.retrieval_query_plan_provider.plan(question, plan)

        gnn_subgraph: GnnSubgraph | None = None
        initial_gnn_subgraph: GnnSubgraph | None = None
        embedding_calls_total = 0
        embedding_tokens_total = 0
        embedding_cache_hits_total = 0
        if (
            requires_retrieval
            and self.use_retrieval_prior
            and self.gnn_retriever is not None
        ):
            try:
                gnn_subgraph = retrieve_subgraph(
                    self.gnn_retriever,
                    SubgraphRetrievalRequest(
                        question=question,
                        query_plan=retrieval_plan,
                        seed_entities=resolved_seeds,
                        graph_backend=backend,
                        expand=self.config.gnn_use_expansion,
                    )
                )
                gnn_subgraph.retrieval_stage = "initial"
                initial_gnn_subgraph = copy.deepcopy(gnn_subgraph)
                embedding_calls_total += int(gnn_subgraph.query_embedding_calls or 0)
                embedding_tokens_total += int(gnn_subgraph.query_embedding_tokens or 0)
                embedding_cache_hits_total += int(
                    gnn_subgraph.query_embedding_cache_hits or 0
                )
            except TimeoutError:
                raise
            except Exception as exc:
                raise RuntimeError("frozen v5 retrieval failed closed") from exc
        seeds = list(resolved_seeds)
        if (
            gnn_subgraph is not None
            and gnn_subgraph.target_anchors
            and self.config.gnn_use_anchors
        ):
            exact = [seed for seed in seeds if seed.score >= 0.9]
            remaining = [seed for seed in seeds if seed.score < 0.9]
            merged: list[EntityRef] = list(exact)
            seen_seed_ids = {seed.node_id for seed in merged}
            anchor_count = 0
            for anchor in gnn_subgraph.target_anchors:
                if anchor_count >= self.config.retained_entities:
                    break
                if anchor.node_id in seen_seed_ids:
                    continue
                entity = backend.get_node(anchor.node_id)
                if entity is None:
                    continue
                if (
                    plan.target_kind in {"space", "object", "system"}
                    and backend.action_target_kind(entity) != plan.target_kind
                ):
                    continue
                entity.score = anchor.similarity
                entity.match_reason = f"gnn_target_{anchor.level}"
                merged.append(entity)
                seen_seed_ids.add(entity.node_id)
                anchor_count += 1
            for seed in remaining:
                if len(merged) >= len(exact) + self.config.retained_entities:
                    break
                if seed.node_id not in seen_seed_ids:
                    merged.append(seed)
                    seen_seed_ids.add(seed.node_id)
            seeds = merged
        # Execute the exact operator before traversal so routing is based on
        # scoped IFC facts rather than on an already-spent beam budget.
        operator_evidence: list[TripleEvidence] = []
        operator_result: OperatorResult | None = None
        target_selection = TargetSelectionResult(reason="not_applicable")
        deterministic_answer = ""
        selected_entities = list(seeds)
        if self.config.variant in {"bim", "bim-gnn"}:
            operator_result = backend.execute_operator_result(plan, seeds)
            deterministic_answer = operator_result.answer
            operator_evidence = list(operator_result.evidence)
            selected_entities = list(operator_result.candidates)
            for entity in selected_entities:
                entity.metadata["_deterministic_operator_candidate"] = True
                entity.match_reason = entity.match_reason or "operator"
            if self.config.profile == "lean-grounding":
                # Candidate merging is part of grounding, not traversal.  It
                # must happen before the first audit so GNN/hierarchy-supported
                # targets cannot be discarded merely because an explicit IFC
                # relation is missing from the deterministic operator result.
                merged_initial = self._merge_target_context(
                    backend,
                    plan,
                    seeds,
                    selected_entities,
                    (),
                    gnn_subgraph,
                    require_bounded_retrieval_universe=requires_retrieval,
                )
                if merged_initial:
                    selected_entities = merged_initial
                # Hierarchy completes and validates executable target
                # bindings.  Attribute/aggregation questions have no PDDL
                # binding to complete, so expanding their GNN context through
                # four graph hops only adds latency and cannot change the
                # deterministic report answer.
                if (
                    self.config.hierarchy_reasoning
                    and gnn_subgraph is not None
                    and (
                        self._hierarchy_binding_needed(plan, operator_result)
                        or self.config.profile != "lean-grounding"
                    )
                    and not (
                        self.config.profile == "lean-grounding"
                        and operator_result is not None
                        and operator_result.complete
                        and len(operator_result.candidates)
                        > 4 * max(1, len(plan.action_bindings))
                        and backend.binding_complete(
                            plan, seeds, operator_result.candidates
                        )
                    )
                ):
                    # The GNN subgraph is already merged above.  Validate its
                    # best actionable targets first and invoke an additional
                    # graph expansion only as a rescue when at least one
                    # action slot still lacks a complete typed path.  This
                    # preserves hierarchy-guided completion without traversing
                    # the same high-degree support hubs twice on every query.
                    selected_entities = backend.assess_hierarchy_candidates(
                        plan,
                        seeds,
                        selected_entities,
                        max_paths_per_binding=min(
                            12, self.config.effective_hierarchy_max_paths
                        ),
                        max_depth=min(4, self.config.hierarchy_max_depth),
                    )
                    if not self._hierarchy_path_bindings_covered(
                        plan, selected_entities
                    ):
                        hierarchy_candidates = (
                            backend.expand_hierarchy_candidates(
                                plan,
                                seeds,
                                gnn_subgraph,
                                max_depth=min(
                                    4, self.config.hierarchy_max_depth
                                ),
                                max_candidates=min(
                                    4 * max(
                                        1, len(plan.action_bindings)
                                    ),
                                    self.config.gnn_max_subgraph_nodes,
                                ),
                                per_relation_quota=4,
                            )
                        )
                        selected_entities = self._merge_target_context(
                            backend,
                            plan,
                            seeds,
                            selected_entities,
                            hierarchy_candidates,
                            gnn_subgraph,
                            require_bounded_retrieval_universe=requires_retrieval,
                        )
                        selected_entities = (
                            backend.assess_hierarchy_candidates(
                                plan,
                                seeds,
                                selected_entities,
                                max_paths_per_binding=min(
                                    12,
                                    self.config.effective_hierarchy_max_paths,
                                ),
                                max_depth=min(
                                    4, self.config.hierarchy_max_depth
                                ),
                            )
                        )
        if plan.action_bindings:
            selected_entities, target_selection = self._select_target_candidates(
                backend,
                question,
                plan,
                selected_entities,
                seeds,
                llm,
                errors,
                phase_token_usage,
                phase_call_usage,
                embedding_input_tokens=embedding_tokens_total,
                allow_semantic_llm=(
                    self.config.semantic_enabled
                    and llm.calls
                    < self.config.effective_semantic_llm_calls_per_question
                ),
            )
        if plan.action_bindings:
            action_answer = self._action_answer(
                plan,
                selected_entities,
                seeds,
                target_selection,
            )
            if action_answer:
                deterministic_answer = action_answer
            elif plan.action_bindings:
                deterministic_answer = ""
        preliminary_mode = self._reasoning_mode(question, plan, deterministic_answer)
        deterministic_complete = bool(
            deterministic_answer
        ) and not deterministic_answer.startswith(
            "I cannot determine the answer from the IFC knowledge graph."
        )
        preliminary_audit: TargetAudit | None = None
        if (
            self.config.hierarchy_reasoning
            or self.config.profile == "lean-grounding"
        ) and self.config.variant in {"bim", "bim-gnn"}:
            preliminary_audit = backend.target_audit(
                plan, seeds, selected_entities, excluded_limit=0
            )
        deterministic_short_circuit = bool(
            not self.config.llm_answering_required
            and preliminary_audit is not None
            and (
                (
                    not plan.action_bindings
                    and plan.operator in {
                        "lookup",
                        "list",
                        "distinct",
                        "count",
                        "group_count",
                        "argmax",
                        "all_matching",
                        "unconnected",
                    }
                    and not plan.action_bindings
                    and operator_result is not None
                    and operator_result.complete
                )
                or (
                    self.config.profile == "lean-grounding"
                    and bool(plan.action_bindings)
                    and bool(plan.action_bindings)
                )
            )
            and deterministic_complete
            and (
                not plan.action_bindings
                or self._target_audit_is_valid(preliminary_audit)
            )
        )

        phase_start = llm.calls
        usage_start = self._usage_snapshot(llm)
        # The paper-style ToG family must always execute graph traversal.  A
        # deterministic operator result may inform the final answer, but it is
        # not a substitute for relation selection, graph expansion, and entity
        # pruning.  Only the explicitly separate lean-grounding profile (or an
        # opt-in non-LLM deterministic short circuit) may bypass exploration.
        chains: list[list[TripleEvidence]]
        traversal_evidence: list[TripleEvidence]
        frontier: list[EntityRef]
        if self.config.profile == "lean-grounding" or deterministic_short_circuit:
            chains, traversal_evidence, frontier = [], [], list(selected_entities or seeds)
            visited_nodes = len({entity.node_id for entity in frontier})
            visited_edges = depth_reached = 0
        else:
            (
                chains,
                traversal_evidence,
                frontier,
                visited_nodes,
                visited_edges,
                depth_reached,
            ) = self._explore(
                question,
                plan,
                seeds,
                backend,
                llm,
                errors,
                gnn_subgraph,
                llm_call_limit=(
                    traversal_call_limit if self.config.hierarchy_reasoning else None
                ),
                input_token_offset=embedding_tokens_total,
            )
        phase_call_usage["traversal"] += llm.calls - phase_start
        self._record_phase_usage(phase_token_usage, "traversal", usage_start, llm)

        if self.config.variant == "canonical" and traversal_evidence:
            deterministic_answer = ", ".join(
                dict.fromkeys(item.target_label for item in traversal_evidence[-self.config.width :])
            )
            selected_entities = list(frontier or seeds)

        if self.config.variant in {"bim", "bim-gnn"}:
            merged_targets = self._merge_target_context(
                backend,
                plan,
                seeds,
                selected_entities,
                frontier,
                gnn_subgraph,
                require_bounded_retrieval_universe=requires_retrieval,
            )
            if merged_targets:
                # A complete operator result followed by a constrained
                # selection is a closed target set.  Traversal/GNN context may
                # support that decision, but must not silently re-introduce
                # candidates the selector rejected.
                closed_ids = set(target_selection.selected_ids)
                closed_targets = [
                    entity for entity in merged_targets
                    if entity.node_id in closed_ids
                ]
                if plan.action_bindings and self.config.profile == "lean-grounding":
                    # The lean selector has already seen deterministic, exact,
                    # GNN and hierarchy-supported candidates.  Its empty result
                    # is a reason-coded abstention, not permission to promote
                    # every context node into a PDDL target during the later
                    # evidence merge.
                    selected_entities = closed_targets
                else:
                    selected_entities = closed_targets or merged_targets
            if (
                bool(plan.action_bindings)
                and len(selected_entities) > 1
                and not (
                    operator_result is not None
                    and operator_result.complete
                    and target_selection.selected_ids
                )
                and target_selection.candidate_count <= 1
            ):
                selected_entities, target_selection = self._select_target_candidates(
                    backend,
                    question,
                    plan,
                    selected_entities,
                    seeds,
                    llm,
                    errors,
                    phase_token_usage,
                    phase_call_usage,
                    embedding_input_tokens=embedding_tokens_total,
                    allow_semantic_llm=(
                        self.config.semantic_enabled
                        and llm.calls
                        < self.config.effective_semantic_llm_calls_per_question
                    ),
                )

        evidence = self._dedupe_evidence([*traversal_evidence, *operator_evidence])
        audited_nodes = {
            node_id
            for item in evidence
            for node_id in (item.source_id, item.target_id)
            if node_id not in {"query", "literal"} and not node_id.startswith("literal_")
        }
        audited_nodes.update(seed.node_id for seed in seeds if seed.kind == "entity")
        visited_nodes = max(visited_nodes, len(audited_nodes))
        visited_edges = max(visited_edges, len(evidence))
        if plan.action_bindings:
            action_answer = self._action_answer(
                plan,
                selected_entities,
                seeds,
                target_selection,
            )
            if action_answer:
                deterministic_answer = action_answer
            elif plan.action_bindings:
                deterministic_answer = ""
        if not deterministic_answer:
            deterministic_answer = "I cannot determine the answer from the IFC knowledge graph."

        hierarchy_context: HierarchyContext | None = None
        hierarchy_validation: HierarchyValidation | None = None
        reasoning_mode: ReasoningMode | None = None
        reasoning_trace: HierarchyReasoningTrace | None = None
        evidence_review: EvidenceReview | None = None
        repair_count = 0
        finalization_mode = "legacy"
        target_audit = (
            preliminary_audit
            if deterministic_short_circuit and preliminary_audit is not None
            else backend.target_audit(
                plan,
                seeds,
                selected_entities,
                excluded_limit=self.config.effective_audit_excluded_limit,
            )
        )
        if closure_flow:
            if gnn_subgraph is None:
                raise RuntimeError(
                    "retrieval closure requires a completed frozen v5 retrieval"
                )
            packet = self._build_retrieval_packet(
                question=question,
                plan=plan,
                retrieval_plan=retrieval_plan,
                backend=backend,
                report=report,
                source_path=source_path,
                seeds=seeds,
                selected_entities=selected_entities,
                traversal_evidence=traversal_evidence,
                evidence=evidence,
                gnn_subgraph=gnn_subgraph,
                target_audit=target_audit,
                target_selection=target_selection,
                operator_result=operator_result,
                deterministic_answer=deterministic_answer,
                llm=llm,
                phase_call_usage=phase_call_usage,
                phase_token_usage=phase_token_usage,
                embedding_calls_total=embedding_calls_total,
                embedding_tokens_total=embedding_tokens_total,
                embedding_cache_hits_total=embedding_cache_hits_total,
                chains=chains,
                visited_nodes=visited_nodes,
                visited_edges=visited_edges,
                depth_reached=depth_reached,
                errors=errors,
            )
            if prepare_only:
                return packet
            arm = {
                "none": "no_hierarchy",
                "full": "typed_hierarchy",
                "shuffled": "relation_shuffled_control",
            }[self.config.hierarchy_path_control]
            if arm != "typed_hierarchy":
                raise ValueError(
                    "closure adjudication is isolated to the typed ToG-Hierarchy arm"
                )
            return self.answer_retrieval_packet_v3(packet)
        validation_status = "not_run"
        if self.config.hierarchy_reasoning and (
            self._hierarchy_binding_needed(plan, operator_result)
            or self.config.profile != "lean-grounding"
        ):
            hierarchy_validation = backend.validate_hierarchy(
                plan,
                seeds,
                selected_entities,
                max_paths=self.config.effective_hierarchy_max_paths,
                max_depth=self.config.hierarchy_max_depth,
            )
        deterministic_complete = bool(deterministic_answer) and not deterministic_answer.startswith(
            "I cannot determine the answer from the IFC knowledge graph."
        )

        if deterministic_short_circuit:
            answer = deterministic_answer
            reasoning_mode = "deterministic"
            finalization_mode = "deterministic"
            validation_status = "passed"
        elif self.config.hierarchy_reasoning and (
            self._hierarchy_binding_needed(plan, operator_result)
            or self.config.profile != "lean-grounding"
        ):
            hierarchy_context = self._hierarchy_context(
                backend, seeds, selected_entities, evidence
            )
            reasoning_mode = preliminary_mode
            reasoning_trace, evidence_review = self._reason_and_review(
                llm,
                question,
                plan,
                hierarchy_context,
                evidence,
                deterministic_answer,
                reasoning_mode,
                target_audit,
                errors,
                phase_call_usage,
                phase_token_usage,
                input_token_offset=embedding_tokens_total,
            )

            while (
                not self._review_allows_finalization(evidence_review)
                and target_selection.reason
                != "ambiguous:multiple_single_scope_spaces"
                and not (
                    bool(plan.action_bindings)
                    and self._target_audit_is_valid(target_audit)
                    and deterministic_answer
                    and not deterministic_answer.startswith(
                        "I cannot determine the answer from the IFC knowledge graph."
                    )
                )
                and evidence_review.decision != "abstain"
                and reasoning_mode != "deterministic"
                and repair_count < repair_limit
                and (
                    self.config.profile != "lean-grounding"
                    or self._has_delta_repair_target(evidence_review)
                )
            ):
                repair_count += 1
                repair_plan = (
                    plan
                    if self.config.profile == "lean-grounding"
                    else replace(
                        plan,
                        gnn_hops=min(self.config.gnn_max_hops, plan.gnn_hops + 1),
                        tog_depth=min(self.config.gnn_max_depth, plan.tog_depth + 1),
                        tog_width=min(self.config.gnn_max_width, plan.tog_width + 1),
                    )
                )
                hints = [
                    *evidence_review.missing_requirements,
                    *(f"relation={item}" for item in evidence_review.needed_relations),
                    *(
                        f"target={validation.target_id}:{issue}"
                        for validation in evidence_review.target_validations
                        for issue in validation.issues
                    ),
                ]
                if not evidence_review.action_bindings_covered:
                    hints.append("missing action binding target")
                if evidence_review.conflicting_extras:
                    hints.append("remove conflicting extra targets")
                # Delta repair receives only the missing slots/relations.  It
                # must not inherit an unbounded review transcript.
                hints = list(dict.fromkeys(str(item) for item in hints if item))[:12]
                repair_question = question
                if hints:
                    repair_question += "\nGrounding repair targets: " + "; ".join(hints)
                if self.config.profile == "lean-grounding":
                    repair_question = repair_question[:2_000]

                repair_subgraph = gnn_subgraph
                if self.use_retrieval_prior and self.gnn_retriever is not None:
                    try:
                        repair_subgraph = retrieve_subgraph(
                            self.gnn_retriever,
                            SubgraphRetrievalRequest(
                                question=question,
                                query_plan=(
                                    retrieval_plan
                                    if self.retrieval_query_plan_provider is not None
                                    else repair_plan
                                ),
                                seed_entities=seeds,
                                graph_backend=backend,
                                expand=self.config.gnn_use_expansion,
                            )
                        )
                        repair_subgraph.retrieval_stage = f"repair_{repair_count}"
                        gnn_subgraph = repair_subgraph
                        embedding_calls_total += int(
                            repair_subgraph.query_embedding_calls or 0
                        )
                        embedding_tokens_total += int(
                            repair_subgraph.query_embedding_tokens or 0
                        )
                        embedding_cache_hits_total += int(
                            repair_subgraph.query_embedding_cache_hits or 0
                        )
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        raise RuntimeError(
                            "frozen v5 repair retrieval failed closed"
                        ) from exc

                repair_operator_evidence: list[TripleEvidence] = []
                repair_answer = deterministic_answer
                repair_selected = list(selected_entities)
                if self.config.variant in {"bim", "bim-gnn"}:
                    (
                        repair_answer,
                        repair_operator_evidence,
                        repair_selected,
                    ) = backend.execute_operator(repair_plan, seeds)
                if repair_plan.action_bindings:
                    action_answer = self._action_answer(
                        repair_plan,
                        repair_selected,
                        seeds,
                    )
                    if action_answer:
                        repair_answer = action_answer
                    elif repair_plan.action_bindings:
                        repair_answer = ""
                repair_mode = self._reasoning_mode(
                    question, repair_plan, repair_answer
                )
                retrieval_calls_used = (
                    phase_call_usage["planner"]
                    + phase_call_usage["traversal"]
                    + phase_call_usage["repair_traversal"]
                )
                remaining_retrieval_calls = max(
                    0, traversal_call_limit - retrieval_calls_used
                )
                repair_llm_call_limit = min(
                    max_llm_calls,
                    llm.calls + remaining_retrieval_calls,
                )
                phase_start = llm.calls
                usage_start = self._usage_snapshot(llm)
                repair_chains: list[list[TripleEvidence]]
                repair_traversal: list[TripleEvidence]
                repair_frontier: list[EntityRef]
                if repair_mode == "deterministic":
                    repair_chains, repair_traversal = [], []
                    repair_frontier = list(repair_selected or seeds)
                    repair_visited_nodes = len({item.node_id for item in repair_frontier})
                    repair_visited_edges = repair_depth = 0
                else:
                    (
                        repair_chains,
                        repair_traversal,
                        repair_frontier,
                        repair_visited_nodes,
                        repair_visited_edges,
                        repair_depth,
                    ) = self._explore(
                        repair_question,
                        repair_plan,
                        seeds,
                        backend,
                        llm,
                        errors,
                        repair_subgraph,
                        llm_call_limit=repair_llm_call_limit,
                        input_token_offset=embedding_tokens_total,
                    )
                phase_call_usage["repair_traversal"] += llm.calls - phase_start
                self._record_phase_usage(
                    phase_token_usage, "repair", usage_start, llm
                )
                chains.extend(repair_chains)
                traversal_evidence = self._dedupe_evidence(
                    [*traversal_evidence, *repair_traversal]
                )
                merged_repair_targets = self._merge_target_context(
                    backend,
                    repair_plan,
                    seeds,
                    repair_selected,
                    repair_frontier,
                    repair_subgraph,
                    require_bounded_retrieval_universe=requires_retrieval,
                )
                if merged_repair_targets:
                    selected_entities = merged_repair_targets
                if repair_plan.action_bindings:
                    repair_answer = self._action_answer(
                        repair_plan, selected_entities, seeds
                    )
                if repair_answer:
                    deterministic_answer = repair_answer

                evidence = self._dedupe_evidence(
                    [*traversal_evidence, *operator_evidence, *repair_operator_evidence]
                )
                operator_evidence = self._dedupe_evidence(
                    [*operator_evidence, *repair_operator_evidence]
                )
                visited_nodes = max(visited_nodes, repair_visited_nodes)
                visited_edges = max(visited_edges, repair_visited_edges, len(evidence))
                depth_reached = max(depth_reached, repair_depth)
                plan = repair_plan
                hierarchy_context = self._hierarchy_context(
                    backend, seeds, selected_entities, evidence
                )
                target_audit = backend.target_audit(
                    repair_plan,
                    seeds,
                    selected_entities,
                    excluded_limit=self.config.effective_audit_excluded_limit,
                )
                hierarchy_validation = backend.validate_hierarchy(
                    repair_plan,
                    seeds,
                    selected_entities,
                    max_paths=self.config.effective_hierarchy_max_paths,
                    max_depth=self.config.hierarchy_max_depth,
                )
                reasoning_mode = self._reasoning_mode(
                    question, plan, deterministic_answer
                )
                reasoning_trace, evidence_review = self._reason_and_review(
                    llm,
                    question,
                    plan,
                    hierarchy_context,
                    evidence,
                    deterministic_answer,
                    reasoning_mode,
                    target_audit,
                    errors,
                    phase_call_usage,
                    phase_token_usage,
                    "repair_",
                    input_token_offset=embedding_tokens_total,
                )

            deterministic_binding_valid = bool(
                bool(plan.action_bindings)
                and self._target_audit_is_valid(target_audit)
                and deterministic_answer
                and not deterministic_answer.startswith(
                    "I cannot determine the answer from the IFC knowledge graph."
                )
            )
            if deterministic_binding_valid:
                validation_status = "passed"
                if self.config.llm_answering_required:
                    phase_start = llm.calls
                    usage_start = self._usage_snapshot(llm)
                    try:
                        answer = str(
                            self._budgeted(
                                llm,
                                "finalize_answer",
                                question,
                                plan,
                                hierarchy_context,
                                evidence,
                                deterministic_answer,
                                reasoning_trace,
                                evidence_review,
                                call_limit=min(max_llm_calls, llm.calls + 1),
                                input_token_offset=embedding_tokens_total,
                                estimated_input_tokens=self._estimated_finalizer_input_tokens(
                                    question,
                                    plan,
                                    hierarchy_context,
                                    evidence,
                                    deterministic_answer,
                                    reasoning_trace,
                                    evidence_review,
                                ),
                            )
                        ).strip()
                        if not answer:
                            answer = deterministic_answer
                            finalization_mode = "required_llm_deterministic_fallback"
                        else:
                            finalization_mode = "required_llm"
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        errors.append(
                            f"required_finalizer:{type(exc).__name__}:{exc}"
                        )
                        answer = deterministic_answer
                        finalization_mode = "required_llm_error_fallback"
                    finally:
                        phase_call_usage["finalizer"] += llm.calls - phase_start
                        self._record_phase_usage(
                            phase_token_usage,
                            "final_explanation",
                            usage_start,
                            llm,
                        )
                else:
                    answer = deterministic_answer
                    finalization_mode = "deterministic_binding"
            elif plan.action_bindings:
                validation_status = "failed_after_repairs"
                if self.config.llm_answering_required:
                    # The experiment requires a real answering-model call for
                    # every question, including safely abstained cases.  The
                    # call is observable but cannot override a failed backend
                    # target audit or promote unsupported targets.
                    phase_start = llm.calls
                    usage_start = self._usage_snapshot(llm)
                    try:
                        self._budgeted(
                            llm,
                            "finalize_answer",
                            question,
                            plan,
                            hierarchy_context,
                            evidence,
                            deterministic_answer,
                            reasoning_trace,
                            evidence_review,
                            call_limit=min(max_llm_calls, llm.calls + 1),
                            input_token_offset=embedding_tokens_total,
                            estimated_input_tokens=self._estimated_finalizer_input_tokens(
                                question,
                                plan,
                                hierarchy_context,
                                evidence,
                                deterministic_answer,
                                reasoning_trace,
                                evidence_review,
                            ),
                        )
                        finalization_mode = "required_llm_safety_abstention"
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        errors.append(
                            f"required_abstention_finalizer:{type(exc).__name__}:{exc}"
                        )
                        finalization_mode = "required_llm_error_safety_abstention"
                    finally:
                        phase_call_usage["finalizer"] += llm.calls - phase_start
                        self._record_phase_usage(
                            phase_token_usage,
                            "final_explanation",
                            usage_start,
                            llm,
                        )
                answer = "I cannot determine the answer from the IFC knowledge graph."
                if not self.config.llm_answering_required:
                    finalization_mode = "deterministic_binding_abstention"
            elif self._review_allows_finalization(evidence_review):
                validation_status = "passed"
                if (
                    not self.config.llm_answering_required
                    and (
                        reasoning_mode == "deterministic"
                        or self.config.effective_deterministic_finalization
                    )
                ):
                    answer = deterministic_answer
                    finalization_mode = (
                        "validated_deterministic"
                        if reasoning_mode != "deterministic"
                        else "deterministic"
                    )
                else:
                    phase_start = llm.calls
                    usage_start = self._usage_snapshot(llm)
                    try:
                        answer = str(
                            self._budgeted(
                                llm,
                                "finalize_answer",
                                question,
                                plan,
                                hierarchy_context,
                                evidence,
                                deterministic_answer,
                                reasoning_trace,
                                evidence_review,
                                call_limit=min(
                                    max_llm_calls, llm.calls + 1
                                ),
                                input_token_offset=embedding_tokens_total,
                                estimated_input_tokens=self._estimated_finalizer_input_tokens(
                                    question,
                                    plan,
                                    hierarchy_context,
                                    evidence,
                                    deterministic_answer,
                                    reasoning_trace,
                                    evidence_review,
                                ),
                            )
                        ).strip()
                        if not answer:
                            answer = deterministic_answer
                            finalization_mode = "validated_deterministic_fallback"
                        else:
                            finalization_mode = "validated_llm"
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        errors.append(f"finalizer:{type(exc).__name__}:{exc}")
                        answer = deterministic_answer
                        finalization_mode = "validated_deterministic_fallback"
                    finally:
                        phase_call_usage["finalizer"] += llm.calls - phase_start
                        self._record_phase_usage(
                            phase_token_usage,
                            "final_explanation",
                            usage_start,
                            llm,
                        )
            else:
                validation_status = "failed_after_repairs"
                if (
                    not self.config.llm_answering_required
                    and (
                        self.config.hierarchy_precision_gate
                        or self.config.effective_deterministic_finalization
                    )
                ):
                    answer = "I cannot determine the answer from the IFC knowledge graph."
                    finalization_mode = (
                        "precision_gated_abstention"
                        if self.config.hierarchy_precision_gate
                        else "deterministic_validation_abstention"
                    )
                else:
                    phase_start = llm.calls
                    usage_start = self._usage_snapshot(llm)
                    try:
                        answer = str(
                            self._budgeted(
                                llm,
                                "finalize_answer",
                                question,
                                plan,
                                hierarchy_context,
                                evidence,
                                deterministic_answer,
                                reasoning_trace,
                                evidence_review,
                                call_limit=min(
                                    max_llm_calls, llm.calls + 1
                                ),
                                input_token_offset=embedding_tokens_total,
                                estimated_input_tokens=self._estimated_finalizer_input_tokens(
                                    question,
                                    plan,
                                    hierarchy_context,
                                    evidence,
                                    deterministic_answer,
                                    reasoning_trace,
                                    evidence_review,
                                ),
                            )
                        ).strip()
                        if not answer:
                            answer = deterministic_answer
                        if answer.lower().startswith("abstain:"):
                            answer = deterministic_answer
                        finalization_mode = "best_effort_evidence"
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        errors.append(f"best_effort_finalizer:{type(exc).__name__}:{exc}")
                        answer = deterministic_answer
                        finalization_mode = "best_effort_deterministic_fallback"
                    finally:
                        phase_call_usage["finalizer"] += llm.calls - phase_start
                        self._record_phase_usage(
                            phase_token_usage,
                            "final_explanation",
                            usage_start,
                            llm,
                        )
        else:
            grounded_abstention = deterministic_answer.startswith(
                "I cannot determine the answer from the IFC knowledge graph."
            )
            target_audit_failed = (
                bool(plan.action_bindings)
                and self.config.variant in {"bim", "bim-gnn"}
                and not self._target_audit_is_valid(target_audit)
            )
            if self.config.llm_answering_required:
                # A paired answering experiment requires an observable system
                # model call even for direct operators and safe abstentions.
                # The call may verbalize a valid backend result, but it cannot
                # promote targets rejected by the deterministic target audit.
                phase_start = llm.calls
                usage_start = self._usage_snapshot(llm)
                generated_answer = ""
                try:
                    generated_answer = str(
                        self._budgeted(
                            llm,
                            "generate_answer",
                            question,
                            plan,
                            evidence,
                            deterministic_answer,
                            call_limit=min(max_llm_calls, llm.calls + 1),
                            input_token_offset=embedding_tokens_total,
                        )
                    ).strip()
                except TimeoutError:
                    raise
                except Exception as exc:
                    errors.append(
                        f"required_answer_generation:{type(exc).__name__}:{exc}"
                    )
                finally:
                    phase_call_usage["finalizer"] += llm.calls - phase_start
                    self._record_phase_usage(
                        phase_token_usage,
                        "final_explanation",
                        usage_start,
                        llm,
                    )
                if target_audit_failed:
                    answer = (
                        "I cannot determine the answer from the IFC knowledge graph."
                    )
                    validation_status = "failed"
                    finalization_mode = "required_llm_generate_safety_abstention"
                elif generated_answer:
                    answer = generated_answer
                    validation_status = (
                        "passed" if not grounded_abstention else "failed"
                    )
                    finalization_mode = "required_llm_generate"
                else:
                    answer = deterministic_answer
                    validation_status = (
                        "passed" if not grounded_abstention else "failed"
                    )
                    finalization_mode = "required_llm_generate_fallback"
            elif self.config.effective_deterministic_finalization:
                if (
                    target_audit_failed
                ):
                    answer = "I cannot determine the answer from the IFC knowledge graph."
                    finalization_mode = "deterministic_validation_abstention"
                    validation_status = "failed"
                else:
                    answer = deterministic_answer
                    finalization_mode = "deterministic"
                    validation_status = "passed" if not grounded_abstention else "failed"
            elif llm.calls < max_llm_calls and evidence and not grounded_abstention:
                phase_start = llm.calls
                usage_start = self._usage_snapshot(llm)
                try:
                    answer = str(
                        self._budgeted(
                            llm,
                            "generate_answer",
                            question,
                            plan,
                            evidence,
                            deterministic_answer,
                            input_token_offset=embedding_tokens_total,
                        )
                    ).strip()
                except TimeoutError:
                    raise
                except Exception as exc:
                    errors.append(f"answer_generation:{type(exc).__name__}:{exc}")
                    answer = deterministic_answer
                finally:
                    phase_call_usage["finalizer"] += llm.calls - phase_start
                    self._record_phase_usage(
                        phase_token_usage,
                        "final_explanation",
                        usage_start,
                        llm,
                    )
            else:
                answer = deterministic_answer

        if (
            bool(plan.action_bindings)
            and deterministic_answer
            and not answer.startswith(
                "I cannot determine the answer from the IFC knowledge graph."
            )
        ):
            answer, validation_error = self._validate_pddl_answer(
                answer, deterministic_answer, plan, evidence, seeds
            )
            if validation_error:
                errors.append(validation_error)

        debug = {}
        if self.config.debug:
            debug = {
                "plan": plan.__dict__ if hasattr(plan, "__dict__") else {
                    field: getattr(plan, field)
                    for field in plan.__dataclass_fields__
                },
                "index_report": report.to_dict(),
                "selected_entities": [entity.node_id for entity in selected_entities],
                "gnn_artifact_report": (
                    self.gnn_artifact_reports[source_path].to_dict()
                    if source_path in self.gnn_artifact_reports
                    else None
                ),
            }
            if route_decision is not None:
                debug["retrieval_route"] = route_decision.to_dict()
        gnn_artifact_hash = ""
        if source_path in self.gnn_artifact_reports:
            gnn_artifact_hash = self.gnn_artifact_reports[source_path].artifact_hash
        phase_token_usage["gnn_retrieval"] = {
            "calls": embedding_calls_total,
            "input_tokens": embedding_tokens_total,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": embedding_tokens_total,
            "cache_hits": embedding_cache_hits_total,
        }
        input_budget = self.config.effective_input_token_budget
        system_input_tokens = llm.input_tokens + embedding_tokens_total
        budget_reasons: list[str] = []
        if llm.calls >= max_llm_calls:
            budget_reasons.append(f"ToG LLM call budget exhausted ({max_llm_calls})")
        if input_budget is not None and system_input_tokens >= input_budget:
            budget_reasons.append(
                "ToG system input token budget exhausted "
                f"({system_input_tokens}>={input_budget}; "
                f"reasoning={llm.input_tokens}, embeddings={embedding_tokens_total})"
            )
        budget_reasons = list(dict.fromkeys(budget_reasons))
        phase_budget_errors = [
            item for item in errors if "budget exhausted" in item.lower()
        ]
        if budget_reasons:
            budget_status = "global_budget_exhausted"
        elif phase_budget_errors:
            budget_status = "phase_budget_exhausted"
        elif (
            not answer
            or answer.startswith("I cannot determine the answer from the IFC knowledge graph.")
        ):
            budget_status = "insufficient_graph_evidence"
        else:
            budget_status = "ok"
        reported_budget_reasons = list(
            dict.fromkeys([*budget_reasons, *phase_budget_errors])
        )
        return ToGResponse(
            answer=answer,
            variant=self.config.variant,
            operator=plan.operator,
            execution_profile=self.config.profile,
            reasoning_chains=chains,
            seed_entities=seeds,
            evidence=evidence,
            visited_nodes=visited_nodes,
            visited_edges=visited_edges,
            depth_reached=depth_reached,
            llm_calls=llm.calls,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            cached_input_tokens=int(
                getattr(llm, "cached_input_tokens", 0) or 0
            ),
            reasoning_output_tokens=int(
                getattr(llm, "reasoning_output_tokens", 0) or 0
            ),
            graph_hash=backend.graph_hash,
            graph_schema=backend.graph_schema,
            planned_hops=(
                retrieval_plan.gnn_hops
                if requires_retrieval and self.use_gnn_prior
                else 0
            ),
            planned_depth=(plan.tog_depth if self.config.hierarchy_reasoning else self.config.depth),
            planned_width=(plan.tog_width if self.config.hierarchy_reasoning else self.config.width),
            gnn_retrieval_levels=(
                list(retrieval_plan.gnn_retrieval_levels)
                if requires_retrieval and self.use_retrieval_prior
                else []
            ),
            gnn_artifact_hash=gnn_artifact_hash,
            gnn_subgraph=gnn_subgraph,
            initial_gnn_subgraph=initial_gnn_subgraph,
            hierarchy_context=hierarchy_context,
            reasoning_mode=reasoning_mode,
            reasoning_trace=reasoning_trace,
            evidence_review=evidence_review,
            target_audit=target_audit,
            operator_result=operator_result,
            query_hypotheses=query_hypotheses,
            query_hypothesis_selection=query_hypothesis_selection,
            target_selection=target_selection,
            hierarchy_validation=hierarchy_validation,
            selected_entities=list(selected_entities),
            constraint_valid_node_ids=[
                item.target_id
                for item in target_audit.target_validations
                if item.valid
            ],
            binding_complete=bool(target_audit.action_bindings_covered),
            phase_call_usage={
                **dict(phase_call_usage),
                "repair": (
                    phase_call_usage["repair_traversal"]
                    + phase_call_usage["repair_reasoner"]
                    + phase_call_usage["repair_reviewer"]
                ),
                "reserved": reserved_calls,
                "traversal_ceiling": traversal_call_limit,
                "total": llm.calls,
            },
            phase_token_usage=phase_token_usage,
            gnn_embedding_calls_total=embedding_calls_total,
            gnn_embedding_input_tokens_total=embedding_tokens_total,
            reasoning_input_tokens_total=llm.input_tokens,
            system_input_tokens_total=system_input_tokens,
            repair_count=repair_count,
            finalization_mode=finalization_mode,
            validation_status=validation_status,
            budget_exhausted=bool(reported_budget_reasons),
            budget_reason="; ".join(reported_budget_reasons),
            budget_status=budget_status,
            errors=errors,
            debug=debug,
        )
