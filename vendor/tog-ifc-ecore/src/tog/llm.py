from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .models import (
    EntityRef,
    EvidenceReview,
    HierarchyContext,
    HierarchyReasoningTrace,
    QueryHypothesis,
    QueryHypothesisSelectionResult,
    QueryPlan,
    ReasoningMode,
    RelationRef,
    TargetAudit,
    TargetSelectionResult,
    TripleEvidence,
)


class ToGLlm(Protocol):
    """Semantic LLM operations required by ToG.

    Implementations may use BAML, OpenAI, or deterministic test doubles. Usage
    counters are exposed so the evaluation runner can account for every call.
    """

    @property
    def calls(self) -> int: ...

    @property
    def input_tokens(self) -> int: ...

    @property
    def output_tokens(self) -> int: ...

    @property
    def cached_input_tokens(self) -> int: ...

    @property
    def reasoning_output_tokens(self) -> int: ...

    def plan_query(
        self,
        question: str,
        fallback: QueryPlan,
        max_gnn_hops: int,
        max_tog_depth: int,
        max_tog_width: int,
    ) -> QueryPlan: ...

    def select_query_hypothesis(
        self,
        question: str,
        hypotheses: Sequence[QueryHypothesis],
    ) -> QueryHypothesisSelectionResult: ...

    def select_relations(
        self,
        question: str,
        entity: EntityRef,
        relations: Sequence[RelationRef],
        width: int,
    ) -> list[RelationRef]: ...

    def score_entities(
        self,
        question: str,
        relation: RelationRef,
        entities: Sequence[EntityRef],
        width: int,
    ) -> list[EntityRef]: ...

    def resolve_targets(
        self,
        question: str,
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
    ) -> TargetSelectionResult: ...

    def is_sufficient(
        self,
        question: str,
        evidence: Sequence[TripleEvidence],
    ) -> bool: ...

    def generate_answer(
        self,
        question: str,
        plan: QueryPlan,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
    ) -> str: ...

    def reason_hierarchy(
        self,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_mode: ReasoningMode,
    ) -> HierarchyReasoningTrace: ...

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
    ) -> EvidenceReview: ...

    def finalize_answer(
        self,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_trace: HierarchyReasoningTrace,
        evidence_review: EvidenceReview,
    ) -> str: ...

    def compile_retrieval_intent_contract_v3(
        self,
        question: str,
        compact_plan_json: str,
    ) -> dict[str, Any]: ...

    def adjudicate_and_answer_evidence_groups_v3(
        self,
        question: str,
        intent_contract_json: str,
        candidate_group_ledger_json: str,
    ) -> dict[str, Any]: ...

    def review_and_answer_evidence_groups_v3(
        self,
        question: str,
        intent_contract_json: str,
        candidate_group_ledger_json: str,
        initial_selection_json: str,
    ) -> dict[str, Any]: ...


class CallBudgetExceeded(RuntimeError):
    pass
