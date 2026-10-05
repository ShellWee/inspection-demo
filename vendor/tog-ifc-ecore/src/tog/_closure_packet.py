from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .models import RetrievalClosureArm

ClosureRequirementKind = Literal[
    "operator",
    "binding",
    "scope",
    "relation",
    "cardinality",
    "order",
]
ClosureRequirementStatus = Literal[
    "supported",
    "recoverable_retrieval_miss",
    "recoverable_selection_error",
    "graph_or_schema_absent",
    "conflict",
]
ClosureStatus = Literal["pass", "repair", "abstain"]

_RECOVERABLE: frozenset[str] = frozenset(
    {"recoverable_retrieval_miss", "recoverable_selection_error"}
)
_ABSTAIN: frozenset[str] = frozenset({"graph_or_schema_absent", "conflict"})
_ACTION_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\(([^()]+)\)")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ClosureRequirementV1:
    requirement_id: str
    kind: ClosureRequirementKind
    description: str
    binding_index: int | None = None
    required_values: list[str] = field(default_factory=list)
    relation: str | None = None
    direction: str | None = None
    cardinality: str | None = None
    order_index: int | None = None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClosureRequirementAssessmentV1:
    requirement: ClosureRequirementV1
    status: ClosureRequirementStatus
    evidence_ids: list[str] = field(default_factory=list)
    path_ids: list[str] = field(default_factory=list)
    certified_target_ids: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalEvidencePacketV1:
    question: str
    query_plan: dict[str, Any]
    retrieval_query_plan: dict[str, Any]
    query_plan_sha256: str
    retrieval_query_plan_sha256: str
    target_anchors: list[dict[str, Any]]
    support_anchors: list[dict[str, Any]]
    evaluation_anchors: list[dict[str, Any]]
    binding_candidate_ids: dict[str, list[str]]
    traversal_evidence: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    typed_path_inventory: list[dict[str, Any]]
    candidate_universe: dict[str, Any]
    selected_entities: list[dict[str, Any]]
    seed_entities: list[dict[str, Any]]
    target_audit: dict[str, Any]
    target_selection: dict[str, Any]
    operator_result: dict[str, Any] | None
    deterministic_answer: str
    graph_hash: str
    graph_schema: str
    gnn_artifact_hash: str
    lineage_hashes: dict[str, str]
    upstream_usage: dict[str, Any]
    traversal_metrics: dict[str, Any]
    gnn_subgraph: dict[str, Any]
    schema_version: str = "retrieval-evidence-packet-v1"
    packet_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("packet_sha256", None)
        return payload

    def seal(self) -> RetrievalEvidencePacketV1:
        self.packet_sha256 = sha256_json(self._payload())
        return self

    def verify(self) -> bool:
        return bool(self.packet_sha256) and self.packet_sha256 == sha256_json(
            self._payload()
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.packet_sha256:
            self.seal()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetrievalEvidencePacketV1:
        packet = cls(**copy.deepcopy(dict(payload)))
        if not packet.verify():
            raise ValueError("RetrievalEvidencePacketV1 self-hash mismatch")
        return packet


@dataclass(slots=True)
class EvidenceClosureCertificateV1:
    packet_sha256: str
    arm: RetrievalClosureArm
    requirements: list[ClosureRequirementAssessmentV1]
    certified_target_ids: dict[str, list[str]] = field(default_factory=dict)
    closure_status: ClosureStatus = "repair"
    stop_reason: str = ""
    authoritative_hierarchy: bool = False
    arm_view_sha256: str = ""
    schema_version: str = "evidence-closure-certificate-v1"
    certificate_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("certificate_sha256", None)
        return payload

    def seal(self) -> EvidenceClosureCertificateV1:
        self.certificate_sha256 = sha256_json(self._payload())
        return self

    def verify(self) -> bool:
        return bool(self.certificate_sha256) and self.certificate_sha256 == sha256_json(
            self._payload()
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.certificate_sha256:
            self.seal()
        return asdict(self)


@dataclass(slots=True)
class ClosureRepairStepV1:
    step_index: int
    repair_kind: Literal["path", "selection_cardinality"]
    before_certificate_sha256: str
    after_certificate_sha256: str
    changed_requirement_ids: list[str] = field(default_factory=list)
    added_path_ids: list[str] = field(default_factory=list)
    certified_target_ids: dict[str, list[str]] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClosureRepairTraceV1:
    packet_sha256: str
    arm: RetrievalClosureArm
    initial_certificate_sha256: str
    final_certificate_sha256: str
    steps: list[ClosureRepairStepV1] = field(default_factory=list)
    max_steps: int = 2
    schema_version: str = "closure-repair-trace-v1"
    trace_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("trace_sha256", None)
        return payload

    def seal(self) -> ClosureRepairTraceV1:
        if len(self.steps) > self.max_steps:
            raise ValueError("Closure repair trace exceeds the two-step limit")
        self.trace_sha256 = sha256_json(self._payload())
        return self

    def verify(self) -> bool:
        return bool(self.trace_sha256) and self.trace_sha256 == sha256_json(
            self._payload()
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.trace_sha256:
            self.seal()
        return asdict(self)


@dataclass(slots=True)
class RetrievalClosureResultV1:
    answer: str
    certificate: EvidenceClosureCertificateV1
    repair_trace: ClosureRepairTraceV1
    reasoning_trace: dict[str, Any]
    evidence_review: dict[str, Any]
    downstream_usage: dict[str, Any]
    answer_call_count: int
    finalization_mode: str
    errors: list[str] = field(default_factory=list)


def compile_closure_requirements(plan: Mapping[str, Any]) -> list[ClosureRequirementV1]:
    requirements: list[ClosureRequirementV1] = [
        ClosureRequirementV1(
            requirement_id="operator",
            kind="operator",
            description=f"execute operator={plan.get('operator', 'lookup')}",
            required_values=[str(plan.get("operator", "lookup"))],
        )
    ]
    bindings = list(plan.get("action_bindings") or [])
    if bindings:
        for offset, raw_binding in enumerate(bindings):
            binding = dict(raw_binding)
            binding_index = int(binding.get("binding_index", offset))
            target_values = _flatten_values(
                binding,
                (
                    "target_kind",
                    "target_roles",
                    "target_names",
                    "target_domains",
                    "function_types",
                    "system_categories",
                ),
            )
            requirements.append(
                ClosureRequirementV1(
                    requirement_id=f"binding:{binding_index}:target",
                    kind="binding",
                    binding_index=binding_index,
                    description=(
                        f"bind {binding.get('action', 'action')} occurrence "
                        f"{binding_index} to an executable target"
                    ),
                    required_values=target_values,
                    cardinality=str(
                        binding.get("cardinality_policy")
                        or plan.get("cardinality_policy")
                        or "single"
                    ),
                    order_index=offset,
                )
            )
            requirements.append(
                ClosureRequirementV1(
                    requirement_id=f"binding:{binding_index}:cardinality",
                    kind="cardinality",
                    binding_index=binding_index,
                    description=(
                        "apply cardinality="
                        + str(
                            binding.get("cardinality_policy")
                            or plan.get("cardinality_policy")
                            or "single"
                        )
                    ),
                    cardinality=str(
                        binding.get("cardinality_policy")
                        or plan.get("cardinality_policy")
                        or "single"
                    ),
                    order_index=offset,
                )
            )
        if len(bindings) > 1:
            requirements.append(
                ClosureRequirementV1(
                    requirement_id="action_order",
                    kind="order",
                    description="preserve requested action occurrence order",
                    required_values=[str(item.get("action", "")) for item in bindings],
                )
            )
    else:
        target_values = _flatten_values(
            plan,
            (
                "target_kind",
                "target_ifc_class",
                "target_role",
                "target_roles",
                "target_domain",
                "target_space_type",
                "target_name",
                "target_names",
                "target_family_terms",
                "target_type_terms",
                "target_keywords",
            ),
        )
        requirements.append(
            ClosureRequirementV1(
                requirement_id="target",
                kind="binding",
                description="identify the query target set",
                required_values=target_values,
                cardinality=str(plan.get("cardinality_policy") or "single"),
            )
        )

    scope_values = _flatten_values(
        plan,
        ("storey", "room", "room_names", "scope_predicates"),
    )
    if scope_values:
        requirements.append(
            ClosureRequirementV1(
                requirement_id="scope",
                kind="scope",
                description="ground the requested spatial or typed scope",
                required_values=scope_values,
            )
        )

    for index, raw_reference in enumerate(plan.get("relation_references") or []):
        reference = dict(raw_reference)
        relation = str(reference.get("relation") or "")
        requirements.append(
            ClosureRequirementV1(
                requirement_id=f"relation:{index}:{relation or 'unspecified'}",
                kind="relation",
                description=f"ground relation={relation or 'unspecified'}",
                required_values=_flatten_values(
                    reference, ("mentions", "node_ids", "reference_kind")
                ),
                relation=relation or None,
                direction=str(reference.get("direction") or "") or None,
            )
        )
    return requirements


def arm_path_inventory(
    packet: RetrievalEvidencePacketV1,
    arm: RetrievalClosureArm,
) -> list[dict[str, Any]]:
    paths = copy.deepcopy(packet.typed_path_inventory)
    if arm == "no_hierarchy":
        return []
    if arm == "typed_hierarchy":
        for path in paths:
            path["authoritative"] = bool(path.get("authoritative", True))
            path["control"] = "typed"
        return paths
    if arm != "relation_shuffled_control":
        raise ValueError(f"Unsupported retrieval closure arm: {arm}")

    vocabulary = sorted(
        {
            str(relation)
            for path in paths
            for relation in list(path.get("relations") or [])
            if str(relation)
        }
    )
    if len(vocabulary) < 2:
        vocabulary = ["contains", "assigned_to_system", "serves"]
    shuffled = {
        relation: vocabulary[(index + 1) % len(vocabulary)]
        for index, relation in enumerate(vocabulary)
    }
    for path in paths:
        path["path_id"] = f"shuffled_control_{path.get('path_id', '')}"
        path["relations"] = [
            shuffled.get(str(relation), vocabulary[0])
            for relation in list(path.get("relations") or [])
        ]
        path["authoritative"] = False
        path["control"] = "relation_shuffled"
        path["provenance"] = "non_authoritative_relation_shuffled_control"
    return paths


def build_initial_certificate(
    packet: RetrievalEvidencePacketV1,
    arm: RetrievalClosureArm,
) -> EvidenceClosureCertificateV1:
    _require_packet(packet)
    requirements = compile_closure_requirements(packet.query_plan)
    direct_valid = _direct_valid_target_ids(packet)
    evidence_relations = {
        str(item.get("relation") or "")
        for item in packet.evidence
        if str(item.get("relation") or "")
    }
    selected_ids = {
        str(item.get("node_id") or "")
        for item in packet.selected_entities
        if str(item.get("node_id") or "")
    }
    assessments: list[ClosureRequirementAssessmentV1] = []
    certified: dict[str, list[str]] = {}
    conflicting = bool(packet.target_audit.get("conflicting_extras"))

    for requirement in requirements:
        if requirement.kind == "operator":
            assessment = ClosureRequirementAssessmentV1(
                requirement=requirement,
                status="supported",
                evidence_ids=["plan:operator"],
                note="operator compiled deterministically from the core QueryPlan",
            )
        elif requirement.kind == "binding":
            candidates = _binding_candidates(packet, requirement.binding_index)
            valid = [node_id for node_id in candidates if node_id in direct_valid]
            if requirement.binding_index is None and not valid:
                valid = sorted(selected_ids.intersection(direct_valid))
            if conflicting:
                status: ClosureRequirementStatus = "conflict"
                note = "backend target audit reports conflicting extra targets"
            elif valid:
                status = "supported"
                note = "target is directly supported by deterministic traversal/audit evidence"
            elif candidates and _has_authoritative_candidate_path(
                packet, requirement.binding_index, candidates
            ):
                status = "recoverable_retrieval_miss"
                note = "sealed binding candidate has a typed path withheld until path repair"
            elif candidates:
                status = "recoverable_selection_error"
                note = "sealed binding candidates exist but direct selection did not close"
            else:
                status = "graph_or_schema_absent"
                note = "no target exists in the sealed binding candidate universe"
            key = _binding_key(requirement.binding_index)
            if valid:
                certified[key] = _ordered_unique(valid)
            assessment = ClosureRequirementAssessmentV1(
                requirement=requirement,
                status=status,
                evidence_ids=[
                    f"target_audit:{node_id}" for node_id in _ordered_unique(valid)
                ],
                certified_target_ids=_ordered_unique(valid),
                note=note,
            )
        elif requirement.kind == "scope":
            scope_supported = _scope_supported(packet)
            assessment = ClosureRequirementAssessmentV1(
                requirement=requirement,
                status=(
                    "supported"
                    if scope_supported
                    else "recoverable_retrieval_miss"
                    if packet.typed_path_inventory
                    else "graph_or_schema_absent"
                ),
                evidence_ids=_scope_evidence_ids(packet),
                note=(
                    "scope is present in traversal/operator evidence"
                    if scope_supported
                    else "scope requires a typed endpoint path"
                ),
            )
        elif requirement.kind == "relation":
            relation = str(requirement.relation or "")
            supported = bool(relation and relation in evidence_relations)
            path_available = any(
                relation in {str(item) for item in path.get("relations") or []}
                and bool(path.get("authoritative", True))
                for path in packet.typed_path_inventory
            )
            assessment = ClosureRequirementAssessmentV1(
                requirement=requirement,
                status=(
                    "supported"
                    if supported
                    else "recoverable_retrieval_miss"
                    if path_available
                    else "graph_or_schema_absent"
                ),
                evidence_ids=[
                    f"evidence:{index + 1}"
                    for index, item in enumerate(packet.evidence)
                    if str(item.get("relation") or "") == relation
                ],
                note=(
                    "relation is present in ordinary ToG evidence"
                    if supported
                    else "relation requires an authoritative typed path"
                ),
            )
        elif requirement.kind == "cardinality":
            candidates = certified.get(_binding_key(requirement.binding_index), [])
            policy = str(requirement.cardinality or "single")
            assessment = ClosureRequirementAssessmentV1(
                requirement=requirement,
                status=(
                    "supported"
                    if _cardinality_closed(policy, candidates)
                    else "recoverable_selection_error"
                ),
                certified_target_ids=list(candidates),
                note="cardinality is closed over certified targets",
            )
        else:
            assessment = ClosureRequirementAssessmentV1(
                requirement=requirement,
                status="supported",
                evidence_ids=["plan:action_order"],
                note="action order is preserved by deterministic finalization",
            )
        assessments.append(assessment)

    paths = arm_path_inventory(packet, arm)
    certificate = EvidenceClosureCertificateV1(
        packet_sha256=packet.packet_sha256,
        arm=arm,
        requirements=assessments,
        certified_target_ids=certified,
        authoritative_hierarchy=arm == "typed_hierarchy",
        arm_view_sha256=sha256_json(
            {
                "packet_sha256": packet.packet_sha256,
                "arm": arm,
                "paths": paths,
            }
        ),
    )
    _refresh_certificate_status(certificate)
    return certificate.seal()


def apply_bounded_repairs(
    packet: RetrievalEvidencePacketV1,
    certificate: EvidenceClosureCertificateV1,
) -> tuple[EvidenceClosureCertificateV1, ClosureRepairTraceV1]:
    _require_packet(packet)
    initial_hash = certificate.certificate_sha256 or certificate.seal().certificate_sha256
    steps: list[ClosureRepairStepV1] = []
    current = copy.deepcopy(certificate)

    if current.closure_status == "repair" and current.arm == "typed_hierarchy":
        before = current.certificate_sha256
        changed: list[str] = []
        added_paths: list[str] = []
        path_view = arm_path_inventory(packet, current.arm)
        by_binding: dict[str, list[str]] = {}
        for path in path_view:
            if not bool(path.get("authoritative")):
                continue
            if not bool(path.get("complete_typed_path")):
                continue
            target_id = str(path.get("target_id") or "")
            binding_index = path.get("binding_index")
            candidates = set(_binding_candidates(packet, _optional_int(binding_index)))
            if not target_id or target_id not in candidates:
                continue
            key = _binding_key(_optional_int(binding_index))
            by_binding.setdefault(key, []).append(target_id)
            added_paths.append(str(path.get("path_id") or ""))

        for assessment in current.requirements:
            requirement = assessment.requirement
            if assessment.status != "recoverable_retrieval_miss":
                continue
            if requirement.kind == "binding":
                ids = _ordered_unique(
                    by_binding.get(_binding_key(requirement.binding_index), [])
                )
                if ids:
                    assessment.status = "supported"
                    assessment.certified_target_ids = ids
                    assessment.path_ids = _path_ids_for_targets(
                        path_view, requirement.binding_index, ids
                    )
                    assessment.note = "closed by one bounded authoritative typed-path repair"
                    current.certified_target_ids[_binding_key(requirement.binding_index)] = ids
                    changed.append(requirement.requirement_id)
            elif requirement.kind in {"scope", "relation"}:
                matching = _matching_paths(path_view, requirement)
                if matching:
                    assessment.status = "supported"
                    assessment.path_ids = [
                        str(path.get("path_id") or "") for path in matching
                    ]
                    assessment.note = "closed by one bounded authoritative typed-path repair"
                    changed.append(requirement.requirement_id)
        _sync_cardinality_assessments(current)
        _refresh_certificate_status(current)
        current.certificate_sha256 = ""
        current.seal()
        if changed:
            steps.append(
                ClosureRepairStepV1(
                    step_index=1,
                    repair_kind="path",
                    before_certificate_sha256=before,
                    after_certificate_sha256=current.certificate_sha256,
                    changed_requirement_ids=_ordered_unique(changed),
                    added_path_ids=_ordered_unique(added_paths),
                    certified_target_ids=copy.deepcopy(current.certified_target_ids),
                    note=(
                        "No embedding, GNN scoring, or traversal was rerun; only sealed "
                        "authoritative paths were exposed."
                    ),
                )
            )

    if current.closure_status == "repair":
        before = current.certificate_sha256
        changed = []
        for assessment in current.requirements:
            requirement = assessment.requirement
            if assessment.status != "recoverable_selection_error":
                continue
            if requirement.kind == "binding":
                candidates = _selection_eligible_candidates(
                    packet, current, requirement.binding_index
                )
                selected = _apply_cardinality(
                    str(requirement.cardinality or "single"), candidates
                )
                if selected:
                    assessment.status = "supported"
                    assessment.certified_target_ids = selected
                    assessment.note = "closed by deterministic selection/cardinality repair"
                    current.certified_target_ids[_binding_key(requirement.binding_index)] = selected
                    changed.append(requirement.requirement_id)
            elif requirement.kind == "cardinality":
                ids = current.certified_target_ids.get(
                    _binding_key(requirement.binding_index), []
                )
                if _cardinality_closed(str(requirement.cardinality or "single"), ids):
                    assessment.status = "supported"
                    assessment.certified_target_ids = list(ids)
                    assessment.note = "cardinality closed deterministically"
                    changed.append(requirement.requirement_id)
        _sync_cardinality_assessments(current)
        _refresh_certificate_status(current)
        current.certificate_sha256 = ""
        current.seal()
        if changed:
            steps.append(
                ClosureRepairStepV1(
                    step_index=len(steps) + 1,
                    repair_kind="selection_cardinality",
                    before_certificate_sha256=before,
                    after_certificate_sha256=current.certificate_sha256,
                    changed_requirement_ids=_ordered_unique(changed),
                    certified_target_ids=copy.deepcopy(current.certified_target_ids),
                    note=(
                        "Applied only to sealed candidates; action order and cardinality "
                        "were compiled from the core QueryPlan."
                    ),
                )
            )

    trace = ClosureRepairTraceV1(
        packet_sha256=packet.packet_sha256,
        arm=current.arm,
        initial_certificate_sha256=initial_hash,
        final_certificate_sha256=current.certificate_sha256,
        steps=steps,
    ).seal()
    return current, trace


def requirement_evidence_lines(
    packet: RetrievalEvidencePacketV1,
    certificate: EvidenceClosureCertificateV1,
) -> list[str]:
    evidence_by_id = {
        f"evidence:{index + 1}": _evidence_prompt_line(item)
        for index, item in enumerate(packet.evidence)
    }
    path_by_id = {
        str(path.get("path_id") or ""): _path_prompt_line(path)
        for path in arm_path_inventory(packet, certificate.arm)
    }
    lines: list[str] = []
    for assessment in certificate.requirements:
        payload = {
            "requirement_id": assessment.requirement.requirement_id,
            "kind": assessment.requirement.kind,
            "status": assessment.status,
            "description": assessment.requirement.description,
            "certified_target_ids": assessment.certified_target_ids,
            "evidence": [
                evidence_by_id[item]
                for item in assessment.evidence_ids
                if item in evidence_by_id
            ],
            "paths": [path_by_id[item] for item in assessment.path_ids if item in path_by_id],
            "note": assessment.note,
        }
        lines.append(canonical_json(payload))
    # The typed and shuffled arms must expose structurally matched path context
    # to the downstream LLM.  Only the typed arm can place those paths in the
    # certificate; shuffled rows remain explicitly non-authoritative.  Keeping
    # these context rows separate from ``assessment.path_ids`` prevents the
    # negative control from silently collapsing into the no-hierarchy arm.
    for path in arm_path_inventory(packet, certificate.arm):
        lines.append(
            canonical_json(
                {
                    "requirement_id": (
                        "hierarchy_path_context:"
                        + str(path.get("binding_index", "unbound"))
                    ),
                    "kind": "hierarchy_path_context",
                    "status": (
                        "supported_context"
                        if bool(path.get("authoritative"))
                        else "non_authoritative_control"
                    ),
                    "path_id": str(path.get("path_id") or ""),
                    "target_id": str(path.get("target_id") or ""),
                    "scope_id": str(path.get("scope_id") or ""),
                    "node_ids": list(path.get("node_ids") or []),
                    "relations": list(path.get("relations") or []),
                    "complete_typed_path": bool(path.get("complete_typed_path")),
                    "authoritative": bool(path.get("authoritative")),
                    "score": float(path.get("score") or 0.0),
                    "note": (
                        "May support the certificate."
                        if bool(path.get("authoritative"))
                        else "Negative-control context only; cannot certify or promote."
                    ),
                }
            )
        )
    return lines


def certified_deterministic_answer(
    packet: RetrievalEvidencePacketV1,
    certificate: EvidenceClosureCertificateV1,
) -> str:
    if certificate.closure_status != "pass":
        return "I cannot determine the answer from the IFC knowledge graph."
    bindings = list(packet.query_plan.get("action_bindings") or [])
    if not bindings:
        return packet.deterministic_answer
    entities = {
        str(item.get("node_id") or ""): item for item in packet.selected_entities
    }
    for item in packet.gnn_subgraph.get("candidate_entities", []) or []:
        entities.setdefault(str(item.get("node_id") or ""), item)
    rendered: list[str] = []
    for offset, binding in enumerate(bindings):
        binding_index = int(binding.get("binding_index", offset))
        action = str(binding.get("action") or "Inspect")
        node_ids = certificate.certified_target_ids.get(str(binding_index), [])
        policy = str(
            binding.get("cardinality_policy")
            or packet.query_plan.get("cardinality_policy")
            or "single"
        )
        if policy not in {"all", "count", "distinct", "group_count"}:
            node_ids = node_ids[:1]
        for node_id in node_ids:
            entity = entities.get(node_id, {})
            global_id = str(entity.get("global_id") or "")
            if not global_id:
                return "I cannot determine the answer from the IFC knowledge graph."
            call = f"{action}({global_id})"
            if call not in rendered:
                rendered.append(call)
    return ", ".join(rendered) or "I cannot determine the answer from the IFC knowledge graph."


def validate_or_repair_final_answer(
    generated_answer: str,
    certified_answer: str,
    packet: RetrievalEvidencePacketV1,
    certificate: EvidenceClosureCertificateV1,
) -> tuple[str, str | None]:
    generated = str(generated_answer or "").strip()
    if certificate.closure_status != "pass":
        return (
            "I cannot determine the answer from the IFC knowledge graph.",
            "closure_not_passed",
        )
    if not generated:
        return certified_answer, "empty_llm_answer_reverted_to_certificate"
    if not packet.query_plan.get("action_bindings"):
        return generated, None

    expected = _ACTION_RE.findall(certified_answer)
    observed = _ACTION_RE.findall(generated)
    if observed == expected:
        return generated, None
    return certified_answer, "llm_action_or_guid_mismatch_reverted_to_certificate"


def packet_arm_invariants(
    packet: RetrievalEvidencePacketV1,
) -> dict[str, Any]:
    views = {
        arm: arm_path_inventory(packet, arm)
        for arm in (
            "no_hierarchy",
            "typed_hierarchy",
            "relation_shuffled_control",
        )
    }
    typed = views["typed_hierarchy"]
    shuffled = views["relation_shuffled_control"]
    return {
        "packet_hash_valid": packet.verify(),
        "same_path_count_typed_shuffled": len(typed) == len(shuffled),
        "same_path_node_ids_typed_shuffled": [
            list(item.get("node_ids") or []) for item in typed
        ]
        == [list(item.get("node_ids") or []) for item in shuffled],
        "same_path_lengths_typed_shuffled": [
            len(item.get("relations") or []) for item in typed
        ]
        == [len(item.get("relations") or []) for item in shuffled],
        "shuffled_non_authoritative": all(
            not bool(item.get("authoritative")) for item in shuffled
        ),
        "no_hierarchy_paths_withheld": not views["no_hierarchy"],
        "operational_anchor_count": len(packet.target_anchors),
        "evaluation_anchor_count": len(packet.evaluation_anchors),
        "top20_top50_separated": len(packet.target_anchors) <= 20
        and len(packet.evaluation_anchors) <= 50,
    }


def _flatten_values(payload: Mapping[str, Any], keys: Iterable[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, Mapping):
            values.append(canonical_json(value))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                values.append(canonical_json(item) if isinstance(item, Mapping) else str(item))
        else:
            values.append(str(value))
    return _ordered_unique(values)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _binding_key(binding_index: int | None) -> str:
    return "target" if binding_index is None else str(binding_index)


def _binding_candidates(
    packet: RetrievalEvidencePacketV1, binding_index: int | None
) -> list[str]:
    if binding_index is None:
        candidates = [
            str(item.get("node_id") or "") for item in packet.target_anchors
        ]
    else:
        candidates = list(packet.binding_candidate_ids.get(str(binding_index), []))
    universe = {
        str(item)
        for item in packet.candidate_universe.get("target_node_ids", [])
    }
    if universe:
        candidates = [item for item in candidates if item in universe]
    return _ordered_unique(candidates)


def _direct_valid_target_ids(packet: RetrievalEvidencePacketV1) -> set[str]:
    valid = {
        str(item.get("target_id") or "")
        for item in packet.target_audit.get("target_validations", [])
        if bool(item.get("valid")) and str(item.get("target_id") or "")
    }
    if packet.target_audit.get("action_bindings_covered") and not valid:
        valid.update(
            str(item.get("node_id") or "")
            for item in packet.selected_entities
            if str(item.get("node_id") or "")
        )
    return valid


def _has_authoritative_candidate_path(
    packet: RetrievalEvidencePacketV1,
    binding_index: int | None,
    candidates: Sequence[str],
) -> bool:
    candidate_set = set(candidates)
    return any(
        bool(path.get("authoritative", True))
        and bool(path.get("complete_typed_path"))
        and str(path.get("target_id") or "") in candidate_set
        and (
            binding_index is None
            or _optional_int(path.get("binding_index")) == binding_index
        )
        for path in packet.typed_path_inventory
    )


def _scope_supported(packet: RetrievalEvidencePacketV1) -> bool:
    plan = packet.query_plan
    needles = {
        str(item).lower()
        for item in _flatten_values(
            plan, ("storey", "room", "room_names", "scope_predicates")
        )
    }
    if not needles:
        return True
    haystack = canonical_json(packet.evidence).lower()
    return any(needle and needle in haystack for needle in needles)


def _scope_evidence_ids(packet: RetrievalEvidencePacketV1) -> list[str]:
    plan = packet.query_plan
    needles = {
        str(item).lower()
        for item in _flatten_values(
            plan, ("storey", "room", "room_names", "scope_predicates")
        )
    }
    return [
        f"evidence:{index + 1}"
        for index, item in enumerate(packet.evidence)
        if any(needle and needle in canonical_json(item).lower() for needle in needles)
    ]


def _cardinality_closed(policy: str, target_ids: Sequence[str]) -> bool:
    if policy in {"all", "count", "distinct", "group_count"}:
        return bool(target_ids)
    return len(target_ids) == 1


def _apply_cardinality(policy: str, target_ids: Sequence[str]) -> list[str]:
    ids = _ordered_unique(target_ids)
    if policy in {"all", "count", "distinct", "group_count"}:
        return ids
    return ids[:1]


def _path_ids_for_targets(
    paths: Sequence[Mapping[str, Any]],
    binding_index: int | None,
    target_ids: Sequence[str],
) -> list[str]:
    targets = set(target_ids)
    return _ordered_unique(
        str(path.get("path_id") or "")
        for path in paths
        if str(path.get("target_id") or "") in targets
        and (
            binding_index is None
            or _optional_int(path.get("binding_index")) == binding_index
        )
    )


def _matching_paths(
    paths: Sequence[Mapping[str, Any]],
    requirement: ClosureRequirementV1,
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for path in paths:
        if not bool(path.get("authoritative")):
            continue
        if not bool(path.get("complete_typed_path")):
            continue
        relations = {str(item) for item in path.get("relations") or []}
        if requirement.kind == "relation" and requirement.relation not in relations:
            continue
        result.append(path)
    return result


def _selection_eligible_candidates(
    packet: RetrievalEvidencePacketV1,
    certificate: EvidenceClosureCertificateV1,
    binding_index: int | None,
) -> list[str]:
    key = _binding_key(binding_index)
    already = certificate.certified_target_ids.get(key, [])
    if already:
        return list(already)
    direct = _direct_valid_target_ids(packet)
    return [
        item for item in _binding_candidates(packet, binding_index) if item in direct
    ]


def _sync_cardinality_assessments(
    certificate: EvidenceClosureCertificateV1,
) -> None:
    for assessment in certificate.requirements:
        requirement = assessment.requirement
        if requirement.kind != "cardinality":
            continue
        ids = certificate.certified_target_ids.get(
            _binding_key(requirement.binding_index), []
        )
        assessment.certified_target_ids = list(ids)
        if _cardinality_closed(str(requirement.cardinality or "single"), ids):
            assessment.status = "supported"
            assessment.note = "cardinality is closed over certified targets"


def _refresh_certificate_status(certificate: EvidenceClosureCertificateV1) -> None:
    statuses = {item.status for item in certificate.requirements if item.requirement.required}
    if statuses.intersection(_ABSTAIN):
        certificate.closure_status = "abstain"
        certificate.stop_reason = (
            "conflict" if "conflict" in statuses else "graph_or_schema_absent"
        )
    elif statuses.intersection(_RECOVERABLE):
        certificate.closure_status = "repair"
        certificate.stop_reason = "bounded_repair_available"
    else:
        certificate.closure_status = "pass"
        certificate.stop_reason = "all_requirements_supported"


def _evidence_prompt_line(item: Mapping[str, Any]) -> str:
    return (
        f"({item.get('source_label', item.get('source_id', ''))}, "
        f"{item.get('relation', '')}, "
        f"{item.get('target_label', item.get('target_id', ''))})"
    )


def _path_prompt_line(path: Mapping[str, Any]) -> str:
    nodes = " > ".join(str(item) for item in path.get("node_ids") or [])
    relations = " > ".join(str(item) for item in path.get("relations") or [])
    return (
        f"{path.get('path_id', '')}: nodes={nodes}; relations={relations}; "
        f"authoritative={bool(path.get('authoritative'))}"
    )


def _require_packet(packet: RetrievalEvidencePacketV1) -> None:
    if not packet.verify():
        raise ValueError("Retrieval packet is unsealed or its self-hash is invalid")
    if len(packet.target_anchors) > 20:
        raise ValueError("Operational target anchors exceed top-20")
    if len(packet.evaluation_anchors) > 50:
        raise ValueError("Evaluation anchors exceed top-50")
    universe = set(packet.candidate_universe.get("target_node_ids", []))
    for values in packet.binding_candidate_ids.values():
        if universe and not set(values).issubset(universe):
            raise ValueError("Binding candidate IDs escape the sealed target universe")
