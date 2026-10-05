from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ._closure_packet import (
    RetrievalEvidencePacketV1,
    canonical_json,
    sha256_json,
)

CandidateDecisionV2 = Literal["select", "reject", "uncertain"]
ClosureStatusV2 = Literal["pass", "abstain"]

_ACTION_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\(([^()]+)\)")
_SET_CARDINALITIES = frozenset({"all", "count", "distinct", "group_count"})
_OPERATOR_BOUND = frozenset({"nearest", "argmax"})


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _normal(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _tokens(value: Any) -> set[str]:
    return set(_normal(value).split())


def _known(value: Any) -> bool:
    return _normal(value) not in {
        "",
        "unknown",
        "missing",
        "none",
        "not available",
        "no local text",
    }


def _flatten(payload: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    result: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, Mapping):
            result.extend(str(item) for item in value.values() if item not in (None, ""))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, Mapping):
                    result.extend(
                        str(nested)
                        for nested in item.values()
                        if nested not in (None, "", [], {})
                    )
                elif item not in (None, ""):
                    result.append(str(item))
        else:
            result.append(str(value))
    return _ordered_unique(result)


def _match_state(required: Sequence[str], observed: Sequence[str]) -> str:
    required_norm = [_normal(item) for item in required if _normal(item)]
    if not required_norm:
        return "not_required"
    observed_norm = [_normal(item) for item in observed if _known(item)]
    if not observed_norm:
        return "unknown"
    observed_joined = " | ".join(observed_norm)
    observed_tokens = set().union(*(_tokens(item) for item in observed_norm))
    for value in required_norm:
        required_tokens = _tokens(value)
        if (
            value in observed_joined
            or any(item in value for item in observed_norm)
            or (required_tokens and required_tokens.issubset(observed_tokens))
        ):
            return "match"
    return "mismatch"


def _candidate_level(metadata: Mapping[str, Any]) -> str:
    level = _normal(metadata.get("level"))
    if "l0" in level or "space" in level:
        return "space"
    if "l1" in level or "object" in level:
        return "object"
    if "l2" in level or "system" in level:
        return "system"
    if "l3" in level or "function" in level:
        return "function"
    return _normal(
        metadata.get("action_target_kind")
        or metadata.get("category")
        or metadata.get("kind")
    )


def _scope_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    predicates = []
    for row in list(plan.get("scope_predicates") or []):
        item = dict(row)
        predicates.append(
            {
                "predicate": str(item.get("predicate") or ""),
                "values": _ordered_unique(item.get("values") or []),
            }
        )
    return {
        "storey": str(plan.get("storey") or ""),
        "room": str(plan.get("room") or ""),
        "room_names": _ordered_unique(plan.get("room_names") or []),
        "predicates": predicates,
    }


def _default_relation_direction(relation: str) -> str:
    relation = _normal(relation).replace(" ", "_")
    if relation == "contains":
        return "scope_to_target"
    if relation in {"part_of", "assigned_to_system", "serves"}:
        return "target_to_scope"
    if relation in {"nearest", "adjacent_to"}:
        return "symmetric"
    return "unspecified"


def _ifc_class_kind(ifc_class: str) -> str:
    value = _normal(ifc_class).replace(" ", "")
    if value in {"ifcspace", "ifcbuildingstorey", "ifcsite", "ifcbuilding"}:
        return "space"
    if "system" in value:
        return "system"
    if value:
        return "object"
    return ""


def compact_query_plan_v2(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Remove catalogues/IDs/rationale and retain only answer-relevant structure."""

    bindings: list[dict[str, Any]] = []
    for offset, raw in enumerate(list(plan.get("action_bindings") or [])):
        binding = dict(raw)
        target_kind = str(binding.get("target_kind") or plan.get("target_kind") or "")
        target_ifc_class = str(
            binding.get("target_ifc_class") or plan.get("target_ifc_class") or ""
        )
        if (
            target_ifc_class
            and target_kind
            and _ifc_class_kind(target_ifc_class) != _normal(target_kind)
        ):
            # A global reference class must not be projected into an action-local
            # target of a different graph level (for example a door reference
            # attached to a space-valued nearest operator).
            target_ifc_class = ""
        bindings.append(
            {
                "binding_index": int(binding.get("binding_index", offset)),
                "action": str(binding.get("action") or "Inspect"),
                "cardinality": str(
                    binding.get("cardinality_policy")
                    or plan.get("cardinality_policy")
                    or "single"
                ),
                "target": {
                    "kind": target_kind,
                    "ifc_class": target_ifc_class,
                    "roles": _ordered_unique(
                        list(binding.get("target_roles") or [])
                        or list(plan.get("target_roles") or [])
                    ),
                    "names": _ordered_unique(
                        list(binding.get("target_names") or [])
                        or list(plan.get("target_names") or [])
                    ),
                    "domains": _ordered_unique(
                        list(binding.get("target_domains") or [])
                        or ([str(plan.get("target_domain"))] if plan.get("target_domain") else [])
                    ),
                    "function_types": _ordered_unique(
                        binding.get("function_types") or []
                    ),
                    "system_categories": _ordered_unique(
                        binding.get("system_categories") or []
                    ),
                },
            }
        )
    if not bindings:
        bindings.append(
            {
                "binding_index": 0,
                "action": "",
                "cardinality": str(plan.get("cardinality_policy") or "single"),
                "target": {
                    "kind": str(plan.get("target_kind") or ""),
                    "ifc_class": str(plan.get("target_ifc_class") or ""),
                    "roles": _ordered_unique(plan.get("target_roles") or []),
                    "names": _ordered_unique(plan.get("target_names") or []),
                    "domains": _ordered_unique(
                        [str(plan.get("target_domain"))]
                        if plan.get("target_domain")
                        else []
                    ),
                    "function_types": [],
                    "system_categories": [],
                },
            }
        )

    relations: list[dict[str, Any]] = []
    for raw in list(plan.get("relation_references") or []):
        reference = dict(raw)
        relation = str(reference.get("relation") or "")
        relations.append(
            {
                "relation": relation,
                "direction": str(reference.get("direction") or "")
                or _default_relation_direction(relation),
                "reference_kind": str(reference.get("reference_kind") or ""),
                "mentions": _ordered_unique(reference.get("mentions") or []),
            }
        )

    function_intents: list[dict[str, Any]] = []
    for raw in list(plan.get("function_intents") or []):
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        function_intents.append(
            {
                key: copy.deepcopy(item[key])
                for key in (
                    "function_type",
                    "function_kind",
                    "system_category",
                    "target_kind",
                )
                if item.get(key) not in (None, "", [], {})
            }
        )

    return {
        "schema_version": "compact-query-plan-v2",
        "operator": str(plan.get("operator") or "lookup"),
        "action_order": [item["action"] for item in bindings if item["action"]],
        "bindings": bindings,
        "scope": _scope_payload(plan),
        "relations": relations,
        "function_intents": function_intents,
    }


@dataclass(slots=True)
class CandidateLedgerV2:
    packet_sha256: str
    compact_plan: dict[str, Any]
    candidates: list[dict[str, Any]]
    path_counts: dict[str, int]
    top20_by_binding: dict[str, list[str]]
    top50_by_binding: dict[str, list[str]]
    global_conflicting_extras_diagnostic: bool = False
    schema_version: str = "candidate-ledger-v2"
    ledger_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("ledger_sha256", None)
        return payload

    def seal(self) -> CandidateLedgerV2:
        self.ledger_sha256 = sha256_json(self._payload())
        return self

    def verify(self) -> bool:
        return bool(self.ledger_sha256) and self.ledger_sha256 == sha256_json(
            self._payload()
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.ledger_sha256:
            self.seal()
        return asdict(self)

    def prompt_payload(self) -> dict[str, Any]:
        """Return the model-facing ledger without node IDs or raw IFC GUIDs."""

        rows: list[dict[str, Any]] = []
        fixed_reject_counts: dict[tuple[int, str], int] = {}
        for candidate in self.candidates:
            if candidate["eligibility"] != "reviewable":
                reasons = list(candidate["hard_reject_reasons"]) or [
                    "deterministic_eligibility_rejection"
                ]
                for reason in reasons:
                    key = (int(candidate["binding_index"]), str(reason))
                    fixed_reject_counts[key] = fixed_reject_counts.get(key, 0) + 1
                continue
            row = {
                "candidate_id": candidate["candidate_alias"],
                "binding_index": candidate["binding_index"],
                "rank": candidate["rank"],
                "rrf_score": candidate["rrf_score"],
                "operational_top20": candidate["operational_top20"],
                "eligibility": candidate["eligibility"],
                "hard_reject_reasons": list(candidate["hard_reject_reasons"]),
                "facts": copy.deepcopy(candidate["facts"]),
                "target_match": copy.deepcopy(candidate["target_match"]),
                "evidence_ids": list(candidate["evidence_ids"]),
            }
            if candidate.get("best_path"):
                path = dict(candidate["best_path"])
                row["best_typed_path"] = {
                    "evidence_id": path["evidence_id"],
                    "scope_endpoint": path["scope_endpoint"],
                    "directed_relations": list(path["directed_relations"]),
                    "scope_match": path["scope_match"],
                    "relation_match": path["relation_match"],
                    "authoritative": path["authoritative"],
                    "complete": path["complete"],
                }
            rows.append(row)
        return {
            "schema_version": "candidate-ledger-model-view-v2",
            "packet_sha256": self.packet_sha256,
            "ledger_sha256": self.ledger_sha256,
            "compact_plan": copy.deepcopy(self.compact_plan),
            "candidate_count": len(self.candidates),
            "reviewable_candidate_count": len(rows),
            "candidates": rows,
            "deterministic_fixed_reject_summary": [
                {
                    "binding_index": binding_index,
                    "reason": reason,
                    "count": count,
                }
                for (binding_index, reason), count in sorted(
                    fixed_reject_counts.items()
                )
            ],
            "classification_contract": (
                "Classify every eligibility=reviewable candidate exactly once. "
                "Candidates represented only in deterministic_fixed_reject_summary "
                "are already rejected and must not be added. An empty reviewable "
                "candidate list is valid and requires a grounded abstention."
            ),
        }


@dataclass(slots=True)
class CandidateSelectionV2:
    ledger_sha256: str
    phase: Literal["selector", "review"]
    decisions: list[dict[str, Any]]
    provider_complete: bool
    errors: list[str] = field(default_factory=list)
    summary: str = ""
    schema_version: str = "candidate-selection-v2"
    selection_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("selection_sha256", None)
        return payload

    def seal(self) -> CandidateSelectionV2:
        self.selection_sha256 = sha256_json(self._payload())
        return self

    def verify(self) -> bool:
        return bool(self.selection_sha256) and self.selection_sha256 == sha256_json(
            self._payload()
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.selection_sha256:
            self.seal()
        return asdict(self)


@dataclass(slots=True)
class EvidenceClosureCertificateV2:
    packet_sha256: str
    ledger_sha256: str
    selector_sha256: str
    review_sha256: str
    certified_target_ids: dict[str, list[str]]
    certified_candidate_aliases: dict[str, list[str]]
    binding_status: dict[str, dict[str, Any]]
    closure_status: ClosureStatusV2
    stop_reason: str
    selection_repair_count: int
    global_conflicting_extras_diagnostic: bool
    validation_errors: list[str] = field(default_factory=list)
    schema_version: str = "evidence-closure-certificate-v2"
    certificate_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("certificate_sha256", None)
        return payload

    def seal(self) -> EvidenceClosureCertificateV2:
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
class RetrievalClosureResultV2:
    answer: str
    ledger: CandidateLedgerV2
    selector: CandidateSelectionV2
    review: CandidateSelectionV2
    certificate: EvidenceClosureCertificateV2
    review_answer: str
    answer_validation: str
    downstream_usage: dict[str, Any]
    answer_call_count: int
    errors: list[str] = field(default_factory=list)


def _binding_top50(
    packet: RetrievalEvidencePacketV1,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    rows = list((packet.gnn_subgraph.get("rrf_trace") or {}).get("frozen_unit_top50") or [])
    scores: dict[str, dict[str, float]] = {}
    best_rank: dict[str, dict[str, int]] = {}
    for offset, raw in enumerate(rows):
        row = dict(raw)
        binding = str(int(row.get("binding_index", offset)))
        for rank, node_id in enumerate(list(row.get("node_ids") or [])[:50], start=1):
            node_id = str(node_id)
            scores.setdefault(binding, {})[node_id] = (
                scores.setdefault(binding, {}).get(node_id, 0.0) + 1.0 / (60.0 + rank)
            )
            best_rank.setdefault(binding, {})[node_id] = min(
                best_rank.setdefault(binding, {}).get(node_id, rank), rank
            )
    output: dict[str, list[str]] = {}
    output_scores: dict[str, dict[str, float]] = {}
    bindings = list(packet.query_plan.get("action_bindings") or [])
    if not bindings:
        bindings = [{"binding_index": 0}]
    evaluation = [str(item.get("node_id") or "") for item in packet.evaluation_anchors]
    for offset, raw in enumerate(bindings):
        binding = str(int(dict(raw).get("binding_index", offset)))
        if scores.get(binding):
            output[binding] = sorted(
                scores[binding],
                key=lambda node_id: (
                    -scores[binding][node_id],
                    best_rank[binding][node_id],
                    node_id,
                ),
            )[:50]
            output_scores[binding] = {
                node_id: float(scores[binding][node_id])
                for node_id in output[binding]
            }
        else:
            fallback = list(packet.binding_candidate_ids.get(binding, [])) + evaluation
            output[binding] = _ordered_unique(fallback)[:50]
            output_scores[binding] = {
                node_id: 1.0 / (60.0 + rank)
                for rank, node_id in enumerate(output[binding], start=1)
            }
    return output, output_scores


def _entity_map(
    packet: RetrievalEvidencePacketV1,
    node_identities: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in list(packet.gnn_subgraph.get("candidate_entities") or []):
        item = copy.deepcopy(dict(raw))
        node_id = str(item.get("node_id") or "")
        if node_id:
            output[node_id] = item
    for raw in [*packet.selected_entities, *packet.target_anchors, *packet.evaluation_anchors]:
        item = copy.deepcopy(dict(raw))
        node_id = str(item.get("node_id") or "")
        if node_id:
            output.setdefault(node_id, item)
    for node_id, raw in (node_identities or {}).items():
        item = output.setdefault(str(node_id), {"node_id": str(node_id), "metadata": {}})
        for key, value in dict(raw).items():
            item.setdefault(key, value)
    return output


def _path_steps(
    path: Mapping[str, Any], edges: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    nodes = [str(item) for item in path.get("node_ids") or []]
    declared = [str(item) for item in path.get("relations") or []]
    result: list[dict[str, str]] = []
    for index, (left, right) in enumerate(zip(nodes, nodes[1:])):
        edge = next(
            (
                dict(item)
                for item in edges
                if str(item.get("source_id") or "") == left
                and str(item.get("target_id") or "") == right
            ),
            None,
        )
        traversed = "target_to_scope"
        if edge is None:
            edge = next(
                (
                    dict(item)
                    for item in edges
                    if str(item.get("source_id") or "") == right
                    and str(item.get("target_id") or "") == left
                ),
                None,
            )
            traversed = "scope_to_target"
        relation = str((edge or {}).get("relation") or "")
        if not relation and index < len(declared):
            relation = declared[index]
        result.append(
            {
                "from": left,
                "relation": relation,
                "to": right,
                "traversal": traversed,
            }
        )
    return result


def _dedup_paths(
    packet: RetrievalEvidencePacketV1,
    node_identities: Mapping[str, Mapping[str, Any]] | None,
    *,
    path_inventory: Sequence[Mapping[str, Any]] | None = None,
    require_authoritative: bool = True,
    use_declared_relations: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    edges = list(packet.gnn_subgraph.get("edges") or [])
    entities = _entity_map(packet, node_identities)
    source_paths = list(
        packet.typed_path_inventory if path_inventory is None else path_inventory
    )
    chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in source_paths:
        path = copy.deepcopy(dict(raw))
        if require_authoritative and not bool(path.get("authoritative", True)):
            continue
        if not bool(path.get("complete_typed_path")):
            continue
        steps = _path_steps(path, edges)
        if use_declared_relations:
            declared = [str(item) for item in list(path.get("relations") or [])]
            for index, step in enumerate(steps):
                if index < len(declared):
                    step["relation"] = declared[index]
        signature = tuple(
            (item["from"], item["relation"], item["to"], item["traversal"])
            for item in steps
        )
        key = (
            str(path.get("target_id") or ""),
            _optional_int(path.get("binding_index")),
            str(path.get("scope_id") or ""),
            tuple(str(item) for item in path.get("node_ids") or []),
            signature,
        )
        row = {
            **path,
            "steps": steps,
            "logical_key_sha256": sha256_json(key),
        }
        previous = chosen.get(key)
        if previous is None or (
            float(row.get("score") or 0.0), str(row.get("path_id") or "")
        ) > (
            float(previous.get("score") or 0.0), str(previous.get("path_id") or "")
        ):
            chosen[key] = row
    result = sorted(
        chosen.values(),
        key=lambda row: (
            _optional_int(row.get("binding_index")) or 0,
            str(row.get("target_id") or ""),
            -float(row.get("score") or 0.0),
            str(row.get("path_id") or ""),
        ),
    )
    for index, row in enumerate(result, start=1):
        scope_id = str(row.get("scope_id") or "")
        scope = entities.get(scope_id, {})
        row["evidence_id"] = f"path:{index:04d}"
        row["scope_endpoint_label"] = str(scope.get("label") or scope_id)
    return result, len(source_paths)


def _scope_required(compact_plan: Mapping[str, Any]) -> bool:
    scope = dict(compact_plan.get("scope") or {})
    return any(
        scope.get(key) not in (None, "", [], {})
        for key in ("storey", "room", "room_names", "predicates")
    )


def _relation_requirement_matches_path(
    relation: Mapping[str, Any], path: Mapping[str, Any]
) -> bool:
    required = _normal(relation.get("relation")).replace(" ", "_")
    expected_direction = str(relation.get("direction") or "unspecified")
    if required in {"nearest", "adjacent_to"}:
        return False
    for step in list(path.get("steps") or []):
        observed = _normal(step.get("relation")).replace(" ", "_")
        traversal = str(step.get("traversal") or "")
        semantic_match = observed == required
        if required == "contains":
            semantic_match = semantic_match or (
                observed == "part_of" and traversal == "target_to_scope"
            )
        elif required == "part_of":
            semantic_match = semantic_match or (
                observed == "contains" and traversal == "scope_to_target"
            )
        if not semantic_match:
            continue
        if expected_direction in {"", "unspecified", "symmetric"}:
            return True
        if required == "contains" and observed == "part_of":
            observed_direction = "scope_to_target"
        else:
            observed_direction = traversal
        if expected_direction == observed_direction:
            return True
    return False


def _path_context_match(
    path: Mapping[str, Any],
    plan: Mapping[str, Any],
    raw_plan: Mapping[str, Any],
) -> tuple[bool, bool]:
    scope_match = True
    if _scope_required(plan):
        scope_match = str(path.get("scope_match_mode") or "") == "exact-typed-scope"
    raw_refs = [dict(item) for item in list(raw_plan.get("relation_references") or [])]
    if raw_refs:
        reference_ids = {
            str(node_id)
            for item in raw_refs
            for node_id in list(item.get("node_ids") or [])
        }
        if reference_ids and str(path.get("scope_id") or "") not in reference_ids:
            scope_match = False
    relations = list(plan.get("relations") or [])
    relation_match = all(
        _relation_requirement_matches_path(relation, path) for relation in relations
    ) if relations else True
    return scope_match, relation_match


def _candidate_facts(entity: Mapping[str, Any]) -> dict[str, str]:
    metadata = dict(entity.get("metadata") or {})
    return {
        "label": str(entity.get("label") or metadata.get("name") or ""),
        "ifc_class": str(entity.get("ifc_class") or metadata.get("ifc_class") or ""),
        "kind": _candidate_level(metadata),
        "role": str(metadata.get("role") or ""),
        "domain": str(metadata.get("domain") or metadata.get("object_domain") or ""),
        "family": str(metadata.get("family") or ""),
        "type": str(metadata.get("type_name") or metadata.get("type") or ""),
        "storey": str(metadata.get("storey") or metadata.get("storey_or_floor") or ""),
        "system": str(metadata.get("system_category") or ""),
        "function": str(metadata.get("function_type") or ""),
    }


def _target_match(target: Mapping[str, Any], facts: Mapping[str, str]) -> dict[str, str]:
    semantic = [
        facts.get("label", ""),
        facts.get("role", ""),
        facts.get("family", ""),
        facts.get("type", ""),
    ]
    return {
        "kind": _match_state([str(target.get("kind") or "")], [facts.get("kind", "")]),
        "ifc_class": _match_state(
            [str(target.get("ifc_class") or "")], [facts.get("ifc_class", "")]
        ),
        "role": _match_state(list(target.get("roles") or []), semantic),
        "name": _match_state(list(target.get("names") or []), semantic),
        "domain": _match_state(
            list(target.get("domains") or []), [facts.get("domain", ""), *semantic]
        ),
        "function": _match_state(
            list(target.get("function_types") or []),
            [facts.get("function", "")],
        ),
        "system": _match_state(
            list(target.get("system_categories") or []),
            [facts.get("system", "")],
        ),
    }


def _candidate_closes_scope(
    facts: Mapping[str, str], compact_plan: Mapping[str, Any]
) -> bool:
    """A space target can itself be the exact typed scope endpoint."""

    if _normal(facts.get("kind")) != "space":
        return False
    scope = dict(compact_plan.get("scope") or {})
    checks: list[bool] = []
    if scope.get("storey"):
        checks.append(
            _match_state([str(scope["storey"])], [facts.get("storey", "")]) == "match"
        )
    direct_names = [str(scope.get("room") or ""), *list(scope.get("room_names") or [])]
    if any(_known(item) for item in direct_names):
        checks.append(
            _match_state(
                direct_names,
                [facts.get("label", ""), facts.get("family", ""), facts.get("type", "")],
            )
            == "match"
        )
    for raw in list(scope.get("predicates") or []):
        predicate = dict(raw)
        values = list(predicate.get("values") or [])
        if not values:
            continue
        name = str(predicate.get("predicate") or "")
        observed = (
            [facts.get("storey", "")]
            if name == "storey"
            else [facts.get("label", ""), facts.get("family", ""), facts.get("type", "")]
        )
        checks.append(_match_state(values, observed) == "match")
    return bool(checks) and all(checks)


def _direct_constraint_evidence(
    packet: RetrievalEvidencePacketV1,
    *,
    binding_index: int,
    node_id: str,
    relations_required: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for raw in list((packet.operator_result or {}).get("candidates") or []):
        candidate = dict(raw)
        if str(candidate.get("node_id") or "") != node_id:
            continue
        metadata = dict(candidate.get("metadata") or {})
        constraint_status = dict(metadata.get("_constraint_status") or {})
        binding_rows = [
            dict(item)
            for item in list(constraint_status.get("bindings") or [])
            if int(dict(item).get("binding_index", -1)) == binding_index
        ]
        if not binding_rows or not all(
            str(item.get("overall") or "") == "pass" for item in binding_rows
        ):
            continue
        constraints = dict(binding_rows[0].get("constraints") or {})
        if constraints.get("scope") not in {"pass", None}:
            continue
        provenance = dict(metadata.get("_operator_scope_provenance") or {})
        observed_relation = str(provenance.get("relation") or "")
        if relations_required:
            required = {
                _normal(item.get("relation")).replace(" ", "_")
                for item in relations_required
            }
            if observed_relation not in required and not (
                observed_relation == "part_of" and "contains" in required
            ):
                continue
        return {
            "source": "bounded_operator_constraint_matrix",
            "scope": constraints.get("scope", "pass"),
            "relation": observed_relation,
            "scope_ids": _ordered_unique(provenance.get("scope_ids") or []),
        }
    return None


def _operator_candidate_ids(packet: RetrievalEvidencePacketV1) -> list[str]:
    return _ordered_unique(
        str(item.get("node_id") or "")
        for item in list((packet.operator_result or {}).get("candidates") or [])
    )


def build_candidate_ledger_v2(
    packet: RetrievalEvidencePacketV1,
    *,
    node_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> CandidateLedgerV2:
    if not packet.verify():
        raise ValueError("RetrievalEvidencePacketV1 self-hash mismatch")
    if len(packet.target_anchors) > 20 or len(packet.evaluation_anchors) > 50:
        raise ValueError("sealed top-20/top-50 anchor boundary is invalid")

    compact = compact_query_plan_v2(packet.query_plan)
    top50, top50_scores = _binding_top50(packet)
    top20 = {
        key: _ordered_unique(values)[:20]
        for key, values in packet.binding_candidate_ids.items()
    }
    entities = _entity_map(packet, node_identities)
    paths, raw_path_count = _dedup_paths(packet, node_identities)
    paths_by_binding_target: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in paths:
        key = (
            str(_optional_int(path.get("binding_index")) or 0),
            str(path.get("target_id") or ""),
        )
        paths_by_binding_target.setdefault(key, []).append(path)
    operator_ids = set(_operator_candidate_ids(packet))
    operator = str(compact.get("operator") or "lookup")

    binding_by_index = {
        str(int(row["binding_index"])): row for row in compact["bindings"]
    }
    candidates: list[dict[str, Any]] = []
    for binding_key, binding in binding_by_index.items():
        ranked = top50.get(binding_key, [])
        direct_ids = set(top20.get(binding_key, []))
        for rank, node_id in enumerate(ranked, start=1):
            entity = entities.get(node_id, {"node_id": node_id, "metadata": {}})
            facts = _candidate_facts(entity)
            matches = _target_match(dict(binding.get("target") or {}), facts)
            candidate_paths = paths_by_binding_target.get((binding_key, node_id), [])
            matched_paths: list[tuple[dict[str, Any], bool, bool]] = []
            for path in candidate_paths:
                scope_match, relation_match = _path_context_match(
                    path, compact, packet.query_plan
                )
                if scope_match and relation_match:
                    matched_paths.append((path, scope_match, relation_match))
            matched_paths.sort(
                key=lambda item: (
                    -float(item[0].get("score") or 0.0),
                    str(item[0].get("path_id") or ""),
                )
            )
            best = matched_paths[0] if matched_paths else None

            reasons: list[str] = []
            # Only graph-level type contradictions are deterministic hard
            # rejects. Name/role/domain/function/system metadata may be sparse
            # or classifier-derived; their states remain visible to the LLM
            # adjudicator instead of silently deleting a candidate.
            for field_name in ("kind", "ifc_class"):
                if matches.get(field_name) == "mismatch":
                    reasons.append(f"target_{field_name}_mismatch")
            if rank > 20 and best is None:
                reasons.append("rank_21_50_requires_complete_authoritative_typed_path")
            contextual = _scope_required(compact) or bool(compact.get("relations"))
            self_scope = _candidate_closes_scope(facts, compact)
            direct_constraint = _direct_constraint_evidence(
                packet,
                binding_index=int(binding_key),
                node_id=node_id,
                relations_required=list(compact.get("relations") or []),
            )
            context_closed = best is not None or self_scope or direct_constraint is not None
            if contextual and operator not in _OPERATOR_BOUND and not context_closed:
                reasons.append("scope_relation_endpoint_not_closed")
            if operator in _OPERATOR_BOUND and node_id not in operator_ids:
                reasons.append("outside_bounded_operator_result")
            global_id = str(entity.get("global_id") or "")
            action = str(binding.get("action") or "")
            if action and not global_id:
                reasons.append("missing_ifc_global_id")

            alias = f"B{int(binding_key):02d}C{rank:02d}"
            score = float(top50_scores.get(binding_key, {}).get(node_id, 0.0))
            path_payload = None
            evidence_ids = [f"candidate:{alias}"]
            if best is not None:
                path, scope_match, relation_match = best
                evidence_ids.append(str(path["evidence_id"]))
                path_payload = {
                    "evidence_id": str(path["evidence_id"]),
                    "path_id": str(path.get("path_id") or ""),
                    "target_id": node_id,
                    "scope_id": str(path.get("scope_id") or ""),
                    "scope_endpoint": str(path.get("scope_endpoint_label") or ""),
                    "node_ids": list(path.get("node_ids") or []),
                    "directed_relations": [
                        f"{item['traversal']}:{item['relation']}"
                        for item in list(path.get("steps") or [])
                    ],
                    "scope_match": "exact" if scope_match else "mismatch",
                    "relation_match": "exact" if relation_match else "mismatch",
                    "authoritative": True,
                    "complete": True,
                    "score": float(path.get("score") or 0.0),
                }
            elif self_scope:
                evidence_ids.append(f"scope-self:{alias}")
                path_payload = {
                    "evidence_id": f"scope-self:{alias}",
                    "path_id": "",
                    "target_id": node_id,
                    "scope_id": node_id,
                    "scope_endpoint": facts.get("label", ""),
                    "node_ids": [node_id],
                    "directed_relations": [],
                    "scope_match": "exact-self",
                    "relation_match": "not_required",
                    "authoritative": True,
                    "complete": True,
                    "score": score,
                }
            elif direct_constraint is not None:
                evidence_ids.append(f"operator-constraint:{alias}")
                path_payload = {
                    "evidence_id": f"operator-constraint:{alias}",
                    "path_id": "",
                    "target_id": node_id,
                    "scope_id": "",
                    "scope_endpoint": "bounded operator constraint",
                    "node_ids": [node_id],
                    "directed_relations": [
                        value
                        for value in [str(direct_constraint.get("relation") or "")]
                        if value
                    ],
                    "scope_match": "exact-operator-constraint",
                    "relation_match": "exact" if compact.get("relations") else "not_required",
                    "authoritative": True,
                    "complete": True,
                    "score": score,
                }
            candidates.append(
                {
                    "candidate_alias": alias,
                    "binding_index": int(binding_key),
                    "node_id": node_id,
                    "global_id": global_id,
                    "rank": rank,
                    "rrf_score": score,
                    "operational_top20": node_id in direct_ids or rank <= 20,
                    "promotion_source": "top20" if rank <= 20 else "top50_typed_path",
                    "eligibility": "reviewable" if not reasons else "deterministic_reject",
                    "hard_reject_reasons": _ordered_unique(reasons),
                    "facts": facts,
                    "target_match": matches,
                    "best_path": path_payload,
                    "evidence_ids": evidence_ids,
                }
            )

    aliases = [row["candidate_alias"] for row in candidates]
    if len(aliases) != len(set(aliases)):
        raise RuntimeError("candidate aliases are not unique")
    if any(len(values) > 50 for values in top50.values()):
        raise RuntimeError("binding top-50 pool exceeds the frozen limit")
    ledger = CandidateLedgerV2(
        packet_sha256=packet.packet_sha256,
        compact_plan=compact,
        candidates=candidates,
        path_counts={
            "raw": raw_path_count,
            "logical_unique": len(paths),
            "duplicates_removed": raw_path_count - len(paths),
        },
        top20_by_binding=top20,
        top50_by_binding=top50,
        global_conflicting_extras_diagnostic=bool(
            packet.target_audit.get("conflicting_extras")
        ),
    ).seal()
    if not ledger.verify():
        raise RuntimeError("candidate ledger self-hash failed")
    return ledger


def normalize_provider_selection_v2(
    raw: Mapping[str, Any],
    ledger: CandidateLedgerV2,
    *,
    phase: Literal["selector", "review"],
) -> CandidateSelectionV2:
    if not ledger.verify():
        raise ValueError("candidate ledger is not sealed")
    by_alias = {row["candidate_alias"]: row for row in ledger.candidates}
    reviewable = {
        alias
        for alias, row in by_alias.items()
        if row["eligibility"] == "reviewable"
    }
    observed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for raw_decision in list(raw.get("decisions") or []):
        item = dict(raw_decision)
        alias = str(item.get("candidate_id") or item.get("candidate_alias") or "")
        decision = str(getattr(item.get("decision"), "value", item.get("decision") or "")).casefold()
        if decision not in {"select", "reject", "uncertain"}:
            errors.append(f"invalid_decision:{alias}:{decision}")
            continue
        if alias not in by_alias:
            errors.append(f"unknown_candidate:{alias}")
            continue
        if alias in observed:
            errors.append(f"duplicate_candidate:{alias}")
            continue
        if by_alias[alias]["eligibility"] != "reviewable" and decision != "reject":
            errors.append(f"selected_deterministic_reject:{alias}")
            decision = "reject"
        evidence_ids = _ordered_unique(item.get("evidence_ids") or [])
        allowed_evidence = set(by_alias[alias]["evidence_ids"])
        invalid_evidence = sorted(set(evidence_ids) - allowed_evidence)
        if invalid_evidence:
            errors.append(f"invalid_evidence:{alias}:{','.join(invalid_evidence)}")
            evidence_ids = [value for value in evidence_ids if value in allowed_evidence]
        observed[alias] = {
            "candidate_id": alias,
            "decision": decision,
            "evidence_ids": evidence_ids,
            "reason": str(item.get("reason") or "")[:400],
            "provider_explicit": True,
        }

    decisions: list[dict[str, Any]] = []
    for alias, row in by_alias.items():
        if row["eligibility"] != "reviewable":
            decisions.append(
                {
                    "candidate_id": alias,
                    "decision": "reject",
                    "evidence_ids": list(row["evidence_ids"]),
                    "reason": "deterministic eligibility rejection",
                    "provider_explicit": alias in observed,
                }
            )
            continue
        decision = observed.get(alias)
        if decision is None:
            decisions.append(
                {
                    "candidate_id": alias,
                    "decision": "uncertain",
                    "evidence_ids": [],
                    "reason": "provider omitted reviewable candidate",
                    "provider_explicit": False,
                }
            )
        else:
            decisions.append(decision)
    explicit_reviewable = {
        alias for alias in observed if alias in reviewable
    }
    return CandidateSelectionV2(
        ledger_sha256=ledger.ledger_sha256,
        phase=phase,
        decisions=decisions,
        provider_complete=explicit_reviewable == reviewable and not errors,
        errors=errors,
        summary=str(raw.get("summary") or raw.get("review_summary") or "")[:1000],
    ).seal()


def adjudicate_closure_v2(
    packet: RetrievalEvidencePacketV1,
    ledger: CandidateLedgerV2,
    selector: CandidateSelectionV2,
    review: CandidateSelectionV2,
) -> EvidenceClosureCertificateV2:
    if not packet.verify() or not ledger.verify() or not selector.verify() or not review.verify():
        raise ValueError("v2 adjudication inputs are not hash-valid")
    if selector.ledger_sha256 != ledger.ledger_sha256 or review.ledger_sha256 != ledger.ledger_sha256:
        raise ValueError("selection lineage does not match candidate ledger")
    rows = {row["candidate_alias"]: row for row in ledger.candidates}
    selector_decisions = {
        row["candidate_id"]: row["decision"] for row in selector.decisions
    }
    review_decisions = {
        row["candidate_id"]: row["decision"] for row in review.decisions
    }
    repairs = sum(
        selector_decisions.get(alias) != review_decisions.get(alias) for alias in rows
    )
    errors = [*selector.errors, *review.errors]
    certified_ids: dict[str, list[str]] = {}
    certified_aliases: dict[str, list[str]] = {}
    binding_status: dict[str, dict[str, Any]] = {}
    operator = str(ledger.compact_plan.get("operator") or "lookup")
    operator_ids = _operator_candidate_ids(packet)

    for binding in list(ledger.compact_plan.get("bindings") or []):
        key = str(int(binding["binding_index"]))
        binding_rows = [row for row in rows.values() if str(row["binding_index"]) == key]
        reviewable = [row for row in binding_rows if row["eligibility"] == "reviewable"]
        selected = [
            row for row in reviewable if review_decisions.get(row["candidate_alias"]) == "select"
        ]
        uncertain = [
            row for row in reviewable if review_decisions.get(row["candidate_alias"]) == "uncertain"
        ]
        policy = str(binding.get("cardinality") or "single")
        status = "supported"
        reason = "selection closed over the sealed candidate pool"

        if operator in _OPERATOR_BOUND:
            by_node = {row["node_id"]: row for row in reviewable}
            selected = [by_node[node_id] for node_id in operator_ids if node_id in by_node]
            if not selected:
                status = "graph_or_schema_absent"
                reason = "bounded operator result has no constraint-valid sealed candidate"
            elif policy not in _SET_CARDINALITIES:
                selected = selected[:1]
        elif policy in _SET_CARDINALITIES:
            if uncertain or not review.provider_complete:
                status = "incomplete_candidate_classification"
                reason = "set-valued cardinality requires explicit select/reject for every candidate"
            elif not selected:
                status = "graph_or_schema_absent"
                reason = "no selected target remains in the sealed candidate pool"
        else:
            if len(selected) != 1:
                status = "cardinality_conflict" if len(selected) > 1 else "graph_or_schema_absent"
                reason = "single cardinality requires exactly one selected target"

        if any(row["eligibility"] != "reviewable" for row in selected):
            status = "candidate_graph_contradiction"
            reason = "selected target violates deterministic candidate eligibility"
        aliases = [row["candidate_alias"] for row in selected] if status == "supported" else []
        node_ids = [row["node_id"] for row in selected] if status == "supported" else []
        certified_aliases[key] = aliases
        certified_ids[key] = node_ids
        binding_status[key] = {
            "status": status,
            "reason": reason,
            "cardinality": policy,
            "candidate_count": len(binding_rows),
            "reviewable_count": len(reviewable),
            "selected_count": len(aliases),
            "uncertain_count": len(uncertain),
            "all_reviewable_candidates_classified": review.provider_complete,
        }

    if errors:
        closure_status: ClosureStatusV2 = "abstain"
        stop_reason = "invalid_provider_selection"
    elif any(row["status"] != "supported" for row in binding_status.values()):
        closure_status = "abstain"
        stop_reason = next(
            row["status"] for row in binding_status.values() if row["status"] != "supported"
        )
    else:
        closure_status = "pass"
        stop_reason = "all_bindings_certified"
    return EvidenceClosureCertificateV2(
        packet_sha256=packet.packet_sha256,
        ledger_sha256=ledger.ledger_sha256,
        selector_sha256=selector.selection_sha256,
        review_sha256=review.selection_sha256,
        certified_target_ids=certified_ids,
        certified_candidate_aliases=certified_aliases,
        binding_status=binding_status,
        closure_status=closure_status,
        stop_reason=stop_reason,
        selection_repair_count=repairs,
        global_conflicting_extras_diagnostic=ledger.global_conflicting_extras_diagnostic,
        validation_errors=_ordered_unique(errors),
    ).seal()


def certified_answers_v2(
    packet: RetrievalEvidencePacketV1,
    ledger: CandidateLedgerV2,
    certificate: EvidenceClosureCertificateV2,
) -> tuple[str, str]:
    if certificate.closure_status != "pass":
        abstention = "I cannot determine the answer from the IFC knowledge graph."
        return abstention, abstention
    rows = {row["candidate_alias"]: row for row in ledger.candidates}
    bindings = list(ledger.compact_plan.get("bindings") or [])
    alias_calls: list[str] = []
    guid_calls: list[str] = []
    for binding in bindings:
        key = str(int(binding["binding_index"]))
        action = str(binding.get("action") or "")
        if not action:
            continue
        for alias in certificate.certified_candidate_aliases.get(key, []):
            row = rows[alias]
            global_id = str(row.get("global_id") or "")
            if not global_id:
                abstention = "I cannot determine the answer from the IFC knowledge graph."
                return abstention, abstention
            alias_calls.append(f"{action}({alias})")
            guid_calls.append(f"{action}({global_id})")
    if alias_calls:
        return ", ".join(alias_calls), ", ".join(guid_calls)
    answer = str(packet.deterministic_answer or "").strip()
    if answer:
        return answer, answer
    abstention = "I cannot determine the answer from the IFC knowledge graph."
    return abstention, abstention


def validate_review_answer_v2(
    generated_answer: str,
    *,
    expected_alias_answer: str,
    certified_answer: str,
    certificate: EvidenceClosureCertificateV2,
) -> tuple[str, str]:
    if certificate.closure_status != "pass":
        return "I cannot determine the answer from the IFC knowledge graph.", "closure_not_passed"
    generated = str(generated_answer or "").strip()
    if not generated:
        return certified_answer, "empty_answer_reverted_to_certificate"
    expected_alias = _ACTION_RE.findall(expected_alias_answer)
    expected_guid = _ACTION_RE.findall(certified_answer)
    observed = _ACTION_RE.findall(generated)
    if expected_alias and observed != expected_alias and observed != expected_guid:
        return certified_answer, "answer_action_or_candidate_mismatch_reverted_to_certificate"
    return certified_answer if expected_alias else generated, "validated"


def selection_repair_trace_v2(
    selector: CandidateSelectionV2,
    review: CandidateSelectionV2,
    certificate: EvidenceClosureCertificateV2,
) -> dict[str, Any]:
    before = {row["candidate_id"]: row["decision"] for row in selector.decisions}
    after = {row["candidate_id"]: row["decision"] for row in review.decisions}
    changed = [alias for alias in before if before.get(alias) != after.get(alias)]
    payload = {
        "schema_version": "closure-selection-repair-trace-v2",
        "selector_sha256": selector.selection_sha256,
        "review_sha256": review.selection_sha256,
        "certificate_sha256": certificate.certificate_sha256,
        "repair_count": int(bool(changed)),
        "changed_candidate_count": len(changed),
        "changed_candidate_aliases": changed,
        "maximum_repairs": 1,
    }
    payload["trace_sha256"] = sha256_json(payload)
    return payload


def v2_synthetic_contract_checks() -> dict[str, bool]:
    """Pure checks used by paid-run preflight; no benchmark data or provider calls."""

    contains = {"relation": "contains", "direction": "scope_to_target"}
    part_of_path = {
        "steps": [
            {
                "from": "target",
                "relation": "part_of",
                "to": "scope",
                "traversal": "target_to_scope",
            }
        ]
    }
    wrong_direction = {
        "steps": [
            {
                "from": "target",
                "relation": "part_of",
                "to": "scope",
                "traversal": "scope_to_target",
            }
        ]
    }
    return {
        "contains_inverse_direction_matches": _relation_requirement_matches_path(
            contains, part_of_path
        ),
        "wrong_direction_rejected": not _relation_requirement_matches_path(
            contains, wrong_direction
        ),
        "target_match_changes_eligibility_signal": _match_state(
            ["sensor"], ["air terminal"]
        )
        == "mismatch",
        "unknown_is_distinct_from_missing_string": _match_state(["sensor"], [])
        == "unknown",
        "compact_plan_forbidden_fields_absent": all(
            token not in canonical_json(
                compact_query_plan_v2(
                    {
                        "operator": "path",
                        "action_bindings": [],
                        "mention_links": [{"node_ids": ["forbidden"]}],
                        "rationale": "forbidden",
                    }
                )
            )
            for token in ("mention_links", "rationale", "forbidden")
        ),
    }


__all__ = [
    "CandidateLedgerV2",
    "CandidateSelectionV2",
    "EvidenceClosureCertificateV2",
    "RetrievalClosureResultV2",
    "adjudicate_closure_v2",
    "build_candidate_ledger_v2",
    "certified_answers_v2",
    "compact_query_plan_v2",
    "normalize_provider_selection_v2",
    "selection_repair_trace_v2",
    "v2_synthetic_contract_checks",
    "validate_review_answer_v2",
]
