from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ._closure_ledger import (
    _binding_top50,
    _candidate_facts,
    _dedup_paths,
    _entity_map,
    _match_state,
    _normal,
    _ordered_unique,
    compact_query_plan_v2,
)
from ._closure_packet import (
    RetrievalEvidencePacketV1,
    arm_path_inventory,
    sha256_json,
)
from .models import RetrievalClosureArm

SelectionModeV3 = Literal[
    "one_graph_node",
    "best_equivalence_class",
    "all_matching",
    "bounded_operator",
]
ClosureStatusV3 = Literal["pass", "abstain"]

_GUID_RE = re.compile(r"(?<![0-9A-Za-z_$])([0-9A-Za-z_$]{22})(?![0-9A-Za-z_$])")
_IFC_NODE_ID_RE = re.compile(
    r"(?<![0-9A-Za-z_$])ifc_[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])",
    re.IGNORECASE,
)
_ACTIONS = frozenset({"Navigate", "Inspect", "Scan"})
_SELECTION_MODES = frozenset(
    {
        "one_graph_node",
        "best_equivalence_class",
        "all_matching",
        "bounded_operator",
    }
)
_BOUNDED_OPERATORS = frozenset({"nearest", "argmax"})
_SET_CARDINALITIES = frozenset({"all", "count", "distinct", "group_count"})
_RELATIONS = frozenset(
    {
        "",
        "contains",
        "part_of",
        "adjacent_to",
        "serves",
        "assigned_to_system",
        "controls",
        "feeds",
        "connected_to",
        "unconnected",
        "nearest",
    }
)
_DIRECTIONS = frozenset(
    {"", "unspecified", "symmetric", "target_to_scope", "scope_to_target"}
)
_UNKNOWN_VALUES = frozenset(
    {"", "unknown", "missing", "none", "not available", "no local text"}
)


def _known(value: Any) -> bool:
    return _normal(value) not in _UNKNOWN_VALUES


def contract_question_view(question: str) -> str:
    """Prevent a question's GUID from leaking into identifier-free semantic terms.

    Exact source identities remain in the sealed QueryPlan and graph references.
    This view is only for the intent contract compiler, never for graph lookup.
    """
    aliases: dict[str, str] = {}
    pattern = re.compile(r"(?<![0-9A-Za-z_$])(?:ifc_)?[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])")
    def replace_identifier(match: re.Match[str]) -> str:
        identity = match.group().removeprefix("ifc_")
        if identity not in aliases:
            aliases[identity] = f"[entity reference {len(aliases) + 1}]"
        return aliases[identity]
    return pattern.sub(replace_identifier, question)


def _clean_text(value: Any, *, errors: list[str], field_name: str) -> str:
    text = " ".join(str(value or "").split())
    if "ifc_" in text.casefold() or _GUID_RE.search(text):
        errors.append(f"forbidden_identifier_in_{field_name}")
        return ""
    return text[:500]


def _clean_explanatory_text(value: Any) -> str:
    """Redact identifiers from prose fields that never drive selection.

    A user may explicitly include an IFC GUID in a what-if question. Models
    sometimes repeat that identifier in ``scope_phrase`` or summary prose even
    though the sealed compact plan already owns scope and executable bindings.
    Dropping the identifier is safer than treating harmless prose as a new
    executable target or failing the entire closure.
    """

    text = " ".join(str(value or "").split())
    text = _IFC_NODE_ID_RE.sub("[identifier omitted]", text)
    text = _GUID_RE.sub("[identifier omitted]", text)
    text = re.sub(
        r"(?<![0-9A-Za-z_$])ifc_[^\s,.;:()]+",
        "[identifier omitted]",
        text,
        flags=re.IGNORECASE,
    )
    return text[:500]


def _clean_terms(
    values: Iterable[Any], *, errors: list[str], field_name: str, limit: int = 16
) -> list[str]:
    output: list[str] = []
    for value in values:
        text = _clean_text(value, errors=errors, field_name=field_name)
        if text:
            output.append(text)
    return _ordered_unique(output)[:limit]


def _fallback_selection_mode(compact: Mapping[str, Any], binding: Mapping[str, Any]) -> str:
    operator = str(compact.get("operator") or "lookup")
    cardinality = str(binding.get("cardinality") or "single")
    if operator in _BOUNDED_OPERATORS:
        return "bounded_operator"
    if cardinality in _SET_CARDINALITIES:
        return "all_matching"
    # A natural-language singular denotes one semantic referent, not
    # necessarily one IFC row.  Repeated leaves/panels/components therefore
    # remain eligible as one evidence-equivalence class.
    return "best_equivalence_class"


@dataclass(slots=True)
class RetrievalIntentContractV3:
    compact_plan_sha256: str
    bindings: list[dict[str, Any]]
    provider_summary: str = ""
    validation_errors: list[str] = field(default_factory=list)
    schema_version: str = "retrieval-intent-contract-v3"
    contract_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("contract_sha256", None)
        return payload

    def seal(self) -> RetrievalIntentContractV3:
        self.contract_sha256 = sha256_json(self._payload())
        return self

    def verify(self) -> bool:
        return bool(self.contract_sha256) and self.contract_sha256 == sha256_json(
            self._payload()
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.contract_sha256:
            self.seal()
        return asdict(self)

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "retrieval-intent-contract-model-view-v3",
            "contract_sha256": self.contract_sha256,
            "bindings": copy.deepcopy(self.bindings),
        }


def normalize_intent_contract_v3(
    raw: Mapping[str, Any], compact_plan: Mapping[str, Any]
) -> RetrievalIntentContractV3:
    """Validate a question-only contract repair without candidate identifiers."""

    compact_hash = sha256_json(compact_plan)
    errors: list[str] = []
    expected = [dict(item) for item in list(compact_plan.get("bindings") or [])]
    raw_rows = [dict(item) for item in list(raw.get("bindings") or []) if isinstance(item, Mapping)]
    by_index: dict[int, dict[str, Any]] = {}
    expected_indices = {int(item["binding_index"]) for item in expected}
    for item in raw_rows:
        try:
            index = int(str(item.get("binding_index")))
        except (TypeError, ValueError):
            errors.append("invalid_binding_index")
            continue
        if index not in expected_indices:
            # A structured provider can split an explanatory request (for
            # example "which rooms are impacted") into an extra conceptual
            # step. The sealed compact plan is authoritative: reject that row
            # without allowing it to invalidate or extend executable actions.
            continue
        if index in by_index:
            errors.append(f"duplicate_binding_index:{index}")
            continue
        by_index[index] = item

    bindings: list[dict[str, Any]] = []
    for fallback in expected:
        index = int(fallback["binding_index"])
        proposed = by_index.get(index, {})
        action = str(proposed.get("action") or fallback.get("action") or "")
        if action not in _ACTIONS:
            errors.append(f"invalid_action:{index}")
            action = str(fallback.get("action") or "")
        mode = str(
            proposed.get("selection_mode")
            or _fallback_selection_mode(compact_plan, fallback)
        )
        if mode not in _SELECTION_MODES:
            errors.append(f"invalid_selection_mode:{index}")
            mode = _fallback_selection_mode(compact_plan, fallback)

        fallback_target = dict(fallback.get("target") or {})
        target_terms = _clean_terms(
            list(proposed.get("target_terms") or [])
            or [
                *list(fallback_target.get("names") or []),
                *list(fallback_target.get("roles") or []),
                *list(fallback_target.get("domains") or []),
            ],
            errors=errors,
            field_name=f"target_terms_{index}",
        )
        scope = dict(compact_plan.get("scope") or {})
        fallback_scope_terms = [
            str(scope.get("storey") or ""),
            str(scope.get("room") or ""),
            *list(scope.get("room_names") or []),
            *[
                value
                for predicate in list(scope.get("predicates") or [])
                for value in list(dict(predicate).get("values") or [])
            ],
        ]
        scope_terms = _clean_terms(
            list(proposed.get("scope_terms") or []) or fallback_scope_terms,
            errors=errors,
            field_name=f"scope_terms_{index}",
        )
        relations = [dict(item) for item in list(compact_plan.get("relations") or [])]
        fallback_relation = str(relations[0].get("relation") or "") if relations else ""
        fallback_direction = str(relations[0].get("direction") or "") if relations else ""
        relation = _normal(proposed.get("relation") or fallback_relation).replace(" ", "_")
        direction = _normal(proposed.get("direction") or fallback_direction).replace(" ", "_")
        if relation not in _RELATIONS:
            errors.append(f"invalid_relation:{index}")
            relation = _normal(fallback_relation).replace(" ", "_")
        if direction not in _DIRECTIONS:
            errors.append(f"invalid_direction:{index}")
            direction = _normal(fallback_direction).replace(" ", "_")

        target_kind = _normal(
            proposed.get("target_kind") or fallback_target.get("kind") or ""
        )
        if target_kind not in {"", "space", "object", "system", "function"}:
            errors.append(f"invalid_target_kind:{index}")
            target_kind = _normal(fallback_target.get("kind") or "")
        target_ifc_class = _clean_text(
            proposed.get("target_ifc_class")
            or fallback_target.get("ifc_class")
            or "",
            errors=errors,
            field_name=f"target_ifc_class_{index}",
        )
        bindings.append(
            {
                "binding_index": index,
                "action": action,
                "selection_mode": mode,
                "target_phrase": _clean_text(
                    proposed.get("target_phrase") or "",
                    errors=errors,
                    field_name=f"target_phrase_{index}",
                ),
                "target_terms": target_terms,
                "target_kind": target_kind,
                "target_ifc_class": target_ifc_class,
                "scope_phrase": _clean_explanatory_text(
                    proposed.get("scope_phrase") or ""
                ),
                "scope_terms": scope_terms,
                "relation": relation,
                "direction": direction,
                "requires_exhaustive_set": bool(
                    proposed.get("requires_exhaustive_set")
                    or mode == "all_matching"
                ),
            }
        )

    if len(bindings) != len(expected):
        errors.append("binding_count_mismatch")
    summary = _clean_explanatory_text(raw.get("summary") or "")
    return RetrievalIntentContractV3(
        compact_plan_sha256=compact_hash,
        bindings=bindings,
        provider_summary=summary,
        validation_errors=_ordered_unique(errors),
    ).seal()


@dataclass(slots=True)
class CandidateGroupLedgerV3:
    packet_sha256: str
    compact_plan: dict[str, Any]
    contract: dict[str, Any]
    candidates: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    source_diagnostics: dict[str, Any]
    schema_version: str = "candidate-group-ledger-v3"
    ledger_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("ledger_sha256", None)
        return payload

    def seal(self) -> CandidateGroupLedgerV3:
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
        rows: list[dict[str, Any]] = []
        for group in self.groups:
            if group["eligibility"] != "reviewable":
                continue
            rows.append(
                {
                    "group_id": group["group_alias"],
                    "binding_index": group["binding_index"],
                    "member_count": group["member_count"],
                    "sample_labels": list(group["sample_labels"]),
                    "facts": copy.deepcopy(group["facts"]),
                    "target_match": copy.deepcopy(group["target_match"]),
                    "scope_contexts": copy.deepcopy(group["scope_contexts"]),
                    "source_summary": copy.deepcopy(group["source_summary"]),
                    "eligibility": group["eligibility"],
                    "evidence_ids": list(group["evidence_ids"]),
                }
            )
        return {
            "schema_version": "candidate-group-ledger-model-view-v3",
            "packet_sha256": self.packet_sha256,
            "ledger_sha256": self.ledger_sha256,
            "intent_contract": copy.deepcopy(self.contract),
            "candidate_union_contract": (
                "Frozen per-binding top-50 union bounded operator targets and complete "
                "authoritative typed-path targets; maximum 200 candidates per binding."
            ),
            "equivalence_contract": (
                "Selecting one group selects every member because members share target "
                "family/type/role/class and the same bounded scope/path signature."
            ),
            "groups": rows,
            "deterministic_reject_summary": [
                {
                    "binding_index": group["binding_index"],
                    "reason": reason,
                    "member_count": group["member_count"],
                }
                for group in self.groups
                if group["eligibility"] != "reviewable"
                for reason in group.get("model_exclusion_reasons", [])
            ],
        }


def _merge_entity(base: dict[str, Any], richer: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in dict(richer).items():
        if key == "metadata":
            metadata = dict(output.get("metadata") or {})
            metadata.update(copy.deepcopy(dict(value or {})))
            output["metadata"] = metadata
        elif value not in (None, "", [], {}):
            output[key] = copy.deepcopy(value)
    return output


def _operator_binding_row(entity: Mapping[str, Any], binding_index: int) -> dict[str, Any]:
    status = dict(dict(entity.get("metadata") or {}).get("_constraint_status") or {})
    for raw in list(status.get("bindings") or []):
        row = dict(raw)
        if int(row.get("binding_index", -1)) == int(binding_index):
            return row
    return {}


def _operator_scope_ids(entity: Mapping[str, Any]) -> list[str]:
    metadata = dict(entity.get("metadata") or {})
    provenance = dict(metadata.get("_operator_scope_provenance") or {})
    return _ordered_unique(provenance.get("scope_ids") or [])


def _path_relations(path: Mapping[str, Any]) -> list[str]:
    steps = list(path.get("steps") or [])
    if steps:
        return [
            f"{step.get('traversal') or 'unspecified'!s}:{step.get('relation') or ''!s}"
            for step in steps
        ]
    return [str(value) for value in list(path.get("relations") or [])]


def _scope_facts(scope_id: str, entities: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    entity = dict(entities.get(scope_id) or {})
    metadata = dict(entity.get("metadata") or {})
    return {
        "label": str(entity.get("label") or metadata.get("name") or ""),
        "kind": str(metadata.get("action_target_kind") or metadata.get("kind") or ""),
        "ifc_class": str(entity.get("ifc_class") or metadata.get("ifc_class") or ""),
        "storey": str(metadata.get("storey") or metadata.get("storey_or_floor") or ""),
    }


def _relation_context_match(required: str, relations: Sequence[str]) -> str:
    required = _normal(required).replace(" ", "_")
    if not required:
        return "not_required"
    if not relations:
        return "unknown"
    observed = {_normal(value).replace(" ", "_") for value in relations}
    if any(required in value or value.endswith(f":{required}") for value in observed):
        return "match"
    inverse = {"contains": "part_of", "part_of": "contains"}
    if required in inverse and any(inverse[required] in value for value in observed):
        return "match"
    return "mismatch"


def _candidate_group_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    facts = dict(candidate["facts"])
    contexts = list(candidate.get("scope_contexts") or [])
    context_signature = tuple(
        sorted(
            (
                str(context.get("scope_alias") or ""),
                tuple(context.get("directed_relations") or []),
                str(context.get("scope_label") or ""),
            )
            for context in contexts
        )
    )
    return (
        int(candidate["binding_index"]),
        str(candidate["eligibility"]),
        (
            str(candidate.get("context_certification") or "")
            if str(candidate.get("selection_mode") or "")
            in {"one_graph_node", "best_equivalence_class", "bounded_operator"}
            else ""
        ),
        _normal(facts.get("kind")),
        _normal(facts.get("ifc_class")),
        _normal(facts.get("role")),
        _normal(facts.get("domain")),
        _normal(facts.get("family")),
        _normal(facts.get("type")),
        context_signature,
    )


def build_candidate_group_ledger_v3(
    packet: RetrievalEvidencePacketV1,
    contract: RetrievalIntentContractV3,
    *,
    node_identities: Mapping[str, Mapping[str, Any]] | None = None,
    max_candidates_per_binding: int = 200,
    arm: RetrievalClosureArm = "typed_hierarchy",
) -> CandidateGroupLedgerV3:
    if not packet.verify() or not contract.verify():
        raise ValueError("v3 ledger requires sealed packet and contract")
    compact = compact_query_plan_v2(packet.query_plan)
    if sha256_json(compact) != contract.compact_plan_sha256:
        raise ValueError("v3 intent contract does not match compact QueryPlan")

    entities = _entity_map(packet, node_identities)
    operator_rows = [dict(item) for item in list((packet.operator_result or {}).get("candidates") or [])]
    for row in operator_rows:
        node_id = str(row.get("node_id") or "")
        if node_id:
            entities[node_id] = _merge_entity(entities.get(node_id, {"node_id": node_id}), row)
    for node_id, identity in dict(node_identities or {}).items():
        entities[str(node_id)] = _merge_entity(
            entities.get(str(node_id), {"node_id": str(node_id)}), identity
        )

    top50, top50_scores = _binding_top50(packet)
    top20 = {
        str(key): _ordered_unique(values)
        for key, values in dict(packet.binding_candidate_ids or {}).items()
    }
    arm_paths = arm_path_inventory(packet, arm)
    paths, raw_path_count = _dedup_paths(
        packet,
        node_identities,
        path_inventory=arm_paths,
        require_authoritative=False,
        use_declared_relations=arm == "relation_shuffled_control",
    )
    path_targets: dict[str, list[str]] = {}
    for path in paths:
        binding = path.get("binding_index")
        if binding is None:
            continue
        path_targets.setdefault(str(int(binding)), []).append(str(path.get("target_id") or ""))

    operator_by_binding: dict[str, list[str]] = {
        str(int(binding["binding_index"])): [] for binding in contract.bindings
    }
    for row in operator_rows:
        node_id = str(row.get("node_id") or "")
        status = dict(dict(row.get("metadata") or {}).get("_constraint_status") or {})
        matched_bindings = [
            str(int(item.get("binding_index")))
            for item in list(status.get("bindings") or [])
            if item.get("binding_index") is not None
        ]
        if not matched_bindings:
            matched_bindings = list(operator_by_binding)
        for binding in matched_bindings:
            if binding in operator_by_binding and node_id:
                operator_by_binding[binding].append(node_id)

    scope_ids = {
        str(path.get("scope_id") or "") for path in paths if path.get("scope_id")
    }
    scope_ids.update(
        scope_id for entity in entities.values() for scope_id in _operator_scope_ids(entity)
    )
    scope_aliases = {
        scope_id: f"S{index:03d}" for index, scope_id in enumerate(sorted(scope_ids), start=1)
    }

    candidates: list[dict[str, Any]] = []
    source_counts: dict[str, Any] = {
        "raw_typed_paths": raw_path_count,
        "deduplicated_typed_paths": len(paths),
        "bindings": {},
    }
    for binding_contract in contract.bindings:
        binding_index = int(binding_contract["binding_index"])
        key = str(binding_index)
        ranked = list(top50.get(key, []))
        operator_ids = _ordered_unique(operator_by_binding.get(key, []))
        typed_ids = _ordered_unique(path_targets.get(key, []))
        ordered_union = _ordered_unique([*ranked, *operator_ids, *typed_ids])
        truncated = len(ordered_union) > max_candidates_per_binding
        ordered_union = ordered_union[:max_candidates_per_binding]
        source_counts["bindings"][key] = {
            "top50_count": len(ranked),
            "operator_target_count": len(operator_ids),
            "typed_path_target_count": len(typed_ids),
            "union_before_cap": len(_ordered_unique([*ranked, *operator_ids, *typed_ids])),
            "union_after_cap": len(ordered_union),
            "truncated": truncated,
        }
        top_rank = {node_id: rank for rank, node_id in enumerate(ranked, start=1)}
        direct = set(top20.get(key, []))
        operator_set = set(operator_ids)
        path_set = set(typed_ids)
        authoritative_path_set = {
            str(path.get("target_id") or "")
            for path in paths
            if bool(path.get("authoritative", True))
        }
        for union_rank, node_id in enumerate(ordered_union, start=1):
            entity = entities.get(node_id, {"node_id": node_id, "metadata": {}})
            facts = _candidate_facts(entity)
            observed_semantic = [
                facts.get(name, "")
                for name in ("label", "role", "domain", "family", "type", "system", "function")
            ]
            target_match = {
                "terms": _match_state(binding_contract.get("target_terms") or [], observed_semantic),
                "kind": _match_state([binding_contract.get("target_kind") or ""], [facts.get("kind", "")]),
                "ifc_class": _match_state(
                    [binding_contract.get("target_ifc_class") or ""],
                    [facts.get("ifc_class", "")],
                ),
            }
            hard_rejects: list[str] = []
            if target_match["kind"] == "mismatch":
                hard_rejects.append("authoritative_target_kind_contradiction")
            if target_match["ifc_class"] == "mismatch":
                hard_rejects.append("authoritative_ifc_class_contradiction")
            global_id = str(entity.get("global_id") or "")
            if binding_contract.get("action") and not global_id:
                hard_rejects.append("missing_ifc_global_id")

            candidate_paths = [
                path
                for path in paths
                if str(path.get("target_id") or "") == node_id
                and (
                    path.get("binding_index") is None
                    or int(path.get("binding_index")) == binding_index
                )
            ]
            candidate_paths.sort(
                key=lambda path: (-float(path.get("score") or 0.0), str(path.get("path_id") or ""))
            )
            contexts: list[dict[str, Any]] = []
            context_signatures: set[tuple[Any, ...]] = set()
            for path in candidate_paths:
                scope_id = str(path.get("scope_id") or "")
                relations = _path_relations(path)
                scope = _scope_facts(scope_id, entities)
                context = {
                    "scope_alias": scope_aliases.get(scope_id, ""),
                    "scope_label": scope["label"],
                    "scope_kind": scope["kind"],
                    "scope_ifc_class": scope["ifc_class"],
                    "scope_storey": scope["storey"],
                    "directed_relations": relations,
                    "relation_match": _relation_context_match(
                        str(binding_contract.get("relation") or ""), relations
                    ),
                    "source": (
                        "authoritative_typed_path"
                        if bool(path.get("authoritative", True))
                        else "non_authoritative_relation_shuffled_control"
                    ),
                }
                signature = (
                    context["scope_alias"],
                    tuple(context["directed_relations"]),
                    context["scope_label"],
                )
                if signature not in context_signatures:
                    context_signatures.add(signature)
                    contexts.append(context)
                if len(contexts) >= 4:
                    break
            for scope_id in _operator_scope_ids(entity):
                scope = _scope_facts(scope_id, entities)
                context = {
                    "scope_alias": scope_aliases.get(scope_id, ""),
                    "scope_label": scope["label"],
                    "scope_kind": scope["kind"],
                    "scope_ifc_class": scope["ifc_class"],
                    "scope_storey": scope["storey"],
                    "directed_relations": ["target_to_scope:part_of"],
                    "relation_match": _relation_context_match(
                        str(binding_contract.get("relation") or ""),
                        ["target_to_scope:part_of", "scope_to_target:contains"],
                    ),
                    "source": "bounded_operator_scope",
                }
                signature = (
                    context["scope_alias"],
                    tuple(context["directed_relations"]),
                    context["scope_label"],
                )
                if signature not in context_signatures:
                    context_signatures.add(signature)
                    contexts.append(context)
                if len(contexts) >= 4:
                    break

            operator_binding = _operator_binding_row(entity, binding_index)
            scope_predicates = list(dict(compact.get("scope") or {}).get("predicates") or [])
            endpoint_scope_predicates = [
                row
                for row in scope_predicates
                if _normal(dict(row).get("predicate") or "") != "storey"
            ]
            endpoint_context_required = bool(
                binding_contract.get("relation")
                or (
                    _normal(binding_contract.get("target_kind") or "") != "space"
                    and endpoint_scope_predicates
                )
            )
            authoritative_context = any(
                context.get("source") == "authoritative_typed_path"
                and context.get("relation_match") != "mismatch"
                for context in contexts
            )
            operator_context_pass = operator_binding.get("overall") == "pass"
            context_certification = (
                "not_required"
                if not endpoint_context_required
                else "authoritative"
                if authoritative_context
                else "operator_pass"
                if operator_context_pass
                else "unproven"
            )
            context_required = bool(
                binding_contract.get("scope_terms") or binding_contract.get("relation")
            )
            context_state = (
                "closed"
                if contexts or operator_binding.get("overall") == "pass"
                else "unknown"
                if context_required
                else "not_required"
            )
            candidates.append(
                {
                    "candidate_alias": f"B{binding_index:02d}C{union_rank:03d}",
                    "binding_index": binding_index,
                    "node_id": node_id,
                    "global_id": global_id,
                    "union_rank": union_rank,
                    "frozen_rank": top_rank.get(node_id),
                    "rrf_score": float(top50_scores.get(key, {}).get(node_id, 0.0)),
                    "operational_top20": node_id in direct or (top_rank.get(node_id) or 999) <= 20,
                    "source_flags": {
                        "frozen_top50": node_id in set(ranked),
                        "bounded_operator": node_id in operator_set,
                        "authoritative_typed_path": node_id in authoritative_path_set,
                    },
                    "operator_constraint": {
                        "overall": str(operator_binding.get("overall") or "unknown"),
                        "constraints": copy.deepcopy(dict(operator_binding.get("constraints") or {})),
                    },
                    "selection_mode": str(binding_contract.get("selection_mode") or ""),
                    "endpoint_context_required": endpoint_context_required,
                    "context_certification": context_certification,
                    "facts": facts,
                    "target_match": target_match,
                    "scope_contexts": contexts,
                    "context_state": context_state,
                    "eligibility": "contradicted" if hard_rejects else "reviewable",
                    "hard_reject_reasons": hard_rejects,
                    "evidence_ids": [
                        f"candidate:B{binding_index:02d}C{union_rank:03d}",
                        *(
                            [f"operator:B{binding_index:02d}C{union_rank:03d}"]
                            if node_id in operator_set
                            else []
                        ),
                        *(
                            [f"typed-path:B{binding_index:02d}C{union_rank:03d}"]
                            if node_id in path_set
                            else []
                        ),
                    ],
                }
            )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(_candidate_group_key(candidate), []).append(candidate)
    groups: list[dict[str, Any]] = []
    group_counter: dict[int, int] = {}
    for _key, members in sorted(
        grouped.items(), key=lambda item: min(row["union_rank"] for row in item[1])
    ):
        binding_index = int(members[0]["binding_index"])
        group_counter[binding_index] = group_counter.get(binding_index, 0) + 1
        alias = f"B{binding_index:02d}G{group_counter[binding_index]:03d}"
        for member in members:
            member["group_alias"] = alias
        facts = copy.deepcopy(members[0]["facts"])
        group_contexts: list[dict[str, Any]] = []
        seen_contexts: set[str] = set()
        for member in members:
            for context in member["scope_contexts"]:
                key = sha256_json(context)
                if key not in seen_contexts:
                    seen_contexts.add(key)
                    group_contexts.append(copy.deepcopy(context))
        endpoint_context_required = any(
            bool(row.get("endpoint_context_required")) for row in members
        )
        context_certified_members = sum(
            str(row.get("context_certification") or "") != "unproven"
            for row in members
        )
        base_eligibility = str(members[0]["eligibility"])
        group_eligibility = (
            base_eligibility
            if base_eligibility == "contradicted"
            else "context_unproven"
            if endpoint_context_required and context_certified_members == 0
            else "reviewable"
        )
        model_exclusion_reasons = _ordered_unique(
            [
                *(
                    reason
                    for row in members
                    for reason in row["hard_reject_reasons"]
                ),
                *(
                    ["required_endpoint_context_unproven"]
                    if group_eligibility == "context_unproven"
                    else []
                ),
            ]
        )
        groups.append(
            {
                "group_alias": alias,
                "binding_index": binding_index,
                "candidate_aliases": [row["candidate_alias"] for row in members],
                "node_ids": [row["node_id"] for row in members],
                "member_count": len(members),
                "sample_labels": _ordered_unique(row["facts"].get("label", "") for row in members)[:4],
                "facts": {
                    key: value
                    for key, value in facts.items()
                    if key != "label"
                },
                "target_match": copy.deepcopy(members[0]["target_match"]),
                "scope_contexts": group_contexts[:8],
                "source_summary": {
                    "best_frozen_rank": min(
                        (row["frozen_rank"] for row in members if row["frozen_rank"] is not None),
                        default=None,
                    ),
                    "frozen_top50_members": sum(row["source_flags"]["frozen_top50"] for row in members),
                    "bounded_operator_members": sum(row["source_flags"]["bounded_operator"] for row in members),
                    "bounded_operator_pass_members": sum(
                        row["source_flags"]["bounded_operator"]
                        and row["operator_constraint"]["overall"] == "pass"
                        for row in members
                    ),
                    "authoritative_path_members": sum(
                        row["source_flags"]["authoritative_typed_path"] for row in members
                    ),
                    "operator_constraint_states": sorted(
                        {str(row["operator_constraint"]["overall"]) for row in members}
                    ),
                    "context_certification_states": sorted(
                        {str(row.get("context_certification") or "") for row in members}
                    ),
                    "context_certified_members": context_certified_members,
                },
                "endpoint_context_required": endpoint_context_required,
                "eligibility": group_eligibility,
                "hard_reject_reasons": _ordered_unique(
                    reason for row in members for reason in row["hard_reject_reasons"]
                ),
                "model_exclusion_reasons": model_exclusion_reasons,
                "evidence_ids": [f"group:{alias}"],
            }
        )

    ledger = CandidateGroupLedgerV3(
        packet_sha256=packet.packet_sha256,
        compact_plan=compact,
        contract=contract.to_dict(),
        candidates=candidates,
        groups=groups,
        source_diagnostics=source_counts,
    ).seal()
    if not ledger.verify():
        raise RuntimeError("v3 candidate group ledger self-hash failed")
    if any(
        int(row["union_after_cap"]) > max_candidates_per_binding
        for row in source_counts["bindings"].values()
    ):
        raise RuntimeError("v3 candidate union cap failed")
    return ledger


@dataclass(slots=True)
class GroupSelectionV3:
    ledger_sha256: str
    binding_selections: list[dict[str, Any]]
    closure_supported: bool
    provider_summary: str = ""
    validation_errors: list[str] = field(default_factory=list)
    schema_version: str = "candidate-group-selection-v3"
    selection_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("selection_sha256", None)
        return payload

    def seal(self) -> GroupSelectionV3:
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


def normalize_group_selection_v3(
    raw: Mapping[str, Any], ledger: CandidateGroupLedgerV3
) -> GroupSelectionV3:
    if not ledger.verify():
        raise ValueError("cannot normalize against an unsealed v3 ledger")
    groups = {str(row["group_alias"]): row for row in ledger.groups}
    expected = {
        int(row["binding_index"])
        for row in list(ledger.contract.get("bindings") or [])
    }
    errors: list[str] = []
    seen_bindings: set[int] = set()
    selections: list[dict[str, Any]] = []
    for raw_row in list(raw.get("binding_selections") or []):
        row = dict(raw_row)
        try:
            binding_index = int(str(row.get("binding_index")))
        except (TypeError, ValueError):
            errors.append("invalid_binding_index")
            continue
        if binding_index not in expected:
            # Unknown rows cannot name a valid ledger group for an executable
            # binding. Ignore the rejected provider extension and retain the
            # authoritative sealed binding set.
            continue
        if binding_index in seen_bindings:
            errors.append(f"duplicate_binding_selection:{binding_index}")
            continue
        seen_bindings.add(binding_index)
        selected = _ordered_unique(row.get("selected_group_ids") or [])
        uncertain = _ordered_unique(row.get("uncertain_group_ids") or [])
        if set(selected) & set(uncertain):
            errors.append(f"selected_uncertain_overlap:{binding_index}")
        for alias in [*selected, *uncertain]:
            group = groups.get(alias)
            if group is None:
                errors.append(f"unknown_group:{alias}")
            elif int(group["binding_index"]) != binding_index:
                errors.append(f"cross_binding_group:{alias}")
            elif alias in selected and group["eligibility"] != "reviewable":
                errors.append(f"uncertifiable_group_selected:{alias}")
        selections.append(
            {
                "binding_index": binding_index,
                "selected_group_ids": selected,
                "uncertain_group_ids": uncertain,
                "reason": _clean_explanatory_text(row.get("reason") or ""),
            }
        )
    for binding_index in sorted(expected - seen_bindings):
        errors.append(f"missing_binding_selection:{binding_index}")
        selections.append(
            {
                "binding_index": binding_index,
                "selected_group_ids": [],
                "uncertain_group_ids": [],
                "reason": "provider omitted binding",
            }
        )
    return GroupSelectionV3(
        ledger_sha256=ledger.ledger_sha256,
        binding_selections=sorted(selections, key=lambda row: row["binding_index"]),
        closure_supported=bool(raw.get("closure_supported")),
        provider_summary=_clean_explanatory_text(
            raw.get("answer_summary") or raw.get("summary") or ""
        ),
        validation_errors=_ordered_unique(errors),
    ).seal()


def reconcile_group_selections_v3(
    packet: RetrievalEvidencePacketV1,
    contract: RetrievalIntentContractV3,
    ledger: CandidateGroupLedgerV3,
    initial: GroupSelectionV3,
    review: GroupSelectionV3,
) -> GroupSelectionV3:
    """Reconcile two provider selections with monotone evidence invariants.

    The reviewer may add evidence to an exhaustive ``all_matching`` result, but
    it cannot silently remove a previously selected, model-visible group. A
    removal requires a machine-checkable graph contradiction; contradicted and
    otherwise uncertifiable groups never reach either provider in the first
    place.  This prevents a stochastic review call from reducing a closed set
    without adding new evidence. Disagreement is retained for review and must
    abstain; the first provider's selection is not itself a certification.

    Space lookup is the second deterministic boundary.  When the bounded graph
    operator completed with no failed or unresolved predicates, its candidates
    already passed the typed name/kind/scope contract.  The same is true when a
    ``best_equivalence_class`` has per-candidate passes and the sole aggregate
    failure is the legacy ``action_bindings`` single-cardinality marker: several
    equivalent spaces are then alternatives, not failed predicates.  If both
    provider calls return an empty selection, those operator witnesses may close
    the binding without another semantic guess.  This rule is based only on the
    sealed operator trace and applies to arbitrary IFC graphs.
    """

    if not all(
        (
            packet.verify(),
            contract.verify(),
            ledger.verify(),
            initial.verify(),
            review.verify(),
        )
    ):
        raise ValueError("v3 reconciliation inputs are not hash-valid")
    if initial.ledger_sha256 != ledger.ledger_sha256:
        raise ValueError("v3 initial selection lineage mismatch")
    if review.ledger_sha256 != ledger.ledger_sha256:
        raise ValueError("v3 review selection lineage mismatch")

    groups = {str(row["group_alias"]): row for row in ledger.groups}
    initial_rows = {
        int(row["binding_index"]): row for row in initial.binding_selections
    }
    review_rows = {
        int(row["binding_index"]): row for row in review.binding_selections
    }
    operator_result = dict(packet.operator_result or {})
    operator_complete = (
        operator_result.get("complete") is True
        and not list(operator_result.get("unresolved_slots") or [])
        and not list(operator_result.get("failed_predicates") or [])
    )
    aggregate_failure_fields = {
        str(value)
        for value in [
            *list(operator_result.get("unresolved_slots") or []),
            *list(operator_result.get("failed_predicates") or []),
        ]
        if str(value)
    }
    decisions: list[str] = []
    reconciled_rows: list[dict[str, Any]] = []
    disagreements: list[str] = []
    fallback_used = False

    for binding in contract.bindings:
        binding_index = int(binding["binding_index"])
        initial_row = initial_rows.get(
            binding_index,
            {"selected_group_ids": [], "uncertain_group_ids": [], "reason": ""},
        )
        review_row = review_rows.get(
            binding_index,
            {"selected_group_ids": [], "uncertain_group_ids": [], "reason": ""},
        )
        selected = [
            alias
            for alias in _ordered_unique(review_row.get("selected_group_ids") or [])
            if alias in groups and groups[alias]["eligibility"] == "reviewable"
        ]
        uncertain = [
            alias
            for alias in _ordered_unique(review_row.get("uncertain_group_ids") or [])
            if alias in groups and groups[alias]["eligibility"] == "reviewable"
        ]
        mode = str(binding.get("selection_mode") or "best_equivalence_class")

        if mode == "all_matching" and not initial.validation_errors:
            retained = [
                alias
                for alias in _ordered_unique(
                    initial_row.get("selected_group_ids") or []
                )
                if alias in groups
                and groups[alias]["eligibility"] == "reviewable"
                and int(groups[alias]["binding_index"]) == binding_index
            ]
            before = set(selected)
            selected = _ordered_unique([*selected, *retained])
            if set(selected) != before:
                disagreements.append(f"provider_selection_disagreement:binding_{binding_index}")
                decisions.append(
                    f"binding_{binding_index}:retained_initial_all_matching_groups"
                )

        if (
            not selected
            and not uncertain
            and str(binding.get("target_kind") or "") == "space"
        ):
            operator_groups = [
                row
                for row in ledger.groups
                if int(row["binding_index"]) == binding_index
                and row["eligibility"] == "reviewable"
                and int(
                    row["source_summary"].get("bounded_operator_pass_members")
                    or 0
                )
                > 0
            ]
            operator_groups.sort(key=lambda row: str(row["group_alias"]))
            equivalence_cardinality_only = (
                mode == "best_equivalence_class"
                and aggregate_failure_fields == {"action_bindings"}
            )
            if not operator_complete and not equivalence_cardinality_only:
                operator_groups = []
            if mode == "one_graph_node":
                operator_groups = (
                    operator_groups
                    if len(operator_groups) == 1
                    and int(operator_groups[0]["member_count"]) == 1
                    else []
                )
            elif mode == "bounded_operator":
                operator_groups = (
                    operator_groups if len(operator_groups) == 1 else []
                )
            if operator_groups:
                selected = [str(row["group_alias"]) for row in operator_groups]
                fallback_used = True
                decisions.append(
                    f"binding_{binding_index}:complete_space_operator_witness"
                )

        selected_set = set(selected)
        uncertain = [alias for alias in uncertain if alias not in selected_set]
        reason_parts = [str(review_row.get("reason") or "").strip()]
        binding_decisions = [
            decision
            for decision in decisions
            if decision.startswith(f"binding_{binding_index}:")
        ]
        if binding_decisions:
            reason_parts.append("; ".join(binding_decisions))
        reconciled_rows.append(
            {
                "binding_index": binding_index,
                "selected_group_ids": selected,
                "uncertain_group_ids": uncertain,
                "reason": " | ".join(part for part in reason_parts if part),
            }
        )

    complete_after_reconciliation = all(
        row["selected_group_ids"] and not row["uncertain_group_ids"]
        for row in reconciled_rows
    )
    closure_supported = bool(review.closure_supported)
    if initial.closure_supported and any(
        decision.endswith("retained_initial_all_matching_groups")
        for decision in decisions
    ):
        closure_supported = complete_after_reconciliation
    if fallback_used:
        closure_supported = complete_after_reconciliation

    summary_parts = [review.provider_summary.strip()]
    if decisions:
        summary_parts.append("deterministic reconciliation: " + "; ".join(decisions))
    return GroupSelectionV3(
        ledger_sha256=ledger.ledger_sha256,
        binding_selections=reconciled_rows,
        closure_supported=closure_supported,
        provider_summary=" | ".join(part for part in summary_parts if part),
        validation_errors=[*review.validation_errors, *disagreements],
    ).seal()


@dataclass(slots=True)
class EvidenceClosureCertificateV3:
    packet_sha256: str
    contract_sha256: str
    ledger_sha256: str
    selection_sha256: str
    certified_target_ids: dict[str, list[str]]
    certified_group_aliases: dict[str, list[str]]
    binding_status: dict[str, dict[str, Any]]
    closure_status: ClosureStatusV3
    stop_reason: str
    validation_errors: list[str] = field(default_factory=list)
    schema_version: str = "evidence-closure-certificate-v3"
    certificate_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("certificate_sha256", None)
        return payload

    def seal(self) -> EvidenceClosureCertificateV3:
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


def adjudicate_group_selection_v3(
    packet: RetrievalEvidencePacketV1,
    contract: RetrievalIntentContractV3,
    ledger: CandidateGroupLedgerV3,
    selection: GroupSelectionV3,
) -> EvidenceClosureCertificateV3:
    if not all((packet.verify(), contract.verify(), ledger.verify(), selection.verify())):
        raise ValueError("v3 adjudication inputs are not hash-valid")
    if ledger.ledger_sha256 != selection.ledger_sha256:
        raise ValueError("v3 selection lineage mismatch")
    groups = {str(row["group_alias"]): row for row in ledger.groups}
    candidates = {str(row["candidate_alias"]): row for row in ledger.candidates}
    selected_by_binding = {
        str(int(row["binding_index"])): row for row in selection.binding_selections
    }
    certified_ids: dict[str, list[str]] = {}
    certified_groups: dict[str, list[str]] = {}
    binding_status: dict[str, dict[str, Any]] = {}
    errors = [*contract.validation_errors, *selection.validation_errors]

    for binding in contract.bindings:
        key = str(int(binding["binding_index"]))
        provider = selected_by_binding.get(
            key,
            {"selected_group_ids": [], "uncertain_group_ids": []},
        )
        selected_aliases = list(provider.get("selected_group_ids") or [])
        uncertain = list(provider.get("uncertain_group_ids") or [])
        chosen_groups = [groups[alias] for alias in selected_aliases if alias in groups]
        chosen_candidates = [
            candidates[candidate_alias]
            for group in chosen_groups
            for candidate_alias in group["candidate_aliases"]
            if candidate_alias in candidates
            and candidates[candidate_alias]["eligibility"] != "contradicted"
        ]
        chosen_candidates.sort(key=lambda row: int(row["union_rank"]))
        chosen_nodes = _ordered_unique(row["node_id"] for row in chosen_candidates)
        mode = str(binding.get("selection_mode") or "best_equivalence_class")
        status = "supported"
        reason = "bounded evidence-equivalence selection certified"
        if uncertain:
            status = "ambiguous_candidate_groups"
            reason = "provider left candidate groups uncertain"
        elif not chosen_nodes:
            status = "graph_or_schema_absent"
            reason = "no supported target group selected"
        elif mode == "one_graph_node" and len(chosen_nodes) != 1:
            status = "cardinality_conflict"
            reason = "one_graph_node requires exactly one physical IFC node"
        elif mode == "bounded_operator" and not any(
            int(group["source_summary"]["bounded_operator_members"]) > 0
            for group in chosen_groups
        ):
            status = "operator_witness_missing"
            reason = "bounded operator selection lacks an operator-result witness"
        if any(group["eligibility"] != "reviewable" for group in chosen_groups):
            status = "candidate_graph_contradiction"
            reason = "selected group is not certifiable from authoritative graph evidence"

        certified_ids[key] = chosen_nodes if status == "supported" else []
        certified_groups[key] = selected_aliases if status == "supported" else []
        binding_status[key] = {
            "status": status,
            "reason": reason,
            "selection_mode": mode,
            "selected_group_count": len(selected_aliases),
            "selected_node_count": len(chosen_nodes),
            "uncertain_group_count": len(uncertain),
            "bounded_candidate_closure": True,
        }

    if errors:
        closure_status: ClosureStatusV3 = "abstain"
        stop_reason = (
            "provider_selection_disagreement"
            if any(error.startswith("provider_selection_disagreement:") for error in errors)
            else "invalid_provider_or_contract"
        )
    elif not selection.closure_supported:
        closure_status = "abstain"
        stop_reason = "provider_declared_evidence_insufficient"
    elif any(row["status"] != "supported" for row in binding_status.values()):
        closure_status = "abstain"
        stop_reason = next(
            row["status"] for row in binding_status.values() if row["status"] != "supported"
        )
    else:
        closure_status = "pass"
        stop_reason = "all_bindings_certified"
    return EvidenceClosureCertificateV3(
        packet_sha256=packet.packet_sha256,
        contract_sha256=contract.contract_sha256,
        ledger_sha256=ledger.ledger_sha256,
        selection_sha256=selection.selection_sha256,
        certified_target_ids=certified_ids,
        certified_group_aliases=certified_groups,
        binding_status=binding_status,
        closure_status=closure_status,
        stop_reason=stop_reason,
        validation_errors=_ordered_unique(errors),
    ).seal()


def certified_answer_v3(
    contract: RetrievalIntentContractV3,
    ledger: CandidateGroupLedgerV3,
    certificate: EvidenceClosureCertificateV3,
) -> str:
    if certificate.closure_status != "pass":
        return "I cannot determine the answer from the IFC knowledge graph."
    candidates = {str(row["node_id"]): row for row in ledger.candidates}
    calls: list[str] = []
    for binding in contract.bindings:
        key = str(int(binding["binding_index"]))
        action = str(binding.get("action") or "")
        if action not in _ACTIONS:
            return "I cannot determine the answer from the IFC knowledge graph."
        for node_id in certificate.certified_target_ids.get(key, []):
            global_id = str(candidates.get(node_id, {}).get("global_id") or "")
            if not global_id:
                return "I cannot determine the answer from the IFC knowledge graph."
            calls.append(f"{action}({global_id})")
    return ", ".join(calls) if calls else "I cannot determine the answer from the IFC knowledge graph."


@dataclass(slots=True)
class RetrievalClosureResultV3:
    answer: str
    contract: RetrievalIntentContractV3
    ledger: CandidateGroupLedgerV3
    selection: GroupSelectionV3
    certificate: EvidenceClosureCertificateV3
    downstream_usage: dict[str, Any]
    answer_call_count: int
    errors: list[str] = field(default_factory=list)


def v3_synthetic_contract_checks() -> dict[str, bool]:
    compact = {
        "operator": "path",
        "bindings": [
            {
                "binding_index": 0,
                "action": "Inspect",
                "cardinality": "single",
                "target": {
                    "kind": "object",
                    "ifc_class": "IfcSensor",
                    "roles": ["sensor"],
                    "names": [],
                    "domains": ["controls"],
                },
            }
        ],
        "scope": {"storey": "LEVEL 1", "room": "Room 101", "room_names": [], "predicates": []},
        "relations": [{"relation": "contains", "direction": "scope_to_target"}],
        "function_intents": [],
        "action_order": ["Inspect"],
    }
    contract = normalize_intent_contract_v3(
        {
            "bindings": [
                {
                    "binding_index": 0,
                    "action": "Inspect",
                    "selection_mode": "best_equivalence_class",
                    "target_phrase": "temperature sensor",
                    "target_terms": ["temperature", "sensor"],
                    "target_kind": "object",
                    "target_ifc_class": "IfcSensor",
                    "scope_phrase": "Room 101 on Level 1",
                    "scope_terms": ["Room 101", "Level 1"],
                    "relation": "contains",
                    "direction": "scope_to_target",
                    "requires_exhaustive_set": False,
                }
            ],
            "summary": "synthetic",
        },
        compact,
    )
    contaminated = normalize_intent_contract_v3(
        {
            "bindings": [
                {
                    "binding_index": 0,
                    "action": "Inspect",
                    "selection_mode": "one_graph_node",
                    "target_phrase": "ifc_AAAAAAAAAAAAAAAAAAAAAA",
                    "target_terms": [],
                    "target_kind": "object",
                    "target_ifc_class": "IfcSensor",
                    "scope_phrase": "",
                    "scope_terms": [],
                    "relation": "",
                    "direction": "",
                    "requires_exhaustive_set": False,
                }
            ]
        },
        compact,
    )
    return {
        "semantic_singular_defaults_to_equivalence_class": (
            contract.bindings[0]["selection_mode"] == "best_equivalence_class"
        ),
        "contract_hash_valid": contract.verify(),
        "question_contract_rejects_identifiers": bool(contaminated.validation_errors),
        "action_order_preserved": contract.bindings[0]["action"] == "Inspect",
        "compact_plan_hash_bound": contract.compact_plan_sha256 == sha256_json(compact),
    }


__all__ = [
    "CandidateGroupLedgerV3",
    "EvidenceClosureCertificateV3",
    "GroupSelectionV3",
    "RetrievalClosureResultV3",
    "RetrievalIntentContractV3",
    "adjudicate_group_selection_v3",
    "build_candidate_group_ledger_v3",
    "certified_answer_v3",
    "normalize_group_selection_v3",
    "normalize_intent_contract_v3",
    "reconcile_group_selections_v3",
    "v3_synthetic_contract_checks",
]
