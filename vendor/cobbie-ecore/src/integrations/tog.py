from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unicodedata
from contextvars import ContextVar
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import baml_py
from baml_py.baml_py import Collector

from src.baml.baml_client import b
from src.schemas.agent_error import AgentError
from src.util.baml_retry import call_baml_with_retry


def ensure_tog_importable() -> None:
    try:
        import tog  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root.parent / "ToG" / "src",
        root / "ToG" / "src",
        Path.cwd().parent / "ToG" / "src",
        Path.cwd() / "ToG" / "src",
    ]
    for candidate in candidates:
        if (candidate / "tog").exists():
            sys.path.insert(0, str(candidate))
            return
    raise ModuleNotFoundError(
        "Could not import tog. Install it with `python -m pip install -e ../ToG` "
        "or keep the ToG checkout beside cobbie."
    )


ensure_tog_importable()

from tog import (  # noqa: E402
    GraphIndexManager,
    QueryPlan,
    ToGConfig,
    ToGSystem,
)
from tog.index import FrozenInspectionGraphBuilder  # noqa: E402
from tog.models import (  # noqa: E402
    ActionTargetBinding,
    EntityRef,
    EvidenceCoverage,
    EvidenceReview,
    HierarchyContext,
    HierarchyReasoningTrace,
    MentionLink,
    QueryHypothesis,
    QueryHypothesisSelectionResult,
    ReasoningMode,
    ReasoningRequirement,
    RelationRef,
    RelationReference,
    ScopePredicate,
    TargetAudit,
    TargetPredicate,
    TargetSelectionResult,
    TripleEvidence,
)
from tog.retriever_plugin import (  # noqa: E402
    RetrievalPluginContext,
    load_retriever_plugin,
)

from src.integrations.tog_wire import enum_value as _enum_value

PLANNER_PROMPT_VERSION = "generic-query-compiler-v3"
_TOG_BAML_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "baml" / "baml_src" / "tog.baml"
)
_TOG_BAML_CLIENTS_PATH = (
    Path(__file__).resolve().parents[1] / "baml" / "baml_src" / "clients.baml"
)


def _planner_prompt_lineage() -> dict[str, str]:
    """Bind planner caches to the exact prompt/client source, not a label."""

    if not _TOG_BAML_PROMPT_PATH.is_file() or not _TOG_BAML_CLIENTS_PATH.is_file():
        raise RuntimeError("ToG planner prompt lineage files are missing")
    return {
        "planner_prompt_sha256": hashlib.sha256(
            _TOG_BAML_PROMPT_PATH.read_bytes()
        ).hexdigest(),
        "planner_clients_sha256": hashlib.sha256(
            _TOG_BAML_CLIENTS_PATH.read_bytes()
        ).hexdigest(),
    }
PLANNER_CACHE_SCHEMA_VERSION = "structured-query-plan-cache-v1"
_PLANNER_LINK_CANDIDATE_LIMIT = 5
QUERY_HYPOTHESIS_PROMPT_VERSION = "bounded-query-hypothesis-v1"
_QUERY_HYPOTHESIS_LIMIT = 4
_QUERY_HYPOTHESIS_PAYLOAD_MAX_CHARS = 6_000
_QUERY_HYPOTHESIS_SUMMARY_LIMIT = 3
_QUERY_HYPOTHESIS_TEXT_MAX_CHARS = 160
_ACTIVE_PLANNER_GRAPH_HASH: ContextVar[str] = ContextVar(
    "active_planner_graph_hash", default=""
)

_IFC_OR_OPAQUE_ID_RE = re.compile(
    r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])"
)
_UUID_RE = re.compile(
    r"(?i)(?<![0-9a-f])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?![0-9a-f])"
)
_INTERNAL_IFC_ID_RE = re.compile(r"(?i)\bifc[_:-][0-9A-Za-z_$.:~-]+\b")
_EXPLICIT_NODE_ID_RE = re.compile(
    r"(?i)\b(?:node|source|target)_id\s*[:=]\s*[\"']?[^,;\\s\"']+"
)


def _normalized_planner_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).split()).casefold()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _query_plan_from_dict(value: Any) -> QueryPlan:
    if not isinstance(value, dict):
        raise ValueError("cached query_plan must be an object")
    allowed = {item.name for item in fields(QueryPlan)}
    if any(key not in allowed for key in value):
        raise ValueError("cached query_plan contains unsupported fields")
    data = dict(value)
    nested = {
        "action_bindings": ActionTargetBinding,
        "mention_links": MentionLink,
        "scope_predicates": ScopePredicate,
        "relation_references": RelationReference,
        "target_predicates": TargetPredicate,
    }
    for field_name, constructor in nested.items():
        rows = data.get(field_name, [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"cached {field_name} must be a list of objects")
        data[field_name] = [constructor(**row) for row in rows]
    plan = QueryPlan(**data)
    if plan.operator not in {
        "lookup", "list", "distinct", "count", "group_count", "argmax",
        "nearest", "all_matching", "unconnected", "path",
    }:
        raise ValueError("cached query_plan has an invalid operator")
    return plan


def _nonempty(values: dict[str, Any]) -> dict[str, Any]:
    """Drop empty planner fields without discarding meaningful false/zero values."""
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def _safe_hypothesis_text(value: Any, *, max_chars: int = 160) -> str:
    """Return a bounded semantic string with graph identities redacted."""

    text = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    text = _EXPLICIT_NODE_ID_RE.sub("[node-id-redacted]", text)
    text = _INTERNAL_IFC_ID_RE.sub("[node-id-redacted]", text)
    text = _UUID_RE.sub("[guid-redacted]", text)
    text = _IFC_OR_OPAQUE_ID_RE.sub("[guid-redacted]", text)
    return text[:max_chars]


def _bounded_hypothesis_strings(
    values: Sequence[Any],
    *,
    limit: int = 4,
    max_chars: int = 80,
) -> list[str]:
    return [
        text
        for value in list(values)[:limit]
        if (text := _safe_hypothesis_text(value, max_chars=max_chars))
    ]


def _compact_hypothesis_plan(plan: QueryPlan) -> dict[str, Any]:
    """Serialize typed semantics only; never serialize graph-linked node IDs."""

    scope_predicates = [
        _nonempty(
            {
                "stage": _safe_hypothesis_text(item.stage_id, max_chars=40),
                "predicate": item.predicate,
                "values": _bounded_hypothesis_strings(item.values, limit=3),
            }
        )
        for item in plan.scope_predicates[:4]
    ]
    target_predicates = [
        _nonempty(
            {
                "stage": _safe_hypothesis_text(item.stage_id, max_chars=40),
                "predicate": item.predicate,
                "values": _bounded_hypothesis_strings(item.values, limit=3),
                "source_stage": _safe_hypothesis_text(
                    item.source_stage, max_chars=40
                ),
                "required": bool(item.required),
            }
        )
        for item in plan.target_predicates[:4]
    ]
    relation_references = [
        _nonempty(
            {
                "relation": item.relation,
                "mentions": _bounded_hypothesis_strings(item.mentions, limit=3),
                "kind": item.reference_kind,
                "source_stage": _safe_hypothesis_text(
                    item.source_stage, max_chars=40
                ),
            }
        )
        for item in plan.relation_references[:4]
    ]
    action_bindings = [
        _nonempty(
            {
                "index": int(item.binding_index),
                "action": _safe_hypothesis_text(item.action, max_chars=24),
                "source_stage": _safe_hypothesis_text(
                    item.source_stage, max_chars=40
                ),
                "target_kind": item.target_kind,
                "roles": _bounded_hypothesis_strings(item.target_roles),
                "names": _bounded_hypothesis_strings(item.target_names),
                "domains": _bounded_hypothesis_strings(item.target_domains),
                "functions": _bounded_hypothesis_strings(item.function_types),
                "systems": _bounded_hypothesis_strings(item.system_categories),
                "target_mode": item.target_mode,
                "cardinality": item.cardinality_policy,
            }
        )
        for item in plan.action_bindings[:4]
    ]
    return _nonempty(
        {
            "operator": plan.operator,
            "mentions": _bounded_hypothesis_strings(plan.mentions, limit=6),
            "scope": _nonempty(
                {
                    "storey": _safe_hypothesis_text(plan.storey, max_chars=80)
                    if plan.storey
                    else None,
                    "room": _safe_hypothesis_text(plan.room, max_chars=80)
                    if plan.room
                    else None,
                    "room_names": _bounded_hypothesis_strings(
                        plan.room_names, limit=3
                    ),
                    "space_type": _safe_hypothesis_text(
                        plan.target_space_type, max_chars=80
                    )
                    if plan.target_space_type
                    else None,
                    "cardinality": plan.scope_cardinality,
                    "predicates": scope_predicates,
                }
            ),
            "target": _nonempty(
                {
                    "kind": plan.target_kind,
                    "ifc_class": _safe_hypothesis_text(
                        plan.target_ifc_class, max_chars=80
                    )
                    if plan.target_ifc_class
                    else None,
                    "roles": _bounded_hypothesis_strings(
                        plan.target_roles
                        or ([plan.target_role] if plan.target_role else [])
                    ),
                    "names": _bounded_hypothesis_strings(
                        plan.target_names
                        or ([plan.target_name] if plan.target_name else [])
                    ),
                    "domain": _safe_hypothesis_text(
                        plan.target_domain, max_chars=80
                    )
                    if plan.target_domain
                    else None,
                    "functions": _bounded_hypothesis_strings(
                        plan.function_intents
                    ),
                    "binding_mode": plan.target_binding_mode,
                    "cardinality": plan.cardinality_policy,
                    "predicates": target_predicates,
                }
            ),
            "actions": action_bindings,
            "relations": relation_references,
            "search_exhaustive": bool(plan.search_exhaustive),
            "unresolved_slots": _bounded_hypothesis_strings(
                plan.unresolved_slots, limit=4
            ),
        }
    )


def _minimal_hypothesis_plan(plan: QueryPlan) -> dict[str, Any]:
    """Second-stage compaction that preserves interpretation differences."""

    scope_terms = [
        _safe_hypothesis_text(
            f"{item.predicate}:{'|'.join(item.values[:1])}", max_chars=48
        )
        for item in plan.scope_predicates[:2]
    ]
    target_terms = [
        _safe_hypothesis_text(
            f"{item.predicate}:{'|'.join(item.values[:1])}", max_chars=48
        )
        for item in plan.target_predicates[:2]
    ]
    return _nonempty(
        {
            "operator": plan.operator,
            "scope": _nonempty(
                {
                    "storey": _safe_hypothesis_text(
                        plan.storey, max_chars=32
                    )
                    if plan.storey
                    else None,
                    "room": _safe_hypothesis_text(plan.room, max_chars=32)
                    if plan.room
                    else None,
                    "space_type": _safe_hypothesis_text(
                        plan.target_space_type, max_chars=48
                    )
                    if plan.target_space_type
                    else None,
                    "terms": scope_terms,
                    "cardinality": plan.scope_cardinality,
                }
            ),
            "target": _nonempty(
                {
                    "kind": plan.target_kind,
                    "roles": _bounded_hypothesis_strings(
                        plan.target_roles
                        or ([plan.target_role] if plan.target_role else []),
                        limit=1,
                        max_chars=32,
                    ),
                    "names": _bounded_hypothesis_strings(
                        plan.target_names
                        or ([plan.target_name] if plan.target_name else []),
                        limit=1,
                        max_chars=32,
                    ),
                    "domain": _safe_hypothesis_text(
                        plan.target_domain, max_chars=32
                    )
                    if plan.target_domain
                    else None,
                    "functions": _bounded_hypothesis_strings(
                        plan.function_intents, limit=1, max_chars=32
                    ),
                    "terms": target_terms,
                    "cardinality": plan.cardinality_policy,
                }
            ),
            "actions": [
                _nonempty(
                    {
                        "i": int(item.binding_index),
                        "action": _safe_hypothesis_text(
                            item.action, max_chars=24
                        ),
                        "kind": item.target_kind,
                        "roles": _bounded_hypothesis_strings(
                            item.target_roles, limit=1, max_chars=32
                        ),
                        "names": _bounded_hypothesis_strings(
                            item.target_names, limit=1, max_chars=32
                        ),
                        "functions": _bounded_hypothesis_strings(
                            item.function_types, limit=1, max_chars=32
                        ),
                        "systems": _bounded_hypothesis_strings(
                            item.system_categories, limit=1, max_chars=32
                        ),
                        "cardinality": item.cardinality_policy,
                    }
                )
                for item in plan.action_bindings[:2]
            ],
            "relations": [
                item.relation for item in plan.relation_references[:3]
            ],
            "search_exhaustive": bool(plan.search_exhaustive),
            "unresolved_count": len(plan.unresolved_slots),
        }
    )


def _query_plan_output_contract(plan: QueryPlan) -> str:
    """Serialize answer shape from QueryPlan without evaluator metadata."""

    if plan.action_bindings:
        payload = {
            "mode": "ordered_action_sequence",
            "operator": plan.operator,
            "bindings": [
                {
                    "binding_index": int(binding.binding_index),
                    "action": binding.action,
                    "target_kind": binding.target_kind,
                    "cardinality_policy": binding.cardinality_policy,
                }
                for binding in plan.action_bindings
            ],
            "syntax": "Action(raw_IFC_GUID)",
            "preserve_binding_order": True,
        }
    else:
        payload = {
            "mode": "direct_operator_result",
            "operator": plan.operator,
            "cardinality_policy": plan.cardinality_policy,
            "include_concise_auditable_support": True,
        }
    return _canonical_json(payload)


def _query_hypothesis_aliases(
    hypotheses: Sequence[QueryHypothesis],
) -> dict[str, str]:
    if not hypotheses:
        return {}
    if len(hypotheses) > _QUERY_HYPOTHESIS_LIMIT:
        raise ValueError(
            "query hypothesis selection accepts at most "
            f"{_QUERY_HYPOTHESIS_LIMIT} hypotheses"
        )
    identities = [str(item.hypothesis_id).strip() for item in hypotheses]
    if any(not value for value in identities):
        raise ValueError("query hypothesis IDs must be non-empty")
    if len(set(identities)) != len(identities):
        raise ValueError("query hypothesis IDs must be unique")
    return {
        f"H{index}": hypothesis_id
        for index, hypothesis_id in enumerate(identities, start=1)
    }


def _compact_query_hypotheses_payload(
    hypotheses: Sequence[QueryHypothesis],
) -> str:
    """Build a compact, identity-free payload for bounded hypothesis choice."""

    aliases = _query_hypothesis_aliases(hypotheses)
    rows = []
    for alias, hypothesis in zip(aliases, hypotheses):
        rows.append(
            {
                "id": alias,
                "plan": _compact_hypothesis_plan(hypothesis.plan),
                "diagnostics": {
                    "hard_valid_count": max(
                        0, int(hypothesis.hard_valid_count)
                    ),
                    "contradiction": bool(hypothesis.contradiction),
                    "missing_metric": bool(hypothesis.missing_metric),
                    "binding_complete": bool(hypothesis.binding_complete),
                    "evidence": _bounded_hypothesis_strings(
                        hypothesis.evidence_summary,
                        limit=_QUERY_HYPOTHESIS_SUMMARY_LIMIT,
                        max_chars=_QUERY_HYPOTHESIS_TEXT_MAX_CHARS,
                    ),
                    "paths": _bounded_hypothesis_strings(
                        hypothesis.path_summary,
                        limit=_QUERY_HYPOTHESIS_SUMMARY_LIMIT,
                        max_chars=_QUERY_HYPOTHESIS_TEXT_MAX_CHARS,
                    ),
                },
            }
        )
    payload = {
        "prompt_version": QUERY_HYPOTHESIS_PROMPT_VERSION,
        "hypotheses": rows,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _QUERY_HYPOTHESIS_PAYLOAD_MAX_CHARS:
        # Diagnostic prose is useful but optional.  Typed plan structure and
        # scalar support signals remain intact when the hard payload ceiling is
        # approached.
        for row in rows:
            row["diagnostics"]["evidence"] = []
            row["diagnostics"]["paths"] = []
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _QUERY_HYPOTHESIS_PAYLOAD_MAX_CHARS:
        for row, hypothesis in zip(rows, hypotheses):
            row["plan"] = _minimal_hypothesis_plan(hypothesis.plan)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _QUERY_HYPOTHESIS_PAYLOAD_MAX_CHARS:
        raise ValueError(
            "bounded query hypothesis payload exceeds "
            f"{_QUERY_HYPOTHESIS_PAYLOAD_MAX_CHARS} characters"
        )
    return encoded


def _compact_planner_payload(plan: QueryPlan) -> str:
    """Serialize only typed query evidence needed by the fallback planner.

    The former ``asdict(plan)`` payload repeated global and staged constraints,
    included verbose rationale text, and could include every graph candidate ID
    attached to a mention.  This representation keeps the same deterministic
    constraints, but groups them by stage and caps graph-backed alternatives at
    five per mention.  It deliberately has no evaluation category, question ID,
    reference answer, benchmark example, or full graph content.
    """

    bindings = [
        _nonempty(
            {
                "action": item.action,
                "index": int(item.binding_index),
                "source_stage": item.source_stage,
                "result_stage": item.result_stage,
                "target_kind": item.target_kind,
                "roles": list(item.target_roles),
                "names": list(item.target_names),
                "domains": list(item.target_domains),
                "functions": list(item.function_types),
                "systems": list(item.system_categories),
                "target_mode": item.target_mode,
                "cardinality": item.cardinality_policy,
            }
        )
        for item in plan.action_bindings
    ]
    scope_predicates = [
        _nonempty(
            {
                "stage": item.stage_id,
                "predicate": item.predicate,
                "values": list(item.values),
            }
        )
        for item in plan.scope_predicates
    ]
    relation_references = [
        _nonempty(
            {
                "stage": item.stage_id,
                "relation": item.relation,
                "mentions": list(item.mentions),
                "candidate_ids": list(item.node_ids)[:_PLANNER_LINK_CANDIDATE_LIMIT],
                "kind": item.reference_kind,
                "source_stage": item.source_stage,
            }
        )
        for item in plan.relation_references
    ]
    target_predicates = [
        _nonempty(
            {
                "stage": item.stage_id,
                "predicate": item.predicate,
                "values": list(item.values),
                "source_stage": item.source_stage,
                "required": bool(item.required),
            }
        )
        for item in plan.target_predicates
    ]

    # MentionLink records may contain several node IDs, and multiple semantic
    # alternatives can share the same text span.  Group them so the cap applies
    # to the mention as a whole rather than independently to every record.
    linked_mentions: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int, int | None], list[Any]] = {}
    for link in plan.mention_links:
        key = (link.text, int(link.char_start), int(link.char_end), link.action_index)
        grouped.setdefault(key, []).append(link)
    for (text, char_start, char_end, action_index), links in grouped.items():
        alternatives: list[dict[str, Any]] = []
        remaining = _PLANNER_LINK_CANDIDATE_LIMIT
        total_candidates = max(int(item.candidate_count or 0) for item in links)
        for link in links:
            node_ids = list(dict.fromkeys(str(value) for value in link.node_ids))
            # Schema/ontology links without instance IDs are still one useful
            # typed alternative and do not consume a graph-node candidate slot.
            visible_ids = node_ids[:remaining]
            if node_ids and not visible_ids:
                continue
            alternative = _nonempty(
                {
                    "canonical": link.canonical,
                    "kind": link.kind,
                    "candidate_ids": visible_ids,
                    "ifc_class": link.ifc_class,
                    "level": link.graph_level,
                    "role": link.role,
                    "domain": link.domain,
                    "space_type": link.space_type,
                    "function": link.function_type,
                    "function_kind": link.function_kind,
                    "system": link.system_category,
                    "compatible_roles": list(link.compatible_roles),
                    "compatible_names": list(link.compatible_names),
                    "compatible_domains": list(link.compatible_domains),
                    "compatible_space_types": list(link.compatible_space_types),
                    "related_systems": list(link.related_system_categories),
                    "confidence": round(float(link.confidence), 4),
                    "source": link.source,
                    "source_field": link.source_field,
                    "query_focus": bool(link.query_focus),
                }
            )
            alternatives.append(alternative)
            remaining -= len(visible_ids)
        linked_mentions.append(
            _nonempty(
                {
                    "text": text,
                    "span": [char_start, char_end]
                    if char_start >= 0 and char_end >= 0
                    else None,
                    "action_index": action_index,
                    "candidate_count": total_candidates,
                    "candidates": alternatives,
                }
            )
        )

    payload = {
        "prompt_version": PLANNER_PROMPT_VERSION,
        "plan": _nonempty(
            {
                "operator": plan.operator,
                "mentions": list(plan.mentions),
                "scope": _nonempty(
                    {
                        "storey": plan.storey,
                        "room": plan.room,
                        "room_names": list(plan.room_names),
                        "space_type": plan.target_space_type,
                        "cardinality": plan.scope_cardinality,
                        "predicates": scope_predicates,
                    }
                ),
                "target": _nonempty(
                    {
                        "kind": plan.target_kind,
                        "ifc_class": plan.target_ifc_class,
                        "role": plan.target_role,
                        "roles": list(plan.target_roles),
                        "domain": plan.target_domain,
                        "name": plan.target_name,
                        "names": list(plan.target_names),
                        "family_terms": list(plan.target_family_terms),
                        "type_terms": list(plan.target_type_terms),
                        "keywords": list(plan.target_keywords),
                        "functions": list(plan.function_intents),
                        "binding_mode": plan.target_binding_mode,
                        "cardinality": plan.cardinality_policy,
                        "predicates": target_predicates,
                    }
                ),
                "properties": list(plan.property_terms),
                "actions": _nonempty(
                    {
                        "sequence": list(plan.action_sequence),
                        "bindings": bindings,
                    }
                ),
                "relation_references": relation_references,
                "search_exhaustive": bool(plan.search_exhaustive),
                "requires_exhaustive": bool(plan.requires_exhaustive),
                "grouping": list(plan.target_grouping),
                "unresolved_slots": list(plan.unresolved_slots),
            }
        ),
        "linked_mentions": linked_mentions,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _nonnegative_token_count(value: Any) -> int | None:
    """Normalize provider usage values without letting diagnostics fail a run."""

    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return count if count >= 0 else None


def _selected_response_usage(
    collector: Any,
    *path: str,
) -> int:
    """Sum one Responses API usage detail from each selected BAML call."""

    total = 0
    for log in list(getattr(collector, "logs", ()) or ()):
        try:
            call = getattr(log, "selected_call", None)
            response = getattr(call, "http_response", None)
            body = getattr(response, "body", None)
            payload = body.json() if body is not None else None
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        count = _nonnegative_token_count(value)
        if count is not None:
            total += count
    return total


class BamlToGLlm:
    def __init__(
        self,
        client: str,
        max_calls: int,
        *,
        prompt_evidence_limit: int = 500,
        prompt_evidence_max_chars: int = 50_000,
        planner_cache_dir: str | Path | None = None,
        planner_graph_hash: str | None = None,
    ) -> None:
        self.collector = Collector(name="ToG")
        self.registry = baml_py.ClientRegistry()
        self.registry.set_primary(client)
        self._calls = 0
        self._max_calls = max_calls
        self._call_limit = max_calls
        self._prompt_evidence_limit = max(1, int(prompt_evidence_limit))
        self._prompt_evidence_max_chars = max(256, int(prompt_evidence_max_chars))
        self._planner_model = str(client)
        self._planner_graph_hash = str(planner_graph_hash or "")
        self._planner_cache_dir = (
            Path(planner_cache_dir).expanduser().resolve()
            if planner_cache_dir is not None
            else None
        )
        if self._planner_cache_dir is not None:
            self._planner_cache_dir.mkdir(parents=True, exist_ok=True)

    def _planner_cache_lineage(self, question: str) -> dict[str, str] | None:
        cache_dir = getattr(self, "_planner_cache_dir", None)
        graph_hash = getattr(self, "_planner_graph_hash", "") or _ACTIVE_PLANNER_GRAPH_HASH.get()
        model = getattr(self, "_planner_model", "")
        if cache_dir is None or not graph_hash or not model:
            return None
        normalized = _normalized_planner_question(question)
        return {
            "normalized_question_sha256": hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest(),
            "graph_hash": str(graph_hash),
            "prompt_version": PLANNER_PROMPT_VERSION,
            "planner_model": str(model),
            **_planner_prompt_lineage(),
        }

    def _planner_cache_path(
        self, question: str
    ) -> tuple[Path, dict[str, str]] | None:
        lineage = self._planner_cache_lineage(question)
        cache_dir = getattr(self, "_planner_cache_dir", None)
        if lineage is None or cache_dir is None:
            return None
        # The normalized question itself participates in the key.  Only its
        # digest is retained as lineage in the cache value.
        key_material = {
            "normalized_question": _normalized_planner_question(question),
            "graph_hash": lineage["graph_hash"],
            "prompt_version": lineage["prompt_version"],
            "planner_model": lineage["planner_model"],
            "planner_prompt_sha256": lineage["planner_prompt_sha256"],
            "planner_clients_sha256": lineage["planner_clients_sha256"],
        }
        key = hashlib.sha256(_canonical_json(key_material).encode("utf-8")).hexdigest()
        return Path(cache_dir) / f"{key}.json", lineage

    def _load_planner_cache(self, question: str) -> QueryPlan | None:
        location = self._planner_cache_path(question)
        if location is None:
            return None
        path, expected_lineage = location
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None
            if set(value) != {"lineage", "query_plan"}:
                return None
            lineage = value.get("lineage")
            if not isinstance(lineage, dict) or any(
                lineage.get(key) != expected
                for key, expected in expected_lineage.items()
            ):
                return None
            if lineage.get("cache_schema_version") != PLANNER_CACHE_SCHEMA_VERSION:
                return None
            plan_value = value.get("query_plan")
            expected_hash = lineage.get("query_plan_sha256")
            if not isinstance(expected_hash, str) or expected_hash != hashlib.sha256(
                _canonical_json(plan_value).encode("utf-8")
            ).hexdigest():
                return None
            return _query_plan_from_dict(plan_value)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _store_planner_cache(self, question: str, plan: QueryPlan) -> None:
        location = self._planner_cache_path(question)
        if location is None:
            return
        path, lineage = location
        plan_value = asdict(plan)
        lineage = {
            **lineage,
            "cache_schema_version": PLANNER_CACHE_SCHEMA_VERSION,
            "query_plan_sha256": hashlib.sha256(
                _canonical_json(plan_value).encode("utf-8")
            ).hexdigest(),
        }
        value = {
            "lineage": lineage,
            "query_plan": plan_value,
        }
        temp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(
                    value,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                temp_path = Path(handle.name)
            temp_path.replace(path)
        except (OSError, TypeError, ValueError):
            # Cache persistence must never turn a valid planner response into
            # an evaluation failure.
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def set_call_limit(self, limit: int) -> None:
        self._call_limit = max(0, min(self._max_calls, int(limit)))

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def input_tokens(self) -> int:
        usage = getattr(self.collector, "usage", None)
        return int(getattr(usage, "input_tokens", 0) or 0)

    @property
    def output_tokens(self) -> int:
        usage = getattr(self.collector, "usage", None)
        return int(getattr(usage, "output_tokens", 0) or 0)

    @property
    def cached_input_tokens(self) -> int:
        usage = getattr(self.collector, "usage", None)
        baml_total = _nonnegative_token_count(
            getattr(usage, "cached_input_tokens", None)
        )
        responses_total = max(
            _selected_response_usage(
                self.collector,
                "usage",
                "input_tokens_details",
                "cached_tokens",
            ),
            _selected_response_usage(
                self.collector,
                "usage",
                "prompt_tokens_details",
                "cached_tokens",
            ),
        )
        return max(baml_total or 0, responses_total)

    @property
    def reasoning_output_tokens(self) -> int:
        return max(
            _selected_response_usage(
                self.collector,
                "usage",
                "output_tokens_details",
                "reasoning_tokens",
            ),
            _selected_response_usage(
                self.collector,
                "usage",
                "completion_tokens_details",
                "reasoning_tokens",
            ),
        )

    def _call(self, name: str, fn):
        remaining = min(self._max_calls, self._call_limit) - self._calls
        if remaining <= 0:
            raise RuntimeError(
                f"ToG LLM call budget exhausted ({min(self._max_calls, self._call_limit)})"
            )

        def counted_call():
            self._calls += 1
            try:
                return fn()
            except (KeyboardInterrupt, SystemExit):
                raise
            except TimeoutError:
                raise
            except BaseException as exc:
                # pyo3 surfaces Rust panics as PanicException(BaseException),
                # which would otherwise bypass the shared bounded retry/error
                # path and terminate the entire 59-question run.
                raise RuntimeError(f"BAML runtime failure: {type(exc).__name__}: {exc}") from exc

        result = call_baml_with_retry(
            counted_call,
            # Every ToG phase has a deterministic fallback. Retrying the same
            # 180-second HTTP failure multiplies latency and consumes the call
            # budget without adding evidence.
            max_retries=1,
            context_name=name,
        )
        if isinstance(result, AgentError):
            raise RuntimeError(f"{name}: {result.error_type}: {result.error_message}")
        return result

    def _options(self):
        return b.with_options(collector=self.collector, client_registry=self.registry)

    def _prompt_evidence(self, evidence: Sequence[TripleEvidence]) -> list[str]:
        result: list[str] = []
        used = 0
        for item in evidence[: self._prompt_evidence_limit]:
            line = item.prompt_line()
            remaining = self._prompt_evidence_max_chars - used
            if remaining <= 0:
                break
            if len(line) > remaining:
                if result:
                    break
                line = line[:remaining]
            result.append(line)
            used += len(line)
        return result

    def plan_query(
        self,
        question: str,
        fallback: QueryPlan,
        max_gnn_hops: int,
        max_tog_depth: int,
        max_tog_width: int,
    ) -> QueryPlan:
        cached = self._load_planner_cache(question)
        if cached is not None:
            return cached
        result = self._call(
            "PlanIfcToGQuery",
            lambda: self._options().PlanIfcToGQuery(
                question=question,
                fallback_plan_json=_compact_planner_payload(fallback),
                max_gnn_hops=max_gnn_hops,
                max_tog_depth=max_tog_depth,
                max_tog_width=max_tog_width,
            ),
        )
        plan = QueryPlan(
            operator=_enum_value(result.operator),
            mentions=list(result.mentions),
            storey=result.storey,
            room=result.room,
            room_names=list(result.room_names),
            target_kind=result.target_kind,
            target_ifc_class=result.target_ifc_class,
            target_role=result.target_role,
            target_roles=list(result.target_roles),
            target_domain=result.target_domain,
            target_space_type=result.target_space_type,
            target_name=result.target_name,
            target_names=list(result.target_names),
            target_family_terms=list(result.target_family_terms),
            target_type_terms=list(result.target_type_terms),
            target_keywords=list(result.target_keywords),
            property_terms=list(result.property_terms),
            action_sequence=list(result.action_sequence),
            action_bindings=[
                ActionTargetBinding(
                    action=item.action,
                    binding_index=int(item.binding_index),
                    target_kind=item.target_kind,
                    target_roles=list(item.target_roles),
                    target_names=list(item.target_names),
                    target_domains=list(item.target_domains),
                    function_types=list(item.function_types),
                    system_categories=list(item.system_categories),
                    cardinality_policy=str(item.cardinality_policy),
                )
                for item in result.action_bindings
            ],
            requires_exhaustive=result.requires_exhaustive,
            gnn_hops=result.gnn_hops,
            tog_depth=result.tog_depth,
            tog_width=result.tog_width,
            gnn_retrieval_levels=list(result.gnn_retrieval_levels),
            rationale=result.rationale,
        )
        self._store_planner_cache(question, plan)
        return plan

    def select_query_hypothesis(
        self,
        question: str,
        hypotheses: Sequence[QueryHypothesis],
    ) -> QueryHypothesisSelectionResult:
        """Select one supplied interpretation without exposing graph IDs.

        Hypotheses are represented to the model as call-local aliases (H1,
        H2, ...).  The returned alias is checked against that exact whitelist
        and mapped back to the caller-owned hypothesis identity.
        """

        aliases = _query_hypothesis_aliases(hypotheses)
        if not aliases:
            return QueryHypothesisSelectionResult(
                status="unsupported",
                hypothesis_count=0,
                reason="no_query_hypotheses",
            )
        payload = _compact_query_hypotheses_payload(hypotheses)
        result = self._call(
            "SelectIfcQueryHypothesis",
            lambda: self._options().SelectIfcQueryHypothesis(
                question=question,
                hypotheses_json=payload,
            ),
        )
        status = _enum_value(result.status)
        allowed_statuses = {
            "selected", "ambiguous", "unsupported", "contradictory"
        }
        if status not in allowed_statuses:
            status = "unsupported"

        selected_alias = str(
            getattr(result, "selected_hypothesis_id", "") or ""
        )
        selected_hypothesis_id = aliases.get(selected_alias)
        reason = str(getattr(result, "reason", "") or "")
        if status == "selected" and selected_hypothesis_id is None:
            status = "unsupported"
            reason = (
                "invalid_or_missing_hypothesis_id"
                + (f":{reason}" if reason else "")
            )
        elif status != "selected":
            selected_hypothesis_id = None

        rejected: list[str] = []
        for value in getattr(result, "rejected_hypothesis_ids", []) or []:
            actual = aliases.get(str(value))
            if (
                actual is not None
                and actual != selected_hypothesis_id
                and actual not in rejected
            ):
                rejected.append(actual)
        return QueryHypothesisSelectionResult(
            status=status,
            selected_hypothesis_id=selected_hypothesis_id,
            rejected_hypothesis_ids=rejected,
            hypothesis_count=len(hypotheses),
            llm_used=True,
            reason=f"bounded_query_hypothesis:{reason or status}",
        )

    def select_relations(
        self,
        question: str,
        entity: EntityRef,
        relations: Sequence[RelationRef],
        width: int,
    ) -> list[RelationRef]:
        result = self._call(
            "SelectIfcToGRelations",
            lambda: self._options().SelectIfcToGRelations(
                question=question,
                entity_label=entity.label,
                relation_candidates=[f"{item.direction}|{item.name}" for item in relations],
                width=width,
            ),
        )
        allowed = {(item.name, item.direction) for item in relations}
        selected = [
            RelationRef(item.name, item.direction, max(0.0, min(1.0, float(item.score))))
            for item in result.relations
            if (item.name, item.direction) in allowed
        ]
        return selected[:width]

    def score_entities(
        self,
        question: str,
        relation: RelationRef,
        entities: Sequence[EntityRef],
        width: int,
    ) -> list[EntityRef]:
        result = self._call(
            "ScoreIfcToGEntities",
            lambda: self._options().ScoreIfcToGEntities(
                question=question,
                relation=f"{relation.direction}|{relation.name}",
                entity_candidates=[
                    f"{item.node_id}|{item.label}|{item.ifc_class or item.kind}|"
                    f"{json.dumps(item.metadata, ensure_ascii=False, default=str)}"
                    for item in entities
                ],
                width=width,
            ),
        )
        by_id = {item.node_id: item for item in entities}
        selected: list[EntityRef] = []
        for scored in result.entities:
            entity = by_id.get(scored.node_id)
            if entity is None:
                continue
            entity.score = relation.score * max(0.0, min(1.0, float(scored.score)))
            selected.append(entity)
        selected.sort(key=lambda item: (-item.score, item.node_id))
        return selected[:width]

    def resolve_targets(
        self,
        question: str,
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
    ) -> TargetSelectionResult:
        """Resolve one bounded candidate set without permitting new node IDs."""
        allowed = {item.node_id for item in candidates}
        payload = [
            {
                "node_id": item.node_id,
                "label": item.label,
                "global_id": item.global_id,
                "ifc_class": item.ifc_class,
                "score": item.score,
                "provenance": item.match_reason,
                "constraints": item.metadata.get("_constraint_status", {}),
                "features": {
                    key: value
                    for key, value in item.metadata.items()
                    if key in {
                        "name", "long_name", "family", "type_name", "role",
                        "domain", "storey", "space_type", "system_category",
                        "action_target_kind", "function_type",
                    }
                    and value not in (None, "")
                },
            }
            for item in candidates
        ]
        result = self._call(
            "ResolveIfcToGTargets",
            lambda: self._options().ResolveIfcToGTargets(
                question=question,
                plan_json=json.dumps(asdict(plan), ensure_ascii=False, default=str),
                candidate_constraints_json=json.dumps(
                    payload, ensure_ascii=False, default=str
                ),
            ),
        )
        selected_ids = list(dict.fromkeys(
            str(value) for value in result.selected_ids if str(value) in allowed
        ))
        rejected_ids = list(dict.fromkeys(
            str(value) for value in result.rejected_ids
            if str(value) in allowed and str(value) not in selected_ids
        ))
        status = _enum_value(result.status)
        if status not in {"resolved", "ambiguous", "unsupported", "contradictory"}:
            status = "unsupported"
        if status != "resolved":
            selected_ids = []
        return TargetSelectionResult(
            selected_ids=selected_ids,
            rejected_ids=rejected_ids,
            candidate_count=len(candidates),
            llm_used=True,
            reason=f"constrained_slot_resolver:{status}:{result.reason}",
        )

    def is_sufficient(
        self,
        question: str,
        evidence: Sequence[TripleEvidence],
    ) -> bool:
        result = self._call(
            "AssessIfcToGEvidence",
            lambda: self._options().AssessIfcToGEvidence(
                question=question,
                evidence=[item.prompt_line() for item in evidence[-100:]],
            ),
        )
        return bool(result.sufficient)

    def generate_answer(
        self,
        question: str,
        plan: QueryPlan,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
    ) -> str:
        result = self._call(
            "GenerateIfcToGAnswer",
            lambda: self._options().GenerateIfcToGAnswer(
                question=question,
                output_contract_json=_query_plan_output_contract(plan),
                plan_json=json.dumps(asdict(plan), ensure_ascii=False),
                evidence=self._prompt_evidence(evidence),
                deterministic_answer=deterministic_answer,
            ),
        )
        return result.answer

    @staticmethod
    def _audit_ids(values: Sequence[str], evidence_count: int, hierarchy_count: int) -> list[str]:
        allowed = {
            *(f"evidence:{index}" for index in range(1, evidence_count + 1)),
            *(f"hierarchy:{index}" for index in range(1, hierarchy_count + 1)),
        }
        return [str(value) for value in values if str(value) in allowed]

    def reason_hierarchy(
        self,
        question: str,
        plan: QueryPlan,
        hierarchy_context: HierarchyContext,
        evidence: Sequence[TripleEvidence],
        deterministic_answer: str,
        reasoning_mode: ReasoningMode,
    ) -> HierarchyReasoningTrace:
        prompt_evidence = list(evidence[: self._prompt_evidence_limit])
        prompt_evidence_lines = self._prompt_evidence(prompt_evidence)
        prompt_paths = hierarchy_context.prompt_lines(100)
        result = self._call(
            "ReasonIfcToGHierarchy",
            lambda: self._options().ReasonIfcToGHierarchy(
                question=question,
                output_contract_json=_query_plan_output_contract(plan),
                plan_json=json.dumps(asdict(plan), ensure_ascii=False),
                reasoning_mode=reasoning_mode,
                hierarchy_summary_json=json.dumps(
                    hierarchy_context.summary, ensure_ascii=False, default=str
                ),
                hierarchy_paths=prompt_paths,
                evidence=prompt_evidence_lines,
                deterministic_answer=deterministic_answer,
            ),
        )
        requirements: list[ReasoningRequirement] = []
        for item in result.requirements:
            status = str(item.status).lower()
            if status not in {"satisfied", "missing", "conflicting"}:
                status = "missing"
            requirements.append(
                ReasoningRequirement(
                    requirement=item.requirement,
                    status=status,
                    evidence_ids=self._audit_ids(
                        item.evidence_ids, len(prompt_evidence_lines), len(prompt_paths)
                    ),
                    conclusion=item.conclusion,
                )
            )
        return HierarchyReasoningTrace(
            requirements=requirements,
            hierarchy_claims=list(result.hierarchy_claims),
            conclusion=result.conclusion,
            sufficient=bool(result.sufficient),
            missing_evidence=list(result.missing_evidence),
            conflicting_evidence=list(result.conflicting_evidence),
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
        prompt_evidence = list(evidence[: self._prompt_evidence_limit])
        prompt_evidence_lines = self._prompt_evidence(prompt_evidence)
        prompt_paths = hierarchy_context.prompt_lines(100)
        result = self._call(
            "ReviewIfcToGEvidence",
            lambda: self._options().ReviewIfcToGEvidence(
                question=question,
                output_contract_json=_query_plan_output_contract(plan),
                plan_json=json.dumps(asdict(plan), ensure_ascii=False),
                reasoning_mode=reasoning_mode,
                hierarchy_summary_json=json.dumps(
                    hierarchy_context.summary, ensure_ascii=False, default=str
                ),
                hierarchy_paths=prompt_paths,
                evidence=prompt_evidence_lines,
                deterministic_answer=deterministic_answer,
                reasoning_trace_json=json.dumps(
                    asdict(reasoning_trace), ensure_ascii=False, default=str
                ),
                target_validation_summary_json=json.dumps(
                    target_audit.validation_summary, ensure_ascii=False, default=str
                ),
                invalid_targets_json=json.dumps(
                    [
                        asdict(item) for item in target_audit.target_validations
                        if not item.valid
                    ],
                    ensure_ascii=False,
                    default=str,
                ),
                excluded_candidates_json=json.dumps(
                    target_audit.excluded_candidates, ensure_ascii=False, default=str
                ),
                action_bindings_json=json.dumps(
                    [asdict(item) for item in plan.action_bindings], ensure_ascii=False
                ),
            ),
        )
        decision = _enum_value(result.decision)
        if decision not in {"pass", "repair", "abstain"}:
            decision = "repair"
        coverage = [
            EvidenceCoverage(
                requirement=item.requirement,
                covered=bool(item.covered),
                evidence_ids=self._audit_ids(
                    item.evidence_ids, len(prompt_evidence_lines), len(prompt_paths)
                ),
                note=item.note,
            )
            for item in result.coverage
        ]
        allowed_relations = {
            "contains",
            "part_of",
            "adjacent_to",
            "connects_to",
            "serves",
            "requires_inspection_of",
            "related_to_system",
            "assigned_to_system",
        }
        return EvidenceReview(
            decision=decision,
            evidence_summary=result.evidence_summary,
            intent_aligned=bool(result.intent_aligned),
            reasoning_supported=bool(result.reasoning_supported),
            hierarchy_consistent=bool(result.hierarchy_consistent),
            deterministic_complete=bool(result.deterministic_complete),
            hierarchy_required=bool(result.hierarchy_required),
            coverage=coverage,
            target_validations=list(target_audit.target_validations),
            audited_target_count=len(target_audit.included_targets),
            action_bindings_covered=bool(result.action_bindings_covered),
            conflicting_extras=bool(result.conflicting_extras),
            missing_requirements=list(result.missing_requirements),
            needed_relations=[
                relation for relation in result.needed_relations
                if relation in allowed_relations
            ],
            reason=result.reason,
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
        result = self._call(
            "FinalizeIfcToGAnswer",
            lambda: self._options().FinalizeIfcToGAnswer(
                question=question,
                output_contract_json=_query_plan_output_contract(plan),
                plan_json=json.dumps(asdict(plan), ensure_ascii=False),
                deterministic_answer=deterministic_answer,
                hierarchy_paths=hierarchy_context.prompt_lines(100),
                evidence=self._prompt_evidence(evidence),
                reasoning_trace_json=json.dumps(
                    asdict(reasoning_trace), ensure_ascii=False, default=str
                ),
                evidence_review_json=json.dumps(
                    asdict(evidence_review), ensure_ascii=False, default=str
                ),
            ),
        )
        return result.answer

    def compile_retrieval_intent_contract_v3(
        self,
        question: str,
        compact_plan_json: str,
    ) -> dict[str, Any]:
        result = self._call(
            "CompileIfcRetrievalIntentContractV3",
            lambda: self._options().CompileIfcRetrievalIntentContractV3(
                question=question,
                compact_plan_json=compact_plan_json,
            ),
        )
        return {
            "bindings": [
                {
                    "binding_index": int(item.binding_index),
                    "action": str(item.action),
                    "selection_mode": _enum_value(item.selection_mode),
                    "target_phrase": str(item.target_phrase),
                    "target_terms": list(item.target_terms),
                    "target_kind": str(item.target_kind),
                    "target_ifc_class": str(item.target_ifc_class),
                    "scope_phrase": str(item.scope_phrase),
                    "scope_terms": list(item.scope_terms),
                    "relation": str(item.relation),
                    "direction": str(item.direction),
                    "requires_exhaustive_set": bool(item.requires_exhaustive_set),
                }
                for item in result.bindings
            ],
            "summary": str(result.summary),
        }

    def adjudicate_and_answer_evidence_groups_v3(
        self,
        question: str,
        intent_contract_json: str,
        candidate_group_ledger_json: str,
    ) -> dict[str, Any]:
        result = self._call(
            "AdjudicateAndAnswerIfcEvidenceGroupsV3",
            lambda: self._options().AdjudicateAndAnswerIfcEvidenceGroupsV3(
                question=question,
                intent_contract_json=intent_contract_json,
                candidate_group_ledger_json=candidate_group_ledger_json,
            ),
        )
        return {
            "binding_selections": [
                {
                    "binding_index": int(item.binding_index),
                    "selected_group_ids": list(item.selected_group_ids),
                    "uncertain_group_ids": list(item.uncertain_group_ids),
                    "reason": str(item.reason),
                }
                for item in result.binding_selections
            ],
            "closure_supported": bool(result.closure_supported),
            "issues": list(result.issues),
            "answer_summary": str(result.answer_summary),
        }

    def review_and_answer_evidence_groups_v3(
        self,
        question: str,
        intent_contract_json: str,
        candidate_group_ledger_json: str,
        initial_selection_json: str,
    ) -> dict[str, Any]:
        result = self._call(
            "ReviewAndAnswerIfcEvidenceGroupsV3",
            lambda: self._options().ReviewAndAnswerIfcEvidenceGroupsV3(
                question=question,
                intent_contract_json=intent_contract_json,
                candidate_group_ledger_json=candidate_group_ledger_json,
                initial_selection_json=initial_selection_json,
            ),
        )
        return {
            "binding_selections": [
                {
                    "binding_index": int(item.binding_index),
                    "selected_group_ids": list(item.selected_group_ids),
                    "uncertain_group_ids": list(item.uncertain_group_ids),
                    "reason": str(item.reason),
                }
                for item in result.binding_selections
            ],
            "closure_supported": bool(result.closure_supported),
            "issues": list(result.issues),
            "answer_summary": str(result.answer_summary),
        }

CURRENT_PROFILE = "paper-v14-clean-v1"
CURRENT_RETRIEVAL_PROFILE = "query-conditioned-plan-v5.0"
CURRENT_PLUGIN_ID = "text-gnn-v5.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_v5_inspection_graph(
    artifact_dir: str | Path,
    requested_path: str | Path | None,
    *,
    expected_manifest_sha256: str,
    expected_graph_sha256: str,
) -> tuple[Path, str]:
    """Resolve the query-free graph shipped in the hash-bound v5 runtime."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256.casefold():
        raise RuntimeError("Text-GNN v5 runtime manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_ifc_sha256 = str(manifest.get("source_ifc_sha256") or "").casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", source_ifc_sha256):
        raise RuntimeError("Text-GNN v5 runtime source IFC lineage is missing")
    graph_row = manifest.get("inspection_graph")
    if not isinstance(graph_row, Mapping):
        raise RuntimeError("Text-GNN v5 runtime does not contain an inspection graph")
    relative = Path(str(graph_row.get("path") or ""))
    packaged = (root / relative).resolve()
    if root not in packaged.parents or not packaged.is_file():
        raise RuntimeError("Text-GNN v5 inspection graph path escapes the runtime")
    packaged_sha256 = _sha256_file(packaged)
    expected = expected_graph_sha256.casefold()
    if (
        packaged_sha256 != expected
        or str(graph_row.get("sha256") or "").casefold() != expected
        or str(manifest.get("tog_runtime_graph_sha256") or "").casefold()
        != expected
    ):
        raise RuntimeError("Text-GNN v5 inspection graph lineage mismatch")
    if requested_path is None:
        return packaged, source_ifc_sha256
    requested = Path(requested_path).expanduser().resolve()
    if not requested.is_file() or _sha256_file(requested) != expected:
        raise RuntimeError(
            "Requested inspection graph differs from the frozen v5 runtime graph"
        )
    return requested, source_ifc_sha256


class _GraphScopedPlannerCacheToGSystem(ToGSystem):
    """Bind generic planner-cache entries to the active graph lineage."""

    def __init__(self, *args, expected_source_ifc_sha256: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_source_ifc_sha256 = expected_source_ifc_sha256

    def _set_graph_scope(self, model_path: str | Path):
        source_path = Path(model_path).expanduser().resolve()
        if (
            self.expected_source_ifc_sha256 is not None
            and _sha256_file(source_path) != self.expected_source_ifc_sha256
        ):
            raise RuntimeError("IFC SHA-256 differs from the frozen Text-GNN v5 lineage")
        report = self.index_reports.get(source_path)
        if report is None:
            report = self.index_manager.ensure_index(
                source_path, rebuild=self.rebuild_index
            )
            self.index_reports[source_path] = report
        backend = self._backend(report)
        graph_hash = str(
            backend.meta.get("representation_graph_hash") or backend.graph_hash
        )
        return _ACTIVE_PLANNER_GRAPH_HASH.set(graph_hash)

    def ask(self, question: str, *, model_path: str | Path):
        token = self._set_graph_scope(model_path)
        try:
            return super().ask(question, model_path=model_path)
        finally:
            _ACTIVE_PLANNER_GRAPH_HASH.reset(token)

    def prepare_retrieval_packet(
        self, question: str, *, model_path: str | Path
    ):
        token = self._set_graph_scope(model_path)
        try:
            return super().prepare_retrieval_packet(question, model_path=model_path)
        finally:
            _ACTIVE_PLANNER_GRAPH_HASH.reset(token)


def create_tog_system(
    *,
    cache_dir: str | Path,
    variant: str,
    client: str,
    profile: str = CURRENT_PROFILE,
    retrieval_flow_profile: str = "legacy",
    width: int = 3,
    depth: int = 3,
    max_llm_calls: int = 48,
    rebuild_index: bool = False,
    debug: bool = False,
    inspection_graph_path: str | Path | None = None,
    gnn_artifact_dir: str | Path | None = None,
    gnn_retrieval_profile: str = CURRENT_RETRIEVAL_PROFILE,
    gnn_representation: str = "four-level",
    retriever_plugin: str | None = None,
    retriever_plugin_manifest_sha256: str | None = None,
    retriever_plugin_runtime_graph_sha256: str | None = None,
    retriever_plugin_device: str = "cuda",
    gnn_max_hops: int = 3,
    gnn_max_depth: int = 4,
    gnn_max_width: int = 5,
    gnn_anchor_k: int = 20,
    gnn_max_subgraph_nodes: int = 200,
    gnn_max_subgraph_edges: int = 2_000,
    use_gnn_prior: bool | None = None,
    hierarchy_reasoning: bool = False,
    hierarchy_max_repairs: int = 3,
    hierarchy_precision_gate: bool = False,
    hierarchy_path_control: str = "full",
    llm_answering_required: bool = True,
    deterministic_planner: bool = False,
    bounded_query_hypotheses: bool = False,
    semantic_enabled: bool = True,
    semantic_llm_calls_per_question: int = 1,
    semantic_input_token_budget: int = 2_000,
    input_token_budget: int | None = None,
    query_hypothesis_cap: int = 4,
) -> ToGSystem:
    """Create one member of the current ToG family.

    The adapter accepts no evaluator category, gold, question ID, frozen
    benchmark plan or historical retrieval lane.  The two experimental switches
    are represented by ``variant``/``use_gnn_prior`` and
    ``hierarchy_reasoning``; closure v3 is legal only for the hierarchy arm.
    """

    if profile != CURRENT_PROFILE:
        raise ValueError(f"Unsupported ToG profile: {profile}")
    if variant not in {"bim", "bim-gnn"}:
        raise ValueError("Current ToG family supports only 'bim' and 'bim-gnn'")
    if gnn_retrieval_profile != CURRENT_RETRIEVAL_PROFILE:
        raise ValueError(
            f"Unsupported GNN retrieval profile: {gnn_retrieval_profile}"
        )
    if gnn_representation not in {"four-level", "flat"}:
        raise ValueError(f"Unsupported GNN representation: {gnn_representation}")
    if gnn_representation != "four-level":
        raise ValueError("The current ToG family freezes four-level v5 retrieval")
    if hierarchy_path_control not in {"full", "none", "shuffled"}:
        raise ValueError(f"Unsupported hierarchy path control: {hierarchy_path_control}")
    if retrieval_flow_profile not in {"legacy", "closure-adjudication-v3"}:
        raise ValueError(f"Unsupported retrieval flow profile: {retrieval_flow_profile}")

    gnn_enabled = variant == "bim-gnn" if use_gnn_prior is None else bool(use_gnn_prior)
    if (variant == "bim-gnn") is not gnn_enabled:
        raise ValueError("variant and use_gnn_prior must describe the same ToG arm")
    if hierarchy_reasoning and not gnn_enabled:
        raise ValueError("Hierarchy requires frozen Text-GNN v5 retrieval")
    if hierarchy_reasoning is not (
        retrieval_flow_profile == "closure-adjudication-v3"
    ):
        raise ValueError(
            "The current hierarchy arm and closure-adjudication-v3 must be enabled together"
        )

    frozen_source_ifc_sha256 = None
    if gnn_enabled:
        if retriever_plugin != CURRENT_PLUGIN_ID:
            raise ValueError(
                f"GNN arms require --tog-retriever-plugin {CURRENT_PLUGIN_ID}"
            )
        if gnn_artifact_dir is None:
            raise ValueError("Text-GNN v5 requires a runtime artifact")
        if not retriever_plugin_manifest_sha256:
            raise ValueError("Text-GNN v5 requires the runtime manifest SHA-256")
        if not retriever_plugin_runtime_graph_sha256:
            raise ValueError("Text-GNN v5 requires the runtime graph SHA-256")
        inspection_graph_path, frozen_source_ifc_sha256 = _resolve_v5_inspection_graph(
            gnn_artifact_dir,
            inspection_graph_path,
            expected_manifest_sha256=str(retriever_plugin_manifest_sha256),
            expected_graph_sha256=str(retriever_plugin_runtime_graph_sha256),
        )
    elif any(
        value is not None
        for value in (
            retriever_plugin,
            gnn_artifact_dir,
            retriever_plugin_manifest_sha256,
            retriever_plugin_runtime_graph_sha256,
        )
    ):
        raise ValueError("ToG-BIM must not load a GNN plugin or runtime artifact")

    builder = (
        FrozenInspectionGraphBuilder(
            inspection_graph_path,
            edge_profile=CURRENT_RETRIEVAL_PROFILE,
        )
        if inspection_graph_path is not None
        else None
    )
    manager = GraphIndexManager(cache_dir, builder=builder)
    plugin = None
    if gnn_enabled:
        plugin = load_retriever_plugin(
            CURRENT_PLUGIN_ID,
            RetrievalPluginContext(
                artifact_dir=str(Path(gnn_artifact_dir).expanduser().resolve()),
                expected_artifact_manifest_sha256=str(
                    retriever_plugin_manifest_sha256
                ),
                expected_runtime_graph_sha256=str(
                    retriever_plugin_runtime_graph_sha256
                ),
                settings={
                    "anchor_k": int(gnn_anchor_k),
                    "evaluation_anchor_k": 50,
                    "max_nodes": int(gnn_max_subgraph_nodes),
                    "max_edges": int(gnn_max_subgraph_edges),
                    "max_hops": int(gnn_max_hops),
                    "representation": gnn_representation,
                    "device": retriever_plugin_device,
                    "message_passing": True,
                    "query_embedding_cache_dir": str(
                        Path(cache_dir).expanduser().resolve()
                        / "text_gnn_v5_query_embeddings"
                    ),
                },
            ),
        )

    config = ToGConfig(
        # Core ``legacy`` means the original full ToG traversal, not an old GNN
        # integration.  The public adapter profile above is the leakage-safe
        # paper-v14-clean-v1 contract.
        profile="legacy",
        retrieval_flow_profile=retrieval_flow_profile,
        variant=variant,
        width=int(width),
        depth=int(depth),
        max_llm_calls=int(max_llm_calls),
        include_evidence=("explicit", "inferred", "candidate"),
        gnn_max_hops=int(gnn_max_hops),
        gnn_max_depth=int(gnn_max_depth),
        gnn_max_width=int(gnn_max_width),
        gnn_anchor_k=int(gnn_anchor_k),
        gnn_max_subgraph_nodes=int(gnn_max_subgraph_nodes),
        gnn_max_subgraph_edges=int(gnn_max_subgraph_edges),
        use_gnn_prior=gnn_enabled,
        gnn_retrieval_profile=CURRENT_RETRIEVAL_PROFILE,
        gnn_representation=gnn_representation,
        query_routing_profile="query-only-v1",
        gnn_use_anchors=gnn_enabled,
        gnn_use_expansion=bool(hierarchy_reasoning),
        hierarchy_reasoning=bool(hierarchy_reasoning),
        hierarchy_max_repairs=int(hierarchy_max_repairs),
        hierarchy_precision_gate=bool(hierarchy_precision_gate),
        hierarchy_path_control=hierarchy_path_control,
        llm_answering_required=bool(llm_answering_required),
        bounded_query_hypotheses=bool(bounded_query_hypotheses),
        semantic_enabled=bool(semantic_enabled),
        semantic_llm_calls_per_question=int(semantic_llm_calls_per_question),
        semantic_input_token_budget=int(semantic_input_token_budget),
        input_token_budget=input_token_budget,
        query_hypothesis_cap=int(query_hypothesis_cap),
        deterministic_planner=bool(deterministic_planner),
        debug=bool(debug),
    )
    return _GraphScopedPlannerCacheToGSystem(
        index_manager=manager,
        llm_factory=lambda: BamlToGLlm(
            client,
            config.effective_max_llm_calls,
            prompt_evidence_limit=config.effective_prompt_evidence_limit,
            prompt_evidence_max_chars=config.effective_prompt_evidence_max_chars,
            planner_cache_dir=Path(cache_dir).expanduser().resolve()
            / "structured_planner",
        ),
        config=config,
        rebuild_index=bool(rebuild_index),
        gnn_retriever=plugin,
        expected_source_ifc_sha256=frozen_source_ifc_sha256,
    )


__all__ = [
    "BamlToGLlm",
    "CURRENT_PLUGIN_ID",
    "CURRENT_PROFILE",
    "CURRENT_RETRIEVAL_PROFILE",
    "create_tog_system",
    "ensure_tog_importable",
]
