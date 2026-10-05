from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .models import (
    ActionTargetBinding,
    CardinalityPolicy,
    EntityRef,
    MentionLink,
    QueryPlan,
    RelationReference,
    ScopePredicate,
    TargetPredicate,
)

# Only language-level constructs are built in here.  Asset, space, system and
# function terms are supplied by ``PlanningSchema`` from the active graph.
_ACTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:navigate|go|move|proceed|travel|be)\s+(?:to|into|in|by)\b", "Navigate"),
    (r"\b(?:inspect|check|examine|verify|assess)\b", "Inspect"),
    (
        r"\b(?:investigate|diagnose|troubleshoot|infer)\b"
        r"(?=.{0,320}\b(?:abnormal(?:ly)?|root\s+cause|fault|failure|"
        r"insufficient|inspection\s+targets?|components?)\b)",
        "Inspect",
    ),
    (r"\b(?:scan|survey|image)\b", "Scan"),
)
_SCENARIO_PREFIX = re.compile(r"^(?:what\s+(?:happens\s+)?if|if)\b", re.IGNORECASE)
_SCENARIO_STATE_CHANGE = re.compile(
    r"\b(?:shut(?:s|ting)?\s+down|turned?\s+off|fails?|malfunctions?)\b",
    re.IGNORECASE,
)
_SCENARIO_IMPACT = re.compile(r"\b(?:impact(?:ed)?|affect(?:ed)?)\b", re.IGNORECASE)
_SPACE_HEADS = {
    "area",
    "classroom",
    "corridor",
    "floor",
    "hall",
    "hallway",
    "laboratory",
    "lab",
    "lobby",
    "office",
    "restroom",
    "room",
    "space",
    "stair",
    "studio",
    "suite",
    "workshop",
    "zone",
}
_GENERIC_NOUNS = {
    "asset",
    "component",
    "device",
    "element",
    "equipment",
    "fixture",
    "item",
    "object",
}
_DETERMINERS = re.compile(
    r"^(?:(?:the|a|an|all|each|every|any|one|those|these)\s+)+",
    flags=re.IGNORECASE,
)
_LINK_STOPWORDS = {
    "a", "an", "and", "any", "all", "each", "every", "for", "in", "inside",
    "inspect", "of", "on", "scan", "the", "those", "these", "to", "within",
}


def normalize_question_text(text: str) -> str:
    """Return stable matching text without altering stored graph labels.

    Unicode compatibility normalization makes typographic hyphens and width
    variants comparable.  Ampersand is normalized for matching only; linked
    names retain their exact graph spelling in the resulting plan.
    """

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("&", " and ")
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    # Split mixed-case graph labels before case folding.  The two boundaries
    # cover both ordinary CamelCase (``FireGuard``) and acronym-to-word
    # transitions (``HVACController``).  This is tokenization only: trailing
    # identifiers remain present rather than being stripped as noisy suffixes.
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    # Treat lexical hyphens as token boundaries while preserving every suffix
    # token (for example, A-101 -> A 101).  This makes graph labels written as
    # "fire protection" match instructions written as "fire-protection"
    # without stripping identifiers or injecting a synonym.
    value = value.replace("-", " ")
    value = value.casefold()
    value = re.sub(r"[_/]+", " ", value)
    value = re.sub(r"[^\w\s+\-']", " ", value)
    return " ".join(value.split())


def _singular_token(token: str) -> str:
    """Conservative English inflection matching for graph labels."""

    if len(token) > 5 and token.endswith("ing"):
        # Conservative derivational normalization lets schema role labels such
        # as ``light_fixture`` match ordinary language such as "lighting
        # fixtures" without an asset-specific synonym table.
        stem = token[:-3]
        if len(stem) >= 4:
            return stem
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes", "sses")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _signature(text: str) -> tuple[str, ...]:
    return tuple(_singular_token(token) for token in normalize_question_text(text).split())


def _compact_identifier(text: str) -> str:
    """Normalize a schema identifier independently of display tokenization."""

    return "".join(normalize_question_text(text).split())


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        key = normalize_question_text(value)
        if not value or not key or key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (Mapping, bytes)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _record_dict(record: Mapping[str, Any] | EntityRef) -> dict[str, Any]:
    if isinstance(record, EntityRef):
        return {
            **record.metadata,
            "node_id": record.node_id,
            "label": record.label,
            "global_id": record.global_id,
            "ifc_class": record.ifc_class,
            "kind": record.kind,
        }
    result = dict(record)
    metadata = result.get("metadata")
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            result.setdefault(str(key), value)
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        for key in (
            "aliases",
            "synonyms",
            "target_roles",
            "function_type",
            "function_kind",
            "system_category",
            "target_name_terms",
            "target_domains",
            "target_space_types",
            "related_system_categories",
        ):
            if key in properties:
                result.setdefault(key, properties[key])
    return result


def _record_kind(record: Mapping[str, Any]) -> str:
    level = str(record.get("level") or "").casefold()
    category = str(record.get("category") or record.get("kind") or "").casefold()
    ifc_class = str(record.get("ifc_class") or "").casefold()
    if "function" in level or category == "function":
        return "function"
    if "system" in level or category == "system" or ifc_class in {"ifcsystem", "ifcgroup"}:
        return "system"
    if "space" in level or category == "space" or ifc_class in {"ifcspace", "ifcbuildingstorey"}:
        return "space"
    if category in {"object", "entity", "component"} or ifc_class:
        return "object"
    return "unknown"


@dataclass(frozen=True, slots=True)
class PlanningTerm:
    alias: str
    canonical: str
    kind: str
    source_field: str
    node_id: str = ""
    ifc_class: str | None = None
    graph_level: str | None = None
    role: str | None = None
    domain: str | None = None
    space_type: str | None = None
    function_type: str | None = None
    function_kind: str | None = None
    system_category: str | None = None
    compatible_roles: tuple[str, ...] = ()
    compatible_names: tuple[str, ...] = ()
    compatible_domains: tuple[str, ...] = ()
    compatible_space_types: tuple[str, ...] = ()
    related_system_categories: tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(slots=True)
class PlanningSchema:
    """Backend-independent vocabulary projected from a building graph.

    The schema may be cached and reused for every question in a run.  It
    intentionally stores no answer labels or evaluation metadata.
    """

    terms: list[PlanningTerm] = field(default_factory=list)
    identifier_node_ids: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any] | EntityRef],
    ) -> PlanningSchema:
        terms: list[PlanningTerm] = []
        identifier_node_ids: dict[str, list[str]] = {}
        seen: set[tuple[str, str, str, str, str]] = set()
        for raw_record in records:
            record = _record_dict(raw_record)
            kind = _record_kind(record)
            node_id = str(record.get("node_id") or record.get("id") or "")
            for identifier in (
                record.get("global_id"),
                record.get("ifc_guid"),
                node_id.removeprefix("ifc_") if node_id.startswith("ifc_") else "",
            ):
                normalized_identifier = str(identifier or "").strip().casefold()
                if normalized_identifier and node_id:
                    bucket = identifier_node_ids.setdefault(normalized_identifier, [])
                    if node_id not in bucket:
                        bucket.append(node_id)
            ifc_class = str(record.get("ifc_class") or "") or None
            graph_level = str(record.get("level") or "") or None

            def semantic_value(value: Any) -> str | None:
                rendered = str(value or "").strip()
                if normalize_question_text(rendered) in {
                    "", "unknown", "none", "null", "n a", "na", "unspecified",
                }:
                    return None
                return rendered

            role = semantic_value(record.get("asset_role") or record.get("role"))
            domain = semantic_value(record.get("object_domain") or record.get("domain"))
            space_type = semantic_value(record.get("space_type"))
            function_type = semantic_value(record.get("function_type"))
            function_kind = semantic_value(record.get("function_kind"))
            system_category = semantic_value(record.get("system_category"))
            compatible_roles = tuple(_ordered_unique(_as_list(record.get("target_roles"))))
            compatible_names = tuple(
                _ordered_unique(_as_list(record.get("target_name_terms")))
            )
            compatible_domains = tuple(
                _ordered_unique(_as_list(record.get("target_domains")))
            )
            compatible_space_types = tuple(
                _ordered_unique(_as_list(record.get("target_space_types")))
            )
            related_system_categories = tuple(
                _ordered_unique(_as_list(record.get("related_system_categories")))
            )
            try:
                confidence = float(
                    record.get("classification_confidence")
                    or record.get("confidence")
                    or 1.0
                )
            except (TypeError, ValueError):
                confidence = 1.0

            field_values: list[tuple[str, str]] = []
            for source_field in (
                "label",
                "name",
                "long_name",
                "family",
                "type_name",
                "type",
                "object_type",
                "tag",
            ):
                field_values.extend(
                    (source_field, value) for value in _as_list(record.get(source_field))
                )
            for source_field in ("aliases", "synonyms", "target_name_terms"):
                field_values.extend(
                    (source_field, value) for value in _as_list(record.get(source_field))
                )
            for source_field, value in (
                ("role", role),
                ("domain", domain),
                ("space_type", space_type),
                ("function_type", function_type),
                ("system_category", system_category),
            ):
                if value:
                    field_values.append((source_field, value.replace("_", " ")))

            canonical = (
                function_type
                or system_category
                or role
                or space_type
                or str(record.get("name") or record.get("label") or "")
            )
            for source_field, alias in field_values:
                normalized = normalize_question_text(alias)
                if not normalized:
                    continue
                # Single-character labels and raw GUID-like strings are not
                # useful language mentions and create many accidental hits.
                if len(normalized) < 2 or re.fullmatch(r"[0-9a-z_$]{18,}", normalized):
                    continue
                key = (normalized, canonical, kind, source_field, node_id)
                if key in seen:
                    continue
                seen.add(key)
                terms.append(
                    PlanningTerm(
                        alias=alias,
                        canonical=canonical or alias,
                        kind=kind,
                        source_field=source_field,
                        node_id=node_id,
                        ifc_class=ifc_class,
                        graph_level=graph_level,
                        role=role,
                        domain=domain,
                        space_type=space_type,
                        function_type=function_type,
                        function_kind=function_kind,
                        system_category=system_category,
                        compatible_roles=compatible_roles,
                        compatible_names=compatible_names,
                        compatible_domains=compatible_domains,
                        compatible_space_types=compatible_space_types,
                        related_system_categories=related_system_categories,
                        confidence=max(0.0, min(1.0, confidence)),
                    )
                )
        terms.sort(
            key=lambda term: (
                -len(_signature(term.alias)),
                -len(normalize_question_text(term.alias)),
                normalize_question_text(term.alias),
                term.node_id,
            )
        )
        return cls(terms, identifier_node_ids)

    def nodes_for_identifier(self, identifier: str) -> list[str]:
        """Resolve only exact graph-provided IFC identifiers."""

        return list(self.identifier_node_ids.get(str(identifier).strip().casefold(), []))

    def link(
        self,
        text: str,
        *,
        target_spans: Sequence[tuple[int, int, int]] = (),
        focus_spans: Sequence[tuple[int, int]] = (),
    ) -> list[MentionLink]:
        normalized = normalize_question_text(text)
        question_tokens = normalized.split()
        question_signature = [_singular_token(token) for token in question_tokens]
        token_offsets: list[int] = []
        offset = 0
        for token in question_tokens:
            position = normalized.find(token, offset)
            token_offsets.append(max(0, position))
            offset = max(0, position) + len(token)

        matched: list[tuple[int, int, PlanningTerm]] = []
        for term in self.terms:
            signature = _signature(term.alias)
            if not signature or len(signature) > len(question_signature):
                continue
            for index in range(len(question_signature) - len(signature) + 1):
                if tuple(question_signature[index : index + len(signature)]) != signature:
                    continue
                start = token_offsets[index]
                end_index = index + len(signature) - 1
                end = token_offsets[end_index] + len(question_tokens[end_index])
                matched.append((start, end, term))

        # Prefer longer aliases at the same location, while allowing multiple
        # graph nodes for an identical alias to remain as linking candidates.
        longest: dict[tuple[int, str], int] = {}
        for start, end, term in matched:
            key = (start, term.kind)
            longest[key] = max(longest.get(key, 0), end - start)
        matched = [
            item
            for item in matched
            if item[1] - item[0] == longest[(item[0], item[2].kind)]
        ]

        grouped: dict[tuple[Any, ...], MentionLink] = {}
        candidate_ids: dict[tuple[Any, ...], set[str]] = {}
        for start, end, term in matched:
            action_index = next(
                (
                    index
                    for span_start, span_end, index in target_spans
                    if span_start <= start < span_end
                ),
                None,
            )
            key = (
                start,
                end,
                term.canonical,
                term.kind,
                term.role or "",
                term.domain or "",
                term.function_type or "",
                term.system_category or "",
                term.ifc_class or "",
                term.graph_level or "",
                term.source_field,
            )
            link = grouped.get(key)
            if link is None:
                link = MentionLink(
                    text=term.alias,
                    canonical=term.canonical,
                    kind=term.kind,  # type: ignore[arg-type]
                    char_start=start,
                    char_end=end,
                    action_index=action_index,
                    query_focus=any(
                        span_start <= start < span_end
                        for span_start, span_end in focus_spans
                    ),
                    ifc_class=term.ifc_class,
                    graph_level=term.graph_level,
                    role=term.role,
                    domain=term.domain,
                    space_type=term.space_type,
                    function_type=term.function_type,
                    function_kind=term.function_kind,
                    system_category=term.system_category,
                    compatible_roles=list(term.compatible_roles),
                    compatible_names=list(term.compatible_names),
                    compatible_domains=list(term.compatible_domains),
                    compatible_space_types=list(term.compatible_space_types),
                    related_system_categories=list(term.related_system_categories),
                    confidence=term.confidence,
                    source="graph" if term.node_id else "schema",
                    source_field=term.source_field,
                )
                grouped[key] = link
                candidate_ids[key] = set()
            if term.node_id:
                candidate_ids[key].add(term.node_id)
            link.compatible_roles = _ordered_unique(
                [*link.compatible_roles, *term.compatible_roles]
            )
            link.compatible_names = _ordered_unique(
                [*link.compatible_names, *term.compatible_names]
            )
            link.compatible_domains = _ordered_unique(
                [*link.compatible_domains, *term.compatible_domains]
            )
            link.compatible_space_types = _ordered_unique(
                [*link.compatible_space_types, *term.compatible_space_types]
            )
            link.related_system_categories = _ordered_unique(
                [*link.related_system_categories, *term.related_system_categories]
            )
            link.confidence = max(link.confidence, term.confidence)
        for key, link in grouped.items():
            all_ids = sorted(candidate_ids[key])
            link.candidate_count = len(all_ids)
            link.node_ids = all_ids[:5]
        return sorted(
            grouped.values(),
            key=lambda link: (
                link.action_index is None,
                link.action_index if link.action_index is not None else 10_000,
                normalize_question_text(link.text),
                link.canonical,
            ),
        )

    def partial_link(
        self,
        phrase: str,
        *,
        preferred_kinds: Sequence[str] = (),
        action_index: int | None = None,
        max_candidates: int = 5,
    ) -> list[MentionLink]:
        """Conservatively link a shorter mention to longer graph vocabulary.

        Exact linking remains authoritative.  This fallback only accepts graph
        terms grounding every significant mention token in the alias, or in a
        conservative alias-plus-canonical/role combination, then ranks by
        specificity and caps the candidate identities exposed downstream.
        """

        query_tokens = tuple(
            token for token in _signature(phrase)
            if token not in _LINK_STOPWORDS and (len(token) > 1 or token.isdigit())
        )
        if not query_tokens or (len(query_tokens) == 1 and len(query_tokens[0]) < 4):
            return []
        allowed = set(preferred_kinds)
        best_by_node: dict[str, tuple[tuple[Any, ...], PlanningTerm]] = {}
        for term in self.terms:
            if allowed and term.kind not in allowed:
                continue
            if term.source_field not in {
                "label", "name", "long_name", "family", "type_name", "type",
                "object_type", "aliases", "synonyms", "target_name_terms",
            }:
                continue
            term_tokens = tuple(
                token for token in _signature(term.alias)
                if token not in _LINK_STOPWORDS
            )
            if not term_tokens:
                continue
            alias_tokens = set(term_tokens)
            alias_match = all(token in alias_tokens for token in query_tokens)

            # Some instance labels carry the discriminating modifier while a
            # graph role supplies only the generic asset head.  For example,
            # an alias containing "basin" with role ``sanitary_fixture`` may
            # satisfy "basin fixtures" even though neither field does alone.
            # Keep this fallback conservative: it applies only to multiword
            # mentions, every query token must be grounded across the alias and
            # canonical/role vocabulary, and at least one non-generic query
            # modifier must occur in the instance alias itself.
            semantic_tokens = {
                token
                for value in (term.canonical, term.role or "")
                for token in _signature(value)
                if token not in _LINK_STOPWORDS
            }
            combined_tokens = alias_tokens | semantic_tokens
            cross_field_match = (
                len(query_tokens) > 1
                and all(token in combined_tokens for token in query_tokens)
                and any(
                    token not in _GENERIC_NOUNS and token in alias_tokens
                    for token in query_tokens
                )
            )
            if not (alias_match or cross_field_match):
                continue
            node_key = term.node_id or (
                f"schema:{term.kind}:{normalize_question_text(term.canonical)}"
            )
            rank = (
                0 if alias_match else 1,
                len(combined_tokens) - len(query_tokens),
                -term.confidence,
                normalize_question_text(term.alias),
                node_key,
            )
            previous = best_by_node.get(node_key)
            if previous is None or rank < previous[0]:
                best_by_node[node_key] = (rank, term)

        ranked = sorted(best_by_node.values(), key=lambda item: item[0])
        total = len(ranked)
        links: list[MentionLink] = []
        for _, term in ranked[: max(0, max_candidates)]:
            links.append(
                MentionLink(
                    # Keep the user's mention as the executable constraint;
                    # the graph term remains in canonical/source metadata.
                    # This lets one short mention close over size/suffix variants.
                    text=phrase,
                    canonical=term.canonical,
                    kind=term.kind,  # type: ignore[arg-type]
                    action_index=action_index,
                    node_ids=[term.node_id] if term.node_id else [],
                    candidate_count=total,
                    ifc_class=term.ifc_class,
                    graph_level=term.graph_level,
                    role=term.role,
                    domain=term.domain,
                    space_type=term.space_type,
                    function_type=term.function_type,
                    function_kind=term.function_kind,
                    system_category=term.system_category,
                    compatible_roles=list(term.compatible_roles),
                    compatible_names=list(term.compatible_names),
                    compatible_domains=list(term.compatible_domains),
                    compatible_space_types=list(term.compatible_space_types),
                    related_system_categories=list(term.related_system_categories),
                    confidence=term.confidence,
                    source="graph" if term.node_id else "schema",
                    source_field=f"partial:{term.source_field}",
                )
            )
        return links

    def link_phrase(
        self,
        phrase: str,
        *,
        preferred_kinds: Sequence[str] = (),
        action_index: int | None = None,
    ) -> list[MentionLink]:
        """Return exact typed links when available, otherwise bounded partial links."""

        allowed = set(preferred_kinds)
        exact = [
            replace(link, action_index=action_index)
            for link in self.link(phrase)
            if not allowed or link.kind in allowed
        ]
        exact_entities = [
            link for link in exact
            if link.kind in {"object", "system", "space"}
            and link.source_field in {
                "label", "name", "long_name", "family", "type_name", "type",
                "object_type", "aliases", "synonyms",
            }
        ]
        exact_semantics = [
            link for link in exact
            if link.source_field in {
                "role", "domain", "space_type", "system_category", "function_type",
            }
        ]
        exact_functions = [link for link in exact if link.kind == "function"]
        selected_sources = (
            [*exact_entities, *exact_functions]
            if exact_entities
            else [*exact_semantics, *exact_functions]
        )
        if selected_sources:
            selected: list[MentionLink] = []
            seen: set[tuple[Any, ...]] = set()
            for link in selected_sources:
                key = (link.kind, link.canonical, link.source_field, tuple(link.node_ids))
                if key not in seen:
                    selected.append(link)
                    seen.add(key)
            return selected
        partial = self.partial_link(
            phrase,
            preferred_kinds=preferred_kinds,
            action_index=action_index,
        )
        return partial or exact

    def suggest(
        self,
        phrase: str,
        *,
        action_index: int | None = None,
        max_candidates: int = 5,
    ) -> list[MentionLink]:
        """Attach low-confidence graph candidates to a genuinely unknown mention.

        Suggestions are deliberately typed as ``unknown`` so they can inform a
        bounded planner without becoming deterministic answer candidates.
        """

        query_tokens = {
            token for token in _signature(phrase)
            if token not in _LINK_STOPWORDS and len(token) >= 3
        }
        if not query_tokens:
            return []
        best_by_node: dict[str, tuple[tuple[Any, ...], PlanningTerm]] = {}
        source_rank = {
            "name": 0, "label": 0, "long_name": 0, "aliases": 1,
            "family": 1, "type_name": 1, "type": 1, "object_type": 1,
            "role": 2, "domain": 3, "system_category": 3,
        }
        for term in self.terms:
            if term.kind not in {"object", "system", "function"} or not term.node_id:
                continue
            term_tokens = set(_signature(term.alias)) - _LINK_STOPWORDS
            overlap = query_tokens.intersection(term_tokens)
            if not overlap:
                continue
            rank = (
                -len(overlap) / len(query_tokens),
                source_rank.get(term.source_field, 4),
                len(term_tokens - query_tokens),
                -term.confidence,
                term.node_id,
            )
            previous = best_by_node.get(term.node_id)
            if previous is None or rank < previous[0]:
                best_by_node[term.node_id] = (rank, term)
        ranked = sorted(best_by_node.values(), key=lambda item: item[0])
        total = len(ranked)
        return [
            MentionLink(
                text=phrase,
                canonical=term.canonical,
                kind="unknown",
                action_index=action_index,
                node_ids=[term.node_id],
                candidate_count=total,
                ifc_class=term.ifc_class,
                graph_level=term.graph_level,
                role=term.role,
                domain=term.domain,
                function_type=term.function_type,
                function_kind=term.function_kind,
                system_category=term.system_category,
                compatible_roles=list(term.compatible_roles),
                compatible_names=list(term.compatible_names),
                compatible_domains=list(term.compatible_domains),
                compatible_space_types=list(term.compatible_space_types),
                related_system_categories=list(term.related_system_categories),
                confidence=min(0.25, term.confidence),
                source="graph",
                source_field=f"suggestion:{term.source_field}",
            )
            for _, term in ranked[: max(0, max_candidates)]
        ]


SchemaContext = PlanningSchema | Iterable[Mapping[str, Any] | EntityRef]


def coerce_planning_schema(schema_context: SchemaContext | None) -> PlanningSchema:
    if schema_context is None:
        return PlanningSchema()
    if isinstance(schema_context, PlanningSchema):
        return schema_context
    return PlanningSchema.from_records(schema_context)


def _is_space_use_link(link: MentionLink) -> bool:
    explicit = normalize_question_text(link.function_kind or "")
    if explicit:
        return explicit == "space use"
    # Backward-compatible structural inference for older ontology artifacts.
    return bool(
        link.kind == "function"
        and link.compatible_space_types
        and not link.compatible_roles
        and not link.compatible_domains
        and not link.related_system_categories
    )


def _is_specific_name_source(source_field: str) -> bool:
    return source_field in {
        "label", "name", "long_name", "family", "type_name", "type", "object_type",
    } or source_field.startswith("partial:")


def _space_label_duplicates_type(
    label: str,
    space_types: Sequence[str],
) -> bool:
    label_tokens = _signature(label)
    type_signatures = {_signature(value) for value in space_types if _signature(value)}
    if not label_tokens or not type_signatures:
        return False
    without_generic_head = (
        label_tokens[:-1]
        if len(label_tokens) > 1 and label_tokens[-1] in _SPACE_HEADS
        else label_tokens
    )
    return label_tokens in type_signatures or without_generic_head in type_signatures


def _longest_non_overlapping_space_links(
    links: Sequence[MentionLink],
) -> list[MentionLink]:
    """Keep the most specific exact space label at overlapping text spans."""

    ranked = sorted(
        links,
        key=lambda link: (
            -(max(0, link.char_end - link.char_start)),
            link.char_start if link.char_start >= 0 else 10_000_000,
            normalize_question_text(link.text),
        ),
    )
    selected: list[MentionLink] = []
    occupied: list[tuple[int, int]] = []
    for link in ranked:
        span = (link.char_start, link.char_end)
        if span[0] >= 0 and span[1] > span[0] and any(
            max(span[0], start) < min(span[1], end) for start, end in occupied
        ):
            continue
        selected.append(link)
        if span[0] >= 0 and span[1] > span[0]:
            occupied.append(span)
    return sorted(
        selected,
        key=lambda link: (
            link.char_start if link.char_start >= 0 else 10_000_000,
            normalize_question_text(link.text),
        ),
    )


_EXACT_IDENTITY_FIELDS = frozenset({
    "label", "name", "long_name", "aliases", "synonyms",
})


def _suppress_contained_cross_kind_links(
    links: Sequence[MentionLink],
) -> list[MentionLink]:
    """Let a longer exact graph identity own its complete text span.

    Building labels routinely contain shorter vocabulary that is also a valid
    object, system, or function term.  Once the active graph links the whole
    span to an identity, a shorter cross-kind match inside that same span is
    incidental rather than an independent mention.  Non-overlapping mentions
    and same-kind alternatives remain available.
    """

    exact = [
        link
        for link in links
        if link.source == "graph"
        and link.source_field in _EXACT_IDENTITY_FIELDS
        and link.char_start >= 0
        and link.char_end > link.char_start
        and link.node_ids
    ]
    result: list[MentionLink] = []
    for link in links:
        suppressed = False
        for owner in exact:
            if owner is link or owner.kind == link.kind:
                continue
            if (
                owner.action_index is not None
                and link.action_index is not None
                and owner.action_index != link.action_index
            ):
                continue
            owner_length = owner.char_end - owner.char_start
            link_length = link.char_end - link.char_start
            contained_span = (
                link.char_start >= 0
                and link.char_end > link.char_start
                and owner_length > link_length
                and owner.char_start <= link.char_start
                and link.char_end <= owner.char_end
            )
            partial_same_surface = (
                link.source_field.startswith("partial:")
                and normalize_question_text(link.text)
                == normalize_question_text(owner.text)
            )
            if contained_span or partial_same_surface:
                suppressed = True
                break
        if not suppressed:
            result.append(link)
    return result


def _suppress_contained_function_links(
    links: Sequence[MentionLink],
) -> list[MentionLink]:
    """Prefer the longest exact function phrase at an overlapping span."""

    exact = [
        link
        for link in links
        if link.kind == "function"
        and link.char_start >= 0
        and link.char_end > link.char_start
        and link.source_field in {
            "label", "name", "long_name", "aliases", "synonyms",
            "function_type",
        }
    ]
    result: list[MentionLink] = []
    for link in links:
        contained = bool(
            link.kind == "function"
            and link.char_start >= 0
            and any(
                owner is not link
                and owner.action_index == link.action_index
                and owner.char_start <= link.char_start
                and link.char_end <= owner.char_end
                and (owner.char_end - owner.char_start)
                > (link.char_end - link.char_start)
                for owner in exact
            )
        )
        if not contained:
            result.append(link)
    return result


@dataclass(frozen=True, slots=True)
class _ActionOccurrence:
    action: str
    start: int
    end: int


def _is_relative_clause_scope_mention(
    text: str,
    start: int,
) -> bool:
    """Return whether a mention belongs to a descriptive relative clause.

    Words such as ``scan`` and ``inspect`` can name a capability or required
    method (``a room which requires a scan``) rather than command the robot.
    A local relative marker plus a descriptive predicate is a conservative
    syntax signal for that reading.  An explicit ``then`` marker after the
    relative marker starts a new executable sequence and overrides it.
    """

    clause_start = max(
        text.rfind(",", 0, start),
        text.rfind(".", 0, start),
        text.rfind(";", 0, start),
        text.rfind("?", 0, start),
        text.rfind("!", 0, start),
    ) + 1
    prefix = text[clause_start:start]
    relative_matches = list(re.finditer(r"\b(?:that|which|where|whose)\b", prefix))
    if not relative_matches:
        return False
    relative_end = relative_matches[-1].end()
    relative_tail = prefix[relative_end:]
    if re.search(r"\b(?:and\s+)?then\b", relative_tail):
        return False
    return bool(
        re.search(
            r"\b(?:requires?|requiring|needs?|needing|performs?|performing|"
            r"supports?|supporting|uses?|using|allows?|allowing|enables?|"
            r"enabling|receives?|receiving|undergoes?|undergoing|involves?|"
            r"involving|calls\s+for)\b[^,.;!?]*$",
            relative_tail,
        )
    )


def _actions(text: str) -> list[_ActionOccurrence]:
    matches: list[_ActionOccurrence] = []
    for pattern, action in _ACTION_PATTERNS:
        for match in re.finditer(pattern, text):
            if _is_relative_clause_scope_mention(text, match.start()):
                continue
            matches.append(_ActionOccurrence(action, match.start(), match.end()))
    if not matches:
        scenario_prefix = _SCENARIO_PREFIX.search(text)
        if (
            scenario_prefix is not None
            and _SCENARIO_STATE_CHANGE.search(text)
            and _SCENARIO_IMPACT.search(text)
        ):
            # The state-changing equipment is a reference; the requested
            # affected entities after the final "which/what" are targets.
            target_questions = list(
                re.finditer(r"\b(?:which|what)\b", text[scenario_prefix.end():])
            )
            anchor = (
                scenario_prefix.end() + target_questions[-1].end()
                if target_questions
                else scenario_prefix.end()
            )
            matches.append(
                _ActionOccurrence(
                    "Inspect",
                    anchor,
                    anchor,
                )
            )
    matches.sort(key=lambda item: (item.start, item.end))
    explicit = list(matches)
    for index, occurrence in enumerate(explicit):
        segment_end = explicit[index + 1].start if index + 1 < len(explicit) else len(text)
        segment = text[occurrence.end:segment_end]
        for continuation in re.finditer(r"(?:,\s*)?(?:and\s+)?then\s+", segment):
            following = segment[continuation.end():].strip()
            if not following or any(
                re.match(pattern, following) for pattern, _ in _ACTION_PATTERNS
            ):
                continue
            marker_start = occurrence.end + continuation.start()
            marker_end = occurrence.end + continuation.end()
            matches.append(
                _ActionOccurrence(occurrence.action, marker_start, marker_end)
            )
    matches.sort(key=lambda item: (item.start, item.end))
    return matches


def _action_target_spans(
    text: str,
    actions: Sequence[_ActionOccurrence],
) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for index, occurrence in enumerate(actions):
        start = occurrence.end
        end = actions[index + 1].start if index + 1 < len(actions) else len(text)
        clause = text[start:end]
        nominal_object = re.match(r"\s+of\s+", clause)
        if nominal_object:
            start += nominal_object.end()
            clause = text[start:end]
        boundary = re.search(
            r"\b(?:in|inside|within|on|at|from|near|nearby|nearest|closest|"
            r"adjacent(?:\s+to)?|served(?:\s+by)?|assigned(?:\s+to)?|"
            r"associated\s+with|related\s+(?:to|with)|of|for|that|which|where|whose|then)\b",
            clause,
        )
        if boundary:
            end = start + boundary.start()
        if end > start:
            result.append((start, end, index))
    return result


def _target_phrases(
    text: str,
    target_spans: Sequence[tuple[int, int, int]],
) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for start, end, action_index in target_spans:
        phrase = _DETERMINERS.sub("", text[start:end].strip(" ,.;:"))
        phrase = re.sub(r"\b(?:and\s+)?then\b.*$", "", phrase).strip()
        if not phrase or phrase in {"it", "them", "there"}:
            continue
        pieces = re.split(r"\s*(?:,|\band\b)\s*", phrase)
        for piece in pieces:
            value = _DETERMINERS.sub("", piece.strip(" ,.;:"))
            if value and value not in {"it", "them", "there"}:
                result.append((action_index, value))
    return result


def _complete_shared_target_heads(
    phrases: Sequence[tuple[int, str]],
    links: Sequence[MentionLink],
) -> list[tuple[int, str]]:
    """Complete elliptical conjuncts from the final coordinated noun head.

    In ``red and blue valves`` the first conjunct is syntactically ``red
    valves``.  The completion is intentionally graph-aware: an earlier phrase
    that already links to an object/system vocabulary item is left untouched.
    This prevents a generic
    shared-head rule from turning an independently meaningful noun into a
    different asset class.
    """

    by_action: dict[int, list[tuple[int, str]]] = {}
    for position, (action_index, phrase) in enumerate(phrases):
        by_action.setdefault(action_index, []).append((position, phrase))
    result = list(phrases)
    for action_index, entries in by_action.items():
        if len(entries) < 2:
            continue
        final_tokens = normalize_question_text(entries[-1][1]).split()
        if len(final_tokens) < 2:
            continue
        shared_head = final_tokens[-1]
        for position, phrase in entries[:-1]:
            phrase_tokens = normalize_question_text(phrase).split()
            if not phrase_tokens or shared_head in phrase_tokens:
                continue
            independently_linked = any(
                link.action_index == action_index
                and link.kind in {"object", "system"}
                and _signature(link.text) == _signature(phrase)
                for link in links
            )
            if independently_linked:
                continue
            result[position] = (action_index, f"{phrase} {shared_head}")
    return result


def _phrase_cardinality(phrase: str) -> CardinalityPolicy:
    """Return the universal surface cardinality of one coordinated phrase."""

    tokens = normalize_question_text(phrase).split()
    if not tokens:
        return "single"
    return "all" if _singular_token(tokens[-1]) != tokens[-1] else "single"


def _pronoun_action_links(
    text: str,
    actions: Sequence[_ActionOccurrence],
    links: Sequence[MentionLink],
) -> list[MentionLink]:
    """Bind an action pronoun to the nearest preceding explicit object mention."""

    result: list[MentionLink] = []
    for index, occurrence in enumerate(actions):
        end = actions[index + 1].start if index + 1 < len(actions) else len(text)
        target = text[occurrence.end:end].strip(" ,.;:")
        if not re.match(r"^(?:the\s+)?(?:it|them|this|these|that|those)\b", target):
            continue
        antecedents = [
            link for link in links
            if link.kind == "object"
            and link.action_index is None
            and 0 <= link.char_end <= occurrence.start
        ]
        if antecedents:
            result.append(replace(max(antecedents, key=lambda link: link.char_end), action_index=index))
    return result


def _assign_action_local_support_links(
    text: str,
    actions: Sequence[_ActionOccurrence],
    links: Sequence[MentionLink],
) -> list[MentionLink]:
    """Attach function/system mentions to the action clause containing them.

    Such support phrases commonly follow ``for``, ``of`` or ``served by`` and
    therefore sit outside the direct-object span.  Clause position provides a
    generic, ordered binding without leaking one function/system across every
    action in a compound instruction.
    """

    if not actions:
        return list(links)
    result: list[MentionLink] = []
    for link in links:
        if (
            link.kind == "function"
            and link.char_start >= 0
            and _is_relative_clause_scope_mention(text, link.char_start)
        ):
            result.append(replace(link, action_index=None))
            continue
        if (
            link.action_index is not None
            or link.kind not in {"function", "system"}
            or link.char_start < 0
        ):
            result.append(link)
            continue
        action_index = next(
            (
                index
                for index, action in enumerate(actions)
                if action.end <= link.char_start
                < (actions[index + 1].start if index + 1 < len(actions) else 10**12)
            ),
            None,
        )
        result.append(
            replace(link, action_index=action_index)
            if action_index is not None
            else link
        )
    return result


def _function_target_role_links(
    actions: Sequence[_ActionOccurrence],
    phrases: Sequence[tuple[int, str]],
    links: Sequence[MentionLink],
) -> list[MentionLink]:
    """Narrow multi-role function compatibility using the target phrase.

    Compatible roles come only from the active function ontology.  The
    explicit target's normalized head/modifiers select among those roles; no
    building asset vocabulary is embedded here.  If no role has lexical
    support, this function returns nothing and the existing fallback remains
    responsible for ambiguity handling.
    """

    result: list[MentionLink] = []
    for action_index, phrase in phrases:
        direct_target = any(
            link.action_index == action_index
            and link.kind in {"object", "system"}
            and (
                link.source_field != "domain"
                and not link.source_field.startswith("suggestion:")
            )
            and bool(
                link.role
                or link.system_category
                or _is_specific_name_source(link.source_field)
            )
            for link in links
        )
        if direct_target:
            continue
        function_links = [
            link
            for link in links
            if link.kind == "function"
            and not _is_space_use_link(link)
            and (
                link.action_index == action_index
                or (len(actions) == 1 and link.action_index is None)
            )
        ]
        role_owners: dict[str, MentionLink] = {}
        for link in function_links:
            for role in link.compatible_roles:
                role_owners.setdefault(normalize_question_text(role), link)
        phrase_tokens = [
            token for token in _signature(phrase) if token not in _LINK_STOPWORDS
        ]
        if len(role_owners) <= 1 or not phrase_tokens:
            continue
        head = _singular_token(phrase_tokens[-1])
        modifiers = set(phrase_tokens[:-1])
        scored: list[tuple[int, str, MentionLink]] = []
        for normalized_role, owner in role_owners.items():
            role_tokens = set(_signature(normalized_role))
            if head in _GENERIC_NOUNS:
                if head not in role_tokens:
                    continue
                score = len(modifiers.intersection(role_tokens))
            else:
                overlap = set(phrase_tokens).intersection(role_tokens)
                if not overlap:
                    continue
                score = len(overlap)
            scored.append((score, normalized_role, owner))
        if not scored:
            continue
        best_score = max(item[0] for item in scored)
        for score, normalized_role, owner in scored:
            if score != best_score:
                continue
            role = next(
                value
                for value in owner.compatible_roles
                if normalize_question_text(value) == normalized_role
            )
            result.append(
                MentionLink(
                    text=phrase,
                    canonical=role,
                    kind="object",
                    action_index=action_index,
                    role=role,
                    confidence=owner.confidence,
                    source="schema",
                    source_field="derived:function_target_lexical",
                )
            )
    return result


def _is_explicit_space_target(phrase: str) -> bool:
    normalized = normalize_question_text(phrase)
    tokens = normalized.split()
    if not tokens:
        return False
    # The action's target head is authoritative.  A space modifier inside an
    # equipment phrase (for example, "laboratory manifold") must not turn an
    # object inspection into navigation.
    if _singular_token(tokens[-1]) in _SPACE_HEADS:
        return True
    return (
        len(tokens) > 1
        and _singular_token(tokens[-2]) in _SPACE_HEADS
        and bool(re.fullmatch(r"(?:[a-z]|[a-z]*\d+[a-z]*)", tokens[-1]))
    )


def _unresolved_generic_compound_actions(
    phrases: Sequence[tuple[int, str]],
    links: Sequence[MentionLink],
) -> set[int]:
    """Find compound asset mentions whose modifier has no schema grounding.

    A graph can legitimately contain a broad authored role whose instances
    happen to belong to one domain.  Matching only the generic head must not
    copy an exemplar domain into the query and discard semantically different
    assets.  A modifier remains grounded when a separate graph link covers it.
    The check is entirely schema- and syntax-driven.
    """

    unresolved: set[int] = set()
    for action_index, phrase in phrases:
        tokens = [
            token
            for token in _signature(phrase)
            if token not in _LINK_STOPWORDS
        ]
        if len(tokens) < 2 or tokens[-1] not in _GENERIC_NOUNS:
            continue
        modifiers = set(tokens[:-1])
        modifier_grounded = False
        for link in links:
            if link.action_index != action_index:
                continue
            link_tokens = {
                token
                for token in _signature(link.text)
                if token not in _LINK_STOPWORDS
            }
            if modifiers.intersection(link_tokens):
                modifier_grounded = True
                break
        if not modifier_grounded:
            unresolved.add(action_index)
    return unresolved


def _schema_semantic_options(
    schema: PlanningSchema,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return bounded, building-derived vocabularies for semantic fallback."""

    roles = _ordered_unique(
        value
        for term in schema.terms
        for value in ([term.role] if term.role else []) + list(term.compatible_roles)
    )
    domains = _ordered_unique(
        value
        for term in schema.terms
        for value in ([term.domain] if term.domain else []) + list(term.compatible_domains)
    )
    systems = _ordered_unique(
        value
        for term in schema.terms
        for value in (
            ([term.system_category] if term.system_category else [])
            + list(term.related_system_categories)
        )
    )
    # These are schema labels rather than graph instances.  Keeping the
    # vocabularies bounded also keeps the one-shot planner payload compact.
    return tuple(roles[:64]), tuple(domains[:24]), tuple(systems[:24])


def _normalize_space_actions(
    actions: Sequence[_ActionOccurrence],
    links: Sequence[MentionLink],
    phrases: Sequence[tuple[int, str]],
) -> list[_ActionOccurrence]:
    """Map room-level inspection language to executable navigation."""

    result: list[_ActionOccurrence] = []
    for index, occurrence in enumerate(actions):
        if occurrence.action != "Inspect":
            result.append(occurrence)
            continue
        raw_space = any(
            action_index == index
            and _is_explicit_space_target(phrase)
            for action_index, phrase in phrases
        )
        exact_named_space = any(
            link.action_index == index
            and link.kind == "space"
            and link.source_field in {"label", "name", "long_name", "aliases", "synonyms"}
            and any(
                action_index == index
                and normalize_question_text(link.text) == normalize_question_text(phrase)
                for action_index, phrase in phrases
            )
            for link in links
        )
        space_only = raw_space or exact_named_space
        result.append(
            replace(occurrence, action="Navigate") if space_only else occurrence
        )
    return result


def _storey(text: str) -> str | None:
    match = re.search(
        r"\b(?:level|floor)\s*[-#:]?\s*"
        r"((?:[a-z]*\d+[a-z]*|basement|ground|roof|mezzanine)"
        r"(?:\s*\+\s*\d+'?)?)\b",
        text,
    )
    return f"LEVEL {match.group(1).upper()}" if match else None


def _room_number(text: str) -> str | None:
    for pattern in (
        r"\b(?:room|space)\s*#?\s*([0-9]{2,4}[a-z]?)\b",
        r"\(([0-9]{2,4}[a-z]?)\)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _named_space_phrases(text: str) -> list[str]:
    """Extract exact named scopes using syntax, not a building name list."""

    result: list[str] = []
    pattern = re.compile(
        r"\b(?:in|inside|within|to|into|at|from|of|by|with)\s+(?:the\s+)?"
        r"([a-z0-9][a-z0-9&/'\-.]*(?:\s+[a-z0-9][a-z0-9&/'\-.]*){0,8})"
        r"(?=\s+(?:on|at|in|that|which|with|containing|nearest|closest|adjacent)\b|[().,;?]|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        phrase = " ".join(match.group(1).split()).strip(" \t\r\n.,;:?!()")
        phrase = re.split(
            r"\s+(?:that|which|where|whose|containing|having)\b",
            phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        phrase = re.split(
            r"\s+(?:used|located|found|installed|present)\s+"
            r"(?:in|inside|within|on|at)\s+",
            phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        phrase = re.split(
            r"\s+(?:used|suitable|intended)\s+(?:for|as)\s+",
            phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        phrase = re.sub(
            r"^(?:level|floor)\s*[-#:]?\s*"
            r"(?:[a-z]*\d+[a-z]*|basement|ground|roof|mezzanine)\s+",
            "",
            phrase,
            flags=re.IGNORECASE,
        ).strip()
        phrase = re.sub(
            r"\s+(?:on|at|in)\s+(?:level|floor)\b.*$",
            "",
            phrase,
            flags=re.IGNORECASE,
        ).strip()
        phrase = _DETERMINERS.sub("", phrase).strip()
        tokens = normalize_question_text(phrase).split()
        if not tokens:
            continue
        if re.match(
            r"^(?:room|space)\s*#?\s*[0-9]{2,4}[a-z]?\b",
            phrase,
            flags=re.IGNORECASE,
        ):
            # The numeric scope is represented by the typed space-number
            # predicate.  Trailing diagnostic prose is not a room name.
            continue
        if tokens[0] in {"all", "both", "each", "every", "any", "same"}:
            continue
        if normalize_question_text(phrase) in {
            "room", "rooms", "space", "spaces", "same room", "same space",
            "room number", "space number",
        }:
            continue
        if any(
            token in {"what", "which", "how", "count", "list", "type", "types"}
            for token in tokens
        ):
            continue
        if tokens and (
            tokens[-1] in _SPACE_HEADS
            or any(token in _SPACE_HEADS for token in tokens)
            or bool(re.search(r"\b(?:room|space)\s+\w+$", phrase, re.IGNORECASE))
        ):
            result.append(phrase)
    return _ordered_unique(result)


def _is_typed_space_collection_phrase(value: str) -> bool:
    normalized = normalize_question_text(value)
    heads = "|".join(sorted(_SPACE_HEADS, key=len, reverse=True))
    return bool(
        re.search(rf"\b[\w-]+(?:-|\s+)type\s+(?:{heads})s?\b$", normalized)
    )


def _is_generic_typed_scope_phrase(
    value: str,
    space_types: Sequence[str],
) -> bool:
    """Keep descriptive or superlative scopes out of the room-name slot."""

    ignored = {
        "all", "both", "each", "every", "any", "same", "the",
        "largest", "smallest", "highest", "lowest", "maximum", "minimum",
    }
    reduced: list[str] = []
    skip_identifier = False
    for token in _signature(value):
        if token in ignored:
            continue
        if token in {"level", "floor"}:
            skip_identifier = True
            continue
        if skip_identifier and re.fullmatch(r"[a-z]*\d+[a-z]*", token):
            skip_identifier = False
            continue
        skip_identifier = False
        reduced.append(token)
    if not reduced:
        return False
    allowed_heads = {_singular_token(head) for head in _SPACE_HEADS}
    for space_type in space_types:
        type_tokens = set(_signature(space_type))
        if type_tokens and type_tokens.issubset(reduced) and all(
            token in type_tokens or token in allowed_heads for token in reduced
        ):
            return True
    return False


def _typed_space_modifier_links(
    text: str,
    schema: PlanningSchema,
) -> list[MentionLink]:
    """Resolve graph-backed ``<space-type>-type rooms`` modifiers."""

    heads = "|".join(sorted(_SPACE_HEADS, key=len, reverse=True))
    pattern = re.compile(
        rf"\b([a-z0-9][a-z0-9_-]*(?:\s+[a-z0-9][a-z0-9_-]*){{0,3}})"
        rf"(?:-|\s+)type\s+(?:{heads})s?\b",
        flags=re.IGNORECASE,
    )
    result: list[MentionLink] = []
    seen: set[tuple[Any, ...]] = set()
    for match in pattern.finditer(text):
        captured = match.group(1)
        captured_tokens = captured.split()
        phrase = captured
        candidates: list[MentionLink] = []
        # The regex intentionally accepts a short multiword type.  Resolve the
        # longest graph-backed suffix so preceding query verbs/determiners can
        # never become part of the semantic space type.
        for offset in range(len(captured_tokens)):
            candidate_phrase = " ".join(captured_tokens[offset:])
            variants = _ordered_unique(
                [candidate_phrase, candidate_phrase.replace("-", " ")]
            )
            exact: list[MentionLink] = []
            for variant in variants:
                exact.extend(
                    link
                    for link in schema.link_phrase(
                        variant, preferred_kinds=("space", "function")
                    )
                    if (
                        (link.kind == "space" and link.source_field == "space_type")
                        or (link.kind == "function" and _is_space_use_link(link))
                    )
                )
            if exact:
                phrase = candidate_phrase
                candidates = exact
                break
        for link in candidates:
            if not (
                (link.kind == "space" and link.source_field == "space_type")
                or (link.kind == "function" and _is_space_use_link(link))
            ):
                continue
            linked = replace(
                link,
                text=phrase,
                char_start=match.end(1) - len(phrase),
                char_end=match.end(1),
                action_index=None,
            )
            key = (
                linked.kind,
                linked.canonical,
                linked.source_field,
                tuple(linked.node_ids),
            )
            if key not in seen:
                result.append(linked)
                seen.add(key)
    return result


def _space_subject_contains_query(text: str) -> bool:
    heads = "|".join(sorted(_SPACE_HEADS, key=len, reverse=True))
    return bool(
        re.search(
            rf"\b(?:what|which|list|enumerate)\b[^?.;]*?"
            rf"\b(?:{heads})s?\b[^?.;]*?\b(?:that|which)\s+"
            r"(?:contains?|has|have|includes?)\b",
            text,
        )
    )


def _has_coordinated_space_scope(
    text: str,
    links: Sequence[MentionLink],
) -> bool:
    positioned = sorted(
        (
            link for link in links
            if link.char_start >= 0 and link.char_end > link.char_start
        ),
        key=lambda link: (link.char_start, link.char_end),
    )
    for left, right in zip(positioned, positioned[1:]):
        separator = text[left.char_end:right.char_start]
        if re.fullmatch(r"\s*(?:,\s*(?:and\s+)?|and\s+)", separator):
            return True
    return False


def _coordinated_space_links(
    text: str,
    schema: PlanningSchema,
) -> list[MentionLink]:
    """Expand compact named scopes such as ``Laboratory A/B/C`` or ``A, B and C``."""

    heads = "|".join(sorted(_SPACE_HEADS, key=len, reverse=True))
    pattern = re.compile(
        rf"\b((?:[A-Za-z0-9&'\-]+\s+){{0,5}}(?:{heads})s?)\s+"
        r"([A-Za-z0-9]+(?:(?:\s*/\s*|\s*,\s*(?:and\s+)?|\s+and\s+)"
        r"[A-Za-z0-9]+){1,5})\b",
        flags=re.IGNORECASE,
    )
    result: list[MentionLink] = []
    seen: set[tuple[str, str]] = set()
    for match in pattern.finditer(text):
        base = " ".join(match.group(1).split())
        base = re.sub(
            r"^(?:(?:in|inside|within|to|at|from|of|by|with|the|both|all)\s+)+",
            "",
            base,
            flags=re.IGNORECASE,
        )
        suffixes = [
            part.strip()
            for part in re.split(
                r"\s*/\s*|\s*,\s*(?:and\s+)?|\s+and\s+",
                match.group(2),
            )
            if part.strip()
        ]
        for suffix in suffixes:
            candidate = f"{base} {suffix}"
            for link in schema.link(candidate):
                if link.kind != "space" or link.source_field not in {
                    "label", "name", "long_name", "aliases", "synonyms",
                }:
                    continue
                key = (link.canonical, ",".join(link.node_ids))
                if key in seen:
                    continue
                seen.add(key)
                # These are graph-verified expansions of one compact surface
                # mention, not competing overlapping labels.
                result.append(replace(link, char_start=-1, char_end=-1))
    return result


def _reference_phrase(text: str, match_end: int) -> str:
    tail = text[match_end:]
    boundary = re.search(
        r"(?:[,.;?]|\b(?:on|at|in)\s+(?:level|floor)\b|"
        r"\b(?:that|which|where|and\s+then|then|for|with)\b)",
        tail,
    )
    phrase = tail[: boundary.start()] if boundary else tail
    return _DETERMINERS.sub("", phrase.strip(" ,.;:"))


def _relation_references(
    text: str,
    schema: PlanningSchema,
) -> tuple[list[RelationReference], list[tuple[int, int]]]:
    references: list[RelationReference] = []
    spans: list[tuple[int, int]] = []
    patterns: tuple[tuple[str, str], ...] = (
        (r"\b(?:nearest|closest)(?:\s+to)?\s+", "nearest"),
        (r"\bnear(?:by)?(?:\s+to)?\s+", "nearest"),
        (r"\badjacent\s+to\s+", "adjacent_to"),
        # Association to a space is represented as a scope-to-target
        # containment reference; it must not be merged into the action noun.
        (r"\b(?:associated|related)\s+with\s+", "contains"),
    )
    for pattern, relation in patterns:
        for match in re.finditer(pattern, text):
            phrase = _reference_phrase(text, match.end())
            if not phrase:
                continue
            preferred = ("space",) if relation == "contains" else ()
            linked = schema.link_phrase(phrase, preferred_kinds=preferred)
            exact_named_spaces = [
                link
                for link in linked
                if link.kind == "space"
                and link.source_field in {
                    "label", "name", "long_name", "aliases", "synonyms",
                }
                and normalize_question_text(link.text) == normalize_question_text(phrase)
            ]
            if exact_named_spaces:
                linked = exact_named_spaces
            elif relation == "contains":
                linked_spaces = [link for link in linked if link.kind == "space"]
                if linked_spaces:
                    linked = linked_spaces
            node_ids = _ordered_unique(
                node_id for link in linked for node_id in link.node_ids
            )
            kinds = _ordered_unique(link.kind for link in linked if link.kind != "unknown")
            references.append(
                RelationReference(
                    stage_id=f"reference_{len(references) + 1}",
                    relation=relation,  # type: ignore[arg-type]
                    mentions=[link.text for link in linked] or [phrase],
                    node_ids=node_ids,
                    reference_kind=kinds[0] if len(kinds) == 1 else None,
                )
            )
            spans.append((match.start(), min(len(text), match.end() + len(phrase))))
    return references, spans


def _without_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(max(0, start), min(len(chars), end)):
            chars[index] = " "
    return "".join(chars)


def _operator(text: str, actions: Sequence[_ActionOccurrence]) -> str:
    # The requested operation is authoritative over a property phrase such as
    # "by room number".
    if re.search(r"\b(?:list|enumerate)\b", text):
        return "list"
    if _space_subject_contains_query(text):
        return "list"
    if re.search(
        r"\b(?:what|which)\s+.+?\s+(?:are|were)\s+"
        r"(?:contained|located|found|installed|present)\s+"
        r"(?:in|inside|within)\b",
        text,
    ):
        return "list"
    if re.search(r"\b(?:room|space)\s+(?:name|number)\b", text):
        return "lookup"
    if re.search(r"\b(?:most\s+common|predominant(?:ly)?)\b", text):
        return "group_count"
    if re.search(
        r"\b(?:rooms?|spaces?|areas?|zones?)\b[^?.;]*?"
        r"\b(?:has|have|contains?|includes?)\s+(?:the\s+)?most\b",
        text,
    ):
        return "argmax"
    if re.search(r"\b(?:largest|smallest|highest|lowest|maximum|minimum)\b", text):
        return "argmax"
    if re.search(r"\b(?:unconnected|not\s+connected|without\s+(?:a\s+)?connection)\b", text):
        return "unconnected"
    if re.search(r"\b(?:nearest|closest|nearby|adjacent\s+to)\b", text):
        return "nearest"
    if re.search(r"\b(?:how\s+many|count|number\s+of)\b", text):
        return "count"
    if re.search(r"\b(?:types?|kinds?|categories?)\s+of\b|\bwhat\s+types?\b", text):
        return "distinct"
    if re.search(r"\b(?:all|every|each)\b", text):
        return "all_matching"
    if actions:
        return "path"
    return "lookup"


def _query_focus_spans(text: str, operator: str) -> list[tuple[int, int]]:
    """Locate the noun phrase operated on by non-action queries."""

    if operator not in {"list", "distinct", "count", "group_count", "argmax", "all_matching"}:
        return []
    patterns = (
        r"\bhow\s+many\s+(.+?)(?=\s+(?:are|is|exist|occur|remain|located|found)\b|[?.]|$)",
        r"\b(?:count|list|enumerate)\s+(?:all\s+|the\s+)?(.+?)"
        r"(?=\s+(?:in|inside|within|on|at|from|by|with|that|which)\b|[?.]|$)",
        r"\b(?:what|which)(?:\s+are)?\s+(?:the\s+)?"
        r"(?:types?|kinds?|categories?)\s+of\s+(.+?)"
        r"(?=\s+(?:are|is|used|exist|occur|located|found|in|on|at)\b|[?.]|$)",
        r"\b(?:rooms?|spaces?|areas?|zones?)\b[^?.;]*?"
        r"\b(?:has|have|contains?|includes?)\s+(?:the\s+)?most\s+(.+?)"
        r"(?=\s+(?:in|inside|within|on|at|from|by|with)\b|[?.]|$)",
        r"\b(?:most\s+common|predominant(?:ly)?)\s+(.+?)"
        r"(?=\s+(?:famil(?:y|ies)|types?|kinds?|classes?)\b|[?.]|$)",
        r"\b(?:what|which)\s+(.+?)\s+(?:famil(?:y|ies)|types?|kinds?)\s+"
        r"(?:is|are)\s+(?:the\s+)?(?:most\s+common|predominant)\b",
        r"\b(?:famil(?:y|ies)|types?|kinds?)\s+of\s+(.+?)\s+"
        r"(?:is|are)\s+(?:the\s+)?(?:most\s+common|predominant)\b",
        r"\b(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+"
        r"(?:famil(?:y|ies)|types?|kinds?|classes?)\s+"
        r"(?:used|installed|found)\s+(?:most\s+)?predominant(?:ly)?\b",
        r"\b(?:what|which)\s+(.+?)\s+(?:are|were)\s+"
        r"(?:contained|located|found|installed|present)\s+"
        r"(?:in|inside|within)\b",
    )
    result: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            result.append((match.start(1), match.end(1)))
    return result


def _space_collection_query(text: str, operator: str) -> bool:
    return bool(
        operator == "distinct"
        and re.search(
            r"\b(?:types?|kinds?|categories?)\s+of\s+"
            r"(?:rooms?|spaces?|areas?|zones?)\b",
            text,
        )
    )


def _implicit_collection_kind(text: str) -> str | None:
    match = re.search(
        r"\b(?:what|which)\s+(.+?)\s+(?:are|were)\s+"
        r"(?:contained|located|found|installed|present)\s+"
        r"(?:in|inside|within)\b",
        text,
    )
    if not match:
        return None
    tokens = normalize_question_text(match.group(1)).split()
    if not tokens:
        return None
    head = _singular_token(tokens[-1])
    if head in _SPACE_HEADS:
        return "space"
    if head in _GENERIC_NOUNS or _singular_token(tokens[-1]) != tokens[-1]:
        return "object"
    return None


def _space_surface_property_terms(text: str, operator: str) -> list[str]:
    """Extract a space-surface property phrase without using building labels."""

    if operator != "lookup":
        return []
    match = re.search(
        r"\b(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+"
        r"(?:in|inside|within|of)\s+(?:the\s+)?",
        text,
    )
    if not match:
        return []
    tokens = list(_signature(match.group(1)))
    while tokens and tokens[-1] in {"type", "kind", "class", "classification", "value"}:
        tokens.pop()
    # ``_signature`` applies conservative derivational normalization, so
    # "covering" appears as ``cover`` here.
    surface_terms = {"cover", "covering", "finish", "material", "surface"}
    if not surface_terms.intersection(tokens):
        return []
    normalized = [
        "finish" if token in {"cover", "covering"} else token for token in tokens
    ]
    return _ordered_unique(normalized)


def _scope_cardinality(text: str, operator: str) -> str:
    if re.search(r"\b(?:all|every|each)\s+(?:\w+\s+){0,3}(?:rooms?|spaces?|areas?|zones?)\b", text):
        return "all"
    if re.search(r"\bboth\b", text):
        return "all"
    if operator in {"count", "distinct", "group_count", "argmax"}:
        return "aggregation"
    return "single"


def _answer_cardinality(text: str, operator: str) -> CardinalityPolicy:
    if operator == "count":
        return "count"
    if operator == "distinct":
        return "distinct"
    if operator == "group_count":
        return "group_count"
    if operator == "argmax":
        return "argmax"
    if operator in {"list", "all_matching"} or re.search(r"\b(?:all|every|each)\b", text):
        return "all"
    return "single"


def _binding_cardinalities(
    text: str,
    actions: Sequence[_ActionOccurrence],
    default: CardinalityPolicy,
) -> dict[int, CardinalityPolicy]:
    """Derive quantifiers independently for each ordered action clause."""

    result: dict[int, CardinalityPolicy] = {}
    phrases_by_action: dict[int, list[str]] = {}
    for action_index, phrase in _target_phrases(
        text, _action_target_spans(text, actions)
    ):
        phrases_by_action.setdefault(action_index, []).append(phrase)
    for index, action in enumerate(actions):
        end = actions[index + 1].start if index + 1 < len(actions) else len(text)
        clause = text[action.end:end]
        if re.search(r"\b(?:all|every|each)\b", clause):
            result[index] = "all"
        elif re.search(r"\b(?:how\s+many|count|number\s+of)\b", clause):
            result[index] = "count"
        else:
            plural_target = any(
                tokens
                and _singular_token(tokens[-1]) != tokens[-1]
                for phrase in phrases_by_action.get(index, [])
                if (tokens := normalize_question_text(phrase).split())
            )
            if plural_target:
                result[index] = "all"
            elif default in {"argmax"}:
                result[index] = "single"
            else:
                result[index] = "single" if len(actions) > 1 else default
    return result


def _target_kind(
    actions: Sequence[_ActionOccurrence],
    links: Sequence[MentionLink],
    target_phrases: Sequence[tuple[int, str]],
) -> str | None:
    relevant_links = (
        [link for link in links if link.action_index is not None]
        if actions
        else list(links)
    )
    kinds = [link.kind for link in relevant_links if link.kind != "unknown"]
    if any(action.action == "Navigate" for action in actions) and not any(
        action.action in {"Inspect", "Scan"} for action in actions
    ):
        return "space"
    if "object" in kinds:
        return "object"
    if "system" in kinds:
        return "system"
    if any(link.kind == "function" and link.compatible_roles for link in relevant_links):
        return "object"
    # Inspect/Scan semantics plus a non-space target head is authoritative over
    # an overlapping space modifier linked inside the noun phrase.
    if any(action.action in {"Inspect", "Scan"} for action in actions):
        return "object"
    if "space" in kinds:
        return "space"
    for _, phrase in target_phrases:
        if _is_explicit_space_target(phrase):
            return "space"
    return None


def _append_scope(
    predicates: list[ScopePredicate],
    predicate: str,
    values: Iterable[str] = (),
) -> None:
    unique_values = _ordered_unique(values)
    key = (predicate, tuple(normalize_question_text(value) for value in unique_values))
    if any(
        (item.predicate, tuple(normalize_question_text(value) for value in item.values)) == key
        for item in predicates
    ):
        return
    predicates.append(
        ScopePredicate(
            stage_id=f"scope_{len(predicates) + 1}",
            predicate=predicate,  # type: ignore[arg-type]
            values=unique_values,
        )
    )


def _scope_function_types(plan: QueryPlan) -> set[str]:
    return {
        normalize_question_text(value)
        for predicate in plan.scope_predicates
        if predicate.predicate == "space_function"
        for value in predicate.values
    }


def _binding_functions(plan: QueryPlan, action_index: int) -> list[str]:
    single_action = len(plan.action_sequence) == 1
    scoped = _scope_function_types(plan)
    linked = [
        link.function_type
        for link in plan.mention_links
        if link.kind == "function"
        and link.function_type
        and not _is_space_use_link(link)
        and (
            link.action_index == action_index
            or (
                single_action
                and link.action_index is None
                and normalize_question_text(link.function_type) not in scoped
            )
        )
    ]
    return _ordered_unique(
        [*linked, *(plan.function_intents if single_action else [])]
    )


def _binding_function_links(
    plan: QueryPlan,
    action_index: int,
) -> list[MentionLink]:
    single_action = len(plan.action_sequence) == 1
    scoped = _scope_function_types(plan)
    return [
        link
        for link in plan.mention_links
        if link.kind == "function"
        and not _is_space_use_link(link)
        and (
            link.action_index == action_index
            or (
                single_action
                and link.action_index is None
                and normalize_question_text(link.function_type or "") not in scoped
            )
        )
    ]


def _binding_system_links(
    plan: QueryPlan,
    action_index: int,
) -> list[MentionLink]:
    single_action = len(plan.action_sequence) == 1
    return [
        link
        for link in plan.mention_links
        if link.kind == "system"
        and link.system_category
        and (
            link.action_index == action_index
            or (single_action and link.action_index is None)
        )
    ]


def _links_for_action(
    links: Sequence[MentionLink],
    action_index: int,
) -> list[MentionLink]:
    """Apply exact semantic-role precedence within one action binding."""

    linked = [link for link in links if link.action_index == action_index]
    role_links = [
        link for link in linked
        if link.kind == "object" and link.source_field == "role" and link.role
    ]
    if not role_links:
        return linked
    authoritative_roles = {link.role for link in role_links if link.role}
    return [
        link for link in linked
        if link.kind != "object"
        or link.source_field == "role"
        or (
            _is_specific_name_source(link.source_field)
            and not link.source_field.startswith("suggestion:")
        )
        or (
            link.role in authoritative_roles
            and (
                link.source_field in {"aliases", "synonyms"}
                or (
                    _is_specific_name_source(link.source_field)
                    and not link.source_field.startswith("partial:")
                )
            )
        )
    ]


def _domain_head_role_links(
    links: Sequence[MentionLink],
    phrase: str,
    action_index: int,
) -> list[MentionLink]:
    """Derive an operational role from graph domain plus generic head.

    The derivation uses only roles carried by active-graph domain matches.  A
    generic head therefore narrows a domain when that head is actually part of
    the graph role (for example ``service_fixture`` + ``fixtures``), without a
    built-in asset vocabulary or cross-domain guessing.
    """

    tokens = [
        token for token in _signature(phrase) if token not in _LINK_STOPWORDS
    ]
    if not tokens:
        return []
    head = _singular_token(tokens[-1])
    if head not in _GENERIC_NOUNS:
        return []
    derived: list[MentionLink] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for link in links:
        if (
            link.kind != "object"
            or link.source_field != "domain"
            or not link.domain
            or not link.role
            or head not in _signature(link.role)
        ):
            continue
        key = (
            normalize_question_text(link.domain),
            normalize_question_text(link.role),
            tuple(link.node_ids),
        )
        if key in seen:
            continue
        seen.add(key)
        derived.append(
            replace(
                link,
                text=phrase,
                canonical=link.role,
                action_index=action_index,
                source_field="derived:domain_head",
            )
        )
    return derived


def build_action_bindings(plan: QueryPlan) -> list[ActionTargetBinding]:
    """Build independent, ordered target slots for every action occurrence."""

    actions = list(plan.action_sequence)
    if not actions:
        return []
    global_roles = list(plan.target_roles or ([plan.target_role] if plan.target_role else []))
    global_names = list(plan.target_names or ([plan.target_name] if plan.target_name else []))
    global_domains = [plan.target_domain] if plan.target_domain else []
    target_action_indices = [
        index for index, action in enumerate(actions)
        if action.casefold() not in {"navigate", "nav"}
    ]
    bindings: list[ActionTargetBinding] = []

    for index, action in enumerate(actions):
        if action.casefold() in {"navigate", "nav"}:
            bindings.append(
                ActionTargetBinding(
                    action=action,
                    binding_index=index,
                    source_stage="scope",
                    target_kind="space",
                    target_names=list(plan.room_names),
                    result_stage="scope",
                    target_mode="auto",
                    cardinality_policy=(
                        "all" if plan.scope_cardinality == "all" else "single"
                    ),
                )
            )
            continue

        linked = _links_for_action(plan.mention_links, index)
        function_links = _binding_function_links(plan, index)
        system_links = _binding_system_links(plan, index)
        has_action_mention = bool(linked or function_links or system_links)
        roles = _ordered_unique(
            link.role
            for link in linked
            if link.kind == "object"
            and link.role
            and link.source_field != "domain"
        )
        names = _ordered_unique(
            link.text
            for link in linked
            if (
                link.kind in {"object", "system"}
                and _is_specific_name_source(link.source_field)
            )
        )
        unresolved_names = _ordered_unique(
            link.text
            for link in linked
            if link.kind == "unknown" and link.source_field == "action_object"
        )
        domains = _ordered_unique(
            link.domain
            for link in linked
            if link.kind in {"object", "system"} and link.domain
        )
        kinds = _ordered_unique(link.kind for link in linked if link.kind != "unknown")

        explicit_target_evidence = bool(roles or names or domains)
        if not roles:
            roles = _ordered_unique(
                role for link in function_links for role in link.compatible_roles
            )
        if not names and not explicit_target_evidence:
            names = _ordered_unique(
                name for link in function_links for name in link.compatible_names
            )
        if not names:
            names = unresolved_names
        if not domains and not explicit_target_evidence:
            domains = _ordered_unique(
                domain for link in function_links for domain in link.compatible_domains
            )
        system_categories = _ordered_unique(
            link.system_category for link in system_links if link.system_category
        )

        # Compatibility fallback for manually-created/legacy QueryPlan values.
        if not roles and not has_action_mention:
            if len(target_action_indices) == len(global_roles) and len(target_action_indices) > 1:
                roles = [global_roles[target_action_indices.index(index)]]
            else:
                roles = list(global_roles)
        if not names and not has_action_mention:
            if len(target_action_indices) == len(global_names) and len(target_action_indices) > 1:
                names = [global_names[target_action_indices.index(index)]]
            else:
                names = list(global_names)
        if not domains and not has_action_mention:
            domains = list(global_domains)

        if len(actions) == 1 and global_names:
            # The global compiler has already converted coordinated noun
            # phrases under one action into an OR-list. Preserve that union in
            # the action slot itself; otherwise a name-qualified right
            # conjunct can silently erase a role-linked left conjunct during
            # candidate×binding validation. This happens after ontology
            # fallback so compatible domain/function fields remain intact.
            names = _ordered_unique([*names, *global_names])

        constraint_branches: list[dict[str, Any]] = []
        coordinated_phrases = list(plan.target_phrases)
        if len(actions) == 1 and len(coordinated_phrases) > 1:
            for phrase in coordinated_phrases:
                phrase_links = [
                    link
                    for link in linked
                    if _signature(link.text) == _signature(phrase)
                    and link.kind in {"object", "system"}
                ]
                branch_roles = _ordered_unique(
                    link.role
                    for link in phrase_links
                    if link.role and link.source_field != "domain"
                )
                branch_domains = _ordered_unique(
                    link.domain for link in phrase_links if link.domain
                )
                requires_name = any(
                    _is_specific_name_source(link.source_field)
                    for link in phrase_links
                ) or not (branch_roles or branch_domains)
                constraint_branches.append(
                    {
                        "roles": branch_roles,
                        "names": [phrase] if requires_name else [],
                        "domains": branch_domains,
                        "cardinality": _phrase_cardinality(phrase),
                    }
                )

        target_kind = (
            "object" if "object" in kinds
            else "system" if "system" in kinds
            else "object" if any((roles, names, domains, function_links))
            else plan.target_kind or "object"
        )
        target_mode = plan.target_binding_mode
        if len(actions) > 1 and system_categories:
            target_mode = "system_members" if target_kind == "object" else "system_entity"
        bindings.append(
            ActionTargetBinding(
                action=action,
                binding_index=index,
                source_stage=f"targets_{index + 1}",
                target_kind=target_kind,
                target_roles=roles,
                target_names=names,
                target_domains=domains,
                constraint_branches=constraint_branches,
                result_stage="targets",
                function_types=_binding_functions(plan, index),
                system_categories=system_categories,
                target_mode=target_mode,
                cardinality_policy=plan.cardinality_policy,
            )
        )
    return bindings


def build_staged_constraints(plan: QueryPlan, question: str) -> QueryPlan:
    """Compile linked mentions into scope, target and relation stages."""

    normalized = normalize_question_text(question)
    scopes: list[ScopePredicate] = []
    if plan.storey:
        _append_scope(scopes, "storey", [plan.storey])
    if plan.room:
        _append_scope(scopes, "space_number", [plan.room])
    if plan.room_names:
        _append_scope(scopes, "space_name", plan.room_names)
    if plan.target_space_type:
        _append_scope(scopes, "space_type", [plan.target_space_type])
    relative_scope_function_links = [
        link
        for link in plan.mention_links
        if link.kind == "function"
        and link.function_type
        and link.char_start >= 0
        and _is_relative_clause_scope_mention(normalized, link.char_start)
    ]
    relative_scope_function_types = {
        normalize_question_text(link.function_type or "")
        for link in relative_scope_function_links
    }
    space_functions = _ordered_unique(
        link.function_type
        for link in plan.mention_links
        if link.kind == "function"
        and link.function_type
        and (
            _is_space_use_link(link)
            or normalize_question_text(link.function_type)
            in relative_scope_function_types
        )
    )
    if space_functions:
        _append_scope(scopes, "space_function", space_functions)

    contextual = re.search(
        r"\b(?:space|room|area|zone|corridor|hall|hallway|office|laboratory|lab|"
        r"classroom|restroom|lobby|stair|studio|suite|workshop)s?\b[^.?,;]*?"
        r"\b(?:that|which)\s+(?:also\s+)?(?:contains?|has|includes?)\s+"
        r"(?:(?:an|a|the|any)\s+)?([^,.?;]+)",
        normalized,
    )
    if contextual:
        phrase = contextual.group(1).strip()
        phrase = re.split(
            r"\s+(?:are|is)\s+(?:on|in|at|within)\b",
            phrase,
            maxsplit=1,
        )[0].strip()
        phrase_signature = set(_signature(phrase))
        linked = [
            link
            for link in plan.mention_links
            if normalize_question_text(link.text) in phrase
            or all(
                token in phrase_signature
                for token in _signature(link.text)
                if token not in _LINK_STOPWORDS
            )
        ]
        roles = _ordered_unique(link.role for link in linked if link.role)
        specific_roles = [
            role for role in roles
            if _singular_token(normalize_question_text(role).split()[-1])
            not in _GENERIC_NOUNS
        ]
        explicit_domains = _ordered_unique(
            link.domain
            for link in linked
            if link.domain and link.source_field == "domain"
        )
        domains = explicit_domains or _ordered_unique(
            link.domain for link in linked if link.domain
        )
        names = _ordered_unique(
            link.text
            for link in linked
            if link.kind == "object" and _is_specific_name_source(link.source_field)
        )
        phrase_tokens = normalize_question_text(phrase).split()
        plural_collection = bool(
            phrase_tokens
            and _singular_token(phrase_tokens[-1]) != phrase_tokens[-1]
        )
        generic_head = bool(
            phrase_tokens
            and _singular_token(phrase_tokens[-1]) in _GENERIC_NOUNS
        )
        if generic_head and explicit_domains:
            _append_scope(scopes, "contains_domain", explicit_domains)
        elif plural_collection and specific_roles:
            _append_scope(scopes, "contains_role", specific_roles)
        elif names:
            _append_scope(scopes, "contains_name", names)
        elif specific_roles:
            _append_scope(scopes, "contains_role", specific_roles)
        elif domains:
            _append_scope(scopes, "contains_domain", domains)
        elif roles:
            _append_scope(scopes, "contains_role", roles)
        else:
            _append_scope(scopes, "contains_name", [phrase])
    if plan.operator == "argmax" and "area" in plan.property_terms and any(
        token in _SPACE_HEADS for token in normalized.split()
    ):
        _append_scope(scopes, "argmax_area")

    allow_global_support = len(plan.action_sequence) <= 1
    target_functions = (
        _ordered_unique(
            link.function_type
            for link in plan.mention_links
            if link.function_type
            and not _is_space_use_link(link)
            and normalize_question_text(link.function_type)
            not in relative_scope_function_types
        )
        if allow_global_support and plan.target_kind != "space"
        else []
    )
    # Only an explicit system mention is a required system predicate.
    # Function ontology relations remain on MentionLink as expansion/ranking
    # hints and must not silently become hard target constraints.
    system_categories = (
        _ordered_unique(
            link.system_category
            for link in plan.mention_links
            if link.kind == "system" and link.system_category
        )
        if allow_global_support and plan.target_kind != "space"
        else []
    )
    targets: list[TargetPredicate] = []
    for predicate, values in (
        ("kind", [plan.target_kind] if plan.target_kind else []),
        ("ifc_class", [plan.target_ifc_class] if plan.target_ifc_class else []),
        ("role", list(plan.target_roles or ([plan.target_role] if plan.target_role else []))),
        ("name", list(plan.target_names or ([plan.target_name] if plan.target_name else []))),
        ("domain", [plan.target_domain] if plan.target_domain else []),
        ("function", target_functions),
        ("system", system_categories),
    ):
        if values:
            targets.append(
                TargetPredicate(
                    predicate=predicate,  # type: ignore[arg-type]
                    values=_ordered_unique(values),
                    required=True,
                )
            )

    plan.scope_predicates = scopes
    plan.target_predicates = targets
    plan.function_intents = target_functions
    grouping: list[str] = []
    if len(plan.room_names) > 1 or plan.scope_cardinality == "all":
        grouping.append("scope_space")
    if len(plan.target_roles) > 1:
        grouping.append("target_role")
    if plan.operator == "group_count":
        grouping.extend(
            term
            for term in plan.property_terms
            if term in {"family", "type", "kind", "class", "role"}
        )
    plan.target_grouping = _ordered_unique(grouping)

    unresolved: list[str] = []
    if any(action.casefold() not in {"navigate", "nav"} for action in plan.action_sequence):
        if plan.target_kind not in {"object", "system"}:
            unresolved.append("action_target_kind")
        grounded_action_mention = any(
            link.action_index is not None and link.kind in {"object", "system"}
            for link in plan.mention_links
        )
        action_local_function = any(
            link.action_index is not None
            and link.kind == "function"
            and not _is_space_use_link(link)
            for link in plan.mention_links
        )
        if (
            not grounded_action_mention
            and not action_local_function
            and not any((target_functions, system_categories))
        ):
            unresolved.append("target_mention")
    for reference in plan.relation_references:
        if not reference.node_ids:
            unresolved.append(f"relation_reference:{reference.stage_id}")
    plan.unresolved_slots = _ordered_unique(unresolved)
    plan.action_bindings = build_action_bindings(plan)
    for binding in plan.action_bindings:
        if binding.target_kind == "space":
            continue
        typed_name = any(
            link.action_index == binding.binding_index
            and link.kind in {"object", "system"}
            and link.source == "graph"
            and not link.source_field.startswith("suggestion:")
            for link in plan.mention_links
        )
        if binding.target_kind == "object":
            system_member_scope = (
                binding.target_mode == "system_members"
                and bool(binding.system_categories)
            )
            typed = bool(
                binding.target_roles
                or binding.target_domains
                or binding.function_types
                or typed_name
                or system_member_scope
            )
            if not typed:
                unresolved.append(
                    f"action_binding:{binding.binding_index}:target_mention"
                )
        elif binding.target_kind == "system":
            exact_system = any(
                link.action_index == binding.binding_index
                and link.kind == "system"
                and link.node_ids
                and link.source_field in {
                    "label", "name", "long_name", "aliases", "synonyms",
                }
                for link in plan.mention_links
            )
            if not exact_system:
                unresolved.append(
                    f"action_binding:{binding.binding_index}:target_mention"
                )
    plan.unresolved_slots = _ordered_unique(unresolved)
    return plan


def _semantic_budget(plan: QueryPlan) -> None:
    functional = bool(plan.function_intents) or any(
        link.kind in {"function", "system"} for link in plan.mention_links
    )
    relational = bool(plan.relation_references) or any(
        predicate.predicate.startswith("contains_") for predicate in plan.scope_predicates
    )
    action_task = bool(plan.action_sequence)
    aggregation = plan.operator in {
        "list", "distinct", "count", "group_count", "argmax", "all_matching", "unconnected"
    }
    if functional:
        plan.gnn_hops, plan.tog_depth, plan.tog_width = 3, 4, 5
        plan.gnn_retrieval_levels = ["function", "system", "object", "space"]
    elif relational:
        plan.gnn_hops, plan.tog_depth, plan.tog_width = 2, 3, 4
        plan.gnn_retrieval_levels = ["space", "object", "system"]
    elif action_task or aggregation:
        plan.gnn_hops, plan.tog_depth, plan.tog_width = 1, 2, 3
        plan.gnn_retrieval_levels = ["object", "space"]
    else:
        plan.gnn_hops, plan.tog_depth, plan.tog_width = 0, 1, 2
        plan.gnn_retrieval_levels = (
            ["space", "object"] if plan.target_kind == "space" else ["object", "space"]
        )


def infer_query_plan(
    question: str,
    category: int | None = None,
    *,
    schema_context: SchemaContext | None = None,
) -> QueryPlan:
    """Compile a category-independent, graph-grounded query plan.

    ``category`` remains in the signature for existing callers but is
    intentionally ignored.  Evaluation labels must never change planning,
    retrieval depth or action binding.
    """

    del category
    normalized = normalize_question_text(question)
    schema = coerce_planning_schema(schema_context)
    actions = _actions(normalized)
    operator = _operator(normalized, actions)
    space_collection = _space_collection_query(normalized, operator)
    space_subject_contains = _space_subject_contains_query(normalized)
    implicit_collection_kind = _implicit_collection_kind(normalized)
    space_property_terms = _space_surface_property_terms(normalized, operator)
    target_spans = _action_target_spans(normalized, actions)
    focus_spans = _query_focus_spans(normalized, operator) if not actions else []
    linked_mentions = schema.link(
        question,
        target_spans=target_spans,
        focus_spans=focus_spans,
    )
    existing_exact_spaces = {
        (normalize_question_text(link.text), tuple(link.node_ids))
        for link in linked_mentions
        if link.kind == "space"
        and link.source_field in {
            "label", "name", "long_name", "aliases", "synonyms",
        }
    }
    coordinated_space_links = _coordinated_space_links(question, schema)
    for link in coordinated_space_links:
        key = (normalize_question_text(link.text), tuple(link.node_ids))
        if key in existing_exact_spaces:
            continue
        linked_mentions.append(link)
        existing_exact_spaces.add(key)
    for link in _typed_space_modifier_links(normalized, schema):
        key = (normalize_question_text(link.text), tuple(link.node_ids))
        if key in existing_exact_spaces:
            continue
        linked_mentions.append(link)
        existing_exact_spaces.add(key)
    phrases = _complete_shared_target_heads(
        _target_phrases(normalized, target_spans),
        linked_mentions,
    )
    unresolved_generic_actions = _unresolved_generic_compound_actions(
        phrases, linked_mentions
    )
    if unresolved_generic_actions:
        # A generic head is not sufficient grounding for an uncovered
        # modifier.  Remove only the broad role link; exact names, domains,
        # functions, systems, and independently grounded compounds remain.
        linked_mentions = [
            link
            for link in linked_mentions
            if not (
                link.action_index in unresolved_generic_actions
                and link.kind in {"object", "system"}
                and link.source_field == "role"
                and _singular_token(normalize_question_text(link.text))
                in _GENERIC_NOUNS
            )
        ]

    # Exact graph mentions remain authoritative.  Only genuinely unresolved
    # target heads receive conservative longer-label matching.
    def mention_key(link: MentionLink) -> tuple[Any, ...]:
        # A semantic role and a graph-backed lexical qualifier can point to the
        # same node while contributing different constraints.  Preserve both.
        return (
            link.action_index,
            link.kind,
            tuple(link.node_ids),
            link.canonical,
            link.source_field,
            normalize_question_text(link.text),
        )

    existing_keys = {mention_key(link) for link in linked_mentions}
    for action_index, phrase in phrases:
        if _is_explicit_space_target(phrase):
            continue
        candidate_links = schema.link_phrase(
            phrase,
            preferred_kinds=("object", "system", "function"),
            action_index=action_index,
        )
        if candidate_links and all(_is_space_use_link(link) for link in candidate_links):
            candidate_links = schema.partial_link(
                phrase,
                preferred_kinds=("object", "system"),
                action_index=action_index,
            )
        phrase_tokens = [
            token for token in _signature(phrase) if token not in _LINK_STOPWORDS
        ]
        exact_named_target = any(
            link.kind in {"object", "system"}
            and _is_specific_name_source(link.source_field)
            and not link.source_field.startswith("partial:")
            for link in candidate_links
        )
        exact_role_signatures = {
            tuple(
                token
                for token in _signature(link.role or link.canonical)
                if token not in _LINK_STOPWORDS
            )
            for link in candidate_links
            if link.kind in {"object", "system"}
            and link.source_field == "role"
            and not link.source_field.startswith("partial:")
            and (link.role or link.canonical)
        }
        exact_role_covers_phrase = (
            len(exact_role_signatures) == 1
            and all(
                token in next(iter(exact_role_signatures))
                for token in phrase_tokens
            )
        )
        if (
            len(phrase_tokens) > 1
            and not exact_named_target
            and not exact_role_covers_phrase
        ):
            candidate_links.extend(
                schema.partial_link(
                    phrase,
                    preferred_kinds=("object", "system"),
                    action_index=action_index,
                )
            )
        candidate_links.extend(
            _domain_head_role_links(candidate_links, phrase, action_index)
        )
        if (
            phrase_tokens
            and not any(
                link.kind in {"object", "system"} and link.role
                for link in candidate_links
            )
            and any(
                link.kind in {"object", "system"}
                and (link.domain or link.system_category)
                for link in candidate_links
            )
        ):
            # A compound's rightmost noun is a generic head fallback.  The
            # preceding graph-linked domain/name constraints remain in place,
            # so a broad head such as "fixture" cannot become an answer by
            # itself.
            candidate_links.extend(
                schema.link_phrase(
                    phrase_tokens[-1],
                    preferred_kinds=("object", "system"),
                    action_index=action_index,
                )
            )
        for link in candidate_links:
            key = mention_key(link)
            if key not in existing_keys:
                linked_mentions.append(link)
                existing_keys.add(key)

    if unresolved_generic_actions:
        # Phrase-level linking above may rediscover the generic head after the
        # initial pass, so enforce the same modifier-aware rule on the complete
        # target-link set.
        linked_mentions = [
            link
            for link in linked_mentions
            if not (
                link.action_index in unresolved_generic_actions
                and link.kind in {"object", "system"}
                and link.source_field == "role"
                and _singular_token(normalize_question_text(link.text))
                in _GENERIC_NOUNS
            )
        ]

    if not actions and not space_collection:
        for start, end in focus_spans:
            if any(
                link.query_focus and link.kind in {"object", "system", "function"}
                for link in linked_mentions
            ):
                break
            phrase = normalized[start:end]
            for link in schema.link_phrase(
                phrase, preferred_kinds=("object", "system", "function")
            ):
                link.query_focus = True
                key = mention_key(link)
                if key not in existing_keys:
                    linked_mentions.append(link)
                    existing_keys.add(key)

    # Nested scope objects are support evidence, not action targets.
    for match in re.finditer(
        r"\b(?:that|which)\s+(?:also\s+)?(?:contains?|has|includes?)\s+"
        r"(?:(?:an|a|the|any)\s+)?([^,.?;]+)",
        normalized,
    ):
        phrase = match.group(1).strip()
        for link in schema.link_phrase(
            phrase, preferred_kinds=("object", "function")
        ):
            link = replace(link, action_index=None)
            key = mention_key(link)
            if key not in existing_keys:
                linked_mentions.append(link)
                existing_keys.add(key)

    linked_mentions = _suppress_contained_cross_kind_links(linked_mentions)
    linked_mentions = _suppress_contained_function_links(linked_mentions)
    linked_mentions = _assign_action_local_support_links(
        normalized, actions, linked_mentions
    )
    for link in _function_target_role_links(actions, phrases, linked_mentions):
        key = mention_key(link)
        if key not in existing_keys:
            linked_mentions.append(link)
            existing_keys.add(key)

    for link in _pronoun_action_links(normalized, actions, linked_mentions):
        key = mention_key(link)
        if key not in existing_keys:
            linked_mentions.append(link)
            existing_keys.add(key)
    actions = _normalize_space_actions(actions, linked_mentions, phrases)
    navigate_only = bool(actions) and all(
        action.action == "Navigate" for action in actions
    )

    # Preserve unresolved syntactic mentions so a bounded structured planner
    # can resolve them against graph candidates.  These are not classifications.
    linked_action_indices = {
        link.action_index
        for link in linked_mentions
        if link.action_index is not None
        and link.kind in {"object", "system", "function"}
        and not (link.kind == "function" and _is_space_use_link(link))
    }
    semantic_roles, semantic_domains, semantic_systems = (
        _schema_semantic_options(schema)
    )
    for action_index, phrase in phrases:
        if action_index in linked_action_indices:
            continue
        if (
            action_index < len(actions)
            and actions[action_index].action == "Navigate"
            and any(
                _singular_token(token) in _SPACE_HEADS
                for token in normalize_question_text(phrase).split()
            )
        ):
            continue
        suggestions = schema.suggest(phrase, action_index=action_index)
        fallback_links = list(suggestions)
        if action_index in unresolved_generic_actions or not fallback_links:
            fallback_links.append(
                MentionLink(
                    text=phrase,
                    canonical=phrase,
                    kind="unknown",
                    action_index=action_index,
                    confidence=0.0,
                    source="syntax",
                    source_field="action_object",
                    compatible_roles=(
                        semantic_roles
                        if action_index in unresolved_generic_actions
                        else ()
                    ),
                    compatible_domains=(
                        semantic_domains
                        if action_index in unresolved_generic_actions
                        else ()
                    ),
                    related_system_categories=(
                        semantic_systems
                        if action_index in unresolved_generic_actions
                        else ()
                    ),
                )
            )
        linked_mentions.extend(fallback_links)

    relation_references, _ = _relation_references(normalized, schema)
    scenario_source_node_ids: list[str] = []
    scenario_state_change = _SCENARIO_STATE_CHANGE.search(normalized)
    if (
        _SCENARIO_PREFIX.search(normalized)
        and scenario_state_change is not None
        and _SCENARIO_IMPACT.search(normalized)
    ):
        identifier_match = re.search(
            r"(?<![0-9A-Za-z_$])([0-9A-Za-z_$]{22})(?![0-9A-Za-z_$])",
            question,
        )
        scenario_source_node_ids = (
            schema.nodes_for_identifier(identifier_match.group(1))
            if identifier_match is not None
            else []
        )
        if scenario_source_node_ids:
            relation_references.append(
                RelationReference(
                    stage_id=f"reference_{len(relation_references) + 1}",
                    relation="assigned_to_system",
                    mentions=["changed source equipment"],
                    node_ids=scenario_source_node_ids,
                    reference_kind="object",
                    source_stage="targets",
                )
            )
    cardinality = _answer_cardinality(normalized, operator)
    if actions and operator == "argmax":
        action_cardinalities = _binding_cardinalities(
            normalized, actions, "single"
        )
        cardinality = (
            "all" if any(value == "all" for value in action_cardinalities.values())
            else "single"
        )
    scope_cardinality = _scope_cardinality(normalized, operator)

    reference_ids = {
        node_id for reference in relation_references for node_id in reference.node_ids
    }
    metric_reference_ids = {
        node_id
        for reference in relation_references
        if reference.relation in {"nearest", "adjacent_to"}
        for node_id in reference.node_ids
    }
    metric_reference_spans: list[tuple[int, int]] = []
    for reference in relation_references:
        if reference.relation not in {"nearest", "adjacent_to"}:
            continue
        for mention in reference.mentions:
            normalized_mention = normalize_question_text(mention)
            if not normalized_mention:
                continue
            metric_reference_spans.extend(
                (match.start(), match.end())
                for match in re.finditer(
                    re.escape(normalized_mention), normalized
                )
            )
    target_links = [
        link
        for link in linked_mentions
        if link.action_index is not None
        and not (reference_ids and reference_ids.intersection(link.node_ids))
    ]
    if not actions:
        focus_links = [link for link in linked_mentions if link.query_focus]
        target_links = focus_links or list(linked_mentions)

    if actions:
        object_links = [
            link
            for index in range(len(actions))
            for link in _links_for_action(target_links, index)
            if link.kind == "object"
        ]
    else:
        object_links = [link for link in target_links if link.kind == "object"]
    system_links = [link for link in target_links if link.kind == "system"]
    mentioned_system_links = [
        link for link in linked_mentions if link.kind == "system"
    ]
    space_links = [
        link
        for link in linked_mentions
        if link.kind == "space"
        and (
            link.action_index is None
            or (
                link.action_index < len(actions)
                and actions[link.action_index].action == "Navigate"
            )
            or (
                link.action_index < len(actions)
                and actions[link.action_index].action in {"Inspect", "Scan"}
                and any(
                    candidate.action_index == link.action_index
                    and candidate.kind in {"object", "system", "function"}
                    and not (
                        candidate.kind == "function"
                        and _is_space_use_link(candidate)
                    )
                    for candidate in target_links
                )
            )
        )
    ]
    function_links = [
        link for link in linked_mentions if link.kind == "function" and link.function_type
    ]
    space_function_links = [
        link for link in function_links
        if _is_space_use_link(link)
    ]
    relative_scope_function_links = [
        link
        for link in function_links
        if link.char_start >= 0
        and _is_relative_clause_scope_mention(normalized, link.char_start)
    ]
    target_function_links = [
        link for link in function_links
        if not _is_space_use_link(link)
        and link not in relative_scope_function_links
    ]
    navigation_reference_links = [
        link
        for link in [*object_links, *system_links]
        if link.action_index is not None
        and link.action_index < len(actions)
        and actions[link.action_index].action == "Navigate"
    ]
    if (
        navigate_only
        and navigation_reference_links
        and not any(
            reference.relation in {"nearest", "adjacent_to"}
            for reference in relation_references
        )
        and any(
            not _is_explicit_space_target(phrase)
            for action_index, phrase in phrases
            if action_index < len(actions)
            and actions[action_index].action == "Navigate"
        )
    ):
        # A proximity action can name both an object reference and a hard room
        # scope without asking the robot to navigate to the room centre.
        # Keep the executable Navigate target spatial, but require a grounded
        # proximity relation to the referenced object/system.  When the graph
        # has no containment, adjacency, or metric evidence the nearest
        # operator will abstain instead of silently binding the scope itself.
        relation_references.append(
            RelationReference(
                stage_id=f"reference_{len(relation_references) + 1}",
                relation="nearest",
                mentions=_ordered_unique(
                    link.text for link in navigation_reference_links
                ),
                node_ids=_ordered_unique(
                    node_id
                    for link in navigation_reference_links
                    for node_id in link.node_ids
                ),
                reference_kind=(
                    navigation_reference_links[0].kind
                    if len({link.kind for link in navigation_reference_links}) == 1
                    else None
                ),
                source_stage="targets",
            )
        )
        operator = "nearest"
    if navigate_only:
        # Every executable binding is spatial.  Incidental object/system words
        # inside an exact room label may remain useful explanation evidence,
        # but they must not change the global target contract.
        object_links = []
        system_links = []
        mentioned_system_links = []
        target_function_links = []
    allow_global_function_fallback = len(actions) <= 1
    global_system_links = system_links if allow_global_function_fallback else []
    target_names = _ordered_unique(
        link.text
        for link in object_links + global_system_links
        if _is_specific_name_source(link.source_field)
        and (
            link.kind == "system"
            or link.source_field.startswith("partial:")
            or not any(item.role for item in object_links)
        )
    )
    coordinated_target_phrases = _ordered_unique(
        phrase for action_index, phrase in phrases if action_index == 0
    ) if len(actions) == 1 else []
    if len(coordinated_target_phrases) > 1 and target_names:
        # Coordinated noun phrases are alternatives under one action.  If one
        # branch required a graph-label qualifier, retain the other branch's
        # surface phrase too so the shared name predicate remains OR, e.g.
        # both surface alternatives rather than accidentally conjoining them.
        # No asset vocabulary is embedded here.
        target_names = _ordered_unique(
            [*target_names, *coordinated_target_phrases]
        )
    # A phrase linked through graph role/domain/function vocabulary is already
    # semantically grounded.  Reusing its surface wording as an exact BIM name
    # constraint (for example, requiring an instance literally named
    # "lighting fixtures") would erase otherwise valid role matches.  Keep a
    # raw phrase only when the linker found no typed target at all; the bounded
    # planner/resolver can then disambiguate that genuinely unresolved mention.
    if (
        not target_names
        and not (object_links or system_links)
        and allow_global_function_fallback
        and not navigate_only
        and not unresolved_generic_actions
    ):
        target_names = _ordered_unique(
            name for link in target_function_links for name in link.compatible_names
        ) or _ordered_unique(phrase for _, phrase in phrases if _signature(phrase))
    # Explicit object mentions take precedence.  Function ontology constraints
    # are a fallback for elliptical requests, not a union expansion over every
    # role compatible with the function.
    explicit_roles = _ordered_unique(
        link.role
        for link in object_links
        if link.role and link.source_field != "domain"
    )
    target_roles = explicit_roles or (
        _ordered_unique(
            role for link in target_function_links for role in link.compatible_roles
        )
        if allow_global_function_fallback
        else []
    )
    explicit_domains = _ordered_unique(
        link.domain for link in object_links + global_system_links if link.domain
    )
    target_domains = explicit_domains or (
        _ordered_unique(
            domain for link in target_function_links for domain in link.compatible_domains
        )
        if allow_global_function_fallback
        else []
    )
    function_intents = (
        _ordered_unique(
            link.function_type for link in target_function_links if link.function_type
        )
        if allow_global_function_fallback
        else []
    )

    # Explicit schema-level space-type mentions are authoritative over
    # incidental graph labels/families that happen to contain the same word.
    # This keeps a query such as "... in office spaces" scoped to the linked
    # ontology type even when an object or named room also contains "office".
    # Metric/adjacency reference entities are evidence anchors, not hard
    # containment scopes. Exclude their labels *and* their schema types from
    # the scope compiler; otherwise "nearest Lab A" is silently rewritten as
    # "inside any lab and nearest Lab A".
    hard_scope_space_links = [
        link
        for link in space_links
        if not metric_reference_ids.intersection(link.node_ids)
        and not (
            link.char_start >= 0
            and any(
                start < link.char_end and link.char_start < end
                for start, end in metric_reference_spans
            )
        )
    ]
    semantic_space_links = [
        link
        for link in hard_scope_space_links
        if link.source_field == "space_type"
        and link.space_type
        and _compact_identifier(link.ifc_class or "") != "ifcbuildingstorey"
        and "storey" not in normalize_question_text(link.graph_level or "")
    ]
    space_type_links = semantic_space_links or hard_scope_space_links
    known_space_types = _ordered_unique(
        link.space_type
        for link in space_type_links
        if link.space_type
        and _compact_identifier(link.ifc_class or "") != "ifcbuildingstorey"
        and "storey" not in normalize_question_text(link.graph_level or "")
    )
    named_space_links = _longest_non_overlapping_space_links(
        [
            link
            for link in hard_scope_space_links
            if link.source_field in {"label", "name", "long_name"}
            and _compact_identifier(link.ifc_class or "") != "ifcbuildingstorey"
            and "storey" not in normalize_question_text(link.graph_level or "")
            and not metric_reference_ids.intersection(link.node_ids)
            and not _space_label_duplicates_type(link.text, known_space_types)
        ]
    )
    linked_space_names = _ordered_unique(
        link.text
        for link in named_space_links
    )
    reference_mentions = [
        mention
        for reference in relation_references
        if reference.relation in {"nearest", "adjacent_to"}
        for mention in reference.mentions
    ]
    # A space-use ontology's compatible types are fallback evidence, not an
    # explicit type assertion made by the user.  Compiling them into a hard
    # ``space_type`` scope would reject spaces connected to the function by a
    # direct materialized graph path merely because their type label differs.
    # ``build_staged_constraints`` retains the function as ``space_function``;
    # the backend can therefore prefer graph paths and use compatible types
    # only when those paths are missing.
    space_types = list(known_space_types)
    syntax_space_names = [
        value
        for value in _named_space_phrases(question)
        if not _is_typed_space_collection_phrase(value)
        if not _is_generic_typed_scope_phrase(value, space_types)
        if not _space_label_duplicates_type(value, space_types)
        if not any(
            normalize_question_text(link.text) in normalize_question_text(value)
            or normalize_question_text(value) in normalize_question_text(link.text)
            for link in space_function_links
        )
        if not any(
            normalize_question_text(value) in normalize_question_text(reference)
            or normalize_question_text(reference) in normalize_question_text(value)
            for reference in reference_mentions
        )
        if not any(
            normalize_question_text(value) in normalize_question_text(linked)
            or normalize_question_text(linked) in normalize_question_text(value)
            for linked in linked_space_names
        )
    ]
    room_names = _ordered_unique([*linked_space_names, *syntax_space_names])
    if len(room_names) > 1 and (
        re.search(r"\b(?:both|all)\b", normalized)
        or "/" in question
        or _has_coordinated_space_scope(normalized, named_space_links)
        or bool(coordinated_space_links)
    ):
        scope_cardinality = "all"
    target_kind = _target_kind(
        actions,
        [*target_links, *target_function_links],
        phrases,
    )
    if space_collection or space_subject_contains:
        target_kind = "space"
    elif implicit_collection_kind:
        target_kind = implicit_collection_kind

    room = _room_number(question)
    if room and any(
        re.search(rf"\b{re.escape(room.casefold())}\b", normalize_question_text(value))
        for value in reference_mentions
    ):
        room = None
    if room:
        # A parenthesized/identifier suffix is already represented by the
        # typed space-number predicate.  Do not also treat the same token as a
        # second named space, which would create spurious grouping/ambiguity.
        room_names = [
            name
            for name in room_names
            if normalize_question_text(name) != normalize_question_text(room)
        ]
    property_terms: list[str] = []
    for term in ("name", "number", "type", "kind", "class", "family", "area", "count"):
        if re.search(rf"\b{term}s?\b", normalized):
            property_terms.append(term)
    if space_collection:
        object_links = []
        system_links = []
        target_roles = []
        target_domains = []
        target_names = []
        property_terms = ["space_type"]
    elif space_subject_contains:
        object_links = []
        system_links = []
        target_roles = []
        target_domains = []
        target_names = []
    if re.search(r"\b(?:room|space)\s+(?:name|number)\b", normalized):
        # IFC stores the human room number in IfcSpace.Name.
        property_terms = ["name"]
    if operator == "argmax":
        area_superlative = bool(
            re.search(
                r"\b(?:largest|biggest|greatest\s+area|maximum\s+area)\b",
                normalized,
            )
        )
        superlative_property = "area" if area_superlative else "count"
        if superlative_property not in property_terms:
            property_terms.append(superlative_property)

    # A request for a finish/material/covering *in a named or typed space* is
    # a property lookup on the space.  Object mentions such as "floor" or
    # "wall" describe the property key; they must not turn the query into an
    # enumeration of every contained slab/opening/furnishing element.
    if space_property_terms and (room or room_names or space_types):
        target_kind = "space"
        object_links = []
        system_links = []
        target_roles = []
        target_domains = []
        target_names = []
        property_terms = space_property_terms

    # For grouped aggregations, a modifier attached to a graph-linked role is
    # a family/type qualifier rather than an exact instance-name constraint.
    # Example pattern: "most common suspended sensing type" where "sensing"
    # links to the role and "suspended" narrows its graph type/family.
    aggregation_qualifiers: list[str] = []
    if operator == "group_count":
        role_tokens = {
            token for role in target_roles for token in _signature(role)
        }
        qualifier_phrases = [
            normalized[start:end] for start, end in focus_spans
        ] or [
            link.text
            for link in object_links
            if link.source_field.startswith("partial:") and link.role
        ]
        for phrase in qualifier_phrases:
            phrase_tokens = list(_signature(phrase))
            qualifier_tokens = [
                token
                for token in phrase_tokens
                if token not in role_tokens
                and token not in _GENERIC_NOUNS
                and token not in {"family", "type", "kind", "class"}
            ]
            if role_tokens.intersection(phrase_tokens) and qualifier_tokens:
                aggregation_qualifiers.append(" ".join(qualifier_tokens))
        aggregation_qualifiers = _ordered_unique(aggregation_qualifiers)
        if aggregation_qualifiers:
            target_names = [
                name
                for name in target_names
                if not any(
                    normalize_question_text(name) == normalize_question_text(link.text)
                    for link in object_links
                    if link.source_field.startswith("partial:") and link.role
                )
            ]

    if target_kind is None and not actions:
        if room or room_names or any(token in _SPACE_HEADS for token in normalized.split()):
            target_kind = "space"
        elif any(token in _GENERIC_NOUNS for token in normalized.split()):
            target_kind = "object"

    target_family_terms = _ordered_unique(
        link.text for link in object_links if link.source_field == "family"
    )
    target_type_terms = _ordered_unique(
        link.text
        for link in object_links
        if link.source_field in {"type", "type_name", "object_type"}
    )
    if aggregation_qualifiers:
        if "family" in property_terms:
            target_family_terms = _ordered_unique(
                [*target_family_terms, *aggregation_qualifiers]
            )
        elif any(term in property_terms for term in ("type", "kind", "class")):
            target_type_terms = _ordered_unique(
                [*target_type_terms, *aggregation_qualifiers]
            )

    plan = QueryPlan(
        operator=operator,  # type: ignore[arg-type]
        mentions=_ordered_unique(link.text for link in linked_mentions),
        mention_links=linked_mentions,
        # A level/floor modifier remains a hard building scope even when it
        # follows a relational reference phrase.
        storey=_storey(normalized),
        room=room,
        room_names=room_names,
        target_kind=target_kind,
        target_role=target_roles[0] if target_roles else None,
        target_roles=target_roles,
        target_domain=target_domains[0] if len(target_domains) == 1 else None,
        target_space_type=space_types[0] if len(space_types) == 1 else None,
        target_name=target_names[0] if target_names else None,
        target_names=target_names,
        target_phrases=coordinated_target_phrases,
        target_family_terms=target_family_terms,
        target_type_terms=target_type_terms,
        target_keywords=_ordered_unique(
            link.text for link in object_links if link.source_field in {"role", "aliases", "synonyms"}
        ) or list(target_names),
        property_terms=property_terms,
        action_sequence=[action.action for action in actions],
        relation_references=relation_references,
        function_intents=function_intents,
        cardinality_policy=cardinality,
        scope_cardinality=scope_cardinality,  # type: ignore[arg-type]
        search_exhaustive=operator in {
            "list", "distinct", "count", "group_count", "argmax", "nearest",
            "all_matching", "unconnected",
        },
        requires_exhaustive=cardinality in {
            "all", "count", "distinct", "group_count", "argmax"
        },
    )

    if scenario_state_change is not None:
        # A room named before the shutdown/failure describes the source asset,
        # not a containment restriction on downstream impacted targets. A
        # later storey constraint such as "which Level 1 rooms" is preserved.
        source_prefix = normalized[: scenario_state_change.start()]
        room_is_source = bool(
            plan.room
            and re.search(
                rf"\b(?:room|space)\s+{re.escape(normalize_question_text(plan.room))}\b",
                source_prefix,
            )
        )
        named_space_is_source = any(
            link.kind == "space"
            and link.char_start >= 0
            and link.char_start < scenario_state_change.start()
            for link in plan.mention_links
        )
        if room_is_source or named_space_is_source:
            plan.room = None
            plan.room_names = []
            plan.target_space_type = None
        target_question = normalized[scenario_state_change.end():]
        if re.search(
            r"\b(?:which|what)\b[^?.;]*\b(?:rooms|spaces|components|fixtures|equipment)\b",
            target_question,
        ):
            plan.cardinality_policy = "all"
            plan.requires_exhaustive = True
        if not plan.target_domain and scenario_source_node_ids:
            source_domains = _ordered_unique(
                term.domain
                for term in schema.terms
                if term.node_id in scenario_source_node_ids and term.domain
            )
            if len(source_domains) == 1:
                plan.target_domain = source_domains[0]
                plan.mention_links.append(
                    MentionLink(
                        text=f"affected {source_domains[0]} components",
                        canonical=source_domains[0],
                        kind="object",
                        action_index=0,
                        domain=source_domains[0],
                        confidence=1.0,
                        source="graph",
                        source_field="scenario:source_domain",
                    )
                )

    if mentioned_system_links and len(actions) <= 1:
        member_request = (
            plan.target_kind == "object"
            or bool(object_links)
            or bool(function_intents)
            or any(
            _singular_token(token) in _GENERIC_NOUNS
            for _, phrase in phrases
            for token in normalize_question_text(phrase).split()
        )
        )
        plan.target_binding_mode = "system_members" if member_request else "system_entity"
        plan.target_kind = "object" if member_request else "system"

    build_staged_constraints(plan, question)
    binding_cardinalities = _binding_cardinalities(
        normalized, actions, plan.cardinality_policy
    )
    for binding in plan.action_bindings:
        binding.cardinality_policy = binding_cardinalities.get(
            binding.binding_index, plan.cardinality_policy
        )
        if scenario_state_change is not None and plan.cardinality_policy == "all":
            binding.cardinality_policy = "all"
        if binding.target_mode == "system_members":
            binding.cardinality_policy = "all"
    _semantic_budget(plan)
    plan.rationale = "Generic syntax and active graph-schema query compilation"
    return plan


def merge_query_plan(
    fallback: QueryPlan,
    proposed: QueryPlan,
    *,
    max_gnn_hops: int = 3,
    max_tog_depth: int = 4,
    max_tog_width: int = 5,
) -> QueryPlan:
    """Merge a structured fallback without weakening graph-grounded scopes."""

    result = replace(proposed)
    for field_name in (
        "target_kind",
        "target_domain",
        "target_space_type",
        "target_name",
    ):
        fallback_value = getattr(fallback, field_name)
        if fallback_value not in (None, "") or getattr(result, field_name) in (None, ""):
            setattr(result, field_name, fallback_value)

    # Exact scope can only originate from the question/graph linker.
    result.storey = fallback.storey
    result.room = fallback.room
    result.room_names = list(fallback.room_names)
    for field_name in (
        "property_terms",
        "target_roles",
        "target_names",
        "target_family_terms",
        "target_type_terms",
        "target_keywords",
        "mentions",
    ):
        setattr(
            result,
            field_name,
            _ordered_unique([*getattr(fallback, field_name), *getattr(result, field_name)]),
        )

    # Typed stages and graph links are deterministic evidence, not LLM prose.
    result.mention_links = list(fallback.mention_links)
    result.scope_predicates = list(fallback.scope_predicates)
    result.relation_references = list(fallback.relation_references)
    result.target_predicates = list(fallback.target_predicates)
    result.function_intents = list(fallback.function_intents)
    result.target_binding_mode = fallback.target_binding_mode
    result.cardinality_policy = fallback.cardinality_policy
    result.scope_cardinality = fallback.scope_cardinality
    result.search_exhaustive = fallback.search_exhaustive
    result.target_grouping = list(fallback.target_grouping)
    result.unresolved_slots = list(fallback.unresolved_slots)
    # Executable actions are syntax-grounded by the deterministic parser.  A
    # semantic planner may refine unresolved target slots, but it must not turn
    # a QA operator such as lookup/count/distinct into a robot action or invent
    # an action absent from the question.
    result.action_sequence = list(fallback.action_sequence)

    if fallback.operator in {
        "list", "distinct", "count", "group_count", "argmax", "nearest",
        "all_matching", "unconnected",
    } or result.operator == "lookup" and fallback.operator != "lookup":
        result.operator = fallback.operator
    result.requires_exhaustive = fallback.requires_exhaustive

    for field_name, minimum, maximum, fallback_value in (
        ("gnn_hops", 0, max_gnn_hops, fallback.gnn_hops),
        ("tog_depth", 1, max_tog_depth, fallback.tog_depth),
        ("tog_width", 1, max_tog_width, fallback.tog_width),
    ):
        try:
            value = max(minimum, min(maximum, int(getattr(result, field_name))))
        except (TypeError, ValueError):
            value = fallback_value
        setattr(result, field_name, value)

    valid_levels = {"object", "space", "system", "function"}
    result.gnn_retrieval_levels = _ordered_unique(
        level for level in result.gnn_retrieval_levels if level in valid_levels
    )
    if not result.gnn_retrieval_levels:
        result.gnn_retrieval_levels = list(fallback.gnn_retrieval_levels)

    merged_bindings = build_action_bindings(result)
    fallback_by_index = {
        binding.binding_index: binding for binding in fallback.action_bindings
    }
    proposed_by_index = {
        binding.binding_index: binding for binding in proposed.action_bindings
    }
    for binding in merged_bindings:
        protected = fallback_by_index.get(binding.binding_index)
        suggestion = proposed_by_index.get(binding.binding_index)
        if protected is not None:
            binding.source_stage = protected.source_stage
            binding.result_stage = protected.result_stage
            binding.target_roles = _ordered_unique(
                [*protected.target_roles, *binding.target_roles]
            )
            binding.target_names = _ordered_unique(
                [*protected.target_names, *binding.target_names]
            )
            binding.target_domains = _ordered_unique(
                [*protected.target_domains, *binding.target_domains]
            )
            binding.function_types = _ordered_unique(
                [*protected.function_types, *binding.function_types]
            )
            binding.system_categories = _ordered_unique(
                [*protected.system_categories, *binding.system_categories]
            )
            binding.cardinality_policy = protected.cardinality_policy
        # The structured planner may fill a genuinely unresolved semantic
        # slot, but it cannot replace an exact graph-linked constraint or the
        # deterministic cardinality policy.
        if suggestion is not None:
            if not binding.target_roles:
                binding.target_roles = _ordered_unique(suggestion.target_roles)
            if not binding.target_names:
                binding.target_names = _ordered_unique(suggestion.target_names)
            if not binding.target_domains:
                binding.target_domains = _ordered_unique(suggestion.target_domains)
            if not binding.function_types:
                binding.function_types = _ordered_unique(suggestion.function_types)
            if not binding.system_categories:
                binding.system_categories = _ordered_unique(
                    suggestion.system_categories
                )
            if binding.target_kind not in {"space", "object", "system"}:
                binding.target_kind = suggestion.target_kind
    result.action_bindings = merged_bindings
    if result.action_bindings and all(
        binding.target_kind in {"space", "object", "system"}
        and (
            binding.target_kind == "space"
            or binding.target_roles
            or binding.target_names
            or binding.target_domains
            or binding.function_types
            or binding.system_categories
        )
        for binding in result.action_bindings
    ):
        result.unresolved_slots = [
            slot for slot in result.unresolved_slots
            if (
                slot not in {"action_target_kind", "target_mention"}
                and not (
                    slot.startswith("action_binding:")
                    and slot.endswith(":target_mention")
                )
            )
        ]
    return result
