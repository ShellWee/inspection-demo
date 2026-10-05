from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import (
    ActionTargetBinding,
    EntityRef,
    GnnSubgraph,
    HierarchyContext,
    HierarchyEdge,
    HierarchyPath,
    HierarchyValidation,
    LogicalTargetGroup,
    OperatorResult,
    QueryPlan,
    RelationRef,
    TargetAudit,
    TargetValidation,
    TripleEvidence,
)

GUID_RE = re.compile(r"(?<![0-9A-Za-z_$])([0-9A-Za-z_$]{22})(?![0-9A-Za-z_$])")

ROLE_IFC_CLASS_ALLOWLIST: dict[str, frozenset[str]] = {
    "light_fixture": frozenset({"IfcFlowTerminal"}),
    "sprinkler": frozenset({"IfcFlowTerminal"}),
    "sanitary_fixture": frozenset({"IfcFlowTerminal", "IfcSanitaryTerminal"}),
    "furnishing": frozenset({"IfcFurnishingElement"}),
    "door": frozenset({"IfcDoor"}),
    "ceiling": frozenset({"IfcCovering"}),
    "outlet": frozenset({"IfcBuildingElementProxy"}),
    "switch": frozenset({"IfcBuildingElementProxy"}),
    "panel": frozenset({"IfcBuildingElementProxy", "IfcElectricDistributionBoard"}),
    "proxy_object": frozenset({"IfcBuildingElementProxy"}),
    "diffuser": frozenset({"IfcFlowTerminal", "IfcBuildingElementProxy"}),
    "air_terminal": frozenset({"IfcFlowTerminal", "IfcBuildingElementProxy"}),
    "fire_extinguisher": frozenset({
        "IfcBuildingElementProxy", "IfcFireSuppressionTerminal",
    }),
    "drain": frozenset({
        "IfcBuildingElementProxy", "IfcFlowTerminal", "IfcWasteTerminal",
    }),
    "electrical_device": frozenset({
        "IfcBuildingElementProxy", "IfcElectricAppliance",
    }),
    "duct": frozenset({"IfcBuildingElementProxy", "IfcFlowSegment"}),
    "water_supply": frozenset({
        "IfcFlowTerminal", "IfcSanitaryTerminal", "IfcBuildingElementProxy",
    }),
}

# A text-only role correction is appropriate only when the graph explicitly
# says that its classification is weak.  Missing confidence is not evidence of
# weakness: older artifacts may omit it while still carrying an authoritative
# IFC-class role.
ROLE_FALLBACK_MAX_CONFIDENCE = 0.5
UNKNOWN_DECLARED_ROLES = frozenset({
    "", "unknown", "unknown_mep", "unclassified", "proxy_object",
})

GENERIC_MATCH_TOKENS = {
    "all", "system", "fixture", "fixtures", "equipment", "element", "elements",
    "object", "objects", "inspect", "scan", "navigate", "level", "floor",
    "room", "space", "spaces", "type", "types", "family", "used",
    "and", "the",
}

SPACE_NAME_STOP_TOKENS = frozenset({
    "the", "a", "an", "in", "on", "at", "of", "to", "for", "and",
})

# Operational gating is deliberately expressed in IFC/domain terms rather
# than benchmark vocabulary.  A fault should normally bind to a component
# that participates in a building service, while a representation/container
# is retained only when no operational alternative exists.
OPERATIONAL_IFC_CLASSES = frozenset({
    "IfcDistributionElement",
    "IfcDistributionControlElement",
    "IfcDistributionFlowElement",
    "IfcEnergyConversionDevice",
    "IfcFlowController",
    "IfcFlowFitting",
    "IfcFlowMovingDevice",
    "IfcFlowSegment",
    "IfcFlowStorageDevice",
    "IfcFlowTerminal",
    "IfcFlowTreatmentDevice",
})
OPERATIONAL_IFC_CLASS_SUFFIXES = (
    "Actuator", "Appliance", "Controller", "Device", "Equipment", "Fan",
    "Instrument", "Outlet", "Pump", "Sensor", "Terminal", "Transformer",
    "Valve",
)
NON_OPERATIONAL_IFC_CLASSES = frozenset({
    "IfcAnnotation", "IfcCovering", "IfcFurnishingElement", "IfcVirtualElement",
})
OPERATIONAL_REQUEST_TOKENS = frozenset({
    "actuator", "appliance", "component", "controller", "device", "equipment",
    "fan", "fixture", "instrument", "outlet", "pump", "sensor", "terminal",
    "transformer", "valve",
})
REPRESENTATION_TOKENS = frozenset({
    "assembly", "cabinet", "casework", "ceiling", "enclosure", "furnishing",
    "housing", "placeholder", "representation", "symbol",
})
# IFC authoring practices frequently leave the intended asset role in family,
# type, or object names while assigning a generic/proxy class.  These aliases
# are domain vocabulary, not benchmark identifiers, and provide a conservative
# semantic fallback after exact role classification.
ROLE_SEMANTIC_TERMS: dict[str, tuple[str, ...]] = {
    "panel": ("panel", "switchboard"),
    "proxy_object": ("access panel",),
    "diffuser": ("diffuser", "chilled beam", "air terminal"),
    "air_terminal": ("diffuser", "chilled beam", "air terminal"),
    "fire_extinguisher": ("fire extinguisher", "fire protection cabinet"),
    "sanitary_fixture": (
        "sink", "floor drain", "safety shower", "emergency shower",
        "eyewash", "eye wash", "plumbing fixture",
    ),
    "drain": ("drain",),
    "electrical_device": ("projector", "display", "tv", "television"),
    "water_supply": ("gas manifold", "gas turret", "eyewash", "eye wash"),
}

# Some graph roles intentionally describe a broad maintainable class while a
# query asks for a narrower asset subtype.  The subtype may be recovered only
# when graph-backed object text explicitly names it and the IFC class remains
# compatible.  This is an ontology relation, not a permission to replace one
# authoritative peer role with another (for example, door -> panel).
ROLE_PARENT_RELATIONS: dict[str, frozenset[str]] = {
    "drain": frozenset({"sanitary_fixture"}),
}

DOMAIN_SEMANTIC_TERMS: dict[str, tuple[str, ...]] = {
    "hvac": ("hvac", "diffuser", "air terminal", "duct", "snorkel", "exhaust", "chilled beam"),
    "plumbing": ("plumbing", "pipe", "drain", "sink", "shower", "eyewash"),
    "electrical": ("electrical", "panel", "outlet", "switch", "receptacle", "transformer"),
    "fire_protection": ("fire protection", "fire extinguisher", "sprinkler", "fire alarm"),
    "structural": ("structural", "beam", "column", "footing"),
}

ACTION_TARGET_KINDS = frozenset({"space", "object", "system", "none"})

# Query-plan system labels intentionally use the graph ontology categories.
# These aliases are domain vocabulary and do not encode benchmark question IDs.
SYSTEM_CATEGORY_ALIASES: dict[str, str] = {
    "hvac": "hvac_system",
    "ventilation": "hvac_system",
    "plumbing": "plumbing_system",
    "drainage": "plumbing_system",
    "electrical": "electrical_system",
    "power": "electrical_system",
    "fire_protection": "fire_protection_system",
    "fire_suppression": "fire_protection_system",
    "structural": "structural_system",
}

# A stored space label remains useful under incomplete BIM, but a weak
# geometry/heuristic classification is not an explicit contradiction.  The
# tri-state matcher below retains such scopes as ``unknown`` so a separate,
# positive graph predicate can still ground the requested target.
SPACE_TYPE_LOW_CONFIDENCE_MAX = 0.7
SPACE_TYPE_UNCERTAIN_SOURCES = frozenset({
    "candidate",
    "geometry",
    "geometry_rule",
    "heuristic",
    "inferred",
    "missing",
    "unknown",
})

# Singular surface nouns can denote one logical building fabric assembled from
# several IFC products, but only graph-declared grouping/aggregation is strong
# enough to prove that membership.  Family prevalence alone is intentionally
# excluded because it would turn a benchmark-specific majority into ontology.
CONTINUOUS_SURFACE_ROLES = frozenset({
    "ceiling", "floor", "roof", "wall",
})

LOGICAL_GROUP_ACTIONS = frozenset({"inspect", "scan"})

STRICT_SPACE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "conference": (re.compile(r"\bconf(?:erence)?\b", re.IGNORECASE),),
    "lab": (
        re.compile(r"\blab(?:oratory)?\b", re.IGNORECASE),
        re.compile(r"\bfuture\s+lab\b", re.IGNORECASE),
    ),
    "office": (
        re.compile(
            r"\b(?:office|faculty|dean|director|department\s*head|admin|pi)\b",
            re.IGNORECASE,
        ),
    ),
    "workshop": (re.compile(r"\b(?:workshop|research\s+shop)\b", re.IGNORECASE),),
}


class IfcGraphBackend:
    def __init__(self, index_path: str | Path, include_evidence: Sequence[str]) -> None:
        self.index_path = Path(index_path).resolve()
        self.connection = sqlite3.connect(
            f"file:{self.index_path}?mode=ro", uri=True, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.include_evidence = tuple(include_evidence)
        self.meta = dict(self.connection.execute("SELECT key, value FROM graph_meta"))
        self._node_cache: dict[str, EntityRef] = {}
        # These caches are intentionally backend-local.  A backend owns one
        # immutable graph artifact and one evidence policy, so cached ontology
        # definitions and function paths can never leak across graphs/runs.
        self._function_ontology_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            tuple[tuple[str, bool, tuple[str, ...]], ...],
        ] = {}
        self._function_evidence_index_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            tuple[
                dict[str, tuple[str, ...]],
                dict[str, tuple[str, ...]],
            ],
        ] = {}
        self._target_function_evidence_cache: dict[
            tuple[str, tuple[str, ...], tuple[str, ...]], tuple[str, ...]
        ] = {}
        self._system_participant_ids_cache: set[str] | None = None
        self._authoritative_containment_cache: dict[str, frozenset[str]] | None = None

    def close(self) -> None:
        self.connection.close()

    @property
    def graph_hash(self) -> str:
        # Different inspection-graph representations may be built from the
        # same IFC source.  Cache keys and response lineage must identify the
        # actual graph, while legacy indexes fall back to the source hash.
        return self.meta.get("representation_graph_hash") or self.meta.get(
            "source_hash", ""
        )

    @property
    def graph_schema(self) -> str:
        return self.meta.get("schema_version", "")

    def _entity(self, row: sqlite3.Row, score: float = 0.0, reason: str = "") -> EntityRef:
        node_id = str(row["node_id"])
        cached = self._node_cache.get(node_id)
        if cached is None:
            label = row["long_name"] or row["name"] or row["type_name"] or row["ifc_class"] or node_id
            metadata = {
                key: row[key]
                for key in (
                    "name", "long_name", "family", "type_name", "object_type", "tag",
                    "storey", "level", "category", "domain", "role", "space_type",
                    "system_category", "function_type",
                )
                if key in row.keys() and row[key] not in (None, "")
            }
            if "attrs_json" in row.keys() and row["attrs_json"]:
                try:
                    attrs = json.loads(row["attrs_json"])
                except (TypeError, json.JSONDecodeError):
                    attrs = {}
                for key in (
                    "classification_confidence",
                    "classification_source",
                    "level",
                    "action_target_kind",
                    "actionability_source",
                    "actionability_reason",
                ):
                    if attrs.get(key) not in (None, ""):
                        metadata[key] = attrs[key]
            cached = EntityRef(
                node_id=node_id,
                label=str(label),
                global_id=row["global_id"],
                ifc_class=row["ifc_class"],
                kind="entity",
                metadata=metadata,
            )
            self._node_cache[node_id] = cached
        return EntityRef(
            node_id=cached.node_id,
            label=cached.label,
            global_id=cached.global_id,
            ifc_class=cached.ifc_class,
            kind=cached.kind,
            score=score,
            match_reason=reason,
            metadata=dict(cached.metadata),
        )

    def get_node(self, node_id: str) -> EntityRef | None:
        cached = self._node_cache.get(str(node_id))
        if cached is not None:
            return EntityRef(
                node_id=cached.node_id,
                label=cached.label,
                global_id=cached.global_id,
                ifc_class=cached.ifc_class,
                kind=cached.kind,
                score=cached.score,
                match_reason=cached.match_reason,
                metadata=dict(cached.metadata),
            )
        row = self.connection.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        return self._entity(row) if row else None

    @staticmethod
    def _search_text(entity: EntityRef) -> str:
        return " ".join(
            str(value)
            for value in (
                entity.label,
                entity.metadata.get("name"),
                entity.metadata.get("long_name"),
                entity.metadata.get("family"),
                entity.metadata.get("type_name"),
                entity.metadata.get("object_type"),
                entity.metadata.get("role"),
                entity.metadata.get("space_type"),
            )
            if value
        )

    @staticmethod
    def _name_search_text(entity: EntityRef) -> str:
        """Text that can satisfy an explicit name/family/type constraint."""

        return " ".join(
            str(value)
            for value in (
                entity.label,
                entity.metadata.get("name"),
                entity.metadata.get("long_name"),
                entity.metadata.get("family"),
                entity.metadata.get("type_name"),
                entity.metadata.get("object_type"),
                entity.metadata.get("tag"),
            )
            if value
        )

    @staticmethod
    def action_target_kind(entity: EntityRef) -> str:
        """Return the executable target kind independently of graph level.

        New inspection-graph artifacts carry the explicit contract.  The
        conservative fallback keeps legacy IFC indexes usable without making
        generic groups or abstract function nodes executable.
        """
        declared = str(entity.metadata.get("action_target_kind") or "").lower()
        if declared in ACTION_TARGET_KINDS:
            return declared
        if entity.global_id and entity.ifc_class == "IfcSpace":
            return "space"
        if entity.global_id and entity.metadata.get("category") == "object":
            return "object"
        if entity.global_id and entity.ifc_class in {
            "IfcSystem", "IfcDistributionSystem", "IfcBuildingSystem"
        }:
            return "system"
        # Legacy IFC indexes did not persist the actionability contract.  A
        # GUID-bearing physical IFC product remains a valid object target,
        # while spatial containers, systems/groups and ontology nodes stay
        # conservative.  New frozen graphs never depend on this fallback.
        if (
            entity.global_id
            and str(entity.ifc_class or "").startswith("Ifc")
            and entity.ifc_class not in {
                "IfcProject",
                "IfcSite",
                "IfcBuilding",
                "IfcBuildingStorey",
                "IfcSpatialZone",
                "IfcZone",
                "IfcGroup",
            }
        ):
            return "object"
        return "none"

    @classmethod
    def action_compatible(
        cls,
        entity: EntityRef,
        *,
        action: str | None = None,
        requested_kind: str | None = None,
    ) -> bool:
        kind = cls.action_target_kind(entity)
        normalized_action = str(action or "").lower()
        normalized_kind = str(requested_kind or "").lower()
        if kind == "none":
            return False
        if normalized_kind in {"space", "object", "system"} and kind != normalized_kind:
            return False
        if normalized_action in {"navigate", "nav"}:
            return kind == "space"
        if normalized_action in {"inspect", "scan"}:
            return kind in {"object", "system"}
        return True

    @staticmethod
    def _system_categories(plan: QueryPlan) -> list[str]:
        values = [
            value
            for predicate in plan.target_predicates
            if predicate.predicate == "system" and predicate.required
            for value in predicate.values
        ]
        values.extend(
            category
            for binding in plan.action_bindings
            for category in binding.system_categories
        )
        if (
            plan.target_domain
            and getattr(plan, "target_binding_mode", "auto") != "auto"
        ):
            values.append(SYSTEM_CATEGORY_ALIASES.get(plan.target_domain, plan.target_domain))
        return list(dict.fromkeys(str(value) for value in values if value))

    @staticmethod
    def _role_class_allowed(entity: EntityRef, roles: Sequence[str]) -> bool:
        entity_role = str(entity.metadata.get("role") or "")
        if entity_role in roles and entity_role in ROLE_IFC_CLASS_ALLOWLIST:
            if entity.ifc_class in ROLE_IFC_CLASS_ALLOWLIST[entity_role]:
                return True
            # IFC class is often an authoring/container choice rather than the
            # maintainable role.  Permit a GUID-backed L1 object outside the
            # conventional class only when its graph-assigned role is also
            # supported by the versioned role ontology in its own family/type
            # text.  This is one rule for every role, not an asset-specific
            # exception.
            text = IfcGraphBackend._normalized_text(
                IfcGraphBackend._search_text(entity)
            )
            ontology_supported = any(
                IfcGraphBackend._semantic_phrase_match(text, phrase)
                for phrase in ROLE_SEMANTIC_TERMS.get(entity_role, ())
            )
            return bool(
                ontology_supported
                and IfcGraphBackend.action_target_kind(entity) == "object"
            )
        constrained = [
            ROLE_IFC_CLASS_ALLOWLIST[role]
            for role in roles if role in ROLE_IFC_CLASS_ALLOWLIST
        ]
        if not constrained or entity_role:
            return True
        return any(entity.ifc_class in allowed for allowed in constrained)

    @staticmethod
    def _role_fallback_permitted(entity: EntityRef) -> bool:
        """Return whether graph text may repair the declared asset role."""

        declared = str(entity.metadata.get("role") or "").strip().lower()
        if declared in UNKNOWN_DECLARED_ROLES:
            return True
        raw_confidence = entity.metadata.get("classification_confidence")
        if raw_confidence in (None, ""):
            return False
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return False
        return math.isfinite(confidence) and confidence <= ROLE_FALLBACK_MAX_CONFIDENCE

    @staticmethod
    def _role_fallback_class_allowed(entity: EntityRef, role: str) -> bool:
        """Require an auditable IFC-class contract for a text role repair."""

        allowed = ROLE_IFC_CLASS_ALLOWLIST.get(role)
        return bool(
            allowed
            and entity.ifc_class in allowed
            and IfcGraphBackend.action_target_kind(entity) == "object"
        )

    @classmethod
    def _semantic_role_match(cls, entity: EntityRef, roles: Sequence[str]) -> bool:
        entity_role = str(entity.metadata.get("role") or "")
        if entity_role in roles:
            return cls._role_class_allowed(entity, roles)
        # A requested ontology child can refine an authoritative broad role,
        # but only when the object's own graph attributes explicitly support
        # the child and the IFC/actionability contract also permits it.
        text = cls._normalized_text(cls._name_search_text(entity))
        for role in roles:
            if entity_role not in ROLE_PARENT_RELATIONS.get(role, frozenset()):
                continue
            if not cls._role_fallback_class_allowed(entity, role):
                continue
            if any(
                cls._semantic_phrase_match(text, phrase)
                for phrase in ROLE_SEMANTIC_TERMS.get(role, ())
            ):
                return True
        # Do not let a coincidental label token overwrite an authoritative graph
        # role.  Text repair is reserved for missing/unknown or explicitly
        # low-confidence classifications, and the requested role must remain
        # compatible with the object's IFC class.
        if not cls._role_fallback_permitted(entity):
            return False
        for role in roles:
            for phrase in ROLE_SEMANTIC_TERMS.get(role, ()):
                if (
                    cls._semantic_phrase_match(text, phrase)
                    and cls._role_fallback_class_allowed(entity, role)
                ):
                    return True
        return False

    @classmethod
    def _semantic_domain_match(cls, entity: EntityRef, domain: str) -> bool:
        if str(entity.metadata.get("domain") or "") == domain:
            return True
        text = cls._normalized_text(cls._name_search_text(entity))
        return any(
            cls._semantic_phrase_match(text, phrase)
            for phrase in DOMAIN_SEMANTIC_TERMS.get(domain, ())
        )

    @classmethod
    def semantic_profile(cls, entity: EntityRef) -> dict[str, Any]:
        """Return auditable role/domain normalization from graph attributes.

        IFC proxies frequently carry their useful semantics in family, type or
        object names.  The profile never mutates the graph classification; it
        records which graph-backed text supports a conservative override.
        """
        text = cls._normalized_text(cls._search_text(entity))
        declared_role = str(entity.metadata.get("role") or "")
        declared_domain = str(entity.metadata.get("domain") or "")
        inferred_roles = sorted(
            role
            for role, phrases in ROLE_SEMANTIC_TERMS.items()
            if cls._role_fallback_permitted(entity)
            and cls._role_fallback_class_allowed(entity, role)
            and any(cls._semantic_phrase_match(text, phrase) for phrase in phrases)
        )
        inferred_domains = sorted(
            domain
            for domain, phrases in DOMAIN_SEMANTIC_TERMS.items()
            if any(cls._semantic_phrase_match(text, phrase) for phrase in phrases)
        )
        return {
            "declared_role": declared_role or None,
            "declared_domain": declared_domain or None,
            "normalized_roles": list(
                dict.fromkeys(([declared_role] if declared_role else []) + inferred_roles)
            ),
            "normalized_domains": list(
                dict.fromkeys(([declared_domain] if declared_domain else []) + inferred_domains)
            ),
            "classification_confidence": entity.metadata.get("classification_confidence"),
            "source": "graph_attributes:name|long_name|family|type_name|object_type",
        }

    @staticmethod
    def _normalized_text(value: object) -> str:
        # IFC authoring labels commonly use '&' where natural-language job
        # plans use 'and' (for example, "Automation & Robotics Lab").
        # Canonicalizing it here keeps exact scope resolution deterministic.
        rendered = unicodedata.normalize("NFKC", str(value or ""))
        # BIM family/type identifiers often concatenate words (FireGuard,
        # VAVBoxType).  Split both ordinary CamelCase and acronym-to-word
        # boundaries before case folding; hyphens/underscores remain ordinary
        # token separators and numeric identity suffixes are preserved.
        rendered = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", rendered)
        rendered = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", rendered)
        rendered = rendered.replace("&", " and ").casefold()
        return " ".join(re.findall(r"[a-z0-9]+", rendered))

    @classmethod
    def _semantic_phrase_match(cls, normalized_text: str, phrase: str) -> bool:
        normalized_phrase = cls._normalized_text(phrase)
        if not normalized_phrase:
            return False
        if f" {normalized_phrase} " in f" {normalized_text} ":
            return True
        text_tokens = set(normalized_text.split())
        phrase_tokens = cls._meaningful_tokens([normalized_phrase])
        return bool(phrase_tokens) and phrase_tokens.issubset(text_tokens)

    @classmethod
    def _normalized_space_name(cls, value: object) -> str:
        """Normalize a graph space label without discarding identity suffixes."""
        aliases = {"laboratory": "lab", "rm": "room"}
        tokens = [
            aliases.get(token, token)
            for token in cls._normalized_text(value).split()
            if token not in SPACE_NAME_STOP_TOKENS
        ]
        if (
            len(tokens) == 2
            and tokens[0] in {"room", "space"}
            and any(character.isdigit() for character in tokens[1])
        ):
            tokens = tokens[1:]
        return " ".join(tokens)

    @classmethod
    def _space_name_tokens(cls, value: object) -> tuple[str, ...]:
        # Unlike ``_meaningful_tokens``, single-letter and numeric suffixes are
        # intentional identity tokens here (for example, Lab A vs Lab B).
        return tuple(cls._normalized_space_name(value).split())

    @classmethod
    def _named_space_matches(
        cls,
        spaces: Sequence[EntityRef],
        requested_names: Sequence[str],
    ) -> set[str]:
        """Resolve each space mention exact-first, then conservatively fuzzy.

        Fuzzy matches are considered only when that individual mention has no
        exact graph label.  This prevents a named entity from expanding to a
        sibling whose label differs only by a short identity suffix.
        """
        result: set[str] = set()
        for requested in requested_names:
            wanted = cls._normalized_space_name(requested)
            if not wanted:
                continue
            labels_by_id = {
                space.node_id: {
                    cls._normalized_space_name(space.metadata.get("name", "")),
                    cls._normalized_space_name(space.metadata.get("long_name", "")),
                    cls._normalized_space_name(space.label),
                }
                for space in spaces
            }
            exact = {
                node_id for node_id, labels in labels_by_id.items()
                if wanted in labels
            }
            if exact:
                result.update(exact)
                continue
            wanted_tokens = set(cls._space_name_tokens(wanted))
            if not wanted_tokens:
                continue
            result.update(
                node_id
                for node_id, labels in labels_by_id.items()
                if any(wanted_tokens.issubset(set(label.split())) for label in labels if label)
            )
        return result

    @classmethod
    def _entity_name_match(
        cls,
        entities: Sequence[EntityRef],
        requested_names: Sequence[str],
    ) -> bool:
        """Exact-first graph-label match for a nested ``contains_name`` join."""
        for requested in requested_names:
            wanted = cls._normalized_text(requested)
            if not wanted:
                continue
            labels = {
                cls._normalized_text(value)
                for entity in entities
                for value in (
                    entity.label,
                    entity.metadata.get("name"),
                    entity.metadata.get("long_name"),
                    entity.metadata.get("family"),
                    entity.metadata.get("type_name"),
                    entity.metadata.get("object_type"),
                )
                if value
            }
            if wanted in labels:
                return True
            wanted_tokens = {
                cls._match_token(token)
                for token in wanted.split()
                if token not in SPACE_NAME_STOP_TOKENS
            }
            if wanted_tokens and any(
                wanted_tokens.issubset(
                    {cls._match_token(token) for token in label.split()}
                )
                for label in labels
            ):
                return True
        return False

    @staticmethod
    def _has_specific_space_scope(plan: QueryPlan) -> bool:
        """Whether object/system targets must be tied to selected spaces.

        A storey is already a direct node attribute and must not be converted
        into containment through every space on that storey.  IFC doors and
        other products are often storey-assigned without an IfcSpace
        containment edge; treating storey-only scope as room scope silently
        removes otherwise valid targets.
        """
        return bool(
            plan.room
            or plan.room_names
            or plan.target_space_type
            or any(
                predicate.predicate != "storey"
                for predicate in plan.scope_predicates
            )
        )

    @classmethod
    def _meaningful_tokens(cls, values: Sequence[str]) -> set[str]:
        return {
            token
            for value in values
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if len(token) > 2 and token not in GENERIC_MATCH_TOKENS
        }

    @classmethod
    def _match_token(cls, token: str) -> str:
        """Conservative morphology for graph-label constraint matching."""
        if len(token) > 5 and token.endswith("ing"):
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

    @classmethod
    def _constraint_tokens(cls, value: str) -> tuple[str, ...]:
        """Tokens used by graph entity constraints, preserving short names."""
        return tuple(
            cls._match_token(token)
            for token in cls._normalized_text(value).split()
            if token not in GENERIC_MATCH_TOKENS
        )

    @classmethod
    def _matches_family_type_name(cls, plan: QueryPlan, entity: EntityRef) -> bool:
        groups = [
            plan.target_names or ([plan.target_name] if plan.target_name else []),
            plan.target_family_terms,
            plan.target_type_terms,
            plan.target_keywords,
        ]
        terms = [value for group in groups for value in group if value]
        if not any(cls._constraint_tokens(term) for term in terms):
            return True
        text = cls._normalized_text(cls._name_search_text(entity))
        text_tokens = {cls._match_token(token) for token in text.split()}
        requested_roles = list(
            plan.target_roles or ([plan.target_role] if plan.target_role else [])
        )
        role_tokens = {
            cls._match_token(token)
            for role in requested_roles
            for token in cls._normalized_text(role).split()
        }
        role_matches = bool(
            requested_roles and cls._semantic_role_match(entity, requested_roles)
        )
        # Multiple target phrases (for example drain + sink or outlet + switch)
        # are a union.  Each phrase itself remains an AND of meaningful tokens.
        for term in terms:
            phrase_tokens = set(cls._constraint_tokens(term))
            # A role-bearing generic head (``lighting`` in ``pendant
            # lighting``) is already certified by the role constraint.  Only
            # the remaining family/type modifier must occur in the authored
            # graph label.
            effective_tokens = (
                phrase_tokens - role_tokens if role_matches else phrase_tokens
            )
            if phrase_tokens and (
                effective_tokens.issubset(text_tokens)
                and (effective_tokens or role_matches)
            ):
                return True
        return False

    @staticmethod
    def _uses_global_name_constraint(plan: QueryPlan) -> bool:
        target_bindings = [
            binding for binding in plan.action_bindings
            if binding.target_kind != "space" or binding.result_stage != "scope"
        ]
        # In a multi-action request, names belong to their individual binding.
        # Applying the union as one SQL predicate would require a panel to also
        # be an AV display (or vice versa).
        return len(target_bindings) <= 1

    def _room_scope_ids(
        self, plan: QueryPlan, seeds: Sequence[EntityRef]
    ) -> list[str]:
        if plan.scope_predicates:
            return self._scope_space_ids(plan, seeds)
        spaces = [
            self._entity(row)
            for row in self.connection.execute(
                "SELECT * FROM nodes WHERE ifc_class='IfcSpace' ORDER BY node_id"
            )
        ]
        by_number: set[str] = set()
        if plan.room:
            wanted = self._normalized_text(plan.room)
            by_number = {
                item.node_id
                for item in spaces
                if self._normalized_text(item.metadata.get("name", "")) == wanted
            }
        by_name: set[str] = set()
        if plan.room_names:
            by_name = self._named_space_matches(spaces, plan.room_names)
        if by_number and by_name:
            overlap = by_number & by_name
            return sorted(overlap or by_number)
        if by_number or by_name:
            return sorted(by_number | by_name)
        if plan.target_space_type:
            return sorted(
                space.node_id
                for space in spaces
                if (
                    (not plan.storey or str(
                        space.metadata.get("storey") or ""
                    ).lower() == plan.storey.lower())
                    and self._space_type_status(
                        space, plan.target_space_type
                    ) != "fail"
                )
            )
        generic_space_labels = {
            "office", "lab", "laboratory", "conference", "classroom",
            "corridor", "stair", "kitchenette", "restroom", "lobby",
            "space", "room",
        }
        return [
            seed.node_id
            for seed in seeds
            if seed.ifc_class == "IfcSpace" and seed.score >= 0.9
            and self._normalized_text(seed.label) not in generic_space_labels
        ]

    def _scope_space_ids(
        self, plan: QueryPlan, seeds: Sequence[EntityRef]
    ) -> list[str]:
        """Execute ordered spatial predicates before selecting action targets."""
        spaces = [
            self._entity(row)
            for row in self.connection.execute(
                "SELECT * FROM nodes WHERE ifc_class='IfcSpace' ORDER BY node_id"
            )
        ]
        candidates = {space.node_id: space for space in spaces}
        has_selector = False
        has_identity_selector = False
        for predicate in plan.scope_predicates:
            values = [self._normalized_text(value) for value in predicate.values if value]
            if predicate.predicate == "storey":
                has_selector = True
                candidates = {
                    node_id: space for node_id, space in candidates.items()
                    if self._normalized_text(space.metadata.get("storey", "")) in values
                }
            elif predicate.predicate == "space_number":
                has_selector = True
                has_identity_selector = True
                candidates = {
                    node_id: space for node_id, space in candidates.items()
                    if self._normalized_text(space.metadata.get("name", "")) in values
                }
            elif predicate.predicate == "space_name":
                has_selector = True
                has_identity_selector = True
                matched_ids = self._named_space_matches(
                    list(candidates.values()), predicate.values
                )
                candidates = {
                    node_id: space for node_id, space in candidates.items()
                    if node_id in matched_ids
                }
            elif predicate.predicate == "space_type":
                has_selector = True
                outcomes = {
                    node_id: (
                        "pass"
                        if any(
                            self._space_type_status(space, value) == "pass"
                            for value in predicate.values
                        )
                        else "unknown"
                        if any(
                            self._space_type_status(space, value) == "unknown"
                            for value in predicate.values
                        )
                        else "fail"
                    )
                    for node_id, space in candidates.items()
                }
                # If an exact name/number already identifies a positively
                # typed space, do not union weakly classified same-number
                # siblings.  Broad storey/type collections still retain
                # unknowns for incomplete-BIM reasoning.
                allowed = (
                    {"pass"}
                    if has_identity_selector and "pass" in outcomes.values()
                    else {"pass", "unknown"}
                )
                candidates = {
                    node_id: space
                    for node_id, space in candidates.items()
                    if outcomes[node_id] in allowed
                }
            elif predicate.predicate in {
                "contains_role", "contains_domain", "contains_name"
            }:
                has_selector = True
                retained: dict[str, EntityRef] = {}
                for node_id, space in candidates.items():
                    child_ids = self._contained_targets([node_id])
                    if not child_ids:
                        continue
                    placeholders = ",".join("?" for _ in child_ids)
                    children = [
                        self._entity(row)
                        for row in self.connection.execute(
                            f"SELECT * FROM nodes WHERE node_id IN ({placeholders})",
                            tuple(sorted(child_ids)),
                        )
                    ]
                    if predicate.predicate == "contains_role":
                        matched = any(
                            self._semantic_role_match(child, predicate.values)
                            for child in children
                        )
                    elif predicate.predicate == "contains_domain":
                        matched = any(
                            any(
                                self._semantic_domain_match(child, domain)
                                for domain in predicate.values
                            )
                            for child in children
                        )
                    else:
                        matched = self._entity_name_match(children, predicate.values)
                    if matched:
                        retained[node_id] = space
                candidates = retained
            elif predicate.predicate == "space_function":
                has_selector = True
                function_space_ids: set[str] = set()
                ontology_constraints: dict[str, dict[str, list[str]]] = {}
                if predicate.values:
                    placeholders = ",".join("?" for _ in predicate.values)
                    evidence_placeholders = ",".join("?" for _ in self.include_evidence)
                    rows = self.connection.execute(
                        f"""
                        SELECT DISTINCT e.target FROM nodes f
                        JOIN edges e ON e.source=f.node_id
                        JOIN nodes s ON s.node_id=e.target
                        WHERE f.function_type IN ({placeholders})
                          AND s.ifc_class='IfcSpace'
                          AND e.relation IN ('requires_inspection_of','affects')
                          AND e.evidence_type IN ({evidence_placeholders})
                        """,
                        (*predicate.values, *self.include_evidence),
                    )
                    function_space_ids = {str(row["target"]) for row in rows}
                    constraint_rows = self.connection.execute(
                        f"""
                        SELECT f.node_id, v.predicate, v.value_text
                        FROM nodes f JOIN node_values v ON v.node_id=f.node_id
                        WHERE f.function_type IN ({placeholders})
                          AND v.predicate IN (
                            'property.target_space_types',
                            'property.target_name_terms'
                          )
                        ORDER BY f.node_id, v.predicate, v.value_text
                        """,
                        tuple(predicate.values),
                    )
                    for row in constraint_rows:
                        function_constraints = ontology_constraints.setdefault(
                            str(row["node_id"]),
                            {"space_types": [], "name_terms": []},
                        )
                        key = (
                            "space_types"
                            if row["predicate"] == "property.target_space_types"
                            else "name_terms"
                        )
                        function_constraints[key].append(str(row["value_text"]))

                # Intensional function definitions live in the active graph,
                # not in Python.  Constraints within one function node are
                # conjunctive by dimension (type AND name); values within a
                # dimension are alternatives.  Separate function nodes remain
                # alternatives, like separate materialized typed paths.
                for node_id, space in candidates.items():
                    for constraints in ontology_constraints.values():
                        space_types = constraints["space_types"]
                        name_terms = constraints["name_terms"]
                        type_match = not space_types or any(
                            self._strict_space_match(space, space_type)
                            for space_type in space_types
                        )
                        name_match = not name_terms or self._entity_name_match(
                            [space], name_terms
                        )
                        if (space_types or name_terms) and type_match and name_match:
                            function_space_ids.add(node_id)
                            break
                candidates = {
                    node_id: space for node_id, space in candidates.items()
                    if node_id in function_space_ids
                }
            elif predicate.predicate == "argmax_area" and candidates:
                has_selector = True
                ids = sorted(candidates)
                placeholders = ",".join("?" for _ in ids)
                rows = self.connection.execute(
                    f"""
                    SELECT node_id, MAX(value_num) AS area FROM node_values
                    WHERE node_id IN ({placeholders}) AND value_num IS NOT NULL
                      AND lower(predicate) LIKE '%area%'
                    GROUP BY node_id
                    """,
                    tuple(ids),
                )
                areas = {str(row["node_id"]): float(row["area"]) for row in rows}
                if areas:
                    maximum = max(areas.values())
                    candidates = {
                        node_id: space for node_id, space in candidates.items()
                        if abs(areas.get(node_id, float("-inf")) - maximum) < 1e-9
                    }
        if not has_selector:
            return [
                seed.node_id for seed in seeds
                if seed.ifc_class == "IfcSpace" and seed.score >= 0.9
            ]
        return sorted(candidates)

    @classmethod
    def _strict_space_match(cls, entity: EntityRef, space_type: str) -> bool:
        patterns = STRICT_SPACE_PATTERNS.get(space_type)
        if not patterns:
            return entity.metadata.get("space_type") == space_type
        # Do not trust the stored classification here: an index built with an
        # older broad taxonomy may already say "lab" for a workshop.  Reapply
        # the strict rules to raw labels/properties at query time.
        text = " ".join(
            str(value) for value in (
                entity.label,
                entity.metadata.get("name"),
                entity.metadata.get("long_name"),
                entity.metadata.get("object_type"),
            ) if value
        )
        return any(pattern.search(text) for pattern in patterns)

    @classmethod
    def _space_type_status(cls, entity: EntityRef, space_type: str) -> str:
        """Return pass/fail/unknown for one graph-backed space-type claim.

        Raw labels and exact graph taxonomy are positive evidence.  A
        conflicting high-confidence classification is a hard failure.  Weak
        geometry/heuristic classifications, missing labels, and explicitly
        low-confidence values remain unknown rather than being promoted to the
        requested type or discarded as contradictions.
        """

        if cls._strict_space_match(entity, space_type):
            return "pass"
        declared = cls._normalized_text(entity.metadata.get("space_type", ""))
        if not declared or declared in {"unknown", "unclassified"}:
            return "unknown"
        source = cls._normalized_text(
            entity.metadata.get("classification_source", "")
        ).replace(" ", "_")
        raw_confidence = entity.metadata.get("classification_confidence")
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = None
        if (
            source in SPACE_TYPE_UNCERTAIN_SOURCES
            or (
                confidence is not None
                and math.isfinite(confidence)
                and confidence <= SPACE_TYPE_LOW_CONFIDENCE_MAX
            )
        ):
            return "unknown"
        return "fail"

    def _space_scope_status(self, plan: QueryPlan, space_id: str) -> str:
        """Evaluate every requested space-type constraint for one scope node."""

        space = self.get_node(space_id)
        if space is None or space.ifc_class != "IfcSpace":
            return "unknown"
        requested = [
            value
            for predicate in plan.scope_predicates
            if predicate.predicate == "space_type"
            for value in predicate.values
        ]
        if plan.target_space_type:
            requested.append(plan.target_space_type)
        requested = list(dict.fromkeys(value for value in requested if value))
        if not requested:
            return "pass"
        outcomes = [
            self._space_type_status(space, value) for value in requested
        ]
        if "fail" in outcomes:
            return "fail"
        if "unknown" in outcomes:
            return "unknown"
        return "pass"

    def all_node_ids(self) -> set[str]:
        return {
            str(row["node_id"])
            for row in self.connection.execute("SELECT node_id FROM nodes")
        }

    @staticmethod
    def _hierarchy_authority(provenance: str) -> int:
        return {"explicit": 0, "inferred": 1, "candidate": 2}.get(provenance, 3)

    def _hierarchy_parent_rows(self, node_id: str) -> list[sqlite3.Row]:
        """Return spatial and typed inspection parents in stored edge direction."""
        evidence_placeholders = ",".join("?" for _ in self.include_evidence)
        rows = list(
            self.connection.execute(
                f"""
                SELECT e.source, e.target, e.relation, e.evidence_type, e.confidence,
                       s.ifc_class AS source_ifc_class,
                       coalesce(s.long_name, s.name, s.type_name, s.ifc_class, s.node_id) AS source_label,
                       t.ifc_class AS target_ifc_class,
                       coalesce(t.long_name, t.name, t.type_name, t.ifc_class, t.node_id) AS target_label,
                       CASE
                         WHEN e.relation='contains' AND e.target=? THEN e.source
                         WHEN e.relation='part_of' AND e.source=? THEN e.target
                         WHEN e.relation='assigned_to_system' AND e.source=? THEN e.target
                         WHEN e.relation='requires_inspection_of' AND e.target=? THEN e.source
                         WHEN e.relation='related_to_system' AND e.target=? THEN e.source
                         WHEN e.relation='serves' AND e.source=? THEN e.target
                       END AS parent_id
                FROM edges e
                JOIN nodes s ON s.node_id=e.source
                JOIN nodes t ON t.node_id=e.target
                WHERE e.evidence_type IN ({evidence_placeholders})
                  AND (
                    (e.relation='contains' AND e.target=?)
                    OR (e.relation='part_of' AND e.source=?)
                    OR (e.relation='assigned_to_system' AND e.source=?)
                    OR (e.relation='requires_inspection_of' AND e.target=?)
                    OR (e.relation='related_to_system' AND e.target=?)
                    OR (e.relation='serves' AND e.source=?)
                  )
                """,
                (
                    node_id, node_id, node_id, node_id, node_id, node_id,
                    *self.include_evidence,
                    node_id, node_id, node_id, node_id, node_id, node_id,
                ),
            )
        )
        rows.sort(
            key=lambda row: (
                self._hierarchy_authority(str(row["evidence_type"])),
                -float(row["confidence"]),
                str(row["parent_id"]),
                str(row["relation"]),
            )
        )
        return rows

    def build_hierarchy_context(
        self,
        relevant_node_ids: Sequence[str],
        *,
        max_paths: int = 100,
        max_depth: int = 6,
    ) -> HierarchyContext:
        """Build a bounded, auditable root-to-target hierarchy context."""
        spatial_counts = {
            str(row["ifc_class"]): int(row["n"])
            for row in self.connection.execute(
                """
                SELECT ifc_class, COUNT(*) AS n FROM nodes
                WHERE ifc_class IN (
                  'IfcProject','IfcSite','IfcBuilding','IfcBuildingStorey','IfcSpace','IfcZone'
                )
                GROUP BY ifc_class ORDER BY ifc_class
                """
            )
        }
        relation_counts: dict[str, dict[str, int]] = {}
        evidence_placeholders = ",".join("?" for _ in self.include_evidence)
        for row in self.connection.execute(
            f"""
            SELECT relation, evidence_type, COUNT(*) AS n FROM edges
            WHERE relation IN (
              'contains','part_of','assigned_to_system',
              'requires_inspection_of','related_to_system','serves'
            )
              AND evidence_type IN ({evidence_placeholders})
            GROUP BY relation, evidence_type ORDER BY relation, evidence_type
            """,
            self.include_evidence,
        ):
            relation_counts.setdefault(str(row["relation"]), {})[
                str(row["evidence_type"])
            ] = int(row["n"])
        actionability_counts = {
            str(row["value_text"]): int(row["n"])
            for row in self.connection.execute(
                """
                SELECT value_text, COUNT(DISTINCT node_id) AS n
                FROM node_values
                WHERE predicate='action_target_kind'
                GROUP BY value_text ORDER BY value_text
                """
            )
        }

        relevant: list[str] = []
        for node_id in relevant_node_ids:
            value = str(node_id)
            if value and value not in relevant and self.get_node(value) is not None:
                relevant.append(value)

        context = HierarchyContext(
            summary={
                "spatial_node_counts": spatial_counts,
                "hierarchy_edge_counts": relation_counts,
                "relation_semantics": {
                    "contains": "parent spatial container -> child",
                    "part_of": "child -> parent aggregate",
                    "assigned_to_system": "object -> system membership",
                    "requires_inspection_of": "function -> target object or space",
                    "related_to_system": "function -> relevant building system",
                    "serves": "system or component -> served space",
                },
                "actionability_counts": actionability_counts,
                "target_policy": (
                    "action_target_kind=space|object|system with IFC GUID; "
                    "graph level alone does not determine PDDL actionability"
                ),
                "support_policy": "action_target_kind=none may support paths but not actions",
            },
            relevant_node_ids=relevant,
        )
        seen_signatures: set[tuple[str, ...]] = set()

        def walk(
            node_id: str,
            visited: frozenset[str],
            depth: int,
            limit: int,
        ) -> list[tuple[list[str], list[HierarchyEdge]]]:
            if limit <= 0:
                context.truncated = True
                if "path_cap" not in context.truncation_reasons:
                    context.truncation_reasons.append("path_cap")
                return []
            if depth >= max_depth:
                if "depth_cap" not in context.truncation_reasons:
                    context.truncation_reasons.append("depth_cap")
                context.truncated = True
                return [([node_id], [])]
            parents = self._hierarchy_parent_rows(node_id)
            if not parents:
                return [([node_id], [])]
            results: list[tuple[list[str], list[HierarchyEdge]]] = []
            for row in parents:
                if len(results) >= limit:
                    context.truncated = True
                    if "path_cap" not in context.truncation_reasons:
                        context.truncation_reasons.append("path_cap")
                    break
                parent_id = str(row["parent_id"])
                if parent_id in visited:
                    context.truncated = True
                    if "cycle" not in context.truncation_reasons:
                        context.truncation_reasons.append("cycle")
                    continue
                edge = HierarchyEdge(
                    source_id=str(row["source"]),
                    source_label=str(row["source_label"]),
                    source_ifc_class=row["source_ifc_class"],
                    relation=str(row["relation"]),
                    target_id=str(row["target"]),
                    target_label=str(row["target_label"]),
                    target_ifc_class=row["target_ifc_class"],
                    provenance=str(row["evidence_type"]),
                    confidence=float(row["confidence"]),
                )
                parent_paths = walk(
                    parent_id,
                    visited | {parent_id},
                    depth + 1,
                    limit - len(results),
                )
                for node_ids, edges in parent_paths:
                    results.append((node_ids + [node_id], edges + [edge]))
                    if len(results) >= limit:
                        break
            return results or [([node_id], [])]

        for target_id in relevant:
            if len(context.paths) >= max_paths:
                context.truncated = True
                if "path_cap" not in context.truncation_reasons:
                    context.truncation_reasons.append("path_cap")
                break
            remaining_paths = max_paths - len(context.paths)
            for node_ids, edges in walk(
                target_id,
                frozenset({target_id}),
                0,
                remaining_paths,
            ):
                signature = tuple(node_ids)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                nodes = [self.get_node(node_id) for node_id in node_ids]
                context.paths.append(
                    HierarchyPath(
                        path_id=f"hierarchy_{len(context.paths) + 1}",
                        target_id=target_id,
                        node_ids=node_ids,
                        node_labels=[node.label if node else node_id for node, node_id in zip(nodes, node_ids)],
                        node_ifc_classes=[node.ifc_class if node else None for node in nodes],
                        edges=edges,
                    )
                )
                if len(context.paths) >= max_paths:
                    context.truncated = True
                    if "path_cap" not in context.truncation_reasons:
                        context.truncation_reasons.append("path_cap")
                    break
        return context

    @staticmethod
    def _hierarchy_relations_for_binding(
        plan: QueryPlan,
        binding: ActionTargetBinding | None,
    ) -> set[str]:
        relations = {"contains", "part_of"}
        functions = list(
            getattr(binding, "function_types", []) or plan.function_intents
        )
        systems = list(getattr(binding, "system_categories", []) or [])
        domains = list(getattr(binding, "target_domains", []) or [])
        if plan.target_domain:
            domains.append(plan.target_domain)
        if functions:
            relations.update(
                {
                    "requires_inspection_of",
                    "related_to_system",
                    "assigned_to_system",
                    "serves",
                }
            )
        if systems or domains or plan.target_binding_mode in {
            "system_entity", "system_members"
        }:
            relations.update(
                {
                    "assigned_to_system",
                    "related_to_system",
                    "serves",
                    "connects_to",
                }
            )
        if plan.operator in {"path", "unconnected"}:
            relations.update({"connects_to", "has_port"})
        return relations

    @classmethod
    def _hierarchy_node_matches_terms(
        cls,
        entity: EntityRef | None,
        terms: Sequence[str],
    ) -> bool:
        if entity is None or not terms:
            return False
        haystack = cls._normalized_text(cls._search_text(entity))
        haystack_tokens = set(haystack.split())
        for term in terms:
            normalized = cls._normalized_text(term)
            if not normalized:
                continue
            if normalized == haystack or set(normalized.split()).issubset(
                haystack_tokens
            ):
                return True
        return False

    def expand_hierarchy_candidates(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        subgraph: GnnSubgraph | None,
        *,
        max_depth: int = 4,
        max_candidates: int = 100,
        per_relation_quota: int = 24,
    ) -> list[EntityRef]:
        """Complete typed scope/support paths before target selection."""

        if subgraph is None:
            return []
        bindings: list[ActionTargetBinding | None] = (
            list(plan.action_bindings) if plan.action_bindings else [None]
        )
        exact_ids = [
            seed.node_id
            for seed in seeds
            if seed.kind == "entity" and seed.score >= 0.9
        ]
        target_anchor_ids = [anchor.node_id for anchor in subgraph.target_anchors]
        support_anchor_ids = [anchor.node_id for anchor in subgraph.support_anchors]
        result: dict[str, EntityRef] = {}
        degree_cache: dict[str, int] = {}

        def degree(node_id: str) -> int:
            if node_id not in degree_cache:
                row = self.connection.execute(
                    """
                    SELECT COUNT(*) AS n FROM edges
                    WHERE source=? OR target=?
                    """,
                    (node_id, node_id),
                ).fetchone()
                degree_cache[node_id] = int(row["n"] if row else 0)
            return degree_cache[node_id]

        per_binding_cap = max(8, max_candidates // max(1, len(bindings)))
        for binding in bindings:
            binding_index = int(getattr(binding, "binding_index", 0))
            allowed = self._hierarchy_relations_for_binding(plan, binding)
            starts = list(dict.fromkeys(
                [*exact_ids, *support_anchor_ids, *target_anchor_ids]
            ))
            visited = set(starts)
            frontier = list(starts)
            parents: dict[str, tuple[str, str, float, str]] = {}
            binding_added = 0
            for _depth in range(max(0, max_depth)):
                ranked: list[
                    tuple[int, int, float, int, str, str, TripleEvidence, EntityRef]
                ] = []
                for source_id in frontier:
                    source = self.get_node(source_id)
                    if source is None:
                        continue
                    for relation in self.relations(source_id):
                        if (
                            relation.direction == "value"
                            or relation.name not in allowed
                        ):
                            continue
                        for entity, edge in self.expand(
                            source,
                            relation,
                            limit=per_relation_quota,
                        ):
                            status = self.candidate_constraint_status(
                                plan, seeds, entity, binding=binding
                            )
                            constraints = status.get("constraints", {})
                            states = (
                                {
                                    str(key): str(value).lower()
                                    for key, value in constraints.items()
                                }
                                if isinstance(constraints, dict)
                                else {}
                            )
                            hard_fail = (
                                str(
                                    status.get("overall", "unknown")
                                ).lower()
                                == "fail"
                                or "fail" in states.values()
                            )
                            actionable = (
                                self.action_target_kind(entity) != "none"
                            )
                            if actionable and hard_fail:
                                continue
                            positive = sum(
                                states.get(key) == "pass"
                                for key in (
                                    "name",
                                    "role",
                                    "domain",
                                    "function",
                                    "system",
                                )
                            )
                            bridge = int(not actionable)
                            ranked.append(
                                (
                                    1,
                                    positive,
                                    float(edge.confidence)
                                    - 0.03
                                    * math.log1p(degree(entity.node_id)),
                                    bridge,
                                    entity.node_id,
                                    source_id,
                                    edge,
                                    entity,
                                )
                            )
                ranked.sort(
                    key=lambda item: (
                        -item[0], -item[1], -item[2], -item[3],
                        item[4], item[5],
                    )
                )
                next_frontier: list[str] = []
                relation_counts: Counter[str] = Counter()
                for (
                    _relation_match,
                    _positive,
                    _score,
                    _bridge,
                    node_id,
                    source_id,
                    edge,
                    entity,
                ) in ranked:
                    if relation_counts[edge.relation] >= per_relation_quota:
                        continue
                    relation_counts[edge.relation] += 1
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    parents[node_id] = (
                        source_id,
                        edge.relation,
                        float(edge.confidence),
                        str(edge.provenance),
                    )
                    next_frontier.append(node_id)
                    if self.action_target_kind(entity) == "none":
                        continue
                    path_nodes = [node_id]
                    path_relations: list[str] = []
                    path_confidences: list[float] = []
                    path_provenance: list[str] = []
                    current = node_id
                    while current in parents:
                        parent, relation, confidence, provenance = parents[current]
                        path_nodes.append(parent)
                        path_relations.append(relation)
                        path_confidences.append(confidence)
                        path_provenance.append(provenance)
                        current = parent
                    path_nodes.reverse()
                    path_relations.reverse()
                    record = {
                        "binding_index": binding_index,
                        "node_ids": path_nodes,
                        "relations": path_relations,
                        "confidence": (
                            min(path_confidences)
                            if path_confidences
                            else 0.0
                        ),
                        "provenance": sorted(set(path_provenance)),
                    }
                    existing = result.get(node_id)
                    if existing is None:
                        entity.match_reason = "hierarchy"
                        entity.metadata["_hierarchy_expansion_paths"] = [record]
                        result[node_id] = entity
                        binding_added += 1
                    else:
                        paths = existing.metadata.setdefault(
                            "_hierarchy_expansion_paths", []
                        )
                        if record not in paths:
                            paths.append(record)
                    if binding_added >= per_binding_cap:
                        break
                frontier = next_frontier
                if not frontier or binding_added >= per_binding_cap:
                    break
        return sorted(
            result.values(),
            key=lambda entity: (
                -float(entity.score),
                math.log1p(degree(entity.node_id)),
                entity.node_id,
            ),
        )[:max_candidates]

    def _hierarchy_candidate_paths(
        self,
        target_id: str,
        *,
        max_paths: int,
        max_depth: int,
    ) -> list[HierarchyPath]:
        context = self.build_hierarchy_context(
            [target_id],
            max_paths=max_paths,
            max_depth=max_depth,
        )
        return [
            path for path in context.paths if path.target_id == target_id
        ]

    def assess_hierarchy_candidates(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
        *,
        max_paths_per_binding: int = 12,
        max_depth: int = 4,
    ) -> list[EntityRef]:
        """Attach query-directed typed path evidence before selection."""

        # Apply the same graph-backed operational-asset gate after downstream
        # candidate merging.  The deterministic operator already uses this
        # gate, but GNN support expansion can re-introduce cabinets,
        # assemblies, or representation nodes that share the asset name and
        # room.  Hierarchy should validate missing topology, not undo an
        # independently established operational-vs-container distinction.
        operator_candidates = [
            candidate
            for candidate in candidates
            if candidate.metadata.get("_deterministic_operator_candidate")
        ]
        gated_candidates = self._apply_fault_operational_gate(plan, candidates)
        # The post-merge gate governs retrieval additions; it must not revoke
        # candidates already produced by the scoped graph operator.  Those
        # candidates still pass through the same per-binding constraints and
        # hierarchy audit below, so preserving them cannot bypass validation.
        # This separation also prevents an operational GNN decoy from causing
        # a legitimately requested cabinet/assembly to be dropped.
        candidates = list(
            {
                candidate.node_id: candidate
                for candidate in [*operator_candidates, *gated_candidates]
            }.values()
        )
        bindings: list[ActionTargetBinding | None] = (
            list(plan.action_bindings) if plan.action_bindings else [None]
        )
        # ``max_paths_per_binding`` is a global path budget, not a
        # per-candidate allowance.  Applying it independently to a hundred
        # merged candidates made a nominal 12-path configuration enumerate
        # more than a thousand root paths.  First retain a small,
        # evidence-ranked candidate beam for each action slot, then divide the
        # path budget across that beam.
        # The configured twelve-path allowance is also the candidate ceiling.
        # A smaller fixed beam can split an otherwise valid logical cohort
        # (for example, several IFC elements authored as one continuous
        # surface) before target grouping has a chance to close it.
        candidate_beam = max(1, min(12, max_paths_per_binding))
        status_cache: dict[tuple[str, int], dict[str, Any]] = {}

        def binding_key(binding: ActionTargetBinding | None) -> int:
            return int(getattr(binding, "binding_index", 0))

        def base_status(
            candidate: EntityRef,
            binding: ActionTargetBinding | None,
        ) -> dict[str, Any]:
            key = (candidate.node_id, binding_key(binding))
            if key not in status_cache:
                status = dict(self.candidate_constraint_status(
                    plan, seeds, candidate, binding=binding
                ))
                status_cache[key] = status
            return status_cache[key]

        def gnn_path_priority(
            candidate: EntityRef,
            binding: ActionTargetBinding | None,
        ) -> int:
            rows = candidate.metadata.get("_gnn_path_evidence", [])
            relevant = [
                row for row in rows
                if isinstance(row, dict)
                and int(row.get("binding_index", 0))
                == binding_key(binding)
            ]
            order = {
                "complete": 2,
                "partial": 1,
                "unsupported": 0,
                "contradictory": -1,
            }
            return max(
                (
                    order.get(str(row.get("path_status", "unsupported")), 0)
                    for row in relevant
                ),
                default=0,
            )

        retained_ids: set[str] = set()
        for binding in bindings:
            ranked: list[
                tuple[
                    int, int, int, int, int, float, str, EntityRef
                ]
            ] = []
            for candidate in candidates:
                if self.action_target_kind(candidate) == "none":
                    continue
                status = base_status(candidate, binding)
                constraints = status.get("constraints", {})
                states = (
                    {
                        str(key): str(value).lower()
                        for key, value in constraints.items()
                    }
                    if isinstance(constraints, dict)
                    else {}
                )
                hard_fail = (
                    str(status.get("overall", "unknown")).lower() == "fail"
                    or "fail" in states.values()
                )
                if hard_fail:
                    continue
                semantic_passes = sum(
                    states.get(key) == "pass"
                    for key in (
                        "name", "role", "domain", "function", "system"
                    )
                )
                ranked.append(
                    (
                        int(
                            bool(
                                candidate.metadata.get(
                                    "_deterministic_operator_candidate"
                                )
                            )
                        ),
                        int(
                            str(status.get("overall", "unknown")).lower()
                            == "pass"
                        ),
                        gnn_path_priority(candidate, binding),
                        int(states.get("scope") == "pass"),
                        semantic_passes,
                        float(candidate.score),
                        candidate.node_id,
                        candidate,
                    )
                )
            ranked.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    -item[3],
                    -item[4],
                    -item[5],
                    item[6],
                )
            )
            retained_ids.update(
                item[-1].node_id for item in ranked[:candidate_beam]
            )

        bounded_candidates = [
            candidate for candidate in candidates
            if candidate.node_id in retained_ids
        ]
        paths_per_candidate = max(
            1,
            max_paths_per_binding // max(1, len(bounded_candidates)),
        )
        specific_scope = self._has_specific_space_scope(plan)
        scope_ids = (
            set(self._room_scope_ids(plan, seeds))
            if specific_scope
            else set()
        )
        nested_scope = any(
            predicate.predicate
            in {
                "contains_role", "contains_domain", "contains_name",
                "space_function",
            }
            for predicate in plan.scope_predicates
        )
        result: list[EntityRef] = []
        degree_cache: dict[str, int] = {}

        def degree(node_id: str) -> int:
            if node_id not in degree_cache:
                row = self.connection.execute(
                    """
                    SELECT COUNT(*) AS n FROM edges
                    WHERE source=? OR target=?
                    """,
                    (node_id, node_id),
                ).fetchone()
                degree_cache[node_id] = int(row["n"] if row else 0)
            return degree_cache[node_id]

        for candidate in bounded_candidates:
            if self.action_target_kind(candidate) == "none":
                continue
            paths = self._hierarchy_candidate_paths(
                candidate.node_id,
                max_paths=paths_per_candidate,
                max_depth=max_depth,
            )
            rows: list[dict[str, Any]] = []
            for binding in bindings:
                binding_index = int(getattr(binding, "binding_index", 0))
                status = base_status(candidate, binding)
                raw_constraints = status.get("constraints", {})
                constraints = (
                    {
                        str(key): str(value).lower()
                        for key, value in raw_constraints.items()
                    }
                    if isinstance(raw_constraints, dict)
                    else {}
                )
                positive = sorted(
                    key for key, value in constraints.items()
                    if value == "pass"
                )
                unknown = sorted(
                    key for key, value in constraints.items()
                    if value == "unknown"
                )
                contradiction = (
                    str(status.get("overall", "unknown")).lower() == "fail"
                    or "fail" in constraints.values()
                )
                function_terms = list(
                    getattr(binding, "function_types", [])
                    or plan.function_intents
                )
                system_terms = list(
                    getattr(binding, "system_categories", []) or []
                )
                allowed = self._hierarchy_relations_for_binding(plan, binding)
                best: tuple[
                    tuple[int, int, int, float, float, int, tuple[str, ...]],
                    list[str],
                    list[str],
                    list[str],
                    bool,
                    bool,
                    bool,
                    float,
                    float,
                ] | None = None
                path_options: list[
                    tuple[list[str], list[str], list[str], float, float]
                ] = []
                for path in paths:
                    node_ids = list(path.node_ids)
                    relations = [edge.relation for edge in path.edges]
                    confidences = [float(edge.confidence) for edge in path.edges]
                    path_options.append(
                        (
                            node_ids,
                            relations,
                            sorted({edge.provenance for edge in path.edges}),
                            min(confidences) if confidences else 0.0,
                            sum(
                                math.log1p(degree(node_id))
                                for node_id in node_ids[1:-1]
                            ),
                        )
                    )
                if len(path_options) > 1:
                    bundled_nodes = list(
                        dict.fromkeys(
                            node_id
                            for node_ids, *_rest in path_options
                            for node_id in node_ids
                        )
                    )
                    bundled_relations = list(
                        dict.fromkeys(
                            relation
                            for _nodes, relations, *_rest in path_options
                            for relation in relations
                        )
                    )
                    bundled_provenance = sorted(
                        {
                            provenance
                            for _nodes, _relations, provenances, *_rest
                            in path_options
                            for provenance in provenances
                        }
                    )
                    path_options.append(
                        (
                            bundled_nodes,
                            bundled_relations,
                            bundled_provenance,
                            min(
                                confidence
                                for *_prefix, confidence, _hub in path_options
                            ),
                            sum(
                                math.log1p(degree(node_id))
                                for node_id in bundled_nodes[1:-1]
                            ),
                        )
                    )
                if not path_options:
                    path_options = [
                        ([candidate.node_id], [], [], 0.0, 0.0)
                    ]
                for (
                    node_ids,
                    relations,
                    provenance,
                    path_confidence,
                    hub_exposure,
                ) in path_options:
                    path_nodes = [self.get_node(node_id) for node_id in node_ids]
                    scope_connected = (
                        not specific_scope
                        or bool(scope_ids.intersection(node_ids))
                        or candidate.node_id in scope_ids
                    )
                    function_connected = (
                        not function_terms
                        or (
                            bool(
                                {
                                    "requires_inspection_of",
                                    "related_to_system",
                                }.intersection(relations)
                            )
                            and any(
                                self._hierarchy_node_matches_terms(
                                    node, function_terms
                                )
                                for node in path_nodes
                            )
                        )
                    )
                    system_connected = (
                        not system_terms
                        or (
                            bool(
                                {
                                    "assigned_to_system",
                                    "related_to_system",
                                    "serves",
                                }.intersection(relations)
                            )
                            and any(
                                self._hierarchy_node_matches_terms(
                                    node, system_terms
                                )
                                for node in path_nodes
                            )
                        )
                    )
                    typed = all(relation in allowed for relation in relations)
                    semantic_positive = bool(
                        {"name", "role", "domain", "function", "system"}
                        .intersection(positive)
                    ) or not any(
                        (
                            getattr(binding, "target_roles", []),
                            getattr(binding, "target_names", []),
                            getattr(binding, "target_domains", []),
                            function_terms,
                            system_terms,
                            plan.target_roles,
                            plan.target_role,
                            plan.target_names,
                            plan.target_name,
                            plan.target_domain,
                        )
                    )
                    positive_identity = bool(
                        {"name", "role", "domain"}.intersection(positive)
                    )
                    credible_support_connection = bool(relations) and bool(
                        {
                            "contains",
                            "part_of",
                            "assigned_to_system",
                            "serves",
                            "requires_inspection_of",
                            "related_to_system",
                        }.intersection(relations)
                    )
                    # Missing function/system relations are unknown, not a
                    # contradiction.  A candidate may still complete a
                    # binding when an independent semantic discriminator and
                    # a typed scope/support connection identify it.  This is
                    # the same missingness contract used by target audit; it
                    # never turns similarity alone into path evidence.
                    function_missingness_supported = bool(
                        function_terms
                        and constraints.get("function") == "unknown"
                        and positive_identity
                        and scope_connected
                        and credible_support_connection
                    )
                    system_missingness_supported = bool(
                        system_terms
                        and constraints.get("system") == "unknown"
                        and positive_identity
                        and scope_connected
                        and credible_support_connection
                    )
                    # A graph-backed ontology match is itself positive
                    # function/system evidence.  Requiring a second
                    # materialized function/system edge after the constraint
                    # evaluator has already established that match turns
                    # sparse-but-valid BIM data back into a false negative.
                    # Keep the scope/support path requirement so semantic
                    # similarity alone can never complete a binding.
                    function_semantic_supported = bool(
                        function_terms
                        and constraints.get("function") == "pass"
                        and scope_connected
                        and credible_support_connection
                    )
                    system_semantic_supported = bool(
                        system_terms
                        and constraints.get("system") == "pass"
                        and scope_connected
                        and credible_support_connection
                    )
                    function_satisfied = bool(
                        function_connected
                        or function_missingness_supported
                        or function_semantic_supported
                    )
                    system_satisfied = bool(
                        system_connected
                        or system_missingness_supported
                        or system_semantic_supported
                    )
                    complete = bool(
                        not contradiction
                        and typed
                        and scope_connected
                        and function_satisfied
                        and system_satisfied
                        and semantic_positive
                    )
                    partial = bool(
                        not contradiction
                        and bool(relations)
                        and typed
                    )
                    ranking = (
                        int(complete),
                        int(scope_connected),
                        int(function_connected and system_connected),
                        path_confidence,
                        -hub_exposure,
                        -len(relations),
                        tuple(node_ids),
                    )
                    record = (
                        ranking,
                        node_ids,
                        relations,
                        provenance,
                        complete,
                        partial,
                        scope_connected,
                        path_confidence,
                        hub_exposure,
                    )
                    if best is None or record[0] > best[0]:
                        best = record
                assert best is not None
                (
                    _ranking,
                    best_node_ids,
                    best_relations,
                    best_provenance,
                    complete,
                    partial,
                    scope_connected,
                    path_confidence,
                    hub_exposure,
                ) = best
                path_status = (
                    "contradictory"
                    if contradiction
                    else "complete"
                    if complete
                    else "partial"
                    if partial
                    else "unsupported"
                )
                rows.append(
                    {
                        "binding_index": binding_index,
                        "path_status": path_status,
                        "node_ids": best_node_ids,
                        "relations": best_relations,
                        "constraint_states": constraints,
                        "positive_constraints": positive,
                        "unknown_constraints": unknown,
                        "contradiction": contradiction,
                        "scope_connected": scope_connected,
                        "path_confidence": path_confidence,
                        "hub_exposure": hub_exposure,
                        "provenance": best_provenance,
                        "missingness_supported_constraints": [
                            key
                            for key, supported in (
                                (
                                    "function",
                                    function_missingness_supported,
                                ),
                                (
                                    "system",
                                    system_missingness_supported,
                                ),
                            )
                            if supported
                        ],
                        "semantic_supported_constraints": [
                            key
                            for key, supported in (
                                (
                                    "function",
                                    function_semantic_supported,
                                ),
                                (
                                    "system",
                                    system_semantic_supported,
                                ),
                            )
                            if supported
                        ],
                    }
                )
            candidate.metadata["_hierarchy_path_evidence"] = rows
            candidate.metadata["_hierarchy_path_required"] = bool(
                specific_scope
                or nested_scope
                or any(
                    (
                        getattr(binding, "function_types", []),
                        getattr(binding, "system_categories", []),
                    )
                    for binding in bindings
                )
                or plan.function_intents
            )
            result.append(candidate)
        return result

    def validate_hierarchy(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        targets: Sequence[EntityRef],
        *,
        max_paths: int = 12,
        max_depth: int = 6,
    ) -> HierarchyValidation:
        """Validate typed target paths deterministically.

        L2/L3 nodes remain available as path evidence; only each node's
        actionability contract decides whether it may be the terminal target.
        """
        # Selection already attaches binding-scoped hierarchy evidence to
        # every candidate it accepts.  Re-running the bounded path search here
        # is both expensive and semantically unsafe: a different beam
        # allocation can replace the evidence used by selection and make the
        # final audit disagree with the selected set.  Assess only legacy or
        # externally supplied targets that do not yet carry the contract.
        if targets and all(
            isinstance(
                entity.metadata.get("_hierarchy_path_evidence"), list
            )
            and bool(entity.metadata.get("_hierarchy_path_evidence"))
            for entity in targets
        ):
            assessed_targets = list(targets)
        else:
            assessed_targets = self.assess_hierarchy_candidates(
                plan,
                seeds,
                targets,
                max_paths_per_binding=max_paths,
                max_depth=min(max_depth, 4),
            )
        audit = self.target_audit(
            plan, seeds, assessed_targets, excluded_limit=0
        )
        context = self.build_hierarchy_context(
            [entity.node_id for entity in [*seeds, *assessed_targets]],
            max_paths=max_paths,
            max_depth=max_depth,
        )
        valid_target_ids = {
            item.target_id for item in audit.target_validations if item.valid
        }
        valid_paths = [
            path.path_id
            for path in context.paths
            if path.target_id in valid_target_ids
        ]
        per_binding_path_coverage: dict[int, list[str]] = {}
        targets_without_valid_path: list[str] = []
        path_complete_ids: set[str] = set()
        for entity in assessed_targets:
            rows = entity.metadata.get("_hierarchy_path_evidence", [])
            relevant_binding_indices = {
                int(binding.binding_index)
                for binding in plan.action_bindings
                if self._binding_matches(binding, entity)
            } or {
                int(row.get("binding_index", 0))
                for row in rows
                if isinstance(row, dict)
            }
            complete_for_target = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                binding_index = int(row.get("binding_index", 0))
                if (
                    binding_index in relevant_binding_indices
                    and row.get("path_status") == "complete"
                    and not row.get("contradiction")
                ):
                    per_binding_path_coverage.setdefault(
                        binding_index, []
                    ).append(entity.node_id)
                    complete_for_target = True
            if complete_for_target:
                path_complete_ids.add(entity.node_id)
            else:
                targets_without_valid_path.append(entity.node_id)
        per_binding_path_coverage = {
            binding_index: sorted(set(node_ids))
            for binding_index, node_ids in per_binding_path_coverage.items()
        }
        extra_path_valid_targets = sorted(
            path_complete_ids - valid_target_ids
        )
        support_ids: set[str] = set()
        for path in context.paths:
            for node_id in path.node_ids:
                node = self.get_node(node_id)
                if node is not None and self.action_target_kind(node) == "none":
                    support_ids.add(node_id)
        violations = list(
            dict.fromkeys(
                issue
                for validation in audit.target_validations
                for issue in validation.issues
            )
        )
        unresolved = list(plan.unresolved_slots)
        if not audit.action_bindings_covered:
            unresolved.append("action_bindings")
        if audit.validation_summary.get("missing_expected_target_count"):
            unresolved.append("exhaustive_target_closure")
        if targets_without_valid_path:
            unresolved.append("binding_valid_hierarchy_path")
        closure_complete = bool(
            audit.action_bindings_covered
            and not audit.validation_summary.get(
                "missing_expected_target_count"
            )
            and not audit.conflicting_extras
        )
        return HierarchyValidation(
            binding_valid_paths=valid_paths,
            violated_constraints=violations,
            unresolved_slots=list(dict.fromkeys(unresolved)),
            target_node_ids=sorted(valid_target_ids),
            support_node_ids=sorted(support_ids),
            per_binding_path_coverage=per_binding_path_coverage,
            targets_without_valid_path=sorted(
                set(targets_without_valid_path)
            ),
            extra_path_valid_targets=extra_path_valid_targets,
            closure_complete=closure_complete,
            complete=bool(
                closure_complete
                and not targets_without_valid_path
                and not extra_path_valid_targets
                and not audit.conflicting_extras
                and not violations
                and not unresolved
            ),
        )

    def contained_targets_for_room(self, room: str) -> set[str]:
        space_ids = [
            str(row["node_id"])
            for row in self.connection.execute(
                "SELECT node_id FROM nodes WHERE ifc_class='IfcSpace' AND lower(name)=lower(?)",
                (room,),
            )
        ]
        return self._contained_targets(space_ids)

    def contained_targets_for_plan(self, plan: QueryPlan) -> set[str]:
        """Authoritative contained targets for structured room/name scope."""
        if not self._has_specific_space_scope(plan):
            return set()
        return self._contained_targets(self._room_scope_ids(plan, ()))

    def _add_rows(
        self,
        results: dict[str, EntityRef],
        rows: Iterable[sqlite3.Row],
        score: float,
        reason: str,
    ) -> None:
        for row in rows:
            entity = self._entity(row, score, reason)
            previous = results.get(entity.node_id)
            if previous is None or previous.score < score:
                results[entity.node_id] = entity

    def resolve(self, question: str, max_results: int = 5) -> list[EntityRef]:
        results: dict[str, EntityRef] = {}
        lower = question.lower()

        for guid in GUID_RE.findall(question):
            self._add_rows(
                results,
                self.connection.execute("SELECT * FROM nodes WHERE global_id=?", (guid,)),
                1.0,
                "exact_global_id",
            )

        room_matches = re.findall(
            r"\b(?:room|space|lab|office|classroom|future\s+lab)\s*#?([0-9]{2,4}[A-Za-z]?)\b",
            question,
            flags=re.IGNORECASE,
        )
        for room in room_matches:
            self._add_rows(
                results,
                self.connection.execute(
                    "SELECT * FROM nodes WHERE lower(name)=lower(?) AND ifc_class='IfcSpace'",
                    (room,),
                ),
                0.99,
                "exact_room_number",
            )

        level = re.search(r"\b(?:level|floor)\s+([A-Za-z0-9]+)", question, re.IGNORECASE)
        if level:
            storey = f"LEVEL {level.group(1).upper()}"
            self._add_rows(
                results,
                self.connection.execute(
                    "SELECT * FROM nodes WHERE ifc_class='IfcBuildingStorey' AND lower(name)=lower(?)",
                    (storey,),
                ),
                0.96,
                "exact_storey",
            )

        tag_match = re.search(r"\b(?:tag|id)\s*#?([0-9]{5,})\b", question, re.IGNORECASE)
        if tag_match:
            self._add_rows(
                results,
                self.connection.execute("SELECT * FROM nodes WHERE tag=?", (tag_match.group(1),)),
                0.98,
                "exact_tag",
            )

        # Long IFC names are high-quality entity mentions. Restrict this scan to
        # spaces and named assets to avoid matching short family fragments.
        rows = self.connection.execute(
            """
            SELECT * FROM nodes
            WHERE length(coalesce(long_name,'')) >= 6 OR length(coalesce(name,'')) >= 8
            """
        )
        for row in rows:
            candidates = [row["long_name"], row["name"]]
            for candidate in candidates:
                if not candidate or len(str(candidate)) < 6:
                    continue
                candidate_lower = str(candidate).lower()
                if candidate_lower in lower:
                    self._add_rows(results, [row], 0.95, "exact_name_in_question")
                    break
                normalized = re.sub(r"\([^)]*\)", " ", candidate_lower)
                normalized = " ".join(
                    token for token in re.findall(r"[a-z0-9]+", normalized)
                    if token not in {"and", "the"}
                )
                normalized_question = " ".join(
                    token for token in re.findall(r"[a-z0-9]+", lower)
                    if token not in {"and", "the"}
                )
                if len(normalized) >= 6 and normalized in normalized_question:
                    self._add_rows(results, [row], 0.92, "normalized_name_in_question")
                    break

        role_terms = {
            "light_fixture": ("lighting fixture", "light fixture", "pendant lighting"),
            "furnishing": ("furnishing", "furniture"),
            "sprinkler": ("sprinkler", "fire suppression"),
            "fire_extinguisher": ("fire extinguisher",),
            "outlet": ("outlet", "receptacle"),
            "switch": ("switch",),
            "panel": ("panel",),
            "drain": ("drain",),
            "sanitary_fixture": ("sink", "plumbing fixture"),
            "door": ("door",),
        }
        for role, terms in role_terms.items():
            if any(term in lower for term in terms):
                self._add_rows(
                    results,
                    self.connection.execute(
                        "SELECT * FROM nodes WHERE role=? ORDER BY node_id LIMIT 20", (role,)
                    ),
                    0.55,
                    f"generic_role:{role}",
                )

        if len(results) < max_results:
            tokens = [
                token for token in re.findall(r"[A-Za-z0-9]+", question.lower())
                if len(token) >= 3 and token not in {"the", "all", "and", "for", "with", "level", "room"}
            ][:12]
            if tokens and self.meta.get("fts_enabled") == "true":
                query = " OR ".join(f'"{token}"' for token in tokens)
                try:
                    fts_rows = self.connection.execute(
                        """
                        SELECT n.* FROM nodes_fts f
                        JOIN nodes n ON n.node_id=f.node_id
                        WHERE nodes_fts MATCH ?
                        ORDER BY bm25(nodes_fts), n.node_id LIMIT 20
                        """,
                        (query,),
                    )
                    self._add_rows(results, fts_rows, 0.35, "fts")
                except sqlite3.OperationalError:
                    pass

        ordered = sorted(results.values(), key=lambda item: (-item.score, item.node_id))
        high_quality = [item for item in ordered if item.score >= 0.9]
        if high_quality:
            # Keep generic role nodes only after exact scopes, leaving enough room
            # for both a room and its requested target class.
            generic = [item for item in ordered if item.score < 0.9]
            return (high_quality + generic)[:max_results]
        return ordered[:max_results]

    def retrieval_lexical_seeds(
        self,
        question: str,
        *,
        level: str | None,
        max_results: int = 20,
    ) -> list[EntityRef]:
        """Return a bounded full-question lexical lane for hybrid retrieval."""

        if max_results <= 0:
            return []
        candidates = self.resolve(
            question,
            max_results=max_results * 4,
        )
        if level is None:
            return candidates[:max_results]
        expected = {
            "object": "L1_object",
            "space": "L0_space",
            "system": "L2_system",
            "function": "L3_function",
        }.get(level)
        if expected is None:
            raise ValueError(f"Unsupported lexical retrieval level: {level}")
        return [
            candidate
            for candidate in candidates
            if str(candidate.metadata.get("level") or "") == expected
        ][:max_results]

    def relations(self, node_id: str) -> list[RelationRef]:
        placeholders = ",".join("?" for _ in self.include_evidence)
        params = (node_id, *self.include_evidence)
        rows = self.connection.execute(
            f"""
            SELECT relation, 'out' AS direction FROM edges
            WHERE source=? AND evidence_type IN ({placeholders})
            UNION
            SELECT relation, 'in' AS direction FROM edges
            WHERE target=? AND evidence_type IN ({placeholders})
            """,
            (*params, node_id, *self.include_evidence),
        )
        relations = {(row["relation"], row["direction"]) for row in rows}
        value_rows = self.connection.execute(
            "SELECT DISTINCT predicate FROM node_values WHERE node_id=?", (node_id,)
        )
        relations.update((row["predicate"], "value") for row in value_rows)
        return [RelationRef(name=name, direction=direction) for name, direction in sorted(relations)]

    def expand(
        self,
        source: EntityRef,
        relation: RelationRef,
        limit: int = 20,
        preferred_node_ids: set[str] | None = None,
    ) -> list[tuple[EntityRef, TripleEvidence]]:
        if relation.direction == "value":
            rows = self.connection.execute(
                """
                SELECT value_text, value_num, value_type FROM node_values
                WHERE node_id=? AND predicate=? ORDER BY value_text LIMIT ?
                """,
                (source.node_id, relation.name, limit),
            )
            result = []
            for row in rows:
                value = str(row["value_text"])
                value_id = "literal_" + hashlib.sha1(
                    f"{source.node_id}|{relation.name}|{value}".encode()
                ).hexdigest()[:16]
                entity = EntityRef(
                    node_id=value_id,
                    label=value,
                    kind="literal",
                    score=relation.score,
                    match_reason="node_value",
                    metadata={"value_num": row["value_num"], "value_type": row["value_type"]},
                )
                evidence = TripleEvidence(
                    source_id=source.node_id,
                    source_label=source.label,
                    relation=relation.name,
                    target_id=value_id,
                    target_label=value,
                    direction="value",
                    provenance="explicit",
                    relevance_score=relation.score,
                    features={"global_id": source.global_id},
                )
                result.append((entity, evidence))
            return result

        placeholders = ",".join("?" for _ in self.include_evidence)
        if relation.direction == "out":
            endpoint = "e.target"
            where = "e.source=?"
        else:
            endpoint = "e.source"
            where = "e.target=?"
        preferred_node_ids = preferred_node_ids or set()
        order_clause = "n.node_id"
        order_params: list[Any] = []
        if preferred_node_ids:
            preferred_placeholders = ",".join("?" for _ in preferred_node_ids)
            order_clause = f"CASE WHEN n.node_id IN ({preferred_placeholders}) THEN 0 ELSE 1 END, n.node_id"
            order_params.extend(sorted(preferred_node_ids))
        rows = self.connection.execute(
            f"""
            SELECT n.*, e.evidence_type, e.confidence, e.features_json
            FROM edges e JOIN nodes n ON n.node_id={endpoint}
            WHERE {where} AND e.relation=? AND e.evidence_type IN ({placeholders})
            ORDER BY {order_clause} LIMIT ?
            """,
            (source.node_id, relation.name, *self.include_evidence, *order_params, limit),
        )
        result = []
        for row in rows:
            entity = self._entity(row, relation.score, f"{relation.direction}:{relation.name}")
            features = json.loads(row["features_json"] or "{}")
            if entity.global_id:
                features["global_id"] = entity.global_id
            evidence = TripleEvidence(
                source_id=source.node_id,
                source_label=source.label,
                relation=relation.name,
                target_id=entity.node_id,
                target_label=entity.label,
                direction=relation.direction,
                provenance=row["evidence_type"],
                confidence=float(row["confidence"]),
                relevance_score=relation.score,
                features=features,
            )
            result.append((entity, evidence))
        return result

    def neighbors(self, node_id: str) -> list[tuple[EntityRef, TripleEvidence]]:
        """Return every entity neighbor with auditable SQLite edge evidence."""
        entity = self.get_node(node_id)
        if entity is None:
            return []
        result: list[tuple[EntityRef, TripleEvidence]] = []
        for relation in self.relations(node_id):
            if relation.direction == "value":
                continue
            result.extend(self.expand(entity, relation, limit=1_000_000))
        result.sort(
            key=lambda item: (
                item[1].relation,
                item[1].direction,
                item[0].node_id,
                item[1].provenance,
            )
        )
        return result

    def _function_kinds(self, function_types: Sequence[str]) -> set[str]:
        if not function_types:
            return set()
        placeholders = ",".join("?" for _ in function_types)
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT lower(v.value_text) AS function_kind
            FROM nodes f JOIN node_values v ON v.node_id=f.node_id
            WHERE f.function_type IN ({placeholders})
              AND v.predicate='property.function_kind'
            ORDER BY function_kind
            """,
            tuple(function_types),
        )
        return {str(row["function_kind"]) for row in rows if row["function_kind"]}

    @classmethod
    def _operational_ifc_class(cls, ifc_class: str | None) -> bool:
        value = str(ifc_class or "")
        return bool(
            value in OPERATIONAL_IFC_CLASSES
            or value.startswith(("IfcFlow", "IfcDistribution"))
            or value.endswith(OPERATIONAL_IFC_CLASS_SUFFIXES)
        )

    @classmethod
    def _representation_like(cls, entity: EntityRef) -> bool:
        if entity.ifc_class in NON_OPERATIONAL_IFC_CLASSES:
            return True
        if entity.ifc_class == "IfcElementAssembly":
            return True
        tokens = set(cls._normalized_text(cls._search_text(entity)).split())
        return bool(tokens & REPRESENTATION_TOKENS)

    @staticmethod
    def _plan_target_terms(plan: QueryPlan) -> list[str]:
        values = list(
            plan.target_roles
            or ([plan.target_role] if plan.target_role else [])
        )
        values.extend(plan.target_names or ([plan.target_name] if plan.target_name else []))
        values.extend(plan.target_family_terms)
        values.extend(plan.target_type_terms)
        values.extend(plan.target_keywords)
        values.extend(
            value
            for binding in plan.action_bindings
            for value in (*binding.target_roles, *binding.target_names)
        )
        return list(dict.fromkeys(str(value) for value in values if value))

    def _operational_request(
        self,
        plan: QueryPlan,
        function_types: Sequence[str],
    ) -> tuple[bool, list[str]]:
        requested_terms = self._plan_target_terms(plan)
        requested_tokens = {
            token
            for value in requested_terms
            for token in self._normalized_text(value).split()
        }
        # An instruction that explicitly names an enclosure/assembly is about
        # that representation itself; it must not be silently rewritten to a
        # different service component.
        if requested_tokens & REPRESENTATION_TOKENS:
            return False, requested_terms

        ontology_roles = [
            role
            for _function_id, _is_method, roles
            in self._function_ontology_definitions(function_types)
            for role in roles
        ]
        requested_roles = list(
            dict.fromkeys(
                [
                    *(
                        plan.target_roles
                        or ([plan.target_role] if plan.target_role else [])
                    ),
                    *(
                        role
                        for binding in plan.action_bindings
                        for role in binding.target_roles
                    ),
                    *ontology_roles,
                ]
            )
        )
        role_tokens = {
            token
            for role in requested_roles
            for token in self._normalized_text(role).split()
        }
        requested_class = str(plan.target_ifc_class or "")
        operational = bool(
            "fault_condition" in self._function_kinds(function_types)
            or requested_tokens & OPERATIONAL_REQUEST_TOKENS
            or role_tokens & OPERATIONAL_REQUEST_TOKENS
            or self._operational_ifc_class(requested_class)
        )
        return operational, requested_roles

    def _system_participant_ids(self) -> set[str]:
        if self._system_participant_ids_cache is not None:
            return self._system_participant_ids_cache
        evidence_placeholders = ",".join("?" for _ in self.include_evidence)
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT source FROM edges
            WHERE relation='assigned_to_system'
              AND evidence_type IN ({evidence_placeholders})
            """,
            self.include_evidence,
        )
        self._system_participant_ids_cache = {
            str(row["source"]) for row in rows
        }
        return self._system_participant_ids_cache

    def _operational_rank(
        self,
        entity: EntityRef,
        *,
        requested_roles: Sequence[str] = (),
        function_types: Sequence[str] = (),
    ) -> int:
        if self.action_target_kind(entity) == "system":
            return 3
        # A broad or inferred system assignment must not promote cabinets,
        # assemblies, or representation objects above the operational asset
        # they merely enclose. Explicitly requested representations bypass
        # this gate earlier in ``_operational_request``.
        if self._representation_like(entity):
            return 0
        if entity.node_id in self._system_participant_ids():
            return 3
        if self._operational_ifc_class(entity.ifc_class):
            return 3
        if requested_roles and self._semantic_role_match(entity, requested_roles):
            return 2
        if function_types and self._functional_path_evidence(
            entity.node_id, function_types
        ):
            return 2
        profile = self.semantic_profile(entity)
        if any(
            value and value != "unknown"
            for value in profile["normalized_domains"]
        ) and self.action_target_kind(entity) == "object":
            return 1
        return 0

    def _apply_fault_operational_gate(
        self,
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
    ) -> list[EntityRef]:
        """Prefer service participants for ontology-declared fault conditions.

        This is a precision gate, not a destructive ontology rule: when the
        graph has no operational alternative, every candidate is retained so
        incomplete IFC classification remains recoverable downstream.
        """
        function_types = self._target_function_types(plan)
        operational_request, requested_roles = self._operational_request(
            plan, function_types
        )
        if not operational_request:
            return list(candidates)
        ranked = [
            (
                self._operational_rank(
                    candidate,
                    requested_roles=requested_roles,
                    function_types=function_types,
                ),
                candidate,
            )
            for candidate in candidates
        ]
        # Only tiers 2/3 certify an operational component.  Tier 1 is weak
        # domain-only evidence and tier 0 is representation-like.  If no
        # certified component exists, retain the original set as a deliberate
        # incomplete-BIM fallback rather than returning an empty answer.
        if not any(rank >= 2 for rank, _ in ranked):
            return list(candidates)
        return [candidate for rank, candidate in ranked if rank >= 2]

    def _expand_explicit_asset_groups(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
    ) -> list[EntityRef]:
        """Close a matched component over a graph-named explicit asset group.

        IFC authoring tools often model one maintainable asset as several
        GUID-bearing pieces under an ``IfcGroup`` (for example a basin,
        countertop and base assembly).  The closure is deliberately narrow:
        the query phrase must match the group label, membership must be
        explicit, and every added member must satisfy actionability, semantic
        and authoritative spatial constraints.  Generic building-wide system
        membership and inferred mega-hubs cannot trigger it.
        """

        names = list(plan.target_names or ([plan.target_name] if plan.target_name else []))
        if (
            plan.target_kind != "object"
            or not candidates
            or not names
            or "explicit" not in self.include_evidence
        ):
            return list(candidates)
        source_ids = sorted({candidate.node_id for candidate in candidates})
        placeholders = ",".join("?" for _ in source_ids)
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT members.source AS member_id,
                            groups.node_id AS group_id,
                            groups.search_text AS group_search_text
            FROM edges seed_membership
            JOIN nodes groups ON groups.node_id=seed_membership.target
            JOIN edges members ON members.target=groups.node_id
            WHERE seed_membership.source IN ({placeholders})
              AND seed_membership.relation='assigned_to_system'
              AND seed_membership.evidence_type='explicit'
              AND groups.ifc_class='IfcGroup'
              AND members.relation='assigned_to_system'
              AND members.evidence_type='explicit'
            ORDER BY groups.node_id, members.source
            """,
            tuple(source_ids),
        )
        allowed_rooms = (
            set(self._room_scope_ids(plan, seeds))
            if self._has_specific_space_scope(plan)
            else set()
        )
        roles = plan.target_roles or ([plan.target_role] if plan.target_role else [])
        by_id = {candidate.node_id: candidate for candidate in candidates}
        for row in rows:
            group_tokens = {
                self._match_token(token)
                for token in self._normalized_text(row["group_search_text"]).split()
            }
            if not any(
                (tokens := set(self._constraint_tokens(name)))
                and tokens.issubset(group_tokens)
                for name in names
            ):
                continue
            member = self.get_node(str(row["member_id"]))
            if member is None or self.action_target_kind(member) != "object":
                continue
            # The explicitly named group supplies the asset-name identity, so
            # a GUID-bearing sibling need not repeat that name in its own IFC
            # label (a sink assembly may contain a counter and cabinet).
            # Semantic role/domain remain required to avoid pulling unrelated
            # task lights, shelving, or generic modeling pieces from a broad
            # authoring group.
            if roles and not self._semantic_role_match(member, roles):
                continue
            if plan.target_domain and not self._semantic_domain_match(
                member, plan.target_domain
            ):
                continue
            if allowed_rooms:
                room = self._authoritative_room(member.node_id)
                if not room or str(room.get("room_id")) not in allowed_rooms:
                    continue
            member.metadata["_group_closure_provenance"] = {
                "source": "explicit_ifc_group",
                "group_id": str(row["group_id"]),
            }
            member.match_reason = "hierarchy_explicit_group_member"
            by_id.setdefault(member.node_id, member)
        return list(by_id.values())

    def _annotate_explicit_cohort(
        self,
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
    ) -> None:
        """Annotate graph-declared logical cohorts without changing cardinality.

        A singular query remains singular until the typed binding/finalizer
        explicitly supports cohort members.  This helper therefore exposes
        auditable membership for downstream consumers but never promotes a
        same-family majority to an answer set and never mutates ``QueryPlan``.
        """

        candidate_ids = sorted(
            {
                candidate.node_id
                for candidate in candidates
                if self.action_target_kind(candidate) == "object"
            }
        )
        if len(candidate_ids) < 2:
            return
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.connection.execute(
            f"""
            SELECT e.target AS cohort_id, e.source AS member_id, e.relation,
                   p.ifc_class, p.category,
                   coalesce(p.long_name, p.name, p.type_name, p.node_id) AS label
            FROM edges e JOIN nodes p ON p.node_id=e.target
            WHERE e.source IN ({placeholders})
              AND e.relation IN ('assigned_to_system','part_of')
              AND e.evidence_type='explicit'
            UNION ALL
            SELECT e.source AS cohort_id, e.target AS member_id, e.relation,
                   p.ifc_class, p.category,
                   coalesce(p.long_name, p.name, p.type_name, p.node_id) AS label
            FROM edges e JOIN nodes p ON p.node_id=e.source
            WHERE e.target IN ({placeholders})
              AND e.relation IN ('has_part','aggregates')
              AND e.evidence_type='explicit'
            ORDER BY cohort_id, member_id
            """,
            (*candidate_ids, *candidate_ids),
        ).fetchall()
        members_by_parent: dict[str, set[str]] = {}
        parent_details: dict[str, dict[str, str]] = {}
        for row in rows:
            member_id = str(row["member_id"])
            if member_id not in candidate_ids:
                continue
            cohort_id = str(row["cohort_id"])
            members_by_parent.setdefault(cohort_id, set()).add(member_id)
            parent_details.setdefault(
                cohort_id,
                {
                    "relation": str(row["relation"]),
                    "ifc_class": str(row["ifc_class"] or ""),
                    "category": str(row["category"] or ""),
                    "label": str(row["label"] or cohort_id),
                },
            )

        roles = set(
            plan.target_roles
            or ([plan.target_role] if plan.target_role else [])
        )
        roles.update(
            role
            for binding in plan.action_bindings
            for role in binding.target_roles
        )
        surface_request = bool(roles.intersection(CONTINUOUS_SURFACE_ROLES))
        for cohort_id, member_ids in members_by_parent.items():
            if len(member_ids) < 2:
                continue
            details = parent_details[cohort_id]
            kind = (
                "continuous_surface"
                if surface_request
                and details["relation"] in {"part_of", "has_part", "aggregates"}
                else "system_or_asset"
            )
            for candidate in candidates:
                if candidate.node_id not in member_ids:
                    continue
                candidate.metadata["_explicit_cohort"] = {
                    "source": "explicit_graph_membership",
                    "cohort_id": cohort_id,
                    "cohort_label": details["label"],
                    "cohort_kind": kind,
                    "relation": details["relation"],
                    "member_ids": sorted(member_ids),
                    "member_count": len(member_ids),
                    "cardinality_effect": "diagnostic_only",
                }

    @staticmethod
    def _logical_group_record(group: LogicalTargetGroup) -> dict[str, Any]:
        return {
            "group_id": group.group_id,
            "group_kind": group.group_kind,
            "binding_index": group.binding_index,
            "member_ids": list(group.member_ids),
            "source": group.source,
            "identity": group.identity,
            "confidence": group.confidence,
            "evidence": list(group.evidence),
            "diagnostics": dict(group.diagnostics),
        }

    @classmethod
    def mark_logical_target_group(
        cls,
        group: LogicalTargetGroup,
        candidates: Sequence[EntityRef],
    ) -> None:
        """Attach one immutable logical-unit contract to all of its members."""

        record = cls._logical_group_record(group)
        members = set(group.member_ids)
        for candidate in candidates:
            if candidate.node_id in members:
                candidate.metadata["_logical_target_group"] = dict(record)

    @staticmethod
    def _logical_group_id(
        kind: str,
        binding_index: int,
        identity: str,
        member_ids: Sequence[str],
    ) -> str:
        payload = json.dumps(
            {
                "kind": kind,
                "binding_index": int(binding_index),
                "identity": identity,
                "members": sorted({str(node_id) for node_id in member_ids}),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"logical_{kind}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    def _logical_group_scope_candidates(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
        binding: ActionTargetBinding,
    ) -> tuple[str | None, list[EntityRef]]:
        """Return high-confidence executable members in one exact IFC space."""

        if (
            len(plan.action_bindings) != 1
            or str(binding.action).casefold() not in LOGICAL_GROUP_ACTIONS
            or self._binding_cardinality(plan, binding) != "single"
            or not self._has_specific_space_scope(plan)
        ):
            return None, []
        scope_ids = self._room_scope_ids(plan, seeds)
        if len(scope_ids) != 1:
            return None, []
        scope_id = str(scope_ids[0])

        # An exact product mention is an entity instruction, never permission
        # to broaden that entity into a same-family cohort.
        if any(
            seed.kind == "entity"
            and seed.score >= 0.9
            and self.action_target_kind(seed) in {"object", "system"}
            and self._binding_matches(binding, seed)
            for seed in seeds
        ):
            return None, []
        if any(
            link.query_focus
            and link.kind in {"object", "system"}
            and link.candidate_count == 1
            and bool(link.node_ids)
            for link in plan.mention_links
        ):
            return None, []

        eligible: list[EntityRef] = []
        for candidate in candidates:
            room = self._authoritative_room(candidate.node_id)
            confidence = candidate.metadata.get("classification_confidence")
            if (
                not candidate.global_id
                or self.action_target_kind(candidate) != "object"
                or not self.action_compatible(
                    candidate,
                    action=binding.action,
                    requested_kind=binding.target_kind or plan.target_kind,
                )
                or not room
                or str(room.get("room_id")) != scope_id
                or str(room.get("provenance")) != "explicit"
                or float(room.get("confidence") or 0.0) < 0.9
                or (
                    confidence not in (None, "")
                    and float(confidence) < 0.55
                )
            ):
                continue
            eligible.append(candidate)
        return scope_id, sorted(eligible, key=lambda item: item.node_id)

    def _continuous_surface_logical_groups(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
        binding: ActionTargetBinding,
    ) -> list[LogicalTargetGroup]:
        scope_id, eligible = self._logical_group_scope_candidates(
            plan, seeds, candidates, binding
        )
        if scope_id is None or len(eligible) < 2:
            return []

        requested_roles = list(
            binding.target_roles
            or plan.target_roles
            or ([plan.target_role] if plan.target_role else [])
        )
        normalized_roles = {
            self._normalized_text(role).replace(" ", "_")
            for role in requested_roles
            if role
        }
        surface_roles = normalized_roles & CONTINUOUS_SURFACE_ROLES
        if len(surface_roles) != 1 or normalized_roles != surface_roles:
            return []
        function_types = list(binding.function_types) or self._target_function_types(
            plan
        )
        if "fault_condition" not in self._function_kinds(function_types):
            return []

        # Family and type must both be authored graph attributes.  IFC class
        # fallback identities are useful for aggregation, but too weak to
        # assert that several products form one physical surface.
        cohorts: dict[tuple[str, str], list[EntityRef]] = {}
        display: dict[tuple[str, str], tuple[str, str]] = {}
        for candidate in eligible:
            family = str(candidate.metadata.get("family") or "").strip()
            type_name = str(candidate.metadata.get("type_name") or "").strip()
            if (
                not family
                or not type_name
                or not self._semantic_role_match(candidate, list(surface_roles))
            ):
                continue
            key = (
                self._normalized_text(family),
                self._normalized_text(type_name),
            )
            cohorts.setdefault(key, []).append(candidate)
            display.setdefault(key, (family, type_name))
        if not cohorts:
            return []

        ranked = sorted(
            cohorts.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        dominant_key, dominant = ranked[0]
        runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
        eligible_with_identity = sum(len(items) for items in cohorts.values())
        coverage = (
            len(dominant) / eligible_with_identity
            if eligible_with_identity
            else 0.0
        )
        if (
            len(dominant) < 2
            or coverage < 0.60
            or (runner_up and len(dominant) < 2 * runner_up)
            or (
                len(ranked) > 1
                and len(dominant) == len(ranked[1][1])
            )
        ):
            return []

        family, type_name = display[dominant_key]
        member_ids = sorted(item.node_id for item in dominant)
        identity = f"{family}::{type_name}"
        group = LogicalTargetGroup(
            group_id=self._logical_group_id(
                "continuous_surface",
                binding.binding_index,
                identity,
                member_ids,
            ),
            group_kind="continuous_surface",
            binding_index=binding.binding_index,
            member_ids=member_ids,
            source="virtual_authored",
            identity=identity,
            confidence=min(
                1.0,
                min(
                    float(
                        item.metadata.get("classification_confidence")
                        if item.metadata.get("classification_confidence")
                        not in (None, "")
                        else 1.0
                    )
                    for item in dominant
                ),
            ),
            evidence=[
                f"scope:explicit_contains:{scope_id}",
                f"role:{next(iter(surface_roles))}",
                "function_kind:fault_condition",
                f"authored_family_type:{identity}",
            ],
            diagnostics={
                "member_count": len(member_ids),
                "eligible_count": eligible_with_identity,
                "coverage": coverage,
                "runner_up_count": runner_up,
                "dominance_ratio": (
                    len(dominant) / runner_up if runner_up else None
                ),
            },
        )
        return [group]

    @classmethod
    def _system_phrase_tokens(cls, plan: QueryPlan) -> tuple[bool, set[str]]:
        phrases = [
            *plan.target_phrases,
            *plan.target_names,
            *([plan.target_name] if plan.target_name else []),
            *(
                name
                for binding in plan.action_bindings
                for name in binding.target_names
            ),
        ]
        system_phrases = [
            phrase
            for phrase in phrases
            if "system" in cls._normalized_text(phrase).split()
        ]
        target_tokens = {
            cls._match_token(token)
            for phrase in system_phrases
            for token in cls._normalized_text(phrase).split()
            if token not in GENERIC_MATCH_TOKENS and token != "system"
        }
        return bool(system_phrases), target_tokens

    def _asset_system_logical_groups(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
        binding: ActionTargetBinding,
    ) -> list[LogicalTargetGroup]:
        scope_id, eligible = self._logical_group_scope_candidates(
            plan, seeds, candidates, binding
        )
        if scope_id is None or len(eligible) < 2:
            return []
        is_system_phrase, target_tokens = self._system_phrase_tokens(plan)
        function_types = list(binding.function_types) or self._target_function_types(
            plan
        )
        if (
            not is_system_phrase
            or not target_tokens
            or "asset_service" not in self._function_kinds(function_types)
        ):
            return []

        # A graph-declared group is stronger than a virtual authored cohort.
        self._annotate_explicit_cohort(plan, eligible)
        explicit: dict[str, tuple[set[str], dict[str, Any]]] = {}
        eligible_ids = {item.node_id for item in eligible}
        for candidate in eligible:
            record = candidate.metadata.get("_explicit_cohort")
            if (
                not isinstance(record, dict)
                or record.get("cohort_kind") != "system_or_asset"
            ):
                continue
            member_ids = {
                str(node_id) for node_id in record.get("member_ids", [])
            } & eligible_ids
            if len(member_ids) >= 2:
                explicit[str(record.get("cohort_id"))] = (member_ids, record)
        explicit_sets = {
            tuple(sorted(member_ids)): (cohort_id, record)
            for cohort_id, (member_ids, record) in explicit.items()
        }
        if explicit_sets:
            return [
                LogicalTargetGroup(
                    group_id=self._logical_group_id(
                        "asset_system",
                        binding.binding_index,
                        f"explicit:{cohort_id}",
                        member_ids,
                    ),
                    group_kind="asset_system",
                    binding_index=binding.binding_index,
                    member_ids=list(member_ids),
                    source="explicit_graph",
                    identity=str(record.get("cohort_label") or cohort_id),
                    confidence=1.0,
                    evidence=[
                        f"scope:explicit_contains:{scope_id}",
                        f"explicit_group:{cohort_id}",
                    ],
                    diagnostics={
                        "member_count": len(member_ids),
                        "relation": record.get("relation"),
                    },
                )
                for member_ids, (cohort_id, record) in sorted(
                    explicit_sets.items()
                )
            ]

        # Virtual cohorts use a namespace authored in the family identity and
        # a primary/accessory signature relation.  No namespace token or
        # benchmark family prefix is pre-declared: the query asset head locates
        # the boundary dynamically in each family string.
        family_members: dict[str, list[EntityRef]] = {}
        family_display: dict[str, str] = {}
        signatures: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {}
        for candidate in eligible:
            family = str(candidate.metadata.get("family") or "").strip()
            if not family:
                continue
            family_tokens = tuple(
                self._match_token(token)
                for token in self._normalized_text(family).split()
            )
            head_positions = [
                index
                for index, token in enumerate(family_tokens)
                if token in target_tokens
            ]
            if not head_positions:
                continue
            boundary = min(head_positions)
            namespace = family_tokens[:boundary]
            signature = frozenset(family_tokens[boundary:])
            if (
                not namespace
                or not signature
                or not (signature & target_tokens)
            ):
                continue
            key = self._normalized_text(family)
            family_members.setdefault(key, []).append(candidate)
            family_display.setdefault(key, family)
            signatures[key] = (namespace, signature)

        proposed: dict[tuple[str, ...], LogicalTargetGroup] = {}
        for primary_key, primary_members in sorted(family_members.items()):
            namespace, primary_signature = signatures[primary_key]
            accessories: list[str] = []
            for accessory_key in sorted(family_members):
                if accessory_key == primary_key:
                    continue
                other_namespace, accessory_signature = signatures[accessory_key]
                added = accessory_signature - primary_signature
                if (
                    other_namespace == namespace
                    and primary_signature < accessory_signature
                    and added
                ):
                    accessories.append(accessory_key)
            if not accessories:
                continue
            variant_keys = [primary_key, *accessories]
            counts = [len(family_members[key]) for key in variant_keys]
            if min(counts) <= 0 or max(counts) / min(counts) > 2.0:
                continue
            members = [
                candidate
                for key in variant_keys
                for candidate in family_members[key]
            ]
            member_ids = tuple(sorted(item.node_id for item in members))
            if len(member_ids) < 2:
                continue
            identity = " + ".join(
                family_display[key] for key in variant_keys
            )
            proposed[member_ids] = LogicalTargetGroup(
                group_id=self._logical_group_id(
                    "asset_system",
                    binding.binding_index,
                    identity,
                    member_ids,
                ),
                group_kind="asset_system",
                binding_index=binding.binding_index,
                member_ids=list(member_ids),
                source="virtual_authored",
                identity=identity,
                confidence=min(
                    float(
                        item.metadata.get("classification_confidence")
                        if item.metadata.get("classification_confidence")
                        not in (None, "")
                        else 1.0
                    )
                    for item in members
                ),
                evidence=[
                    f"scope:explicit_contains:{scope_id}",
                    "function_kind:asset_service",
                    "authored_namespace:" + " ".join(namespace),
                    "complementary_signatures:"
                    + "|".join(family_display[key] for key in variant_keys),
                ],
                diagnostics={
                    "member_count": len(member_ids),
                    "namespace": list(namespace),
                    "family_variants": [
                        {
                            "family": family_display[key],
                            "count": len(family_members[key]),
                        }
                        for key in variant_keys
                    ],
                    "balance_ratio": max(counts) / min(counts),
                },
            )
        return [
            proposed[key] for key in sorted(proposed)
        ]

    def _authored_asset_logical_groups(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
        binding: ActionTargetBinding,
    ) -> list[LogicalTargetGroup]:
        """Treat repeated instances of one authored asset class as one unit.

        Natural-language singularity does not guarantee that an IFC authoring
        tool emitted one identity.  A named maintainable family can have
        several executable instances in one explicit scope.  This legacy
        closure is deliberately narrower than a
        same-role cohort: the query must name the authored family, every
        member must be a hard-pass executable object, and exactly one family
        may satisfy the mention.  Exact entity mentions never trigger it.
        """

        if (
            len(plan.action_bindings) != 1
            or str(binding.action).casefold() not in LOGICAL_GROUP_ACTIONS
            or self._binding_cardinality(plan, binding) != "single"
            or len(candidates) < 2
        ):
            return []
        if any(
            seed.kind == "entity"
            and seed.score >= 0.9
            and self.action_target_kind(seed) in {"object", "system"}
            and self._binding_matches(binding, seed)
            for seed in seeds
        ):
            return []
        if any(
            link.query_focus
            and link.kind in {"object", "system"}
            and link.candidate_count == 1
            and bool(link.node_ids)
            for link in plan.mention_links
        ):
            return []

        phrases = [
            *binding.target_names,
            *plan.target_names,
            *([plan.target_name] if plan.target_name else []),
            *plan.target_phrases,
        ]
        phrase_token_sets = [
            {
                self._match_token(token)
                for token in self._normalized_text(phrase).split()
                if token not in GENERIC_MATCH_TOKENS
            }
            for phrase in phrases
        ]
        phrase_token_sets = [
            tokens for tokens in phrase_token_sets if len(tokens) >= 2
        ]
        if not phrase_token_sets:
            return []

        by_family: dict[str, list[EntityRef]] = {}
        family_display: dict[str, str] = {}
        for candidate in candidates:
            family = str(candidate.metadata.get("family") or "").strip()
            confidence = candidate.metadata.get("classification_confidence")
            if (
                not family
                or not candidate.global_id
                or self.action_target_kind(candidate) != "object"
                or (
                    confidence not in (None, "")
                    and float(confidence) < 0.55
                )
            ):
                continue
            family_tokens = {
                self._match_token(token)
                for token in self._normalized_text(family).split()
            }
            if not any(tokens.issubset(family_tokens) for tokens in phrase_token_sets):
                continue
            key = self._normalized_text(family)
            by_family.setdefault(key, []).append(candidate)
            family_display.setdefault(key, family)

        viable = {
            key: sorted(items, key=lambda item: item.node_id)
            for key, items in by_family.items()
            if len(items) >= 2
        }
        if len(viable) != 1:
            return []
        family_key, members = next(iter(viable.items()))

        # The cohort must remain within the explicit query scope.  A single
        # room is strongest; a storey-wide type query is accepted only when
        # every member explicitly carries the requested storey.
        if self._has_specific_space_scope(plan):
            scope_ids = set(self._room_scope_ids(plan, seeds))
            if len(scope_ids) != 1:
                return []
            for member in members:
                room = self._authoritative_room(member.node_id)
                if (
                    not room
                    or str(room.get("room_id")) not in scope_ids
                    or str(room.get("provenance")) != "explicit"
                ):
                    return []
            scope_evidence = f"scope:explicit_contains:{next(iter(scope_ids))}"
        elif plan.storey:
            if any(
                str(member.metadata.get("storey") or "").casefold()
                != str(plan.storey).casefold()
                for member in members
            ):
                return []
            scope_evidence = f"scope:storey:{plan.storey}"
        else:
            return []

        member_ids = [member.node_id for member in members]
        identity = family_display[family_key]
        return [
            LogicalTargetGroup(
                group_id=self._logical_group_id(
                    "authored_asset",
                    binding.binding_index,
                    identity,
                    member_ids,
                ),
                group_kind="authored_asset",
                binding_index=binding.binding_index,
                member_ids=member_ids,
                source="virtual_authored",
                identity=identity,
                confidence=min(
                    float(
                        member.metadata.get("classification_confidence")
                        if member.metadata.get("classification_confidence")
                        not in (None, "")
                        else 1.0
                    )
                    for member in members
                ),
                evidence=[
                    scope_evidence,
                    f"authored_family:{identity}",
                    "query_family_token_coverage:complete",
                ],
                diagnostics={
                    "member_count": len(member_ids),
                    "family": identity,
                    "scope_mode": (
                        "space" if self._has_specific_space_scope(plan)
                        else "storey"
                    ),
                },
            )
        ]

    def logical_target_groups(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
        binding: ActionTargetBinding,
    ) -> list[LogicalTargetGroup]:
        """Return safe logical-unit alternatives from a hard-pass pool.

        This method never changes binding cardinality and never selects among
        multiple alternatives.  The caller may adopt only a unique result,
        attach it to the binding, and then rerun the ordinary target audit.
        """

        allowed = set(binding.allowed_logical_group_kinds)
        groups: list[LogicalTargetGroup] = []
        if not allowed or "continuous_surface" in allowed:
            groups.extend(
                self._continuous_surface_logical_groups(
                    plan, seeds, candidates, binding
                )
            )
        if not allowed or "asset_system" in allowed:
            groups.extend(
                self._asset_system_logical_groups(
                    plan, seeds, candidates, binding
                )
            )
        if not allowed or "authored_asset" in allowed:
            groups.extend(
                self._authored_asset_logical_groups(
                    plan, seeds, candidates, binding
                )
            )
        deduped = {
            (group.group_kind, tuple(group.member_ids)): group
            for group in groups
        }
        return [
            deduped[key]
            for key in sorted(deduped, key=lambda item: (item[0], item[1]))
        ]

    @staticmethod
    def _mark_scope_semijoin(
        candidates: Sequence[EntityRef],
        scope_ids: Sequence[str],
        *,
        relation: str,
    ) -> None:
        """Preserve the deterministic scope operator without upgrading it.

        The marker is provenance, not a replacement for an authoritative IFC
        containment/service path.  Constraint evaluation may therefore keep
        ``scope=unknown`` while downstream selection can distinguish an
        operator-produced candidate from an unrelated retrieval hit.
        """
        normalized_scope_ids = sorted({str(node_id) for node_id in scope_ids})
        for candidate in candidates:
            candidate.metadata["_operator_scope_provenance"] = {
                "source": "deterministic_scope_semijoin",
                "relation": relation,
                "scope_ids": normalized_scope_ids,
            }
            candidate.match_reason = candidate.match_reason or "operator_scope_semijoin"

    def _filtered_nodes(self, plan: QueryPlan, seeds: Sequence[EntityRef]) -> list[EntityRef]:
        clauses: list[str] = []
        params: list[Any] = []
        if plan.target_kind == "space":
            clauses.append("ifc_class='IfcSpace'")
        elif plan.target_kind == "object":
            clauses.append("category='object'")
        elif plan.target_kind == "system":
            clauses.append("category='system'")
        if plan.target_ifc_class:
            clauses.append("lower(ifc_class)=lower(?)")
            params.append(plan.target_ifc_class)
        roles = plan.target_roles or ([plan.target_role] if plan.target_role else [])
        # Room/storey remain hard constraints.  Role/domain classification is
        # deliberately resolved after retrieval because IFC proxy objects often
        # retain the intended type only in their family/name text.
        # Space-type constraints describe the target only for a space query.
        # For object/system queries they describe a containing/served scope and
        # must be evaluated through the tri-state spatial semi-join below.
        if plan.target_space_type and plan.target_kind == "space":
            clauses.append("space_type=?")
            params.append(plan.target_space_type)
        if plan.storey:
            clauses.append("lower(storey)=lower(?)")
            params.append(plan.storey)
        names = (
            plan.target_names or ([plan.target_name] if plan.target_name else [])
            if self._uses_global_name_constraint(plan)
            else []
        )
        if names:
            alternatives: list[str] = []
            for name in names:
                tokens = list(self._constraint_tokens(name))
                if tokens:
                    alternatives.append("(" + " AND ".join("lower(search_text) LIKE ?" for _ in tokens) + ")")
                    params.extend(f"%{token}%" for token in tokens)
            if alternatives:
                clauses.append("(" + " OR ".join(alternatives) + ")")

        sql = "SELECT * FROM nodes"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Aggregation and PDDL enumeration must see the complete match set.  A
        # hidden terminal LIMIT would make the baseline depend on IFC row order.
        sql += " ORDER BY node_id"
        candidates = [self._entity(row) for row in self.connection.execute(sql, params)]
        if plan.target_kind in {"space", "object", "system"}:
            candidates = [
                candidate
                for candidate in candidates
                if self.action_target_kind(candidate) == plan.target_kind
            ]
        if roles:
            candidates = [
                candidate for candidate in candidates
                if self._semantic_role_match(candidate, roles)
            ]
        if self._uses_global_name_constraint(plan) and any(
            (plan.target_names, plan.target_family_terms, plan.target_type_terms, plan.target_keywords)
        ):
            candidates = [
                candidate for candidate in candidates
                if self._matches_family_type_name(plan, candidate)
            ]
        if plan.target_space_type and plan.target_kind == "space":
            candidates = [
                candidate for candidate in candidates
                if self._strict_space_match(candidate, plan.target_space_type)
            ]

        room_ids = self._room_scope_ids(plan, seeds)
        specific_space_scope = self._has_specific_space_scope(plan)
        if room_ids and specific_space_scope and plan.target_kind == "system":
            requested = set(room_ids)
            candidates = [
                candidate
                for candidate in candidates
                if self._system_served_space_ids(candidate.node_id) & requested
            ]
            self._mark_scope_semijoin(
                candidates, room_ids, relation="serves"
            )
        elif room_ids and specific_space_scope and plan.target_kind != "space":
            contained = self._contained_targets(room_ids)
            candidates = [candidate for candidate in candidates if candidate.node_id in contained]
            self._mark_scope_semijoin(
                candidates, room_ids, relation="contains"
            )
        elif room_ids and plan.target_kind == "space" and plan.scope_predicates:
            room_id_set = set(room_ids)
            candidates = [
                candidate for candidate in candidates
                if candidate.node_id in room_id_set
            ]
        elif plan.target_space_type and plan.target_kind == "object":
            # Do not let the SQL equality predicate erase low-confidence
            # spaces before the incomplete-BIM tri-state policy can examine
            # them.
            space_clauses = ["ifc_class='IfcSpace'"]
            space_params: list[Any] = []
            if plan.storey:
                space_clauses.append("lower(storey)=lower(?)")
                space_params.append(plan.storey)
            scope_spaces = [
                self._entity(row)
                for row in self.connection.execute(
                    "SELECT * FROM nodes WHERE " + " AND ".join(space_clauses),
                    space_params,
                )
            ]
            scope_space_ids = [
                space.node_id for space in scope_spaces
                if self._space_type_status(
                    space, plan.target_space_type
                ) != "fail"
            ]
            if scope_space_ids:
                contained = self._contained_targets(scope_space_ids)
                candidates = [candidate for candidate in candidates if candidate.node_id in contained]
                self._mark_scope_semijoin(
                    candidates, scope_space_ids, relation="contains"
                )
            else:
                candidates = []
        elif (plan.room or plan.room_names) and plan.target_kind == "space":
            candidates = [
                candidate for candidate in candidates
                if candidate.node_id in set(room_ids)
            ]

        global_function_types = self._target_function_types(plan)
        system_categories = self._system_categories(plan)
        if system_categories:
            system_filtered: list[EntityRef] = []
            for candidate in candidates:
                explicit_system_evidence = self._system_path_evidence(
                    candidate.node_id, system_categories
                )
                inferred_system_evidence = (
                    []
                    if explicit_system_evidence
                    else self._semantic_system_membership_evidence(
                        plan, candidate, system_categories, room_ids
                    )
                )
                if explicit_system_evidence or inferred_system_evidence:
                    system_filtered.append(candidate)
            candidates = system_filtered
        if plan.target_domain and not global_function_types:
            candidates = [
                candidate for candidate in candidates
                if self._semantic_domain_match(candidate, plan.target_domain)
            ]
        if global_function_types:
            candidates = [
                candidate for candidate in candidates
                if not (function_types := self._target_function_types(plan, candidate))
                or self._functional_path_evidence(candidate.node_id, function_types)
            ]

        candidates = self._apply_fault_operational_gate(plan, candidates)
        candidates = self._expand_explicit_asset_groups(
            plan, seeds, candidates
        )
        self._annotate_explicit_cohort(plan, candidates)

        if plan.operator in {"lookup", "path"} and not self._search_requires_exhaustive(plan):
            exact_targets: set[str] = set()
            for seed in seeds:
                if seed.kind != "entity" or seed.score < 0.9:
                    continue
                is_space = seed.ifc_class == "IfcSpace" or seed.metadata.get("category") == "space"
                if plan.target_kind == "space" and is_space or plan.target_kind == "object" and not is_space or (
                    plan.target_kind == "system"
                    and self.action_target_kind(seed) == "system"
                ):
                    exact_targets.add(seed.node_id)
            exact_matches = [candidate for candidate in candidates if candidate.node_id in exact_targets]
            if exact_matches:
                candidates = exact_matches

        return sorted(candidates, key=lambda candidate: candidate.node_id)

    def _contained_targets(
        self, space_ids: Sequence[str], *, prefer_authoritative: bool = True
    ) -> set[str]:
        if not space_ids:
            return set()
        # The graph DB and evidence policy are immutable for the backend's
        # lifetime.  The previous implementation reran this whole-graph window
        # query once per requested space, which made hierarchy validation
        # quadratic in practice.  Materialize the exact same global authority
        # ranking once, then answer calls by set union.  Filtering after the
        # global ROW_NUMBER is deliberate and preserves the old query's
        # semantics when one target has multiple candidate parents.
        if self._authoritative_containment_cache is None:
            evidence_placeholders = ",".join("?" for _ in self.include_evidence)
            rows = self.connection.execute(
                f"""
                WITH ranked AS (
                  SELECT e.source, e.target, e.evidence_type, e.confidence,
                         ROW_NUMBER() OVER (
                           PARTITION BY e.target
                           ORDER BY CASE e.evidence_type
                                      WHEN 'explicit' THEN 0
                                      WHEN 'inferred' THEN 1
                                      WHEN 'candidate' THEN 2 ELSE 3 END,
                                    e.confidence DESC, e.source ASC
                         ) AS authority_rank
                  FROM edges e
                  JOIN nodes parent ON parent.node_id=e.source
                  WHERE e.relation='contains'
                    AND parent.ifc_class='IfcSpace'
                    AND e.evidence_type IN ({evidence_placeholders})
                )
                SELECT source, target FROM ranked WHERE authority_rank=1
                """,
                self.include_evidence,
            )
            mutable: dict[str, set[str]] = {}
            for row in rows:
                mutable.setdefault(str(row["source"]), set()).add(
                    str(row["target"])
                )
            self._authoritative_containment_cache = {
                source: frozenset(targets) for source, targets in mutable.items()
            }
        result: set[str] = set()
        for space_id in space_ids:
            result.update(
                self._authoritative_containment_cache.get(str(space_id), ())
            )
        return result

    def _authoritative_room(self, target_id: str) -> dict[str, Any] | None:
        evidence_placeholders = ",".join("?" for _ in self.include_evidence)
        row = self.connection.execute(
            f"""
            SELECT e.source AS room_id, e.evidence_type, e.confidence,
                   n.global_id, n.name, n.long_name, n.storey
            FROM edges e JOIN nodes n ON n.node_id=e.source
            WHERE e.target=? AND e.relation='contains'
              AND n.ifc_class='IfcSpace'
              AND e.evidence_type IN ({evidence_placeholders})
            ORDER BY CASE e.evidence_type
                       WHEN 'explicit' THEN 0 WHEN 'inferred' THEN 1
                       WHEN 'candidate' THEN 2 ELSE 3 END,
                     e.confidence DESC, e.source ASC
            LIMIT 1
            """,
            (target_id, *self.include_evidence),
        ).fetchone()
        if row is None:
            return None
        return {
            "room_id": str(row["room_id"]),
            "room_guid": row["global_id"],
            "room": row["name"] or row["long_name"] or row["room_id"],
            "room_label": row["long_name"] or row["name"] or row["room_id"],
            "storey": row["storey"],
            "relation": "contains",
            "provenance": row["evidence_type"],
            "confidence": float(row["confidence"]),
        }

    def _system_member_ids(self, system_ids: Sequence[str]) -> set[str]:
        if not system_ids:
            return set()
        system_placeholders = ",".join("?" for _ in system_ids)
        evidence_placeholders = ",".join("?" for _ in self.include_evidence)
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT e.source FROM edges e
            JOIN nodes n ON n.node_id=e.source
            WHERE e.relation='assigned_to_system'
              AND e.target IN ({system_placeholders})
              AND e.evidence_type IN ({evidence_placeholders})
              AND n.category='object'
            """,
            (*system_ids, *self.include_evidence),
        )
        return {str(row["source"]) for row in rows}

    def _system_served_space_ids(self, system_id: str) -> set[str]:
        """Return direct or membership-derived spaces served by a system."""
        evidence_placeholders = ",".join("?" for _ in self.include_evidence)
        direct = self.connection.execute(
            f"""
            SELECT DISTINCT e.target AS space_id FROM edges e
            JOIN nodes s ON s.node_id=e.target
            WHERE e.source=? AND e.relation IN ('serves','affects')
              AND e.evidence_type IN ({evidence_placeholders})
              AND s.ifc_class='IfcSpace'
            """,
            (system_id, *self.include_evidence),
        )
        result = {str(row["space_id"]) for row in direct}
        members = self._system_member_ids([system_id])
        if not members:
            return result
        member_placeholders = ",".join("?" for _ in members)
        contained = self.connection.execute(
            f"""
            SELECT DISTINCT e.source AS space_id FROM edges e
            JOIN nodes s ON s.node_id=e.source
            WHERE e.target IN ({member_placeholders})
              AND e.relation='contains'
              AND e.evidence_type IN ({evidence_placeholders})
              AND s.ifc_class='IfcSpace'
            """,
            (*sorted(members), *self.include_evidence),
        )
        result.update(str(row["space_id"]) for row in contained)
        return result

    def _system_path_evidence(
        self,
        target_id: str,
        system_categories: Sequence[str],
    ) -> list[str]:
        if not system_categories:
            return []
        category_placeholders = ",".join("?" for _ in system_categories)
        evidence_placeholders = ",".join("?" for _ in self.include_evidence)
        target = self.get_node(target_id)
        if target is None:
            return []
        if self.action_target_kind(target) == "system":
            if str(target.metadata.get("system_category") or "") in system_categories:
                return [f"system:{target.node_id}:self"]
            return []
        rows = self.connection.execute(
            f"""
            SELECT e.edge_id, e.target AS system_id FROM edges e
            JOIN nodes s ON s.node_id=e.target
            WHERE e.source=? AND e.relation='assigned_to_system'
              AND s.system_category IN ({category_placeholders})
              AND e.evidence_type IN ({evidence_placeholders})
            ORDER BY e.confidence DESC, e.edge_id
            """,
            (target_id, *system_categories, *self.include_evidence),
        )
        return [
            f"system:{row['system_id']}:{row['edge_id']}" for row in rows
        ]

    @staticmethod
    def _system_category_domains(
        system_categories: Sequence[str],
    ) -> list[str]:
        inverse: dict[str, list[str]] = {}
        for domain, category in SYSTEM_CATEGORY_ALIASES.items():
            inverse.setdefault(category, []).append(domain)
        result: list[str] = []
        for category in system_categories:
            normalized = str(category)
            result.extend(inverse.get(normalized, ()))
            if normalized.endswith("_system"):
                result.append(normalized.removesuffix("_system"))
        return list(
            dict.fromkeys(
                domain
                for domain in result
                if domain in DOMAIN_SEMANTIC_TERMS
            )
        )

    def _semantic_system_membership_evidence(
        self,
        plan: QueryPlan,
        candidate: EntityRef,
        system_categories: Sequence[str],
        room_ids: Sequence[str],
    ) -> list[str]:
        """Infer a missing member edge from independent graph evidence.

        This fallback is intentionally narrower than ordinary text matching:
        it requires an actionable object explicitly contained in the selected
        space, a domain compatible with the requested system category, and a
        second positive asset/function discriminator.  The result is recorded
        as inferred semantic evidence; it is never reported as an explicit IFC
        ``assigned_to_system`` relation.
        """

        if (
            self.action_target_kind(candidate) != "object"
            or not room_ids
            or not self._has_specific_space_scope(plan)
        ):
            return []
        room = self._authoritative_room(candidate.node_id)
        if (
            not room
            or room.get("provenance") != "explicit"
            or str(room.get("room_id")) not in {str(item) for item in room_ids}
        ):
            return []
        compatible_domains = self._system_category_domains(system_categories)
        matched_domains = [
            domain
            for domain in compatible_domains
            if self._semantic_domain_match(candidate, domain)
        ]
        if not matched_domains:
            return []

        roles = list(
            dict.fromkeys(
                [
                    *(
                        plan.target_roles
                        or ([plan.target_role] if plan.target_role else [])
                    ),
                    *(
                        role
                        for binding in plan.action_bindings
                        for role in binding.target_roles
                    ),
                    *(
                        role
                        for _function_id, _is_method, ontology_roles
                        in self._function_ontology_definitions(
                            self._target_function_types(plan)
                        )
                        for role in ontology_roles
                    ),
                ]
            )
        )
        role_supported = bool(roles) and self._semantic_role_match(
            candidate, roles
        )
        names = list(
            dict.fromkeys(
                [
                    *(
                        plan.target_names
                        or ([plan.target_name] if plan.target_name else [])
                    ),
                    *(
                        name
                        for binding in plan.action_bindings
                        for name in binding.target_names
                    ),
                ]
            )
        )
        name_supported = bool(names) and self._entity_name_match(
            [candidate], names
        )
        function_types = self._target_function_types(plan)
        function_supported = bool(
            function_types
            and (
                self._functional_path_evidence(
                    candidate.node_id, function_types
                )
                or any(
                    ontology_roles
                    and self._semantic_role_match(candidate, ontology_roles)
                    for _function_id, _is_method, ontology_roles
                    in self._function_ontology_definitions(function_types)
                )
            )
        )
        if not (role_supported or name_supported or function_supported):
            return []

        evidence = [
            f"system_inferred_semantic:{category}:"
            f"domain={'|'.join(matched_domains)}:"
            f"room={room['room_id']}"
            for category in system_categories
        ]
        candidate.metadata["_inferred_system_membership"] = {
            "source": "graph_semantic_system_member_fallback",
            "categories": list(dict.fromkeys(system_categories)),
            "domains": matched_domains,
            "room_id": str(room["room_id"]),
            "room_relation": "contains",
            "room_provenance": "explicit",
            "role_supported": role_supported,
            "name_supported": name_supported,
            "function_supported": function_supported,
            "evidence": evidence,
        }
        candidate.match_reason = (
            candidate.match_reason or "operator_inferred_system_member"
        )
        return evidence

    @classmethod
    def _binding_matches(cls, binding: ActionTargetBinding, entity: EntityRef) -> bool:
        if not cls.action_compatible(
            entity,
            action=binding.action,
            requested_kind=binding.target_kind,
        ):
            return False
        branches = list(getattr(binding, "constraint_branches", []) or [])
        if branches:
            text_tokens = {
                cls._match_token(token)
                for token in cls._normalized_text(
                    cls._name_search_text(entity)
                ).split()
            }

            def matches_branch(branch: dict[str, list[str]]) -> bool:
                roles = list(branch.get("roles", []))
                names = list(branch.get("names", []))
                domains = list(branch.get("domains", []))
                return bool(
                    (not roles or cls._semantic_role_match(entity, roles))
                    and (
                        not names
                        or any(
                            (tokens := set(cls._constraint_tokens(name)))
                            and tokens.issubset(text_tokens)
                            for name in names
                        )
                    )
                    and (
                        not domains
                        or any(
                            cls._semantic_domain_match(entity, domain)
                            for domain in domains
                        )
                    )
                )

            if not any(matches_branch(branch) for branch in branches):
                return False
        else:
            if binding.target_roles and not cls._semantic_role_match(
                entity, binding.target_roles
            ):
                return False
            if binding.target_names:
                text = cls._normalized_text(cls._name_search_text(entity))
                text_tokens = {cls._match_token(token) for token in text.split()}
                if not any(
                    set(cls._constraint_tokens(name)).issubset(text_tokens)
                    and bool(cls._constraint_tokens(name))
                    for name in binding.target_names
                ):
                    return False
            domains = cls._binding_values(
                binding, "target_domains", "target_domain", "domains", "domain"
            )
            if domains and not any(
                cls._semantic_domain_match(entity, domain) for domain in domains
            ):
                return False
        target_ifc_classes = cls._binding_values(
            binding, "target_ifc_classes", "target_ifc_class"
        )
        if target_ifc_classes and str(entity.ifc_class or "").lower() not in {
            value.lower() for value in target_ifc_classes
        }:
            return False
        return True

    @staticmethod
    def _binding_cardinality(
        plan: QueryPlan,
        binding: ActionTargetBinding | None = None,
    ) -> str:
        value = getattr(binding, "cardinality_policy", None) if binding else None
        return str(value or getattr(plan, "cardinality_policy", "single") or "single")

    @classmethod
    def _answer_requires_exhaustive(
        cls,
        plan: QueryPlan,
        binding: ActionTargetBinding | None = None,
    ) -> bool:
        if binding is None and plan.action_bindings:
            return any(
                cls._answer_requires_exhaustive(plan, item)
                for item in plan.action_bindings
            )
        policy = cls._binding_cardinality(plan, binding)
        if plan.operator == "nearest":
            return policy == "all"
        if policy == "all":
            return True
        return bool(
            plan.requires_exhaustive
            and plan.operator in {"all_matching", "list", "path"}
        )

    @staticmethod
    def _search_requires_exhaustive(plan: QueryPlan) -> bool:
        value = getattr(plan, "search_exhaustive", None)
        return bool(
            IfcGraphBackend._answer_requires_exhaustive(plan)
            or value
            or any(
                len(getattr(binding, "constraint_branches", []) or []) > 1
                for binding in plan.action_bindings
            )
        )

    @classmethod
    def _target_function_types(
        cls, plan: QueryPlan, entity: EntityRef | None = None
    ) -> list[str]:
        global_types = list(
            dict.fromkeys(
                [
                    value
                    for predicate in plan.target_predicates
                    if predicate.predicate == "function" and predicate.required
                    for value in predicate.values
                ]
                + list(plan.function_intents)
            )
        )
        if entity is None or not any(
            binding.function_types for binding in plan.action_bindings
        ):
            return global_types
        return list(
            dict.fromkeys(
                function_type
                for binding in plan.action_bindings
                if cls._binding_matches(binding, entity)
                for function_type in binding.function_types
            )
        )

    def _function_cache_key(
        self, function_types: Sequence[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (
            tuple(sorted({str(value) for value in function_types if value})),
            tuple(self.include_evidence),
        )

    def _function_ontology_definitions(
        self, function_types: Sequence[str]
    ) -> tuple[tuple[str, bool, tuple[str, ...]], ...]:
        """Load intensional function constraints once per backend/policy."""
        key = self._function_cache_key(function_types)
        cached = self._function_ontology_cache.get(key)
        if cached is not None:
            return cached
        normalized_types = key[0]
        if not normalized_types:
            return ()
        placeholders = ",".join("?" for _ in normalized_types)
        rows = self.connection.execute(
            f"""
            SELECT f.node_id, v.predicate, v.value_text
            FROM nodes f
            LEFT JOIN node_values v
              ON v.node_id=f.node_id
             AND v.predicate IN (
               'property.function_kind', 'property.target_roles'
             )
            WHERE f.function_type IN ({placeholders})
            ORDER BY f.node_id, v.predicate, v.value_text
            """,
            normalized_types,
        )
        definitions: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = definitions.setdefault(
                str(row["node_id"]), {"is_method": False, "roles": []}
            )
            if (
                row["predicate"] == "property.function_kind"
                and row["value_text"] == "inspection_method"
            ):
                item["is_method"] = True
            elif (
                row["predicate"] == "property.target_roles"
                and row["value_text"] not in (None, "")
            ):
                item["roles"].append(str(row["value_text"]))
        result = tuple(
            (
                node_id,
                bool(item["is_method"]),
                tuple(dict.fromkeys(item["roles"])),
            )
            for node_id, item in definitions.items()
        )
        self._function_ontology_cache[key] = result
        return result

    def _function_evidence_index(
        self, function_types: Sequence[str]
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
        """Batch all explicit function paths instead of issuing N target SQLs."""
        key = self._function_cache_key(function_types)
        cached = self._function_evidence_index_cache.get(key)
        if cached is not None:
            return cached
        normalized_types, evidence_types = key
        if not normalized_types:
            return {}, {}
        function_placeholders = ",".join("?" for _ in normalized_types)
        evidence_placeholders = ",".join("?" for _ in evidence_types)
        by_target: dict[str, list[str]] = {}
        direct = self.connection.execute(
            f"""
            SELECT e.target AS target_id, f.node_id AS function_id, e.edge_id
            FROM nodes f JOIN edges e ON e.source=f.node_id
            WHERE f.function_type IN ({function_placeholders})
              AND e.relation='requires_inspection_of'
              AND e.evidence_type IN ({evidence_placeholders})
            ORDER BY e.target, f.node_id, e.edge_id
            """,
            (*normalized_types, *evidence_types),
        )
        for row in direct:
            by_target.setdefault(str(row["target_id"]), []).append(
                f"functional:{row['function_id']}:{row['edge_id']}"
            )
        system_paths = self.connection.execute(
            f"""
            SELECT os.source AS target_id, f.node_id AS function_id,
                   fs.edge_id AS function_edge, os.target AS system_id,
                   os.edge_id AS assignment_edge
            FROM nodes f
            JOIN edges fs ON fs.source=f.node_id
            JOIN edges os ON os.target=fs.target
            WHERE f.function_type IN ({function_placeholders})
              AND fs.relation='related_to_system'
              AND os.relation='assigned_to_system'
              AND fs.evidence_type IN ({evidence_placeholders})
              AND os.evidence_type IN ({evidence_placeholders})
            ORDER BY os.source, f.node_id, os.target, fs.edge_id, os.edge_id
            """,
            (*normalized_types, *evidence_types, *evidence_types),
        )
        for row in system_paths:
            by_target.setdefault(str(row["target_id"]), []).append(
                f"functional:{row['function_id']}:{row['function_edge']}:"
                f"{row['system_id']}:{row['assignment_edge']}"
            )
        related_systems: dict[str, list[str]] = {}
        related = self.connection.execute(
            f"""
            SELECT e.target AS system_id, f.node_id AS function_id, e.edge_id
            FROM nodes f JOIN edges e ON e.source=f.node_id
            WHERE f.function_type IN ({function_placeholders})
              AND e.relation='related_to_system'
              AND e.evidence_type IN ({evidence_placeholders})
            ORDER BY e.target, f.node_id, e.edge_id
            """,
            (*normalized_types, *evidence_types),
        )
        for row in related:
            related_systems.setdefault(str(row["system_id"]), []).append(
                f"functional:{row['function_id']}:{row['edge_id']}"
            )
        result = (
            {node_id: tuple(values) for node_id, values in by_target.items()},
            {
                node_id: tuple(values)
                for node_id, values in related_systems.items()
            },
        )
        self._function_evidence_index_cache[key] = result
        return result

    def _functional_path_evidence(
        self, target_id: str, function_types: Sequence[str]
    ) -> list[str]:
        """Return typed function->(system)->target evidence identifiers."""
        key = self._function_cache_key(function_types)
        if not key[0]:
            return []
        target_key = (str(target_id), *key)
        cached = self._target_function_evidence_cache.get(target_key)
        if cached is not None:
            return list(cached)

        by_target, related_systems = self._function_evidence_index(key[0])
        evidence = list(by_target.get(str(target_id), ()))
        entity = self.get_node(target_id)
        if entity is not None and self.action_target_kind(entity) == "system":
            evidence.extend(related_systems.get(str(target_id), ()))
        if not evidence and entity is not None:
            # Intensional ontology constraints are evaluated at query time
            # instead of materializing low-precision function mega-hubs.
            confidence = float(
                entity.metadata.get("classification_confidence") or 0.0
            )
            for function_id, is_method, ontology_roles in (
                self._function_ontology_definitions(key[0])
            ):
                if (
                    ontology_roles
                    and self._semantic_role_match(entity, ontology_roles)
                    and confidence >= 0.55
                ) or (is_method and not ontology_roles):
                    evidence.append(
                        f"functional_virtual:{function_id}:"
                        f"roles={'|'.join(ontology_roles)}"
                    )
        result = tuple(evidence)
        self._target_function_evidence_cache[target_key] = result
        return list(result)

    @staticmethod
    def _binding_values(binding: Any, *names: str) -> list[str]:
        for name in names:
            value = getattr(binding, name, None)
            if isinstance(value, str) and value:
                return [value]
            if isinstance(value, (list, tuple, set)) and value:
                return [str(item) for item in value if item]
        return []

    def candidate_constraint_status(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidate: EntityRef,
        binding: ActionTargetBinding | None = None,
    ) -> dict[str, Any]:
        """Build a compact pass/fail/unknown matrix for constrained selection.

        A missing IFC relation is ``unknown`` rather than an implicit failure.
        Explicit contradictions (wrong kind, room, storey, role/name/domain)
        remain hard failures.  The method is intentionally independent of the
        resolver implementation so deterministic and LLM selection share one
        auditable contract.
        """
        statuses: dict[str, str] = {}
        evidence: dict[str, list[str]] = {}
        executable_kind = self.action_target_kind(candidate)
        requested_kind = str(
            getattr(binding, "target_kind", None) or plan.target_kind or ""
        )
        action = str(getattr(binding, "action", "") or "")
        if requested_kind or action:
            statuses["kind"] = (
                "pass"
                if self.action_compatible(
                    candidate, action=action, requested_kind=requested_kind
                )
                else "fail"
            )
        else:
            statuses["kind"] = "unknown" if executable_kind == "none" else "pass"

        requested_room_ids = (
            set(self._room_scope_ids(plan, seeds))
            if self._has_specific_space_scope(plan)
            else set()
        )
        scope_status = "pass"
        scope_evidence: list[str] = []
        if self._has_specific_space_scope(plan):
            if not requested_room_ids:
                scope_status = "unknown"
            elif executable_kind == "space":
                if candidate.node_id not in requested_room_ids:
                    scope_status = "fail"
                else:
                    scope_status = self._space_scope_status(
                        plan, candidate.node_id
                    )
                    scope_evidence.append(f"scope:{candidate.node_id}")
                    if scope_status == "unknown":
                        scope_evidence.append(
                            f"space_type_unknown:{candidate.node_id}"
                        )
            elif executable_kind == "system":
                served = self._system_served_space_ids(candidate.node_id)
                if not served:
                    scope_status = "unknown"
                elif served & requested_room_ids:
                    matched_spaces = sorted(served & requested_room_ids)
                    space_states = [
                        self._space_scope_status(plan, item)
                        for item in matched_spaces
                    ]
                    scope_status = (
                        "pass" if "pass" in space_states else "unknown"
                    )
                    scope_evidence.extend(
                        f"served_space:{item}" for item in matched_spaces
                    )
                else:
                    scope_status = "fail"
            else:
                room = self._authoritative_room(candidate.node_id)
                if not room:
                    scope_status = "unknown"
                    scope_provenance = candidate.metadata.get(
                        "_operator_scope_provenance", {}
                    )
                    if isinstance(scope_provenance, dict):
                        hinted_scope_ids = {
                            str(node_id)
                            for node_id in scope_provenance.get("scope_ids", [])
                        }
                        if hinted_scope_ids & requested_room_ids:
                            scope_evidence.append(
                                "operator_scope_semijoin:"
                                + "|".join(sorted(hinted_scope_ids & requested_room_ids))
                            )
                elif str(room["room_id"]) in requested_room_ids:
                    room_id = str(room["room_id"])
                    scope_status = self._space_scope_status(plan, room_id)
                    scope_evidence.append(f"room:{room_id}")
                    if scope_status == "unknown":
                        scope_evidence.append(
                            f"space_type_unknown:{room_id}"
                        )
                else:
                    scope_status = "fail"
        actual_storey = candidate.metadata.get("storey")
        if not actual_storey and executable_kind == "object":
            actual_storey = (self._authoritative_room(candidate.node_id) or {}).get("storey")
        if plan.storey:
            if actual_storey and str(actual_storey).lower() != plan.storey.lower():
                scope_status = "fail"
            elif not actual_storey and scope_status != "fail":
                scope_status = "unknown"
        statuses["scope"] = scope_status
        evidence["scope"] = scope_evidence

        roles = (
            self._binding_values(binding, "target_roles", "target_role")
            if binding
            else list(
                plan.target_roles
                or ([plan.target_role] if plan.target_role else [])
            )
        )
        single_binding = binding is not None and len(plan.action_bindings) <= 1
        if single_binding and not roles:
            roles = list(
                plan.target_roles
                or ([plan.target_role] if plan.target_role else [])
            )
        if roles:
            if self._semantic_role_match(candidate, roles):
                statuses["role"] = "pass"
            elif candidate.metadata.get("role"):
                statuses["role"] = "fail"
            else:
                statuses["role"] = "unknown"
        else:
            statuses["role"] = "pass"

        names = (
            self._binding_values(binding, "target_names", "target_name")
            if binding
            else []
        )
        names_from_global_plan = False
        if (binding is None or single_binding) and not names and self._uses_global_name_constraint(plan):
            names = list(plan.target_names or ([plan.target_name] if plan.target_name else []))
            names.extend(plan.target_family_terms)
            names.extend(plan.target_type_terms)
            names.extend(plan.target_keywords)
            names_from_global_plan = True
        if not names:
            statuses["name"] = "pass"
        elif names_from_global_plan:
            statuses["name"] = (
                "pass" if self._matches_family_type_name(plan, candidate) else "fail"
            )
        else:
            binding_name_plan = replace(
                plan,
                target_name=names[0] if names else None,
                target_names=list(names),
                target_family_terms=[],
                target_type_terms=[],
                target_keywords=[],
            )
            statuses["name"] = (
                "pass"
                if self._matches_family_type_name(binding_name_plan, candidate)
                else "fail"
            )

        domains = (
            self._binding_values(
                binding, "target_domains", "target_domain", "domains", "domain"
            )
            if binding
            else ([plan.target_domain] if plan.target_domain else [])
        )
        if single_binding and not domains and plan.target_domain:
            domains = [plan.target_domain]
        if domains:
            if any(self._semantic_domain_match(candidate, domain) for domain in domains):
                statuses["domain"] = "pass"
            elif candidate.metadata.get("domain") not in (None, "", "unknown"):
                statuses["domain"] = "fail"
            else:
                statuses["domain"] = "unknown"
        else:
            statuses["domain"] = "pass"

        branches = list(
            getattr(binding, "constraint_branches", []) or []
        ) if binding is not None else []
        if branches:
            candidate_text_tokens = {
                self._match_token(token)
                for token in self._normalized_text(
                    self._name_search_text(candidate)
                ).split()
            }
            branch_outcomes: list[str] = []
            for branch in branches:
                branch_roles = list(branch.get("roles", []))
                branch_names = list(branch.get("names", []))
                branch_domains = list(branch.get("domains", []))
                branch_statuses: list[str] = []
                if branch_roles:
                    branch_statuses.append(
                        "pass"
                        if self._semantic_role_match(candidate, branch_roles)
                        else "fail"
                        if candidate.metadata.get("role")
                        else "unknown"
                    )
                if branch_names:
                    branch_statuses.append(
                        "pass"
                        if any(
                            (tokens := set(self._constraint_tokens(name)))
                            and tokens.issubset(candidate_text_tokens)
                            for name in branch_names
                        )
                        else "fail"
                    )
                if branch_domains:
                    branch_statuses.append(
                        "pass"
                        if any(
                            self._semantic_domain_match(candidate, domain)
                            for domain in branch_domains
                        )
                        else "fail"
                        if candidate.metadata.get("domain")
                        not in (None, "", "unknown")
                        else "unknown"
                    )
                branch_outcomes.append(
                    "fail"
                    if "fail" in branch_statuses
                    else "unknown"
                    if "unknown" in branch_statuses
                    else "pass"
                )
            statuses["semantic_branch"] = (
                "pass"
                if "pass" in branch_outcomes
                else "unknown"
                if "unknown" in branch_outcomes
                else "fail"
            )
            # Flat fields are a broad retrieval projection. The branch status
            # is the authoritative paired semantic constraint.
            statuses["role"] = "pass"
            statuses["name"] = "pass"
            statuses["domain"] = "pass"
            evidence["semantic_branch"] = [
                f"branch:{index}:{outcome}"
                for index, outcome in enumerate(branch_outcomes)
            ]

        function_types = (
            self._binding_values(binding, "function_types")
            if binding
            else self._target_function_types(plan)
        )
        if single_binding and not function_types:
            function_types = self._target_function_types(plan)
        if function_types:
            function_evidence = self._functional_path_evidence(
                candidate.node_id, function_types
            )
            statuses["function"] = "pass" if function_evidence else "unknown"
            evidence["function"] = function_evidence
        else:
            statuses["function"] = "pass"

        system_categories = (
            self._binding_values(binding, "system_categories")
            if binding
            else self._system_categories(plan)
        )
        if single_binding and not system_categories:
            system_categories = self._system_categories(plan)
        if system_categories:
            system_evidence = self._system_path_evidence(
                candidate.node_id, system_categories
            )
            inferred_membership = candidate.metadata.get(
                "_inferred_system_membership", {}
            )
            inferred_categories = (
                {
                    str(value)
                    for value in inferred_membership.get("categories", [])
                }
                if isinstance(inferred_membership, dict)
                else set()
            )
            inferred_evidence = (
                [
                    str(value)
                    for value in inferred_membership.get("evidence", [])
                ]
                if isinstance(inferred_membership, dict)
                and inferred_categories.intersection(system_categories)
                else []
            )
            statuses["system"] = (
                "pass"
                if system_evidence or inferred_evidence
                else "unknown"
            )
            system_evidence.extend(inferred_evidence)
            evidence["system"] = system_evidence
        else:
            statuses["system"] = "pass"

        overall = (
            "fail" if "fail" in statuses.values()
            else "unknown" if "unknown" in statuses.values()
            else "pass"
        )
        return {
            "candidate_id": candidate.node_id,
            "binding_index": getattr(binding, "binding_index", None),
            "overall": overall,
            "constraints": statuses,
            "evidence": evidence,
            "semantic_profile": self.semantic_profile(candidate),
        }

    def candidate_constraint_matrix(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
    ) -> list[dict[str, Any]]:
        bindings: Sequence[ActionTargetBinding | None] = plan.action_bindings or [None]
        return [
            self.candidate_constraint_status(plan, seeds, candidate, binding)
            for binding in bindings
            for candidate in sorted(candidates, key=lambda item: item.node_id)
        ]

    def _validate_target(
        self,
        plan: QueryPlan,
        entity: EntityRef,
        requested_room_ids: set[str],
    ) -> TargetValidation:
        checks: list[str] = []
        issues: list[str] = []
        evidence_ids = [f"target:{entity.node_id}"]
        is_space = entity.ifc_class == "IfcSpace"
        executable_kind = self.action_target_kind(entity)
        binding_matches = [
            binding for binding in plan.action_bindings
            if self._binding_matches(binding, entity)
        ]
        roles = list(
            dict.fromkeys(
                role
                for binding in binding_matches
                for role in binding.target_roles
            )
        )
        if not roles and len(plan.action_bindings) <= 1:
            roles = list(
                plan.target_roles
                or ([plan.target_role] if plan.target_role else [])
            )
        function_types = self._target_function_types(plan, entity)
        semantic_support = self._positive_target_semantic_support(
            plan,
            entity,
            binding_matches=binding_matches,
            function_types=function_types,
        )
        scope_action_target = bool(
            is_space and any(binding.target_kind == "space" for binding in binding_matches)
        )

        checks.append("guid")
        if not entity.global_id:
            issues.append("missing_guid")

        checks.append("action_target_kind")
        if executable_kind == "none":
            issues.append("support_only_target")
        if executable_kind not in ACTION_TARGET_KINDS:
            issues.append("invalid_action_target_kind")

        checks.append("target_kind_class")
        if (
            not binding_matches
            and plan.target_kind in {"space", "object", "system"}
            and executable_kind != plan.target_kind
            and not (
                plan.target_kind == "object"
                and is_space
                and plan.operator == "argmax"
                and scope_action_target
            )
        ):
            issues.append("target_kind_mismatch")
        if plan.target_ifc_class and str(entity.ifc_class).lower() != plan.target_ifc_class.lower():
            issues.append("ifc_class_mismatch")

        checks.append("role_class_allowlist")
        if roles and executable_kind == "object":
            if not self._semantic_role_match(entity, roles):
                issues.append("role_mismatch")

        checks.append("family_type_name")
        if (
            not is_space
            and self._uses_global_name_constraint(plan)
            and not self._matches_family_type_name(plan, entity)
        ):
            issues.append("family_type_name_mismatch")

        room = (
            self._authoritative_room(entity.node_id)
            if executable_kind == "object"
            else None
        )
        served_spaces = (
            self._system_served_space_ids(entity.node_id)
            if executable_kind == "system"
            else set()
        )
        checks.append("room_path_relation")
        if room:
            evidence_ids.append(f"room:{room['room_id']}")
        if executable_kind == "system" and served_spaces:
            evidence_ids.extend(f"served_space:{node_id}" for node_id in sorted(served_spaces))
        if requested_room_ids and executable_kind == "system":
            if not served_spaces:
                issues.append("missing_system_service_path")
            elif not (served_spaces & requested_room_ids):
                issues.append("outside_system_service_scope")
        elif requested_room_ids and not is_space:
            if not room:
                scope_provenance = entity.metadata.get(
                    "_operator_scope_provenance", {}
                )
                hinted_scope_ids = (
                    {
                        str(node_id)
                        for node_id in scope_provenance.get("scope_ids", [])
                    }
                    if isinstance(scope_provenance, dict)
                    else set()
                )
                if semantic_support and hinted_scope_ids & requested_room_ids:
                    checks.append("room_path_unknown_operator_scope")
                    evidence_ids.append(
                        "operator_scope_semijoin:"
                        + "|".join(sorted(hinted_scope_ids & requested_room_ids))
                    )
                else:
                    issues.append("missing_authoritative_room_path")
            elif str(room["room_id"]) not in requested_room_ids:
                issues.append("outside_authoritative_room")
            elif room.get("relation") != "contains":
                issues.append("required_relation_missing")

        checks.append("storey")
        actual_storey = entity.metadata.get("storey") or (room or {}).get("storey")
        if plan.storey and executable_kind == "system" and not actual_storey:
            served_storeys = {
                str(space.metadata.get("storey") or "").lower()
                for node_id in served_spaces
                if (space := self.get_node(node_id)) is not None
            }
            if plan.storey.lower() not in served_storeys:
                issues.append("storey_mismatch")
        elif plan.storey and str(actual_storey or "").lower() != plan.storey.lower():
            issues.append("storey_mismatch")

        checks.append("action_binding")
        if plan.action_bindings and not any(
            self._binding_matches(binding, entity) for binding in plan.action_bindings
        ):
            issues.append("action_binding_mismatch")

        binding_system_categories = list(
            dict.fromkeys(
                category
                for binding in binding_matches
                for category in binding.system_categories
            )
        )
        if binding_system_categories or any(
            binding.system_categories for binding in plan.action_bindings
        ):
            # In a multi-action plan, a system constraint belongs to the
            # action slot that declared it.  Applying the union to every
            # selected target would make an unrelated, otherwise valid action
            # fail its audit merely because another action uses system
            # evidence.
            system_categories = binding_system_categories
        else:
            system_categories = self._system_categories(plan)
        if system_categories:
            checks.append("system_path")
            system_evidence = self._system_path_evidence(
                entity.node_id, system_categories
            )
            if system_evidence:
                evidence_ids.extend(system_evidence)
            elif semantic_support:
                # A missing support relation in an incomplete BIM graph is an
                # unknown, not a contradiction.  Positive target semantics
                # are sufficient to retain the candidate while recording that
                # the support path could not be verified.
                checks.append("system_path_unknown")
            else:
                issues.append("missing_system_path:" + "|".join(system_categories))

        if function_types and not is_space:
            checks.append("functional_path")
            functional_evidence = self._functional_path_evidence(
                entity.node_id, function_types
            )
            if functional_evidence:
                evidence_ids.extend(functional_evidence)
            elif semantic_support:
                checks.append("functional_path_unknown")
            else:
                issues.append(
                    "missing_functional_path:" + "|".join(function_types)
                )

        return TargetValidation(
            target_id=entity.node_id,
            valid=not issues,
            checks=checks,
            issues=issues,
            evidence_ids=evidence_ids,
        )

    def _positive_target_semantic_support(
        self,
        plan: QueryPlan,
        entity: EntityRef,
        *,
        binding_matches: Sequence[ActionTargetBinding],
        function_types: Sequence[str],
    ) -> bool:
        """Return whether graph-backed target semantics support ``entity``.

        Function/system edges are support evidence and may be absent in an
        incomplete BIM.  They can therefore be treated as unknown only when a
        separate positive discriminator identifies the target: an explicit
        name, a compatible role/domain, or a function ontology role.  Domain
        evidence only qualifies when the query binding requested it and the
        candidate's graph-backed semantic profile matches it.
        """
        roles = list(
            dict.fromkeys(
                role
                for binding in binding_matches
                for role in binding.target_roles
            )
        )
        names = list(
            dict.fromkeys(
                name
                for binding in binding_matches
                for name in binding.target_names
            )
        )
        domains = list(
            dict.fromkeys(
                domain
                for binding in binding_matches
                for domain in binding.target_domains
            )
        )
        if len(plan.action_bindings) <= 1:
            roles.extend(
                role
                for role in (
                    plan.target_roles
                    or ([plan.target_role] if plan.target_role else [])
                )
                if role not in roles
            )
            global_names = list(
                plan.target_names
                or ([plan.target_name] if plan.target_name else [])
            )
            global_names.extend(plan.target_family_terms)
            global_names.extend(plan.target_type_terms)
            global_names.extend(plan.target_keywords)
            names.extend(name for name in global_names if name not in names)
            if plan.target_domain and plan.target_domain not in domains:
                domains.append(plan.target_domain)

        role_supported = bool(roles) and self._semantic_role_match(entity, roles)
        name_supported = bool(names) and bool(
            self._entity_name_match([entity], names)
        )
        domain_supported = bool(domains) and any(
            self._semantic_domain_match(entity, domain) for domain in domains
        )
        function_supported = any(
            ontology_roles and self._semantic_role_match(entity, ontology_roles)
            for _function_id, _is_method, ontology_roles
            in self._function_ontology_definitions(function_types)
        )
        return (
            role_supported
            or name_supported
            or domain_supported
            or function_supported
        )

    @staticmethod
    def _logical_group_coverage(
        binding: ActionTargetBinding,
        included: Sequence[EntityRef],
        valid_ids: set[str],
    ) -> tuple[bool, dict[str, Any]]:
        """Verify that one logical unit expands to its exact audited members."""

        selected = [
            entity
            for entity in included
            if entity.node_id in valid_ids
            and IfcGraphBackend._binding_matches(binding, entity)
        ]
        records_by_id: dict[str, dict[str, Any]] = {}
        for entity in selected:
            record = entity.metadata.get("_logical_target_group")
            if (
                isinstance(record, dict)
                and int(record.get("binding_index", -1))
                == int(binding.binding_index)
                and record.get("group_id")
            ):
                records_by_id[str(record["group_id"])] = record

        selected_ids = {entity.node_id for entity in selected}
        allowed = set(binding.allowed_logical_group_kinds)
        reasons: list[str] = []
        record: dict[str, Any] | None = None
        if len(records_by_id) != 1:
            reasons.append("logical_group_count_not_one")
        else:
            record = next(iter(records_by_id.values()))
        expected_ids = (
            {
                str(node_id)
                for node_id in record.get("member_ids", [])
            }
            if record is not None
            else set()
        )
        group_kind = str(record.get("group_kind", "")) if record else ""
        if allowed and group_kind not in allowed:
            reasons.append("logical_group_kind_not_allowed")
        if not expected_ids:
            reasons.append("logical_group_empty")
        if selected_ids != expected_ids:
            reasons.append("logical_group_member_set_mismatch")
        if any(
            not isinstance(entity.metadata.get("_logical_target_group"), dict)
            or entity.metadata["_logical_target_group"].get("group_id")
            != (record or {}).get("group_id")
            for entity in selected
        ):
            reasons.append("logical_group_member_contract_mismatch")

        missing_ids = sorted(expected_ids - selected_ids)
        extra_ids = sorted(selected_ids - expected_ids)
        complete = bool(record and expected_ids and not reasons)
        return complete, {
            "policy": "logical_group",
            "target_unit": "logical_group",
            "selected_unit_count": 1 if complete else 0,
            "selected_count": len(selected_ids),
            "selected_ids": sorted(selected_ids),
            "expected_count": len(expected_ids),
            "expected_ids": sorted(expected_ids)[:200],
            "missing_ids": missing_ids[:200],
            "extra_ids": extra_ids[:200],
            "logical_group_id": (
                str(record.get("group_id")) if record is not None else None
            ),
            "logical_group_kind": group_kind or None,
            "logical_group_issues": reasons,
            "_expected_ids_full": sorted(expected_ids),
            "_missing_ids_full": missing_ids,
            "_extra_ids_full": extra_ids,
        }

    def _binding_coverage(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        included: Sequence[EntityRef],
        valid_ids: set[str],
    ) -> tuple[dict[int, bool], dict[int, dict[str, Any]]]:
        coverage: dict[int, bool] = {}
        details: dict[int, dict[str, Any]] = {}
        for index, binding in enumerate(plan.action_bindings):
            if binding.target_unit == "logical_group":
                complete, detail = self._logical_group_coverage(
                    binding, included, valid_ids
                )
                coverage[index] = complete
                details[index] = detail
                continue
            expected_pool = self._binding_filtered_nodes(plan, seeds, binding)
            # The selector may conservatively reject candidates whose scope is
            # only ``unknown``.  Audit must not re-promote those rows into the
            # exhaustive expected set.  An unknown remains expected only when
            # the selector recorded its independent positive-evidence waiver;
            # hard-valid rows are always retained.
            positive_unknown_ids = {
                entity.node_id
                for entity in included
                if binding.binding_index
                in (
                    entity.metadata.get(
                        "_positive_evidence_binding_indices", []
                    )
                    or []
                )
            }
            audited_expected_pool: list[EntityRef] = []
            for entity in expected_pool:
                status = self.candidate_constraint_status(
                    plan, seeds, entity, binding=binding
                )
                overall = str(status.get("overall", "unknown")).lower()
                if overall == "pass" or (
                    overall == "unknown"
                    and entity.node_id in positive_unknown_ids
                ):
                    audited_expected_pool.append(entity)
            expected_pool = audited_expected_pool
            branches = list(getattr(binding, "constraint_branches", []) or [])
            if branches:
                selected_ids_union: set[str] = set()
                expected_ids_union: set[str] = set()
                missing_ids_union: set[str] = set()
                extra_ids_union: set[str] = set()
                branch_details: list[dict[str, Any]] = []
                branch_complete = True
                for branch_index, branch in enumerate(branches):
                    branch_binding = replace(
                        binding,
                        target_roles=list(branch.get("roles", []) or []),
                        target_names=list(branch.get("names", []) or []),
                        target_domains=list(branch.get("domains", []) or []),
                        constraint_branches=[],
                        cardinality_policy=str(
                            branch.get("cardinality", "single")
                        ),
                    )
                    branch_selected = sorted(
                        entity.node_id
                        for entity in included
                        if entity.node_id in valid_ids
                        and self._binding_matches(branch_binding, entity)
                    )
                    branch_expected = sorted(
                        entity.node_id
                        for entity in expected_pool
                        if self._binding_matches(branch_binding, entity)
                    )
                    branch_policy = str(branch.get("cardinality", "single"))
                    if branch_policy == "all":
                        branch_missing = sorted(
                            set(branch_expected) - set(branch_selected)
                        )
                        branch_extra = sorted(
                            set(branch_selected) - set(branch_expected)
                        )
                        complete = bool(branch_selected) and not (
                            branch_missing or branch_extra
                        )
                    else:
                        branch_missing = []
                        branch_extra = branch_selected[1:]
                        complete = len(branch_selected) == 1
                    selected_ids_union.update(branch_selected)
                    expected_ids_union.update(branch_expected)
                    missing_ids_union.update(branch_missing)
                    extra_ids_union.update(branch_extra)
                    branch_complete = branch_complete and complete
                    branch_details.append(
                        {
                            "branch_index": branch_index,
                            "policy": branch_policy,
                            "selected_ids": branch_selected,
                            "expected_ids": branch_expected[:200],
                            "missing_ids": branch_missing[:200],
                            "extra_ids": branch_extra[:200],
                            "complete": complete,
                        }
                    )
                selected = sorted(selected_ids_union)
                expected = sorted(expected_ids_union)
                missing_ids = sorted(missing_ids_union)
                extra_ids = sorted(extra_ids_union)
                coverage[index] = branch_complete
                details[index] = {
                    "policy": "coordinated",
                    "selected_count": len(selected),
                    "selected_ids": selected,
                    "expected_count": len(expected),
                    "expected_ids": expected[:200],
                    "missing_ids": missing_ids[:200],
                    "extra_ids": extra_ids[:200],
                    "branches": branch_details,
                    "_expected_ids_full": expected,
                    "_missing_ids_full": missing_ids,
                    "_extra_ids_full": extra_ids,
                }
                continue
            selected = sorted(
                entity.node_id for entity in included
                if entity.node_id in valid_ids and self._binding_matches(binding, entity)
            )
            expected = sorted(
                entity.node_id for entity in expected_pool
                if self._binding_matches(binding, entity)
            )
            selected_ids = set(selected)
            expected_ids = set(expected)
            policy = self._binding_cardinality(plan, binding)
            if policy == "single":
                extra_ids = selected[1:]
                missing_ids: list[str] = []
                complete = len(selected) == 1
            elif self._answer_requires_exhaustive(plan, binding):
                missing_ids = sorted(expected_ids - selected_ids)
                extra_ids = sorted(selected_ids - expected_ids)
                complete = bool(selected) and not missing_ids and not extra_ids
            else:
                missing_ids = []
                extra_ids = []
                complete = bool(selected)
            coverage[index] = complete
            details[index] = {
                "policy": policy,
                "selected_count": len(selected),
                "selected_ids": selected,
                "expected_count": len(expected),
                "expected_ids": expected[:200],
                "missing_ids": missing_ids[:200],
                "extra_ids": extra_ids[:200],
                # Full sets stay internal to the audit so correctness does not
                # depend on the 200-item diagnostic rendering limit.
                "_expected_ids_full": expected,
                "_missing_ids_full": missing_ids,
                "_extra_ids_full": extra_ids,
            }
        return coverage, details

    def _binding_filtered_nodes(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        binding: ActionTargetBinding,
    ) -> list[EntityRef]:
        """Evaluate one action slot without leaking another slot's filters."""
        return self._filtered_nodes(self._plan_for_binding(plan, binding), seeds)

    def _plan_for_binding(
        self,
        plan: QueryPlan,
        binding: ActionTargetBinding,
    ) -> QueryPlan:
        """Project a staged plan onto one independent action binding."""
        roles = list(binding.target_roles)
        names = list(binding.target_names)
        domains = list(getattr(binding, "target_domains", []) or [])
        single_binding = len(plan.action_bindings) == 1
        # With one action slot, global target constraints necessarily describe
        # that slot.  Graph-linked function mentions may populate the global
        # compatible role/name/domain fields while leaving the binding itself
        # function-only; retaining them prevents an exhaustive closure from
        # widening back to every object in the building.  Multi-action plans
        # remain strictly isolated so one action's constraints cannot leak into
        # another action.
        if single_binding:
            if not roles:
                roles = list(
                    plan.target_roles
                    or ([plan.target_role] if plan.target_role else [])
                )
            if not names:
                names = list(
                    plan.target_names
                    or ([plan.target_name] if plan.target_name else [])
                )
            if not domains and plan.target_domain:
                domains = [plan.target_domain]
        return replace(
            plan,
            target_kind=binding.target_kind or plan.target_kind,
            target_role=roles[0] if roles else None,
            target_roles=roles,
            target_domain=domains[0] if domains else None,
            target_name=names[0] if names else None,
            target_names=names,
            target_family_terms=(list(plan.target_family_terms) if single_binding else []),
            target_type_terms=(list(plan.target_type_terms) if single_binding else []),
            target_keywords=(
                list(plan.target_keywords or names) if single_binding else names
            ),
            function_intents=list(binding.function_types),
            target_binding_mode=binding.target_mode,
            target_predicates=[
                predicate
                for predicate in plan.target_predicates
                if predicate.predicate
                not in {"kind", "role", "name", "domain", "function", "system"}
            ],
            action_sequence=[binding.action],
            action_bindings=[binding],
            cardinality_policy=self._binding_cardinality(plan, binding),
            requires_exhaustive=self._answer_requires_exhaustive(plan, binding),
        )

    def target_audit(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        included: Sequence[EntityRef],
        *,
        excluded_limit: int = 200,
    ) -> TargetAudit:
        """Return bounded positive and negative evidence for target review."""
        included_ids = {entity.node_id for entity in included}
        roles = plan.target_roles or ([plan.target_role] if plan.target_role else [])
        names = (
            plan.target_names or ([plan.target_name] if plan.target_name else [])
            if self._uses_global_name_constraint(plan)
            else []
        )

        def record(entity: EntityRef, reasons: list[str] | None = None) -> dict[str, Any]:
            room = self._authoritative_room(entity.node_id)
            result: dict[str, Any] = {
                "target_id": entity.node_id,
                "guid": entity.global_id,
                "label": entity.label,
                "ifc_class": entity.ifc_class,
                "action_target_kind": self.action_target_kind(entity),
                "role": entity.metadata.get("role"),
                "family": entity.metadata.get("family"),
                "type": entity.metadata.get("type_name"),
                "storey": entity.metadata.get("storey") or (room or {}).get("storey"),
                "room_path": room,
                "relation": (room or {}).get("relation"),
                "provenance": (room or {}).get("provenance"),
                "confidence": (room or {}).get("confidence"),
                "logical_target_group": entity.metadata.get(
                    "_logical_target_group"
                ),
            }
            if reasons is not None:
                result["exclusion_reasons"] = reasons
            return result

        included_records = [record(entity) for entity in included]
        requested_room_ids = (
            set(self._room_scope_ids(plan, seeds))
            if self._has_specific_space_scope(plan)
            else set()
        )
        kind_clause = {
            "object": "category='object'",
            "system": "category='system'",
            "space": "ifc_class='IfcSpace'",
        }.get(plan.target_kind, "1=1")
        pool = (
            [
                self._entity(row)
                for row in self.connection.execute(
                    f"SELECT * FROM nodes WHERE {kind_clause} ORDER BY node_id LIMIT 5000"
                )
            ]
            if excluded_limit > 0
            else []
        )
        excluded_records: list[dict[str, Any]] = []
        for entity in pool:
            if entity.node_id in included_ids:
                continue
            reasons: list[str] = []
            if roles and not self._semantic_role_match(entity, roles):
                reasons.append("role_mismatch")
            if any((names, plan.target_family_terms, plan.target_type_terms, plan.target_keywords)) and not self._matches_family_type_name(plan, entity):
                reasons.append("name_or_type_mismatch")
            room = self._authoritative_room(entity.node_id)
            if requested_room_ids:
                if self.action_target_kind(entity) == "system":
                    if not (self._system_served_space_ids(entity.node_id) & requested_room_ids):
                        reasons.append("outside_system_service_scope")
                elif str((room or {}).get("room_id", "")) not in requested_room_ids:
                    reasons.append("outside_authoritative_room")
            if plan.target_space_type:
                scope = self.get_node(str((room or {}).get("room_id", ""))) if room else entity
                if scope is None or not self._strict_space_match(scope, plan.target_space_type):
                    reasons.append("outside_strict_space_taxonomy")
            if plan.storey and str(entity.metadata.get("storey") or (room or {}).get("storey") or "").lower() != plan.storey.lower():
                reasons.append("storey_mismatch")
            if not reasons:
                reasons.append("not_selected_by_operator")
            excluded_records.append(record(entity, reasons))
            if len(excluded_records) >= excluded_limit:
                break
        validations = [
            self._validate_target(plan, entity, requested_room_ids)
            for entity in included
        ]
        valid_ids = {
            validation.target_id for validation in validations if validation.valid
        }
        binding_coverage, binding_details = self._binding_coverage(
            plan, seeds, included, valid_ids
        )
        action_bindings_covered = all(binding_coverage.values())
        expected_ids: set[str] = set()
        answer_exhaustive = self._answer_requires_exhaustive(plan)
        binding_extra_ids: set[str] = set()
        binding_missing_ids: set[str] = set()
        exhaustive_extra_ids: set[str] = set()
        if plan.action_bindings:
            # Expected, missing, and extra targets are action-slot properties.
            # A target selected for a legitimate single-cardinality action is
            # not an extra merely because another action in the same plan is
            # exhaustive.  The previous union-level comparison made exactly
            # that mistake for mixed ``single`` + ``all`` plans.
            for index, binding in enumerate(plan.action_bindings):
                details = binding_details.get(index, {})
                selected_for_binding = {
                    str(node_id) for node_id in details.get("selected_ids", [])
                }
                extra_for_binding = {
                    str(node_id)
                    for node_id in details.get(
                        "_extra_ids_full", details.get("extra_ids", [])
                    )
                }
                binding_extra_ids.update(extra_for_binding)
                if (
                    self._answer_requires_exhaustive(plan, binding)
                    or binding.target_unit == "logical_group"
                ):
                    expected_for_binding = {
                        str(node_id)
                        for node_id in details.get(
                            "_expected_ids_full", details.get("expected_ids", [])
                        )
                    }
                    expected_ids.update(expected_for_binding)
                    binding_missing_ids.update(
                        str(node_id)
                        for node_id in details.get(
                            "_missing_ids_full", details.get("missing_ids", [])
                        )
                    )
                    exhaustive_extra_ids.update(extra_for_binding)
                else:
                    # For a non-exhaustive slot the audit validates the chosen
                    # binding; it does not invent an exact answer from the
                    # entire candidate pool.  Its selected target therefore
                    # belongs to the plan-level expected set.
                    expected_ids.update(selected_for_binding)
        elif answer_exhaustive:
            expected_ids = {
                entity.node_id for entity in self._filtered_nodes(plan, seeds)
            }
            binding_missing_ids.update(expected_ids - included_ids)
            exhaustive_extra_ids.update(included_ids - expected_ids)
            binding_extra_ids.update(exhaustive_extra_ids)

        missing_expected_ids = sorted(binding_missing_ids)
        invalid_target_ids = {
            validation.target_id for validation in validations if not validation.valid
        }
        bound_target_ids = {
            entity.node_id
            for entity in included
            if any(
                self._binding_matches(binding, entity)
                for binding in plan.action_bindings
            )
        }
        unbound_target_ids = (
            included_ids - bound_target_ids if plan.action_bindings else set()
        )
        extra_target_ids = sorted(
            invalid_target_ids | binding_extra_ids | unbound_target_ids
        )
        if missing_expected_ids or binding_extra_ids or unbound_target_ids:
            action_bindings_covered = False
        issue_counts: Counter[str] = Counter(
            issue for validation in validations for issue in validation.issues
        )
        invalid_ids = [item.target_id for item in validations if not item.valid]
        return TargetAudit(
            included_targets=included_records,
            excluded_candidates=excluded_records,
            target_validations=validations,
            validation_summary={
                "included_count": len(included_records),
                "valid_count": len(validations) - len(invalid_ids),
                "invalid_count": len(invalid_ids),
                "invalid_target_ids": invalid_ids,
                "issue_counts": dict(issue_counts),
                "action_binding_coverage": {
                    str(index): covered for index, covered in binding_coverage.items()
                },
                "action_binding_details": {
                    str(index): {
                        key: value
                        for key, value in detail.items()
                        if not key.startswith("_")
                    }
                    for index, detail in binding_details.items()
                },
                "binding_complete": action_bindings_covered,
                "expected_target_count": len(expected_ids),
                "missing_expected_target_count": len(missing_expected_ids),
                "missing_expected_target_ids": missing_expected_ids[:200],
                "extra_target_ids": extra_target_ids,
                "binding_extra_target_ids": sorted(binding_extra_ids)[:200],
                "unbound_target_ids": sorted(unbound_target_ids)[:200],
                "unexpected_exhaustive_target_ids": sorted(exhaustive_extra_ids)[:200],
                "exact_target_set": bool(
                    answer_exhaustive
                    and not missing_expected_ids
                    and not extra_target_ids
                ),
            },
            action_bindings_covered=action_bindings_covered,
            conflicting_extras=bool(extra_target_ids),
        )

    def constraint_valid_targets(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
    ) -> list[EntityRef]:
        """Deterministically convert retrieval context into action candidates."""
        requested_room_ids = (
            set(self._room_scope_ids(plan, seeds))
            if self._has_specific_space_scope(plan)
            else set()
        )
        result: list[EntityRef] = []
        seen: set[str] = set()
        for entity in sorted(candidates, key=lambda item: item.node_id):
            if entity.kind != "entity" or entity.node_id in seen:
                continue
            if self.action_target_kind(entity) == "none":
                continue
            validation = self._validate_target(plan, entity, requested_room_ids)
            if validation.valid:
                result.append(entity)
                seen.add(entity.node_id)
        return result

    def binding_complete(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
    ) -> bool:
        valid = self.constraint_valid_targets(plan, seeds, candidates)
        valid_ids = {entity.node_id for entity in valid}
        if plan.action_bindings:
            coverage, _ = self._binding_coverage(
                plan, seeds, valid, valid_ids
            )
            if not all(coverage.values()):
                return False
        if self._answer_requires_exhaustive(plan):
            exhaustive_bindings = [
                binding for binding in plan.action_bindings
                if self._answer_requires_exhaustive(plan, binding)
            ]
            if exhaustive_bindings:
                expected = {
                    entity.node_id
                    for binding in exhaustive_bindings
                    for entity in self._binding_filtered_nodes(plan, seeds, binding)
                    if self._binding_matches(binding, entity)
                }
            else:
                expected = {
                    entity.node_id
                    for entity in self._filtered_nodes(plan, seeds)
                }
            if expected - valid_ids:
                return False
        return bool(valid) if plan.action_bindings else True

    @classmethod
    def _family_type_identity(
        cls, entity: EntityRef
    ) -> tuple[tuple[str, str], str, str, str]:
        """Return a graph-backed family/type identity and stable display.

        IFC type names are scoped by their family.  Treating ``Type A`` as a
        global identity can incorrectly merge unrelated authored families.
        Missing family/type metadata falls back to class/domain taxonomy, never
        to the instance label, so aggregate output cannot become an ID dump.
        """
        metadata = entity.metadata
        family = str(
            metadata.get("family")
            or metadata.get("object_type")
            or entity.ifc_class
            or metadata.get("domain")
            or "unknown"
        ).strip()
        type_name = str(metadata.get("type_name") or family).strip()
        family_key = cls._normalized_text(family)
        type_key = cls._normalized_text(type_name)
        display = (
            f"{family}: {type_name}"
            if family_key and type_key and type_key != family_key
            else family or type_name
        )
        return (family_key, type_key), display, family, type_name

    @classmethod
    def _project_value(cls, plan: QueryPlan, entity: EntityRef) -> str:
        terms = {term.lower() for term in plan.property_terms}
        metadata = entity.metadata
        if "name" in terms:
            return str(metadata.get("name") or entity.label)
        if "ifc_class" in terms:
            return str(entity.ifc_class or entity.label)
        if "role" in terms and len(terms) == 1:
            return str(metadata.get("role") or entity.label)
        if "long_name" in terms:
            return str(metadata.get("long_name") or metadata.get("space_type") or entity.label)
        if "space_type" in terms:
            return str(metadata.get("space_type") or metadata.get("long_name") or entity.label)
        _, family_type_display, _, _ = cls._family_type_identity(entity)
        if "family" in terms:
            return family_type_display
        if "type" in terms:
            return family_type_display
        if "role" in terms:
            return str(metadata.get("role") or entity.ifc_class or entity.label)
        return str(metadata.get("long_name") or entity.label)

    @classmethod
    def _canonical_values(cls, values: Iterable[str]) -> list[str]:
        canonical: dict[str, str] = {}
        for value in values:
            rendered = " ".join(str(value).split())
            key = cls._normalized_text(rendered)
            if not key:
                continue
            previous = canonical.get(key)
            if previous is None or (rendered.casefold(), rendered) < (
                previous.casefold(), previous
            ):
                canonical[key] = rendered
        return [canonical[key] for key in sorted(canonical)]

    @classmethod
    def _hierarchical_group_count(
        cls,
        plan: QueryPlan,
        candidates: Sequence[EntityRef],
    ) -> tuple[str, list[EntityRef]]:
        """Aggregate a dominant family while retaining its type variants."""
        property_terms = {term.lower() for term in plan.property_terms}
        grouping = {
            str(value).lower() for value in getattr(plan, "target_grouping", [])
        }
        use_family = bool(
            not property_terms
            or "family" in property_terms
            or "family" in grouping
        ) and plan.target_kind != "space"
        if not use_family:
            # Type identity is family-scoped in IFC authoring tools. Other
            # explicit projections (role, class, normalized space type) retain
            # their requested identity.
            use_type_identity = "type" in property_terms or "type" in grouping
            grouped: dict[tuple[str, str], list[EntityRef]] = {}
            displays: dict[tuple[str, str], str] = {}
            for candidate in candidates:
                if use_type_identity:
                    identity, display, _, _ = cls._family_type_identity(candidate)
                else:
                    display = cls._project_value(plan, candidate)
                    identity = ("value", cls._normalized_text(display))
                grouped.setdefault(identity, []).append(candidate)
                previous = displays.get(identity)
                if previous is None or (display.casefold(), display) < (
                    previous.casefold(), previous
                ):
                    displays[identity] = display
            counts = {key: len(members) for key, members in grouped.items()}
            if not counts:
                return "", []
            maximum = max(counts.values())
            winners = sorted(
                (key for key, count in counts.items() if count == maximum),
                key=lambda key: (key, displays[key].casefold(), displays[key]),
            )
            selected = [
                candidate
                for key in winners
                for candidate in sorted(grouped[key], key=lambda item: item.node_id)
            ]
            return (
                f"{', '.join(displays[key] for key in winners)} "
                f"({maximum} instances)",
                selected,
            )

        family_members: dict[str, list[EntityRef]] = {}
        family_display: dict[str, str] = {}
        for candidate in candidates:
            _, _, family, _ = cls._family_type_identity(candidate)
            key = cls._normalized_text(family)
            family_members.setdefault(key, []).append(candidate)
            previous = family_display.get(key)
            if previous is None or (family.casefold(), family) < (
                previous.casefold(), previous
            ):
                family_display[key] = family
        if not family_members:
            return "", []
        maximum = max(len(members) for members in family_members.values())
        winner_keys = sorted(
            (key for key, members in family_members.items() if len(members) == maximum)
        )
        rendered_groups: list[str] = []
        selected: list[EntityRef] = []
        for key in winner_keys:
            members = sorted(family_members[key], key=lambda item: item.node_id)
            selected.extend(members)
            family = family_display[key]
            variant_counts: Counter[str] = Counter()
            variant_display: dict[str, str] = {}
            for member in members:
                type_name = str(member.metadata.get("type_name") or "").strip()
                variant_key = cls._normalized_text(type_name)
                if not variant_key or variant_key == key:
                    continue
                variant_counts[variant_key] += 1
                previous = variant_display.get(variant_key)
                if previous is None or (type_name.casefold(), type_name) < (
                    previous.casefold(), previous
                ):
                    variant_display[variant_key] = type_name
            if variant_counts:
                rendered_variants = ", ".join(
                    f"{variant_display[variant_key]} ({count})"
                    for variant_key, count in sorted(
                        variant_counts.items(), key=lambda item: item[0]
                    )
                )
                rendered_groups.append(f"{family}: {rendered_variants}")
            else:
                rendered_groups.append(family)
        return f"{'; '.join(rendered_groups)} ({maximum} instances)", selected

    @classmethod
    def _class_domain_histogram(cls, candidates: Sequence[EntityRef]) -> str:
        """Summarize an unconstrained object collection without listing IDs."""
        counts: Counter[tuple[str, str]] = Counter()
        display: dict[tuple[str, str], tuple[str, str]] = {}
        for candidate in candidates:
            ifc_class = str(candidate.ifc_class or "UnknownIfcClass").strip()
            profile = cls.semantic_profile(candidate)
            normalized_domains = [
                str(value).strip()
                for value in profile.get("normalized_domains", [])
                if str(value).strip() and str(value).strip() != "unknown"
            ]
            domain = (
                normalized_domains[0]
                if normalized_domains
                else str(candidate.metadata.get("domain") or "unknown").strip()
            )
            key = (cls._normalized_text(ifc_class), cls._normalized_text(domain))
            counts[key] += 1
            previous = display.get(key)
            rendered = (ifc_class, domain)
            if previous is None or tuple(value.casefold() for value in rendered) < tuple(
                value.casefold() for value in previous
            ):
                display[key] = rendered
        return "; ".join(
            f"{display[key][0]} / {display[key][1]} ({count})"
            for key, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0], display[item[0]]),
            )
        )

    @staticmethod
    def _is_broad_object_listing(plan: QueryPlan) -> bool:
        """Identify taxonomy-level listings rather than named asset requests."""
        return bool(
            plan.operator == "list"
            and plan.target_kind == "object"
            and not plan.action_sequence
            and not plan.action_bindings
            and not plan.property_terms
            and not plan.target_ifc_class
            and not plan.target_role
            and not plan.target_roles
            and not plan.target_name
            and not plan.target_names
            and not plan.target_family_terms
            and not plan.target_type_terms
            and not plan.target_keywords
        )

    def _synthetic_evidence(
        self,
        entities: Sequence[EntityRef],
        relation: str,
        source_label: str = "IFC query",
    ) -> list[TripleEvidence]:
        return [
            TripleEvidence(
                source_id="query",
                source_label=source_label,
                relation=relation,
                target_id=entity.node_id,
                target_label=entity.label,
                provenance="derived",
                confidence=1.0,
                features={"global_id": entity.global_id, **entity.metadata},
            )
            for entity in entities
        ]

    def _property_evidence(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        *,
        preferred_subject_ids: set[str] | None = None,
    ) -> list[TripleEvidence]:
        terms = [term.lower() for term in plan.property_terms if term]
        if not terms:
            return []
        prefer_scoped_subject = False
        if set(terms).issubset({"type", "value", "property", "properties"}):
            qualifiers = [
                *plan.target_family_terms,
                *plan.target_keywords,
            ]
            qualifier_tokens = sorted(
                {
                    self._match_token(token)
                    for qualifier in qualifiers
                    for token in self._normalized_text(qualifier).split()
                    if token not in {"type", "types", "value", "property", "properties"}
                }
            )
            if qualifier_tokens:
                terms = qualifier_tokens
                prefer_scoped_subject = True
        preferred_subject_ids = preferred_subject_ids or set()
        scored_rows: list[tuple[int, int, EntityRef, sqlite3.Row]] = []
        for seed in seeds:
            if seed.kind != "entity":
                continue
            rows = list(
                self.connection.execute(
                    "SELECT predicate, value_text FROM node_values WHERE node_id=? ORDER BY predicate, value_text",
                    (seed.node_id,),
                )
            )
            for row in rows:
                predicate_path = re.sub(
                    r"(?<=[a-z0-9])(?=[A-Z])",
                    " ",
                    str(row["predicate"]),
                )
                predicate = self._normalized_text(predicate_path)
                predicate_tokens = {
                    self._match_token(token) for token in predicate.split()
                }
                if terms == ["name"]:
                    score = 2 if predicate == "name" else 0
                else:
                    score = sum(
                        1
                        for term in terms
                        if self._match_token(self._normalized_text(term))
                        in predicate_tokens
                    )
                if score:
                    scored_rows.append(
                        (
                            score,
                            1
                            if (
                                seed.node_id in preferred_subject_ids
                            ) == prefer_scoped_subject
                            else 0,
                            seed,
                            row,
                        )
                    )
        if not scored_rows:
            return []
        best = max((score, preferred) for score, preferred, _, _ in scored_rows)
        result: list[TripleEvidence] = []
        for _, _, seed, row in sorted(
            (
                item for item in scored_rows
                if (item[0], item[1]) == best
            ),
            key=lambda item: (
                item[2].node_id,
                str(item[3]["predicate"]),
                str(item[3]["value_text"]),
            ),
        )[:50]:
            result.append(
                TripleEvidence(
                    source_id=seed.node_id,
                    source_label=seed.label,
                    relation=row["predicate"],
                    target_id="literal",
                    target_label=str(row["value_text"]),
                    provenance="explicit",
                    confidence=1.0,
                    features={"global_id": seed.global_id},
                )
            )
        return result

    def _property_subjects(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
        candidates: Sequence[EntityRef],
    ) -> tuple[list[EntityRef], set[str]]:
        subjects: dict[str, EntityRef] = {}
        preferred: set[str] = set()
        if plan.operator == "lookup" and self._has_specific_space_scope(plan):
            for node_id in self._room_scope_ids(plan, seeds):
                entity = self.get_node(node_id)
                if entity is not None:
                    subjects[entity.node_id] = entity
                    preferred.add(entity.node_id)
        for entity in candidates:
            subjects[entity.node_id] = entity
        if not subjects:
            for entity in seeds:
                if entity.kind == "entity":
                    subjects[entity.node_id] = entity
        return [subjects[node_id] for node_id in sorted(subjects)], preferred

    def _relation_reference_ids(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
    ) -> list[str]:
        """Resolve typed relation references without treating target seeds as scope."""
        references = getattr(plan, "relation_references", None) or []
        node_ids: set[str] = set()
        names: list[str] = []
        for reference in references:
            if isinstance(reference, str):
                if self.get_node(reference) is not None:
                    node_ids.add(reference)
                else:
                    names.append(reference)
                continue
            getter = reference.get if isinstance(reference, dict) else lambda key, default=None: getattr(reference, key, default)
            for key in ("node_id", "reference_node_id"):
                value = getter(key)
                if value and self.get_node(str(value)) is not None:
                    node_ids.add(str(value))
            for key in ("node_ids", "reference_node_ids", "resolved_node_ids", "entity_ids"):
                values = getter(key, []) or []
                node_ids.update(
                    str(value) for value in values
                    if value and self.get_node(str(value)) is not None
                )
            for key in ("values", "mentions"):
                values = getter(key, []) or []
                names.extend(str(value) for value in values if value)
            for key in ("value", "mention", "name", "reference_name"):
                value = getter(key)
                if value:
                    names.append(str(value))
        if names:
            spaces = [
                self._entity(row)
                for row in self.connection.execute(
                    "SELECT * FROM nodes WHERE ifc_class='IfcSpace' ORDER BY node_id"
                )
            ]
            node_ids.update(self._named_space_matches(spaces, names))
        if references:
            return sorted(node_ids)
        return sorted({
            seed.node_id for seed in seeds
            if seed.kind == "entity" and self.get_node(seed.node_id) is not None
        })

    @staticmethod
    def _nearest_requires_single(plan: QueryPlan) -> bool:
        policies = [
            str(getattr(binding, "cardinality_policy", "") or "")
            for binding in plan.action_bindings
        ]
        policy = str(
            getattr(plan, "scope_cardinality", "")
            or getattr(plan, "cardinality_policy", "")
            or "single"
        )
        return "all" not in policies and policy != "all"

    def _metric_geometry(self, node_id: str) -> dict[str, Any] | None:
        """Return geometry only when the graph marks it as measured/usable.

        IFC placements may yield a centroid even when representation geometry
        extraction failed. Treating that placement point as a measured shape
        creates false nearest-neighbour claims, so the explicit
        ``has_geometry`` flag is authoritative.
        """

        row = self.connection.execute(
            """
            SELECT g.*, n.attrs_json
            FROM geometry g JOIN nodes n ON n.node_id=g.node_id
            WHERE g.node_id=?
            """,
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            attrs = json.loads(row["attrs_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            return None
        geometry = attrs.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("has_geometry") is not True:
            return None
        center = tuple(row[key] for key in ("center_x", "center_y", "center_z"))
        if any(value is None or not math.isfinite(float(value)) for value in center):
            return None
        bbox_min = geometry.get("bbox_min")
        bbox_max = geometry.get("bbox_max")
        bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        if (
            isinstance(bbox_min, list)
            and isinstance(bbox_max, list)
            and len(bbox_min) == len(bbox_max) == 3
        ):
            try:
                lower = tuple(float(value) for value in bbox_min)
                upper = tuple(float(value) for value in bbox_max)
            except (TypeError, ValueError):
                pass
            else:
                if all(
                    math.isfinite(value) for value in (*lower, *upper)
                ):
                    bbox = (lower, upper)
        return {
            "center": tuple(float(value) for value in center),
            "bbox": bbox,
        }

    @staticmethod
    def _point_to_bbox_distance(
        point: Sequence[float],
        bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> float:
        lower, upper = bbox
        squared = 0.0
        for coordinate, minimum, maximum in zip(point, lower, upper):
            lo, hi = sorted((minimum, maximum))
            delta = lo - coordinate if coordinate < lo else coordinate - hi if coordinate > hi else 0.0
            squared += delta * delta
        return math.sqrt(squared)

    def _nearest_result(
        self,
        plan: QueryPlan,
        scored: Sequence[tuple[float, EntityRef]],
        relation: str,
    ) -> tuple[str, list[TripleEvidence], list[EntityRef]] | None:
        usable = [
            (float(distance), candidate)
            for distance, candidate in scored
            if math.isfinite(float(distance))
        ]
        if not usable:
            return None
        usable.sort(key=lambda item: (item[0], item[1].node_id))
        minimum = usable[0][0]
        selected = [
            candidate for distance, candidate in usable
            if abs(distance - minimum) < 1e-9
        ]
        if self._nearest_requires_single(plan) and len(selected) != 1:
            return "", [], []
        evidence = self._synthetic_evidence(selected, relation)
        for item in evidence:
            item.features["distance"] = minimum
        return ", ".join(item.label for item in selected), evidence, selected

    def _unique_explicit_containment_nearest(
        self,
        reference_ids: Sequence[str],
        candidates: dict[str, EntityRef],
    ) -> tuple[str, list[TripleEvidence], list[EntityRef]] | None:
        """Use explicit containment as a zero-hop topological metric.

        The rule is deliberately uniqueness-gated.  Containment proves that a
        candidate is inside the reference space, but it cannot rank two
        contained assets without geometry or another metric relation.
        """

        if not reference_ids or not candidates:
            return None
        space_reference_ids = [
            node_id
            for node_id in reference_ids
            if (
                (reference := self.get_node(node_id)) is not None
                and reference.ifc_class == "IfcSpace"
            )
        ]
        if not space_reference_ids:
            return None
        reference_placeholders = ",".join("?" for _ in space_reference_ids)
        candidate_ids = sorted(candidates)
        candidate_placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.connection.execute(
            f"""
            SELECT e.source, e.target, e.edge_id
            FROM edges e
            WHERE e.relation='contains'
              AND e.evidence_type='explicit'
              AND e.source IN ({reference_placeholders})
              AND e.target IN ({candidate_placeholders})
            ORDER BY e.source, e.target, e.edge_id
            """,
            (*space_reference_ids, *candidate_ids),
        ).fetchall()
        contained_ids = sorted({str(row["target"]) for row in rows})
        if len(contained_ids) != 1:
            return None
        selected = [candidates[contained_ids[0]]]
        evidence = self._synthetic_evidence(
            selected, "nearest_by_explicit_containment"
        )
        for item in evidence:
            item.provenance = "explicit"
            item.features.update(
                {
                    "topological_distance": 0,
                    "metric": "explicit_containment",
                    "reference_ids": space_reference_ids,
                }
            )
        return selected[0].label, evidence, selected

    def execute_operator(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
    ) -> tuple[str, list[TripleEvidence], list[EntityRef]]:
        if len(plan.action_bindings) > 1:
            # Each action owns its target constraints and cardinality.  Running
            # one flattened global plan can satisfy the first binding while
            # silently removing every candidate required by a later action.
            # Project and execute the same generic operator independently, then
            # merge only at the typed EntityRef/evidence boundary.
            answers: list[str] = []
            merged_candidates: dict[str, EntityRef] = {}
            merged_evidence: dict[
                tuple[str, str, str, str, str], TripleEvidence
            ] = {}
            for binding in sorted(
                plan.action_bindings, key=lambda item: item.binding_index
            ):
                binding_plan = self._plan_for_binding(plan, binding)
                answer, evidence, candidates = self.execute_operator(
                    binding_plan, seeds
                )
                if answer:
                    answers.append(answer)
                for candidate in candidates:
                    indices = candidate.metadata.setdefault(
                        "_operator_binding_indices", []
                    )
                    if binding.binding_index not in indices:
                        indices.append(binding.binding_index)
                    candidate.match_reason = candidate.match_reason or (
                        f"operator_binding_{binding.binding_index}"
                    )
                    existing = merged_candidates.get(candidate.node_id)
                    if existing is None:
                        merged_candidates[candidate.node_id] = candidate
                    else:
                        existing_indices = existing.metadata.setdefault(
                            "_operator_binding_indices", []
                        )
                        if binding.binding_index not in existing_indices:
                            existing_indices.append(binding.binding_index)
                for item in evidence:
                    key = (
                        item.source_id,
                        item.relation,
                        item.target_id,
                        item.direction,
                        item.provenance,
                    )
                    merged_evidence.setdefault(key, item)
            return (
                "; ".join(answers),
                list(merged_evidence.values()),
                [
                    merged_candidates[node_id]
                    for node_id in sorted(merged_candidates)
                ],
            )

        candidates = self._filtered_nodes(plan, seeds)
        property_subjects, preferred_property_subjects = self._property_subjects(
            plan, seeds, candidates
        )
        property_evidence = self._property_evidence(
            plan,
            property_subjects,
            preferred_subject_ids=preferred_property_subjects,
        )
        evidence: list[TripleEvidence] = []
        selected = candidates

        if plan.operator == "count":
            answer = str(len(candidates))
            evidence = self._synthetic_evidence(candidates[:200], "counted_match")
            return answer, evidence, candidates

        if plan.operator == "distinct":
            values = self._canonical_values(
                self._project_value(plan, candidate) for candidate in candidates
            )
            answer = ", ".join(values)
            evidence = self._synthetic_evidence(candidates, "distinct_match")
            return answer, evidence, candidates

        if plan.operator == "group_count":
            answer, winner_entities = self._hierarchical_group_count(plan, candidates)
            if not winner_entities:
                return "", property_evidence, []
            evidence = self._synthetic_evidence(winner_entities[:200], "member_of_most_common_group")
            return answer, evidence, winner_entities

        if plan.operator == "argmax":
            if "area" in plan.property_terms and plan.target_space_type:
                space_params: list[Any] = [plan.target_space_type]
                storey_clause = ""
                if plan.storey:
                    storey_clause = " AND lower(n.storey)=lower(?)"
                    space_params.append(plan.storey)
                area_rows = self.connection.execute(
                    f"""
                    SELECT n.node_id, max(v.value_num) AS area
                    FROM nodes n JOIN node_values v ON v.node_id=n.node_id
                    WHERE n.ifc_class='IfcSpace' AND n.space_type=?
                      AND lower(v.predicate) LIKE '%area%' AND v.value_num IS NOT NULL
                      {storey_clause}
                    GROUP BY n.node_id ORDER BY area DESC, n.node_id
                    """,
                    space_params,
                ).fetchall()
                if not area_rows:
                    geometry_params: list[Any] = [plan.target_space_type]
                    geometry_storey_clause = ""
                    if plan.storey:
                        geometry_storey_clause = " AND lower(n.storey)=lower(?)"
                        geometry_params.append(plan.storey)
                    area_rows = self.connection.execute(
                        f"""
                        SELECT n.node_id,
                               abs((g.max_x-g.min_x) * (g.max_y-g.min_y)) AS area
                        FROM nodes n JOIN geometry g ON g.node_id=n.node_id
                        WHERE n.ifc_class='IfcSpace' AND n.space_type=?
                          AND g.max_x IS NOT NULL AND g.min_x IS NOT NULL
                          AND g.max_y IS NOT NULL AND g.min_y IS NOT NULL
                          {geometry_storey_clause}
                        ORDER BY area DESC, n.node_id
                        """,
                        geometry_params,
                    ).fetchall()
                if area_rows:
                    max_area = float(area_rows[0]["area"])
                    winner_space_ids = [
                        row["node_id"]
                        for row in area_rows
                        if abs(float(row["area"]) - max_area) < 1e-9
                    ]
                    winner_spaces = [self.get_node(node_id) for node_id in winner_space_ids]
                    winner_spaces = [item for item in winner_spaces if item is not None]
                    if plan.target_kind == "object":
                        contained = self._contained_targets(winner_space_ids)
                        selected = [candidate for candidate in candidates if candidate.node_id in contained]
                    else:
                        selected = winner_spaces
                    evidence = self._synthetic_evidence(winner_spaces, "has_maximum_area")
                    evidence.extend(self._synthetic_evidence(selected, "contained_in_largest_space"))
                    answer = ", ".join(item.label for item in selected)
                    return answer, evidence, selected

            candidate_ids = {candidate.node_id for candidate in candidates}
            if candidate_ids:
                counts_by_room: Counter[str] = Counter()
                for target_id in candidate_ids:
                    room = self._authoritative_room(target_id)
                    if room:
                        counts_by_room[str(room["room_id"])] += 1
                rows = [
                    {"space_id": room_id, "n": count}
                    for room_id, count in sorted(
                        counts_by_room.items(), key=lambda item: (-item[1], item[0])
                    )
                ]
            else:
                rows = []
            if rows:
                maximum = int(rows[0]["n"])
                winner_ids = [row["space_id"] for row in rows if int(row["n"]) == maximum]
                selected = [self.get_node(node_id) for node_id in winner_ids]
                selected = [item for item in selected if item is not None]
                answer = f"{', '.join(item.label for item in selected)} ({maximum})"
                evidence = self._synthetic_evidence(selected, "has_maximum_count")
                return answer, evidence, selected

        if plan.operator == "nearest":
            reference_ids = self._relation_reference_ids(plan, seeds)
            candidate_by_id = {
                candidate.node_id: candidate
                for candidate in candidates
                if candidate.node_id not in reference_ids
            }
            containment_result = self._unique_explicit_containment_nearest(
                reference_ids, candidate_by_id
            )
            if containment_result is not None:
                return containment_result
            adjacent_ids: set[str] = set()
            if reference_ids and candidate_by_id:
                reference_placeholders = ",".join("?" for _ in reference_ids)
                candidate_placeholders = ",".join("?" for _ in candidate_by_id)
                evidence_placeholders = ",".join("?" for _ in self.include_evidence)
                adjacent_rows = self.connection.execute(
                    f"""
                    SELECT CASE WHEN source IN ({reference_placeholders}) THEN target ELSE source END AS node_id,
                           features_json
                    FROM edges
                    WHERE relation='adjacent_to'
                      AND evidence_type IN ({evidence_placeholders})
                      AND ((source IN ({reference_placeholders}) AND target IN ({candidate_placeholders}))
                        OR (target IN ({reference_placeholders}) AND source IN ({candidate_placeholders})))
                    ORDER BY node_id
                    """,
                    (
                        *reference_ids,
                        *self.include_evidence,
                        *reference_ids, *candidate_by_id,
                        *reference_ids, *candidate_by_id,
                    ),
                ).fetchall()
                edge_distances: dict[str, float] = {}
                for row in adjacent_rows:
                    node_id = str(row["node_id"])
                    if node_id not in candidate_by_id:
                        continue
                    adjacent_ids.add(node_id)
                    try:
                        features = json.loads(row["features_json"] or "{}")
                        distance = float(features.get("distance"))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if math.isfinite(distance):
                        edge_distances[node_id] = min(
                            distance, edge_distances.get(node_id, float("inf"))
                        )
                if edge_distances:
                    result = self._nearest_result(
                        plan,
                        [
                            (distance, candidate_by_id[node_id])
                            for node_id, distance in edge_distances.items()
                        ],
                        "nearest_adjacent_to_reference",
                    )
                    if result is not None:
                        return result
                if len(adjacent_ids) == 1:
                    selected = [candidate_by_id[next(iter(adjacent_ids))]]
                    evidence = self._synthetic_evidence(selected, "unique_adjacent_to_reference")
                    return selected[0].label, evidence, selected
            candidates = list(candidate_by_id.values())
            if adjacent_ids:
                candidates = [
                    candidate for candidate in candidates
                    if candidate.node_id in adjacent_ids
                ]
            reference_geometries: list[tuple[EntityRef, dict[str, Any]]] = []
            for node_id in reference_ids:
                reference = self.get_node(node_id)
                reference_geometry = self._metric_geometry(node_id)
                if reference is not None and reference_geometry is not None:
                    reference_geometries.append((reference, reference_geometry))
            if reference_geometries:
                distances: list[tuple[float, EntityRef]] = []
                for candidate in candidates:
                    geometry = self._metric_geometry(candidate.node_id)
                    if geometry is None:
                        continue
                    candidate_center = geometry["center"]
                    distance = min(
                        self._point_to_bbox_distance(
                            candidate_center, reference_geometry["bbox"]
                        )
                        if reference.ifc_class == "IfcSpace"
                        and reference_geometry["bbox"] is not None
                        else math.dist(
                            reference_geometry["center"], candidate_center
                        )
                        for reference, reference_geometry in reference_geometries
                    )
                    distances.append((distance, candidate))
                if distances:
                    result = self._nearest_result(
                        plan, distances, "nearest_to_reference_geometry"
                    )
                    if result is not None:
                        return result
            # Do not silently reinterpret an unresolved nearest request as an
            # exhaustive list.  Missing/tied metrics stay unresolved for the
            # constrained resolver or a reason-coded abstention.
            return "", [], []

        if plan.operator == "unconnected":
            connected = {
                row["node_id"]
                for row in self.connection.execute(
                    """
                    SELECT source AS node_id FROM edges WHERE relation IN ('connects_to','has_port')
                    UNION SELECT target AS node_id FROM edges WHERE relation IN ('connects_to','has_port')
                    """
                )
            }
            selected = [candidate for candidate in candidates if candidate.node_id not in connected]

        if property_evidence and plan.operator == "lookup":
            values = []
            for item in property_evidence:
                if item.target_label not in values:
                    values.append(item.target_label)
            matched_subject_ids = {item.source_id for item in property_evidence}
            matched_subjects = [
                subject for subject in property_subjects
                if subject.node_id in matched_subject_ids
            ]
            return ", ".join(values), property_evidence, matched_subjects

        if plan.property_terms and plan.operator in {"lookup", "all_matching", "list"}:
            values = self._canonical_values(
                self._project_value(plan, candidate) for candidate in candidates
            )
            evidence = self._synthetic_evidence(candidates, "projected_match")
            return ", ".join(values), evidence, candidates

        if self._is_broad_object_listing(plan):
            answer = self._class_domain_histogram(selected)
            evidence = self._synthetic_evidence(selected, "class_domain_histogram")
            return answer, evidence, selected

        answer = ", ".join(entity.label for entity in selected)
        relation = "matched_query" if plan.operator in {"lookup", "path"} else plan.operator
        evidence = self._synthetic_evidence(selected, relation)
        return answer, evidence, selected

    def execute_operator_result(
        self,
        plan: QueryPlan,
        seeds: Sequence[EntityRef],
    ) -> OperatorResult:
        """Execute a staged plan and expose completeness without breaking the
        historical tuple-returning API used by baseline tests and callers.
        """
        answer, evidence, candidates = self.execute_operator(plan, seeds)
        matched = [
            *(f"scope:{item.stage_id}:{item.predicate}" for item in plan.scope_predicates),
            *(f"target:{item.stage_id}:{item.predicate}" for item in plan.target_predicates),
        ]
        unresolved = list(dict.fromkeys(plan.unresolved_slots))
        if plan.action_bindings and not candidates:
            unresolved.append("action_targets")
        complete = bool(answer) or (
            bool(candidates) and not plan.property_terms
        )
        if plan.action_bindings:
            complete = self.binding_complete(plan, seeds, candidates)
            if not complete and "action_bindings" not in unresolved:
                unresolved.append("action_bindings")
        failed = [value for value in unresolved if value]
        return OperatorResult(
            answer=answer,
            candidates=list(candidates),
            evidence=list(evidence),
            matched_predicates=matched,
            failed_predicates=failed,
            evidence_paths=list(
                dict.fromkeys(
                    f"{item.source_id}|{item.relation}|{item.target_id}"
                    for item in evidence
                )
            )[:100],
            complete=complete and not failed,
            unresolved_slots=failed,
            excluded_counts={},
        )
