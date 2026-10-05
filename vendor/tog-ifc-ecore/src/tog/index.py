from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .models import IndexBuildReport

try:  # ``resource`` is unavailable on native Windows Python.
    import resource
except ImportError:  # pragma: no cover - exercised by Windows CI/import smoke.
    resource = None  # type: ignore[assignment]


GRAPH_SCHEMA_VERSION = "ifc-inspection-sqlite-v1"
VALUE_ALIAS_VERSION = "quantity-alias-v1"
ACTIONABILITY_VERSION = "ifc-actionability-v1"
# Immutable on-disk schema name retained because the frozen v5 graph was
# derived from this protected runtime format.  This is data compatibility,
# not a public v3 retrieval implementation.
PROTECTED_RUNTIME_GRAPH_SCHEMA = "text-gnn-runtime-graph-v3.1"
ACTION_TARGET_KINDS = frozenset({"space", "object", "system", "none"})
MAINTENANCE_SYSTEM_CATEGORIES = frozenset(
    {
        "hvac_system",
        "plumbing_system",
        "electrical_system",
        "fire_protection_system",
        "structural_system",
    }
)
EXPLICIT_SYSTEM_CLASSES = frozenset(
    {"IfcSystem", "IfcDistributionSystem", "IfcBuildingSystem"}
)
INFERENCE_CONFIG = {
    "candidate_connect_threshold": 0.75,
    "max_candidate_connects_per_node": 8,
    "include_all_elements": True,
    "taxonomy_version": "graph-vocabulary-normalization-v4",
    "containment_version": "authoritative-space-parent-v2",
    "target_validation_version": "backend-per-target-v1",
    "actionability_version": ACTIONABILITY_VERSION,
}
INFERENCE_CONFIG_HASH = hashlib.sha256(
    json.dumps(INFERENCE_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()[:16]


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_memory_mb() -> float:
    if resource is None:
        # Keep index generation portable without adding a process-monitoring
        # dependency. Zero means "not reported on this platform".
        return 0.0
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _scalar_values(value: Any, prefix: str, depth: int = 0) -> Iterator[tuple[str, Any]]:
    if depth > 6 or value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _scalar_values(item, child, depth + 1)
        return
    if isinstance(value, (list, tuple, set)):
        for item in list(value)[:200]:
            yield from _scalar_values(item, prefix, depth + 1)
        return
    if isinstance(value, (str, int, float, bool)):
        yield prefix, value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", value)
        if match:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _node_component_count(node: dict[str, Any]) -> int:
    properties = node.get("properties")
    components = properties.get("components") if isinstance(properties, dict) else None
    return len(components) if isinstance(components, (list, tuple, set)) else 0


def _actionability_for_node(node: dict[str, Any]) -> tuple[str, str, str]:
    level = str(node.get("level") or "")
    ifc_class = str(node.get("ifc_class") or "")
    guid = node.get("ifc_guid")
    if level == "L0_space":
        if ifc_class == "IfcSpace" and guid:
            return "space", "ifc_class", "explicit IfcSpace with IFC GUID"
        return "none", "support_only", "spatial hierarchy node is not an executable space"
    if level == "L1_object":
        if guid:
            return "object", "ifc_identity", "GUID-backed IFC inspection object"
        return "none", "support_only", "object has no stable IFC GUID"
    if level == "L2_system":
        if not guid:
            return "none", "support_only", "inferred system has no executable IFC identity"
        if ifc_class in EXPLICIT_SYSTEM_CLASSES:
            return "system", "ifc_class", "explicit IFC system with IFC GUID"
        if ifc_class == "IfcGroup":
            name = str(node.get("name") or "").strip().lower()
            source = str(node.get("classification_source") or "")
            category = str(node.get("system_category") or "")
            confidence = float(node.get("classification_confidence") or 0.0)
            if (
                not name.startswith("model group:")
                and category in MAINTENANCE_SYSTEM_CATEGORIES
                and _node_component_count(node) > 0
                and confidence >= 0.8
                and source in {"keyword", "property_set", "ifc_class"}
            ):
                return (
                    "system",
                    "classified_ifc_group",
                    "high-confidence maintenance group with IFC GUID and members",
                )
        return "none", "support_only", "generic, modeling, empty, or low-confidence IFC group"
    if level == "L3_function":
        return "none", "support_only", "abstract function ontology node"
    return "none", "support_only", "unknown inspection-graph level"


def _runtime_actionability_sha256(nodes: list[dict[str, Any]]) -> str:
    rows = [
        {
            "node_id": str(node.get("node_id") or ""),
            "action_target_kind": node.get("action_target_kind"),
            "actionability_source": node.get("actionability_source"),
            "actionability_reason": node.get("actionability_reason"),
        }
        for node in nodes
    ]
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ensure_actionability_contract(graph: dict[str, Any]) -> dict[str, Any]:
    """Backfill and validate actionability for rich, compact, and frozen graphs."""

    nodes = graph.get("nodes", [])
    graph_metadata = graph.get("metadata")
    runtime = (
        graph_metadata.get("runtime_graph_v31")
        if isinstance(graph_metadata, dict)
        else None
    )
    protected = isinstance(runtime, dict) and runtime.get(
        "actionability_protected"
    ) is True
    if protected and (
        runtime.get("schema_version") != PROTECTED_RUNTIME_GRAPH_SCHEMA
        or runtime.get("node_count") != len(nodes)
        or runtime.get("actionability_sha256")
        != _runtime_actionability_sha256(nodes)
    ):
        raise RuntimeError("protected runtime actionability lineage mismatch")
    counts: Counter[str] = Counter()
    level_kind_counts: dict[str, Counter[str]] = {}
    actionable_without_guid: list[str] = []
    invalid_kinds: list[str] = []
    for node in nodes:
        if protected:
            kind = str(node.get("action_target_kind") or "")
            if not str(node.get("actionability_source") or "").strip() or not str(
                node.get("actionability_reason") or ""
            ).strip():
                raise RuntimeError(
                    "protected runtime actionability is incomplete"
                )
        else:
            kind, source, reason = _actionability_for_node(node)
            node["action_target_kind"] = kind
            node["actionability_source"] = source
            node["actionability_reason"] = reason
        counts[kind] += 1
        level = str(node.get("level") or "unknown")
        level_kind_counts.setdefault(level, Counter())[kind] += 1
        if kind not in ACTION_TARGET_KINDS:
            invalid_kinds.append(str(node.get("node_id")))
        if kind != "none" and not node.get("ifc_guid"):
            actionable_without_guid.append(str(node.get("node_id")))

    system_identity_counts: Counter[str] = Counter()
    for node in nodes:
        if node.get("level") != "L2_system":
            continue
        if not node.get("ifc_guid"):
            system_identity_counts["inferred_without_guid"] += 1
        elif node.get("ifc_class") in EXPLICIT_SYSTEM_CLASSES:
            system_identity_counts["explicit_ifc_system"] += 1
        elif node.get("action_target_kind") == "system":
            system_identity_counts["targetable_ifc_group"] += 1
        else:
            system_identity_counts["support_ifc_group"] += 1

    report = {
        "version": ACTIONABILITY_VERSION,
        "counts_by_kind": dict(sorted(counts.items())),
        "counts_by_level_and_kind": {
            level: dict(sorted(values.items()))
            for level, values in sorted(level_kind_counts.items())
        },
        "system_identity_counts": dict(sorted(system_identity_counts.items())),
        "actionable_without_ifc_guid": actionable_without_guid[:50],
        "invalid_action_target_kinds": invalid_kinds[:50],
        "valid": not actionable_without_guid and not invalid_kinds,
    }
    metadata = graph.setdefault("metadata", {})
    metadata["actionability_version"] = ACTIONABILITY_VERSION
    metadata["node_counts_by_action_target_kind"] = report["counts_by_kind"]
    validation = metadata.setdefault("validation", {})
    validation["actionability"] = report
    validation.setdefault("checks", {})[
        "11_action_targets_have_stable_ifc_identity"
    ] = report["valid"]
    return report


def _insert_quantity_aliases(connection: sqlite3.Connection) -> None:
    """Expose IFC exports that encode QTOs as property sets as quantity relations."""
    connection.execute(
        """
        INSERT INTO node_values(node_id, predicate, value_text, value_num, value_type)
        SELECT v.node_id,
               'quantity.' || substr(v.predicate, length('property.') + 1),
               v.value_text, v.value_num, v.value_type
        FROM node_values v
        WHERE lower(v.predicate) LIKE 'property.%'
          AND (
            lower(v.predicate) LIKE '%quantitytakeoff%'
            OR lower(v.predicate) LIKE '%basequantities%'
            OR lower(v.predicate) LIKE 'property.qto_%'
          )
          AND NOT EXISTS (
            SELECT 1 FROM node_values existing
            WHERE existing.node_id=v.node_id
              AND existing.predicate='quantity.' || substr(v.predicate, length('property.') + 1)
              AND existing.value_text=v.value_text
          )
        """
    )


def _node_id(entity: Any) -> str:
    guid = getattr(entity, "GlobalId", None)
    return f"ifc_{guid}" if guid else f"ifc_id_{int(entity.id())}"


def _clean(value: Any) -> str | None:
    if value in (None, "", "$"):
        return None
    return str(value)


def _role_from_text(ifc_class: str, text: str) -> tuple[str, str]:
    value = f"{ifc_class} {text}".lower()
    rules = [
        ("fire_protection", "sprinkler", ("sprinkler", "fire suppression")),
        ("fire_protection", "fire_extinguisher", ("extinguisher",)),
        ("hvac", "duct", ("duct",)),
        ("hvac", "air_terminal", ("diffuser", "air terminal", "grille")),
        ("plumbing", "drain", ("drain", "waste terminal")),
        ("plumbing", "sanitary_fixture", ("sink", "toilet", "lavatory", "urinal")),
        ("plumbing", "water_supply", ("faucet", "eyewash", "gas manifold", "gas turret")),
        ("electrical", "light_fixture", ("light fixture", "pendant", "downlight", "led strip")),
        ("electrical", "panel", ("panel", "distribution board")),
        ("electrical", "outlet", ("outlet", "receptacle")),
        ("electrical", "switch", ("switch",)),
        ("electrical", "electrical_device", ("projector", "display", "tv", "monitor")),
        ("architectural", "door", ("ifcdoor", " door")),
        ("architectural", "furnishing", ("furnishing", "furniture")),
    ]
    for domain, role, terms in rules:
        if any(term in value for term in terms):
            return domain, role
    return "unknown", "unknown"


def _space_type(text: str) -> str:
    value = text.lower()
    rules = [
        ("conference", ("conference", "conf ", "group rm", "meeting")),
        ("classroom", ("classroom", "lecture", "teaching")),
        ("lab", ("lab", "laboratory", "workshop", "research")),
        ("office", ("office", "faculty", "director", "admin")),
        ("corridor", ("corridor", "circulation", "hall")),
        ("restroom", ("restroom", "mens", "womens", "toilet")),
        ("kitchenette", ("kitchenette",)),
        ("stair", ("stair",)),
        ("lobby", ("lobby", "vestibule")),
        ("storage", ("storage", "closet")),
    ]
    for result, terms in rules:
        if any(term in value for term in terms):
            return result
    return "unknown"


class CompactIfcBuilder:
    """Portable fallback builder used when the richer workspace builder is absent."""

    def build(self, source_path: Path) -> dict[str, Any]:
        import ifcopenshell
        import ifcopenshell.util.element as element_util
        import ifcopenshell.util.placement as placement_util

        model = ifcopenshell.open(str(source_path))
        spatial: list[Any] = []
        for class_name in ("IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace"):
            try:
                spatial.extend(model.by_type(class_name, include_subtypes=False))
            except TypeError:
                spatial.extend(model.by_type(class_name))
        elements = list(model.by_type("IfcElement"))
        entities = {int(entity.id()): entity for entity in spatial + elements if int(entity.id()) > 0}

        nodes: list[dict[str, Any]] = []
        for entity in entities.values():
            ifc_class = str(entity.is_a())
            name = _clean(getattr(entity, "Name", None))
            long_name = _clean(getattr(entity, "LongName", None))
            object_type = _clean(getattr(entity, "ObjectType", None))
            tag = _clean(getattr(entity, "Tag", None))
            text = " ".join(x for x in (name, long_name, object_type, tag) if x)
            domain, role = _role_from_text(ifc_class, text)
            container = None
            try:
                container = element_util.get_container(entity)
            except Exception:
                pass
            storey = None
            current = container
            for _ in range(8):
                if current is None:
                    break
                if current.is_a("IfcBuildingStorey"):
                    storey = _clean(getattr(current, "Name", None))
                    break
                try:
                    current = element_util.get_container(current)
                except Exception:
                    break
            if entity.is_a("IfcBuildingStorey"):
                storey = name

            properties: dict[str, Any] = {}
            quantities: dict[str, Any] = {}
            try:
                properties = element_util.get_psets(entity, psets_only=True)
                quantities = element_util.get_psets(entity, qtos_only=True)
            except Exception:
                pass

            geometry = {"bbox_min": None, "bbox_max": None, "centroid": None, "has_geometry": False}
            placement = getattr(entity, "ObjectPlacement", None)
            if placement is not None:
                try:
                    matrix = placement_util.get_local_placement(placement)
                    geometry["centroid"] = [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
                except Exception:
                    pass

            nodes.append(
                {
                    "node_id": _node_id(entity),
                    "ifc_guid": _clean(getattr(entity, "GlobalId", None)),
                    "ifc_class": ifc_class,
                    "name": name,
                    "long_name": long_name,
                    "type": object_type,
                    "level": "L0_space" if entity.is_a("IfcSpatialStructureElement") or ifc_class == "IfcProject" else "L1_object",
                    "category": "space" if entity.is_a("IfcSpatialStructureElement") or ifc_class == "IfcProject" else "object",
                    "object_domain": domain,
                    "asset_role": role,
                    "space_type": _space_type(text) if ifc_class == "IfcSpace" else None,
                    "storey": storey,
                    "properties": properties,
                    "quantities": quantities,
                    "materials": [],
                    "geometry": geometry,
                    "_express_id": int(entity.id()),
                    "_tag": tag,
                }
            )

        endpoint_ids = {entity_id: _node_id(entity) for entity_id, entity in entities.items()}
        edges: list[dict[str, Any]] = []
        edge_no = 0

        def add(source: Any, target: Any, relation: str, rel_class: str) -> None:
            nonlocal edge_no
            if source is None or target is None:
                return
            source_id = endpoint_ids.get(int(source.id()))
            target_id = endpoint_ids.get(int(target.id()))
            if not source_id or not target_id:
                return
            edge_no += 1
            edges.append(
                {
                    "edge_id": f"edge_{edge_no:08d}",
                    "source": source_id,
                    "target": target_id,
                    "relation": relation,
                    "evidence_type": "explicit",
                    "confidence": 1.0,
                    "source_ifc_relation": rel_class,
                    "features": {},
                }
            )

        rel_specs = [
            ("IfcRelContainedInSpatialStructure", "RelatingStructure", "RelatedElements", "contains"),
            ("IfcRelAggregates", "RelatingObject", "RelatedObjects", "contains"),
            ("IfcRelAssignsToGroup", "RelatingGroup", "RelatedObjects", "assigned_to_system"),
            ("IfcRelConnectsPorts", "RelatingPort", "RelatedPort", "connects_to"),
            ("IfcRelConnectsPortToElement", "RelatingElement", "RelatedPort", "has_port"),
            ("IfcRelSpaceBoundary", "RelatingSpace", "RelatedBuildingElement", "bounds"),
            ("IfcRelVoidsElement", "RelatingBuildingElement", "RelatedOpeningElement", "has_opening"),
            ("IfcRelFillsElement", "RelatingOpeningElement", "RelatedBuildingElement", "fills_opening"),
        ]
        for rel_class, source_attr, target_attr, relation in rel_specs:
            try:
                rels = model.by_type(rel_class)
            except RuntimeError:
                continue
            for rel in rels:
                source = getattr(rel, source_attr, None)
                targets = getattr(rel, target_attr, None)
                if not isinstance(targets, (list, tuple)):
                    targets = [targets]
                for target in targets:
                    add(source, target, relation, rel_class)

        return {
            "metadata": {
                "stage": "compact_ifc_fallback",
                "source_ifc": str(source_path),
                "ifc_schema": getattr(model, "schema", None),
                "validation": {},
            },
            "nodes": nodes,
            "edges": edges,
        }


class InspectionGraphBuilder:
    """Use the rich four-level builder when available, otherwise the fallback."""

    def _workspace_builder_path(self) -> Path | None:
        configured = os.environ.get("TOG_INSPECTION_BUILDER")
        candidates = [Path(configured)] if configured else []
        package_root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                package_root / "parser" / "build_inspection_graph.py",
                package_root.parent / "parser" / "build_inspection_graph.py",
                Path.cwd().parent / "parser" / "build_inspection_graph.py",
                Path.cwd() / "parser" / "build_inspection_graph.py",
            ]
        )
        return next((path.resolve() for path in candidates if path and path.exists()), None)

    def build(self, source_path: Path) -> dict[str, Any]:
        builder_path = self._workspace_builder_path()
        if builder_path is None:
            return CompactIfcBuilder().build(source_path)

        spec = importlib.util.spec_from_file_location("tog_workspace_inspection_builder", builder_path)
        if spec is None or spec.loader is None:
            return CompactIfcBuilder().build(source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ifcopenshell.open(str(source_path))
        builder = module.GraphBuilder(
            model,
            source_path,
            min(4, max(1, (os.cpu_count() or 2) - 1)),
            include_all_elements=True,
            geometry_cache_path=None,
            skip_missing_geometry=False,
        )
        return builder.build(
            INFERENCE_CONFIG["candidate_connect_threshold"],
            INFERENCE_CONFIG["max_candidate_connects_per_node"],
        )


class FrozenInspectionGraphBuilder:
    """Load the exact inspection graph used to produce GNN artifacts."""

    def __init__(
        self,
        graph_path: str | Path,
        *,
        edge_profile: str = "query-conditioned-plan-v5.0",
    ) -> None:
        self.graph_path = Path(graph_path).expanduser().resolve()
        if not self.graph_path.is_file():
            raise FileNotFoundError(f"Inspection graph not found: {self.graph_path}")
        self.edge_profile = edge_profile
        self._graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        if edge_profile != "query-conditioned-plan-v5.0":
            raise ValueError(f"Unsupported frozen graph edge profile: {edge_profile}")
        self.graph_hash = _sha256(self.graph_path)
        self.inference_config_hash = (
            f"frozen-v5-{self.graph_hash[:16]}-{ACTIONABILITY_VERSION}"
        )

    def build(self, source_path: Path) -> dict[str, Any]:
        del source_path
        graph = self._graph
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
            raise RuntimeError(f"Invalid frozen inspection graph: {self.graph_path}")
        if not isinstance(graph.get("edges"), list):
            raise RuntimeError(f"Invalid frozen inspection graph edges: {self.graph_path}")
        report = _ensure_actionability_contract(graph)
        if not report["valid"]:
            raise RuntimeError(
                f"Invalid actionability contract in frozen graph: {self.graph_path}"
            )
        return graph


class GraphIndexManager:
    def __init__(
        self,
        cache_dir: str | Path,
        builder: Any | None = None,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.builder = builder or InspectionGraphBuilder()
        self.inference_config_hash = str(
            getattr(self.builder, "inference_config_hash", INFERENCE_CONFIG_HASH)
        )
        self.progress_callback = progress_callback
        self._reports: dict[Path, IndexBuildReport] = {}
        self._source_signatures: dict[Path, tuple[int, int]] = {}

    def set_progress_callback(
        self,
        callback: Callable[[str, int, int, str], None] | None,
    ) -> None:
        self.progress_callback = callback

    def _notify(self, stage: str, current: int = 0, total: int = 0, detail: str = "") -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, current, total, detail)

    def ensure_index(self, ifc_path: str | Path, rebuild: bool = False) -> IndexBuildReport:
        source_path = Path(ifc_path).expanduser().resolve()
        self._notify("start", detail=source_path.name)
        if not source_path.exists():
            raise FileNotFoundError(f"IFC file not found: {source_path}")
        stat = source_path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if (
            source_path in self._reports
            and self._source_signatures.get(source_path) == signature
            and not rebuild
        ):
            self._notify("memory_cache_hit", 1, 1, source_path.name)
            return self._reports[source_path]

        self._notify("hashing", detail=source_path.name)
        source_hash = _sha256(source_path)
        profile_suffix = (
            "" if self.inference_config_hash == INFERENCE_CONFIG_HASH
            else f"-{self.inference_config_hash}"
        )
        index_path = self.cache_dir / (
            f"{source_hash[:20]}-{GRAPH_SCHEMA_VERSION}{profile_suffix}.sqlite"
        )
        if index_path.exists() and not rebuild:
            try:
                report = self._read_report(index_path, source_path, source_hash, cache_hit=True)
            except RuntimeError:
                # Keep the old file readable until the rebuilt temporary database is
                # atomically swapped in below.
                pass
            else:
                self._reports[source_path] = report
                self._source_signatures[source_path] = signature
                self._notify("disk_cache_hit", 1, 1, source_path.name)
                return report

        started = time.perf_counter()
        self._notify("rebuilding_graph", detail=source_path.name)
        graph = self.builder.build(source_path)
        actionability = _ensure_actionability_contract(graph)
        if not actionability["valid"]:
            raise RuntimeError("Inspection graph contains actionable nodes without stable IFC identity")
        self._notify(
            "graph_extracted",
            len(graph.get("nodes", [])),
            len(graph.get("nodes", [])),
            f"{len(graph.get('edges', []))} edges",
        )
        temp_path = index_path.with_suffix(f".tmp-{os.getpid()}.sqlite")
        if temp_path.exists():
            temp_path.unlink()
        try:
            node_count, edge_count, validation = self._write_index(
                temp_path, source_path, source_hash, graph
            )
            total_build_seconds = time.perf_counter() - started
            # sqlite3.Connection's context manager commits/rolls back but does
            # not close the handle.  Keeping that handle alive prevents the
            # atomic os.replace below on Windows.
            connection = sqlite3.connect(temp_path)
            try:
                index_write_row = connection.execute(
                    "SELECT value FROM graph_meta WHERE key='build_seconds'"
                ).fetchone()
                connection.execute(
                    "INSERT OR REPLACE INTO graph_meta(key, value) VALUES ('index_write_seconds', ?)",
                    (index_write_row[0] if index_write_row else "0",),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO graph_meta(key, value) VALUES ('build_seconds', ?)",
                    (str(total_build_seconds),),
                )
                connection.commit()
            finally:
                connection.close()
            os.replace(temp_path, index_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        report = IndexBuildReport(
            index_path=index_path,
            source_path=source_path,
            source_hash=source_hash,
            schema_version=GRAPH_SCHEMA_VERSION,
            cache_hit=False,
            build_seconds=total_build_seconds,
            peak_memory_mb=_peak_memory_mb(),
            node_count=node_count,
            edge_count=edge_count,
            validation=validation,
        )
        self._reports[source_path] = report
        self._source_signatures[source_path] = signature
        self._notify(
            "complete",
            node_count + edge_count,
            node_count + edge_count,
            f"{total_build_seconds:.1f}s",
        )
        return report

    def _read_report(
        self,
        index_path: Path,
        source_path: Path,
        source_hash: str,
        cache_hit: bool,
    ) -> IndexBuildReport:
        # sqlite3.Connection's context manager commits/rolls back but does not
        # close the handle.  Keep the lifecycle explicit so a stale index can
        # be atomically replaced on Windows immediately after this read.
        connection = sqlite3.connect(index_path)
        try:
            meta = dict(connection.execute("SELECT key, value FROM graph_meta"))
            if meta.get("value_alias_version") != VALUE_ALIAS_VERSION:
                _insert_quantity_aliases(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO graph_meta(key, value) VALUES ('value_alias_version', ?)",
                    (VALUE_ALIAS_VERSION,),
                )
                connection.commit()
                meta["value_alias_version"] = VALUE_ALIAS_VERSION
        finally:
            connection.close()
        if meta.get("source_hash") != source_hash:
            raise RuntimeError(f"Stale ToG graph index: {index_path}")
        if meta.get("schema_version") != GRAPH_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported ToG graph schema: {meta.get('schema_version')}")
        if meta.get("inference_config_hash") != self.inference_config_hash:
            raise RuntimeError(
                f"Stale ToG inference configuration in graph index: {index_path}"
            )
        return IndexBuildReport(
            index_path=index_path,
            source_path=source_path,
            source_hash=source_hash,
            schema_version=GRAPH_SCHEMA_VERSION,
            cache_hit=cache_hit,
            build_seconds=float(meta.get("build_seconds", 0.0)),
            peak_memory_mb=float(meta.get("peak_memory_mb", 0.0)),
            node_count=int(meta.get("node_count", 0)),
            edge_count=int(meta.get("edge_count", 0)),
            validation=json.loads(meta.get("validation", "{}")),
        )

    def _write_index(
        self,
        path: Path,
        source_path: Path,
        source_hash: str,
        graph: dict[str, Any],
    ) -> tuple[int, int, dict[str, Any]]:
        started = time.perf_counter()
        nodes = graph.get("nodes", [])
        raw_edges = graph.get("edges", [])
        node_ids = {str(node["node_id"]) for node in nodes}
        edges = [
            edge
            for edge in raw_edges
            if str(edge.get("source")) in node_ids and str(edge.get("target")) in node_ids
        ]
        orphan_count = len(raw_edges) - len(edges)
        validation = dict(graph.get("metadata", {}).get("validation", {}))
        validation["orphan_edges_dropped"] = orphan_count
        validation["node_ids_unique"] = len(node_ids) == len(nodes)
        self._notify("writing_schema", detail=path.name)

        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=MEMORY;
                CREATE TABLE graph_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE nodes (
                    pk INTEGER PRIMARY KEY,
                    node_id TEXT NOT NULL UNIQUE,
                    global_id TEXT,
                    express_id INTEGER,
                    ifc_class TEXT,
                    name TEXT,
                    long_name TEXT,
                    family TEXT,
                    type_name TEXT,
                    object_type TEXT,
                    tag TEXT,
                    storey TEXT,
                    level TEXT,
                    category TEXT,
                    domain TEXT,
                    role TEXT,
                    space_type TEXT,
                    system_category TEXT,
                    function_type TEXT,
                    search_text TEXT NOT NULL,
                    attrs_json TEXT NOT NULL
                );
                CREATE TABLE edges (
                    edge_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    distance REAL,
                    source_ifc_relation TEXT,
                    features_json TEXT NOT NULL
                );
                CREATE TABLE node_values (
                    node_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value_text TEXT NOT NULL,
                    value_num REAL,
                    value_type TEXT NOT NULL
                );
                CREATE TABLE geometry (
                    node_id TEXT PRIMARY KEY,
                    min_x REAL, max_x REAL,
                    min_y REAL, max_y REAL,
                    min_z REAL, max_z REAL,
                    center_x REAL, center_y REAL, center_z REAL
                );
                CREATE INDEX idx_nodes_guid ON nodes(global_id);
                CREATE INDEX idx_nodes_name ON nodes(name);
                CREATE INDEX idx_nodes_long_name ON nodes(long_name);
                CREATE INDEX idx_nodes_tag ON nodes(tag);
                CREATE INDEX idx_nodes_storey ON nodes(storey);
                CREATE INDEX idx_nodes_role ON nodes(role);
                CREATE INDEX idx_nodes_class ON nodes(ifc_class);
                CREATE INDEX idx_nodes_function_type ON nodes(function_type);
                CREATE INDEX idx_edges_source_relation ON edges(source, relation);
                CREATE INDEX idx_edges_target_relation ON edges(target, relation);
                CREATE INDEX idx_values_node_predicate ON node_values(node_id, predicate);
                CREATE INDEX idx_values_predicate ON node_values(predicate);
                """
            )
            fts_enabled = True
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE nodes_fts USING fts5(node_id UNINDEXED, search_text)"
                )
            except sqlite3.OperationalError:
                fts_enabled = False
            rtree_enabled = True
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE geometry_rtree USING rtree(
                        node_pk, min_x, max_x, min_y, max_y, min_z, max_z
                    )
                    """
                )
            except sqlite3.OperationalError:
                rtree_enabled = False

            node_notify_every = max(1, len(nodes) // 100)
            for node_index, node in enumerate(nodes, 1):
                properties = node.get("properties") or {}
                ifc_meta = properties.get("_ifc", {}) if isinstance(properties, dict) else {}
                express_id = node.get("_express_id") or ifc_meta.get("ifc_id")
                tag = node.get("_tag") or ifc_meta.get("tag")
                object_type = node.get("object_type") or ifc_meta.get("object_type")
                fields = [
                    node.get("node_id"), node.get("ifc_guid"), node.get("ifc_class"),
                    node.get("name"), node.get("long_name"), node.get("family"),
                    node.get("type"), object_type, tag, node.get("storey"),
                    node.get("object_domain"), node.get("asset_role"),
                    node.get("space_type"), node.get("system_category"),
                    node.get("function_type"),
                    node.get("action_target_kind"),
                ]
                searchable_values: list[str] = []
                for predicate, value in list(_scalar_values(properties, "property"))[:200]:
                    searchable_values.extend((predicate, str(value)))
                for predicate, value in list(
                    _scalar_values(node.get("quantities") or {}, "quantity")
                )[:100]:
                    searchable_values.extend((predicate, str(value)))
                searchable_values.extend(
                    str(value) for value in (node.get("materials") or [])[:50]
                )
                fields.extend(searchable_values)
                search_text = " ".join(
                    str(value) for value in fields if value not in (None, "")
                )
                cursor = connection.execute(
                    """
                    INSERT INTO nodes(
                        node_id, global_id, express_id, ifc_class, name, long_name,
                        family, type_name, object_type, tag, storey, level, category,
                        domain, role, space_type, system_category, function_type,
                        search_text, attrs_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(node["node_id"]), node.get("ifc_guid"), express_id,
                        node.get("ifc_class"), node.get("name"), node.get("long_name"),
                        node.get("family"), node.get("type"), object_type, tag,
                        node.get("storey"), node.get("level"), node.get("category"),
                        node.get("object_domain"), node.get("asset_role"),
                        node.get("space_type"), node.get("system_category"),
                        node.get("function_type"), search_text, _json(node),
                    ),
                )
                node_pk = int(cursor.lastrowid)
                if fts_enabled:
                    connection.execute(
                        "INSERT INTO nodes_fts(node_id, search_text) VALUES (?,?)",
                        (str(node["node_id"]), search_text),
                    )

                direct_values = {
                    "ifc_guid": node.get("ifc_guid"),
                    "ifc_class": node.get("ifc_class"),
                    "name": node.get("name"),
                    "long_name": node.get("long_name"),
                    "family": node.get("family"),
                    "type": node.get("type"),
                    "object_type": object_type,
                    "tag": tag,
                    "storey": node.get("storey"),
                    "domain": node.get("object_domain"),
                    "role": node.get("asset_role"),
                    "space_type": node.get("space_type"),
                    "system_category": node.get("system_category"),
                    "function_type": node.get("function_type"),
                    "action_target_kind": node.get("action_target_kind"),
                }
                values: list[tuple[str, Any]] = [
                    (key, value) for key, value in direct_values.items() if value not in (None, "")
                ]
                values.extend(_scalar_values(properties, "property"))
                values.extend(_scalar_values(node.get("quantities") or {}, "quantity"))
                values.extend(_scalar_values(node.get("materials") or [], "material"))
                seen_values: set[tuple[str, str]] = set()
                for predicate, value in values:
                    value_text = str(value)
                    key = (predicate.lower(), value_text)
                    if key in seen_values:
                        continue
                    seen_values.add(key)
                    connection.execute(
                        "INSERT INTO node_values VALUES (?,?,?,?,?)",
                        (str(node["node_id"]), predicate, value_text, _number(value), type(value).__name__),
                    )

                geometry = node.get("geometry") or {}
                bbox_min = geometry.get("bbox_min")
                bbox_max = geometry.get("bbox_max")
                center = geometry.get("centroid")
                if center and len(center) >= 3 and all(value is not None for value in center[:3]):
                    if not bbox_min or not bbox_max:
                        bbox_min = bbox_max = center
                    if any(value is None for value in [*bbox_min[:3], *bbox_max[:3]]):
                        bbox_min = bbox_max = center
                    connection.execute(
                        "INSERT INTO geometry VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            str(node["node_id"]), bbox_min[0], bbox_max[0],
                            bbox_min[1], bbox_max[1], bbox_min[2], bbox_max[2],
                            center[0], center[1], center[2],
                        ),
                    )
                    if rtree_enabled:
                        connection.execute(
                            "INSERT INTO geometry_rtree VALUES (?,?,?,?,?,?,?)",
                            (
                                node_pk, bbox_min[0], bbox_max[0], bbox_min[1],
                                bbox_max[1], bbox_min[2], bbox_max[2],
                            ),
                        )
                if node_index == len(nodes) or node_index % node_notify_every == 0:
                    self._notify("writing_nodes", node_index, len(nodes), path.name)

            relation_counts: Counter[str] = Counter()
            evidence_counts: Counter[str] = Counter()
            edge_notify_every = max(1, len(edges) // 100)
            for index, edge in enumerate(edges, 1):
                features = dict(edge.get("features") or {})
                distance = features.get("distance")
                edge_id = str(edge.get("edge_id") or f"edge_{index:08d}")
                relation = str(edge.get("relation") or "related_to")
                evidence_type = str(edge.get("evidence_type") or "explicit")
                relation_counts[relation] += 1
                evidence_counts[evidence_type] += 1
                connection.execute(
                    "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        edge_id, str(edge["source"]), str(edge["target"]), relation,
                        evidence_type, float(edge.get("confidence", 1.0)),
                        _number(distance), edge.get("source_ifc_relation"), _json(features),
                    ),
                )
                if index == len(edges) or index % edge_notify_every == 0:
                    self._notify("writing_edges", index, len(edges), path.name)

            self._notify("finalizing_index", detail=path.name)
            _insert_quantity_aliases(connection)
            build_seconds = time.perf_counter() - started
            meta = {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "inference_config_hash": self.inference_config_hash,
                "value_alias_version": VALUE_ALIAS_VERSION,
                "source_path": str(source_path),
                "source_hash": source_hash,
                "node_count": str(len(nodes)),
                "edge_count": str(len(edges)),
                "build_seconds": str(build_seconds),
                "peak_memory_mb": str(_peak_memory_mb()),
                "validation": _json(validation),
                "relation_counts": _json(relation_counts),
                "evidence_counts": _json(evidence_counts),
                "source_metadata": _json(graph.get("metadata", {})),
                "model_text_search_policy": "runtime-graph-search-text-v5",
                "fts_enabled": str(fts_enabled).lower(),
                "rtree_enabled": str(rtree_enabled).lower(),
            }
            representation_graph_hash = getattr(self.builder, "graph_hash", None)
            if representation_graph_hash:
                meta["representation_graph_hash"] = str(representation_graph_hash)
            connection.executemany(
                "INSERT INTO graph_meta(key, value) VALUES (?,?)", meta.items()
            )
            connection.commit()
        finally:
            connection.close()
        return len(nodes), len(edges), validation
