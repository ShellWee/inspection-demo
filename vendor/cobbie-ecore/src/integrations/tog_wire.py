"""Side-effect-free wire normalization shared by ToG adapters and tests."""

from __future__ import annotations

from typing import Any

_ENUM_WIRE_VALUES = {
    "Lookup": "lookup",
    "List": "list",
    "Distinct": "distinct",
    "Count": "count",
    "GroupCount": "group_count",
    "Argmax": "argmax",
    "Nearest": "nearest",
    "AllMatching": "all_matching",
    "Unconnected": "unconnected",
    "Path": "path",
    "OneGraphNode": "one_graph_node",
    "BestEquivalenceClass": "best_equivalence_class",
    "BoundedOperator": "bounded_operator",
}


def enum_value(value: Any) -> str:
    """Return the stable snake-case value used by the ToG wire contract."""

    raw = str(getattr(value, "value", value))
    return _ENUM_WIRE_VALUES.get(raw, raw.lower())


__all__ = ["enum_value"]
