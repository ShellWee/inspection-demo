"""Query-only routing for inspection requests.

The formal v3 route is intentionally independent of evaluator categories,
question identifiers, graph contents, and expected answers.  It distinguishes
direct IFC queries/aggregations from executable inspection instructions using
only the exact question text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass

ROUTER_SCHEMA_VERSION = "ifc-retrieval-route-v2"
ROUTER_VERSION = "query-intent-actions-v2"
DETERMINISTIC_ROUTE = "deterministic"
RETRIEVAL_ROUTE = "retrieval"


_LEADING_DECORATION = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:first|next|finally)\s*[,;:]?\s*)+",
    re.IGNORECASE,
)
_INTERROGATIVE = re.compile(
    r"^(?:what|which|who|where|when|why|how|how\s+many|how\s+much|"
    r"list|count|name|show|tell|report|identify|find|is|are|do|does|did|can|could|would)\b",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(?:navigate(?:\s+to)?|go\s+to|move\s+to|travel\s+to|head\s+to|"
    r"inspect|examine|check|scan|survey|test|measure|"
    r"be\s+(?:by|near|inside|at)|stand\s+(?:by|near|inside|at)|"
    r"position\s+(?:by|near|inside|at))\b",
    re.IGNORECASE,
)
_PERFORM_ACTION = re.compile(
    r"\bperform\b.{0,48}\b(?:inspection|scan|test|measurement|survey|check)\b",
    re.IGNORECASE,
)
_CLAUSE_ACTION = re.compile(
    r"(?:^|[.;:]|\bthen\b|\band\s+then\b)\s*"
    r"(?:please\s+)?(?:navigate(?:\s+to)?|go\s+to|move\s+to|inspect|"
    r"examine|check|scan|survey|test|measure|be\s+(?:by|near|inside|at)|"
    r"stand\s+(?:by|near|inside|at)|position\s+(?:by|near|inside|at))\b",
    re.IGNORECASE,
)
_POLITE_ACTION_REQUEST = re.compile(
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:navigate(?:\s+to)?|go\s+to|move\s+to|inspect|examine|check|scan|"
    r"survey|test|measure|be\s+(?:by|near|inside|at)|"
    r"stand\s+(?:by|near|inside|at)|position\s+(?:by|near|inside|at))\b",
    re.IGNORECASE,
)
_DECLARATIVE_ACTION_REQUEST = re.compile(
    r"\b(?:need|want|ask|require)\s+you\s+to\s+"
    r"(?:navigate(?:\s+to)?|go\s+to|move\s+to|inspect|examine|check|scan|"
    r"survey|test|measure|be\s+(?:by|near|inside|at)|"
    r"stand\s+(?:by|near|inside|at)|position\s+(?:by|near|inside|at))\b",
    re.IGNORECASE,
)
_DIRECT_LOOKUP_IMPERATIVE = re.compile(
    r"^(?:check|find|identify)\s+(?:how\s+many|how\s+much|which|what|whether|if)\b",
    re.IGNORECASE,
)
_DIAGNOSTIC_ACTION = re.compile(
    r"\b(?:investigate|diagnose|troubleshoot|infer)\b"
    r"(?=.{0,320}\b(?:abnormal(?:ly)?|root\s+cause|fault|failure|"
    r"insufficient|inspection\s+targets?|components?)\b)",
    re.IGNORECASE,
)
_SCENARIO_PREFIX = re.compile(r"^(?:what\s+(?:happens\s+)?if|if)\b", re.IGNORECASE)
_SCENARIO_STATE_CHANGE = re.compile(
    r"\b(?:shut(?:s|ting)?\s+down|turned?\s+off|fails?|malfunctions?)\b",
    re.IGNORECASE,
)
_SCENARIO_IMPACT = re.compile(r"\b(?:impact(?:ed)?|affect(?:ed)?)\b", re.IGNORECASE)


def normalize_query_text(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(question)).split())


@dataclass(frozen=True, slots=True)
class RetrievalRouteDecision:
    schema_version: str
    router_version: str
    normalized_query_sha256: str
    route: str
    requires_retrieval: bool
    recognized_action_spans: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_query_intent(question: str) -> RetrievalRouteDecision:
    """Classify a question without accepting benchmark metadata.

    An interrogative/reporting request remains deterministic even when it
    mentions an inspection verb (for example, "which assets should be
    inspected?").  Executable imperative clauses enter retrieval.
    """

    normalized = normalize_query_text(question)
    if not normalized:
        raise ValueError("Question text is required for retrieval routing")
    stripped = _LEADING_DECORATION.sub("", normalized)
    interrogative = bool(_INTERROGATIVE.match(stripped))
    matches = list(_ACTION.finditer(stripped))
    perform_match = _PERFORM_ACTION.search(stripped)
    clause_matches = list(_CLAUSE_ACTION.finditer(stripped))
    polite_request = _POLITE_ACTION_REQUEST.search(stripped)
    declarative_request = _DECLARATIVE_ACTION_REQUEST.search(stripped)
    leading_action = bool(matches and matches[0].start() <= 8)
    direct_lookup = bool(_DIRECT_LOOKUP_IMPERATIVE.search(stripped))
    diagnostic_action = _DIAGNOSTIC_ACTION.search(stripped)
    scenario_prefix = _SCENARIO_PREFIX.search(stripped)
    scenario_state_change = _SCENARIO_STATE_CHANGE.search(stripped)
    scenario_impact = _SCENARIO_IMPACT.search(stripped)
    impact_scenario = bool(
        scenario_prefix and scenario_state_change and scenario_impact
    )
    executable = bool(
        polite_request
        or declarative_request
        or diagnostic_action
        or impact_scenario
        or (
            not interrogative
            and not direct_lookup
            and (leading_action or perform_match or clause_matches)
        )
        or (clause_matches and any(match.start() > 0 for match in clause_matches))
    )
    semantic_spans = [
        match.group(0).strip(" ,.;:")
        for match in (diagnostic_action, scenario_state_change)
        if match is not None
    ]
    spans = tuple(
        dict.fromkeys(
            [match.group(0).strip(" ,.;:") for match in matches]
            + ([perform_match.group(0).strip()] if perform_match else [])
            + semantic_spans
        )
    )
    return RetrievalRouteDecision(
        schema_version=ROUTER_SCHEMA_VERSION,
        router_version=ROUTER_VERSION,
        normalized_query_sha256=hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest(),
        route=RETRIEVAL_ROUTE if executable else DETERMINISTIC_ROUTE,
        requires_retrieval=executable,
        recognized_action_spans=spans,
        reason=(
            "query contains an executable inspection/navigation or scenario action"
            if executable
            else "query is a direct IFC lookup, report, or aggregation"
        ),
    )


__all__ = [
    "DETERMINISTIC_ROUTE",
    "RETRIEVAL_ROUTE",
    "ROUTER_SCHEMA_VERSION",
    "ROUTER_VERSION",
    "RetrievalRouteDecision",
    "classify_query_intent",
    "normalize_query_text",
]
