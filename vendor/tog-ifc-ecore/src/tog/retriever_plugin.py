"""Generic, model-independent subgraph retriever plugin contract.

ToG owns bounded reasoning request/result types. Retrieval models live in
separate distributions and register factories through the
``tog_ifc.retrievers`` entry-point group. Importing :mod:`tog` never imports an
external model plugin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

from .backend import IfcGraphBackend
from .models import (
    EntityRef,
    GnnArtifactReport,
    GnnSubgraph,
    IndexBuildReport,
    QueryPlan,
)

ENTRY_POINT_GROUP = "tog_ifc.retrievers"


@dataclass(frozen=True, slots=True)
class RetrievalPluginContext:
    """Hash-bound construction context supplied by the application."""

    artifact_dir: str
    expected_artifact_manifest_sha256: str
    expected_runtime_graph_sha256: str
    settings: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeContext:
    """Generic graph/index context validated before retrieval starts."""

    index_report: IndexBuildReport
    graph_backend: IfcGraphBackend


@dataclass(frozen=True, slots=True)
class RetrievalProviderReport:
    """Provider identity plus ToG's existing artifact validation report."""

    provider_id: str
    provider_version: str
    artifact_report: GnnArtifactReport


@dataclass(frozen=True, slots=True)
class SubgraphRetrievalRequest:
    """Model-independent bounded retrieval request."""

    question: str
    query_plan: QueryPlan
    seed_entities: Sequence[EntityRef]
    graph_backend: IfcGraphBackend
    expand: bool = True


@runtime_checkable
class SubgraphRetrieverPlugin(Protocol):
    """Structural interface consumed by :class:`tog.engine.ToGSystem`."""

    provider_id: str
    provider_version: str
    retrieval_profile: str

    def validate_runtime(
        self,
        context: RetrievalRuntimeContext,
    ) -> RetrievalProviderReport: ...

    def retrieve_subgraph(self, request: SubgraphRetrievalRequest) -> GnnSubgraph: ...


def as_retriever_plugin(value: Any) -> SubgraphRetrieverPlugin:
    """Validate and return a current entry-point retriever plugin."""

    if isinstance(value, SubgraphRetrieverPlugin):
        return value
    raise TypeError("retriever violates the generic ToG plugin contract")


def validate_retriever_runtime(
    value: Any,
    context: RetrievalRuntimeContext,
) -> RetrievalProviderReport:
    """Validate one current plugin against the active graph/index."""

    return as_retriever_plugin(value).validate_runtime(context)


def retrieve_subgraph(
    value: Any,
    request: SubgraphRetrievalRequest,
) -> GnnSubgraph:
    """Dispatch one bounded request through the current plugin contract."""

    return as_retriever_plugin(value).retrieve_subgraph(request)


RetrieverPluginFactory = Callable[[RetrievalPluginContext], SubgraphRetrieverPlugin]


def available_retriever_plugins() -> tuple[str, ...]:
    """Return registered names without importing their implementations."""

    return tuple(
        sorted(item.name for item in metadata.entry_points(group=ENTRY_POINT_GROUP))
    )


def load_retriever_plugin(
    name: str,
    context: RetrievalPluginContext,
) -> SubgraphRetrieverPlugin:
    """Load one explicitly named plugin and fail closed on ambiguity or drift."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("retriever plugin name must be non-empty")
    matches = [
        item
        for item in metadata.entry_points(group=ENTRY_POINT_GROUP)
        if item.name == normalized
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"retriever plugin {normalized!r} must resolve exactly once; "
            f"found {len(matches)}"
        )
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError(f"retriever plugin {normalized!r} is not callable")
    plugin = factory(context)
    if not isinstance(plugin, SubgraphRetrieverPlugin):
        raise TypeError(f"retriever plugin {normalized!r} violates the ToG contract")
    if plugin.provider_id != normalized:
        raise RuntimeError(
            f"retriever plugin provider_id mismatch: {plugin.provider_id!r} != "
            f"{normalized!r}"
        )
    return plugin
