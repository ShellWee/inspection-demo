from __future__ import annotations

from pathlib import Path

import pytest
from baml_py import logging as baml_logging
from inspection_demo import openai_catalog
from inspection_demo.models import GroundingRequest
from inspection_demo.research_bridge import _private_baml_logs, _validated_navigation_xy
from inspection_demo.settings import Settings
from pydantic import ValidationError


def test_grounding_request_uses_server_bound_assets() -> None:
    request = GroundingRequest(
        api_key="sk-test",
        model_id="gpt-4.1",
        query="Inspect the air terminal in Room 319.",
    )

    assert not hasattr(request, "ifc_asset_id")
    assert not hasattr(request, "gnn_runtime_id")
    with pytest.raises(ValidationError):
        GroundingRequest(
            api_key="sk-test",
            model_id="gpt-4.1",
            query="Inspect the air terminal in Room 319.",
            ifc_asset_id="other-ifc",
        )


def test_model_registry_is_the_only_model_source() -> None:
    assert tuple(openai_catalog.MODEL_PROFILES) == (
        "gpt-4.1",
        "gpt-5",
        "gpt-5.6-luna",
    )
    assert all(
        profile.embedding_model == "text-embedding-3-small"
        for profile in openai_catalog.MODEL_PROFILES.values()
    )


def test_room_404_has_no_navigation_shortcut() -> None:
    assert (
        _validated_navigation_xy(
            "ifc_0aZxGK_jD1LR$XoST2vRo$",
            {},
            floor_map=None,
            expected_ifc_sha256=None,
        )
        is None
    )


def test_settings_are_live_only_and_cuda_by_default(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        asset_mount=tmp_path / "assets",
        runtime_workspace=tmp_path / "runtime",
    )

    assert settings.research_device == "cuda"
    assert not hasattr(settings, "demo_mode")
    assert not hasattr(settings, "research_enabled")
    assert not hasattr(settings, "require_iap")


def test_production_package_contains_no_acceptance_fixture() -> None:
    package = Path(__file__).resolve().parents[1] / "inspection_demo"
    forbidden = (
        "DemoGroundingAdapter",
        "demo-key",
        "ROOM_404_NODE_ID",
        "Acceptance target",
        "sample_data",
    )
    hits: list[str] = []
    for source in package.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        hits.extend(f"{source.name}:{token}" for token in forbidden if token in text)
    assert hits == []


def test_baml_prompt_logging_is_disabled_only_inside_research_scope() -> None:
    original_level = baml_logging.get_log_level()

    with _private_baml_logs():
        assert baml_logging.get_log_level() == "OFF"

    assert baml_logging.get_log_level() == original_level
