from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from inspection_demo.asset_bootstrap import AssetBootstrapError, AssetBootstrapper


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runtime(path: Path, *, ifc_hash: str, traversal: bool = False) -> str:
    files = {
        "checkpoint.bin": b"weights",
        "inspection-graph.json": b"{}",
        "node-facts.jsonl.gz": b"facts",
        "node-ids.json.gz": b"ids",
        "runtime-static-arrays.npz": b"arrays",
        "slot-embeddings.npz": b"embeddings",
        "source/text_gnn_v5/__init__.py": b"",
    }
    runtime_manifest = {
        "schema_version": "text-gnn-v5-generic-runtime-artifact-v1",
        "source_ifc_sha256": ifc_hash,
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    }
    manifest_bytes = json.dumps(runtime_manifest, sort_keys=True).encode()
    files["manifest.json"] = manifest_bytes
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"text-gnn-v5.0-runtime/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if traversal:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    return hashlib.sha256(manifest_bytes).hexdigest()


def _asset_tree(tmp_path: Path, *, traversal: bool = False) -> tuple[Path, Path]:
    mount = tmp_path / "mount"
    (mount / "ecore").mkdir(parents=True)
    (mount / "runtime").mkdir()
    (mount / "index").mkdir()
    (mount / "floorplans").mkdir()
    (mount / "ecore" / "ecore.ifc").write_bytes(b"ISO-10303-21;\nEND-ISO-10303-21;")
    ifc_hash = _sha256(mount / "ecore" / "ecore.ifc")
    runtime_manifest_sha256 = _write_runtime(
        mount / "runtime" / "runtime.tar.gz", ifc_hash=ifc_hash, traversal=traversal
    )
    (mount / "index" / "graph.sqlite").write_bytes(b"sqlite")
    floor = {
        "id": "floor-1",
        "floor_id": "LEVEL 1",
        "units": "m",
        "origin": [0, 0],
        "resolution_m": 0.1,
        "width_m": 1,
        "height_m": 1,
        "polygons": [],
        "occupancy_rows": [],
        "target_positions": {},
        "transform": {
            "local_to_ifc_x": 0,
            "local_to_ifc_y": 0,
            "local_to_ifc_scale": 1,
            "ifc_to_local_scale": 1,
            "ifc_length_unit_to_m": 1,
        },
        "subgraph_projection": [],
        "source_ifc_sha256": ifc_hash,
        "artifact_version": "floorplan-v1",
    }
    (mount / "floorplans" / "floor-1.json").write_text(json.dumps(floor), encoding="utf-8")
    entries = []
    for relative in (
        "ecore/ecore.ifc",
        "runtime/runtime.tar.gz",
        "index/graph.sqlite",
        "floorplans/floor-1.json",
    ):
        path = mount / relative
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "schema_version": "ecore-inspection-assets-v1",
        "ifc_sha256": ifc_hash,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "entries": entries,
    }
    floor_entries = [item for item in entries if item["path"].startswith("floorplans/")]
    manifest["floor_inventory_sha256"] = hashlib.sha256(
        json.dumps(floor_entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = mount / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    expected = tmp_path / "expected.json"
    expected.write_bytes(manifest_path.read_bytes())
    return mount, expected


def test_bootstrap_verifies_and_prepares_fixed_assets(tmp_path: Path) -> None:
    mount, expected = _asset_tree(tmp_path)
    prepared = AssetBootstrapper(
        asset_mount=mount,
        expected_manifest_path=expected,
        workspace=tmp_path / "runtime-work",
        data_dir=tmp_path / "data",
        require_cuda=False,
        verify_plugin=False,
    ).prepare()

    assert prepared.ifc_path == mount / "ecore" / "ecore.ifc"
    assert prepared.runtime_dir.name == "text-gnn-v5.0-runtime"
    assert (prepared.cache_dir / "graph.sqlite").is_file()
    assert [path.name for path in prepared.floorplan_dir.glob("*.json")] == ["floor-1.json"]


def test_bootstrap_rejects_archive_traversal(tmp_path: Path) -> None:
    mount, expected = _asset_tree(tmp_path, traversal=True)
    with pytest.raises(AssetBootstrapError, match="unsafe runtime archive"):
        AssetBootstrapper(
            asset_mount=mount,
            expected_manifest_path=expected,
            workspace=tmp_path / "runtime-work",
            data_dir=tmp_path / "data",
            require_cuda=False,
            verify_plugin=False,
        ).prepare()
