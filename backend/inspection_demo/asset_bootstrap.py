from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any


class AssetBootstrapError(RuntimeError):
    """Raised when the immutable Ecore asset set cannot be trusted or prepared."""


@dataclass(frozen=True, slots=True)
class PreparedAssets:
    ifc_path: Path
    runtime_dir: Path
    cache_dir: Path
    floorplan_dir: Path
    asset_manifest_sha256: str
    ifc_sha256: str
    runtime_manifest_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise AssetBootstrapError(f"unsafe asset path: {value!r}")
    return Path(*posix.parts)


class AssetBootstrapper:
    """Verify and materialize one hash-bound, read-only Ecore asset bundle."""

    def __init__(
        self,
        *,
        asset_mount: Path,
        expected_manifest_path: Path,
        workspace: Path,
        data_dir: Path,
        require_cuda: bool = True,
        verify_plugin: bool = True,
    ) -> None:
        self.asset_mount = asset_mount.resolve()
        self.expected_manifest_path = expected_manifest_path.resolve()
        self.workspace = workspace.resolve()
        self.data_dir = data_dir.resolve()
        self.require_cuda = require_cuda
        self.verify_plugin = verify_plugin

    def prepare(self) -> PreparedAssets:
        expected_bytes = self._read(self.expected_manifest_path, "expected asset manifest")
        mounted_manifest = self.asset_mount / "manifest.json"
        observed_bytes = self._read(mounted_manifest, "mounted asset manifest")
        if observed_bytes != expected_bytes:
            raise AssetBootstrapError("mounted asset manifest does not match the pinned manifest")
        manifest_sha = hashlib.sha256(observed_bytes).hexdigest()
        try:
            manifest = json.loads(observed_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AssetBootstrapError("asset manifest is not valid JSON") from error
        if manifest.get("schema_version") != "ecore-inspection-assets-v1":
            raise AssetBootstrapError("unsupported asset manifest schema")
        entries = self._verify_entries(manifest)

        ifc_path = self._single(entries, prefix="ecore/", suffix=".ifc")
        runtime_archive = self._single(entries, prefix="runtime/", suffix=".tar.gz")
        index_path = self._single(entries, prefix="index/", suffix=".sqlite")
        floor_paths = sorted(
            path
            for relative, path in entries.items()
            if relative.startswith("floorplans/") and path.suffix == ".json"
        )
        if not floor_paths:
            raise AssetBootstrapError("asset manifest contains no floor plans")
        if _sha256(ifc_path) != str(manifest.get("ifc_sha256") or ""):
            raise AssetBootstrapError("pinned IFC hash does not match the IFC asset")
        floor_entries = sorted(
            (item for item in manifest["entries"] if str(item["path"]).startswith("floorplans/")),
            key=lambda item: str(item["path"]),
        )
        floor_inventory = json.dumps(
            floor_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if hashlib.sha256(floor_inventory).hexdigest() != str(
            manifest.get("floor_inventory_sha256") or ""
        ):
            raise AssetBootstrapError("floor-plan inventory hash mismatch")

        runtime_dir = self._extract_runtime(runtime_archive, manifest_sha)
        runtime_manifest_path = runtime_dir / "manifest.json"
        runtime_manifest_sha = _sha256(runtime_manifest_path)
        expected_runtime_manifest_sha = str(manifest.get("runtime_manifest_sha256") or "")
        if runtime_manifest_sha != expected_runtime_manifest_sha:
            raise AssetBootstrapError("runtime manifest hash mismatch")
        self._verify_runtime(runtime_dir, str(manifest.get("ifc_sha256") or ""))

        cache_dir = self.data_dir / "research-cache"
        floor_dir = self.data_dir / "assets" / "floorplans"
        cache_dir.mkdir(parents=True, exist_ok=True)
        floor_dir.mkdir(parents=True, exist_ok=True)
        self._copy_verified(index_path, cache_dir / index_path.name)
        for floor in floor_paths:
            self._copy_verified(floor, floor_dir / floor.name)

        if self.require_cuda:
            self._verify_cuda()
        if self.verify_plugin:
            self._verify_plugin(runtime_dir)
        return PreparedAssets(
            ifc_path=ifc_path,
            runtime_dir=runtime_dir,
            cache_dir=cache_dir,
            floorplan_dir=floor_dir,
            asset_manifest_sha256=manifest_sha,
            ifc_sha256=str(manifest["ifc_sha256"]),
            runtime_manifest_sha256=runtime_manifest_sha,
        )

    @staticmethod
    def _read(path: Path, label: str) -> bytes:
        try:
            return path.read_bytes()
        except OSError as error:
            raise AssetBootstrapError(f"{label} is unavailable") from error

    def _verify_entries(self, manifest: dict[str, Any]) -> dict[str, Path]:
        raw_entries = manifest.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise AssetBootstrapError("asset manifest entries are missing")
        verified: dict[str, Path] = {}
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise AssetBootstrapError("asset manifest entry is invalid")
            relative_text = str(raw.get("path") or "")
            relative = _safe_relative(relative_text)
            if relative_text in verified:
                raise AssetBootstrapError(f"duplicate asset path: {relative_text}")
            path = (self.asset_mount / relative).resolve()
            try:
                path.relative_to(self.asset_mount)
            except ValueError as error:
                raise AssetBootstrapError(f"asset escapes mount: {relative_text}") from error
            if not path.is_file():
                raise AssetBootstrapError(f"asset is missing: {relative_text}")
            if path.stat().st_size != int(raw.get("bytes", -1)):
                raise AssetBootstrapError(f"asset size mismatch: {relative_text}")
            if _sha256(path) != str(raw.get("sha256") or "").lower():
                raise AssetBootstrapError(f"asset hash mismatch: {relative_text}")
            verified[relative_text] = path
        return verified

    @staticmethod
    def _single(entries: dict[str, Path], *, prefix: str, suffix: str) -> Path:
        matches = [
            path
            for relative, path in entries.items()
            if relative.startswith(prefix) and relative.endswith(suffix)
        ]
        if len(matches) != 1:
            raise AssetBootstrapError(f"expected exactly one {prefix}*{suffix} asset")
        return matches[0]

    def _extract_runtime(self, archive_path: Path, manifest_sha: str) -> Path:
        self.workspace.mkdir(parents=True, exist_ok=True)
        destination = self.workspace / f"runtime-{manifest_sha[:16]}"
        if destination.is_dir():
            roots = [item for item in destination.iterdir() if item.is_dir()]
            if len(roots) == 1 and (roots[0] / "manifest.json").is_file():
                return roots[0]
        staging = Path(tempfile.mkdtemp(prefix="runtime-staging-", dir=self.workspace))
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                members = archive.getmembers()
                if not members:
                    raise AssetBootstrapError("runtime archive is empty")
                for member in members:
                    try:
                        relative = _safe_relative(member.name)
                    except AssetBootstrapError as error:
                        raise AssetBootstrapError(
                            f"unsafe runtime archive member: {member.name!r}"
                        ) from error
                    target = (staging / relative).resolve()
                    try:
                        target.relative_to(staging)
                    except ValueError as error:
                        raise AssetBootstrapError("unsafe runtime archive path") from error
                    if member.issym() or member.islnk() or member.isdev():
                        raise AssetBootstrapError("unsafe runtime archive member")
                    if not (member.isfile() or member.isdir()):
                        raise AssetBootstrapError("unsafe runtime archive member type")
                archive.extractall(staging, members=members, filter="data")
            roots = [item for item in staging.iterdir() if item.is_dir()]
            if len(roots) != 1 or not (roots[0] / "manifest.json").is_file():
                raise AssetBootstrapError("runtime archive layout is invalid")
            try:
                os.replace(staging, destination)
            except FileExistsError:
                shutil.rmtree(staging)
            roots = [item for item in destination.iterdir() if item.is_dir()]
            if len(roots) != 1:
                raise AssetBootstrapError("prepared runtime layout is invalid")
            return roots[0]
        except (tarfile.TarError, OSError) as error:
            raise AssetBootstrapError("unsafe runtime archive") from error
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _verify_runtime(runtime_dir: Path, expected_ifc_sha256: str) -> None:
        try:
            manifest = json.loads((runtime_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AssetBootstrapError("runtime manifest is invalid") from error
        if manifest.get("schema_version") != "text-gnn-v5-generic-runtime-artifact-v1":
            raise AssetBootstrapError("runtime schema mismatch")
        if str(manifest.get("source_ifc_sha256") or "") != expected_ifc_sha256:
            raise AssetBootstrapError("runtime and IFC hashes do not match")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise AssetBootstrapError("runtime file manifest is missing")
        for relative_text, expected_hash in files.items():
            relative = _safe_relative(str(relative_text))
            path = (runtime_dir / relative).resolve()
            try:
                path.relative_to(runtime_dir)
            except ValueError as error:
                raise AssetBootstrapError("runtime file escapes extraction root") from error
            if not path.is_file() or _sha256(path) != str(expected_hash).lower():
                raise AssetBootstrapError(f"runtime file hash mismatch: {relative_text}")

    @staticmethod
    def _copy_verified(source: Path, destination: Path) -> None:
        if (
            destination.is_file()
            and destination.stat().st_size == source.stat().st_size
            and _sha256(destination) == _sha256(source)
        ):
            return
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, temporary)
        if _sha256(temporary) != _sha256(source):
            temporary.unlink(missing_ok=True)
            raise AssetBootstrapError(f"copy verification failed: {source.name}")
        os.replace(temporary, destination)

    @staticmethod
    def _verify_cuda() -> None:
        try:
            import torch
        except ImportError as error:
            raise AssetBootstrapError("PyTorch is not installed") from error
        if not torch.cuda.is_available():
            raise AssetBootstrapError("CUDA is required but unavailable")
        if not str(torch.version.cuda or "").startswith("12.8"):
            raise AssetBootstrapError("Torch must use the pinned CUDA 12.8 runtime")

    @staticmethod
    def _verify_plugin(runtime_dir: Path) -> None:
        source_root = str((runtime_dir / "source").resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        matches = [
            item
            for item in metadata.entry_points(group="tog_ifc.retrievers")
            if item.name == "text-gnn-v5.0"
        ]
        if len(matches) != 1:
            raise AssetBootstrapError("Text-GNN v5 plugin registration is missing or ambiguous")
        try:
            factory = matches[0].load()
        except Exception as error:  # noqa: BLE001 - converted to a readiness failure
            raise AssetBootstrapError("Text-GNN v5 plugin cannot be imported") from error
        if not callable(factory):
            raise AssetBootstrapError("Text-GNN v5 plugin factory is invalid")
