from __future__ import annotations

import os
import stat
from pathlib import Path


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        if path.is_dir():
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        path.chmod(mode)
    root.chmod(
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )


def prepare_asset_mount() -> Path:
    configured = Path(os.environ.get("INSPECTION_DEMO_ASSET_MOUNT", "/mnt/default-assets"))
    if (configured / "manifest.json").is_file():
        return configured
    repository = os.environ.get("HF_ASSET_DATASET", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not repository or not token:
        raise RuntimeError(
            "The private Ecore asset Dataset must be mounted or HF_ASSET_DATASET and HF_TOKEN set"
        )
    from huggingface_hub import snapshot_download

    destination = Path("/tmp/inspection-demo/default-assets")
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        token=token,
        local_dir=destination,
    )
    _make_read_only(destination)
    os.environ["INSPECTION_DEMO_ASSET_MOUNT"] = str(destination)
    os.environ.pop("HF_TOKEN", None)
    return destination


def main() -> None:
    prepare_asset_mount()
    os.environ.pop("HF_TOKEN", None)
    from uvicorn import run

    run(
        "inspection_demo.main:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        access_log=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
