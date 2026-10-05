from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT.parent / "inspection-demo-hf-assets"
LOCAL_STATE = PROJECT / ".webapp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=LOCAL_STATE)
    args = parser.parse_args()
    local_state = args.state_dir.resolve()
    os.environ["INSPECTION_DEMO_ASSET_MOUNT"] = str(ASSETS)
    os.environ["INSPECTION_DEMO_RESEARCH_DEVICE"] = "cpu"
    os.environ["INSPECTION_DEMO_DATA_DIR"] = str(local_state / "data")
    os.environ["INSPECTION_DEMO_RUNTIME_WORKSPACE"] = str(local_state / "runtime")
    sys.path.insert(0, str(PROJECT / "backend"))

    from uvicorn import run

    run(
        "inspection_demo.main:app",
        host="127.0.0.1",
        port=7860,
        workers=1,
        access_log=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
