from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PRODUCTION = [
    PROJECT / "backend" / "inspection_demo",
    PROJECT / "frontend" / "src",
    PROJECT / "frontend" / "dist",
    PROJECT / "vendor" / "tog-ifc-ecore" / "src",
]
FORBIDDEN = (
    "Navigate to room 404.",
    "0aZxGK_jD1LR$XoST2vRo$",
    "DemoGroundingAdapter",
    "demo-key",
    "sample_data",
    "if \"room 404\"",
)


def main() -> None:
    hits: list[str] = []
    for root in PRODUCTION:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or ".test." in path.name:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for token in FORBIDDEN:
                if token in text:
                    hits.append(f"{path.relative_to(PROJECT)}: {token}")
    if hits:
        raise SystemExit("release fixture leakage detected:\n" + "\n".join(hits))
    lock = (PROJECT / "uv.lock").read_text(encoding="utf-8")
    if 'name = "torch-geometric"' in lock:
        raise SystemExit("torch-geometric must not be installed in the inference image")
    print("release safety checks passed")


if __name__ == "__main__":
    main()
