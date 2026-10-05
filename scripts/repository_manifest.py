"""Select reviewed source-release files and record their portable SHA-256 inventory.

This is the staging allowlist, not a scan of everything Git happens to see. Model
weights, IFC data, downloaded assets, runtime state, and deployment credentials
belong outside the source release. SOURCE_MANIFEST.json remains upstream provenance;
RELEASE_MANIFEST.json describes the current application without including itself.
Requires Python 3.12+ for the Windows junction safety check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


RELEASE_FILES = frozenset({
    ".dockerignore", ".gitattributes", ".gitignore", "Dockerfile", "README.md",
    "SOURCE_MANIFEST.json", "THIRD_PARTY_NOTICES.md", "assets.lock.json",
    "pyproject.toml", "uv.lock",
    "docs/REFINEMENT.md", "docs/WHATIF_REFINEMENT.md", "docs/NOTEBOOK_QUICKSTART.md",
    "notebooks/Launch_Demo.ipynb",
    "backend/inspection_demo/expected_assets.json",
    "frontend/index.html", "frontend/package.json", "frontend/pnpm-lock.yaml",
    "frontend/THIRD_PARTY_LICENSES.txt",
    "frontend/pnpm-workspace.yaml", "frontend/tsconfig.json",
    "frontend/tsconfig.app.json", "frontend/tsconfig.node.json",
    "frontend/vite.config.ts", "frontend/vitest.config.ts",
    "scripts/demo_launcher.py", "scripts/notebook_client.py",
    "scripts/repository_manifest.py", "scripts/run_local_server.py", "scripts/serve_local.py",
    "scripts/verify_release.py", "scripts/upload_existing_space.py", "scripts/deploy_hf.ps1",
    "scripts/tests/test_demo_launcher.py", "scripts/tests/test_notebook_client.py",
    "scripts/tests/test_repository_manifest.py",
    "scripts/tests/test_launcher_lifecycle.py",
    "vendor/cobbie-ecore/LICENSE", "vendor/cobbie-ecore/README.md",
    "vendor/cobbie-ecore/src/__init__.py", "vendor/cobbie-ecore/src/config.py",
    "vendor/cobbie-ecore/src/integrations/__init__.py",
    "vendor/cobbie-ecore/src/integrations/tog.py",
    "vendor/cobbie-ecore/src/integrations/tog_wire.py",
    "vendor/cobbie-ecore/src/schemas/__init__.py",
    "vendor/cobbie-ecore/src/schemas/agent_error.py",
    "vendor/cobbie-ecore/src/schemas/result.py",
    "vendor/cobbie-ecore/src/util/__init__.py",
    "vendor/cobbie-ecore/src/util/baml_retry.py",
    "vendor/tog-ifc-ecore/LICENSE", "vendor/tog-ifc-ecore/README.md",
    "vendor/tog-ifc-ecore/pyproject.toml", "vendor/tog-ifc-ecore/src/tog/py.typed",
    "vendor/text-gnn-plugin/LICENSE", "vendor/text-gnn-plugin/README.md",
    "vendor/text-gnn-plugin/pyproject.toml",
})

RELEASE_TREES = {
    "backend/inspection_demo": frozenset({".py"}),
    "backend/tests": frozenset({".py"}),
    "frontend/src": frozenset({".ts", ".tsx", ".js", ".jsx", ".css", ".svg"}),
    "frontend/dist": frozenset({
        ".html", ".js", ".css", ".woff", ".woff2", ".ttf", ".otf",
        ".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico",
    }),
    "vendor/cobbie-ecore/src/baml": frozenset({".py", ".baml"}),
    "vendor/tog-ifc-ecore/src/tog": frozenset({".py"}),
}

PRIVATE_COMPONENTS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache", "cache",
    "caches", ".tmp", "tmp", ".webapp", ".demo-state", ".ipynb_checkpoints", "node_modules",
    "data", "datasets", "checkpoints", "checkpoint", "outputs", "output",
    "logs", "responses", "secrets", ".aws", ".ssh",
})


def _private(relative: Path) -> bool:
    for component in relative.parts:
        name = component.lower()
        if (name in PRIVATE_COMPONENTS or name.startswith((".env", ".runtime"))
                or name.endswith(".egg-info")):
            return True
    name = relative.name.lower()
    return any(token in name for token in ("private_key", "private-key", "credentials", "secret"))


def _checked_path(root: Path, path: Path) -> bool:
    """Reject links before inspecting their targets; skip embedded Git checkouts."""
    current = root
    for component in path.relative_to(root).parts:
        current = current / component
        if current.is_symlink() or current.is_junction():
            raise ValueError(f"Symlink or junction is not a release input: {current.relative_to(root)}")
        if current.is_dir() and (current / ".git").exists():
            return False
    return True


def collect_files(project_root: Path) -> list[Path]:
    """Return sorted absolute regular-file paths from the reviewed release allowlist.

    Unknown files are omitted. Links within a selected source tree fail closed;
    embedded repositories and private/cache directories are never traversed.
    """
    root = project_root.absolute()
    if root.is_symlink() or root.is_junction():
        raise ValueError("Symlink or junction cannot be the release project root")
    if not root.is_dir():
        raise ValueError("Release project root must be an existing directory")
    root = root.resolve()
    selected: set[Path] = set()
    for relative in RELEASE_FILES:
        path = root / relative
        if _checked_path(root, path) and path.is_file():
            selected.add(path)
    for relative, suffixes in RELEASE_TREES.items():
        tree = root / relative
        if not _checked_path(root, tree) or not tree.is_dir():
            continue
        pending = [tree]
        while pending:
            directory = pending.pop()
            for path in directory.iterdir():
                if _private(path.relative_to(root)):
                    continue
                if not _checked_path(root, path):
                    continue
                if path.is_dir():
                    pending.append(path)
                elif path.is_file() and path.suffix.lower() in suffixes:
                    selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _inventory(project_root: Path) -> dict:
    root = project_root.resolve()
    files = []
    for path in collect_files(project_root):
        content = path.read_bytes()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    return {
        "schema_version": "inspection-demo-source-release-v1",
        "hash_algorithm": "sha256",
        "files": files,
    }


def write_manifest(project_root: Path, output: Path) -> dict:
    """Write a deterministic inventory without changing upstream provenance."""
    root = project_root.resolve()
    output = output if output.is_absolute() else root / output
    if not output.resolve().is_relative_to(root):
        raise ValueError("Manifest output must stay inside the project root")
    if _private(output.relative_to(root)):
        raise ValueError("Manifest output cannot modify private files or Git metadata")
    if not _checked_path(root, output):
        raise ValueError("Manifest output cannot be inside an embedded repository")
    if output.resolve() in collect_files(project_root):
        raise ValueError("Manifest output cannot overwrite a release source file")
    manifest = _inventory(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("RELEASE_MANIFEST.json"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write the SHA-256 inventory")
    action.add_argument("--check", action="store_true", help="fail when the inventory is stale")
    action.add_argument("--list", action="store_true", help="print relative files for explicit staging")
    args = parser.parse_args(argv)
    try:
        if args.list:
            for path in collect_files(args.project_root):
                print(path.relative_to(args.project_root.resolve()).as_posix())
            return 0
        if args.write:
            manifest = write_manifest(args.project_root, args.output)
            print(f"Wrote release inventory for {len(manifest['files'])} files.")
            return 0
        output = args.output if args.output.is_absolute() else args.project_root / args.output
        expected = json.loads(output.read_text(encoding="utf-8"))
        current = _inventory(args.project_root)
        if expected != current:
            print("Release inventory is stale; review changes before running --write.", file=sys.stderr)
            return 1
        print(f"Release inventory matches {len(current['files'])} files.")
        return 0
    except (OSError, ValueError) as error:
        print(f"Release inventory failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
