from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

SPACE_FILES = [
    ".dockerignore",
    "Dockerfile",
    "README.md",
    "docs/REFINEMENT.md",
    "SOURCE_MANIFEST.json",
    "pyproject.toml",
    "uv.lock",
    "backend/inspection_demo/**",
    "frontend/index.html",
    "frontend/package.json",
    "frontend/pnpm-lock.yaml",
    "frontend/pnpm-workspace.yaml",
    "frontend/tsconfig*.json",
    "frontend/vite.config.ts",
    "frontend/vitest.config.ts",
    "frontend/src/**",
    "vendor/cobbie-ecore/src/__init__.py",
    "vendor/cobbie-ecore/src/config.py",
    "vendor/cobbie-ecore/src/baml/**",
    "vendor/cobbie-ecore/src/integrations/__init__.py",
    "vendor/cobbie-ecore/src/integrations/tog.py",
    "vendor/cobbie-ecore/src/integrations/tog_wire.py",
    "vendor/cobbie-ecore/src/schemas/__init__.py",
    "vendor/cobbie-ecore/src/schemas/agent_error.py",
    "vendor/cobbie-ecore/src/schemas/result.py",
    "vendor/cobbie-ecore/src/util/__init__.py",
    "vendor/cobbie-ecore/src/util/baml_retry.py",
    "vendor/text-gnn-plugin/pyproject.toml",
    "vendor/tog-ifc-ecore/README.md",
    "vendor/tog-ifc-ecore/pyproject.toml",
    "vendor/tog-ifc-ecore/src/tog/**",
]

SPACE_IGNORES = [
    "**/.env*",
    "**/__pycache__/**",
    "**/*.egg-info/**",
    "**/*.py[cod]",
    "**/*.test.*",
    "frontend/src/test/**",
]


def upload_existing_space(
    *, project_root: Path, repo_id: str, api: Any | None = None
) -> str:
    """Commit to an existing Space without the CLI's implicit Gradio create call."""

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.repo_info(repo_id=repo_id, repo_type="space")
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=project_root.resolve(),
        path_in_repo=".",
        allow_patterns=SPACE_FILES,
        ignore_patterns=SPACE_IGNORES,
        commit_message="Deploy fixed Ecore inspection demo",
    )
    return str(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        upload_existing_space(
            project_root=arguments.project_root,
            repo_id=arguments.repo_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
