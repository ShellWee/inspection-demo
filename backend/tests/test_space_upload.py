from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))

from upload_existing_space import upload_existing_space  # noqa: E402


class CapturingApi:
    def __init__(self) -> None:
        self.info_calls: list[dict[str, object]] = []
        self.upload_calls: list[dict[str, object]] = []

    def repo_info(self, **kwargs):
        self.info_calls.append(kwargs)
        return object()

    def upload_folder(self, **kwargs):
        self.upload_calls.append(kwargs)
        return "https://huggingface.co/spaces/test-user/inspection-demo-ecore/commit/test"


def test_existing_space_upload_commits_without_attempting_repo_creation(tmp_path: Path) -> None:
    api = CapturingApi()

    result = upload_existing_space(
        project_root=tmp_path,
        repo_id="test-user/inspection-demo-ecore",
        api=api,
    )

    assert result.endswith("/commit/test")
    assert api.info_calls == [
        {"repo_id": "test-user/inspection-demo-ecore", "repo_type": "space"}
    ]
    assert api.upload_calls[0]["repo_type"] == "space"
    assert "backend/inspection_demo/**" in api.upload_calls[0]["allow_patterns"]
    assert "**/.env*" in api.upload_calls[0]["ignore_patterns"]
    assert not hasattr(api, "create_repo")
