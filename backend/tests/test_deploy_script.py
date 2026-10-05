from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt" or shutil.which("pwsh") is None, reason="Windows only")
def test_deploy_stops_before_asset_upload_when_space_preflight_fails(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    log_path = tmp_path / "hf-calls.txt"
    fake_hf = tmp_path / "hf.cmd"
    fake_hf.write_text(
        "@echo off\n"
        f'echo %*>>"{log_path}"\n'
        'echo %* | findstr /C:"inspection-demo-ecore --type space" >nul\n'
        "if not errorlevel 1 exit /b 42\n"
        "exit /b 0\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"

    completed = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(project / "scripts" / "deploy_hf.ps1"),
            "-Namespace",
            "test-user",
        ],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8")

    assert completed.returncode != 0
    assert "repos create test-user/inspection-demo-ecore-assets-v1" in calls
    assert "repos create test-user/inspection-demo-ecore --type space" in calls
    assert "upload-large-folder" not in calls
    assert "upload test-user/inspection-demo-ecore" not in calls
