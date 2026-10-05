from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
COBBIE_ROOT = WORKSPACE_ROOT / "cobbie-ecore"


def _run_cobbie_import(script: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(COBBIE_ROOT), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=COBBIE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_importing_baml_retry_does_not_load_the_database_stack() -> None:
    """Catches src.util eagerly importing DB dependencies during ToG startup."""
    result = _run_cobbie_import(
        """
import builtins
import sys
import types

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "sqlmodel" or name.startswith("sqlmodel."):
        raise ModuleNotFoundError("database stack must not load")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.modules["mlflow"] = types.ModuleType("mlflow")
loguru = types.ModuleType("loguru")
loguru.logger = object()
sys.modules["loguru"] = loguru
import src.util.baml_retry
assert "src.db" not in sys.modules
"""
    )

    assert result.returncode == 0, result.stderr


def test_importing_baml_retry_does_not_require_optional_mlflow() -> None:
    """Catches optional tracing becoming a hard inference dependency."""
    result = _run_cobbie_import(
        """
import builtins
import sys
import types

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "mlflow" or name.startswith("mlflow."):
        raise ModuleNotFoundError("mlflow is optional")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
loguru = types.ModuleType("loguru")
loguru.logger = object()
sys.modules["loguru"] = loguru
import src.util.baml_retry
"""
    )

    assert result.returncode == 0, result.stderr
