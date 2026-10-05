import shlex
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _stage_docker_cobbie_snapshot(destination: Path) -> Path:
    """Materialize the Cobbie files selected by Docker COPY instructions."""

    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignore_rules = [
        line.strip()
        for line in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    def ignored(relative: str) -> bool:
        result = False
        for rule in ignore_rules:
            negate = rule.startswith("!")
            pattern = rule[1:] if negate else rule
            if fnmatch(relative, pattern):
                result = not negate
        return result

    app_root = destination / "app"
    for line in dockerfile.splitlines():
        if not line.startswith("COPY vendor/cobbie-ecore/"):
            continue
        parts = shlex.split(line)
        sources, target = parts[1:-1], parts[-1]
        target_path = app_root / target.removeprefix("./")
        for source in sources:
            if ignored(source):
                continue
            source_path = PROJECT_ROOT / source
            if len(sources) > 1 or target.endswith("/"):
                copied_path = target_path / source_path.name
            else:
                copied_path = target_path
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.is_dir():
                shutil.copytree(source_path, copied_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, copied_path)
    return app_root / "vendor" / "cobbie-ecore"


def test_frontend_supply_chain_policy_is_available_before_pnpm_install() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    policy_copy = (
        "COPY frontend/package.json frontend/pnpm-lock.yaml "
        "frontend/pnpm-workspace.yaml ./"
    )

    assert policy_copy in dockerfile
    assert dockerfile.index(policy_copy) < dockerfile.index(
        "RUN pnpm install --frozen-lockfile"
    )


def test_runtime_reuses_base_image_uid_1000_when_present() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "if ! getent passwd 1000" in dockerfile
    assert "USER 1000" in dockerfile


def test_docker_cobbie_snapshot_imports_grounding_integration(tmp_path: Path) -> None:
    """Catches omitted package files that crash the isolated grounding worker."""

    cobbie_root = _stage_docker_cobbie_snapshot(tmp_path)
    tog_root = PROJECT_ROOT / "vendor" / "tog-ifc-ecore" / "src"
    code = (
        "import sys; "
        f"sys.path[:0] = [{str(cobbie_root)!r}, {str(tog_root)!r}]; "
        "import src.integrations.tog; "
        "print('IMPORT_OK')"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "IMPORT_OK"
