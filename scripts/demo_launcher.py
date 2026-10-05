"""Portable, loopback-only launcher. No paid inference or credentials in CLI arguments.

Runs with a Python 3.10+ notebook kernel; the application uses its own Python 3.12
environment. Windows Application Control errors require administrator approval,
not an alternative execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

PROJECT = Path(__file__).resolve().parents[1]


class LauncherError(RuntimeError):
    pass


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_assets(assets: Path, expected: Path) -> int:
    assets = assets.resolve()
    manifest = assets / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink() or digest(manifest) != digest(expected):
        raise LauncherError("Asset manifest is missing or does not match this release.")
    entries = json.loads(expected.read_text(encoding="utf-8"))["entries"]
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or "\\" in str(relative):
            raise LauncherError("Unsafe asset path in manifest.")
        path = assets.joinpath(*relative.parts)
        if any(item.is_symlink() for item in [path, *path.parents] if item != assets.parent):
            raise LauncherError("Symlink asset path is not allowed.")
        if not path.resolve().is_relative_to(assets) or not path.is_file():
            raise LauncherError(f"Missing asset: {relative}")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise LauncherError(f"Asset size/hash mismatch: {relative}")
    return len(entries)


def project_python(project: Path = PROJECT) -> Path:
    return project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def uv_command() -> list[str]:
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    if executable := shutil.which("uv"):
        return [executable]
    raise LauncherError("Install uv first: python -m pip install uv==0.12.9")


def setup_commands(project: Path, uv: list[str], device: str) -> list[list[str]]:
    if device not in {"cpu", "cuda"}:
        raise LauncherError("Choose cpu or cuda explicitly.")
    if device == "cuda" and sys.platform != "linux":
        raise LauncherError("This release's CUDA profile supports Linux only; use cpu on Windows.")
    return [
        [*uv, "sync", "--frozen", "--no-dev", "--python", "3.12", "--project", str(project)],
        [*uv, "pip", "install", "--python", str(project_python(project)), "--no-deps",
         str(project / "vendor" / "tog-ifc-ecore"), str(project / "vendor" / "text-gnn-plugin")],
    ]


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    try:
        result = subprocess.run(command, cwd=PROJECT, env=env, check=False)
    except OSError as error:
        raise LauncherError(
            "Unable to execute the approved runtime. If Application Control blocked it, "
            "ask your administrator to approve Python; do not disable the policy."
        ) from error
    if result.returncode:
        raise LauncherError(f"Setup command failed (exit {result.returncode}); see output above.")


def runtime_environment(
    project: Path, assets: Path, state: Path, device: str, base: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for key in ("OPENAI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        env.pop(key, None)
    env.update({
        "INSPECTION_DEMO_ASSET_MOUNT": str(assets),
        "INSPECTION_DEMO_RESEARCH_DEVICE": device,
        "INSPECTION_DEMO_DATA_DIR": str(state / "data"),
        "INSPECTION_DEMO_RUNTIME_WORKSPACE": str(state / "runtime"),
        "INSPECTION_DEMO_FRONTEND_DIST": str(project / "frontend" / "dist"),
        "PYTHONPATH": os.pathsep.join((str(project / "backend"),
                                     str(project / "vendor" / "tog-ifc-ecore" / "src"))),
        "PYTHONUNBUFFERED": "1",
    })
    return env


def download_assets(assets: Path) -> None:
    lock = json.loads((PROJECT / "assets.lock.json").read_text(encoding="utf-8"))
    expected = PROJECT / "backend" / "inspection_demo" / "expected_assets.json"
    if digest(expected) != lock["manifest_sha256"]:
        raise LauncherError("Asset source lock and expected manifest disagree.")
    if len(lock["revision"]) != 40 or any(c not in "0123456789abcdef" for c in lock["revision"]):
        raise LauncherError("Asset download requires an immutable dataset commit.")
    interpreter = project_python()
    if not interpreter.is_file():
        raise LauncherError("Run setup before downloading assets.")
    code = (
        "import json, sys; from huggingface_hub import snapshot_download; "
        "entries=json.load(open(sys.argv[4], encoding='utf-8'))['entries']; "
        "snapshot_download(repo_id=sys.argv[1], repo_type='dataset', revision=sys.argv[2], "
        "local_dir=sys.argv[3], allow_patterns=['manifest.json']+[e['path'] for e in entries])"
    )
    try:
        result = subprocess.run(
            [str(interpreter), "-c", code, lock["repo_id"], lock["revision"], str(assets), str(expected)],
            cwd=PROJECT, capture_output=True, text=True, check=False,
        )
    except OSError as error:
        raise LauncherError("Python could not run; resolve OS execution policy with your administrator.") from error
    if result.returncode:
        # Provider errors can include request details. Do not echo them or cached tokens.
        raise LauncherError("Asset download failed. Check network, disk space and HF login with private dataset access.")


def require_free_port(port: int) -> None:
    if not 1024 <= port <= 65535:
        raise LauncherError("Port must be between 1024 and 65535.")
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise LauncherError(f"Port {port} is in use or unavailable; choose another port.") from error


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read_status(state: Path) -> dict:
    try:
        result = json.loads((state / "status.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "stopped"}
    except (OSError, ValueError):
        return {"status": "failed", "reason": "Launcher status is unreadable; inspect its state directory."}
    if (not isinstance(result, dict)
            or result.get("status") not in {"starting", "running", "stopping", "stopped", "failed"}
            or not isinstance(result.get("heartbeat", 0), (int, float))
            or not math.isfinite(result.get("heartbeat", 0))):
        return {"status": "failed", "reason": "Launcher status is malformed; inspect its state directory."}
    if result.get("status") in {"starting", "running", "stopping"} and time.time() - result.get("heartbeat", 0) > 10:
        return {**result, "status": "stale"}
    return result


def request_stop(state: Path) -> None:
    try:
        token = (state / "active.lock").read_text(encoding="utf-8")
        config = json.loads((state / "launch.json").read_text(encoding="utf-8"))
        if not token or not isinstance(config, dict) or config.get("token") != token:
            raise ValueError("mismatch")
    except (OSError, ValueError) as error:
        raise LauncherError("No launcher-owned worker to stop; no process was killed.") from error
    write_json(state / "stop.json", {"token": token})


def healthy(port: int, token: str) -> bool:
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        # Loopback must not travel through the user's corporate/public proxy.
        with build_opener(ProxyHandler({}), NoRedirect()).open(
            f"http://127.0.0.1:{port}/_launcher/ready", timeout=2,
        ) as response:
            payload = json.loads(response.read(4096))
            return (response.status == 200 and isinstance(payload, dict)
                    and payload.get("status") == "ok"
                    and isinstance(payload.get("ownerToken"), str)
                    and secrets.compare_digest(payload["ownerToken"], token))
    except (URLError, HTTPError, TimeoutError, OSError, ValueError):
        return False


def supervise(state: Path) -> int:
    """Only this supervisor terminates the child it created; never kill a stored PID."""
    from serve_local import OwnedProcessTree, owned_stop_requested

    token = (state / "active.lock").read_text(encoding="utf-8")
    child = None
    tree = None
    result = {"status": "starting", "heartbeat": time.time()}
    try:
        config = json.loads((state / "launch.json").read_text(encoding="utf-8"))
        if not token or config.get("token") != token:
            raise LauncherError("Launcher lock changed.")
        result["url"] = f"http://127.0.0.1:{config['port']}/"
        deadline = time.monotonic() + config["timeout"]
        tree = OwnedProcessTree()
        command = [str(project_python()), str(PROJECT / "scripts" / "serve_local.py"),
                   "--state-dir", str(state)]
        with (state / "server.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                env=runtime_environment(PROJECT, Path(config["assets"]), state, config["device"]),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
        tree.attach(child)
        write_json(state / "worker-ready.json", {"token": token})
        while child.poll() is None:
            result["heartbeat"] = time.time()
            if owned_stop_requested(state, token):
                result["status"] = "stopped"
                break
            if result["status"] == "starting":
                if healthy(config["port"], token):
                    result["status"] = "running"
                elif time.monotonic() > deadline:
                    result.update(status="failed", reason="Readiness timeout; inspect server.log and asset/runtime setup.")
                    break
            write_json(state / "status.json", result)
            time.sleep(0.4)
        else:
            if owned_stop_requested(state, token) and child.returncode == 0:
                result["status"] = "stopped"
            else:
                result.update(status="failed", reason="Server exited; inspect server.log.")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, LauncherError):
        result.update(status="failed", reason="Worker setup failed; inspect server.log and OS execution approval.")
    finally:
        try:
            if child is not None and child.poll() is None:
                write_json(state / "stop.json", {"token": token})
                grace_deadline = time.monotonic() + 10
                while child.poll() is None and time.monotonic() < grace_deadline:
                    write_json(state / "status.json", {**result, "status": "stopping", "heartbeat": time.time()})
                    time.sleep(0.2)
                if child.poll() is None:
                    if tree is not None and tree.child is child:
                        tree.terminate(child)
                    else:
                        # Worker gate was never opened if assignment failed.
                        child.kill()
                    child.wait(timeout=5)
            write_json(state / "status.json", {**result, "heartbeat": time.time()})
        finally:
            if tree is not None:
                tree.close()
            if (state / "active.lock").exists() and (state / "active.lock").read_text() == token:
                (state / "active.lock").unlink()
    return 0 if result["status"] == "stopped" else 1


def start(assets: Path, state: Path, device: str, port: int, timeout: int) -> dict:
    if not project_python().is_file():
        raise LauncherError("Application environment missing; run setup first.")
    if not (PROJECT / "frontend" / "dist" / "index.html").is_file():
        raise LauncherError("Prebuilt UI missing; use a complete release or build frontend first.")
    if state == PROJECT or state == assets or state == Path(state.anchor):
        raise LauncherError("Use a dedicated state directory, not a project/asset/filesystem root.")
    verify_assets(assets, PROJECT / "backend" / "inspection_demo" / "expected_assets.json")
    require_free_port(port)
    state.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(24)
    try:
        with (state / "active.lock").open("x", encoding="utf-8") as stream:
            stream.write(token)
    except FileExistsError as error:
        raise LauncherError(
            "This state directory is busy. Use status/stop. If stale after a crash or reboot, "
            "confirm its old workers have exited before choosing a fresh dedicated state directory. "
            "No process or lock was removed."
        ) from error
    supervisor = None
    try:
        write_json(state / "launch.json", {"token": token, "assets": str(assets), "device": device,
                                          "port": port, "timeout": timeout})
        write_json(state / "status.json", {"status": "starting", "heartbeat": time.time()})
        supervisor = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_supervise", "--state-dir", str(state)],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
            env=runtime_environment(PROJECT, assets, state, device),
        )
        deadline = time.monotonic() + timeout + 10
        while time.monotonic() < deadline:
            status = read_status(state)
            if status["status"] == "running":
                return status
            if status["status"] in {"failed", "stopped", "stale"} or supervisor.poll() is not None:
                raise LauncherError(status.get("reason", "Launcher stopped unexpectedly."))
            time.sleep(0.5)
        raise LauncherError("Startup timed out; stop requested.")
    except (KeyboardInterrupt, LauncherError, OSError):
        if supervisor is None:
            if (state / "active.lock").exists() and (state / "active.lock").read_text() == token:
                (state / "active.lock").unlink()
        elif (state / "active.lock").exists():
            try:
                request_stop(state)
            except (LauncherError, OSError):
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Install locked Python 3.12 dependencies and plugins")
    setup.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    setup.add_argument("--dry-run", action="store_true")
    assets = commands.add_parser("assets", help="Verify assets; explicitly opt into private HF download")
    assets.add_argument("--assets-dir", type=Path, default=PROJECT / "assets")
    assets.add_argument("--download", action="store_true")
    launch = commands.add_parser("start", help="Start a local-only background server")
    launch.add_argument("--assets-dir", type=Path, default=PROJECT / "assets")
    launch.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    launch.add_argument("--port", type=int, default=7860)
    launch.add_argument("--timeout", type=int, default=300)
    for command in (launch, commands.add_parser("status"), commands.add_parser("stop"), commands.add_parser("_supervise", help=argparse.SUPPRESS)):
        command.add_argument("--state-dir", type=Path, default=PROJECT / ".demo-state")
    args = parser.parse_args()
    try:
        if args.command == "setup":
            uv = ["uv"] if args.dry_run else uv_command()
            for command in setup_commands(PROJECT, uv, args.device):
                if args.dry_run:
                    print(json.dumps(command))
                else:
                    # Never let ambient UV_PROJECT_ENVIRONMENT replace a user's other environment.
                    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(PROJECT / ".venv"))
                    run_checked(command, env=env)
            if not args.dry_run:
                print("Environment installed. Next: verify/download assets, then start.")
        elif args.command == "assets":
            root = args.assets_dir.resolve()
            if args.download:
                download_assets(root)
            count = verify_assets(root, PROJECT / "backend" / "inspection_demo" / "expected_assets.json")
            print(f"Verified {count} asset files.")
        elif args.command == "start":
            if not 1 <= args.timeout <= 600:
                raise LauncherError("Startup timeout must be 1–600 seconds.")
            print(json.dumps(start(args.assets_dir.resolve(), args.state_dir.resolve(), args.device, args.port, args.timeout)))
        elif args.command == "status":
            print(json.dumps(read_status(args.state_dir.resolve())))
        elif args.command == "stop":
            request_stop(args.state_dir.resolve())
            deadline = time.monotonic() + 20
            while (args.state_dir / "active.lock").exists() and time.monotonic() < deadline:
                time.sleep(0.3)
            if (args.state_dir / "active.lock").exists():
                raise LauncherError("Stop requested but supervisor did not acknowledge; inspect status. No arbitrary PID killed.")
            print("Stopped launcher-owned server.")
        else:
            return supervise(args.state_dir.resolve())
    except (LauncherError, OSError, ValueError) as error:
        print(f"Launcher error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
