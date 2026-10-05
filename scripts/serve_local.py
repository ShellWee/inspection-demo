"""Owned local Uvicorn worker with cooperative shutdown and process-tree cleanup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time


class OwnedReadiness:
    """Keep launch identity separate from the production API and its frontend."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/_launcher/ready":
            ready = scope["method"] == "GET" and getattr(self.app.state, "jobs", None) is not None
            body = json.dumps({"status": "ok", "ownerToken": self.token} if ready
                              else {"status": "starting"}).encode()
            await send({"type": "http.response.start", "status": 200 if ready else 503,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def owned_stop_requested(state: Path, token: str) -> bool:
    try:
        value = json.loads((state / "stop.json").read_text(encoding="utf-8"))
        return isinstance(value, dict) and value.get("token") == token
    except (OSError, ValueError):
        return False


def watch_stop(server, state: Path, token: str, finished: threading.Event) -> None:
    """Use should_exit on Windows too, allowing FastAPI lifespan cleanup to run."""
    while not finished.wait(0.2):
        try:
            owned = (state / "active.lock").read_text(encoding="utf-8") == token
            status = json.loads((state / "status.json").read_text(encoding="utf-8"))
            alive = time.time() - float(status.get("heartbeat", 0)) < 15
        except (OSError, ValueError, TypeError, AttributeError):
            owned = alive = False
        if not owned or not alive or owned_stop_requested(state, token):
            server.should_exit = True
            return


class OwnedProcessTree:
    """OS ownership established on a current Popen object, never a persisted PID.

    Windows jobs kill descendants even if the supervisor crashes. The worker
    waits for the supervisor's gate before it imports the app or creates work.
    POSIX workers run in a fresh process group for bounded forced cleanup.
    """

    def __init__(self):
        self.handle = None
        self.child = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t),
                        ("maximum", ctypes.c_size_t), ("active", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, child: subprocess.Popen) -> None:
        if self.handle:
            import ctypes
            if not self.kernel.AssignProcessToJobObject(self.handle, int(child._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        elif os.getpgid(child.pid) != child.pid:
            raise RuntimeError("Worker must own its own process group.")
        self.child = child

    def terminate(self, child: subprocess.Popen) -> None:
        if child is not self.child:
            raise RuntimeError("Refusing to terminate an unowned process.")
        if self.handle:
            import ctypes
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
        elif child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
        elif self.child is not None:
            # Also reap descendants left behind by an unexpectedly exited worker.
            # This group was established on this launch's current Popen object.
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.child = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    state = args.state_dir.resolve()
    config = json.loads((state / "launch.json").read_text(encoding="utf-8"))
    token = config["token"]
    deadline = time.monotonic() + 10
    while True:
        if (state / "active.lock").read_text(encoding="utf-8") != token:
            raise RuntimeError("Launch ownership changed before worker startup.")
        if owned_stop_requested(state, token):
            return 0
        try:
            gate = json.loads((state / "worker-ready.json").read_text(encoding="utf-8"))
            if secrets.compare_digest(str(gate.get("token", "")), token):
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("Supervisor did not establish process ownership.")
        time.sleep(0.1)

    import uvicorn
    from inspection_demo.main import app

    server = uvicorn.Server(uvicorn.Config(
        OwnedReadiness(app, token), host="127.0.0.1", port=config["port"], workers=1,
        access_log=False, server_header=False, timeout_graceful_shutdown=3,
    ))
    finished = threading.Event()
    watcher = threading.Thread(target=watch_stop, args=(server, state, token, finished), daemon=True)
    watcher.start()
    try:
        server.run()
    finally:
        finished.set()
        watcher.join(timeout=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
