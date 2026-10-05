from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import demo_launcher as launcher


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.token = "a" * 48

    def wrapper(self):
        self.assertTrue((SCRIPTS / "serve_local.py").is_file(), "owned server wrapper missing")
        import serve_local
        return serve_local

    def ready(self, port, token):
        try:
            return launcher.healthy(port, token)
        except TypeError:
            self.fail("readiness has no launch ownership check")

    @contextmanager
    def listener(self, payload, *, redirect=False):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler):
                handler.send_response(302 if redirect else 200)
                if redirect:
                    handler.send_header("Location", "/target")
                handler.end_headers()
                handler.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_readiness_rejects_unrelated_http_200(self):
        with self.listener({"status": "ok"}) as port:
            self.assertFalse(self.ready(port, self.token))

    def test_readiness_requires_exact_owner_and_never_follows_redirect(self):
        with self.listener({"status": "ok", "ownerToken": self.token}) as port:
            self.assertTrue(self.ready(port, self.token))
            self.assertFalse(self.ready(port, "b" * 48))
        with self.listener({"status": "ok", "ownerToken": self.token}, redirect=True) as port:
            self.assertFalse(self.ready(port, self.token))

    def test_start_releases_owned_lock_when_initial_state_write_fails(self):
        frontend = self.root / "frontend" / "dist"
        frontend.mkdir(parents=True)
        (frontend / "index.html").write_text("test")
        with patch.object(launcher, "PROJECT", self.root), \
             patch.object(launcher, "project_python", return_value=Path(sys.executable)), \
             patch.object(launcher, "verify_assets", return_value=1), \
             patch.object(launcher, "require_free_port"), \
             patch.object(launcher, "write_json", side_effect=OSError("disk unavailable")):
            with self.assertRaises((OSError, launcher.LauncherError)):
                launcher.start(self.root / "assets", self.state, "cpu", 7860, 1)
        self.assertFalse((self.state / "active.lock").exists())

    def test_readiness_wrapper_never_announces_unready_application(self):
        wrapper = self.wrapper()
        app = SimpleNamespace(state=SimpleNamespace(jobs=None))
        wrapped = wrapper.OwnedReadiness(app, self.token)

        async def request():
            messages = []
            async def send(message):
                messages.append(message)
            await wrapped({"type": "http", "path": "/_launcher/ready", "method": "GET"}, None, send)
            return messages

        self.assertEqual(asyncio.run(request())[0]["status"], 503)
        app.state.jobs = object()
        messages = asyncio.run(request())
        self.assertEqual(messages[0]["status"], 200)
        self.assertEqual(json.loads(messages[1]["body"]), {"status": "ok", "ownerToken": self.token})

    def test_owned_stop_requests_graceful_shutdown_without_process_signal(self):
        wrapper = self.wrapper()
        (self.state / "active.lock").write_text(self.token)
        launcher.write_json(self.state / "status.json", {"heartbeat": time.time()})
        launcher.write_json(self.state / "stop.json", {"token": "someone-else"})
        server = SimpleNamespace(should_exit=False)
        finished = threading.Event()
        watcher = threading.Thread(target=wrapper.watch_stop, args=(server, self.state, self.token, finished))
        watcher.start()
        try:
            time.sleep(0.3)
            self.assertFalse(server.should_exit)
            launcher.write_json(self.state / "stop.json", {"token": self.token})
            watcher.join(timeout=3)
            self.assertTrue(server.should_exit)
        finally:
            finished.set()
            watcher.join(timeout=3)

    def test_owned_tree_cleanup_stops_spawned_descendant(self):
        wrapper = self.wrapper()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        code = (
            "import socket,time; s=socket.socket(); "
            f"s.bind(('127.0.0.1',{port})); s.listen(); time.sleep(60)"
        )
        # The gate prevents descendants being created before job assignment on Windows.
        gate = self.state / "gate"
        parent_code = (
            "import pathlib,subprocess,sys,time; p=pathlib.Path(sys.argv[1]); "
            "exec('while not p.exists(): time.sleep(0.02)'); "
            "subprocess.Popen([sys.executable,'-c',sys.argv[2]]); time.sleep(60)"
        )
        tree = wrapper.OwnedProcessTree()
        child = subprocess.Popen([sys.executable, "-c", parent_code, str(gate), code],
                                 start_new_session=sys.platform != "win32")
        try:
            tree.attach(child)
            gate.write_text("ready")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                self.fail("descendant never opened its listener")
            tree.terminate(child)
            child.wait(timeout=5)
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=0.2)
        finally:
            tree.close()
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_supervisor_stops_real_worker_cooperatively_and_releases_lock(self):
        (self.state / "active.lock").write_text(self.token)
        launcher.write_json(self.state / "launch.json", {
            "token": self.token, "port": 7860, "timeout": 2,
            "assets": str(self.root / "assets"), "device": "cpu",
        })
        launcher.write_json(self.state / "status.json", {"status": "starting", "heartbeat": time.time()})
        # The harmless stdlib worker writes a marker only on cooperative exit.
        worker = """
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1])
token = sys.argv[2]
while True:
    try:
        stop = json.loads((p / 'stop.json').read_text())
    except (OSError, ValueError):
        stop = {}
    if stop.get('token') == token:
        break
    time.sleep(.02)
(p / 'graceful-exit').write_text('finished')
"""
        real_popen = subprocess.Popen
        errors = []

        def spawn(command, **kwargs):
            return real_popen([sys.executable, "-c", worker, str(self.state), self.token], **kwargs)

        def supervise():
            try:
                launcher.supervise(self.state)
            except BaseException as error:
                errors.append(error)

        with patch.object(launcher.subprocess, "Popen", side_effect=spawn), \
             patch.object(launcher, "healthy", return_value=True):
            thread = threading.Thread(target=supervise)
            thread.start()
            deadline = time.monotonic() + 5
            while launcher.read_status(self.state)["status"] == "starting" and time.monotonic() < deadline:
                time.sleep(.05)
            launcher.request_stop(self.state)
            thread.join(timeout=15)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue((self.state / "graceful-exit").exists())
        self.assertEqual(launcher.read_status(self.state)["status"], "stopped")
        self.assertFalse((self.state / "active.lock").exists())

    def test_supervisor_cleans_owned_lock_when_launch_json_is_malformed(self):
        (self.state / "active.lock").write_text(self.token)
        (self.state / "launch.json").write_text("{")
        self.assertEqual(launcher.supervise(self.state), 1)
        self.assertFalse((self.state / "active.lock").exists())
        self.assertEqual(launcher.read_status(self.state)["status"], "failed")

    def test_stop_rejects_nonobject_configuration_safely(self):
        (self.state / "active.lock").write_text(self.token)
        (self.state / "launch.json").write_text("[]")
        with self.assertRaises(launcher.LauncherError):
            launcher.request_stop(self.state)
        self.assertFalse((self.state / "stop.json").exists())

    def test_malformed_status_is_reported_as_failed(self):
        for value in ([], {"status": "running", "heartbeat": "bad"}):
            with self.subTest(value=value):
                (self.state / "status.json").write_text(json.dumps(value))
                self.assertEqual(launcher.read_status(self.state)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
