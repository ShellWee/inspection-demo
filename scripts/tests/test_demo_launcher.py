from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("demo_launcher"), "launcher is missing")
        import demo_launcher
        self.launcher = demo_launcher
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def asset_fixture(self):
        assets = self.root / "assets"
        assets.mkdir()
        (assets / "graph.bin").write_bytes(b"verified")
        manifest = {"entries": [{"path": "graph.bin", "bytes": 8,
                                 "sha256": hashlib.sha256(b"verified").hexdigest()}]}
        expected = self.root / "expected.json"
        expected.write_text(json.dumps(manifest))
        (assets / "manifest.json").write_bytes(expected.read_bytes())
        return assets, expected

    def test_accepts_only_matching_asset_contents(self):
        assets, expected = self.asset_fixture()
        self.assertEqual(self.launcher.verify_assets(assets, expected), 1)
        (assets / "graph.bin").write_bytes(b"modified")
        with self.assertRaisesRegex(self.launcher.LauncherError, "hash"):
            self.launcher.verify_assets(assets, expected)

    def test_rejects_manifest_replacement_before_trusting_entries(self):
        assets, expected = self.asset_fixture()
        (assets / "manifest.json").write_text('{"entries": []}')
        with self.assertRaisesRegex(self.launcher.LauncherError, "manifest"):
            self.launcher.verify_assets(assets, expected)

    def test_rejects_traversal_even_in_expected_manifest(self):
        assets, expected = self.asset_fixture()
        manifest = json.loads(expected.read_text())
        manifest["entries"][0]["path"] = "../outside.bin"
        expected.write_text(json.dumps(manifest))
        (assets / "manifest.json").write_bytes(expected.read_bytes())
        with self.assertRaisesRegex(self.launcher.LauncherError, "path"):
            self.launcher.verify_assets(assets, expected)

    def test_setup_commands_pin_python_lock_and_plugin_registration(self):
        commands = self.launcher.setup_commands(self.root, ["uv"], "cpu")
        self.assertIn("--frozen", commands[0])
        self.assertIn("3.12", commands[0])
        self.assertIn("--no-deps", commands[1])
        self.assertIn(str(self.root / "vendor" / "text-gnn-plugin"), commands[1])

    def test_runtime_uses_only_selected_asset_and_state_paths(self):
        env = self.launcher.runtime_environment(
            self.root, self.root / "assets", self.root / "state", "cpu",
            {"OPENAI_API_KEY": "do-not-inherit", "HF_TOKEN": "do-not-inherit", "PATH": "bin"},
        )
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertEqual(env["INSPECTION_DEMO_RESEARCH_DEVICE"], "cpu")
        self.assertEqual(env["INSPECTION_DEMO_ASSET_MOUNT"], str(self.root / "assets"))
        self.assertEqual(env["INSPECTION_DEMO_DATA_DIR"], str(self.root / "state" / "data"))
        self.assertIn(str(self.root / "vendor" / "tog-ifc-ecore" / "src"), env["PYTHONPATH"])

    def test_missing_state_status_is_stopped(self):
        self.assertEqual(self.launcher.read_status(self.root)["status"], "stopped")

    def test_stop_signals_only_launcher_owned_worker_not_arbitrary_pid(self):
        state = self.root / "state"
        state.mkdir()
        (state / "status.json").write_text(json.dumps({"pid": os.getpid(), "status": "running"}))
        with self.assertRaisesRegex(self.launcher.LauncherError, "owned"):
            self.launcher.request_stop(state)
        self.assertFalse((state / "stop.json").exists())

    def test_busy_port_is_rejected_without_killing_listener(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(self.launcher.LauncherError, "in use"):
                self.launcher.require_free_port(port)
            self.assertGreater(listener.fileno(), -1)


if __name__ == "__main__":
    unittest.main()
