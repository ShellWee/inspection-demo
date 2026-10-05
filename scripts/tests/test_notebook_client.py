"""Stdlib tests: no installed application dependencies or paid API calls."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "scripts" / "notebook_client.py"
NOTEBOOK = ROOT / "notebooks" / "Launch_Demo.ipynb"
SECRET = "sk-test-not-a-real-secret-1234567890"


def run_fixture(status="completed"):
    return {
        "id": "run-123",
        "owner": "huggingface-private-user",
        "status": status,
        "created_at": "2026-10-05T12:00:00Z",
        "updated_at": "2026-10-05T12:00:01Z",
        "model_id": "gpt-4.1",
        "ifc_asset_id": "ecore-fixed",
        "gnn_runtime_id": "text-gnn-v5-seed43-epoch10",
        "query": "Inspect the doors",
        "result": {
            "status": status,
            "answer": "No certified targets in this fixture.",
            "agent_response": "",
            "response_status": "unavailable",
            "query_type": "planning",
            "query_type_reason": "Inspection request",
            "planning_score": {
                "value": None,
                "kind": "mean_retrieval_rank_score",
                "calibrated": False,
                "description": "Fixture score",
            },
            "closure_status": "abstain",
            "closure_stop_reason": "Fixture only",
            "tasks": [],
            "reasoning": [],
            "subgraph": None,
            "errors": [],
            "executable": False,
            "graph_hash": "fixture",
            "usage": {
                "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cached_input_tokens": 0, "embedding_input_tokens": 0,
            },
        },
        "events": [],
    }


@contextlib.contextmanager
def fake_server(replies):
    """Serve real HTTP and retain requests only in test memory."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append((self.command, self.path, json.loads(body) if body else None))
            response = replies[min(len(requests) - 1, len(replies) - 1)]
            code, payload, headers = response
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.wfile.write(encoded)

        do_GET = reply
        do_POST = reply

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CLIENT.is_file(), "Notebook HTTP client has not been implemented")
        spec = importlib.util.spec_from_file_location("notebook_client", CLIENT)
        self.client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.client)

    def call(self, base_url, **kwargs):
        options = {
            "base_url": base_url, "query": "Inspect the doors", "model_id": "gpt-4.1",
            "api_key": SECRET, "poll_interval": 0.001, "timeout": 2,
        }
        options.update(kwargs)
        return self.client.run_grounding(**options)

    def test_original_query_is_submitted_once_then_polled_until_complete(self):
        # A changed query or duplicate POST would change the user's paid request.
        running = run_fixture("running")
        running["result"] = None
        with fake_server([
            (202, {"run_id": "run-123", "status": "queued"}, {}),
            (200, running, {}),
            (200, run_fixture(), {}),
        ]) as (url, requests):
            result = self.call(url, query="  Inspect the doors.\n")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(requests, [
            ("POST", "/api/v1/grounding-runs", {
                "query": "  Inspect the doors.\n", "model_id": "gpt-4.1", "api_key": SECRET,
            }),
            ("GET", "/api/v1/grounding-runs/run-123", None),
            ("GET", "/api/v1/grounding-runs/run-123", None),
        ])

    def test_every_terminal_status_returns_without_extra_polling(self):
        for status in ("completed", "abstained", "failed", "cancelled"):
            with self.subTest(status=status), fake_server([
                (202, {"run_id": "run-123", "status": "queued"}, {}),
                (200, run_fixture(status), {}),
            ]) as (url, requests):
                self.assertEqual(self.call(url)["status"], status)
                self.assertEqual(len(requests), 2)

    def test_response_redacts_secret_in_nested_values_before_notebook_display(self):
        response = run_fixture("failed")
        response["result"]["errors"] = [f"Provider echoed {SECRET}", "sk-unrelated-secret-123456"]
        with fake_server([
            (202, {"run_id": "run-123", "status": "queued"}, {}),
            (200, response, {}),
        ]) as (url, _requests), contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.call(url)
        self.assertNotIn(SECRET, json.dumps(result) + output.getvalue())
        self.assertNotIn("sk-unrelated-secret", json.dumps(result))
        self.assertEqual(result["result"]["errors"][0], "Provider echoed [REDACTED]")

    def test_http_error_never_includes_echoed_request_credentials(self):
        with fake_server([(422, {"detail": {"input": {"api_key": SECRET}}}, {})]) as (url, _):
            with self.assertRaises(self.client.NotebookClientError) as raised:
                self.call(url)
        self.assertIn("422", str(raised.exception))
        self.assertNotIn(SECRET, str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)

    def test_redirect_is_not_followed_even_to_another_local_port(self):
        with fake_server([(200, run_fixture(), {})]) as (destination, received):
            with fake_server([(307, {}, {"Location": destination})]) as (url, _):
                with self.assertRaises(self.client.NotebookClientError):
                    self.call(url)
            self.assertEqual(received, [])

    def test_invalid_input_is_rejected_before_any_http_request(self):
        bad_inputs = [
            {"query": "  x "}, {"query": "x" * 2001}, {"api_key": " "},
            {"model_id": "unknown"}, {"timeout": 0}, {"poll_interval": -1},
            {"timeout": float("nan")}, {"request_timeout": 0},
        ]
        with fake_server([(500, {}, {})]) as (url, received):
            for bad in bad_inputs:
                with self.subTest(bad=bad), self.assertRaises(self.client.NotebookClientError):
                    self.call(url, **bad)
            self.assertEqual(received, [])

    def test_remote_or_credential_bearing_urls_are_rejected_before_connection(self):
        for url in (
            "https://example.com", "http://0.0.0.0:7860", "http://127.0.0.1.evil.test",
            "http://name:password@127.0.0.1", "http://127.0.0.1/path",
            "http://127.0.0.1?key=secret", "http://127.0.0.1#fragment",
        ):
            with self.subTest(url=url), self.assertRaises(self.client.NotebookClientError):
                self.call(url)

    def test_unknown_status_or_invalid_run_id_fails_without_unbounded_polling(self):
        for replies in (
            [(202, {"run_id": "../other", "status": "queued"}, {})],
            [(202, {"run_id": "run-123", "status": "queued"}, {}),
             (200, {"status": "mystery"}, {})],
            [(200, b"not json", {})],
        ):
            with self.subTest(replies=replies), fake_server(replies) as (url, _):
                with self.assertRaises(self.client.NotebookClientError):
                    self.call(url)

    def test_pending_run_has_bounded_wait(self):
        with fake_server([
            (202, {"run_id": "run-123", "status": "queued"}, {}),
            (200, run_fixture("running"), {}),
        ]) as (url, _):
            with self.assertRaises(self.client.NotebookClientError) as raised:
                self.call(url, timeout=0.02)
        self.assertIn("timed out", str(raised.exception).lower())


class NotebookTests(unittest.TestCase):
    def load_notebook(self):
        self.assertTrue(NOTEBOOK.is_file(), "Launch notebook has not been created")
        return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_notebook_code_compiles_without_saved_outputs_or_execution_state(self):
        book = self.load_notebook()
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        self.assertEqual(book["nbformat"], 4)

    def test_config_finds_checkout_from_root_notebooks_or_parent(self):
        book = self.load_notebook()
        config = next(cell for cell in book["cells"] if cell["id"] == "config")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkout = parent / "inspection-demo-hf"
            (checkout / "scripts").mkdir(parents=True)
            (checkout / "scripts" / "demo_launcher.py").touch()
            (checkout / "pyproject.toml").touch()
            (checkout / "notebooks").mkdir()
            try:
                for current in (checkout, checkout / "notebooks", parent):
                    with self.subTest(current=current), contextlib.redirect_stdout(io.StringIO()):
                        os.chdir(current)
                        namespace = {}
                        exec("".join(config["source"]), namespace)
                        self.assertEqual(namespace["REPO_ROOT"], checkout.resolve())
                        self.assertEqual(namespace["ASSETS_DIR"], checkout.resolve() / "assets")
                        self.assertEqual(namespace["DEVICE"], "cpu")
                        self.assertFalse(namespace["DOWNLOAD_ASSETS"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
