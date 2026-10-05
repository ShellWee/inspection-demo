"""A dependency-free, loopback-only client for the launch notebook (Python 3.10+)."""

from __future__ import annotations

import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NotebookClientError(RuntimeError):
    """A diagnostic safe to display without echoing HTTP bodies or credentials."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _local_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and (parsed.port is None or 0 < parsed.port < 65536)
        )
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise NotebookClientError("Use an HTTP loopback URL without credentials or a path.")
    return value.rstrip("/")


def _redact(value, secret: str):
    if isinstance(value, str):
        return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value.replace(secret, "[REDACTED]"))
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    return value


def run_grounding(
    *,
    base_url: str,
    query: str,
    model_id: str,
    api_key: str,
    timeout: float = 660,
    poll_interval: float = 1,
    request_timeout: float = 30,
) -> dict:
    """Submit once and poll the existing API; return a redacted terminal run.

    Keys travel only in the initial loopback POST body and are never written to
    disk, environment variables, command arguments, or output by this helper.
    A client timeout does not cancel a run already accepted by the backend.
    """
    base_url = _local_url(base_url)
    if not isinstance(query, str) or len(query.strip()) < 3 or len(query) > 2000:
        raise NotebookClientError("Query must contain 3–2000 characters and nonblank text.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise NotebookClientError("An OpenAI API key is required.")
    if model_id not in {"gpt-4.1", "gpt-5", "gpt-5.6-luna"}:
        raise NotebookClientError("Select a supported model: gpt-4.1, gpt-5, or gpt-5.6-luna.")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in (timeout, poll_interval, request_timeout)
    ):
        raise NotebookClientError("Timeouts and polling interval must be positive finite numbers.")

    opener = build_opener(ProxyHandler({}), _NoRedirect())
    deadline = time.monotonic() + timeout
    payload = {"api_key": api_key, "model_id": model_id, "query": query}

    def request_json(path: str, data=None) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NotebookClientError("Polling timed out; the accepted run may still be running.")
        request = Request(
            base_url + path,
            data=json.dumps(data).encode("utf-8") if data is not None else None,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            with opener.open(request, timeout=min(request_timeout, remaining)) as response:
                result = json.load(response)
        except HTTPError as error:
            code = error.code
            error.close()
            raise NotebookClientError(f"Local demo returned HTTP {code}.") from None
        except (URLError, OSError, ValueError):
            raise NotebookClientError(
                "Local demo request failed or returned invalid JSON; check launcher status."
            ) from None
        finally:
            request.data = None
        if not isinstance(result, dict):
            raise NotebookClientError("Local demo returned an invalid response object.")
        return result

    try:
        created = request_json("/api/v1/grounding-runs", payload)
        payload["api_key"] = ""
        run_id = created.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
            raise NotebookClientError("Local demo returned an invalid run identifier.")
        while True:
            run = request_json(f"/api/v1/grounding-runs/{run_id}")
            status = run.get("status")
            if status in {"completed", "abstained", "failed", "cancelled"}:
                return _redact(run, api_key)
            if status not in {"queued", "running"}:
                raise NotebookClientError("Local demo returned an unknown run status.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NotebookClientError(
                    "Polling timed out; the accepted run may still be running."
                )
            time.sleep(min(poll_interval, remaining))
    finally:
        payload["api_key"] = ""
        api_key = ""
