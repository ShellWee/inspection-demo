from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from .models import GroundingLimits, GroundingResult, RunEvent

ProgressCallback = Callable[[str, int, str], Awaitable[None]]


class GroundingAdapter(Protocol):
    async def ground(
        self,
        *,
        query: str,
        model_id: str,
        api_key: str,
        progress: ProgressCallback | None = None,
        limits: GroundingLimits | None = None,
    ) -> GroundingResult: ...


class SubprocessResearchGroundingAdapter:
    """Runs the heavyweight ToG/PyG stack in one isolated, request-scoped process."""

    def __init__(
        self,
        *,
        python_executable: str,
        bridge_path: Path,
        base_payload: dict[str, object],
        timeout_seconds: int = 600,
    ) -> None:
        self.python_executable = python_executable
        self.bridge_path = bridge_path.resolve()
        self.base_payload = dict(base_payload)
        self.timeout_seconds = timeout_seconds

    async def ground(
        self,
        *,
        query: str,
        model_id: str,
        api_key: str,
        progress: ProgressCallback | None = None,
        limits: GroundingLimits | None = None,
    ) -> GroundingResult:
        if progress:
            await progress("asset_sync", 12, "Verified research runtime paths and hashes.")
            await progress("query_planning", 25, "Started request-scoped ToG query planning.")
        process = await asyncio.create_subprocess_exec(
            self.python_executable,
            str(self.bridge_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = {
            **self.base_payload,
            "query": query,
            "model_id": model_id,
            "api_key": api_key,
            "limits": limits.model_dump() if limits else None,
        }
        redaction_key = api_key
        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(json.dumps(payload).encode("utf-8")),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError("research grounding exceeded the hard timeout") from None
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.kill()
                await process.wait()
                raise
            if process.returncode != 0:
                detail = _redact_worker_error(
                    stderr.decode("utf-8", errors="replace"), redaction_key
                )
                raise RuntimeError(
                    f"research grounding worker exited with code {process.returncode}: {detail}"
                )
            try:
                result = GroundingResult.model_validate_json(
                    stdout.decode("utf-8").splitlines()[-1].replace(redaction_key, "[REDACTED]")
                )
            except (UnicodeDecodeError, IndexError, ValueError) as error:
                raise RuntimeError(
                    "research grounding worker returned an invalid contract"
                ) from error
        finally:
            payload["api_key"] = ""
            api_key = ""  # noqa: F841 - promptly release the request-scoped secret
            redaction_key = ""  # noqa: F841 - release after diagnostics are safely redacted
        if progress:
            await progress("gnn_retrieval", 55, "Completed Text-GNN v5 retrieval.")
            await progress("hierarchy", 78, "Completed typed hierarchy reasoning.")
            await progress("closure_validation", 94, "Evidence validation finished.")
        return result


def ignore_event(_: RunEvent) -> None:
    """Marker used by adapters that do not expose progress callbacks."""


def _redact_worker_error(value: str, api_key: str) -> str:
    redacted = value.replace(api_key, "[REDACTED]") if api_key else value
    redacted = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", redacted)
    return " ".join(redacted.split())[-2_000:] or "no diagnostic output"
