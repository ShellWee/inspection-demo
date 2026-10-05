from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation JSON has no result rows: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _key(row: dict[str, Any]) -> str:
    question_id = row.get("question_id")
    return f"id:{question_id}" if question_id is not None else f"question:{row.get('question', '')}"


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    correct = sum(row.get("classification") == "correct" for row in materialized)
    wrong = sum(row.get("classification") == "wrong" for row in materialized)
    abstained = sum(row.get("classification") == "abstained" for row in materialized)
    errors = sum(
        row.get("classification") == "error" or row.get("status") == "error"
        for row in materialized
    )
    return {
        "n": len(materialized),
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "errors": errors,
        "correct_rate": correct / len(materialized) if materialized else 0.0,
        "accuracy": correct / (correct + wrong) if correct + wrong else 0.0,
        "abstention_rate": abstained / len(materialized) if materialized else 0.0,
        "error_rate": errors / len(materialized) if materialized else 0.0,
    }


def compare_evaluations(paths: Iterable[str | Path]) -> dict[str, Any]:
    path_list = [Path(path) for path in paths]
    indexed = [{_key(row): row for row in _rows(path)} for path in path_list]
    intersection = set(indexed[0]) if indexed else set()
    for rows in indexed[1:]:
        intersection.intersection_update(rows)
    keys = sorted(intersection)
    systems: dict[str, Any] = {}
    for path, rows in zip(path_list, indexed):
        fair_rows = [rows[key] for key in keys]
        by_category = {
            str(category): _metrics(
                row for row in fair_rows if row.get("category") == category
            )
            for category in sorted(
                {row.get("category") for row in fair_rows if row.get("category") is not None}
            )
        }
        systems[str(path)] = {"overall": _metrics(fair_rows), "by_category": by_category}
    return {"fair_intersection_n": len(keys), "question_keys": keys, "systems": systems}
