import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from inspection_demo.repository import RunStore


def test_store_enables_wal_and_removes_expired_runs(tmp_path: Path) -> None:
    """Catches non-WAL persistence and replay records surviving their retention window."""
    database = tmp_path / "runs.sqlite3"
    store = RunStore(database)
    old = store.create(kind="grounding", owner="owner", request={"query": "old"})
    recent = store.create(kind="grounding", owner="owner", request={"query": "recent"})
    store.set_created_at(old.id, datetime.now(UTC) - timedelta(days=31))

    removed = store.delete_older_than(datetime.now(UTC) - timedelta(days=30))
    with sqlite3.connect(database) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode.lower() == "wal"
    assert removed == 1
    assert store.get(old.id) is None
    assert store.get(recent.id) is not None
