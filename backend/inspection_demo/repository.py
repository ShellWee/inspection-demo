from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, Text, delete, event
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .models import RunEvent


class RunRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    kind: str = Field(index=True)
    owner: str = Field(index=True)
    status: str = Field(index=True)
    created_at: datetime
    updated_at: datetime
    request_json: str = Field(sa_column=Column(Text))
    result_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    events_json: str = Field(default="[]", sa_column=Column(Text))


class RunStore:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def configure_sqlite(connection: Any, _: Any) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        SQLModel.metadata.create_all(self.engine)
        self._lock = threading.RLock()

    def create(self, *, kind: str, owner: str, request: dict[str, Any]) -> RunRecord:
        now = datetime.now(UTC)
        record = RunRecord(
            id=uuid4().hex,
            kind=kind,
            owner=owner,
            status="queued",
            created_at=now,
            updated_at=now,
            request_json=json.dumps(request, separators=(",", ":"), default=str),
            events_json="[]",
        )
        with self._lock, Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def get(self, record_id: str) -> RunRecord | None:
        with self._lock, Session(self.engine) as session:
            record = session.exec(select(RunRecord).where(RunRecord.id == record_id)).first()
            if record is not None:
                session.expunge(record)
            return record

    def set_status(self, record_id: str, status: str) -> None:
        self._mutate(record_id, status=status)

    def set_created_at(self, record_id: str, created_at: datetime) -> None:
        self._mutate(record_id, created_at=created_at)

    def delete_older_than(self, cutoff: datetime) -> int:
        with self._lock, Session(self.engine) as session:
            result = session.exec(delete(RunRecord).where(RunRecord.created_at < cutoff))
            session.commit()
            return int(result.rowcount or 0)

    def set_result(self, record_id: str, *, status: str, result: dict[str, Any]) -> None:
        self._mutate(
            record_id,
            status=status,
            result_json=json.dumps(result, separators=(",", ":"), default=str),
        )

    def append_event(self, record_id: str, event: RunEvent) -> None:
        with self._lock, Session(self.engine) as session:
            record = session.get(RunRecord, record_id)
            if record is None:
                raise KeyError(record_id)
            events = json.loads(record.events_json)
            events.append(event.model_dump(mode="json"))
            record.events_json = json.dumps(events, separators=(",", ":"))
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()

    def events(self, record_id: str) -> list[RunEvent]:
        record = self.get(record_id)
        if record is None:
            raise KeyError(record_id)
        return [RunEvent.model_validate(item) for item in json.loads(record.events_json)]

    def request(self, record: RunRecord) -> dict[str, Any]:
        return json.loads(record.request_json)

    def result(self, record: RunRecord) -> dict[str, Any] | None:
        return json.loads(record.result_json) if record.result_json else None

    def _mutate(self, record_id: str, **updates: Any) -> None:
        with self._lock, Session(self.engine) as session:
            record = session.get(RunRecord, record_id)
            if record is None:
                raise KeyError(record_id)
            for key, value in updates.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()
