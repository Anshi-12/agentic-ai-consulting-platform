"""Small persistence layer owned by Group 3.

PostgreSQL is used when DATABASE_URL is configured. Local demo mode defaults to
SQLite so the platform remains runnable without an external database.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from sqlalchemy import JSON, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .contracts import AuditEvent, Project, User

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./group3_runtime.db")
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
if DATABASE_URL.startswith("sqlite:///"):
    Path(DATABASE_URL.removeprefix("sqlite:///" )).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "group3_users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class ProjectRecord(Base):
    __tablename__ = "group3_projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class AuditRecord(Base):
    __tablename__ = "group3_audit_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


Base.metadata.create_all(engine)

ModelT = TypeVar("ModelT")


class PersistentDict(Generic[ModelT]):
    def __init__(self, record_type, model_type):
        self.record_type = record_type
        self.model_type = model_type

    def get(self, key, default=None):
        with SessionLocal() as session:
            record = session.get(self.record_type, str(key))
            return self.model_type.model_validate(record.payload) if record else default

    def __setitem__(self, key, value):
        payload = value.model_dump(mode="json")
        with SessionLocal.begin() as session:
            record = session.get(self.record_type, str(key))
            if record:
                record.payload = payload
            else:
                session.add(self.record_type(id=str(key), payload=payload))

    def values(self) -> list[ModelT]:
        with SessionLocal() as session:
            records = session.scalars(select(self.record_type)).all()
            return [self.model_type.model_validate(record.payload) for record in records]


class PersistentAuditLog:
    def append(self, event: AuditEvent) -> None:
        with SessionLocal.begin() as session:
            session.add(AuditRecord(id=str(event.id), payload=event.model_dump(mode="json")))

    def __iter__(self) -> Iterator[AuditEvent]:
        with SessionLocal() as session:
            records = session.scalars(select(AuditRecord)).all()
            yield from (AuditEvent.model_validate(record.payload) for record in records)


users = PersistentDict(UserRecord, User)
projects = PersistentDict(ProjectRecord, Project)
audits = PersistentAuditLog()
