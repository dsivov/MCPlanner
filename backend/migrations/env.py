"""Alembic environment.

We:
  - Pull `sqlalchemy.url` from the app's settings (so .env is the single source of truth)
  - Reuse the app's Base.metadata so autogenerate sees the live SQLAlchemy models
  - Hook a sqlite-vec extension load on every connection so the vec0 virtual-table
    bits in migrations can run (even though we DON'T autogenerate them — see below)

Note on the `vec_precedents` virtual table: alembic's autogenerate can't see virtual
tables created with CREATE VIRTUAL TABLE. It's safer to (a) exclude it from autogenerate
and (b) keep its creation idempotent at app startup. See app/db.py init_db().
"""

from __future__ import annotations
import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool, event
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Make the app importable when running `alembic` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings              # noqa: E402  — needs path tweak above
from app.db import Base                       # noqa: E402  — imports all models
import sqlite_vec                             # noqa: E402

config = context.config

# Force alembic to use the app's runtime DATABASE_URL.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# Tables alembic should NOT touch (managed by app code at startup):
#   - the vec_precedents virtual table itself (created via CREATE VIRTUAL TABLE … USING vec0)
#   - the internal shadow tables sqlite-vec creates underneath it (vec_precedents_chunks,
#     vec_precedents_rowids, vec_precedents_info, vec_precedents_vector_chunks00, …)
# All are recreated idempotently in app.db.init_db.
def _include_object(object, name, type_, reflected, compare_to):  # noqa: A002
    if type_ == "table" and (name or "").startswith("vec_precedents"):
        return False
    return True


def _resolve_real_sqlite_conn(dbapi_connection):
    """Reach through aiosqlite to the underlying sqlite3.Connection. Same idea as app/db.py."""
    import sqlite3
    paths = (
        lambda c: c,
        lambda c: getattr(c, "_connection", None),
        lambda c: getattr(getattr(c, "_connection", None), "_conn", None),
        lambda c: getattr(getattr(c, "driver_connection", None), "_conn", None),
    )
    for path in paths:
        try:
            cand = path(dbapi_connection)
        except Exception:
            cand = None
        if isinstance(cand, sqlite3.Connection):
            return cand
    return None


def _load_sqlite_vec(dbapi_connection, _record):
    real = _resolve_real_sqlite_conn(dbapi_connection)
    if real is None:
        return
    try:
        real.enable_load_extension(True)
        sqlite_vec.load(real)
        real.enable_load_extension(False)
    except Exception:
        pass


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,        # SQLite needs batch mode for ALTER TABLE
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Load sqlite-vec on every new connection from this engine.
    event.listen(connectable.sync_engine, "connect", _load_sqlite_vec)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
