"""Database schema & helpers (SQLAlchemy Core). SQLite for local dev, PostgreSQL(+Timescale) in production.

Predictions and their outcomes are append-only: enforced by DB triggers, not just application code.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable

from sqlalchemy import (
    BigInteger, Boolean, Column, Float, Index, Integer, LargeBinary, MetaData, String, Table, Text, UniqueConstraint,
    create_engine, event, text,
)
from sqlalchemy.engine import Engine

from .config import settings

metadata = MetaData()

symbols = Table(
    "symbols", metadata,
    Column("symbol", String(32), primary_key=True),
    Column("display_name", String(64), nullable=False),
    Column("family", String(8), nullable=False),          # boom | crash
    Column("n", Integer, nullable=False),                 # 300/500/...
    Column("direction", Integer, nullable=False),         # +1 boom (up spikes), -1 crash (down spikes)
    Column("pip", Float),
    Column("market", String(64)),
    Column("submarket", String(64)),
    Column("available", Boolean, nullable=False, default=True),
    Column("exchange_is_open", Boolean),
    Column("discovered_at", Float),
    Column("last_seen_at", Float),
)

ticks = Table(
    "ticks", metadata,
    Column("symbol", String(32), primary_key=True),
    Column("epoch", BigInteger, primary_key=True),
    Column("quote", Float, nullable=False),
)

candles_native = Table(
    "candles_native", metadata,
    Column("symbol", String(32), primary_key=True),
    Column("granularity", Integer, primary_key=True),
    Column("epoch", BigInteger, primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float), Column("close", Float),
)

backfill_state = Table(
    "backfill_state", metadata,
    Column("symbol", String(32), primary_key=True),
    Column("oldest_epoch", BigInteger),
    Column("newest_epoch", BigInteger),
    Column("ticks_complete", Boolean, default=False),
    Column("candles_done", Text, default="{}"),
    Column("updated_at", Float),
)

data_gaps = Table(
    "data_gaps", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(32), nullable=False),
    Column("start_epoch", BigInteger, nullable=False),
    Column("end_epoch", BigInteger, nullable=False),
    Column("seconds", Float),
    Column("detected_at", Float),
    Column("filled", Boolean, default=False),
    Column("source", String(32)),
    UniqueConstraint("symbol", "start_epoch", "end_epoch", name="uq_gap"),
)

spikes = Table(
    "spikes", metadata,
    Column("symbol", String(32), primary_key=True),
    Column("epoch", BigInteger, primary_key=True),
    Column("price_before", Float), Column("price_after", Float),
    Column("ret", Float),                     # log return of the spike tick
    Column("magnitude_sigma", Float),         # |ret| / robust sigma
    Column("abs_move", Float),
    Column("direction", Integer),
    Column("size_class", String(8)),          # small | medium | large
    Column("ticks_since_prev", Integer),
    Column("secs_since_prev", Float),
    Column("threshold", Float),
    Column("detected_at", Float),
)

jobs = Table(
    "jobs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("type", String(48), nullable=False),
    Column("payload", Text, default="{}"),
    Column("status", String(16), nullable=False, default="queued"),  # queued|running|done|failed
    Column("priority", Integer, default=5),
    Column("run_after", Float, default=0),
    Column("attempts", Integer, default=0),
    Column("max_attempts", Integer, default=3),
    Column("lease_until", Float),
    Column("last_error", Text),
    Column("result", Text),
    Column("dedupe_key", String(128)),
    Column("created_at", Float), Column("started_at", Float), Column("finished_at", Float),
    Index("ix_jobs_status", "status", "run_after"),
)

schedule_state = Table(
    "schedule_state", metadata,
    Column("name", String(96), primary_key=True),
    Column("last_enqueued_at", Float),
)

models = Table(
    "models", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(32), nullable=False),
    Column("horizon", String(8), nullable=False),
    Column("model_name", String(48), nullable=False),
    Column("role", String(16), nullable=False),     # champion|challenger|retired|rejected
    Column("approved", Boolean, default=False),
    Column("created_at", Float),
    Column("path", Text),
    Column("metrics", Text),       # json: walk-forward, validation, test, baselines
    Column("thresholds", Text),    # json: level thresholds derived from validation
    Column("feature_cols", Text),
    Column("train_range", Text),
    Column("notes", Text),
    Index("ix_models_sym", "symbol", "horizon", "role"),
)

# Serialized model artifacts live in the DB so API and worker processes (possibly on different hosts) share them
model_artifacts = Table(
    "model_artifacts", metadata,
    Column("model_id", Integer, primary_key=True),
    Column("data", LargeBinary, nullable=False),
    Column("created_at", Float),
)

experiments = Table(
    "experiments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("timeframe", String(8)),
    Column("family", String(48)),
    Column("params", Text),
    Column("horizon", String(8)),
    Column("split", String(16)),
    Column("n_signals", Integer), Column("tp", Integer), Column("fp", Integer),
    Column("precision", Float), Column("recall", Float), Column("f1", Float),
    Column("lift", Float), Column("p_value", Float), Column("q_value", Float), Column("base_rate", Float),
    Column("trading", Text),
    Column("is_combo", Boolean, default=False),
    Column("selected", Boolean, default=False),
    Column("created_at", Float),
    Index("ix_exp_run", "run_id", "symbol"),
)

backtests = Table(
    "backtests", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("horizon", String(8)),
    Column("kind", String(16)),        # model|baseline
    Column("name", String(48)),
    Column("split", String(24)),       # in_sample|walk_forward|validation|test|forward
    Column("metrics", Text),
    Column("created_at", Float),
    Index("ix_bt_sym", "symbol", "horizon"),
)

research = Table(
    "research", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(32), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("payload", Text),
    Column("created_at", Float),
    Index("ix_research", "symbol", "kind"),
)

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Float, nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("horizon", String(8), nullable=False),
    Column("horizon_kind", String(4), nullable=False),
    Column("horizon_size", Integer, nullable=False),
    Column("issued_epoch", BigInteger, nullable=False),     # epoch of last tick observed at issue time
    Column("expires_epoch", BigInteger),                    # for time horizons
    Column("direction", Integer, nullable=False),
    Column("probability", Float, nullable=False),
    Column("base_rate", Float),
    Column("level", String(8), nullable=False),
    Column("model_id", Integer),
    Column("model_name", String(48)),
    Column("role", String(16), nullable=False),             # champion|shadow|baseline
    Column("approved", Boolean, default=False),
    Column("reasons", Text),
    Column("details", Text),
    Index("ix_pred_sym", "symbol", "horizon", "issued_epoch"),
)

prediction_outcomes = Table(
    "prediction_outcomes", metadata,
    Column("prediction_id", Integer, primary_key=True),
    Column("evaluated_at", Float, nullable=False),
    Column("status", String(16), nullable=False),       # hit|miss|invalid
    Column("outcome", Integer),                          # 1/0, null when invalid
    Column("spike_epoch", BigInteger),
    Column("lead_time_sec", Float),
    Column("note", Text),
)

alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Float),
    Column("symbol", String(32)),
    Column("horizon", String(8)),
    Column("level", String(8)),
    Column("prediction_id", Integer),
    Column("dedupe_key", String(128), unique=True),
    Column("message", Text),
    Column("telegram_sent", Boolean, default=False),
    Column("telegram_error", Text),
)

system_events = Table(
    "system_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float),
    Column("level", String(8)),
    Column("component", String(32)),
    Column("message", Text),
)

_IMMUTABLE_TABLES = ("predictions", "prediction_outcomes")


def _install_triggers(engine: Engine) -> None:
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            for t in _IMMUTABLE_TABLES:
                conn.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS {t}_no_update BEFORE UPDATE ON {t} "
                    f"BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;")
                conn.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS {t}_no_delete BEFORE DELETE ON {t} "
                    f"BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;")
        elif engine.dialect.name == "postgresql":
            conn.exec_driver_sql(
                "CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS trigger AS $$ "
                "BEGIN RAISE EXCEPTION '% is append-only', TG_TABLE_NAME; END; $$ LANGUAGE plpgsql;")
            for t in _IMMUTABLE_TABLES:
                conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS {t}_immutable ON {t};")
                conn.exec_driver_sql(
                    f"CREATE TRIGGER {t}_immutable BEFORE UPDATE OR DELETE ON {t} "
                    f"FOR EACH ROW EXECUTE FUNCTION forbid_mutation();")
                conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS {t}_no_truncate ON {t};")
                conn.exec_driver_sql(
                    f"CREATE TRIGGER {t}_no_truncate BEFORE TRUNCATE ON {t} "
                    f"FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();")
            if os.environ.get("USE_TIMESCALE", "false").lower() == "true":
                try:
                    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                except Exception:  # noqa: BLE001 - optional
                    pass


_engine: Engine | None = None


def normalize_url(url: str) -> str:
    """postgres:// and postgresql:// (Render/Heroku style) -> psycopg3 driver."""
    for pre in ("postgres://", "postgresql://"):
        if url.startswith(pre):
            return "postgresql+psycopg://" + url[len(pre):]
    return url


def make_engine(url: str | None = None) -> Engine:
    url = normalize_url(url or settings.database_url)
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("sqlite"):
        path = url.replace("sqlite:///", "")
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 10
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _pragma(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()
    metadata.create_all(eng)
    _install_triggers(eng)
    return eng


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def set_engine(engine: Engine) -> None:
    global _engine
    _engine = engine


def insert_ignore(engine: Engine, table: Table, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(table).on_conflict_do_nothing()
    else:
        from sqlalchemy.dialects.sqlite import insert as sl_insert
        stmt = sl_insert(table).on_conflict_do_nothing()
    n = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), 2000):
            res = conn.execute(stmt, rows[i:i + 2000])
            n += max(res.rowcount or 0, 0)
    return n


def upsert(engine: Engine, table: Table, row: dict, keys: list[str]) -> None:
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as ins
    else:
        from sqlalchemy.dialects.sqlite import insert as ins
    stmt = ins(table).values(**row)
    upd = {k: v for k, v in row.items() if k not in keys}
    stmt = stmt.on_conflict_do_update(index_elements=keys, set_=upd) if upd else stmt.on_conflict_do_nothing()
    with engine.begin() as conn:
        conn.execute(stmt)


def log_event(level: str, component: str, message: str, engine: Engine | None = None) -> None:
    eng = engine or get_engine()
    try:
        with eng.begin() as conn:
            conn.execute(system_events.insert().values(ts=time.time(), level=level, component=component,
                                                       message=message[:4000]))
    except Exception:  # noqa: BLE001 - logging must never crash the caller
        pass


def jdump(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, allow_nan=False)


def _json_default(o: Any):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if f != f or f in (float("inf"), float("-inf")) else f
    if isinstance(o, np.ndarray):
        return [_json_default(x) if isinstance(x, (np.generic,)) else x for x in o.tolist()]
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def clean_json(obj: Any) -> Any:
    """Replace NaN/inf with None recursively so payloads are strict JSON."""
    import math
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def jloads(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return default


def exec_sql(sql: str, params: dict | None = None, engine: Engine | None = None):
    eng = engine or get_engine()
    with eng.connect() as conn:
        return conn.execute(text(sql), params or {}).mappings().all()
