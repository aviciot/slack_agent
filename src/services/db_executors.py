# src/services/db_executors.py
from __future__ import annotations

import os, sqlite3, logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional Oracle deps: lazy import
_ORACLE_IMPORTED = False
_oracledb = None

class BaseExecutor:
    """Minimal interface all executors must implement."""
    name: str = "base"
    def execute(self, sql: str, *, row_cap: int = 200) -> List[Tuple]:
        raise NotImplementedError

# ------------------------- SQLite Executor -------------------------
class SQLiteExecutor(BaseExecutor):
    name = "sqlite"

    def __init__(self, db_path: str, timeout: float = 12.0) -> None:
        self.db_path = db_path
        self.timeout = timeout
        self._check_same_thread = os.getenv("SQLITE_CHECK_SAME_THREAD", "0") == "1"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=not self._check_same_thread and True,
        )
        # Lightweight, safe pragmas for read-mostly workloads
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
        except Exception:
            pass
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, *, row_cap: int = 200) -> List[Tuple]:
        stmt = (sql or "").strip().rstrip(";")
        logger.info(f"[EXEC][sqlite] running (cap={row_cap}) path={self.db_path}")
        with self._connect() as conn:
            cur = conn.execute(stmt)
            rows = cur.fetchmany(row_cap) if row_cap else cur.fetchall()
            return [tuple(r) for r in rows]

# ------------------------- Oracle Executor -------------------------
def _ensure_oracle_imported() -> None:
    global _ORACLE_IMPORTED, _oracledb
    if _ORACLE_IMPORTED:
        return
    try:
        import oracledb  # type: ignore
        _oracledb = oracledb
        _ORACLE_IMPORTED = True
    except Exception as e:
        logger.warning(f"[EXEC][oracle] oracledb import failed: {e}")
        _ORACLE_IMPORTED = False
        _oracledb = None

class OracleExecutor(BaseExecutor):
    name = "oracle"

    def __init__(self, dsn: str, user: str, password: str, arraysize: int = 1000) -> None:
        _ensure_oracle_imported()
        if not _ORACLE_IMPORTED:
            raise RuntimeError("oracledb is not available. Install `oracledb` to use OracleExecutor.")
        self.dsn, self.user, self.password = dsn, user, password
        self.default_arraysize = max(1, arraysize)

        # Pool config (optional)
        pool_min = int(os.getenv("ORACLE_POOL_MIN", "1"))
        pool_max = int(os.getenv("ORACLE_POOL_MAX", "4"))
        pool_inc = int(os.getenv("ORACLE_POOL_INC", "1"))
        self._pool = None
        try:
            self._pool = _oracledb.create_pool(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=pool_min,
                max=pool_max,
                increment=pool_inc,
                homogeneous=True,
            )
            logger.info(f"[EXEC][oracle] pool created dsn={self.dsn} min={pool_min} max={pool_max}")
        except Exception as e:
            logger.warning(f"[EXEC][oracle] pool create failed, will connect per-call: {e}")

    def _get_connection(self):
        if self._pool:
            return self._pool.acquire()
        return _oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)

    @staticmethod
    def _coerce_row(row):
        # Convert LOBs to str in thin mode if needed
        out = []
        for v in row:
            try:
                if hasattr(v, "read"):
                    out.append(v.read())
                else:
                    out.append(v)
            except Exception:
                out.append(v)
        return tuple(out)

    def execute(self, sql: str, *, row_cap: int = 200) -> List[Tuple]:
        stmt = (sql or "").strip().rstrip(";")
        logger.info(f"[EXEC][oracle] running (cap={row_cap}) dsn={self.dsn}")
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # Tune fetch sizes from cap when provided
            arraysize = max(row_cap, 1) if row_cap else self.default_arraysize
            cur.arraysize = arraysize
            try:
                cur.prefetchrows = arraysize
            except Exception:
                pass
            cur.execute(stmt)
            rows = cur.fetchmany(row_cap) if row_cap else cur.fetchall()
            return [self._coerce_row(r) for r in rows]
        finally:
            try:
                if self._pool:
                    self._pool.release(conn)
                else:
                    conn.close()
            except Exception:
                pass

# ------------------------- Registry Builder -------------------------
def make_registry_from_env(*, validator: Any | None = None) -> Dict[str, BaseExecutor]:
    """
    Build an executor registry keyed by logical route names used by Agent:
      - "sqlite"  → SQLiteExecutor (local cache/analytics DB)
      - "oracle"  → OracleExecutor (statements/billing/etc.)

    ENV:
      SQLITE_DB_PATH=/app/data/db/informatica_insigts_agent.sqlite
      ORACLE_DSN=host:port/service
      ORACLE_USER=username
      ORACLE_PASSWORD=secret
      ORACLE_POOL_MIN=1  ORACLE_POOL_MAX=4  ORACLE_POOL_INC=1
    """
    registry: Dict[str, BaseExecutor] = {}

    # Validator hook wrapper
    def wrap(exec_obj: BaseExecutor) -> BaseExecutor:
        if not validator:
            return exec_obj
        _call = getattr(validator, "check", None) or getattr(validator, "validate", None) or validator
        if not callable(_call):
            return exec_obj

        class Guarded(exec_obj.__class__):  # type: ignore[misc]
            def execute(self, sql: str, *, row_cap: int = 200) -> List[Tuple]:
                _call(sql)  # raise on violation
                return super().execute(sql, row_cap=row_cap)
        g = Guarded.__new__(Guarded)  # type: ignore
        g.__dict__ = exec_obj.__dict__.copy()
        return g  # type: ignore

    # SQLite (always present in dev)
    sqlite_path = os.getenv("SQLITE_DB_PATH") or os.getenv("DB_PATH") or "/app/data/db/informatica_insigts_agent.sqlite"
    try:
        registry["sqlite"] = wrap(SQLiteExecutor(sqlite_path))
        logger.info(f"[EXEC][registry] sqlite -> {sqlite_path}")
    except Exception as e:
        logger.warning(f"[EXEC][registry] failed to init sqlite executor: {e}")

    # Oracle (optional)
    dsn = os.getenv("ORACLE_DSN")
    user = os.getenv("ORACLE_USER")
    pwd  = os.getenv("ORACLE_PASSWORD")
    if dsn and user and pwd:
        try:
            registry["oracle"] = wrap(OracleExecutor(dsn=dsn, user=user, password=pwd))
            logger.info(f"[EXEC][registry] oracle -> {dsn} (user={user})")
        except Exception as e:
            logger.warning(f"[EXEC][registry] failed to init oracle executor: {e}")
    else:
        logger.info("[EXEC][registry] oracle not configured (set ORACLE_DSN/ORACLE_USER/ORACLE_PASSWORD)")

    return registry
