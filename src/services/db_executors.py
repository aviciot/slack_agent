# src/services/db_executors.py
from __future__ import annotations

import os
import json
import sqlite3
import logging
import time  # [ERROR_DECIDER] backoff
from typing import Any, Dict, List, Optional, Tuple
from src import telemetry


# [ERROR_DECIDER] service import
try:
    from src.services.error_decider.handler import ErrorHandlingService  # type: ignore
except Exception:  # fallback relative import if module path differs
    from .error_decider.handler import ErrorHandlingService  # type: ignore

logger = logging.getLogger(__name__)

# Optional Oracle deps: lazy import
_ORACLE_IMPORTED = False
_oracledb = None

# [ERROR_DECIDER] single shared service (reads env/databases.json/redis)
_ERROR_SVC = ErrorHandlingService()


class BaseExecutor:
    """Minimal interface all executors must implement."""
    name: str = "base"
    db_name: str = ""  # [ERROR_DECIDER] filled by registry

    def execute(self, sql: str, *, row_cap: int = 200) -> List[Tuple]:
        raise NotImplementedError




# ------------------------- SQLite Executor -------------------------
class SQLiteExecutor(BaseExecutor):
    name = "sqlite"

    def __init__(self, db_path: str, timeout: float = 12.0) -> None:
        self.db_path = db_path
        self.timeout = timeout
        # If SQLITE_CHECK_SAME_THREAD=1, we DISABLE sqlite's same-thread check (allow cross-thread)
        self._disable_same_thread = os.getenv("SQLITE_CHECK_SAME_THREAD", "0") == "1"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=not self._disable_same_thread,  # True by default, False if env=1
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

    # NOTE: added request_id kw-only param
    def execute(self, sql: str, *, row_cap: int = 200, request_id: str | None = None) -> List[Tuple]:
        stmt = (sql or "").strip().rstrip(";")
        logger.info(f"[EXEC][sqlite] running (cap={row_cap}) path={self.db_path}")

        attempt = 0
        while True:
            try:
                with self._connect() as conn:
                    cur = conn.execute(stmt)
                    rows = cur.fetchmany(row_cap) if row_cap else cur.fetchall()
                    return [tuple(r) for r in rows]
            except Exception as e:
                # [ERROR_DECIDER] ask the decider what to do
                res = _ERROR_SVC.handle(
                    db_vendor=self.name,
                    db_name=getattr(self, "db_name", "") or "informatica",
                    sql=stmt,
                    error=e,
                    retry_count=attempt,
                    exec_phase="execute",
                )
                dec = res["decision"]

                # Telemetry for error decision (guarded by env flag inside telemetry)
                try:
                    telemetry.error_decision(
                        request_id=request_id or "unknown",
                        db_name=self.db_name or "unknown",
                        vendor=self.name,
                        decision=dec,
                    )
                except Exception:
                    pass

                rty = res["retry"]
                logger.warning(
                    "[EXEC][sqlite][decider] db=%s sig=%s action=%s category=%s source=%s reason=%s retry_allowed=%s",
                    getattr(self, "db_name", "unknown"),
                    dec.get("error_signature"), dec.get("action"), dec.get("category"),
                    dec.get("source"), dec.get("reason"), rty.get("allowed"),
                )

                if dec.get("action") == "RETRY" and rty.get("allowed"):
                    attempt = rty.get("next_retry_count", attempt + 1)
                    # simple capped exponential backoff
                    delay = min(0.5 * (2 ** max(0, attempt - 1)), 3.0)
                    time.sleep(delay)
                    continue

                # No retry: raise with structured context (preserve original as __cause__)
                raise RuntimeError(
                    f"[ERROR_DECIDER] db={getattr(self,'db_name','unknown')} vendor={self.name} "
                    f"action={dec.get('action')} category={dec.get('category')} "
                    f"reason={dec.get('reason')} source={dec.get('source')} "
                    f"signature={dec.get('error_signature')}"
                ) from e

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

        attempt = 0
        while True:
            conn = None
            try:
                conn = self._get_connection()
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

            except Exception as e:
                # ensure connection is closed before deciding/retrying
                try:
                    if conn:
                        if self._pool:
                            self._pool.release(conn)
                        else:
                            conn.close()
                except Exception:
                    pass

                # [ERROR_DECIDER] ask the decider
                res = _ERROR_SVC.handle(
                    db_vendor=self.name,
                    db_name=getattr(self, "db_name", "") or "oracle",
                    sql=stmt,
                    error=e,
                    retry_count=attempt,
                    exec_phase="execute",
                )
                dec = res["decision"]
                telemetry.error_decision(
                    request_id=getattr(self, "request_id", "unknown"),  # or pass from agent context
                    db_name=self.db_name,
                    vendor=self.name,
                    decision=dec,
                )                
                rty = res["retry"]
                
                logger.warning(
                    "[EXEC][oracle][decider] db=%s sig=%s action=%s category=%s source=%s reason=%s retry_allowed=%s",
                    getattr(self, "db_name", "unknown"),
                    dec.get("error_signature"), dec.get("action"), dec.get("category"),
                    dec.get("source"), dec.get("reason"), rty.get("allowed"),
                )

                if dec.get("action") == "RETRY" and rty.get("allowed"):
                    attempt = rty.get("next_retry_count", attempt + 1)
                    delay = min(0.5 * (2 ** max(0, attempt - 1)), 3.0)
                    time.sleep(delay)
                    continue

                # No retry: raise with structured context
                raise RuntimeError(
                    f"[ERROR_DECIDER] db={getattr(self,'db_name','unknown')} vendor={self.name} "
                    f"action={dec.get('action')} category={dec.get('category')} "
                    f"reason={dec.get('reason')} source={dec.get('source')} "
                    f"signature={dec.get('error_signature')}"
                ) from e

            finally:
                # close on success path
                if conn:
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
    Build an executor registry keyed by *DB name* (not just 'sqlite'/'oracle').

    Preferred: load /app/config/databases.json (or env DATABASES_JSON_PATH / DATABASES_CONFIG_PATH):
      [
        {"name":"informatica","type":"sqlite","db_path":"/app/data/db/informatica_insigts_agent.sqlite"},
        {"name":"statements","type":"oracle","user":"...","password":"...","dsn":"host:port/svc"},
        {"name":"billing","type":"oracle","user":"...","password":"...","dsn":"host:port/svc"}
      ]

    Fallback: if file not present, build:
      - "informatica" from SQLITE_DB_PATH/DB_PATH
      - single "oracle" from ORACLE_* envs (last resort)

    If a validator is provided, we wrap each executor so it validates SQL before execution.
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

    # --- Load config/databases.json when available ---
    cfg_path = (
        os.getenv("DATABASES_JSON_PATH")
        or os.getenv("DATABASES_CONFIG_PATH")
        or "/app/config/databases.json"
    )
    dbs: List[Dict[str, Any]] = []
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                dbs = json.load(f) or []
        except Exception as e:
            logger.warning(f"[EXEC][registry] cannot read {cfg_path}: {e}")

    # Register one executor per DB entry (preferred path)
    for db in dbs:
        try:
            name = str(db.get("name", "")).lower()
            typ = str(db.get("type", "")).lower()
            if not name or not typ:
                logger.warning(f"[EXEC][registry] skipping malformed entry: {db}")
                continue

            if typ == "sqlite":
                path = (
                    db.get("db_path")
                    or os.getenv("SQLITE_DB_PATH")
                    or os.getenv("DB_PATH")
                    or "/app/data/db/informatica_insigts_agent.sqlite"
                )
                exec_obj = SQLiteExecutor(path)
                exec_obj.db_name = name  # [ERROR_DECIDER]
                exec_obj = wrap(exec_obj)
                registry[name] = exec_obj
                logger.info(f"[EXEC][registry] {name} -> sqlite:{path}")

            elif typ == "oracle":
                _ensure_oracle_imported()
                if not _ORACLE_IMPORTED:
                    logger.warning("[EXEC][registry] skipping oracle db '%s' (oracledb not installed)", name)
                    continue
                dsn = db["dsn"]
                user = db["user"]
                pwd = db["password"]
                arraysize = int(db.get("arraysize", 1000))
                exec_obj = OracleExecutor(dsn=dsn, user=user, password=pwd, arraysize=arraysize)
                exec_obj.db_name = name  # [ERROR_DECIDER]
                exec_obj = wrap(exec_obj)
                registry[name] = exec_obj
                # avoid logging password
                logger.info(f"[EXEC][registry] {name} -> oracle:{dsn} (user={user})")

            else:
                logger.warning(f"[EXEC][registry] unknown type='{typ}' for '{name}', skipping")

        except Exception as e:
            logger.warning(f"[EXEC][registry] failed for entry {db}: {e}")

    # Fallbacks if no config worked
    if not registry:
        # sqlite fallback (as 'informatica')
        sqlite_path = (
            os.getenv("SQLITE_DB_PATH")
            or os.getenv("DB_PATH")
            or "/app/data/db/informatica_insigts_agent.sqlite"
        )
        try:
            exec_obj = SQLiteExecutor(sqlite_path)
            exec_obj.db_name = "informatica"  # [ERROR_DECIDER]
            exec_obj = wrap(exec_obj)
            registry["informatica"] = exec_obj
            logger.info(f"[EXEC][registry] informatica -> sqlite:{sqlite_path}")
        except Exception as e:
            logger.warning(f"[EXEC][registry] failed to init sqlite executor: {e}")

        # single generic "oracle" as last resort
        dsn = os.getenv("ORACLE_DSN")
        user = os.getenv("ORACLE_USER")
        pwd = os.getenv("ORACLE_PASSWORD")
        if dsn and user and pwd:
            try:
                _ensure_oracle_imported()
                if _ORACLE_IMPORTED:
                    exec_obj = OracleExecutor(dsn=dsn, user=user, password=pwd)
                    exec_obj.db_name = "oracle"  # [ERROR_DECIDER]
                    exec_obj = wrap(exec_obj)
                    registry["oracle"] = exec_obj
                    logger.info(f"[EXEC][registry] oracle -> {dsn} (user={user})")
                else:
                    logger.warning("[EXEC][registry] oracledb not installed; cannot init oracle fallback")
            except Exception as e:
                logger.warning(f"[EXEC][registry] failed to init oracle executor: {e}")
        else:
            logger.info("[EXEC][registry] oracle not configured (set ORACLE_* or use databases.json)")

    return registry
