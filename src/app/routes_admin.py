from fastapi import APIRouter, HTTPException, Request ,Query
import os
import sqlite3   # <-- add this
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from src.persistence.db import get_conn
from src.persistence.init_db import ensure_db


router = APIRouter()

# Simple protection using shared admin token
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme123")

def check_admin(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin token")

@router.post("/initdb")
def init_db(token: str):
    """
    Re-run DB schema init manually.
    Example:
        curl -X POST "http://localhost:8090/admin/initdb?token=changeme123"
    """
    check_admin(token)
    ensure_db()
    return {"ok": True, "message": "DB init executed"}

@router.get("/showtables")
def show_tables(token: str):
    """
    List all tables currently in the SQLite DB.
    Example:
        curl "http://localhost:8090/admin/showtables?token=changeme123"
    """
    check_admin(token)

    db_path = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()

    return {"db": db_path, "tables": tables}


@router.get("/showcounts")
def show_counts(token: str):
    """
    Show row counts for each table in the SQLite DB.
    Example:
        curl "http://localhost:8090/admin/showcounts?token=changeme123"
    """
    check_admin(token)

    db_path = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")
    conn = sqlite3.connect(db_path)

    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    counts = {}
    for t in tables:
        try:
            cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
            counts[t] = cur.fetchone()[0]
        except Exception as e:
            counts[t] = f"error: {e}"

    conn.close()
    return {"db": db_path, "counts": counts}



LOG_TABLES = {
    "requests": "requests",
    "sql_runs": "sql_runs",
    "llm_requests": "llm_requests",
    "retrieval": "retrieval",
    "slack_interactions": "slack_interactions",
    "feedback": "feedback",
}

def _cutoff_str(before_iso: Optional[str], before_days: Optional[int]) -> Optional[str]:
    if before_iso:
        dt = datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if before_days is not None:  # allow 0
        dt = datetime.utcnow() - timedelta(days=before_days)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return None


@router.post("/telemetry/purge")
def purge_telemetry(
    token: str,
    tables: Optional[List[str]] = Query(
        default=None,
        description="Repeat ?tables=...; default = all telemetry tables"
    ),
    before_iso: Optional[str] = Query(
        default=None,
        description="UTC ISO timestamp; purge rows older than this (e.g. 2025-08-01T00:00:00Z)"
    ),
    before_days: Optional[int] = Query(
        default=None,
        ge=0,
        description="Purge rows older than N days (0 allowed = now)"
    ),
    dry_run: bool = Query(
        default=False,
        description="If true, only count rows that would be deleted"
    ),
):
    check_admin(token)

    use_tables = tables or list(LOG_TABLES.values())
    unknown = [t for t in use_tables if t not in LOG_TABLES.values()]
    if unknown:
        raise HTTPException(400, f"unknown tables: {unknown}")

    cutoff = _cutoff_str(before_iso, before_days)
    results: List[Dict] = []

    with get_conn() as cx:
        for t in use_tables:
            if dry_run:
                q = f"SELECT COUNT(*) FROM {t}" + (" WHERE created_at < ?" if cutoff else "")
                args = (cutoff,) if cutoff else ()
                cnt = cx.execute(q, args).fetchone()[0]
                results.append({"table": t, "would_delete": int(cnt)})
            else:
                q = f"DELETE FROM {t}" + (" WHERE created_at < ?" if cutoff else "")
                args = (cutoff,) if cutoff else ()
                cur = cx.execute(q, args)
                results.append({"table": t, "deleted": cur.rowcount})

    return {"ok": True, "cutoff": cutoff, "dry_run": dry_run, "results": results}


@router.get("/metrics")
def telemetry_metrics(
    token: str,
    limit: int = Query(10, ge=1, le=100, description="How many recent rows to show per table"),
):
    """
    Show recent telemetry snapshots.
    Example:
        curl.exe "http://localhost:8090/admin/metrics?token=changeme&limit=5"
    """
    check_admin(token)

    db_path = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")
    results = {}

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row

        # Last requests
        results["requests"] = [
            dict(r)
            for r in cx.execute("SELECT * FROM requests ORDER BY created_at DESC LIMIT ?", (limit,))
        ]

        # Last SQL runs
        results["sql_runs"] = [
            dict(r)
            for r in cx.execute("SELECT * FROM sql_runs ORDER BY created_at DESC LIMIT ?", (limit,))
        ]

        # Last LLM calls
        results["llm_requests"] = [
            dict(r)
            for r in cx.execute("SELECT * FROM llm_requests ORDER BY created_at DESC LIMIT ?", (limit,))
        ]

        # Error counts per table

        error_counts = {}
        for t in ["requests", "sql_runs", "llm_requests"]:
            try:
                # Check if table has an "error" column
                cols = [r[1] for r in cx.execute(f"PRAGMA table_info({t})").fetchall()]
                if "error" in cols:
                    cur = cx.execute(f"SELECT COUNT(*) FROM {t} WHERE error IS NOT NULL AND error <> ''")
                    error_counts[t] = cur.fetchone()[0]
                else:
                    error_counts[t] = 0
            except Exception as e:
                error_counts[t] = f"error: {e}"
        results["error_counts"] = error_counts

        # Feedback stats (if table exists)
        try:
            tables = [r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            if "feedback" in tables:
                stats = {}
                cur = cx.execute("SELECT rating, COUNT(*) as cnt FROM feedback GROUP BY rating")
                for r in cur.fetchall():
                    stats[r[0]] = r[1]
                results["feedback"] = stats
        except Exception as e:
            results["feedback"] = {"error": str(e)}

    return {"ok": True, "limit": limit, "metrics": results}
