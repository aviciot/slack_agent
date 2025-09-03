# src/persistence/telemetry_store.py
# Low-level telemetry writers (talk to SQLite only).
# Import THIS module from src/telemetry.py, not directly from app code.

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from src.persistence.db import get_conn


def _conn():
    return get_conn()


# ---------- small helpers: schema safety (idempotent) ----------
def _ensure_column(table: str, column: str, type_decl: str = "TEXT"):
    """
    Add a column if it does not exist. Safe to call often.
    """
    with _conn() as cx:
        cur = cx.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if column not in cols:
            cx.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")


def _ensure_table_qcache_events():
    with _conn() as cx:
        cx.execute(
            """
            CREATE TABLE IF NOT EXISTS qcache_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                event TEXT,         -- e.g., "hit", "miss", "evict", "accept", "reject"
                qid TEXT,           -- q:<sha256> or template key if known
                route TEXT,         -- "bank", "catalog_llm", "reroute_after_error"
                question TEXT,
                created_at TIMESTAMP
            )
            """
        )


# ---------- qcache (bank) event ----------
def log_qcache_event(request_id: str, event: str, qid: str | None, route: str, question: str):
    """
    Lightweight bank/qcache telemetry line. Separate table to avoid overloading routing_decisions.
    """
    _ensure_table_qcache_events()
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO qcache_events
                (request_id, event, qid, route, question, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (request_id, event, qid, route, question, datetime.utcnow()),
        )


# ---------- top-level request ----------
def log_request(
    request_id: str,
    source: str,                      # 'api' | 'slack' | 'scheduler'
    text: Optional[str] = None,
    *,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    route: Optional[str] = None,      # 'bank' | 'llm_local' | 'llm_cloudflare'
    status: str = "ok",               # 'ok' | 'error' | 'partial'
    timings: Optional[Dict[str, Any]] = None,
):
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO requests
                (request_id, source, user_id, user_name, channel_id, thread_ts,
                 text, route, status, timings_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, source, user_id, user_name, channel_id, thread_ts,
                text, route, status, json.dumps(timings or {}), datetime.utcnow(),
            ),
        )


# ---------- retrieval (KNN) ----------
def log_retrieval(
    request_id: str,
    *,
    k: int,
    query_text: str,
    hits: List[Dict[str, Any]],       # [{"id":"...","score":0.12}, ...]
    chosen_id: Optional[str] = None,
    chosen_score: Optional[float] = None,
):
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO retrieval
                (request_id, k, query_text, hits_json, chosen_id, chosen_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (request_id, k, query_text, json.dumps(hits), chosen_id, chosen_score, datetime.utcnow()),
        )


# ---------- llm calls ----------
def log_llm_request(
    request_id: str,
    attempt: int,
    provider: str,
    model: str,
    prompt_version: str | None = None,
    prompt_hash: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
    extra_json: dict | None = None,   # NEW: store arbitrary structured details
) -> None:
    """
    Insert an LLM request telemetry row into llm_requests.
    Ensures 'extra_json' column exists (TEXT).
    """
    _ensure_column("llm_requests", "extra_json", "TEXT")
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO llm_requests
                (request_id, attempt, provider, model,
                 prompt_version, prompt_hash,
                 tokens_in, tokens_out, latency_ms, error, extra_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                attempt,
                provider,
                model,
                prompt_version,
                prompt_hash,
                tokens_in,
                tokens_out,
                latency_ms,
                error,
                json.dumps(extra_json) if extra_json else None,
                datetime.utcnow(),
            ),
        )


# ---------- sql run ----------
def log_sql_run(
    request_id: str,
    *,
    sql_before: Optional[str],
    sql_after: Optional[str],
    safety_flags: Optional[List[str]],
    executed: int = 1,               # 0/1
    duration_ms: int = 0,
    rowcount: int = 0,
    error: Optional[str] = None,
):
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO sql_runs
                (request_id, sql_before, sql_after, safety_flags,
                 executed, duration_ms, rowcount, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, sql_before, sql_after,
                json.dumps(safety_flags or []),
                executed, duration_ms, rowcount, error, datetime.utcnow(),
            ),
        )


# ---------- slack delivery ----------
def log_slack_interaction(
    request_id: str,
    *,
    channel_id: Optional[str],
    message_ts: Optional[str],
    response_bytes: Optional[int],
    rows_shown: Optional[int],
    truncated: int = 0,               # 0/1
):
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO slack_interactions
                (request_id, channel_id, message_ts, response_bytes,
                 rows_shown, truncated, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, channel_id, message_ts, response_bytes,
                rows_shown, truncated, datetime.utcnow(),
            ),
        )


# ---------- user feedback ----------
def log_feedback(
    request_id: str,
    *,
    user_id: Optional[str],
    rating: str,                      # 'up' | 'down' | '1'..'5'
    comment: Optional[str] = None,
):
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO feedback
                (request_id, user_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (request_id, user_id, rating, comment, datetime.utcnow()),
        )


# ---------- routing decisions (NEW fields for 2-stage router) ----------
def log_routing_decision(
    request_id: str,
    *,
    source: Optional[str],                 # 'bank' | 'retrieval' | 'catalog' | etc.
    k: Optional[int],
    candidates: Optional[List[Dict[str, Any]]],  # [{"db":"statements","score":2.9}, ...]
    selected: Optional[List[str]],         # e.g., ["statements"] or ["statements","billing"]
    chosen_rank: Optional[int],
    stage1_shortlist: Optional[List[Dict[str, Any]]] = None,   # [{"db":"statements","score":2.9}, {"db":"billing","score":2.3}]
    stage2_bank: Optional[Dict[str, Any]] = None,              # {"db":"statements","similarity":0.83,"threshold":0.72,"template_key":"q:abcd..."}
    reason: Optional[str] = None                                  # "bank_hit" | "catalog_llm" | "reroute_after_error"
):
    """
    Persists router telemetry, including Stage-1 shortlist and Stage-2 bank details.
    Ensures the new columns exist before insert.
    """
    # Ensure extended columns exist (idempotent)
    _ensure_column("routing_decisions", "stage1_shortlist_json", "TEXT")
    _ensure_column("routing_decisions", "stage2_bank_json", "TEXT")
    _ensure_column("routing_decisions", "chosen_reason", "TEXT")

    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO routing_decisions
                (request_id, source, k, candidates, selected, chosen_rank,
                 stage1_shortlist_json, stage2_bank_json, chosen_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                source,
                k,
                json.dumps(candidates or []),
                json.dumps(selected or []),
                chosen_rank,
                json.dumps(stage1_shortlist or []),
                json.dumps(stage2_bank or {}),
                reason or "",
                datetime.utcnow(),
            ),
        )


def log_error_decision(
    request_id: str,
    *,
    db_name: str,
    vendor: str,
    action: str,
    category: str,
    reason: str,
    source: str,
    signature: str,
    confidence: float,
):
    """
    Persist one error-handling decision.
    """
    with _conn() as cx:
        # ensure table exists
        cx.execute(
            """
            CREATE TABLE IF NOT EXISTS error_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                db_name TEXT,
                vendor TEXT,
                action TEXT,
                category TEXT,
                reason TEXT,
                source TEXT,
                signature TEXT,
                confidence REAL,
                created_at TIMESTAMP
            )
            """
        )
        cx.execute(
            """
            INSERT INTO error_decisions
                (request_id, db_name, vendor, action, category, reason, source, signature, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (request_id, db_name, vendor, action, category, reason, source, signature, confidence, datetime.utcnow()),
        )
