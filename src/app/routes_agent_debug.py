# src/app/routes_agent_debug.py
from __future__ import annotations

import os
import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/debug/agent",   tags=["Agent Debug"]
)

# DB path (override via env if needed)
DB_PATH = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")



# ---------- Pydantic models returned in Swagger ----------

class AgentEvent(BaseModel):
    ts: Optional[str] = Field(None, description="Event timestamp (ISO8601 if available)")
    kind: str = Field(..., description="Event source/table (request, retrieval, llm_request, sql_run, slack, qcache)")
    data: Dict[str, Any] = Field(default_factory=dict)


class RoutingInfo(BaseModel):
    candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Raw candidates (router-specific)")
    selected: List[str] = Field(default_factory=list, description="Selected / attempted routes")
    chosen_rank: Optional[int] = None
    # ROUTE-DB-FIRST: Stage 1 = DB shortlist, Stage 2 = Query Bank evaluation
    stage1_shortlist: List[Dict[str, Any]] = Field(default_factory=list, description="Catalog-first shortlist (db, score)")
    stage2_bank: Dict[str, Any] = Field(default_factory=dict, description="Bank result (db, similarity, threshold, template_key, signature)")
    chosen_reason: Optional[str] = None  # 'bank_hit' | 'catalog_llm' | 'reroute_after_error' | etc.
    source: Optional[str] = None
    k: Optional[int] = None
    created_at: Optional[str] = None


class AgentActivity(BaseModel):
    request_id: str
    started_at: Optional[str] = None
    last_update: Optional[str] = None
    routing: Optional[RoutingInfo] = None
    events: List[AgentEvent] = Field(default_factory=list)


# ---------- helpers ----------

def _conn():
    try:
        cx = sqlite3.connect(DB_PATH, timeout=5)
        cx.row_factory = sqlite3.Row
        return cx
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot open DB at {DB_PATH}: {e}")


def _fetch_one(cx: sqlite3.Connection, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    cur = cx.execute(query, params)
    row = cur.fetchone()
    cur.close()
    return row


def _fetch_all(cx: sqlite3.Connection, query: str, params: tuple = ()) -> List[sqlite3.Row]:
    cur = cx.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _json_load_safe(s: Any, default):
    try:
        if s is None:
            return default
        if isinstance(s, (dict, list)):
            return s
        return json.loads(s)
    except Exception:
        return default


def _row_time_iso(row: Optional[sqlite3.Row], col: str) -> Optional[str]:
    if not row:
        return None
    v = row[col]
    if v is None:
        return None
    try:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)
    except Exception:
        return str(v)


def _collect_routing(cx: sqlite3.Connection, request_id: str) -> Optional[RoutingInfo]:
    """
    Reads the most recent routing_decisions row for the request, including Stage-1/Stage-2 fields.
    To reflect ROUTE-DB-FIRST correctly, your orchestrator should insert this BEFORE retrieval/LLM/SQL logs.
    """
    row = _fetch_one(
        cx,
        """
        SELECT
          source, k, candidates, selected, chosen_rank,
          stage1_shortlist_json, stage2_bank_json, chosen_reason, created_at
        FROM routing_decisions
        WHERE request_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (request_id,),
    )
    if not row:
        return None

    candidates = _json_load_safe(row["candidates"], [])
    selected = _json_load_safe(row["selected"], [])
    stage1 = _json_load_safe(row["stage1_shortlist_json"], [])
    stage2 = _json_load_safe(row["stage2_bank_json"], {})

    return RoutingInfo(
        candidates=candidates,
        selected=selected,
        chosen_rank=row["chosen_rank"],
        stage1_shortlist=stage1,
        stage2_bank=stage2,
        chosen_reason=row["chosen_reason"],
        source=row["source"],
        k=row["k"],
        created_at=_row_time_iso(row, "created_at"),
    )


def _collect_events(cx: sqlite3.Connection, request_id: str) -> List[AgentEvent]:
    """
    Pulls a concise timeline from multiple tables and sorts by ts.
    If you log Stage-1/Stage-2 into routing_decisions BEFORE other steps, the final timeline will reflect ROUTE-DB-FIRST.
    """
    events: List[AgentEvent] = []

    # requests
    for r in _fetch_all(cx, "SELECT * FROM requests WHERE request_id=? ORDER BY created_at ASC", (request_id,)):
        events.append(AgentEvent(
            ts=_row_time_iso(r, "created_at"),
            kind="request",
            data={
                "source": r["source"],
                "route": r["route"],
                "status": r["status"],
                "text": r["text"],
                "timings": _json_load_safe(r["timings_json"], {}),
            },
        ))

    # retrieval
    for r in _fetch_all(cx, "SELECT * FROM retrieval WHERE request_id=? ORDER BY created_at ASC", (request_id,)):
        events.append(AgentEvent(
            ts=_row_time_iso(r, "created_at"),
            kind="retrieval",
            data={
                "k": r["k"],
                "query_text": r["query_text"],
                "hits": _json_load_safe(r["hits_json"], []),
                "chosen_id": r["chosen_id"],
                "chosen_score": r["chosen_score"],
            },
        ))

    # llm_requests
    for r in _fetch_all(cx, "SELECT * FROM llm_requests WHERE request_id=? ORDER BY rowid ASC", (request_id,)):
        events.append(AgentEvent(
            ts=_row_time_iso(r, "created_at") or _row_time_iso(r, "ts"),
            kind="llm_request",
            data={
                "attempt": r["attempt"],
                "provider": r["provider"],
                "model": r["model"],
                "prompt_version": r["prompt_version"],
                "prompt_hash": r["prompt_hash"],
                "tokens_in": r["tokens_in"],
                "tokens_out": r["tokens_out"],
                "latency_ms": r["latency_ms"],
                "error": r["error"],
                "extra": _json_load_safe(r["extra_json"], {}),
            },
        ))

    # sql_runs
    for r in _fetch_all(cx, "SELECT * FROM sql_runs WHERE request_id=? ORDER BY created_at ASC", (request_id,)):
        events.append(AgentEvent(
            ts=_row_time_iso(r, "created_at"),
            kind="sql_run",
            data={
                "executed": r["executed"],
                "duration_ms": r["duration_ms"],
                "rowcount": r["rowcount"],
                "error": r["error"],
            },
        ))

    # slack_interactions
    for r in _fetch_all(cx, "SELECT * FROM slack_interactions WHERE request_id=? ORDER BY created_at ASC", (request_id,)):
        events.append(AgentEvent(
            ts=_row_time_iso(r, "created_at"),
            kind="slack",
            data={
                "channel_id": r["channel_id"],
                "message_ts": r["message_ts"],
                "response_bytes": r["response_bytes"],
                "rows_shown": r["rows_shown"],
                "truncated": r["truncated"],
            },
        ))

    # qcache_events (optional)
    try:
        for r in _fetch_all(cx, "SELECT * FROM qcache_events WHERE request_id=? ORDER BY created_at ASC", (request_id,)):
            events.append(AgentEvent(
                ts=_row_time_iso(r, "created_at"),
                kind="qcache",
                data={
                    "event": r["event"],
                    "qid": r["qid"],
                    "route": r["route"],
                    "question": r["question"],
                },
            ))
    except Exception:
        pass

    # Sort by timestamp (string ISO is fine if consistent)
    events.sort(key=lambda e: e.ts or "")
    return events


def _find_last_active_request_id(cx: sqlite3.Connection) -> Optional[str]:
    """
    Prefer the newest request that has real agent activity:
    - has llm_requests OR sql_runs OR routing_decisions
    - OR originated from Slack (/slack/...) or the manual E2E tester (/debug/agent)
    Excludes /healthz.
    """
    row = _fetch_one(cx, """
        SELECT r.request_id
        FROM requests r
        LEFT JOIN llm_requests       l ON l.request_id = r.request_id
        LEFT JOIN sql_runs           s ON s.request_id = r.request_id
        LEFT JOIN routing_decisions  d ON d.request_id = r.request_id
        WHERE r.path != '/healthz'
          AND (
                l.rowid IS NOT NULL
             OR s.rowid IS NOT NULL
             OR d.rowid IS NOT NULL
             OR r.path LIKE '/slack/%'
             OR r.path = '/debug/agent'
          )
        ORDER BY r.created_at DESC
        LIMIT 1
    """)
    if row:
        return row["request_id"]

    # fallback: newest non-healthz request
    row2 = _fetch_one(cx, """
        SELECT request_id
        FROM requests
        WHERE path != '/healthz'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    return row2["request_id"] if row2 else None



# ---------- endpoints ----------

@router.get(
    "/last",
    response_model=AgentActivity,
    summary="Get the most recent request's activity",
    description=(
        "Looks up the latest request_id in the telemetry DB and returns the same payload as /debug/agent/{request_id}. "
        "Useful for quick post-run inspection."
    ),
)
def get_last_agent_activity() -> AgentActivity:
    with _conn() as cx:
        rid = _find_last_active_request_id(cx)
        if not rid:
            raise HTTPException(status_code=404, detail="No requests logged yet")
    return get_agent_activity(rid)



@router.get(
    "/{request_id}",
    response_model=AgentActivity,
    summary="Get end-to-end activity for a specific request",
    description=(
        "Returns routing info (Stage-1 DB shortlist, Stage-2 bank result) and a time-ordered "
        "timeline of events (request → retrieval → LLM → SQL → Slack). "
        "Use this to verify ROUTE-DB-FIRST order and debug mistakes/latency."
    ),
    responses={
        200: {
            "description": "Agent activity found",
            "content": {
                "application/json": {
                    "examples": {
                        "sample": {
                            "summary": "Sample agent activity",
                            "value": {
                                "request_id": "05dbd22cf0d96709",
                                "started_at": "2025-08-29T13:12:09Z",
                                "last_update": "2025-08-29T13:12:10Z",
                                "routing": {
                                    "stage1_shortlist": [{"db": "statements", "score": 3}],
                                    "stage2_bank": {
                                        "db": "statements",
                                        "similarity": 0.98,
                                        "threshold": 0.9,
                                        "template_key": "q:...",
                                        "signature": "statements count by status in July"
                                    },
                                    "chosen_reason": "bank_hit",
                                    "selected": ["bank"],
                                    "chosen_rank": 0,
                                    "source": "retrieval",
                                    "k": 10,
                                    "created_at": "2025-08-29T13:12:09Z"
                                },
                                "events": [
                                    {"ts":"2025-08-29T13:12:09Z","kind":"request","data":{"source":"slack","route":"bank","status":"OK","text":"...","timings":{}}},
                                    {"ts":"2025-08-29T13:12:09Z","kind":"retrieval","data":{"k":10,"query_text":"@db_name:{statements} ...","hits":[{"id":"...","score":3.0}], "chosen_id":"...", "chosen_score":3.0}},
                                    {"ts":"2025-08-29T13:12:09Z","kind":"llm_request","data":{"attempt":1,"provider":"cloudflare","model":"...","latency_ms":420}},
                                    {"ts":"2025-08-29T13:12:10Z","kind":"sql_run","data":{"executed":"SELECT ...","duration_ms":120,"rowcount":18}},
                                    {"ts":"2025-08-29T13:12:10Z","kind":"slack","data":{"channel_id":"C123","rows_shown":18,"truncated":0}}
                                ]
                            }
                        }
                    }
                }
            }
        },
        404: {"description": "No activity for this request_id"},
        500: {"description": "DB error or schema mismatch"}
    },
)
def get_agent_activity(request_id: str) -> AgentActivity:
    with _conn() as cx:
        req_first = _fetch_one(cx, "SELECT created_at FROM requests WHERE request_id=? ORDER BY created_at ASC LIMIT 1", (request_id,))
        if not req_first:
            raise HTTPException(status_code=404, detail=f"No activity for request_id={request_id}")

        req_last  = _fetch_one(cx, "SELECT created_at FROM requests WHERE request_id=? ORDER BY created_at DESC LIMIT 1", (request_id,))

        routing = _collect_routing(cx, request_id)
        events = _collect_events(cx, request_id)

        return AgentActivity(
            request_id=request_id,
            started_at=_row_time_iso(req_first, "created_at"),
            last_update=_row_time_iso(req_last, "created_at"),
            routing=routing,
            events=events,
        )
