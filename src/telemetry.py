"""
High-level telemetry wrappers used by the app code.
These forward to src.persistence.telemetry_store with light adaptation so
call sites stay simple and stable.

Do NOT import sqlite or touch DBs here; defer to telemetry_store.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
from datetime import datetime

from src.persistence import telemetry_store as S

# ─── Env toggles ──────────────────────────────────────────────
LOG_REQUESTS = os.getenv("LOG_REQUESTS", "true").lower() == "true"
LOG_SQL_RUNS = os.getenv("LOG_SQL_RUNS", "true").lower() == "true"
LOG_LLM_REQUESTS = os.getenv("LOG_LLM_REQUESTS", "true").lower() == "true"
LOG_RETRIEVAL = os.getenv("LOG_RETRIEVAL", "true").lower() == "true"
LOG_SLACK_INTERACTIONS = os.getenv("LOG_SLACK_INTERACTIONS", "true").lower() == "true"
LOG_FEEDBACK = os.getenv("LOG_FEEDBACK", "true").lower() == "true"
LOG_ROUTING_DECISIONS = os.getenv("LOG_ROUTING_DECISIONS", "true").lower() == "true"
LOG_QCACHE_EVENTS = os.getenv("LOG_QCACHE_EVENTS", "true").lower() == "true"
LOG_ERROR_DECISIONS = os.getenv("LOG_ERROR_DECISIONS", "true").lower() == "true"
LOG_LLM_PROMPT = os.getenv("LOG_LLM_PROMPT", "false").lower() == "true"  # kept for consistency


# ------------- HTTP / Top-level request -------------
def http_request(
    *,
    request_id: str,
    source: str,                 # 'api' | 'slack' | 'scheduler' | 'debug'
    path: str,
    text: Optional[str] = None,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    route: Optional[str] = None,
    status: str = "ok",
    timings: Optional[Dict[str, Any]] = None,
) -> None:
    if not LOG_REQUESTS:
        return
    S.log_request(
        request_id=request_id,
        source=source,
        text=text,
        user_id=user_id,
        user_name=user_name,
        channel_id=channel_id,
        thread_ts=thread_ts,
        route=route,
        status=status,
        timings=timings or {},
    )


# ------------- Retrieval / Routing summaries -------------
def retrieval(
    *,
    request_id: str,
    k: int,
    candidates: List[str] | List[Dict[str, Any]],
    chosen: List[str] | None = None,
    chosen_rank: int | None = None,
    source: str | None = None,
) -> None:
    if not LOG_RETRIEVAL:
        return
    hits: List[Dict[str, Any]] = []
    for c in candidates or []:
        if isinstance(c, dict):
            cid = c.get("id") or c.get("db") or c.get("table") or c.get("key") or str(c)
            score = c.get("score")
        else:
            cid, score = str(c), None
        hits.append({"id": cid, "score": score})

    chosen_id = (chosen or [None])[0]
    S.log_retrieval(
        request_id=request_id,
        k=k,
        query_text=(source or "")[:500],
        hits=hits,
        chosen_id=chosen_id,
        chosen_score=None,
    )


# ------------- LLM calls -------------
def llm_request(
    *,
    request_id: str,
    attempt: int,
    provider: str | None,
    model: str | None,
    prompt_version: str | None = None,
    prompt_hash: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
    extra_json: dict | None = None,
) -> None:
    if not LOG_LLM_REQUESTS:
        return
    S.log_llm_request(
        request_id=request_id,
        attempt=attempt,
        provider=provider or "",
        model=model or "",
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        error=error,
        extra_json=extra_json or {},
    )


# ------------- SQL run results -------------
def sql_run(
    *,
    request_id: str,
    sql_before: Optional[str],
    sql_after: Optional[str],
    safety_flags: Optional[List[str]],
    executed: int = 1,
    duration_ms: int | None = 0,
    rowcount: int = 0,
    error: Optional[str] = None,
) -> None:
    if not LOG_SQL_RUNS:
        return
    S.log_sql_run(
        request_id=request_id,
        sql_before=sql_before,
        sql_after=sql_after,
        safety_flags=safety_flags or [],
        executed=int(bool(executed)),
        duration_ms=int(duration_ms or 0),
        rowcount=rowcount,
        error=error,
    )


# ------------- Slack delivery -------------
def slack_interaction(
    *,
    request_id: str,
    channel_id: Optional[str],
    message_ts: Optional[str],
    response_bytes: Optional[int],
    rows_shown: Optional[int],
    truncated: int = 0,
) -> None:
    if not LOG_SLACK_INTERACTIONS:
        return
    S.log_slack_interaction(
        request_id=request_id,
        channel_id=channel_id,
        message_ts=message_ts,
        response_bytes=response_bytes,
        rows_shown=rows_shown,
        truncated=int(bool(truncated)),
    )


# ------------- User feedback -------------
def feedback(
    *,
    request_id: str,
    user_id: Optional[str],
    rating: str,
    comment: Optional[str] = None,
) -> None:
    if not LOG_FEEDBACK:
        return
    S.log_feedback(
        request_id=request_id,
        user_id=user_id,
        rating=rating,
        comment=comment,
    )


# ------------- Routing decisions (extended 2-stage) -------------
def routing_decision(
    *,
    request_id: str,
    source: Optional[str],
    k: Optional[int],
    candidates: Optional[List[Dict[str, Any]]],
    selected: Optional[List[str]],
    chosen_rank: Optional[int],
    stage1_shortlist: Optional[List[Dict[str, Any]]] = None,
    stage2_bank: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> None:
    if not LOG_ROUTING_DECISIONS:
        return
    S.log_routing_decision(
        request_id=request_id,
        source=source,
        k=k,
        candidates=candidates,
        selected=selected,
        chosen_rank=chosen_rank,
        stage1_shortlist=stage1_shortlist,
        stage2_bank=stage2_bank,
        reason=reason,
    )


# ------------- Bank / qcache lightweight events -------------
def qcache_event(
    *,
    request_id: str,
    event: str,
    qid: str | None,
    route: str,
    question: str,
) -> None:
    if not LOG_QCACHE_EVENTS:
        return
    S.log_qcache_event(
        request_id=request_id,
        event=event,
        qid=qid,
        route=route,
        question=question,
    )


# ------------- Error decisions (NEW) -------------
def error_decision(
    *,
    request_id: str,
    db_name: str,
    vendor: str,
    decision: Dict[str, Any],
) -> None:
    if not LOG_ERROR_DECISIONS:
        return
    S.log_error_decision(
        request_id=request_id,
        db_name=db_name,
        vendor=vendor,
        action=decision.get("action", ""),
        category=decision.get("category", ""),
        reason=decision.get("reason", ""),
        source=decision.get("source", ""),
        signature=decision.get("error_signature", ""),
        confidence=float(decision.get("confidence", 0.0)),
    )
