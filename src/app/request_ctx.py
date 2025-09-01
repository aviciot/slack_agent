# src/app/request_ctx.py
import time
import uuid
from typing import Dict, Optional
from contextvars import ContextVar

REQUEST_ID: ContextVar[str] = ContextVar("REQUEST_ID", default="")
ROUTE:      ContextVar[Optional[str]] = ContextVar("ROUTE", default=None)   # 'bank'|'llm_local'|'llm_cloudflare'
STATUS:     ContextVar[str] = ContextVar("STATUS", default="ok")            # 'ok'|'error'|'partial'
TIMINGS:    ContextVar[Dict[str, float]] = ContextVar("TIMINGS", default={})

def new_request_id() -> str:
    rid = uuid.uuid4().hex[:16]
    REQUEST_ID.set(rid)
    TIMINGS.set({})
    STATUS.set("ok")
    ROUTE.set(None)
    return rid

def get_request_id() -> str:
    rid = REQUEST_ID.get()
    return rid or new_request_id()

def set_route(route: Optional[str]) -> None:
    ROUTE.set(route)

def set_status(status: str) -> None:
    STATUS.set(status)

def add_timing(key: str, start_ns: int, end_ns: Optional[int] = None) -> None:
    d = TIMINGS.get().copy()
    if end_ns is None:
        end_ns = time.time_ns()
    d[key] = round((end_ns - start_ns) / 1_000_000, 3)  # ms
    TIMINGS.set(d)

def get_route() -> Optional[str]:
    return ROUTE.get()

def get_status() -> str:
    return STATUS.get()

def get_timings() -> Dict[str, float]:
    return TIMINGS.get()
