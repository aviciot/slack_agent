#src/app/llm_stub.py

"""
LLM Stub — drop‑in for Agent retrieval path

Purpose
-------
Temporary, zero‑dependency stand‑in for your real LLM client so you can
exercise the Agent end‑to‑end in Docker. Implements the two methods the
Agent expects:
  • draft_sql(user_text, tables, rules) -> {sql, provider, model, tokens_in, tokens_out, latency_ms}
  • rephrase_sql(sql_text, reasons, rules) -> same shape

Behavior
--------
- draft_sql: picks the first selected table and returns "SELECT * FROM <table> LIMIT <ROW_LIMIT>".
- rephrase_sql: if no LIMIT found, appends one; otherwise returns the input unchanged.

Swap‑out
--------
When you plug your real provider, keep the same return dict shape so telemetry
remains consistent.
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Any

DEFAULT_LIMIT = 50

class LLMStub:
    def __init__(self, *, default_limit: int | None = None) -> None:
        self.default_limit = default_limit or DEFAULT_LIMIT

    # --- Agent expects this on the retrieval (NL→SQL) step ---
    def draft_sql(self, *, user_text: str, tables: List[str], rules: str = "sqlite_strict") -> Dict[str, Any]:
        t0 = time.time()
        if not tables:
            raise ValueError("LLMStub.draft_sql: no tables provided")
        table = tables[0]
        try:
            limit = int(os.getenv("ROW_LIMIT", str(self.default_limit)))
        except Exception:
            limit = self.default_limit
        sql = f"SELECT * FROM {table} LIMIT {limit}"
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "sql": sql,
            "provider": "stub",
            "model": "llm-stub",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": latency_ms,
        }

    # --- Agent expects this on the constrained rephrase step ---
    def rephrase_sql(self, *, sql_text: str, reasons: List[str], rules: str = "sqlite_strict") -> Dict[str, Any]:
        t0 = time.time()
        sql = self._ensure_limit(sql_text)
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "sql": sql,
            "provider": "stub",
            "model": "llm-stub",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": latency_ms,
        }

    # --- helpers ---
    def _ensure_limit(self, sql: str) -> str:
        if "limit" in sql.lower():
            return sql
        try:
            limit = int(os.getenv("ROW_LIMIT", str(self.default_limit)))
        except Exception:
            limit = self.default_limit
        return f"{sql.rstrip(';')} LIMIT {limit}"
