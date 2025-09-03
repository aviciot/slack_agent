# src/tools//app/src/tools/smoke_error_decider_telemetry.py
from __future__ import annotations

import os
import uuid
import json
import sqlite3
import requests

from src.services.db_executors import SQLiteExecutor
from src.persistence.db import get_conn

API_BASE = os.getenv("API_BASE", "http://localhost:8090")
DB_PATH = os.getenv("SMOKE_SQLITE_DB", "/app/data/db/smoke_decider.sqlite")


"""
Smoke Test: Error Decider + Error Bank Integration
==================================================

This script runs a set of failing SQL queries against a temporary SQLite DB
to validate the full error-decider pipeline, telemetry logging, and Redis
error-bank caching.

Test Cases
----------
1. **Regex path (missing table)**  
   - Query: SELECT * FROM definitely_missing_table  
   - Expected: decision comes from regex rules.  
   - Source in logs/telemetry: "regex".

2. **LLM path (unknown function, first time)**  
   - Query: SELECT oops_broken()  
   - Expected: decision initially comes from the LLM.  
   - Stored in error bank under signature "sqlite:msg:<hash>".  
   - Source in logs/telemetry: "llm-cloudflare".  
   - On replay (second run), should come from "bank".

3. **Bank path (unknown function, repeat)**  
   - Same query as case 2 (SELECT oops_broken()).  
   - Expected: decision is served directly from the error bank
     (no LLM call this time).  
   - Source in logs/telemetry: "bank".

Expected Results
----------------
- Each case raises a `RuntimeError` with structured [ERROR_DECIDER] info.
- A corresponding row is written into `error_decisions` telemetry table
  when `LOG_ERROR_DECISIONS=true`.
- Redis error bank is updated:
  * Regex errors stored as `sqlite:regex:<hash>`
  * Message-based errors stored as `sqlite:msg:<hash>`
- By case 3, the source is "bank" (showing the cache is effective).
- The summary prints sources observed per case, proving fallback order:
  regex → llm → bank.

Usage
-----
Run inside the API container:

    docker compose exec api python /app/src/tools/smoke_error_decider_telemetry.py

Make sure Redis and SQLite paths are available, and the API service is up.
"""


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fetch_error_decision(request_id: str) -> dict | None:
    try:
        with get_conn() as cx:
            cx.row_factory = sqlite3.Row
            rows = cx.execute(
                """
                SELECT request_id, db_name, vendor, action, category, reason,
                       source, signature, confidence, created_at
                FROM error_decisions
                WHERE request_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (request_id,),
            ).fetchall()
        return dict(rows[0]) if rows else None
    except Exception as e:
        print(f"!! Failed to read telemetry error_decisions: {e}")
        return None


def run_case(sql: str, label: str) -> dict:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    execu = SQLiteExecutor(db_path=DB_PATH)
    execu.db_name = "smoketest_sqlite"

    request_id = str(uuid.uuid4())
    print_header(f"{label}  (request_id={request_id})")
    print(f"SQL: {sql}")

    try:
        execu.execute(sql, row_cap=50, request_id=request_id)
        print("UNEXPECTED: query succeeded (should have failed).")
    except Exception as e:
        print(f"Expected failure: {type(e).__name__}: {e}")

    row = fetch_error_decision(request_id)
    if row:
        print("error_decision row:")
        print(json.dumps(row, indent=2, default=str))
    else:
        print("No error_decision row found (check LOG_ERROR_DECISIONS=true).")
    return row or {}


def hit_errorbank_endpoint():
    print_header("Querying /api/store/errorbank/search?vendor=sqlite")
    url = f"{API_BASE}/api/store/errorbank/search?vendor=sqlite"
    try:
        r = requests.get(url, timeout=5)
        print(f"HTTP {r.status_code}")
        if r.ok:
            data = r.json()
            # Show a few entries, prioritize any with source starting llm- or bank
            interesting = [x for x in data if str(x.get("source", "")).startswith(("llm", "bank"))]
            sample = (interesting or data)[:5]
            print(json.dumps(sample, indent=2))
        else:
            print(r.text)
    except Exception as e:
        print(f"!! Failed to call errorbank endpoint: {e}")


def main():
    print(f"LOG_ERROR_DECISIONS={os.getenv('LOG_ERROR_DECISIONS', 'unset')}")
    print(f"API_BASE={API_BASE}")
    print(f"DB_PATH={DB_PATH}")

    # 1) Regex path: "no such table"
    row1 = run_case("SELECT * FROM definitely_missing_table", "Case 1: Regex (missing table)")

    # 2) LLM path: "no such function" (not covered by your regex; should hit LLM then cache)
    row2 = run_case("SELECT oops_broken()", "Case 2: LLM (unknown function)")

    # 3) Same LLM error again — should now hit bank if message-hash caching is enabled
    row3 = run_case("SELECT oops_broken()", "Case 3: Bank (repeat of unknown function)")

    # Print quick summary of sources
    print_header("Summary (sources)")
    print(json.dumps({
        "case1_source": row1.get("source"),
        "case2_source": row2.get("source"),
        "case3_source": row3.get("source"),
    }, indent=2))

    # 4) Peek at the bank via the API endpoint (optional but useful)
    hit_errorbank_endpoint()


if __name__ == "__main__":
    main()

## To run:
## docker compose exec api python /app/src/tools//app/src/tools/smoke_error_decider_telemetry.py