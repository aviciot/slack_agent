# src/tools/smoke_error_bank.py
from __future__ import annotations

import os
import sys
import json
import requests
from src.services.error_decider.decider import ErrorDecider

def run_case(vendor: str, text: str, counters: dict) -> None:
    decider = ErrorDecider()
    out = decider.decide(db_vendor=vendor, db_name="testdb", sql="select 1", error_text=text)
    llm_used = str(out.get("source", "")).startswith("llm")

    print(f"\nVendor={vendor} | Error={text}")
    print(out)
    print(f"llm_used={llm_used}")

    counters["total"] += 1
    if llm_used:
        counters["llm"] += 1
    else:
        # count bank/regex/fallback buckets
        src = str(out.get("source", ""))
        if src == "bank":
            counters["bank"] += 1
        elif src.startswith("regex"):
            counters["regex"] += 1
        else:
            counters["other"] += 1

def test_cloudflare_call() -> bool:
    """
    Minimal direct call to Cloudflare Workers AI so we can fail-fast if the model slug or creds are wrong.
    Requires:
      CF_ACCOUNT_ID
      CF_API_TOKEN
      ERROR_DECIDER_MODEL (full CF slug, e.g. '@cf/meta/llama-3-8b-instruct')
    """
    account = os.getenv("CF_ACCOUNT_ID")
    token = os.getenv("CF_API_TOKEN")
    model = os.getenv("ERROR_DECIDER_MODEL", "@cf/meta/llama-3-8b-instruct")

    if not (account and token):
        print("⚠️  CF_ACCOUNT_ID / CF_API_TOKEN not set; skipping Cloudflare direct test.")
        return False

    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "messages": [
            {"role": "system", "content": "You are a test bot. Reply only with JSON."},
            {"role": "user", "content": "Give me a JSON {action:RETRY,category:transient,reason:Testing}"}
        ]
    }

    print(f"\n=== Direct Cloudflare API Test ({model}) ===")
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=12)
        print(f"status_code={resp.status_code}")
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"❌ Cloudflare call failed: {e}")
        return False

    ok = bool(data.get("success"))
    if ok:
        print("✅ Cloudflare test succeeded")
    else:
        print("❌ Cloudflare test failed — stopping further smoke tests.")
    return ok

if __name__ == "__main__":
    print("=== Smoke Error Bank Test ===")
    print(f"ERROR_DECIDER_MODE     = {os.getenv('ERROR_DECIDER_MODE')}")
    print(f"ERROR_DECIDER_PROVIDER = {os.getenv('ERROR_DECIDER_PROVIDER')}")
    print(f"ERROR_DECIDER_MODEL    = {os.getenv('ERROR_DECIDER_MODEL')}")
    print("-------------------------------")

    # If provider is cloudflare, verify the API first. Abort on failure.
    if (os.getenv("ERROR_DECIDER_PROVIDER") or "").lower() == "cloudflare":
        if not test_cloudflare_call():
            sys.exit(1)

    counters = {"total": 0, "llm": 0, "bank": 0, "regex": 0, "other": 0}

    # Covered cases (bank/regex)
    run_case("sqlite",   "sqlite3.OperationalError: no such table: invoices", counters)
    run_case("sqlite",   "sqlite3.IntegrityError: constraint failed", counters)
    run_case("oracle",   "ORA-00942: table or view does not exist", counters)
    run_case("oracle",   "ORA-01017: invalid username/password", counters)
    run_case("postgres", "ERROR: permission denied for relation users", counters)

    print("\n=== Uncovered cases (should hit LLM) ===")
    run_case("sqlite",    "sqlite3.OperationalError: misuse of aggregate: count()", counters)
    run_case("oracle",    "ORA-22222: custom unknown oracle error", counters)
    run_case("postgres",  "invalid byte sequence for encoding UTF8", counters)
    run_case("snowflake", "Numeric value 'abc' is not recognized", counters)
    run_case("mysql",     "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use", counters)
    run_case("mysql",     "ERROR 1045 (28000): Access denied for user 'user'@'host' (using password: YES)", counters)
    run_case("mssql",     "Msg 18456, Level 14, State 1, Server myserver, Line 1 Login failed for user 'user'.", counters)

    print("\n=== Summary ===")
    print(json.dumps(counters, indent=2))
    print("=== End Smoke Test ===")

##docker compose exec api python /app/src/tools/smoke_error_bank.py




# SQLite
# ✔ no such table → QUIT, schema, source: regex
# ✔ constraint failed → ASK_USER, data, source: regex

# Oracle
# ✔ ORA-00942 → REPHRASE, schema, from bank
# ✔ ORA-01017 → QUIT, auth, from bank

# Postgres (any vendor fallback)
# ✔ permission denied → QUIT, auth, from **regex:any(mapped topg` in signature)