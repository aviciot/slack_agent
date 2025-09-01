# src/tools/qs_debug_run.py
from __future__ import annotations

import os, sys, json, time, re
from typing import Any, Dict, List
from uuid import uuid4

import requests

from src.services.catalog_retriever import CatalogRetriever
from src.services.query_bank_runtime import QueryBankRuntime
from src.persistence.telemetry_store import log_request, log_routing_decision, log_qcache_event

def _unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _strip_code_fences(text: str) -> str:
    # Extract SQL between ``` blocks if present; otherwise return stripped text
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return text.strip()

def llm_generate_sql(question: str, db_name: str, tables_hint: List[str]) -> Dict[str, Any]:
    """
    Generate SQL via Cloudflare Workers AI (default) or Ollama if TXT2SQL_PROVIDER=ollama.
    Returns: {"sql": ..., "prompt": ..., "tables_hint": [...]}
    """
    provider = (os.getenv("TXT2SQL_PROVIDER") or "cloudflare").lower()

    # Prepare the prompt (shared)
    tables_hint = _unique_keep_order(tables_hint)[:8]
    tables_block = ""
    if tables_hint:
        tables_block = "Tables likely relevant:\n" + "\n".join(f"- {t}" for t in tables_hint) + "\n"

    instruction = (
        f"You are an expert SQL generator for the '{db_name}' database. "
        "Write a SINGLE SQL SELECT statement that answers the question. "
        "Prefer the tables listed if they make sense. "
        "Do not include DDL, comments, or multiple statements. "
        "If a row cap is reasonable, include a LIMIT / FETCH depending on dialect."
    )
    prompt = f"""{instruction}

{tables_block}
Question:
{question}
SQL:"""

    if provider == "cloudflare":
        # Cloudflare Workers AI REST API
        base = (os.getenv("CF_BASE_URL")
                or f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CF_ACCOUNT_ID')}/ai/run").rstrip("/")
        model = os.getenv("TXT2SQL_MODEL", "@cf/defog/sqlcoder-7b-2")
        token = os.getenv("CF_API_TOKEN")
        if not token:
            raise RuntimeError("CF_API_TOKEN is not set")

        url = f"{base}/{model.lstrip('/')}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        # Per CF docs, the body uses {"prompt": "..."} for text generation models
        # Ref: Execute AI model endpoint
        payload = {"prompt": prompt}

        r = requests.post(url, headers=headers, json=payload, timeout=(5, 90))
        r.raise_for_status()
        data = r.json()
        # CF returns: {"result": {"response": "..."}}
        resp_text = (data.get("result") or {}).get("response", "")
        sql = _strip_code_fences(resp_text).strip()

        return {"sql": sql, "prompt": prompt, "tables_hint": tables_hint}

    # Fallback: Ollama (only if explicitly configured)
    host = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
    url = os.getenv("OLLAMA_GENERATE_URL") or f"{host}/api/generate"
    model = os.getenv("LLM_MODEL", "llama3")
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=(5, 90))
    r.raise_for_status()
    resp = r.json().get("response") or r.text
    sql = _strip_code_fences(resp).strip()
    return {"sql": sql, "prompt": prompt, "tables_hint": tables_hint}


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.qs_debug_run \"your question here\"")
        sys.exit(1)

    question = sys.argv[1].strip()
    request_id = uuid4().hex[:16]

    # --- Stage 1: shortlist via catalog ---
    retriever = CatalogRetriever()
    t0 = time.time()
    shortlist = retriever.shortlist_dbs(question, k=int(os.getenv("SHORTLIST_MAX", "2")))
    t1 = time.time()

    shortlist_json = [{"db": s.db_name, "score": round(float(s.score), 3)} for s in shortlist]
    db_filter = [s.db_name for s in shortlist] if shortlist else None

    # --- Stage 2: bank within shortlist ---
    bank = QueryBankRuntime()
    # Compute threshold consistent with BANK_ACCEPT
    try:
        BANK_ACCEPT = float(os.getenv("BANK_ACCEPT", "0.26"))
    except ValueError:
        BANK_ACCEPT = 0.26
    bank_threshold = max(0.0, min(1.0, 1.0 - BANK_ACCEPT))

    t2 = time.time()
    bank_hit = bank.bank_best_match(question, k=5, db_filter=db_filter)
    t3 = time.time()

    stage2_bank = None
    chosen_db = None
    chosen_reason = None
    chosen_rank = None

    if bank_hit:
        stage2_bank = {
            "db": bank_hit.db_name,
            "similarity": round(float(bank_hit.score), 3),
            "threshold": bank_threshold,
            "template_key": getattr(bank_hit, "template_key", None),
        }
        chosen_db = bank_hit.db_name
        chosen_reason = "bank_hit"
        log_qcache_event(
            request_id,
            "hit",
            qid=stage2_bank["template_key"],
            route="bank",
            question=question
        )

        # rank from shortlist if present
        if shortlist:
            for i, cand in enumerate(shortlist_json):
                if cand["db"] == chosen_db:
                    chosen_rank = i
                    break

        # --- Reconstruct SQL + params for debug output ---
        from src.services.template_utils import template_question, fill_sql_template
        sig, params = template_question(question)
        sql = fill_sql_template(bank_hit.template, params)

        # map numeric capture to :limit if present (debug parity with runtime)
        if "<NUMBER_0>" in params and (":limit" in sql) and ("limit" not in params):
            params["limit"] = params.pop("<NUMBER_0>")

        # inject default binds for start/end/limit (debug parity with runtime)
        params = bank._inject_default_time_binds(sql, params)  # intentionally calling helper

        stage2_bank["sql"] = sql
        stage2_bank["params"] = params

        # Optional: execute SQL locally if EXECUTE_DEBUG_SQL=1
        if os.getenv("EXECUTE_DEBUG_SQL", "0") == "1":
            from src.services.db_executors import make_registry_from_env
            execs = make_registry_from_env()
            if bank_hit.db_name in execs:
                rows = execs[bank_hit.db_name].execute(
                    sql,
                    row_cap=int(os.getenv("ROW_CAP", "20"))
                )
                stage2_bank["rows"] = rows

    else:
        if shortlist:
            chosen_db = shortlist[0].db_name
            chosen_reason = "catalog_llm"
            chosen_rank = 0
            log_qcache_event(request_id, "miss", qid=None, route="catalog_llm", question=question)
        else:
            chosen_db = None
            chosen_reason = "no_route"
            log_qcache_event(request_id, "miss", qid=None, route="none", question=question)

    timings = {
        "t_catalog_ms": int((t1 - t0) * 1000),
        "t_bank_ms": int((t3 - t2) * 1000),
    }

    # Top-level telemetry
    log_request(
        request_id=request_id,
        source="debug",
        text=question,
        route="bank" if bank_hit else "catalog_llm" if chosen_db else "none",
        status="ok" if chosen_db else "error",
        timings=timings,
    )

    # ---------------- LLM FALLBACK (only if bank missed and we have a chosen DB) ----------------
    stage2_llm = None
    stage2_llm = None
    if not bank_hit and chosen_db:
        # Build a hint list from catalog results (if any)
        tables_hint: List[str] = []
        for s in shortlist:
            if getattr(s, "db_name", None) == chosen_db:
                reasons = getattr(s, "reasons", {}) or {}
                for rec in reasons.get("tables", []):
                    tname = (rec or {}).get("table")
                    if tname:
                        tables_hint.append(tname)

        # If still empty, extract table names directly from the question as a fallback
        if not tables_hint:
            m = re.search(r'\btable\s+([A-Za-z0-9_\."]+)', question, flags=re.I)
            if m:
                q_table = m.group(1).strip('"\'')

                # Add the mentioned table
                tables_hint.append(q_table)

                # Add domain-appropriate anchor tables that often solve the task
                if chosen_db.lower() == "informatica":
                    # This table is your lookup for workflow ↔ table usage
                    tables_hint.append("informatica_related_tables")
                    # Common column hints help steer the LLM
                    tables_hint.extend(["source_name", "target_name", "workflow_name", "folder_name"])

        tables_hint = _unique_keep_order([t for t in tables_hint if t])[:12]

        # Call Ollama LLM to generate SQL
        try:
            gen = llm_generate_sql(question, chosen_db, tables_hint)
            stage2_llm = {
                "db": chosen_db,
                "tables_hint": gen["tables_hint"],
                "prompt": gen["prompt"],
                "sql": gen["sql"],
            }

            # Optional: execute generated SQL
            if os.getenv("EXECUTE_DEBUG_SQL", "0") == "1":
                from src.services.db_executors import make_registry_from_env
                execs = make_registry_from_env()
                if chosen_db in execs:
                    rows = execs[chosen_db].execute(
                        stage2_llm["sql"],
                        row_cap=int(os.getenv("ROW_CAP", "20"))
                    )
                    stage2_llm["rows"] = rows

            # Optional: cache LLM template back to bank
            if os.getenv("CACHE_LLM_TEMPLATE", "0") == "1":
                bank.maybe_cache_template_from_llm(question, stage2_llm["sql"], db_name=chosen_db)

        except Exception as e:
            stage2_llm = {"db": chosen_db, "error": f"{type(e).__name__}: {e}"}

    # Router telemetry
    log_routing_decision(
        request_id=request_id,
        source="catalog_bank",
        k=5,
        candidates=[{"db": c["db"], "score": c["score"]} for c in shortlist_json],
        selected=[chosen_db] if chosen_db else [],
        chosen_rank=chosen_rank,
        stage1_shortlist=shortlist_json,
        stage2_bank=stage2_bank or {},
        reason=chosen_reason,
    )

    # Output
    out: Dict[str, Any] = {
        "ok": bool(chosen_db),
        "request_id": request_id,
        "question": question,
        "shortlist": shortlist_json,
        "bank_hit": bool(bank_hit),
        "picked_db": chosen_db,
        "reason": chosen_reason,
        "timings": timings,
    }
    if bank_hit:
        out["bank"] = stage2_bank
    if stage2_llm:
        out["llm"] = stage2_llm

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
