# src/routes/text2sql_debug.py
from __future__ import annotations

import os, re, requests
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel

from src.services.catalog_retriever import CatalogRetriever
from src.services.query_bank_runtime import QueryBankRuntime
from src.services.template_utils import template_question, fill_sql_template
from src.services.db_executors import make_registry_from_env

router = APIRouter(prefix="/api", tags=["debug"])

# ---------------- Models ----------------

class Hints(BaseModel):
    tables: Optional[List[str]] = None

class Text2SQLReq(BaseModel):
    question: str
    db: Optional[str] = None        # optional override (otherwise auto from shortlist)
    hints: Optional[Hints] = None   # optional override (otherwise auto from catalog)
    execute: bool = False
    row_cap: int = 50

class Text2SQLResp(BaseModel):
    route: str
    bank_hit: bool
    picked_db: Optional[str]
    shortlist: List[Dict[str, Any]]
    bank: Optional[Dict[str, Any]] = None
    llm: Optional[Dict[str, Any]] = None
    rows: Optional[List[Any]] = None
    trace: Optional[Dict[str, Any]] = None   # <-- new: clear trace of what happened

# ---------------- Helpers ----------------

def _strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()

def _uniq_keep(xs: List[str], cap: int = 12) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out[:cap]

def _cf_generate_sql(question: str, db_name: str, tables_hint: List[str]) -> Dict[str, Any]:
    """
    Generate SQL via Cloudflare Workers AI (prod provider).
    Returns a dict that includes everything we SENT and what we RECEIVED (sanitized).
    """
    base = (os.getenv("CF_BASE_URL")
            or f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CF_ACCOUNT_ID')}/ai/run").rstrip("/")
    model = os.getenv("TXT2SQL_MODEL", "@cf/defog/sqlcoder-7b-2")
    token = os.getenv("CF_API_TOKEN")
    if not token:
        raise RuntimeError("CF_API_TOKEN is not set")

    tbls = _uniq_keep(tables_hint or [], 12)
    tables_block = ("Tables likely relevant:\n" + "\n".join(f"- {t}" for t in tbls) + "\n") if tbls else ""

    instruction = (
        f"You are an expert SQL generator for the '{db_name}' database. "
        "Write a SINGLE SQL SELECT statement that answers the question. "
        "Prefer the tables listed if they make sense. "
        "Do not include DDL, comments, or multiple statements. "
        "If a row cap is reasonable, include a LIMIT / FETCH depending on dialect."
    )
    prompt = f"""{instruction}

{tables_block}Question:
{question}
SQL:"""

    url = f"{base}/{model.lstrip('/')}"
    headers = {
        "Authorization": "Bearer ****",             # do NOT leak token
        "Content-Type": "application/json",
    }
    # Actual request (with real token) but we keep the echoed headers sanitized
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"prompt": prompt},
        timeout=(5, 90)
    )
    resp.raise_for_status()
    data = resp.json()
    raw = (data.get("result") or {}).get("response", "") or ""
    sql = _strip_code_fences(raw)

    return {
        # what we sent
        "provider": "cloudflare",
        "model": model,
        "url": url,
        "request": {
            "prompt": prompt,
            "tables_hint": tbls,
            "headers": headers,  # sanitized
        },
        # what we received
        "response": {
            "raw_preview": raw[:300],
            "sql": sql,
        },
        # convenience mirror
        "sql": sql,
        "prompt": prompt,
        "tables_hint": tbls,
    }

def _choose_db(shortlist_objs: List[Any], explicit_db: Optional[str]) -> Tuple[Optional[str], str]:
    if explicit_db:
        return explicit_db.strip().lower(), "request"
    return (shortlist_objs[0].db_name if shortlist_objs else None), "shortlist_auto"

def _build_hints_from_catalog(shortlist_objs: List[Any], picked_db: Optional[str]) -> Tuple[List[str], str]:
    tables: List[str] = []
    if picked_db:
        for s in shortlist_objs:
            if getattr(s, "db_name", None) == picked_db:
                for rec in (getattr(s, "reasons", {}) or {}).get("tables", []):
                    t = (rec or {}).get("table")
                    if t:
                        tables.append(t)
    return _uniq_keep(tables, 12), "catalog_auto"

# ---------------- Route ----------------

@router.post("/text2sql", response_model=Text2SQLResp)
def text2sql(req: Text2SQLReq):
    """
    Production-like text→SQL flow with a clear trace:
      1) Catalog shortlist (auto DB if not provided)
      2) Query bank (threshold via BANK_ACCEPT)
      3) Fallback to Cloudflare LLM
      4) Optional execution via executors
      5) Trace shows exactly what was sent to and received from the LLM
    """
    # Stage 1 — shortlist
    retriever = CatalogRetriever()
    shortlist_objs = retriever.shortlist_dbs(req.question, k=int(os.getenv("SHORTLIST_MAX", "2")))
    shortlist = [{"db": s.db_name, "score": float(s.score)} for s in shortlist_objs]

    picked_db, db_source = _choose_db(shortlist_objs, req.db)

    # Stage 2 — bank
    bank = QueryBankRuntime()
    hit = bank.bank_best_match(req.question, k=5, db_filter=[picked_db] if picked_db else None)

    # threshold (for trace)
    try:
        bank_accept = float(os.getenv("BANK_ACCEPT", "0.26"))
    except Exception:
        bank_accept = 0.26
    bank_threshold = max(0.0, min(1.0, 1.0 - bank_accept))

    trace: Dict[str, Any] = {
        "stage1": {
            "db_source": db_source,
            "picked_db": picked_db,
            "shortlist": shortlist,
        },
        "stage2": {
            "bank_threshold": bank_threshold,
            "accepted": bool(hit),
            "db_filter": [picked_db] if picked_db else [],
        }
    }

    if hit:
        # Fill template with extracted params
        sig, params = template_question(req.question)
        sql = fill_sql_template(hit.template, params)
        # Runtime parity helpers
        if "<NUMBER_0>" in params and (":limit" in sql) and ("limit" not in params):
            params["limit"] = params.pop("<NUMBER_0>")
        params = bank._inject_default_time_binds(sql, params)

        trace["stage2"].update({
            "top_similarity": float(hit.score),
            "template_key": getattr(hit, "template_key", None),
            "signature": getattr(hit, "signature", None),
        })

        resp: Dict[str, Any] = {
            "route": "bank",
            "bank_hit": True,
            "picked_db": picked_db,
            "shortlist": shortlist,
            "bank": {
                "db": hit.db_name,
                "similarity": float(hit.score),
                "threshold": bank_threshold,
                "template_key": getattr(hit, "template_key", None),
                "signature": getattr(hit, "signature", None),
                "sql": sql,
                "params": params,
            },
            "llm": None,
            "trace": trace,
        }

        if req.execute:
            execs = make_registry_from_env()
            if hit.db_name in execs:
                rows = execs[hit.db_name].execute(sql, row_cap=max(1, min(req.row_cap, 500)))
                resp["rows"] = rows
        return resp

    # Fallback — LLM (build hints)
    auto_hints, hints_source = _build_hints_from_catalog(shortlist_objs, picked_db)
    tables_hint = auto_hints[:]
    if req.hints and req.hints.tables:
        tables_hint = _uniq_keep(tables_hint + req.hints.tables, 12)
        hints_source = "catalog+request"

    # Generate via Cloudflare
    gen = _cf_generate_sql(req.question, picked_db or "unknown", tables_hint)

    trace["stage3_llm"] = {
        "provider": gen.get("provider"),
        "model": gen.get("model"),
        "url": gen.get("url"),
        "db_used": picked_db,
        "hints_source": hints_source,
        "tables_hint": gen.get("tables_hint", []),
        "prompt_chars": len(gen.get("prompt", "")),
        "response_preview": gen.get("response", {}).get("raw_preview", "")[:200],
    }

    resp2: Dict[str, Any] = {
        "route": "catalog_llm",
        "bank_hit": False,
        "picked_db": picked_db,
        "shortlist": shortlist,
        "bank": None,
        "llm": {
            "provider": gen["provider"],
            "model": gen["model"],
            "url": gen["url"],
            "request": gen["request"],     # includes full prompt + hints (sanitized headers)
            "response": gen["response"],   # raw preview + final SQL
            "sql": gen["sql"],
            "tables_hint": gen["tables_hint"],
            "prompt": gen["prompt"],
        },
        "trace": trace,
    }

    if req.execute and picked_db:
        execs = make_registry_from_env()
        if picked_db in execs:
            rows = execs[picked_db].execute(gen["sql"], row_cap=max(1, min(req.row_cap, 500)))
            resp2["rows"] = rows

    return resp2
