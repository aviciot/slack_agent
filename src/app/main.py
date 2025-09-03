# src/app/main.py
# [Agent] imports
from src.app.agent import Agent
from src.app.llm_stub import LLMStub
from src.validation.sql_guard import SQLGuard
from src.services.query_bank_runtime import QueryBankRuntime
from src.services.table_selector import TableSelector

from src.app.agent_types import SlackContext

from contextlib import asynccontextmanager
import os, json, time, sqlite3, glob
from typing import Dict, Any
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.providers import get_llm_config

# Routers
from src.app.routes_admin import router as admin_router
from src.app.routes_agent import router as agent_router
from src.app.routes_agent_debug import router as agent_debug_router
from src.app.text2sql_debug import router as text2sql_router
from src.app.store_inspect import router as store_inspect_router
from src.app.routes_task_debug import router as task_debug_router
from src.app.routes_errorbank import router as errorbank_router


# Request context / logging
from src.app.request_ctx import (
    new_request_id, get_route, get_status, get_timings, add_timing, set_status
)
from src.app.slack_handlers import verify_slack_signature
from src.app.logging_setup import configure_logging, set_request_id

# Telemetry facade + DB init
from src import telemetry as T
from src.persistence.init_db import ensure_db

logger = configure_logging()

APP_START = time.time()

DB_PATH = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")

# Provider envs (left as-is)
PROVIDER = os.getenv("PROVIDER", "local")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://llm-local:11434")
CF_BASE_URL = os.getenv("CF_BASE_URL", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")

TIMERS_ENABLED = os.getenv("TIMERS_ENABLED", "true").lower() == "true"

# Catalog config (NEW defaults):
# - TABLE_CATALOG: legacy single-file (disabled by default)
# - TABLE_CATALOGS_DIR / CATALOGS_DIR: directory of multiple catalogs (one per DB)
TABLE_CATALOG = os.getenv("TABLE_CATALOG", "").strip()  # <- empty disables legacy path
TABLE_CATALOGS_DIR = (os.getenv("TABLE_CATALOGS_DIR") or os.getenv("CATALOGS_DIR") or "").strip()

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # keep log correlation id for your logger
        set_request_id()
        return await call_next(request)

# ---- Provider check ----
def provider_ok() -> bool:
    try:
        cfg = get_llm_config("txt2sql")
        return bool(cfg.provider and cfg.model)
    except Exception:
        return False

def _normalize_catalog_obj(obj) -> Dict[str, list[str]]:
    """
    Accepts multiple shapes and returns {table_name: [columns...]}.
    Supported inputs:
      A) [{"name": "wf_runs", "columns": [{"name":"col1"}, ...]}, ...]
      B) {"wf_runs": ["col1","col2", ...], ...}
      C) {"tables": [ <shape A items> ]}
    """
    result: Dict[str, list[str]] = {}
    # Case B
    if isinstance(obj, dict) and all(isinstance(v, list) for v in obj.values()) and "tables" not in obj:
        for t, cols in obj.items():
            result[str(t)] = [str(c) for c in cols]
        return result
    # Case C
    if isinstance(obj, dict) and isinstance(obj.get("tables"), list):
        obj = obj["tables"]
    # Case A (or after C)
    if isinstance(obj, list):
        for table in obj:
            if not isinstance(table, dict):
                continue
            name = table.get("name")
            cols_spec = table.get("columns") or []
            cols: list[str] = []
            for c in cols_spec:
                if isinstance(c, dict) and "name" in c:
                    cols.append(str(c["name"]))
                elif isinstance(c, str):
                    cols.append(c)
            if name:
                result[str(name)] = cols
        return result
    raise ValueError("Unrecognized catalog JSON schema")

def load_catalogs(single_path: str, dir_path: str | None) -> dict[str, list[str]]:
    """
    Returns {table_name: [col,...]} merged from either a single file
    or all *.json files in a directory. Raises if nothing loads.
    """
    merged: dict[str, list[str]] = {}
    loaded_any = False

    # Prefer directory when provided and exists
    if dir_path and os.path.isdir(dir_path):
        for path in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            part = _normalize_catalog_obj(obj)
            merged.update(part)
            loaded_any = True
        logger.info(f"[CATALOG] loaded {len(merged)} tables from dir={dir_path}")

    # Fallback to single file (ONLY if provided)
    if not loaded_any and single_path:
        if os.path.exists(single_path):
            with open(single_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            merged = _normalize_catalog_obj(obj)
            loaded_any = True
            logger.info(f"[CATALOG] loaded {len(merged)} tables from file={single_path}")

    if not loaded_any:
        raise FileNotFoundError("No catalog JSON found (checked dir and single file).")
    return merged

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_db()

    # 1) Load catalogs (multi-file preferred; single-file disabled by default)
    try:
        _catalog = load_catalogs(TABLE_CATALOG, TABLE_CATALOGS_DIR)
    except Exception as e:
        logger.warning(
            json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "level": "WARNING",
                "msg": f"Failed to load catalogs ({e}); building from SQLite DB at {DB_PATH}",
            })
        )
        _catalog = build_catalog(DB_PATH)

    # 2) Core services
    _query_bank = QueryBankRuntime()
    _table_selector = TableSelector()  # kept for legacy/compat paths
    _sql_guard = SQLGuard(db_path=DB_PATH, catalog=_catalog, big_tables=None)
    _llm = LLMStub()

    # 3) Agent (lets Agent build executors from env; multi-DB ready)
    app.state.agent = Agent(
        settings=type("S", (), {
            "RETRY_MAX": int(os.getenv("RETRY_MAX", "3")),
            "RETRY_BASE_MS": int(os.getenv("RETRY_BASE_MS", "150")),
            "RETRY_JITTER_MS": int(os.getenv("RETRY_JITTER_MS", "200")),
            "PLAN_MAX_STEPS": int(os.getenv("PLAN_MAX_STEPS", "6")),
            "ALT_TABLE_TRIES": int(os.getenv("ALT_TABLE_TRIES", "2")),
            "ROW_LIMIT": int(os.getenv("ROW_LIMIT", "200")),
        })(),
        query_bank=_query_bank,
        table_selector=_table_selector,
        validator=_sql_guard,
        llm=_llm,
    )

    yield
    # optional shutdown hooks here

def build_catalog(db_path: str) -> dict[str, list[str]]:
    catalog = {}
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cur.fetchall()]
        for table in tables:
            cur.execute(f"PRAGMA table_info('{table}');")
            columns = [row[1] for row in cur.fetchall()]
            catalog[table] = columns
    logger.info(f"[CATALOG] built {len(catalog)} tables from SQLite at {db_path}")
    return catalog

app = FastAPI(title="Slack Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)

@app.middleware("http")
async def reqid_and_telemetry_middleware(request: Request, call_next):    
    rid = new_request_id()
    try:
        set_request_id(rid)  # for logger correlation
    except Exception:
        pass

    path = request.url.path
    if path.startswith("/slack"):
        source = "slack"
    elif path.startswith("/scheduler"):
        source = "scheduler"
    else:
        source = "api"

    user_id = request.headers.get("X-Slack-User")
    channel_id = request.headers.get("X-Slack-Channel")

    t0 = time.time_ns()
    try:
        response: Response = await call_next(request)
        status_str = get_status()
    except Exception:
        set_status("error")
        status_str = "error"
        add_timing("total_ms", t0)
        try:
            T.http_request(
                request_id=rid,
                source=source,
                path=path,
                text=f"{request.method} {path}",
                user_id=user_id,
                channel_id=channel_id,
                route=get_route(),
                status=status_str,
                timings=get_timings(),
            )
        finally:
            raise

    add_timing("total_ms", t0, time.time_ns())
    T.http_request(
        request_id=rid,
        source=source,
        path=path,
        text=f"{request.method} {path}",
        user_id=user_id,
        channel_id=channel_id,
        route=get_route(),
        status=status_str,
        timings=get_timings(),
    )
    return response

# ---- Routers ----
app.include_router(admin_router, prefix="/admin", tags=["admin"])
app.include_router(agent_router)
app.include_router(agent_debug_router)
app.include_router(text2sql_router)
app.include_router(store_inspect_router)
app.include_router(task_debug_router)
app.include_router(errorbank_router)

# ---- Health ----
def db_ok() -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False

@app.get("/healthz")
async def healthz():
    deps = {
        "db": db_ok(),
        "provider": provider_ok(),   # no live call
        "uptime_s": round(time.time() - APP_START, 2),
    }
    ok = all(v if isinstance(v, bool) else True for v in deps.values())
    code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse({"ok": ok, **deps}, status_code=code)

# ---- Slack endpoints ----
@app.post("/slack/events")
async def slack_events(request: Request):
    body = await request.body()
    if not verify_slack_signature(request.headers, body):
        return PlainTextResponse("bad signature", status_code=401)
    return PlainTextResponse("OK", status_code=200)

@app.post("/slack/interactive")
async def slack_interactive(request: Request):
    body = await request.body()
    if not verify_slack_signature(request.headers, body):
        return PlainTextResponse("bad signature", status_code=401)
    return PlainTextResponse("OK", status_code=200)

@app.get("/debug/agent")
async def debug_agent(text: str = ""):
    """
    Manual E2E test for the Agent without Slack:
      GET /debug/agent?text=top 10 rows from ...
    """
    if not text:
        return PlainTextResponse("Provide ?text=...", status_code=400)

    rid = new_request_id()
    ctx = SlackContext(channel="debug", user="debug")
    reply = app.state.agent.run(rid, text, ctx)

    return JSONResponse({
        "status": reply.status.value,
        "reply": reply.reply,
        "meta": {
            "request_id": reply.meta.request_id,
            "route": reply.meta.route.source,
            "tables": reply.meta.route.selected,
            "retries": reply.meta.retries,
            "plan_steps": reply.meta.plan_steps,
        },
    })
