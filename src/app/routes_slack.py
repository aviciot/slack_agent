# src/app/routes_slack.py

import os, json, re, asyncio
import httpx, redis
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from .slack_handlers import verify_slack_signature
from .routes_sql import plan, run, PlanRequest, RunRequest

router = APIRouter()

REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")

# util: format rows into Slack text block
def _format_result(result: dict) -> str:
    if result["rowcount"] == 0:
        return "No rows found."
    cols = result["columns"]
    header = " | ".join(cols)
    sep = "-|-".join("-" * len(c) for c in cols)
    lines = []
    for row in result["rows"][:10]:   # only first 10 rows for Slack
        vals = [str(row.get(c, "")) for c in cols]
        lines.append(" | ".join(vals))
    return f"```\n{header}\n{sep}\n" + "\n".join(lines) + "\n```"


@router.post("/slack/sql")
async def slack_sql(request: Request):
    """
    Slack slash command handler for /sql
    """
    body = await request.form()
    text = body.get("text", "").strip()

    # ✅ verify Slack signature
    raw_body = await request.body()
    if not verify_slack_signature(request.headers, raw_body):
        return PlainTextResponse("bad signature", status_code=401)

    if not text:
        return PlainTextResponse("Please provide a query text after /sql", status_code=200)

    # TODO: here you’d plug in intent detection / bank search.
    # For now assume the user typed the Redis key directly, e.g. "bank:Q006"
    if text.lower().startswith("bank:"):
        key = text
        # plan + run
        plan_req = PlanRequest(key=key, params={})
        plan_resp = plan(plan_req)
        if plan_resp.missing_params:
            return PlainTextResponse(
                f"Missing params: {plan_resp.missing_params}", status_code=200
            )
        run_req = RunRequest(key=key, params={}, limit=50)
        result = run(run_req)
        return PlainTextResponse(_format_result(result), status_code=200)

    # fallback — not recognized
    return PlainTextResponse(
        f"Unrecognized query: {text}", status_code=200
    )
