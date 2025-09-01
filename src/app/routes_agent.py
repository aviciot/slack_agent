# src/app/routes_agent.py
"""
Agent-backed Slack route (/slack/ask)

- Verifies Slack signature
- Reads slash command text
- Calls Agent.run(...) and returns a Slack-safe reply string

Wire it in main.py:
    from src.app.routes_agent import router as agent_router
    app.include_router(agent_router)
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from src.app.slack_handlers import verify_slack_signature
from src.app.agent_types import SlackContext
from src.app.request_ctx import new_request_id

router = APIRouter()


@router.post("/slack/ask")
async def slack_ask(request: Request):
    # Parse form then verify signature (mirrors your existing routes)
    body = await request.form()
    raw_body = await request.body()
    if not verify_slack_signature(request.headers, raw_body):
        return PlainTextResponse("bad signature", status_code=401)

    text = (body.get("text") or "").strip()
    user_id = body.get("user_id") or "unknown"
    channel_id = body.get("channel_id") or "unknown"
    thread_ts = body.get("thread_ts")

    if not text:
        return PlainTextResponse("Usage: /ask <natural language question>", status_code=200)

    # Ensure Agent is available
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        return PlainTextResponse("Agent not initialized", status_code=503)

    # Build context & request id, then run
    rid = new_request_id()
    ctx = SlackContext(channel=channel_id, user=user_id, thread_ts=thread_ts)

    try:
        reply = agent.run(rid, text, ctx)
        # reply.reply is already Slack-safe (code block / truncation handled by Agent)
        return PlainTextResponse(reply.reply, status_code=200)
    except Exception as e:
        # Keep response terse for Slack; details will be in telemetry/logs
        return PlainTextResponse(f"Error: {type(e).__name__}", status_code=200)
