# src/app/routes_agent.py
from __future__ import annotations

from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, Query, HTTPException, Body
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.app.slack_handlers import verify_slack_signature
from src.app.agent_types import SlackContext
from src.app.request_ctx import new_request_id

router = APIRouter(tags=["Slack"])

# ---------- Swagger-visible JSON for dev mode ----------
class SlackAskDevBody(BaseModel):
    text: str = Field(..., description="Question text (dev mode via Swagger)")
    user_id: str = Field("UDEV", description="Optional: fake Slack user id (dev)")
    channel_id: str = Field("CDEV", description="Optional: fake Slack channel id (dev)")
    thread_ts: Optional[str] = Field(None, description="Optional: fake Slack thread ts (dev)")

def _run_agent(agent: Any, *, text: str, user_id: str, channel_id: str, thread_ts: Optional[str]) -> str:
    rid = new_request_id()
    ctx = SlackContext(channel=channel_id, user=user_id, thread_ts=thread_ts)
    resp = agent.run(rid, text, ctx)  # your existing orchestrator
    return getattr(resp, "reply", "OK")

@router.post(
    "/slack/ask",
    summary="Handle a Slack question",
    description=(
        "Production: Slack sends a signed form payload (verified).\n"
        "Dev (Swagger): set `dev=true` and send JSON body to test without Slack."
    ),
    response_class=PlainTextResponse,
    responses={
        200: {"description": "Plain text reply (Slack-safe)"},
        401: {"description": "Missing/invalid Slack signature (prod)"},
        503: {"description": "Agent not initialized"},
    },
)
async def slack_ask(
    request: Request,
    dev: bool = Query(False, description="Set true to test from Swagger/clients (no Slack signature)."),
    body: Optional[SlackAskDevBody] = Body(
        None,
        description="Used only when dev=true (Swagger/JSON).",
        examples={
            "ask_example": {
                "summary": "Simple dev request",
                "value": {"text": "show me statements count by status in July"}
            }
        },
    ),
):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # ---------- DEV MODE (Swagger-friendly) ----------
    if dev:
        if not body or not body.text.strip():
            return PlainTextResponse(
                'Usage (dev): set dev=true and provide JSON body like {"text": "your question"}',
                status_code=200,
            )
        reply_text = _run_agent(
            agent,
            text=body.text.strip(),
            user_id=body.user_id,
            channel_id=body.channel_id,
            thread_ts=body.thread_ts,
        )
        return PlainTextResponse(reply_text, status_code=200)

    # ---------- PRODUCTION MODE (real Slack) ----------
    raw_body = await request.body()
    if not verify_slack_signature(request.headers, raw_body):
        return PlainTextResponse("bad signature", status_code=401)

    form = await request.form()
    text = (form.get("text") or "").strip()
    user_id = form.get("user_id") or "unknown"
    channel_id = form.get("channel_id") or "unknown"
    thread_ts = form.get("thread_ts")

    if not text:
        return PlainTextResponse("Usage: /ask <natural language question>", status_code=200)

    reply_text = _run_agent(
        agent,
        text=text,
        user_id=user_id,
        channel_id=channel_id,
        thread_ts=thread_ts,
    )
    return PlainTextResponse(reply_text, status_code=200)
