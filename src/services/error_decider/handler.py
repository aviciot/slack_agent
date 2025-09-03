#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional

from .decider import ErrorDecider


def _to_error_text(err: Any) -> str:
    """
    Normalize various DB error shapes to a plain string.
    """
    if err is None:
        return ""
    # cx_Oracle / oracledb often stringify well
    try:
        s = str(err)
        if s and s != repr(err):
            return s
    except Exception:
        pass

    # dict-like? stitch common fields
    if isinstance(err, dict):
        msg = err.get("message") or err.get("msg") or err.get("error") or ""
        code = err.get("code") or err.get("sqlstate") or ""
        return f"{code} {msg}".strip()

    return repr(err)


class ErrorHandlingService:
    """
    Thin wrapper around ErrorDecider that:
      - extracts error text
      - calls the decider (force_quit → bank → regex → LLM → fallback)
      - evaluates simple retry advice against the DB policy
    """

    def __init__(
        self,
        *,
        databases_path: str = "/app/config/databases.json",
        redis_url: Optional[str] = None,
        bank_prefix: Optional[str] = None,
        bank_ttl_days: Optional[int] = None,
    ):
        self.decider = ErrorDecider(
            databases_path=databases_path,
            redis_url=redis_url,
            bank_prefix=bank_prefix,
            bank_ttl_days=bank_ttl_days,
        )

    def handle(
        self,
        *,
        db_vendor: str,
        db_name: str,
        sql: str,
        error: Any,
        retry_count: int = 0,
        exec_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns a single dict with the decision + retry guidance.
        """
        error_text = _to_error_text(error)

        # Decision (includes action/category/reason/confidence/source/error_signature)
        decision = self.decider.decide(
            db_vendor=db_vendor,
            db_name=db_name,
            sql=sql,
            error_text=error_text,
            retry_count=retry_count,
            exec_phase=exec_phase,
        )

        # Pull policy to compute simple retry advice
        # (access the decider's cached policies; ok to call the private getter here)
        policy = self.decider._policy_for(db_name)  # noqa: SLF001 (internal use acceptable)
        max_retries = int(policy.get("max_retries", 0))

        retry_allowed = decision.get("action") == "RETRY" and retry_count < max_retries

        return {
            "db_name": db_name,
            "db_vendor": db_vendor,
            "exec_phase": exec_phase or "execute",
            "sql_redacted": sql,  # already redacted by your executor before calling (recommended)
            "error_text": error_text,
            "decision": decision,
            "retry": {
                "allowed": bool(retry_allowed),
                "retry_count": int(retry_count),
                "max_retries": max_retries,
                "next_retry_count": (retry_count + 1) if retry_allowed else retry_count,
            },
            # lightweight hints your caller can log / inspect
            "llm_used": str(decision.get("source","")).startswith(("llm", "openai", "cloudflare")),
        }
