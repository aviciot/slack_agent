# src/services/error_decider/llm_adapter.py
from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Any, Dict, Optional

import requests


def _normalize_action(s: str) -> str:
    s = (s or "").strip().upper()
    if s not in {"QUIT", "RETRY", "REPHRASE", "ASK_USER"}:
        return "REPHRASE"
    return s


def _normalize_category(s: str) -> str:
    s = (s or "").strip().lower()
    # light aliasing
    aliases = {
        "authentication": "auth",
        "authorization": "auth",
        "network": "connectivity",
        "schema_error": "schema",
        "syntax": "schema",
    }
    s = aliases.get(s, s)
    return s or "schema"


def _safe_str(obj: Any, max_len: int = 240) -> str:
    s = str(obj or "")
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


class LLMAdapter:
    """
    Calls Cloudflare Workers AI (or other providers later) to classify DB errors into:
      action:     QUIT | RETRY | REPHRASE | ASK_USER
      category:   auth | connectivity | schema | data | encoding | transient | other
      reason:     short human explanation
      confidence: float 0..1

    Return payload always includes: source="llm-cloudflare".
    """

    def __init__(self, *, provider: Optional[str], model: Optional[str]) -> None:
        self.provider = (provider or os.getenv("ERROR_DECIDER_PROVIDER") or "cloudflare").lower()
        self.model = model or os.getenv("ERROR_DECIDER_MODEL", "@cf/meta/llama-3-8b-instruct")

        # Tunables (deterministic by default)
        self.temperature = float(os.getenv("ERROR_DECIDER_TEMP", "0"))
        self.max_tokens = int(os.getenv("ERROR_DECIDER_MAXTOKENS", "128"))
        self.timeout_sec = int(os.getenv("ERROR_DECIDER_TIMEOUT_SEC", "15"))

        # Cloudflare creds
        self.cf_account = os.getenv("CF_ACCOUNT_ID")
        self.cf_token = os.getenv("CF_API_TOKEN")

    # ---------------- Public ----------------

    def decide(
        self,
        vendor: str,
        db_name: str,
        sql_redacted: str,
        error_text: str,
        *,
        retry_count: int = 0,
        exec_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return dict {action, category, reason, confidence, source}
        """
        start = time.time()
        try:
            raw = self._call_cloudflare(
                vendor=vendor,
                db_name=db_name,
                sql_redacted=_safe_str(sql_redacted, 160),
                error_text=_safe_str(error_text, 850),
                retry_count=retry_count,
                exec_phase=exec_phase or "execute",
            )
            parsed = self._postprocess(raw)
        except Exception as e:
            # Hard fallback: safe default
            parsed = {
                "action": "REPHRASE",
                "category": "schema",
                "reason": f"LLM fallback ({type(e).__name__})",
                "confidence": 0.5,
                "source": "llm-cloudflare",
            }

        # Attach standard source tag and cap confidence
        parsed["source"] = "llm-cloudflare"
        try:
            c = float(parsed.get("confidence", 0.7))
        except Exception:
            c = 0.7
        parsed["confidence"] = max(0.0, min(1.0, c))

        # Shape final output
        return {
            "action": _normalize_action(parsed.get("action")),
            "category": _normalize_category(parsed.get("category")),
            "reason": parsed.get("reason", "LLM decision").strip()[:240],
            "confidence": parsed["confidence"],
            "source": "llm-cloudflare",
        }

    # ---------------- Internals ----------------

    def _call_cloudflare(
        self,
        *,
        vendor: str,
        db_name: str,
        sql_redacted: str,
        error_text: str,
        retry_count: int,
        exec_phase: str,
    ) -> str:
        if self.provider != "cloudflare":
            raise RuntimeError(f"Unsupported provider: {self.provider}")

        if not (self.cf_account and self.cf_token):
            raise RuntimeError("Cloudflare credentials missing (CF_ACCOUNT_ID / CF_API_TOKEN)")

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.cf_account}/ai/run/{self.model}"
        headers = {"Authorization": f"Bearer {self.cf_token}", "Content-Type": "application/json"}

        system = (
            "You are a reliable classifier for database error handling. "
            "Always return STRICT JSON with keys: action, category, reason, confidence. "
            "No prose, no markdown, only a single JSON object."
        )

        user = {
            "task": "Classify the database error into an action and category.",
            "schema": {
                "action": "QUIT|RETRY|REPHRASE|ASK_USER",
                "category": "auth|connectivity|schema|data|encoding|transient|other",
                "reason": "short explanation (<= 180 chars)",
                "confidence": "0.0..1.0 (float)",
            },
            "hints": [
                "auth → bad credentials or permission denied",
                "connectivity → network/listener/dns/timeouts",
                "schema → syntax errors, invalid identifiers, missing objects",
                "data → constraint violations, type/cast errors",
                "encoding → invalid byte sequence / character set errors",
                "transient → deadlocks, timeouts, locked, snapshot too old (retryable)",
            ],
            "context": {
                "vendor": vendor,
                "db_name": db_name,
                "exec_phase": exec_phase,
                "retry_count": retry_count,
                "sql_redacted_prefix": sql_redacted,
                "error_text": error_text,
            },
            "constraints": [
                "If credentials/permission-related → action=QUIT, category=auth",
                "If dropped connection / DNS / listener → action=QUIT, category=connectivity",
                "If deadlock/locked/snapshot-too-old → action=RETRY, category=transient",
                "If missing table/view/column or syntax → action=REPHRASE, category=schema",
                "If unique/PK/NOT NULL/type-cast → action=ASK_USER or REPHRASE, category=data",
                "Never invent SQL. Do not return anything other than JSON.",
            ],
        }

        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            # Some CF models accept params under top-level. If rejected, remove these two lines.
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(f"CF HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"CF API error: {data.get('errors') or data.get('messages')}")
        # CF returns {"result":{"response":"..."}}
        result = data.get("result") or {}
        return result.get("response") or ""

    def _postprocess(self, raw: str) -> Dict[str, Any]:
        """
        'raw' is expected to be a JSON string like:
          {"action":"REPHRASE","category":"schema","reason":"...","confidence":0.7}

        If the model returned quoted JSON, parse twice. If anything goes wrong,
        return a safe, normalized object.
        """
        s = (raw or "").strip()
        if not s:
            return {"action": "REPHRASE", "category": "schema", "reason": "Empty LLM response", "confidence": 0.5}

        # Some CF models return JSON-as-string
        try:
            obj = json.loads(s)
        except Exception:
            # maybe it's quoted JSON
            try:
                obj = json.loads(json.loads(s))
            except Exception:
                return {"action": "REPHRASE", "category": "schema", "reason": "Unparseable LLM JSON", "confidence": 0.5}

        if not isinstance(obj, dict):
            return {"action": "REPHRASE", "category": "schema", "reason": "Non-object LLM JSON", "confidence": 0.5}

        # Normalize fields
        action = _normalize_action(obj.get("action"))
        category = _normalize_category(obj.get("category"))
        reason = _safe_str(obj.get("reason", "LLM decision"), 180)
        try:
            conf = float(obj.get("confidence", 0.7))
        except Exception:
            conf = 0.7

        return {"action": action, "category": category, "reason": reason, "confidence": conf}
