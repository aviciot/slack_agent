#!/usr/bin/env python3
from __future__ import annotations

import os, json, re
from typing import Any, Dict, Optional

class LLMAdapter:
    """
    decide(vendor, db_name, sql_redacted, error_text) -> {action, category, reason, confidence, source}

    Providers:
      - stub        : offline heuristic (default if no provider creds)
      - cloudflare  : uses Cloudflare Workers AI
      - openai      : uses OpenAI Chat Completions

    Env:
      ERROR_DECIDER_PROVIDER = stub | cloudflare | openai
      ERROR_DECIDER_MODEL    = <model slug> (e.g., gpt-4.1-mini, @cf/llama-3.1-8b-instruct)
      ERROR_DECIDER_TIMEOUT_S= 8
      CF_API_TOKEN           = <token>
      CF_ACCOUNT_ID          = <id>
      OPENAI_API_KEY         = <key>
    """
    def __init__(self,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout_s: Optional[int] = None):
        self.provider = (provider or os.getenv("ERROR_DECIDER_PROVIDER", "stub")).strip().lower()
        self.model = model or os.getenv("ERROR_DECIDER_MODEL", "gpt-4.1-mini")
        self.timeout_s = int(timeout_s if timeout_s is not None else os.getenv("ERROR_DECIDER_TIMEOUT_S", "8"))

        self._cf_token = os.getenv("CF_API_TOKEN")
        self._cf_account = os.getenv("CF_ACCOUNT_ID")
        self._openai_key = os.getenv("OPENAI_API_KEY")

        # auto-downgrade to stub if creds missing
        if self.provider == "cloudflare" and not (self._cf_token and self._cf_account):
            self.provider = "stub"
        if self.provider == "openai" and not self._openai_key:
            self.provider = "stub"

    # ---------------- Heuristic stub ----------------
    def _stub_decide(self, vendor: str, db_name: str, sql_redacted: str, error_text: str) -> Dict[str, Any]:
        et = (error_text or "").lower()
        if "ora-01017" in et or "invalid password" in et or "authentication" in et or "permission denied" in et:
            return {"action": "QUIT", "category": "auth", "reason": "Heuristic auth", "confidence": 0.75, "source": "llm-stub"}
        if any(x in et for x in ["could not connect", "listener", "tns", "refused", "timed out", "dns"]):
            return {"action": "QUIT", "category": "connectivity", "reason": "Heuristic connectivity", "confidence": 0.7, "source": "llm-stub"}
        if any(x in et for x in ["database is locked", "deadlock", "busy", "snapshot too old", "serialization"]):
            return {"action": "RETRY", "category": "transient", "reason": "Heuristic transient", "confidence": 0.65, "source": "llm-stub"}
        if any(x in et for x in ["no such table", "does not exist", "invalid identifier", "syntax error"]):
            return {"action": "REPHRASE", "category": "schema", "reason": "Heuristic schema/syntax", "confidence": 0.7, "source": "llm-stub"}
        if any(x in et for x in ["unique constraint", "constraint failed", "not null", "invalid number"]):
            return {"action": "ASK_USER", "category": "data", "reason": "Heuristic data", "confidence": 0.6, "source": "llm-stub"}
        return {"action": "REPHRASE", "category": "schema", "reason": "Heuristic fallback", "confidence": 0.5, "source": "llm-stub"}

    # ---------------- Cloudflare Workers AI ----------------
    def _cf_decide(self, vendor: str, db_name: str, sql_redacted: str, error_text: str) -> Dict[str, Any]:
        import requests
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._cf_account}/ai/run/{self.model}"
        prompt = self._prompt(vendor, db_name, sql_redacted, error_text)
        headers = {"Authorization": f"Bearer {self._cf_token}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"messages": prompt}, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        text = self._first_text(data)
        return self._parse_json_decision(text, provider="cloudflare")

    # ---------------- OpenAI ----------------
    def _openai_decide(self, vendor: str, db_name: str, sql_redacted: str, error_text: str) -> Dict[str, Any]:
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        prompt = self._prompt(vendor, db_name, sql_redacted, error_text)
        body = {"model": self.model, "messages": prompt, "temperature": 0, "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self._openai_key}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return self._parse_json_decision(text, provider="openai")

    # ---------------- Helpers ----------------
    def _first_text(self, data: Dict[str, Any]) -> str:
        try:
            return data["result"]["response"][0]["content"][0]["text"]
        except Exception:
            pass
        try:
            return data["result"]["message"]["content"][0]["text"]
        except Exception:
            pass
        try:
            return data["result"]["output_text"]
        except Exception:
            pass
        return json.dumps(data)

    def _prompt(self, vendor: str, db_name: str, sql_redacted: str, error_text: str) -> Any:
        system = (
            "You are an error decider for database query agents. "
            "Return ONLY a compact JSON object with keys: action, category, reason, confidence. "
            "Valid actions: QUIT, RETRY, REPHRASE, ASK_USER. "
            "Categories: auth, connectivity, syntax, schema, transient, data. "
            "Be conservative; prefer QUIT for auth/connectivity, RETRY for transient, REPHRASE for schema/syntax."
        )
        user = f"db_vendor={vendor}\ndb_name={db_name}\nsql_redacted={sql_redacted}\nerror_text={error_text}\n"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _parse_json_decision(self, text: str, provider: str) -> Dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.DOTALL)
        try:
            obj = json.loads(cleaned)
        except Exception:
            return {"action": "REPHRASE", "category": "schema", "reason": "LLM parse failure", "confidence": 0.5, "source": f"llm-{provider}-fallback"}
        action = str(obj.get("action", "REPHRASE")).upper()
        category = str(obj.get("category", "schema")).lower()
        reason = obj.get("reason", "LLM decision")
        conf = float(obj.get("confidence", 0.7))
        if action not in {"QUIT", "RETRY", "REPHRASE", "ASK_USER"}:
            action = "REPHRASE"
        return {"action": action, "category": category, "reason": reason, "confidence": conf, "source": f"llm-{provider}"}

    # ---------------- Public API ----------------
    def decide(self, vendor: str, db_name: str, sql_redacted: str, error_text: str) -> Dict[str, Any]:
        if self.provider == "cloudflare":
            try:
                return self._cf_decide(vendor, db_name, sql_redacted, error_text)
            except Exception:
                return self._stub_decide(vendor, db_name, sql_redacted, error_text)
        if self.provider == "openai":
            try:
                return self._openai_decide(vendor, db_name, sql_redacted, error_text)
            except Exception:
                return self._stub_decide(vendor, db_name, sql_redacted, error_text)
        return self._stub_decide(vendor, db_name, sql_redacted, error_text)
