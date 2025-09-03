from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from .llm_adapter import LLMAdapter

from pathlib import Path
import yaml


try:
    import redis
except Exception as e:
    raise RuntimeError(
        "redis-py is required. Install in the container:\n"
        "  docker compose exec api pip install redis"
    ) from e


# -----------------------------
# Helpers / normalization
# -----------------------------
_ALIAS = {
    "oracle": "ora",
    "ora": "ora",
    "postgres": "pg",
    "postgresql": "pg",
    "pg": "pg",
    "sqlite": "sqlite",
    "snowflake": "snowflake",
    "any": "any",
}

def _normalize_err_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _msg_sig(error_text: str) -> str:
    # keep length consistent with what you used when writing
    return _sha1(_normalize_err_text(error_text))[:12]

def _make_msg_key(prefix: str, vendor: str, msg_sig: str) -> str:
    return f"{prefix}:{_vendor_alias(vendor)}:msg:{msg_sig}"


def _vendor_alias(vendor: str) -> str:
    v = (vendor or "").strip().lower()
    return _ALIAS.get(v, v or "any")

def _extract_code(vendor: str, error_text: str) -> Optional[str]:
    """Try to extract a vendor-native error code (e.g., '00942' from ORA-00942)."""
    v = _vendor_alias(vendor)
    et = error_text or ""
    if v == "ora":
        m = re.search(r"ORA-(\d{4,5})", et, flags=re.IGNORECASE)
        return m.group(1) if m else None
    if v == "pg":
        m = re.search(r"SQLSTATE\s+([0-9A-Z]{5})", et, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None
    if v == "snowflake":
        m = re.search(r"\(SQLSTATE\s+([0-9A-Z]+)\)", et, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None
    # sqlite & generic usually text-only
    return None

def _make_code_key(prefix: str, vendor: str, code: str) -> str:
    return f"{prefix}:{_vendor_alias(vendor)}:{code.lower()}"

def _regex_key(prefix: str, vendor: str) -> str:
    return f"{prefix}:regex:{_vendor_alias(vendor)}"

def _redact_sql_literals(sql: str) -> str:
    if not sql:
        return ""
    s = re.sub(r"'[^']*'", "'***'", sql)     # redact strings
    s = re.sub(r"\b\d+\b", "N", s)           # redact integers
    return s

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -----------------------------
# ErrorDecider (Redis + policy)
# -----------------------------
class ErrorDecider:
    """
    Precedence:
      1) force_quit_codes (per-DB) → QUIT
      2) bank(code)                → return cached
      3) regex (from Redis)        → return match
      4) LLM (stub/classifier)     → return + (optionally) cache
      5) fallback                  → REPHRASE

    Output Schema:
      {action, category, reason, confidence, source, error_signature}
    """

    def __init__(
        self,
        *,
        databases_path: str = "/app/config/databases.json",
        redis_url: Optional[str] = None,
        bank_prefix: Optional[str] = None,
        bank_ttl_days: Optional[int] = None,
    ):
        # Redis + bank config
        self.redis = redis.from_url(redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
        self.prefix = bank_prefix or os.getenv("ERROR_BANK_PREFIX", "errorbank")
        days = bank_ttl_days if bank_ttl_days is not None else int(os.getenv("ERROR_BANK_TTL_DAYS", "0"))
        self.ttl_seconds = 0 if days <= 0 else days * 24 * 3600

        self.bank_write = os.getenv("ERROR_BANK_WRITE", "1") == "1"

        # Global LLM settings (selection only; we use a stub here)
        self.global_mode = (os.getenv("ERROR_DECIDER_MODE", "hybrid") or "hybrid").lower()  # rule|llm|hybrid
        self.global_provider = os.getenv("ERROR_DECIDER_PROVIDER", "cloudflare")
        self.global_model = os.getenv("ERROR_DECIDER_MODEL", "gpt-4.1-mini")
        self.allow_rephrase_default = os.getenv("ERROR_DECIDER_ALLOW_REPHRASE_DEFAULT", "1") == "1"

        # Policies per DB
        self.databases_path = databases_path
        self._policies = self._load_db_policies()

    # -------- Policies --------
    def _load_db_policies(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.databases_path, "r", encoding="utf-8") as f:
                arr = json.load(f)
        except Exception:
            arr = []
        out: Dict[str, Dict[str, Any]] = {}
        for x in arr or []:
            name = str(x.get("name", "")).strip()
            pol = (x.get("error_decider") or {}) if isinstance(x.get("error_decider"), dict) else {}
            if not name:
                continue
            out[name] = {
                "mode": (pol.get("mode") or self.global_mode),
                "provider": pol.get("provider") or self.global_provider,
                "model": pol.get("model") or self.global_model,
                "allow_rephrase": bool(pol.get("allow_rephrase", self.allow_rephrase_default)),
                "max_retries": int(pol.get("max_retries", 0)),
                "force_quit_codes": pol.get("force_quit_codes", []),
            }
        return out

    def _policy_for(self, db_name: str) -> Dict[str, Any]:
        return self._policies.get(db_name, {
            "mode": self.global_mode,
            "provider": self.global_provider,
            "model": self.global_model,
            "allow_rephrase": self.allow_rephrase_default,
            "max_retries": 0,
            "force_quit_codes": [],
        })

    # -------- Force-quit --------
    def _force_quit_hit(self, codes: List[str], vendor: str, error_text: str) -> Optional[str]:
        if not codes:
            return None
        et_up = (error_text or "").upper()
        v = _vendor_alias(vendor)
        for c in codes:
            token = str(c).upper().strip()
            if not token:
                continue
            if token in et_up:
                return c
            # Oracle convenience: allow plain digits in policy
            if v == "ora" and re.search(rf"\bORA[-\s]?{re.escape(token)}\b", et_up):
                return c
        return None

    # -------- Bank (code) --------
    def _bank_get(self, vendor: str, code: str) -> Optional[Dict[str, Any]]:
        key = _make_code_key(self.prefix, vendor, code)
        val = self.redis.get(key)
        if not val:
            return None
        try:
            obj = json.loads(val)
        except Exception:
            return None
        # touch metadata (non-critical)
        obj["source"] = "bank"
        obj["error_signature"] = obj.get("error_signature", f"{_vendor_alias(vendor)}:{code}")
        try:
            obj["hits"] = int(obj.get("hits", 0)) + 1
            obj["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if self.bank_write:
                self.redis.set(key, json.dumps(obj))
                if self.ttl_seconds > 0:
                    self.redis.expire(key, self.ttl_seconds)
        except Exception:
            pass
        # return schema subset
        return {
            "action": obj.get("action", "REPHRASE"),
            "category": obj.get("category", "schema"),
            "reason": obj.get("reason", "Bank decision"),
            "confidence": float(obj.get("confidence", 0.95)),
            "source": "bank",
            "error_signature": obj.get("error_signature", f"{_vendor_alias(vendor)}:{code}"),
        }

    def _bank_put(self, vendor: str, code: Optional[str], payload: Dict[str, Any], *, error_text: str = "") -> None:
        if not self.bank_write:
            return
        if code:
            key = _make_code_key(self.prefix, vendor, code)
        else:
            # fallback: use a hash of the full error_text
            sig = _sha1(error_text or "")
            key = f"{self.prefix}:{_vendor_alias(vendor)}:msg:{sig[:12]}"
        try:
            self.redis.set(key, json.dumps(payload))
            if self.ttl_seconds > 0:
                self.redis.expire(key, self.ttl_seconds)
        except Exception:
            pass


    # -------- Regex (from Redis) --------
    def _regex_decide(self, vendor: str, error_text: str) -> Optional[Dict[str, Any]]:
        et = error_text or ""
        for bucket_vendor in (_vendor_alias(vendor), "any"):
            k = _regex_key(self.prefix, bucket_vendor)
            raw = self.redis.get(k)
            if not raw:
                continue
            try:
                rules = json.loads(raw)
            except Exception:
                continue
            for r in rules or []:
                pat = r.get("pattern")
                if not pat:
                    continue
                try:
                    cre = re.compile(pat, flags=re.IGNORECASE | re.DOTALL)
                except re.error:
                    continue
                if cre.search(et):
                    sig = f"{_vendor_alias(vendor)}:regex:{_sha1(pat)[:8]}"
                    return {
                        "action": r.get("action", "REPHRASE"),
                        "category": r.get("category", "schema"),
                        "reason": r.get("reason", "Matched regex"),
                        "confidence": 0.85,
                        "source": "regex",
                        "error_signature": sig,
                    }
        return None

    # -------- LLM (stub) --------
    def _llm_decide_stub(self, *, vendor: str, db_name: str, sql_redacted: str, error_text: str) -> Dict[str, Any]:
        et = (error_text or "").lower()
        # very small heuristic to stand in for the real LLM call
        if "ora-01017" in et or "invalid password" in et or "authentication" in et or "permission denied" in et:
            return {"action": "QUIT", "category": "auth", "reason": "Heuristic auth", "confidence": 0.75, "source": "llm-stub"}
        if any(x in et for x in ["could not connect", "listener", "tns", "refused", "timed out", "dns"]):
            return {"action": "QUIT", "category": "network", "reason": "Heuristic connectivity", "confidence": 0.7, "source": "llm-stub"}
        if any(x in et for x in ["database is locked", "deadlock", "busy", "snapshot too old", "serialization"]):
            return {"action": "RETRY", "category": "transient", "reason": "Heuristic transient", "confidence": 0.65, "source": "llm-stub"}
        if any(x in et for x in ["no such table", "does not exist", "invalid identifier", "syntax error"]):
            return {"action": "REPHRASE", "category": "schema", "reason": "Heuristic schema/syntax", "confidence": 0.7, "source": "llm-stub"}
        if any(x in et for x in ["unique constraint", "constraint failed", "not null", "invalid number"]):
            return {"action": "ASK_USER", "category": "data", "reason": "Heuristic data", "confidence": 0.6, "source": "llm-stub"}
        return {"action": "REPHRASE", "category": "schema", "reason": "Heuristic fallback", "confidence": 0.5, "source": "llm-stub"}


    def _bank_get_msg(self, vendor: str, error_text: str) -> Optional[Dict[str, Any]]:
        sig = _msg_sig(error_text)
        key = _make_msg_key(self.prefix, vendor, sig)
        val = self.redis.get(key)
        if not val:
            return None
        try:
            obj = json.loads(val)
        except Exception:
            return None

        # touch metadata (best-effort)
        obj["source"] = obj.get("source", "bank")
        obj["error_signature"] = obj.get("error_signature", f"{_vendor_alias(vendor)}:msg:{sig}")
        try:
            obj["hits"] = int(obj.get("hits", 0)) + 1
            obj["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if self.bank_write:
                self.redis.set(key, json.dumps(obj))
                if self.ttl_seconds > 0:
                    self.redis.expire(key, self.ttl_seconds)
        except Exception:
            pass

        # return schema subset
        return {
            "action": obj.get("action", "REPHRASE"),
            "category": obj.get("category", "schema"),
            "reason": obj.get("reason", "Bank decision"),
            "confidence": float(obj.get("confidence", 0.95)),
            "source": "bank",
            "error_signature": obj.get("error_signature", f"{_vendor_alias(vendor)}:msg:{sig}"),
        }

    # -------- Public API --------
    def decide(
        self,
        *,
        db_vendor: str,
        db_name: str,
        sql: str,
        error_text: str,
        retry_count: int = 0,
        exec_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        policy = self._policy_for(db_name)
        v = _vendor_alias(db_vendor)
        sql_red = _redact_sql_literals(sql)
        code = _extract_code(db_vendor, error_text)

        # 1) force_quit (per DB)
        fq = self._force_quit_hit(policy.get("force_quit_codes"), db_vendor, error_text)
        if fq:
            return {
                "action": "QUIT",
                "category": "auth" if "01017" in fq else "connectivity",
                "reason": f"Force-quit policy ({fq})",
                "confidence": 1.0,
                "source": "force_quit",
                "error_signature": f"{v}:{(code or fq).lower()}",
            }

        # 2) bank by code
        if code:
            hit = self._bank_get(db_vendor, code)
            if hit:
                return hit
        
        # 2b) bank by message signature (for vendors without codes, e.g., sqlite)

        hit_msg = self._bank_get_msg(db_vendor, error_text)
        if hit_msg:
            return hit_msg
        # 3) regex from Redis
        rx = self._regex_decide(db_vendor, error_text)
        if rx:
            return rx

        # 3.5) Oracle: if an ORA-xxxxx code exists but is unknown to bank/regex → force LLM
        if v == "ora":
            # If the message has an ORA-XXXXX code and we got here (no bank/regex), ask the LLM
            if re.search(r"\bORA-\d{5}\b", error_text or "", flags=re.IGNORECASE):
                mode = (policy.get("mode") or self.global_mode)
                if mode in ("hybrid", "llm"):
                    adapter = LLMAdapter(provider=policy.get("provider"), model=policy.get("model"))
                    llm = adapter.decide(db_vendor, db_name, sql_red, error_text)
                    # bank if we extracted a code
                    if (code or error_text) and self.bank_write:
                        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        sig = f"{v}:{code}" if code else f"{v}:msg:{_sha1(error_text)[:12]}"
                        payload = {
                            "action": llm["action"],
                            "category": llm["category"],
                            "reason": llm.get("reason", "LLM decision"),
                            "confidence": llm.get("confidence", 0.7),
                            "source": llm.get("source", "llm"),
                            "error_signature": sig,
                            "created_at": now,
                            "last_seen": now,
                            "hits": 1,
                            "prompt_version": "v1",
                        }
                        self._bank_put(db_vendor, code, payload, error_text=error_text)

                    return {**llm, "error_signature": sig}



        # 4) LLM (stub) if allowed
     

        mode = (policy.get("mode") or self.global_mode)
        if mode in ("hybrid", "llm"):
            adapter = LLMAdapter(provider=policy.get("provider"), model=policy.get("model"))
            llm = adapter.decide(db_vendor, db_name, sql_red, error_text)
            # bank the outcome if we have a code
            if (code or error_text) and self.bank_write:
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                sig = f"{v}:{code}" if code else f"{v}:msg:{_sha1(error_text)[:12]}"
                payload = {
                    "action": llm["action"],
                    "category": llm["category"],
                    "reason": llm.get("reason", "LLM decision"),
                    "confidence": llm.get("confidence", 0.7),
                    "source": llm.get("source", "llm"),
                    "error_signature": sig,
                    "created_at": now,
                    "last_seen": now,
                    "hits": 1,
                    "prompt_version": "v1",
                }
                self._bank_put(db_vendor, code, payload, error_text=error_text)

            return {**llm, "error_signature": sig}



        # 5) fallback
        return {
            "action": "REPHRASE",
            "category": "schema",
            "reason": "Default fallback",
            "confidence": 0.5,
            "source": "fallback",
            "error_signature": f"{v}:{code or 'unknown'}",
        }
