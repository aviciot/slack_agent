# src/tools/seed_error_bank.py

"""
Full-reset Error Bank seeder.

Behavior (every run):
  1) Drop RediSearch index 'idx:errorbank' (ignore if missing).
  2) Delete ALL keys under ERROR_BANK_PREFIX:* (codes, regex lists, signatures, sentinels).
  3) Load YAML from ERROR_HANDLING_YAML.
  4) Seed:
       - Code entries:   {prefix}:{vendor_alias}:{code}
       - Regex entries:  {prefix}:regex:{vendor_alias}
       - Sentinel:       {prefix}:seed:version
  5) Print a compact JSON summary.
"""

from __future__ import annotations
import json, os, re, sys, hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

try:
    import yaml
except Exception:
    print("ERROR: PyYAML is required. Install in container:\n  docker compose exec api pip install pyyaml", file=sys.stderr)
    raise

try:
    import redis
except Exception:
    print("ERROR: redis-py is required. Install in container:\n  docker compose exec api pip install redis", file=sys.stderr)
    raise

ISO = "%Y-%m-%dT%H:%M:%SZ"

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

def vendor_alias(v: str) -> str:
    return {
        "oracle": "ora", "ora": "ora",
        "postgres": "pg", "postgresql": "pg", "pg": "pg",
        "sqlite": "sqlite",
        "snowflake": "snowflake",
        "any": "any",
    }.get((v or "").strip().lower(), v or "any")

def norm_category(cat: str) -> str:
    cat = (cat or "").strip().lower()
    aliases = {"authentication": "auth", "authorization": "auth", "network": "connectivity"}
    return aliases.get(cat, cat or "other")

def norm_code(vendor: str, code: str) -> str:
    if not code:
        return "unknown"
    v = vendor_alias(vendor)
    c = str(code).strip()
    if v == "ora":
        m = re.search(r"(\d{4,5})", c)
        return (m.group(1) if m else c).lower()
    return c.lower().replace(" ", "").replace(":", "").replace("-", "_")

def make_code_key(prefix: str, vendor: str, code: str) -> str:
    return f"{prefix}:{vendor_alias(vendor)}:{norm_code(vendor, code)}"

def regex_key(prefix: str, vendor: str) -> str:
    return f"{prefix}:regex:{vendor_alias(vendor)}"

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR_HANDLING_YAML not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def iter_code_rows(y: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    vendors = ((y.get("error_handling") or {}).get("vendors") or {})
    for vendor, spec in vendors.items():
        codes = (spec or {}).get("error_codes") or {}
        for raw_code, cfg in codes.items():
            if not isinstance(cfg, dict):
                continue
            action = (cfg.get("action") or "").strip().upper()
            category = norm_category(cfg.get("category") or "")
            reason = (cfg.get("reason") or "").strip()
            if action not in {"QUIT", "RETRY", "REPHRASE", "ASK_USER"}:
                continue
            if not category:
                continue
            out.append((vendor, str(raw_code), {"action": action, "category": category, "reason": reason}))
    return out

def iter_regex_rows(y: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    vendors = ((y.get("error_handling") or {}).get("vendors") or {})
    for vendor, spec in vendors.items():
        rules = (spec or {}).get("regex_rules") or []
        for r in rules:
            pat = r.get("pattern")
            action = (r.get("action") or "").strip().upper()
            category = norm_category(r.get("category") or "")
            reason = (r.get("reason") or "Matched regex").strip()
            if not pat or action not in {"QUIT", "RETRY", "REPHRASE", "ASK_USER"} or not category:
                continue
            try:
                re.compile(pat, flags=re.IGNORECASE | re.DOTALL)
            except re.error:
                continue
            out.append((vendor, {"pattern": pat, "action": action, "category": category, "reason": reason}))
    return out

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def drop_index_if_exists(r: "redis.Redis") -> None:
    try:
        r.execute_command("FT.DROPINDEX", "idx:errorbank", "DD")
        print("[reset] dropped RediSearch index idx:errorbank")
    except Exception:
        print("[reset] no RediSearch index to drop (idx:errorbank)")

def delete_all_under_prefix(r: "redis.Redis", prefix: str) -> int:
    pat = f"{prefix}:*"
    keys = list(r.scan_iter(pat))
    if not keys:
        print(f"[reset] no keys under {pat}")
        return 0
    pipe = r.pipeline(transaction=False)
    for key in keys:
        pipe.delete(key)
    pipe.execute()
    print(f"[reset] deleted keys under {pat}: {len(keys)}")
    return len(keys)


def main() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    prefix = os.getenv("ERROR_BANK_PREFIX", "errorbank")

    args = sys.argv[1:]
    mode = args[0] if args else "seed"

    # decide which file to load
    if mode == "seed":
        yaml_path = "/app/config/error_bank/error_handling.yaml"
    elif mode == "snapshot":
        yaml_path = "./data/error_bank_snapshot.yaml"
    else:
        print("Usage: seed_error_bank.py [seed|snapshot]")
        sys.exit(1)

    r = redis.from_url(redis_url, decode_responses=True)

    drop_index_if_exists(r)
    delete_all_under_prefix(r, prefix)

    y = load_yaml(yaml_path)

    if mode == "seed":
        # base rules seeding (as before) ...
        code_rows = iter_code_rows(y)
        regex_rows = iter_regex_rows(y)
        # <your existing seeding logic>
        print(json.dumps({"ok": True, "mode": "base_rules", "codes": len(code_rows), "regex": len(regex_rows)}))
    elif mode == "snapshot":
        count = 0
        for sig, val in y.items():
            # Case 1: cached error entries (dict)
            if isinstance(val, dict):
                r.set(f"{prefix}:{sig}", json.dumps(val))
                count += 1

            # Case 2: regex rules (list of dicts)
            elif isinstance(val, list):
                r.set(f"{prefix}:{sig}", json.dumps(val))
                count += len(val)

        print(json.dumps({"ok": True, "mode": "snapshot", "entries": count}))




if __name__ == "__main__":
    main()

# # seed from base rules
# docker compose exec api python /app/src/tools/seed_error_bank.py seed

# # seed from snapshot
# docker compose exec api python /app/src/tools/seed_error_bank.py snapshot

