from __future__ import annotations

import os
import json
import redis
from fastapi import APIRouter, Query
from typing import List, Dict, Any

router = APIRouter(prefix="/api/store/errorbank", tags=["Error Bank"])

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
r = redis.from_url(REDIS_URL, decode_responses=True)

PREFIX = os.getenv("ERROR_BANK_PREFIX", "errorbank")


@router.get("/search")
def search_errorbank(
    vendor: str = Query(None, description="Filter by vendor (ora, sqlite, pg, mysql, etc.)"),
    pattern: str = Query(None, description="Substring to filter signatures or reasons"),
    max_results: int = Query(20, ge=1, le=100),
) -> List[Dict[str, Any]]:
    """
    Search cached error decisions in Redis error bank.

    - Filter by vendor (ora/sqlite/pg/etc.)
    - Filter by substring (in signature or reason)
    - Returns action, category, reason, confidence, etc.
    """
    # Gather keys
    if vendor:
        keys = r.keys(f"{PREFIX}:{vendor}*")
    else:
        keys = r.keys(f"{PREFIX}:*")

    out = []
    for k in keys[:max_results]:
        try:
            val = json.loads(r.get(k))
        except Exception:
            continue
        sig = k.replace(f"{PREFIX}:", "")
        if pattern and pattern.lower() not in (sig.lower() + json.dumps(val).lower()):
            continue
        val["signature"] = sig
        out.append(val)

    return out
