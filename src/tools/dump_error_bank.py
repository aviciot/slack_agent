#!/usr/bin/env python3
"""
Export all cached error bank entries from Redis into a YAML file.

Usage:
  docker compose exec api python /app/src/tools/dump_error_bank.py ./data/error_bank_snapshot.yaml
"""

import os
import sys
import redis
import json
import yaml

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
PREFIX = os.getenv("ERROR_BANK_PREFIX", "errorbank")


def dump_error_bank(out_path: str):
    r = redis.from_url(REDIS_URL, decode_responses=True)

    keys = r.keys(f"{PREFIX}:*")
    snapshot = {}

    for k in keys:
        try:
            val = json.loads(r.get(k))
        except Exception:
            continue
        # strip the prefix for nicer export
        sig = k.replace(f"{PREFIX}:", "")
        snapshot[sig] = val

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=True, allow_unicode=True)

    print(f"✅ Dumped {len(snapshot)} error bank entries to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: dump_error_bank.py <output.yaml>")
        sys.exit(1)
    out_path = sys.argv[1]
    dump_error_bank(out_path)
##docker compose exec api python /app/src/tools/dump_error_bank.py ./data/error_bank_snapshot.yaml
