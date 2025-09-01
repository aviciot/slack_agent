# src/services/qcache.py
from __future__ import annotations
import json, hashlib, time
from typing import Optional, Dict, Any, Tuple
import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Key space:
#  - idx:qcache : FT/BM25 index (already exists)
#  - qcache:{id} : HASH per entry { question, normalized, route, dialect, tables(json), sql, meta(json), ts }

def _r():
    return redis.from_url(REDIS_URL, decode_responses=True)

def _qid(question: str, route: str, dialect: str) -> str:
    norm = " ".join(question.split()).lower()
    h = hashlib.sha256(f"{norm}|{route}|{dialect}".encode("utf-8")).hexdigest()[:16]
    return h

def find(question: str, route: str, dialect: str) -> Optional[Dict[str, Any]]:
    r = _r()
    qid = _qid(question, route, dialect)
    key = f"qcache:{qid}"
    if not r.exists(key):
        return None
    doc = r.hgetall(key)
    # inflate json fields
    if "tables" in doc and isinstance(doc["tables"], str):
        try: doc["tables"] = json.loads(doc["tables"])
        except: doc["tables"] = []
    if "meta" in doc and isinstance(doc["meta"], str):
        try: doc["meta"] = json.loads(doc["meta"])
        except: doc["meta"] = {}
    return doc

def store(question: str, route: str, dialect: str, tables, sql: str, meta: Dict[str, Any] = None) -> str:
    r = _r()
    qid = _qid(question, route, dialect)
    key = f"qcache:{qid}"
    payload = {
        "qid": qid,
        "question": question,
        "normalized": " ".join(question.split()).lower(),
        "route": route,
        "dialect": dialect,
        "tables": json.dumps(tables or []),
        "sql": sql,
        "meta": json.dumps(meta or {}),
        "ts": str(int(time.time()))
    }
    r.hset(key, mapping=payload)
    # optional: maintain a lightweight searchable idx doc
    r.set(f"idx:qcache:{qid}", payload["normalized"])
    return qid
