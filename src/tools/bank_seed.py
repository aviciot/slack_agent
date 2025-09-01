# src/tools/bank_seed.py
from __future__ import annotations

import os
import csv
import json
import hashlib
import logging
from typing import Dict, List, Any

import numpy as np
import redis
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bank_seed")

# ---- ENV ----
REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
IDX_QCACHE = os.getenv("IDX_QCACHE", "idx:qcache")

OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL") or f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed(text: str) -> bytes:
    r = requests.post(OLLAMA_EMBEDDING_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=(5, 30))
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32).tobytes()


def ensure_index(r: redis.Redis, dim: int):
    try:
        r.execute_command("FT.INFO", IDX_QCACHE)
        return
    except Exception:
        pass

    r.execute_command(
        "FT.CREATE", IDX_QCACHE,
        "ON", "HASH",
        "PREFIX", 1, "q:",
        "SCHEMA",
            "db_name", "TAG", "SEPARATOR", ",",
            "signature", "TEXT",
            "sql_template", "TEXT",
            "nl_desc_vector", "VECTOR", "HNSW", "10",
                "TYPE", "FLOAT32",
                "DIM", dim,
                "DISTANCE_METRIC", "COSINE",
                "M", "16",
                "EF_CONSTRUCTION", "200"
    )
    log.info(f"[BANK] created index {IDX_QCACHE} with DIM={dim}")


def upsert_template(r: redis.Redis, *, db_name: str, signature: str, sql_template: str):
    dbv = db_name.strip().lower()
    vec = embed(signature)
    key = "q:" + hashlib.sha256(f"{dbv}|{signature}".encode("utf-8")).hexdigest()
    r.hset(key, mapping={
        "db_name": dbv.encode("utf-8"),
        "signature": signature.encode("utf-8"),
        "sql_template": sql_template.encode("utf-8"),
        "nl_desc_vector": vec,
    })
    return key


def seed_from_csv(csv_path: str):
    """
    CSV columns (header required):
      db_name,signature,sql_template
    """
    r = redis.from_url(REDIS_URL, decode_responses=False)

    # probe dim
    dim = int(len(np.frombuffer(embed("dim probe"), dtype=np.float32)))
    ensure_index(r, dim)

    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            db_name = (row.get("db_name") or "").strip()
            signature = (row.get("signature") or "").strip()
            sql_template = (row.get("sql_template") or "").strip()
            if not (db_name and signature and sql_template):
                log.warning(f"[BANK] skip row with missing fields: {row}")
                continue
            key = upsert_template(r, db_name=db_name, signature=signature, sql_template=sql_template)
            count += 1
            if count % 50 == 0:
                log.info(f"[BANK] seeded {count} templates... last={key}")

    log.info(f"[BANK] DONE. Seeded {count} templates into {IDX_QCACHE}")


if __name__ == "__main__":
    path = os.getenv("BANK_SEED_CSV", "./data/bank_templates.csv")
    if not os.path.exists(path):
        raise SystemExit(f"CSV not found: {path}\nExpected columns: db_name,signature,sql_template")
    seed_from_csv(path)
