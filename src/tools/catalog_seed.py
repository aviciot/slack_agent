# src/tools/catalog_seed.py
from __future__ import annotations

import os
import sqlite3
import json
import logging
from typing import Dict, List, Any, Tuple

import numpy as np
import redis
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("catalog_seed")

# ---- ENV ----
REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
IDX_TABLES = os.getenv("IDX_TABLES", "idx:tables")
DB_NAME = os.getenv("SEED_DB_NAME", "informatica")  # logical db label: informatica|statements|billing
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")

OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL") or f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed(text: str) -> bytes:
    resp = requests.post(OLLAMA_EMBEDDING_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=(5, 30))
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"], dtype=np.float32).tobytes()
    return vec


def ensure_index(r: redis.Redis, dim: int):
    try:
        r.execute_command("FT.INFO", IDX_TABLES)
        return
    except Exception:
        pass

    r.execute_command(
        "FT.CREATE", IDX_TABLES,
        "ON", "HASH",
        "PREFIX", 1, "t:",
        "SCHEMA",
            "db_name", "TAG", "SEPARATOR", ",",
            "table", "TEXT",
            "columns", "TEXT",
            "description", "TEXT",
            "domain_tags", "TAG", "SEPARATOR", ",",
            "nl_desc_vector", "VECTOR", "HNSW", "10",
                "TYPE", "FLOAT32",
                "DIM", dim,
                "DISTANCE_METRIC", "COSINE",
                "M", "16",
                "EF_CONSTRUCTION", "200"
    )
    log.info(f"[CATALOG] created index {IDX_TABLES} with DIM={dim}")


def sqlite_tables(db_path: str) -> List[Tuple[str, List[str]]]:
    conn = sqlite3.connect(db_path, timeout=6)
    conn.row_factory = sqlite3.Row
    try:
        tables = []
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        for (tname,) in cur.fetchall():
            cols = []
            ccur = conn.execute(f"PRAGMA table_info('{tname}')")
            for row in ccur.fetchall():
                cols.append(row["name"])
            tables.append((tname, cols))
        return tables
    finally:
        conn.close()


def infer_tags(table: str, cols: List[str]) -> List[str]:
    t = table.lower()
    tags: List[str] = []
    if t.startswith("wf_") or "workflow" in t or any("workflow" in c.lower() for c in cols):
        tags.append("workflow")
        tags.append("etl")
    if t.startswith("ms_") or "statement" in t:
        tags.append("statements")
        tags.append("ms_pdf")
    if "gl" in t or "ledger" in t:
        tags.append("gl")
        tags.append("ledger")
    if "invoice" in t or "ar" in t:
        tags.append("ar")
        tags.append("invoice")
    return sorted(set(tags))


def seed_from_sqlite():
    r = redis.from_url(REDIS_URL, decode_responses=False)

    # Probe embedding dim
    dim = int(len(np.frombuffer(embed("dim probe"), dtype=np.float32)))
    ensure_index(r, dim)

    count = 0
    for table, cols in sqlite_tables(SQLITE_DB_PATH):
        description = f"{DB_NAME}.{table} columns: {', '.join(cols)}"
        text_blob = f"{DB_NAME} {table} {' '.join(cols)}"
        vec = embed(text_blob)
        key = f"t:{DB_NAME}:{table}"

        tags = infer_tags(table, cols)
        r.hset(key, mapping={
            "db_name": DB_NAME.encode("utf-8"),
            "table": table.encode("utf-8"),
            "columns": ", ".join(cols).encode("utf-8"),
            "description": description.encode("utf-8"),
            "domain_tags": ",".join(tags).encode("utf-8"),
            "nl_desc_vector": vec,
        })
        count += 1

    log.info(f"[CATALOG] Seeded {count} tables for db_name={DB_NAME} into {IDX_TABLES}")


if __name__ == "__main__":
    seed_from_sqlite()
