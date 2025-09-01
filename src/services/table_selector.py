# src/services/table_selector.py
from __future__ import annotations
import os, json, sqlite3, logging, time, re
from typing import List, Dict, Any

import numpy as np
import redis
import requests

logger = logging.getLogger(__name__)

# -------- Env / Config --------
REDIS_URL      = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
DB_PATH        = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")

# Embeddings (same as elsewhere)
OLLAMA_HOST          = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL") or f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL          = os.getenv("EMBED_MODEL") or "nomic-embed-text"

# RediSearch index containing table catalog seeded by your tables seeder
IDX_TABLES     = os.getenv("TABLES_INDEX", "idx:tables")

# Selection knobs
TABLE_TOPK         = int(os.getenv("TABLE_TOPK", 3))          # how many to return to the LLM
TABLE_MAX_PASS     = int(os.getenv("TABLE_MAX_PASS", 2))      # max tables to include in LLM prompt
FUSION_MULT        = int(os.getenv("TABLE_FUSION_MULT", 4))   # widen KNN/BM25 candidate pools
MIN_CANDIDATES     = int(os.getenv("TABLE_MIN_CANDIDATES", 8))
RRF_K              = int(os.getenv("TABLE_RRF_K", 60))        # RRF constant

def _hval(b: Any) -> str:
    if b is None:
        return ""
    if isinstance(b, (bytes, bytearray)):
        return b.decode("utf-8", "ignore")
    return str(b)

class TableSelector:
    """
    Chooses the most relevant tables for a user question by combining:
      - semantic KNN over @nl_desc_vector
      - BM25 full-text over name/description/columns
    Then fuses results via Reciprocal Rank Fusion (RRF).
    Returns list of table names + compact schema text for LLM.
    """

    def __init__(self):
        self.r = redis.from_url(REDIS_URL, decode_responses=False)

    # ---------- Embedding ----------
    def embed(self, text: str) -> bytes:
        resp = requests.post(OLLAMA_EMBEDDING_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype=np.float32).tobytes()
        return vec

    # ---------- Build a safe text query ----------
    def _text_query(self, user_text: str) -> str:
        toks = re.findall(r"[A-Za-z0-9_]+", (user_text or "").lower())
        if not toks:
            return "*"
        # OR the tokens so BM25 can match any of them
        q = " | ".join(toks[:64])
        return f"(@name:({q})) | (@description:({q})) | (@columns:({q}))"

    # ---------- KNN over idx:tables ----------
    def knn_tables(self, vec: bytes, k: int) -> List[Dict[str, Any]]:
        res = self.r.execute_command(
            "FT.SEARCH", IDX_TABLES,
            f"*=>[KNN {k} @nl_desc_vector $vec AS score]",
            "PARAMS", 2, "vec", vec,
            "SORTBY", "score",
            "LIMIT", 0, k,
            "RETURN", 5, "name", "description", "ddl", "columns", "score",
            "DIALECT", 2
        )

        hits: List[Dict[str, Any]] = []
        total = res[0] if res else 0
        for i in range(1, len(res), 2):
            key = _hval(res[i])
            fields = res[i + 1]
            d: Dict[str, Any] = {"__key": key}
            for j in range(0, len(fields), 2):
                d[_hval(fields[j])] = _hval(fields[j + 1])
            try:
                d["__knn_dist"] = float(d.get("score", "9.99"))
            except Exception:
                d["__knn_dist"] = 9.99
            hits.append(d)

        if hits:
            logger.info(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "level": "INFO",
                "msg": "[TABLES KNN]",
                "total": total,
                "returned": len(hits),
                "top": hits[0].get("name", hits[0].get("__key")),
                "score": hits[0].get("__knn_dist"),
            }))
        else:
            logger.info("[TABLES KNN] no hits")
        return hits

    # ---------- BM25 text search over idx:tables ----------
    def text_tables(self, user_text: str, k: int) -> List[Dict[str, Any]]:
        query = self._text_query(user_text)
        res = self.r.execute_command(
            "FT.SEARCH", IDX_TABLES,
            query,
            "WITHSCORES",
            "RETURN", 4, "name", "description", "ddl", "columns",
            "LIMIT", 0, k,
            "DIALECT", 2
        )
        hits: List[Dict[str, Any]] = []
        total = res[0] if res else 0
        i = 1
        while i < len(res):
            key = _hval(res[i]); i += 1
            try:
                score = float(_hval(res[i])); i += 1
            except Exception:
                score = 0.0; i += 1
            fields = res[i]; i += 1
            d: Dict[str, Any] = {"__key": key, "__bm25": score}
            for j in range(0, len(fields), 2):
                d[_hval(fields[j])] = _hval(fields[j + 1])
            hits.append(d)
        return hits

    # ---------- SQLite schema snippet ----------
    def _pragma_schema(self, table: str) -> str:
        try:
            conn = sqlite3.connect(DB_PATH, timeout=6)
            try:
                cur = conn.execute(f"PRAGMA table_info('{table}')")
                cols = cur.fetchall()
            finally:
                conn.close()
        except Exception:
            cols = []

        if not cols:
            return f"TABLE {table};\n"

        cols_str = ", ".join([f"{c[1]} {c[2]}".strip() for c in cols])
        return f"TABLE {table} ({cols_str});\n"

    # ---------- Public API ----------
    def choose_tables(self, user_text: str) -> Dict[str, Any]:
        """
        Returns:
          { "tables": ["t1","t2"], "schema_text": "..." }
        """
        t0 = time.time()
        try:
            # widen candidate pools
            k_cands = max(TABLE_TOPK * FUSION_MULT, MIN_CANDIDATES)

            # get candidates
            vec = self.embed(user_text)
            knn = self.knn_tables(vec, k_cands)
            bm25 = self.text_tables(user_text, k_cands)

            # build rank maps
            knn_keys = [d["__key"] for d in knn]
            bm25_keys = [d["__key"] for d in bm25]
            knn_rank = {k: i+1 for i, k in enumerate(knn_keys)}     # 1-based
            bm_rank  = {k: i+1 for i, k in enumerate(bm25_keys)}

            # fuse via RRF
            union_keys = list(dict.fromkeys(knn_keys + bm25_keys))
            knn_by_key = {d["__key"]: d for d in knn}
            bm_by_key  = {d["__key"]: d for d in bm25}

            fused: List[Dict[str, Any]] = []
            for k in union_keys:
                r_knn = knn_rank.get(k)
                r_bm  = bm_rank.get(k)
                score = 0.0
                if r_knn is not None: score += 1.0 / (RRF_K + r_knn)
                if r_bm  is not None: score += 1.0 / (RRF_K + r_bm)
                row: Dict[str, Any] = {"__key": k, "__rrf": score}
                # merge fields from either list
                for src in (knn_by_key.get(k), bm_by_key.get(k)):
                    if src:
                        for f in ("name","description","ddl","columns"):
                            if f in src and (f not in row or not row[f]):
                                row[f] = src[f]
                if k in knn_by_key:
                    row["__knn_dist"] = knn_by_key[k].get("__knn_dist", 9.99)
                fused.append(row)

            fused.sort(key=lambda d: d["__rrf"], reverse=True)

            # --------- SELECT: top fused + top BM25, then fill up to TABLE_MAX_PASS ----------
            selected: List[Dict[str, Any]] = []
            fused_by_key = {d["__key"]: d for d in fused}

            def add_by_key(k: str):
                if not k:
                    return
                d = fused_by_key.get(k)
                if d and d not in selected:
                    selected.append(d)

            top_rrf_key = fused[0]["__key"] if fused else None
            top_bm_key  = bm25_keys[0] if bm25_keys else None

            add_by_key(top_bm_key)
            add_by_key(top_rrf_key)

            for d in fused:
                if len(selected) >= TABLE_MAX_PASS:
                    break
                if d not in selected:
                    selected.append(d)

            chosen = selected[:TABLE_MAX_PASS]
            # -------------------------------------------------------------------------------

            tables: List[str] = []
            blocks: List[str] = []

            for h in chosen:
                name = (h.get("name") or h.get("__key", "")).split(":")[-1]
                if not name:
                    continue
                desc = (h.get("description") or "").strip()
                ddl  = (h.get("ddl") or "").strip()

                tables.append(name)

                block = []
                if desc:
                    block.append(f"Description for table '{name}':\n{desc}\n")
                if ddl:
                    block.append(f"{ddl}\n")
                else:
                    block.append(self._pragma_schema(name))
                blocks.append("\n".join(block).rstrip() + "\n")

            schema_text = "\n".join(blocks).strip()
            took = int((time.time() - t0) * 1000)

            logger.info(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "level": "INFO",
                "msg": "[TABLES CHOOSE]",
                "tables": tables,
                "latency_ms": took
            }))

            return {"tables": tables, "schema_text": schema_text}
        except Exception as e:
            logger.exception(f"[TABLES] choose_tables failed: {e}")
            return {"tables": [], "schema_text": ""}
