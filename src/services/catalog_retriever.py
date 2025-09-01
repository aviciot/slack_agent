# src/services/catalog_retriever.py
from __future__ import annotations

import os, re, logging, json, requests, numpy as np, redis
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# ---- ENV / Config ----
REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")

# Primary index name; may be missing in some envs
IDX_TABLES_ENV = os.getenv("IDX_TABLES", "idx:tables")
# Comma-separated fallbacks (e.g. "idx:tables_v1,idx:tables_dev")
IDX_TABLES_FALLBACK = os.getenv("IDX_TABLES_FALLBACK", "idx:tables")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL") or f"{OLLAMA_HOST}/api/embeddings"

# Routing knobs
CATALOG_MIN_SCORE = float(os.getenv("CATALOG_MIN_SCORE", "1.0"))
DB_GAP_THRESHOLD  = float(os.getenv("DB_GAP_THRESHOLD", "0.6"))
SHORTLIST_MAX     = int(os.getenv("SHORTLIST_MAX", "2"))

# Heuristic keyword → DB boosts (lowercase keys; additive)
DB_KEYWORD_BOOSTS = {
    "informatica": ["workflow", "wf_", "session", "mapping", "repository", "run_id", "pmrep", "pmcmd"],
    "statements":  ["statement", "ms_", "pdf", "statement csv", "merchant statement","parnter statement", "ms_pdf","invoice"],
    "billing":     ["billing", "charge", "gl", "general ledger", "ar", "invoice", "revenue"],
}

# Domain tags per DB (soft boost if present verbatim)
DB_DOMAIN_TAGS = {
    "informatica": ["powercenter", "etl", "repository", "workflow", "session", "mapping"],
    "statements":  ["statements", "merchants", "pdf", "ms_pdf", "commissions"],
    "billing":     ["finance", "billing", "gl", "ar", "ledger"],
}

@dataclass
class DBSuggestion:
    db_name: str
    score: float
    reasons: Dict[str, Any]  # breakdown per DB: {"total", "keywords", "tags", "tables":[...]}

class CatalogRetriever:
    """
    Catalog-first router helper.

    Assumes RediSearch index over table catalog with schema (typical):
      - db_name        TAG
      - table          TEXT
      - columns        TEXT
      - description    TEXT
      - domain_tags    TAG
      - nl_desc_vector VECTOR (FLOAT32, COSINE)
    Keys use prefix 't:'.

    Exposes:
      - shortlist_dbs(question, k=2, min_score=CATALOG_MIN_SCORE, gap_threshold=DB_GAP_THRESHOLD) -> List[DBSuggestion]
    """

    def __init__(self):
        self.r = redis.from_url(REDIS_URL, decode_responses=False)
        self.idx_tables = self._pick_index()

    # ---------- Public: shortlist ----------
    def shortlist_dbs(
        self,
        question: str,
        k: int = SHORTLIST_MAX,
        min_score: float = CATALOG_MIN_SCORE,
        gap_threshold: float = DB_GAP_THRESHOLD
    ) -> List[DBSuggestion]:
        per_db, per_db_tables = self._search_catalog(question, k_tables=20)

        # Heuristic boosts
        kw_boosts = self._keyword_boost(question)
        tag_boosts = self._tag_boost(question)

        # Merge scores + boosts
        all_dbs = set(per_db.keys()) | set(kw_boosts.keys()) | set(tag_boosts.keys())
        totals: Dict[str, float] = {}
        for db in all_dbs:
            totals[db] = per_db.get(db, 0.0) + kw_boosts.get(db, 0.0) + tag_boosts.get(db, 0.0)

        # Drop empty key if present
        totals.pop("", None)

        # Rank by total score
        ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return []

        # Enforce min_score; if none pass, keep top-1 anyway
        filtered = [r for r in ranked if r[1] >= min_score] or ranked[:1]

        # Decide 1 vs 2 results using gap
        out: List[DBSuggestion] = []
        db1, s1 = filtered[0]
        out.append(DBSuggestion(
            db_name=db1,
            score=s1,
            reasons={
                "total": s1,
                "keywords": kw_boosts.get(db1, 0.0),
                "tags": tag_boosts.get(db1, 0.0),
                "tables": per_db_tables.get(db1, []),
            }
        ))

        if len(filtered) > 1 and (s1 - filtered[1][1]) < gap_threshold and len(out) < k:
            db2, s2 = filtered[1]
            out.append(DBSuggestion(
                db_name=db2,
                score=s2,
                reasons={
                    "total": s2,
                    "keywords": kw_boosts.get(db2, 0.0),
                    "tags": tag_boosts.get(db2, 0.0),
                    "tables": per_db_tables.get(db2, []),
                }
            ))

        return out[:k]

    # ---------- Embedding ----------
    def embed(self, text: str) -> bytes:
        r = requests.post(
            OLLAMA_EMBEDDING_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=(5, 30),
        )
        r.raise_for_status()
        return np.array(r.json()["embedding"], dtype=np.float32).tobytes()

    # ---------- Helpers ----------
    @staticmethod
    def _norm(s: Optional[bytes | str]) -> str:
        if s is None:
            return ""
        if isinstance(s, (bytes, bytearray)):
            try:
                s = s.decode("utf-8", "ignore")
            except Exception:
                s = ""
        return (s or "").strip().lower()

    @staticmethod
    def _to_float(x: Any, default: float = 0.0) -> float:
        try:
            if isinstance(x, (bytes, bytearray)):
                x = x.decode("utf-8", "ignore")
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", (text or "").lower())

    def _keyword_boost(self, text: str) -> Dict[str, float]:
        toks = set(self._tokenize(text))
        boosts: Dict[str, float] = {k: 0.0 for k in DB_KEYWORD_BOOSTS.keys()}
        for db, words in DB_KEYWORD_BOOSTS.items():
            for w in words:
                if w.endswith("_"):  # prefix pattern like "wf_"
                    if any(t.startswith(w) for t in toks):
                        boosts[db] += 0.4
                elif w in toks or (" " in w and w in text.lower()):
                    boosts[db] += 0.4
        return boosts

    def _tag_boost(self, text: str) -> Dict[str, float]:
        low = text.lower()
        boosts: Dict[str, float] = {k: 0.0 for k in DB_DOMAIN_TAGS.keys()}
        for db, tags in DB_DOMAIN_TAGS.items():
            for t in tags:
                if t in low:
                    boosts[db] += 0.2
        return boosts

    # ---------- Index pick / fallback ----------
    def _pick_index(self) -> str:
        candidates: List[str] = []
        if IDX_TABLES_ENV:
            candidates.append(IDX_TABLES_ENV)
        if IDX_TABLES_FALLBACK:
            candidates.extend([c.strip() for c in IDX_TABLES_FALLBACK.split(",") if c.strip()])

        seen = set()
        ordered = [c for c in candidates if not (c in seen or seen.add(c))]
        for idx in ordered:
            try:
                self.r.execute_command("FT.INFO", idx)
                logger.info(json.dumps({
                    "ts": datetime.utcnow().isoformat()+"Z",
                    "msg": "[CATALOG] using index",
                    "index": idx
                }))
                return idx
            except Exception as e:
                logger.warning(f"[CATALOG] FT.INFO failed for '{idx}': {e}")
        # Last resort
        logger.error("[CATALOG] No valid index found, falling back to 'idx:tables' blindly")
        return "idx:tables"

    # ---------- Search catalog (BM25 + KNN), aggregate per DB ----------
    def _search_catalog(self, question: str, k_tables: int = 20) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
        per_db_score: Dict[str, float] = {}
        per_db_tables: Dict[str, List[Dict[str, Any]]] = {}

        def add_score(db: str, add: float):
            if not db:
                return
            per_db_score[db] = per_db_score.get(db, 0.0) + add

        def add_reason(db: str, reason: Dict[str, Any]):
            if not db:
                return
            per_db_tables.setdefault(db, []).append(reason)

        # --- KNN ---
        knn_res = None
        try:
            vec = self.embed(question)
            knn_res = self.r.execute_command(
                "FT.SEARCH", self.idx_tables,
                f'*=>[KNN {k_tables} @nl_desc_vector $vec AS dist]',
                "PARAMS", 2, "vec", vec,
                "SORTBY", "dist",
                "LIMIT", 0, k_tables,
                "RETURN", 5, "db_name", "table", "description", "domain_tags", "dist",
                "DIALECT", 2
            )
        except Exception as e:
            logger.warning(f"[CATALOG] KNN failed: {e}")

        if knn_res and len(knn_res) > 1:
            total = knn_res[0]
            i = 1
            while i < len(knn_res):
                # doc id
                _doc_id = knn_res[i]; i += 1
                # fields
                f = knn_res[i]; i += 1
                # fields array comes as [k1,v1,k2,v2,...]
                d = {}
                for j in range(0, len(f), 2):
                    key = f[j].decode("utf-8", "ignore")
                    d[key] = f[j+1]
                db = self._norm(d.get("db_name"))
                dist = self._to_float(d.get("dist"), 9.99)
                # robust similarity mapping
                sim = 1.0 / (1.0 + max(0.0, dist))
                add_score(db, sim)
                add_reason(db, {
                    "src": "knn",
                    "table": self._safe_decode(d.get("table")),
                    "sim": sim
                })
            logger.info(json.dumps({"ts": datetime.utcnow().isoformat()+"Z", "msg": "[CATALOG] knn", "total": int(total)}))

        # --- BM25 (text query over key fields) ---
        tokens = self._tokenize(question)
        if tokens:
            # OR over tokens across main text fields
            or_terms = " | ".join(tokens)
            bm25_query = f"(@table:({or_terms}) | @columns:({or_terms}) | @description:({or_terms}) | @db_name:({or_terms}))"
        else:
            # Fallback to raw question
            bm25_query = question.strip() or "*"

        bm25_res = None
        try:
            bm25_res = self.r.execute_command(
                "FT.SEARCH", self.idx_tables,
                bm25_query,
                "WITHSCORES",
                "LIMIT", 0, k_tables,
                "RETURN", 4, "db_name", "table", "description", "domain_tags",
                "DIALECT", 2
            )
        except Exception as e:
            logger.warning(f"[CATALOG] BM25 failed: {e}")

        if bm25_res and len(bm25_res) > 1:
            total = bm25_res[0]
            i = 1
            while i < len(bm25_res):
                _doc_id = bm25_res[i]; i += 1
                s_raw = bm25_res[i]; i += 1
                score = self._to_float(s_raw, 0.0)
                f = bm25_res[i]; i += 1
                d = {}
                for j in range(0, len(f), 2):
                    key = f[j].decode("utf-8", "ignore")
                    d[key] = f[j+1]
                db = self._norm(d.get("db_name"))
                add_score(db, 0.3 * score)  # BM25 contributes lighter than KNN
                add_reason(db, {
                    "src": "bm25",
                    "table": self._safe_decode(d.get("table")),
                    "bm25": score
                })
            logger.info(json.dumps({"ts": datetime.utcnow().isoformat()+"Z", "msg": "[CATALOG] bm25", "total": int(total)}))

        return per_db_score, per_db_tables

    @staticmethod
    def _safe_decode(x: Any) -> str:
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8", "ignore")
            except Exception:
                return ""
        return str(x or "")
