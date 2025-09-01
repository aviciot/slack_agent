# src/services/query_bank_runtime.py
from __future__ import annotations

import os, re, time, sqlite3, logging, hashlib, json
import requests, numpy as np, redis
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .template_utils import template_question, template_sql, fill_sql_template

logger = logging.getLogger(__name__)

# ---- Config / ENV ----
REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
DB_PATH   = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")

# Embeddings (Ollama)
OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL") or f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "nomic-embed-text"

# Single index we keep (single source of truth)
IDX_QCACHE  = "idx:qcache"   # runtime template cache (prefix q:)

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

# Acceptance knob (single param)
BANK_ACCEPT = _env_float("BANK_ACCEPT", 0.26)
BANK_MIN_SIMILARITY = max(0.0, min(1.0, 1.0 - BANK_ACCEPT))

# HTTP timeouts for Ollama
CONNECT_TO = float(os.getenv("HTTP_CONNECT_TIMEOUT", 5))
READ_TO    = float(os.getenv("HTTP_READ_TIMEOUT", 30))

# Missing time handling for default binds
BANK_DEFAULT_RANGE_HOURS = int(os.getenv("BANK_DEFAULT_RANGE_HOURS", 24))

@dataclass
class BankHit:
    db_name: str
    template: str
    score: float                 # similarity
    template_key: Optional[str]
    signature: Optional[str]
    meta: Dict[str, Any]

class QueryBankRuntime:
    """
    Authoritative query-bank runtime using ONLY RediSearch qcache (no secondary/legacy cache).

    Index schema (HASH keys with prefix 'q:'):
      - db_name        TAG       (routing filter; query with @db_name:{statements|billing})
      - signature      TEXT      (templated NL signature)
      - sql_template   TEXT      (parameterized SQL template)
      - nl_desc_vector VECTOR    (FLOAT32, COSINE)

    Public APIs:
      - bank_best_match(question, k=5, db_filter=None) -> Optional[BankHit]
      - bank_search_topk(question, k=5, db_filter=None) -> List[BankHit]
      - try_answer_from_bank(user_text, db_filter=None, allow_execute_sqlite=True) -> Optional[Dict]
      - maybe_cache_template_from_llm(user_text, concrete_sql, db_name=None)
      - cache_llm_template(templated_sig, templated_sql, db_name)
    """
    def __init__(self):
        # decode_responses=False to keep vectors as raw bytes
        self.r = redis.from_url(REDIS_URL, decode_responses=False)

    # ---------- Embedding ----------
    def embed(self, text: str) -> bytes:
        r = requests.post(
            OLLAMA_EMBEDDING_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=(CONNECT_TO, READ_TO),
        )
        r.raise_for_status()
        vec = np.array(r.json()["embedding"], dtype=np.float32).tobytes()
        return vec

    def _embed_dim(self) -> int:
        v = np.frombuffer(self.embed("dim probe"), dtype=np.float32)
        return int(v.shape[0])

    # ---------- RediSearch helpers ----------
    def _ensure_qcache_index(self, dim: int):
        try:
            self.r.execute_command("FT.INFO", IDX_QCACHE)
            return
        except Exception:
            pass
        # Create qcache index over prefix q:
        self.r.execute_command(
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
        logger.info(f"[QCACHE] created index {IDX_QCACHE} (DIM={dim})")

    # ---------- Low-level KNN over qcache (with optional DB filter) ----------
    def _knn_qcache(self, vec: bytes, k: int = 5, db_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            if db_filter:
                filt = "|".join(sorted({str(x).strip().lower() for x in db_filter if x}))
                base = f"(@db_name:{{{filt}}})"
            else:
                base = "*"

            # Use alias 'dist' for distance and sort ascending (lower distance is better)
            res = self.r.execute_command(
                "FT.SEARCH", IDX_QCACHE,
                f'{base}=>[KNN {k} @nl_desc_vector $vec AS dist]',
                "PARAMS", 2, "vec", vec,
                "SORTBY", "dist",
                "LIMIT", 0, k,
                "RETURN", 4, "db_name", "signature", "sql_template", "dist",
                "DIALECT", 2
            )
        except Exception as e:
            logger.warning(f"[QCACHE] search failed: {e}")
            return []

        hits: List[Dict[str, Any]] = []
        total = res[0] if res else 0
        for i in range(1, len(res), 2):
            key = res[i].decode("utf-8", "ignore")
            f = res[i+1]
            d = {f[j].decode("utf-8","ignore"): f[j+1] for j in range(0, len(f), 2)}
            d["__key"] = key

            # decode fields to text
            try:
                d["db_name"] = (d.get("db_name") or b"").decode("utf-8", "ignore").strip().lower()
            except Exception:
                d["db_name"] = ""
            try:
                d["signature"] = (d.get("signature") or b"").decode("utf-8", "ignore")
            except Exception:
                d["signature"] = ""

            # dist is a string number; convert and compute similarity (stable mapping)
            try:
                dist = float((d.get("dist") or b"9.99").decode("utf-8", "ignore"))
            except Exception:
                dist = 9.99
            d["dist"] = dist
            sim = 1.0 / (1.0 + max(0.0, dist))
            d["similarity"] = sim

            hits.append(d)

        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            "msg": "[QCACHE] find",
            "total": total,
            "returned": len(hits),
            "top": hits[0].get("__key") if hits else None,
            "similarity": hits[0].get("similarity") if hits else None,
            "filter": db_filter or [],
        }))
        return hits

    # ---------- Store templated SQL into qcache ----------
    def cache_llm_template(self, templated_sig: str, templated_sql: str, db_name: str):
        """
        Store (db_name, signature, sql_template, vector) under key 'q:<sha256(db_name|signature)>'.

        IMPORTANT: db_name must be provided; it enables DB-filtered lookup.
        """
        try:
            if not db_name:
                raise ValueError("db_name is required for caching templates")

            dbv = str(db_name).strip().lower()
            dim = self._embed_dim()
            self._ensure_qcache_index(dim)
            vec = self.embed(templated_sig)  # bytes
            key = "q:" + hashlib.sha256(f"{dbv}|{templated_sig}".encode("utf-8")).hexdigest()
            self.r.hset(key, mapping={
                "db_name": dbv.encode("utf-8"),
                "signature": templated_sig.encode("utf-8"),
                "sql_template": templated_sql.encode("utf-8"),
                "nl_desc_vector": vec,
            })
            logger.info(f"[QCACHE] stored template key={key} db={dbv}")
        except Exception as e:
            logger.warning(f"[QCACHE] store failed: {e}", exc_info=True)

    # ---------- Safety ----------
    def validate_sql(self, sql: str, allowed_tables: List[str]) -> Optional[str]:
        s = (sql or "").strip()
        if s.count(";") > 1:
            return "multiple statements not allowed"
        if not re.match(r"(?is)^\s*select\b", s):
            return "only SELECT statements are allowed"
        forbidden = r"(?is)\b(pragma|attach|detach|delete|update|insert|create|alter|drop|replace|vacuum|analyze)\b"
        if re.search(forbidden, s):
            return "dangerous SQL keyword detected"
        tbls_re = r"(?is)\bfrom\s+([a-zA-Z0-9_]+)|\bjoin\s+([a-zA-Z0-9_]+)"
        mentioned = set([t for m in re.findall(tbls_re, s) for t in m if t])
        if allowed_tables and any(t for t in mentioned if t not in set(allowed_tables)):
            return f"query references non-whitelisted tables: {mentioned} not in {allowed_tables}"
        return None

    def _inject_default_time_binds(self, sql: str, params: Dict[str, Any] | None) -> Dict[str, Any]:
        """
        If SQL references :start/:end/:limit but caller didn't supply them,
        inject sane defaults based on env knobs:
          - :start = now - BANK_DEFAULT_RANGE_HOURS
          - :end   = now
          - :limit = 10  (only if referenced and missing)
        """
        p = dict(params or {})
        needs_start = (":start" in sql) and ("start" not in p)
        needs_end   = (":end"   in sql) and ("end"   not in p)
        needs_limit = (":limit" in sql) and ("limit" not in p)
        if needs_start or needs_end:
            now = datetime.utcnow().replace(microsecond=0)
            if needs_end:
                p["end"] = now.strftime("%Y-%m-%d %H:%M:%S")
            if needs_start:
                p["start"] = (now - timedelta(hours=BANK_DEFAULT_RANGE_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
        if needs_limit:
            p["limit"] = 10
        return p

    # ---------- (Optional) SQLite schema check for local 'informatica' DB ----------
    def schema_check_sqlite(self, sql: str, params: dict | None = None) -> str | None:
        try:
            ro_uri = f"file:{DB_PATH}?mode=ro"
            conn = sqlite3.connect(ro_uri, uri=True, timeout=6)
            try:
                conn.row_factory = sqlite3.Row
                test_sql = f"SELECT * FROM ({sql.rstrip(';')}) LIMIT 0;"
                conn.execute(test_sql, params or {})
            finally:
                conn.close()
            return None
        except Exception as e:
            return f"schema check failed: {e}"

    # ---------- (Optional) Execute on local SQLite (informatica) ----------
    def execute_sqlite(self, sql: str, params: Dict[str, Any] | None) -> Tuple[List[str], List[Tuple]]:
        conn = sqlite3.connect(DB_PATH, timeout=12)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, params or {})
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        conn.close()
        return cols, [tuple(r) for r in rows]

    # ---------- Public: Search (returns hits; does NOT execute) ----------
    def bank_search_topk(self, question: str, k: int = 5, db_filter: Optional[List[str]] = None) -> List[BankHit]:
        try:
            templated_sig, _ = template_question(question)
        except Exception:
            templated_sig = question

        try:
            dim = self._embed_dim()
            self._ensure_qcache_index(dim)
            vec_q = self.embed(templated_sig)
            raw_hits = self._knn_qcache(vec_q, k=k, db_filter=db_filter)
        except Exception as e:
            logger.warning(f"[BANK] bank_search_topk failed: {e}")
            raw_hits = []

        hits: List[BankHit] = []
        for h in raw_hits:
            sim = float(h.get("similarity", 0.0))
            if sim < BANK_MIN_SIMILARITY:
                continue
            tmpl = (h.get("sql_template") or b"").decode("utf-8", "ignore")
            hits.append(BankHit(
                db_name=h.get("db_name", ""),
                template=tmpl,
                score=sim,
                template_key=h.get("__key"),
                signature=h.get("signature"),
                meta={"dist": h.get("dist")}
            ))
        return hits

    def bank_best_match(self, question: str, k: int = 5, db_filter: Optional[List[str]] = None) -> Optional[BankHit]:
        hits = self.bank_search_topk(question, k=k, db_filter=db_filter)
        return hits[0] if hits else None

    # Back-compat wrapper (search without DB filter)
    def bank_best_match_any_db(self, question: str, k: int = 5) -> Optional[BankHit]:
        return self.bank_best_match(question, k=k, db_filter=None)

    # ---------- Public: qcache-first helper (executes only if SQLite/informatica) ----------
    def try_answer_from_bank(
        self,
        user_text: str,
        db_filter: Optional[List[str]] = None,
        allow_execute_sqlite: bool = True
    ) -> Dict[str, Any] | None:
        """
        Single-cache flow, now DB-aware:
          1) Build templated signature from the question
          2) KNN over idx:qcache (optionally filtered by db_filter)
          3) If best hit similarity >= BANK_MIN_SIMILARITY:
               a) Fill template with extracted params
               b) Inject default binds (start/end/limit) if needed
               c) If hit.db_name indicates sqlite/informatica and allow_execute_sqlite:
                    - Validate + schema-check locally and execute
                  else:
                    - Return the filled SQL & params without executing (agent will run via proper executor)
          4) Otherwise return None → caller should do retrieval + LLM
        """
        t0 = time.time()

        try:
            templated_sig, extracted_params = template_question(user_text)
        except Exception:
            templated_sig, extracted_params = user_text, {}

        # Search with optional DB filter (use original user text)
        best = self.bank_best_match(user_text, k=5, db_filter=db_filter)
        if not best:
            logger.info("[QCACHE] no acceptable bank match; fall back to LLM path")
            return None

        # Fill stored template with params derived from the question templating
        final_sql = fill_sql_template(best.template, extracted_params)

        # Common placeholder mapping
        params_qc = dict(extracted_params)
        if "<NUMBER_0>" in params_qc and (":limit" in final_sql) and ("limit" not in params_qc):
            params_qc["limit"] = params_qc.pop("<NUMBER_0>")

        # Inject defaults for any referenced binds (start/end/limit) if missing
        params_qc = self._inject_default_time_binds(final_sql, params_qc)

        # If the hit is for the local SQLite (informatica) DB and execution is allowed, run it here.
        dbn = (best.db_name or "").lower()
        is_sqlite_local = dbn in {"informatica", "sqlite", "local"}
        if allow_execute_sqlite and is_sqlite_local:
            verr = self.validate_sql(final_sql, [])
            if verr:
                logger.info(f"[QCACHE] validation error: {verr}")
                return None

            sch = self.schema_check_sqlite(final_sql, params_qc)
            if sch:
                logger.info(f"[QCACHE] schema error: {sch}")
                return None

            cols, data = self.execute_sqlite(final_sql, params_qc)
            took = int((time.time() - t0) * 1000)
            logger.info(json.dumps({
                "ts": datetime.utcnow().isoformat() + "Z",
                "level": "INFO",
                "msg": "[QCACHE] ok (executed sqlite)",
                "db_name": dbn,
                "signature": templated_sig,
                "similarity": best.score,
                "threshold": BANK_MIN_SIMILARITY,
                "latency_ms": took,
                "rows": len(data),
            }))
            return {
                "source": "bank",
                "db_name": dbn,
                "similarity": best.score,
                "threshold": BANK_MIN_SIMILARITY,
                "accepted": True,
                "needs_execution": False,
                "template_key": best.template_key,
                "signature": best.signature,
                "params_used": params_qc,
                "sql": final_sql,
                "columns": cols,
                "rows": data,
            }

        # Otherwise return a match that the agent/executor layer will run (Oracle/MySQL/etc.)
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            "msg": "[QCACHE] ok (returned template only)",
            "db_name": dbn,
            "signature": templated_sig,
            "similarity": best.score,
            "threshold": BANK_MIN_SIMILARITY,
            "executed": False,
        }))
        return {
            "source": "bank",
            "db_name": dbn,
            "similarity": best.score,
            "threshold": BANK_MIN_SIMILARITY,
            "accepted": True,
            "needs_execution": True,
            "template_key": best.template_key,
            "signature": best.signature,
            "params_used": params_qc,
            "sql": final_sql,
        }

    # ---------- Public: called after LLM success to cache template ----------
    def maybe_cache_template_from_llm(self, user_text: str, concrete_sql: str, db_name: Optional[str] = None):
        """
        Build a templated signature/SQL from the user's text and the validated LLM SQL,
        then store into qcache (REQUIRES db_name; will warn and skip if missing).
        """
        try:
            templated_sig, extracted_params = template_question(user_text)
            t_sql = template_sql(concrete_sql, extracted_params)
            if db_name:
                self.cache_llm_template(templated_sig, t_sql, db_name=db_name)
            else:
                logger.warning("[QCACHE] skip caching LLM template: db_name is required")
        except Exception as e:
            logger.warning(f"[QCACHE] maybe_cache_template_from_llm failed: {e}", exc_info=True)
