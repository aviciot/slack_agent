# src/services/qcache_runtime.py
import re, time, logging, hashlib, redis, numpy as np, requests, os
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", "http://ollama:11434/api/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
SIMILARITY_THRESHOLD = float(os.getenv("QCACHE_ACCEPT", 0.90))
IDX_QCACHE = "idx:qcache"

def ensure_qcache_index(r, dim):
    try:
        r.execute_command(
            "FT.CREATE", "idx:qcache",
            "ON", "HASH",
            "PREFIX", "1", "q:",
            "SCHEMA",
            "signature", "TEXT",
            "sql_template", "TEXT",
            "nl_desc_vector", "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", "COSINE"
        )
        logger.info("Created RediSearch index idx:qcache")
    except Exception as e:
        if "Index already exists" in str(e):
            logger.info("Index idx:qcache already exists")
        else:
            raise


class QCacheRuntime:
    def __init__(self):
        self.r = redis.from_url(REDIS_URL, decode_responses=False)
        # Use your actual embedding dimension here (e.g., 768)
        ensure_qcache_index(self.r, 768)

    # --- Embed text ---
    def embed(self, text: str) -> Optional[np.ndarray]:
        try:
            resp = requests.post(OLLAMA_EMBEDDING_URL, json={"model": EMBED_MODEL, "prompt": text})
            resp.raise_for_status()
            return np.array(resp.json()["embedding"], dtype=np.float32)
        except Exception as e:
            logger.warning(f"[QCACHE] embedding failed: {e}")
            return None

    # --- Signature builder ---
    def template_question(self, text: str) -> Tuple[str, Dict[str,str]]:
        # Replace numbers
        params: Dict[str,str] = {}
        templ = text
        for i, m in enumerate(re.findall(r"\b\d+\b", text)):
            placeholder = f"<NUM_{i}>"
            templ = templ.replace(m, placeholder, 1)
            params[placeholder] = m
        # Replace quoted strings
        for i, m in enumerate(re.findall(r"'([^']+)'", text)):
            placeholder = f"<STR_{i}>"
            templ = templ.replace(f"'{m}'", placeholder, 1)
            params[placeholder] = m
        return templ, params

    def template_sql(self, sql: str, params: Dict[str,str]) -> str:
        t = sql
        for k,v in params.items():
            if k.startswith("<NUM_"):
                t = re.sub(rf"\b{re.escape(v)}\b", k, t)
            else:
                t = t.replace(f"'{v}'", k)
        return t

    def fill_sql(self, templated_sql: str, params: Dict[str,str]) -> str:
        s = templated_sql
        for k,v in params.items():
            if k.startswith("<NUM_"):
                s = s.replace(k, v)
            else:
                s = s.replace(k, f"'{v}'")
        return s

    def cache_key(self, sig: str) -> str:
        return "q:" + hashlib.sha256(sig.encode("utf-8")).hexdigest()

    # --- Store into Redis ---
    def store(self, signature: str, templated_sql: str, vec: np.ndarray):
        key = self.cache_key(signature)
        try:
            mapping = {
                "signature": signature.encode("utf-8"),
                "sql_template": templated_sql.encode("utf-8"),
                "nl_desc_vector": vec.astype(np.float32).tobytes()
            }
            self.r.hset(key, mapping=mapping)
            logger.info(f"[QCACHE] stored key={key}")
        except Exception as e:
            logger.warning(f"[QCACHE] store failed: {e}")

    # --- KNN search ---
    def find(self, sig: str, vec: np.ndarray) -> Optional[Dict[str,Any]]:
        try:
            res = self.r.execute_command(
                "FT.SEARCH", IDX_QCACHE,
                f"*=>[KNN 1 @nl_desc_vector $vec AS score]",
                "PARAMS", 2, "vec", vec.tobytes(),
                "SORTBY", "score",
                "RETURN", 3, "signature", "sql_template", "score",
                "LIMIT", 0, 1,
                "DIALECT", 2
            )
            if res and res[0] > 0:
                # res[2] is a list: [b'signature', b'...', b'sql_template', b'...', b'score', b'...']
                doc_list = res[2]
                doc = {doc_list[i]: doc_list[i+1] for i in range(0, len(doc_list), 2)}
                score = float(doc[b"score"].decode())
                similarity = 1.0 - score
                if similarity >= SIMILARITY_THRESHOLD and doc[b"signature"].decode() == sig:
                    return {
                        "signature": doc[b"signature"].decode(),
                        "templated_sql": doc[b"sql_template"].decode(),
                        "similarity": similarity,
                        "key": res[1].decode()
                    }
        except Exception as e:
            logger.warning(f"[QCACHE] find failed: {e}")
        return None
