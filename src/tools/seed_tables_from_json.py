import os, sys, json, time
import redis, requests, numpy as np

REDIS_URL   = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
CATALOG = os.getenv("TABLE_CATALOG_JSON", "/app/data/table_catalog.json")


IDX         = "idx:tables"
PREFIX      = "tables:"

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
EMB_URL     = os.getenv("OLLAMA_EMBEDDING_URL") or f"{OLLAMA_HOST}/api/embeddings"
EMBED_DIM   = int(os.getenv("EMBED_DIM", "0") or "0")

def log(msg): print(msg, flush=True)

def embed(text: str) -> np.ndarray:
    r = requests.post(EMB_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    vec = r.json().get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("bad embedding")
    return np.array(vec, dtype=np.float32)

def create_index(r: redis.Redis, dim: int):
    # Always force drop + create
    try:
        r.execute_command("FT.DROPINDEX", IDX, "DD")
        log(f"INFO: Dropped index {IDX}")
    except redis.ResponseError:
        log(f"INFO: Index {IDX} not present before create")

    r.execute_command(
        "FT.CREATE", IDX,
        "ON", "HASH",
        "PREFIX", 1, PREFIX,
        "SCHEMA",
        "db_name", "TAG", "SEPARATOR", ",",
        "dialect", "TAG", "SEPARATOR", ",",
        "catalog_id", "TEXT",
        "name", "TEXT",
        "description", "TEXT",
        "columns", "TEXT",
        "ddl", "TEXT",
        "nl_desc_vector", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE"
    )
    log(f"INFO: Created {IDX} (DIM={dim})")

def main():
    log("INFO: Connecting to Redis…")
    r = redis.from_url(REDIS_URL, decode_responses=False); r.ping()

    if not os.path.exists(CATALOG):
        log(f"FATAL: Missing {CATALOG}"); sys.exit(1)

    with open(CATALOG, "r", encoding="utf-8") as f:
        tables = json.load(f)

    dim = EMBED_DIM
    if dim <= 0:
        dim = embed("dimension probe").shape[0]
        log(f"INFO: Autodetected EMBED_DIM={dim}")

    create_index(r, dim)

    count=0; t0=time.time()
    for t in tables:
        name = (t.get("name") or "").strip()
        if not name: continue
        db = (t.get("db_name") or os.getenv("DB_NAME","")).strip()
        dialect = (t.get("dialect") or os.getenv("DIALECT","")).strip()
        catalog_id = (t.get("catalog_id") or "").strip()
        purpose = (t.get("purpose") or "").strip()
        cols = t.get("columns", []) or []
        col_names = [c.get("name","") for c in cols]
        col_syns  = [s for c in cols for s in (c.get("synonyms") or [])]
        col_descs = [c.get("desc","") for c in cols]
        qlist = t.get("possible_questions", []) or []

        embed_text = "\n".join([
            f"Table {name}", purpose,
            " ".join(col_names), " ".join(col_syns), " ".join(col_descs),
            " ".join(qlist)
        ])
        try:
            vec = embed(embed_text).tobytes()
        except Exception:
            vec = b""

        mapping = {
            "db_name": db.encode(),
            "dialect": dialect.encode(),
            "catalog_id": catalog_id.encode(),
            "name": name.encode(),
            "description": purpose.encode(),
            "columns": " ".join(col_names+col_syns+col_descs).encode(),
            "ddl": (t.get("ddl") or "").encode(),
        }
        if vec:
            mapping["nl_desc_vector"] = vec

        r.hset(f"{PREFIX}{db}:{name}", mapping=mapping); count+=1

    log(f"INFO: Seeded {count} tables in {int((time.time()-t0)*1000)}ms ✔")

if __name__=="__main__":
    main()
