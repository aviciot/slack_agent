#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import numpy as np
import redis

def get_args():
    p = argparse.ArgumentParser(description="KNN search over query bank (idx:queries)")
    p.add_argument("--q", "--query", dest="query", required=True, help="Natural language query")
    p.add_argument("-k", "--topk", dest="k", type=int, default=5, help="Top-K results (default: 5)")
    return p.parse_args()

def main():
    args = get_args()

    # ---- ENV / Defaults ----
    REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
    EMBED_MODEL = os.getenv("EMBED_MODEL") or "nomic-embed-text"
    OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
    OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL") or f"{OLLAMA_HOST}/api/embeddings"

    # ---- 1) Embed the query with Ollama ----
    try:
        r = requests.post(OLLAMA_EMBEDDING_URL, json={"model": EMBED_MODEL, "prompt": args.query})
        r.raise_for_status()
        emb = r.json().get("embedding")
        if not emb:
            print("ERR: embedding not returned from Ollama", file=sys.stderr)
            sys.exit(2)
        vec = np.array(emb, dtype=np.float32).tobytes()
    except Exception as e:
        print(f"ERR: embedding request failed: {e}", file=sys.stderr)
        sys.exit(2)

    # ---- 2) KNN search in Redis (RediSearch) ----
    try:
        rc = redis.from_url(REDIS_URL, decode_responses=False)
        # FT.SEARCH idx:queries "*=>[KNN k @nl_desc_vector $vec AS score]" PARAMS 2 vec <bytes> SORTBY score RETURN ...
        query = f"*=>[KNN {args.k} @nl_desc_vector $vec AS score]"

        res = rc.execute_command(
            "FT.SEARCH", "idx:queries",
            query,
            "PARAMS", 2, "vec", vec,
            "SORTBY", "score",
            "LIMIT", 0, args.k,
            "RETURN", 6, "nl_desc", "sql_template", "tables", "tags", "params_schema", "score",
            "DIALECT", 2
        )

    except Exception as e:
        print(f"ERR: Redis search failed: {e}", file=sys.stderr)
        sys.exit(3)

    # ---- 3) Pretty print ----
    total = res[0] if res else 0
    print(f"Total hits: {total}")
    if total == 0:
        return

    for i in range(1, len(res), 2):
        key = res[i].decode("utf-8", errors="ignore")
        fields = res[i+1]
        d = {fields[j].decode("utf-8", "ignore"): fields[j+1].decode("utf-8", "ignore")
             for j in range(0, len(fields), 2)}
        print("\n—" * 30)
        print(f"Key: {key}")
        # RediSearch returns cosine distance; smaller = closer
        try:
            score = float(d.get("score", "nan"))
        except ValueError:
            score = d.get("score")
        print(f"Score (cosine distance): {score}")
        print(f"NL: {d.get('nl_desc')}")
        print(f"Tables: {d.get('tables')}")
        print(f"Tags: {d.get('tags')}")
        print(f"Params: {d.get('params_schema')}")
        print("SQL:")
        print(d.get("sql_template"))

if __name__ == "__main__":
    main()
