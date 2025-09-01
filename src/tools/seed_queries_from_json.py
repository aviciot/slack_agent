# src/tools/seed_queries_from_json.py
from __future__ import annotations

import os, sys, json, hashlib
import redis
from collections import defaultdict

# Allow running as a module/script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add src/ to path if needed

from services.query_bank_runtime import QueryBankRuntime
from services.template_utils import template_question, template_sql

REDIS_URL   = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
QUERY_JSON  = os.getenv("QUERY_STORE_JSON", "/app/data/query_store.json")
DEFAULT_DB  = (os.getenv("DEFAULT_DB_NAME") or "").strip().lower()  # used if records don't include db_name
SKIP_CLEAN  = os.getenv("SKIP_CLEAN", "0") == "1"  # set SKIP_CLEAN=1 to keep existing q:* and idx

def _normalize_db_name(rec) -> str:
    """
    Pull db_name from common fields; fall back to DEFAULT_DB if set.
    Accepted keys: 'db_name', 'db', 'database'
    """
    for k in ("db_name", "db", "database"):
        v = (rec.get(k) or "").strip().lower()
        if v:
            return v
    return DEFAULT_DB  # may be ""

def _iter_questions(rec) -> list[str]:
    q = []
    if rec.get("question"):
        q.append(rec["question"])
    if isinstance(rec.get("nl_paraphrases"), list):
        q.extend([x for x in rec["nl_paraphrases"] if x])
    if isinstance(rec.get("questions"), list):  # optional bulk form
        q.extend([x for x in rec["questions"] if x])
    # de-dupe while preserving order
    seen = set()
    out = []
    for s in q:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def main():
    # Load JSON
    try:
        with open(QUERY_JSON, "r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Query-store JSON not found at: {QUERY_JSON}")
    if not isinstance(records, list):
        raise ValueError("QUERY_STORE_JSON must be a list of records")

    r = redis.from_url(REDIS_URL, decode_responses=False)

    if not SKIP_CLEAN:
        # Cleanup (fresh seed)
        try:
            r.execute_command("FT.DROPINDEX", "idx:qcache", "DD")
            print("[cleanup] dropped idx:qcache")
        except Exception:
            print("[cleanup] no idx:qcache to drop")

        pipe = r.pipeline(transaction=False)
        removed = 0
        for key in r.scan_iter("q:*"):
            pipe.delete(key)
            removed += 1
        pipe.execute()
        print(f"[cleanup] removed {removed} q:* keys")
    else:
        print("[cleanup] skipped (SKIP_CLEAN=1)")

    # Ensure index exists with correct vector dim and db_name TAG
    qbr = QueryBankRuntime()
    dim = qbr._embed_dim()
    qbr._ensure_qcache_index(dim)  # single source of truth with db_name TAG

    ok = 0
    skipped = 0
    per_db_counts = defaultdict(int)

    # Seed
    pipe = r.pipeline(transaction=False)
    for rec in records:
        db_name = _normalize_db_name(rec)
        sql = rec.get("sql_template") or rec.get("sql")
        if not sql:
            skipped += 1
            print("[skip] missing 'sql'/'sql_template' field")
            continue

        if not db_name:
            print(f"[skip] missing db_name (and DEFAULT_DB_NAME unset) for sql starting: {str(sql)[:60]}...")
            skipped += 1
            continue

        questions = _iter_questions(rec)
        if not questions:
            # still allow seeding a template without NL (rare), but warn
            print(f"[warn] no NL questions for db={db_name}; skipping this record")
            skipped += 1
            continue

        for q in questions:
            try:
                templated_sig, extracted = template_question(q)  # e.g., normalize parameter placeholders
                sql_template_text = template_sql(sql, extracted)  # ensure template is normalized

                vec = qbr.embed(templated_sig)
                dbv = db_name.strip().lower()
                # IMPORTANT: key uses db_name|signature so collisions across DBs don't overwrite
                key = "q:" + hashlib.sha256(f"{dbv}|{templated_sig}".encode("utf-8")).hexdigest()

                pipe.hset(key, mapping={
                    b"db_name": dbv.encode("utf-8"),
                    b"signature": templated_sig.encode("utf-8"),
                    b"sql_template": sql_template_text.encode("utf-8"),
                    b"nl_desc_vector": vec,
                })
                ok += 1
                per_db_counts[dbv] += 1
                print(f"[ok] db={dbv} sig={templated_sig[:60]}...")
            except Exception as e:
                skipped += 1
                print(f"[skip] {q[:60]}... error: {e}")

    pipe.execute()

    # Summary
    by_db = ", ".join(f"{db}:{cnt}" for db, cnt in sorted(per_db_counts.items()))
    print(
        f"\nSeed done. Loaded {ok} entries (skipped {skipped}). "
        f"Index: idx:qcache (single source of truth with db_name tagging). "
        f"Per DB: {by_db or 'n/a'}"
    )

if __name__ == "__main__":
    main()
