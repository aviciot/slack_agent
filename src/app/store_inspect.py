# src/routes/store_inspect.py
from __future__ import annotations

import os, json
from typing import Any, Dict, List, Optional

import redis
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

# Reuse embedding from the runtime (same model + config)
from src.services.query_bank_runtime import QueryBankRuntime

router = APIRouter(prefix="/api/store", tags=["store"])

# -------------------------
# Env / constants
# -------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://:devpass@ds_redis:6379/0")
IDX_QCACHE = os.getenv("IDX_QCACHE", "idx:qcache")
IDX_TABLES = os.getenv("IDX_TABLES") or os.getenv("TABLES_INDEX", "idx:tables")
DATABASES_JSON_PATH = os.getenv("DATABASES_JSON_PATH") or os.getenv("DATABASES_JSON", "/app/config/databases.json")

# raw redis (vectors need decode_responses=False)
r = redis.from_url(REDIS_URL, decode_responses=False)
qbr = QueryBankRuntime()  # for embedding only

# -------------------------
# Schemas
# -------------------------
class QBankItem(BaseModel):
    key: str
    db_name: str
    signature: Optional[str] = None
    similarity: Optional[float] = None
    dist: Optional[float] = None
    sql_template: Optional[str] = None  # truncated by default

class CatalogItem(BaseModel):
    id: str
    db_name: str
    table: str
    description: Optional[str] = None
    domain_tags: Optional[str] = None
    similarity: Optional[float] = None
    dist: Optional[float] = None

class DBOverview(BaseModel):
    catalog_dbs: List[str]
    qbank_dbs: List[str]
    config: Optional[Any] = None

# -------------------------
# Helpers
# -------------------------
def _truncate(s: str, n: int = 500) -> str:
    return s if len(s) <= n else s[:n] + "…"

def _agg_distinct_dbs(index: str) -> List[str]:
    try:
        # FT.AGGREGATE idx "*" GROUPBY 1 @db_name REDUCE COUNT 0 AS c SORTBY 2 @db_name ASC LIMIT 0 1000
        res = r.execute_command(
            "FT.AGGREGATE", index, "*",
            "GROUPBY", 1, "@db_name",
            "REDUCE", "COUNT", 0, "AS", "c",
            "SORTBY", 2, "@db_name", "ASC",
            "LIMIT", 0, 1000
        )
        out = []
        for row in res[1:]:
            fields = {row[i].decode(): row[i+1].decode() for i in range(0, len(row), 2)}
            dbn = (fields.get("db_name") or "").strip().lower()
            if dbn:
                out.append(dbn)
        return out
    except Exception:
        return []

def _build_text_query(base_filter: Optional[str], q: Optional[str]) -> str:
    """
    Safe builder for RediSearch text queries:
    - If both filter and q → "(filter) (q)"
    - If only filter       → "filter"
    - If only q            → "q"
    - If none              → "*"
    Avoids the invalid pattern "* <terms>" that causes syntax errors.
    """
    q = (q or "").strip()
    if base_filter and q:
        return f"({base_filter}) ({q})"
    if base_filter:
        return base_filter
    if q:
        return q
    return "*"

# -------------------------
# Query Bank search
# -------------------------
@router.get(
    "/qbank/search",
    response_model=List[QBankItem],
    summary="Search the Query Bank (NL→SQL templates)",
    description=(
        "Browse saved NL→SQL templates.\n\n"
        "How to use:\n"
        "• **Text search:** set `q` (e.g., `q=statements by status`).\n"
        "• **Browse a domain:** set `db` (e.g., `db=statements`).\n"
        "• **Fuzzy/semantic:** add `knn=1`.\n"
        "• **Show SQL too:** add `include_sql=1`.\n\n"
        "Note: `db` filters the business/domain tag (e.g., `statements`, `informatica`). "
        "It is **not** the SQL engine name."
    ),
)
def qbank_search(
    q: Optional[str] = Query(
        None,
        description="Plain-English search text (example: 'statements by status').",
        examples={
            "free_text": {"summary": "Free text", "value": "statements by status"},
            "fuzzy": {"summary": "Fuzzy wording", "value": "workflow failures June 2025"},
        },
    ),
    db: Optional[str] = Query(
        None,
        description="Domain tag to filter by (e.g., 'statements', 'informatica'). Not a SQL engine.",
        examples={
            "statements": {"summary": "Statements domain", "value": "statements"},
            "informatica": {"summary": "Informatica domain", "value": "informatica"},
        },
    ),
    k: int = Query(
        25, ge=1, le=200,
        description="How many results to return (top-K).",
        example=10,
    ),
    knn: int = Query(
        0,
        description="0 = text match (exact-ish); 1 = semantic (fuzzy/smarter).",
        examples={"text": {"summary": "Text mode", "value": 0}, "semantic": {"summary": "Semantic mode", "value": 1}},
    ),
    include_sql: int = Query(
        0,
        description="1 = include sql_template (truncated) in results; 0 = hide SQL.",
        examples={"hide": {"summary": "Hide SQL", "value": 0}, "show": {"summary": "Show SQL", "value": 1}},
    ),
) -> List[QBankItem]:
    base = None
    if db:
        base = f"@db_name:{{{db.strip().lower()}}}"
    items: List[QBankItem] = []

    try:
        if knn == 1 and q:
            vec = qbr.embed(q)  # bytes
            filter_str = base if base else "*"
            res = r.execute_command(
                "FT.SEARCH", IDX_QCACHE,
                f'{filter_str}=>[KNN {k} @nl_desc_vector $vec AS dist]',
                "PARAMS", 2, "vec", vec,
                "SORTBY", "dist",
                "LIMIT", 0, k,
                "RETURN", 4, "db_name", "signature", "sql_template", "dist",
                "DIALECT", 2
            )
            for i in range(1, len(res), 2):
                key = res[i].decode("utf-8", "ignore")
                f = res[i+1]
                d = {f[j].decode("utf-8", "ignore"): f[j+1] for j in range(0, len(f), 2)}
                dbn = (d.get("db_name") or b"").decode("utf-8", "ignore").strip().lower()
                sig = (d.get("signature") or b"").decode("utf-8", "ignore")
                try:
                    dist = float((d.get("dist") or b"9.99").decode("utf-8", "ignore"))
                except Exception:
                    dist = 9.99
                similarity = 1.0 - dist
                sql_t = (d.get("sql_template") or b"").decode("utf-8", "ignore")
                items.append(QBankItem(
                    key=key, db_name=dbn, signature=sig,
                    similarity=round(float(similarity), 3),
                    dist=round(float(dist), 6),
                    sql_template=_truncate(sql_t) if include_sql else None
                ))
        else:
            # simple text search (BM25) over signature
            query = _build_text_query(base, q)
            res = r.execute_command(
                "FT.SEARCH", IDX_QCACHE, query,
                "WITHSCORES",
                "LIMIT", 0, k,
                "RETURN", 3, "db_name", "signature", "sql_template",
                "DIALECT", 2
            )
            i = 1
            while i < len(res):
                key = res[i].decode("utf-8", "ignore"); i += 1
                _score = res[i]; i += 1  # not normalized, we omit it
                f = res[i]; i += 1
                d = {f[j].decode("utf-8", "ignore"): f[j+1] for j in range(0, len(f), 2)}
                dbn = (d.get("db_name") or b"").decode("utf-8", "ignore").strip().lower()
                sig = (d.get("signature") or b"").decode("utf-8", "ignore")
                sql_t = (d.get("sql_template") or b"").decode("utf-8", "ignore")
                items.append(QBankItem(
                    key=key, db_name=dbn, signature=sig,
                    similarity=None, dist=None,
                    sql_template=_truncate(sql_t) if include_sql else None
                ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qbank search failed: {type(e).__name__}: {e}")

    return items

# -------------------------
# Catalog search (tables)
# -------------------------
@router.get(
    "/catalog/search",
    response_model=List[CatalogItem],
    summary="Search the Catalog (tables)",
    description=(
        "Explore table metadata from the catalog index.\n\n"
        "How to use:\n"
        "• **Text search:** set `q` (matches table/description).\n"
        "• **Filter a domain:** set `db` (e.g., `db=informatica`).\n"
        "• **Fuzzy/semantic:** add `knn=1`."
    ),
)
def catalog_search(
    q: Optional[str] = Query(None, description="Plain-English search over table/description."),
    db: Optional[str] = Query(None, description="Filter by db_name TAG."),
    k: int = Query(25, ge=1, le=200),
    knn: int = Query(0, description="0 = text search (BM25); 1 = semantic KNN over vectors."),
) -> List[CatalogItem]:
    items: List[CatalogItem] = []
    base = None
    if db:
        base = f"@db_name:{{{db.strip().lower()}}}"

    try:
        if knn == 1 and q:
            vec = qbr.embed(q)
            filter_str = base if base else "*"
            res = r.execute_command(
                "FT.SEARCH", IDX_TABLES,
                f'{filter_str}=>[KNN {k} @nl_desc_vector $vec AS dist]',
                "PARAMS", 2, "vec", vec,
                "SORTBY", "dist",
                "LIMIT", 0, k,
                "RETURN", 5, "db_name", "table", "description", "domain_tags", "dist",
                "DIALECT", 2
            )
            for i in range(1, len(res), 2):
                doc_id = res[i].decode("utf-8", "ignore")
                f = res[i+1]
                d = {f[j].decode("utf-8", "ignore"): f[j+1] for j in range(0, len(f), 2)}
                dbn = (d.get("db_name") or b"").decode("utf-8", "ignore").strip().lower()
                table = (d.get("table") or b"").decode("utf-8", "ignore")
                desc = (d.get("description") or b"").decode("utf-8", "ignore")
                tags = (d.get("domain_tags") or b"").decode("utf-8", "ignore")
                try:
                    dist = float((d.get("dist") or b"9.99").decode("utf-8", "ignore"))
                except Exception:
                    dist = 9.99
                similarity = 1.0 - dist
                items.append(CatalogItem(
                    id=doc_id, db_name=dbn, table=table,
                    description=_truncate(desc, 500), domain_tags=tags,
                    similarity=round(float(similarity), 3), dist=round(float(dist), 6)
                ))
        else:
            query = _build_text_query(base, q)
            res = r.execute_command(
                "FT.SEARCH", IDX_TABLES, query,
                "WITHSCORES",
                "LIMIT", 0, k,
                "RETURN", 4, "db_name", "table", "description", "domain_tags",
                "DIALECT", 2
            )
            i = 1
            while i < len(res):
                doc_id = res[i].decode("utf-8", "ignore"); i += 1
                _score = res[i]; i += 1  # text score not normalized
                f = res[i]; i += 1
                d = {f[j].decode("utf-8", "ignore"): f[j+1] for j in range(0, len(f), 2)}
                dbn = (d.get("db_name") or b"").decode("utf-8", "ignore").strip().lower()
                table = (d.get("table") or b"").decode("utf-8", "ignore")
                desc = (d.get("description") or b"").decode("utf-8", "ignore")
                tags = (d.get("domain_tags") or b"").decode("utf-8", "ignore")
                items.append(CatalogItem(
                    id=doc_id, db_name=dbn, table=table,
                    description=_truncate(desc, 500), domain_tags=tags
                ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"catalog search failed: {type(e).__name__}: {e}")

    return items

# -------------------------
# Databases snapshot
# -------------------------
@router.get("/databases", response_model=DBOverview, summary="List distinct DB tags and config snapshot")
def databases_overview() -> DBOverview:
    cfg = None
    try:
        with open(DATABASES_JSON_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = None

    cat_dbs = _agg_distinct_dbs(IDX_TABLES)
    qb_dbs = _agg_distinct_dbs(IDX_QCACHE)
    return DBOverview(catalog_dbs=cat_dbs, qbank_dbs=qb_dbs, config=cfg)
