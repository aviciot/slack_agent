Source of Truth for DS_Agent (Slack-first Text2SQL Agent)
Last updated: 2025-08-22 (Asia/Jerusalem)

🎯 Purpose

DS_Agent is a Slack-first text-to-SQL assistant focused on Informatica performance data. It converts natural language into safe SQL, executes against a local SQLite mart, and returns results (and previews) back to Slack. This document captures design decisions, architecture, interfaces, and next steps so anyone can pick up the project and continue confidently.

🏗️ Architecture Overview

Containers (docker compose)

api — FastAPI service (REST endpoints, Slack handlers later).

redis — Redis Stack for vector search (RediSearch).

ollama — Local LLM/embeddings server.

kb-seed — One-off seeder; embeds and indexes the query bank.

ollama-init — One-off model puller (generator + embedding models).

ollama-warm (+ optional ollama-warm-embed) — One-off warmers for quicker first call.

ngrok — Optional dev tunnel for Slack.

worker — Reserved for async jobs/retries (stub).

Volumes & mounts

ollama_models — Named volume for persistent model cache.

redis_data — Named volume for Redis persistence.

./data/db/ — Bind mount for SQLite database.

./data/query_bank.json — Bind mount for the query bank source of truth.

Code structure

src/
  app/
    main.py          # FastAPI app, routes registration
    routes_bank.py   # /bank/search → embed → Redis KNN → top templates
    routes_sql.py    # /sql/validate → harden + preview rows
    sql_utils.py     # SQL safety & execution helpers
  persistence/
    sqlite_schema.sql  # telemetry tables
    init_db.py         # ensure DB schema on startup
  tools/
    seed_queries_from_json.py  # embeddings + index + seed bank
    pull_models.py             # pull Ollama models
    test_knn.py                # smoke test: embed → KNN

📑 Query Bank — Source of Truth

File: data/query_bank.json
Schema per entry (required → optional):

{
  "id": "string-unique",
  "nl_desc": "natural-language description",
  "sql_template": "parameterized SQL",
  "params_schema": { "param": "int|float|str|date" },
  "tables": ["table1","table2"],
  "tags": ["performance","sla"]
}


Seeding behavior

Each entry’s embedding text = nl_desc + "\nSQL: " + sql_template (empirically better KNN).

Stored in Redis as bank:{id} (HASH) with:

nl_desc, sql_template, params_schema (JSON string), tables, tags, nl_desc_vector (FLOAT32 bytes).

RediSearch index: idx:queries, HNSW, COSINE, DIM = EMBED_DIM. Text fields for filtering, Tag fields for tables/tags.

🔐 SQL Safety & Validation

Allow only SELECT (ALLOW_SELECT_ONLY=true by default).

Forbid keywords: ATTACH, PRAGMA, INSERT, UPDATE, DELETE, ALTER, DROP, TRUNCATE.

Auto-LIMIT if missing (SQL_LIMIT_DEFAULT, default 200).

Row cap (ROW_CAP, default 2000).

Timeouts via SQLite busy_timeout and container limits.

Preview endpoint: /sql/validate returns sanitized SQL and a small result preview (columns, rows).

🧠 Routing (Bank-First) & Retries

Bank-first retrieval: embed user question → KNN on idx:queries.

If hit, use sql_template + params_schema to fill parameters (TBD UI prompts).

If miss, fallback path will call a text-to-SQL model (local Ollama first, optional Cloudflare later).

Retry strategy: on SQL error, first rephrase the NL and regenerate; if still failing → return actionable message suggesting rephrasing.

🗃️ Telemetry (Planned / Partial)

SQLite tables (defined in sqlite_schema.sql):

llm_requests — prompts, models, token counts, timers.

sql_runs — candidate SQL, validation result, row counts, latency, errors.

routing_decisions — bank hit, provider/model, fallback used.

slack_interactions — user id/name, text, timestamps.

feedback — ratings / notes.

A small helper (telemetry.py) will be added so routes can log consistently with a request_id.

⚙️ Configuration (env)

Key vars (kept minimal here):

DB_PATH=/app/data/db/informatica_insigts_agent.sqlite

REDIS_URL=redis://:devpass@ds_redis:6379/0

REDIS_QUERY_BANK_JSON=/data/query_bank.json

OLLAMA_HOST=http://ollama:11434

EMBED_MODEL=nomic-embed-text (and optionally EMBED_DIM or auto-detect)

SQL_LIMIT_DEFAULT, ROW_CAP, SQL_TIMEOUT_MS

ALLOW_SELECT_ONLY, FORBID_KEYWORDS

✅ Current Status (today)

Containers healthy: api, redis, ollama.

Models pulled & warmed (qwen2.5-coder:7b-instruct, nomic-embed-text).

Query bank seeded and indexed (HNSW/COSINE).

KNN retrieval verified (CLI & /bank/search).

SQL validator working (/sql/validate).

Volumes persist models/data across restarts.

⏭️ Next Steps (short list)

Slack wiring

Events & interactivity → call /bank/search → param prompt → /sql/validate.

Log slack_interactions + feedback.

Telemetry helper

Add telemetry.py; decorate routes with uniform logging.

Param handling UX

Auto-prompt for params_schema (button/select/input in Slack).

LLM fallback path

Implement text-to-SQL generator with safe prompts; add Cloudflare fallback.

Smoke tests

Keep tiny scripts for bank search, validator, Slack webhook.

🧭 Decisions Log (chronological highlights)

Bank-first strategy (chosen over pure on-the-fly NL2SQL)
Rationale: deterministic coverage for critical queries, faster, easier to review; reduces risk of unsafe SQL.
Implication: invest in good nl_desc and templates; add new patterns incrementally.

Query bank schema finalized (id, nl_desc, sql_template, params_schema?, tables?, tags?)
Rationale: minimal but expressive; enables accurate embeddings and parameter prompting.

Embedding text = NL + template
Rationale: empirically improves retrieval precision for semantically similar asks; includes column/table hints without overfitting to a specific literal.

Vector store = Redis Stack (HNSW/COSINE)
Rationale: simple to operate in Docker, fast KNN, plays nicely with metadata filters; no extra infra.

Models via Ollama + named volume cache
Rationale: local, reproducible, efficient. Named volume avoids re-download; separate ollama-init/ollama-warm ensure cold-start is handled.

SQLite execution with guardrails
Rationale: single-file DB is easy to ship and test; strict guardrails make it safer in early stages.

FastAPI with reload; lifespan instead of startup event
Rationale: dev velocity; modern lifecycle; no rebuild needed for code changes.

No dual schemas / no “compat” shims
Rationale: avoid confusion; the bank format above is canonical. Seeder expects exactly this structure.

Step-by-step ops (bring up services individually)
Rationale: clear troubleshooting, no noisy failures; easy to validate each stage.

Logging & telemetry design (tables pre-created, helper pending)
Rationale: we want visibility on latency per step (embed, KNN, validate, execute), routing choices, errors, and user feedback; the helper will ensure consistent inserts.

🧪 Smoke Tests We Keep

src/tools/test_knn.py — quick “embed string → KNN” sanity test.

/sql/validate — quick SQL preview with safety checks.

Seeder re-run — idempotent; shows index health and item counts.

📬 How to Get Productive Tomorrow

docker compose up -d redis ollama api

docker compose run --rm ollama-init (if models changed)

docker compose run --rm kb-seed (if bank changed)

curl "http://localhost:8090/bank/search?q=top%20slow%20workflows"

curl -X POST http://localhost:8090/sql/validate -H "Content-Type: application/json" -d '{"sql":"select name from sqlite_master"}'