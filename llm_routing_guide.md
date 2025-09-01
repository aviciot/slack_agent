# LLM Routing & Search Guide (Catalog → Bank → LLM)

This document explains, in **plain language**, how the system routes a user question to the right database and SQL—using a **two-stage search** (Catalog then Query Bank) with **semantic vectors (KNN)** and **BM25** keyword scores. It also shows how the **fallback LLM** is prompted, plus handy debug endpoints and knobs to tune.

---

## TL;DR

**Flow (2 stages + fallback):**

1. **Pick DB (Catalog):**  
   Search Redis `idx:tables` (one doc per table) with your question.  
   - **Semantic KNN** over `nl_desc_vector` (high signal).  
   - **BM25 text** over `table / columns / description` (supporting signal).  
   - Small **domain keyword/tag boosts**.  
   Aggregate scores **per `db_name`** → shortlist 1–2 DBs.

2. **Find template (Query Bank):**  
   Search Redis `idx:qcache` for a **templated signature** similar to your question (vector KNN), **filtered to that shortlist**.  
   Accept when `similarity ≥ 1 - BANK_ACCEPT`.

3. **Fallback LLM:**  
   If no bank hit, build a prompt for the configured provider (e.g., Cloudflare) and generate SQL.  
   Include **table hints** from the catalog when available.

---

## Data Stores (and what fields matter)

### A) Catalog (`idx:tables`)
Each table is a Redis doc with fields like:
- `db_name` (**TAG**): logical domain (e.g., `statements`, `informatica`)
- `table` (**TEXT**)
- `columns` (**TEXT**) – optional, if available
- `description` (**TEXT**)
- `domain_tags` (**TAG**)
- `nl_desc_vector` (**VECTOR**, FLOAT32, COSINE)

**Vectors (`nl_desc_vector`) are built from concatenated table text** (table + columns + description).

### B) Query Bank (`idx:qcache`)
Each template is a Redis doc with:
- `db_name` (**TAG**) – same domain concept as above
- `signature` (**TEXT**) – *templated NL signature* of the question
- `sql_template` (**TEXT**)
- `nl_desc_vector` (**VECTOR**, FLOAT32, COSINE) – embedding of the signature

> **Note**: We compare **question↔signature**, not question↔SQL text.

### C) `databases.json`
Connection/config metadata for executors. **Not used for matching** (no search or vectors).

---

## Stage 1 — Pick the DB (Catalog shortlist)

**Goal:** choose the most likely `db_name`(s) from table-level hits.

### What we search
- **Semantic KNN** on `@nl_desc_vector` using an embedding of the **user question**.
- **BM25** text search over `table`, `columns`, `description`.

### How we score per DB
For each hit (table doc) we add to its `db_name` bucket:
- `+ semantic_similarity` (computed as `1 - cosine_distance`).
- `+ 0.3 * bm25_score` (BM25 is helpful but down-weighted).
- `+ small keyword/tag boost` if the question contains domain telltales.

Then we **rank DBs** by total score:
- Enforce `CATALOG_MIN_SCORE`; if none pass, keep top-1 anyway.
- If the gap between #1 and #2 is small `(s1 - s2) < DB_GAP_THRESHOLD`, include both (up to `SHORTLIST_MAX`).

### Why KNN *and* BM25?
- **Semantic (KNN)** captures meaning and paraphrases—great for long or natural phrases.
- **BM25** nails **exact tokens** (e.g., `CUSTOMER_TRANSACTIONS`, acronyms) and helps very short queries.
- Combining both makes the shortlist reliable in real-world wording.

---

## Stage 2 — Find a Template (Query Bank)

**Goal:** find a stored **template** that matches the **meaning** of the user question.

1. Build a **templated signature** from the user question (normalize numbers, dates, etc.).
2. Search `idx:qcache` with **KNN** over `nl_desc_vector`, **filtered to the shortlist DB(s)**.
3. Convert distance to similarity: `similarity = 1 - dist`.
4. Accept when `similarity ≥ BANK_MIN_SIMILARITY`, where:
   - `BANK_MIN_SIMILARITY = 1 - BANK_ACCEPT`  
     (e.g., `BANK_ACCEPT=0.26` → threshold `0.74`)

5. If accepted:
   - Fill the `sql_template` with extracted params.
   - If DB is local SQLite (e.g., `informatica`), we may execute directly (if enabled).
   - Else return SQL + params for the proper executor (Oracle, etc.).

6. If **no** template is good enough → fallback LLM.

> **We compare question↔signature** (both embedded). We do **not** compare to SQL text.

---

## Fallback LLM

When there’s no bank hit:
- Use the **picked DB** and any **table hints** from Stage 1.
- Build a **minimal, strict prompt** (“single SELECT, no DDL, limit rows if reasonable”).
- Call the configured provider (e.g., Cloudflare `@cf/defog/sqlcoder-7b-2`).

**Prompt example (simplified):**
```
You are an expert SQL generator for the 'informatica' database.
Write a SINGLE SQL SELECT statement that answers the question.
Prefer the tables listed if they make sense. No DDL, no comments.

Question:
Which workflows use table CUSTOMER_TRANSACTIONS as source?
SQL:
```

---

## Concrete Examples

### Example 1 — Clear domain + good bank match
**Q:** “statements count by merchant for 2025-07 by status”

- **Stage 1 (Catalog):** hits tables in `statements` → pick `db_name=statements`.
- **Stage 2 (Bank):** question signature ≈ a stored template’s signature → similarity `≈ 1.0` ≥ `0.74` → **bank hit**.
- **Result:** Filled `sql_template` for `statements`, returned.

**Inspector checklist:**
- `/api/store/catalog/search?q=statements%20count%20merchant&db=statements&knn=1`
- `/api/store/qbank/search?q=statements%20by%20status&db=statements&knn=1&include_sql=1`

### Example 2 — Table token + strict threshold → LLM
**Q:** “Which workflows use table CUSTOMER_TRANSACTIONS as source?”

- **Stage 1 (Catalog):** `CUSTOMER_TRANSACTIONS` token gives strong BM25; semantic also helps → pick `db_name=informatica`.
- **Stage 2 (Bank):** similarity is below strict threshold (e.g., `0.803 < 0.90`) → **bank miss**.
- **Fallback LLM:** prompt built with DB + (if available) table hints; provider returns SQL.

**Inspector checklist:**
- `/api/store/catalog/search?q=CUSTOMER_TRANSACTIONS&db=informatica&knn=1`
- `/api/store/qbank/search?q=workflows%20table%20source&db=informatica&knn=1`

### Example 3 — Ambiguous wording + boosts help
**Q:** “billing status by merchant”

- Very short; semantic may be weak; **keyword boosts** for `billing` nudge `db_name=billing` above others.
- Stage 2 tries bank; if similarity doesn’t pass threshold → LLM.

---

## Config Knobs (most-used)

- **Catalog → DB shortlist**
  - `CATALOG_MIN_SCORE` (default `1.0`) — minimum total to accept.
  - `DB_GAP_THRESHOLD` (default `0.6`) — include #2 if gap small.
  - `SHORTLIST_MAX` (default `2`) — max DBs returned.
  - `DB_KEYWORD_BOOSTS`, `DB_DOMAIN_TAGS` — tiny nudges; safe to zero out.

- **Bank (template match)**
  - `BANK_ACCEPT` — **lower accept distance** → **higher similarity required**.  
    `BANK_MIN_SIMILARITY = 1 - BANK_ACCEPT` (e.g., `0.26 → 0.74`).
  - `BANK_TOPK` — neighbors to pull from Redis (search breadth).

- **Missing params policies**
  - `BANK_MISSING_TIME_POLICY` = `default | llm | ask`
  - `BANK_DEFAULT_RANGE_HOURS` — default time window if needed.
  - `BANK_MISSING_ENTITY_POLICY` = `llm | ask`

---

## Debug & Visibility

- **Store Inspector (read-only):**
  - `GET /api/store/qbank/search` — search templates (text or KNN).  
    Examples:  
    - `?q=statements by status`  
    - `?db=statements`  
    - `?q=workflow failures&knn=1&k=10&include_sql=1`
  - `GET /api/store/catalog/search` — search tables.  
    Examples:  
    - `?q=CUSTOMER_TRANSACTIONS&db=informatica&knn=1`  
    - `?db=statements`
  - `GET /api/store/databases` — distinct DB tags + config snapshot.

- **Router (end-to-end):**
  - `POST /api/text2sql` — runs the full flow and returns route (`bank_hit` vs `catalog_llm`), prompt, SQL, etc.

---

## FAQ

**Q: What exactly is _semantic search_?**  
It compares **meanings** using vectors. We embed your question and the stored text (table descriptions or signatures) into numbers (vectors) and find the nearest neighbors by **cosine** distance. Great for paraphrases.

**Q: What is _BM25_?**  
A classic keyword scoring method. It boosts matches that contain your exact words (and balances term frequency vs document length). Great for **exact names** and **short queries**.

**Q: Why combine both?**  
Because real questions mix paraphrase and exact tokens. Semantic finds meaning; BM25 catches precise identifiers. Together they’re robust.

**Q: How do you convert vector distance to a score?**  
RediSearch returns **cosine distance** (`0.0` is identical). We use `similarity = 1 - dist` (closer to `1.0` is better).

**Q: `db=oracle` didn’t work in the inspector. Why?**  
`db` filters the **domain tag** (like `statements`, `informatica`), **not** the SQL engine. If you want to filter by engine, add a separate `dialect` tag.

---

## Gotchas & Notes

- Avoid duplicates in keyword boosts; or dedupe with `set(...)` in code.
- RediSearch text queries cannot be `"* <terms>"`; the inspector builds safe queries to avoid this.
- `databases.json` is NOT indexed—only used for executors.
- Bank compares **question↔signature**, not to SQL text.

---

## Quick Tuning Cheatsheet

- Too many false bank hits? **Raise strictness** → lower `BANK_ACCEPT` (e.g., `0.10`) → threshold `0.90`.
- Bank never hits? **Lower threshold** (e.g., `BANK_ACCEPT=0.30` → `0.70` sim).
- Wrong DB shortlisted? Reduce/zero the keyword boosts; increase semantic weight by ensuring catalog vectors include good descriptions/columns.
- LLM choosing weird tables? Make sure `tables_hint` is populated by filtering tables **per picked DB** in Stage 1.
