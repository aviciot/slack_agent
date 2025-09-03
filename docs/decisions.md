# Error Decider + Error Bank (Design + Ops)

## What this module does
On DB errors, choose an action (`RETRY`, `REPHRASE`, `QUIT`, `ASK_USER`) using:
1) **Force-quit codes** (per DB policy)
2) **Bank by code** (e.g., `ora:00942`)
3) **Bank by message** (normalized text → `vendor:msg:<sha1-12>`)
4) **Regex rules**
5) **LLM** (Cloudflare) and **cache** the decision back to the bank

## Redis key shapes
- Code:  `{ERROR_BANK_PREFIX}:{vendor_alias}:{code}`  
  e.g. `errorbank:ora:00942`
- Regex: `{ERROR_BANK_PREFIX}:regex:{vendor_alias}` → JSON list of rules
- Msg:   `{ERROR_BANK_PREFIX}:{vendor_alias}:msg:{sha1_12}`  
  e.g. `errorbank:sqlite:msg:915d51408462`

## Endpoint (inspection)
- `GET /api/store/errorbank/search?vendor=<vendor>&q=<substring>`  
  Returns bank entries (code + msg). Use it to verify caching.

## Telemetry
- `requests`, `routing_decisions`, `sql_runs`, `llm_requests`
- **New**: `error_decisions` with:
  `request_id, db_name, vendor, action, category, reason, source, signature, confidence, created_at`

Enable via env:

LOG_ERROR_DECISIONS=true 
(same for all level/areas telemetry)


## Smoke tests
- `src/tools/smoke_error_bank.py` — seed paths + baseline
- `src/tools/smoke_error_decider_combo.py` — regex → llm → bank flow
  - Case 1: missing table → **regex**
  - Case 2: unknown function (first) → **llm**, cached to **bank**
  - Case 3: same unknown function → **bank** (no LLM)
  - Verifiable via `/api/store/errorbank/search?vendor=sqlite`

## Gotchas
- Message signature must match **write vs read** (same SHA1 length).
- If you disable `ERROR_BANK_WRITE`, LLM decisions won’t be cached.
- Seed scripts:
  - `seed_error_bank.py seed` — load from `error_handling.yaml`
  - `seed_error_bank.py snapshot` — import current Redis bank back to Redis (no-op reset + reinsert)


ERROR_DECIDER_MODE=hybrid
ERROR_DECIDER_PROVIDER=cloudflare
ERROR_DECIDER_MODEL=@cf/meta/llama-3-8b-instruct
ERROR_BANK_PREFIX=errorbank
ERROR_BANK_WRITE=1

TTL disabled if not set or set to 0
ERROR_BANK_TTL_DAYS=0

## Smoke tests
- `src/tools/smoke_error_bank.py` — seed paths + baseline
- `src/tools/smoke_error_decider_combo.py` — regex → llm → bank flow
  - Case 1: missing table → **regex**
  - Case 2: unknown function (first) → **llm**, cached to **bank**
  - Case 3: same unknown function → **bank** (no LLM)
  - Verifiable via `/api/store/errorbank/search?vendor=sqlite`

## Gotchas
- Message signature must match **write vs read** (same SHA1 length).
- If you disable `ERROR_BANK_WRITE`, LLM decisions won’t be cached.
- Seed scripts:
  - `seed_error_bank.py seed` — load from `error_handling.yaml`
  - `seed_error_bank.py snapshot` — import current Redis bank back to Redis (no-op reset + reinsert)