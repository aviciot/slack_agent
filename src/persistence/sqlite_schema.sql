PRAGMA foreign_keys = ON;

-- 1) One row per user request (Slack/API/Scheduler)
CREATE TABLE IF NOT EXISTS requests (
  request_id     TEXT PRIMARY KEY,                 -- UUID
  source         TEXT NOT NULL,                    -- 'slack' | 'api' | 'scheduler'
  user_id        TEXT,
  user_name      TEXT,
  channel_id     TEXT,
  thread_ts      TEXT,
  text           TEXT,                             -- original user text
  route          TEXT,                             -- 'bank' | 'llm_local' | 'llm_cloudflare'
  status         TEXT,                             -- 'ok' | 'error' | 'partial'
  timings_json   TEXT,                             -- {"embed":12,"knn":8,...,"total":390}
  created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_source ON requests(source);

-- 2) Retrieval (KNN) details for each request
CREATE TABLE IF NOT EXISTS retrieval (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id    TEXT NOT NULL,
  k             INTEGER NOT NULL,
  query_text    TEXT NOT NULL,
  hits_json     TEXT NOT NULL,                     -- [{"id":"slow_workflows_top_n","score":0.12},...]
  chosen_id     TEXT,
  chosen_score  REAL,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(request_id) REFERENCES requests(request_id)
);
CREATE INDEX IF NOT EXISTS idx_retrieval_req ON retrieval(request_id);

-- 3) Any LLM calls (rephrase / text2sql / fallback)
CREATE TABLE IF NOT EXISTS llm_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    attempt INTEGER,
    provider TEXT,
    model TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    error TEXT,
    extra_json TEXT,             -- 👈 NEW
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_req_req ON llm_requests(request_id);

-- 4) SQL validation & execution
CREATE TABLE IF NOT EXISTS sql_runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id    TEXT NOT NULL,
  sql_before    TEXT,                              -- candidate before hardening
  sql_after     TEXT,                              -- hardened SQL actually executed
  safety_flags  TEXT,                              -- '["added_limit","select_only"]'
  executed      INTEGER NOT NULL DEFAULT 0,        -- 0/1
  duration_ms   INTEGER DEFAULT 0,
  rowcount      INTEGER DEFAULT 0,
  error         TEXT,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(request_id) REFERENCES requests(request_id)
);
CREATE INDEX IF NOT EXISTS idx_sql_runs_req ON sql_runs(request_id);

-- 5) Slack delivery
CREATE TABLE IF NOT EXISTS slack_interactions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id    TEXT NOT NULL,
  channel_id    TEXT,
  message_ts    TEXT,                              -- Slack ts of the message
  response_bytes INTEGER,
  rows_shown    INTEGER,
  truncated     INTEGER DEFAULT 0,                 -- 0/1
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(request_id) REFERENCES requests(request_id)
);
CREATE INDEX IF NOT EXISTS idx_slack_req ON slack_interactions(request_id);

-- 6) User feedback
CREATE TABLE IF NOT EXISTS feedback (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id    TEXT NOT NULL,
  user_id       TEXT,
  rating        TEXT,                              -- 'up' | 'down' | '1'..'5'
  comment       TEXT,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(request_id) REFERENCES requests(request_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_req ON feedback(request_id);

-- (Optional) Scheduler metadata (kept minimal; we might enable later)
CREATE TABLE IF NOT EXISTS schedules (
  id            TEXT PRIMARY KEY,
  enabled       INTEGER NOT NULL DEFAULT 1,
  cron          TEXT NOT NULL,
  tz            TEXT NOT NULL DEFAULT 'UTC',
  source_type   TEXT NOT NULL,                     -- 'bank' | 'sql'
  source_ref    TEXT NOT NULL,                     -- bank id or sql id
  params_json   TEXT NOT NULL DEFAULT '{}',
  channel       TEXT NOT NULL,
  title         TEXT NOT NULL,
  owner         TEXT,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TRIGGER IF NOT EXISTS trg_schedules_updated
AFTER UPDATE ON schedules
FOR EACH ROW BEGIN
  UPDATE schedules SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS schedule_runs (
  run_id        TEXT PRIMARY KEY,
  schedule_id   TEXT NOT NULL,
  started_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  finished_at   DATETIME,
  status        TEXT NOT NULL,                     -- 'ok' | 'failed'
  rows          INTEGER DEFAULT 0,
  duration_ms   INTEGER DEFAULT 0,
  error         TEXT,
  sql           TEXT,
  slack_ts      TEXT,
  FOREIGN KEY(schedule_id) REFERENCES schedules(id)
);
CREATE INDEX IF NOT EXISTS idx_sched_runs_sched ON schedule_runs(schedule_id);

-- Routing decisions (bank vs retrieval)
CREATE TABLE IF NOT EXISTS routing_decisions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id   TEXT NOT NULL,
  source       TEXT,                 -- 'bank' | 'retrieval'
  k            INTEGER,
  candidates   TEXT,                 -- JSON array string
  selected     TEXT,                 -- JSON array string
  chosen_rank  INTEGER,
  created_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_request
  ON routing_decisions (request_id);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_created
  ON routing_decisions (created_at);
