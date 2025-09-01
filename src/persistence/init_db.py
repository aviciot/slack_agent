import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")
SCHEMA_FILE = os.path.join(Path(__file__).resolve().parent, "sqlite_schema.sql")

PRAGMAS = [
    ("journal_mode", "WAL"),        # better concurrency
    ("synchronous",  "NORMAL"),     # durability vs speed tradeoff
    ("foreign_keys", "ON"),         # enforce FKs
    ("busy_timeout", "8000")        # ms; aligns with SQL_TIMEOUT_MS default
]

def _apply_pragmas(conn: sqlite3.Connection):
    cur = conn.cursor()
    for key, val in PRAGMAS:
        cur.execute(f"PRAGMA {key}={val};")
    cur.close()

def ensure_db() -> None:
    """Create DB directory, apply pragmas, and ensure schema exists."""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    initializing = not db_path.exists()
    conn = sqlite3.connect(DB_PATH)
    try:
        _apply_pragmas(conn)

        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn.executescript(schema_sql)
        conn.commit()

        if initializing:
            print(f"Initialized DB at {DB_PATH}")
        else:
            print(f"DB ready at {DB_PATH}")
    finally:
        conn.close()
