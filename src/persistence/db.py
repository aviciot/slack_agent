import os, sqlite3
from pathlib import Path
from threading import Lock
from typing import Optional

DB_PATH = os.getenv("DB_PATH", "/app/data/db/informatica_insigts_agent.sqlite")

_CONN: Optional[sqlite3.Connection] = None
_LOCK = Lock()

def _ensure_dir():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

def get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    with _LOCK:
        if _CONN is None:
            _ensure_dir()
            conn = sqlite3.connect(
                DB_PATH,
                check_same_thread=False,
                isolation_level=None,   # autocommit-style; use explicit BEGIN if needed
            )
            conn.row_factory = sqlite3.Row
            _apply_pragmas(conn)
            _CONN = conn
    return _CONN
