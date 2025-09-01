# src/validation/sql_guard.py
from __future__ import annotations
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

SQL_FORBID_DEFAULT = r"(?:ATTACH|DETACH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|REINDEX|VACUUM|TRIGGER|INDEX|ANALYZE)"
SQL_READONLY = os.getenv("SQL_READONLY", "1") == "1"
SQL_REQUIRE_LIMIT = os.getenv("SQL_REQUIRE_LIMIT", "1") == "1"
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "1000"))
SQL_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "3000"))
SQL_FORBID = os.getenv("SQL_FORBID", SQL_FORBID_DEFAULT)
VALIDATION_RETRY_MAX = int(os.getenv("VALIDATION_RETRY_MAX", "2"))

# NOTE: our strict SQLite rules
_ILLEGAL_TOKENS = re.compile(SQL_FORBID, re.IGNORECASE)
_MULTISTMT = re.compile(r";\s*\S")  # semicolon followed by more text → multiple statements
_WRITE_HINTS = re.compile(r"\b(TRANSACTION|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE)\b", re.IGNORECASE)

# Extremely defensive: only SELECT, no WITH RECURSIVE that writes, no PRAGMA, no temp tables.
_ALLOWED_LEAD = re.compile(r"^\s*(WITH\b.*\bSELECT\b|SELECT\b)", re.IGNORECASE)
_PRAGMA = re.compile(r"\bPRAGMA\b", re.IGNORECASE)
_TEMP_TABLE = re.compile(r"\bTEMP\b|\bTEMPORARY\b", re.IGNORECASE)

# Simple LIMIT detection (handles LIMIT N and LIMIT N OFFSET M)
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?\b", re.IGNORECASE)

@dataclass
class ValidationResult:
    ok: bool
    sql: str
    reason: Optional[str] = None
    plan: Optional[List[Tuple[Any, ...]]] = None
    tables: Optional[List[str]] = None
    columns: Optional[List[str]] = None
    truncated: bool = False

class SQLGuard:
    """
    Validates LLM-generated SQL for safety, schema correctness, and execution cost.
    Provides a guarded execute() with row/timeout caps.
    """

    def __init__(
        self,
        db_path: str,
        catalog: Dict[str, List[str]],  # {table_name: [col1, col2, ...]}
        big_tables: Optional[Dict[str, int]] = None,  # table_name -> relative size score
    ):
        self.db_path = db_path
        self.catalog = {k.lower(): [c.lower() for c in v] for k, v in catalog.items()}
        self.big_tables = {k.lower(): v for k, v in (big_tables or {}).items()}

    # ---------- Public API ----------

    def validate(self, sql: str) -> ValidationResult:
        s1 = self._safety_gate(sql)
        if not s1.ok:
            return s1

        # Normalize LIMIT if required
        normalized_sql = s1.sql
        if SQL_REQUIRE_LIMIT and not _HAS_LIMIT.search(normalized_sql):
            normalized_sql = self._append_limit(normalized_sql, SQL_MAX_ROWS)

        # Schema checks
        tables, cols = self._extract_identifiers(normalized_sql)
        missing = self._unknown_schema(tables, cols)
        if missing:
            return ValidationResult(False, normalized_sql, reason=f"unknown schema: {missing}")

        # SQLite parse + plan
        plan_res = self._explain(normalized_sql)
        if not plan_res.ok:
            return plan_res

        # Plan heuristics (avoid foot-guns)
        heavy_reason = self._is_heavy(plan_res.plan, tables)
        if heavy_reason:
            return ValidationResult(False, normalized_sql, reason=heavy_reason, plan=plan_res.plan, tables=tables, columns=cols)

        # PASS
        return ValidationResult(True, normalized_sql, plan=plan_res.plan, tables=tables, columns=cols)

    def execute_guarded(self, sql: str, params: Optional[Tuple]=None, row_cap: int=SQL_MAX_ROWS) -> Tuple[List[Tuple], bool]:
        """
        Executes a validated SELECT under caps; returns (rows, truncated).
        """
        params = params or ()
        con = sqlite3.connect(self._ro_uri(), uri=True)
        try:
            con.row_factory = None
            # Timeout & progress guard
            con.execute(f"PRAGMA busy_timeout={SQL_TIMEOUT_MS};")
            aborted = {"stop": False}
            steps = {"n": 0}
            def progress_handler():
                steps["n"] += 1
                # crude step cap; ~1e6 is already huge for SQLite VM steps
                if steps["n"] > 1_000_000:
                    aborted["stop"] = True
                    return 1  # abort
                return 0
            con.set_progress_handler(progress_handler, 1000)

            cur = con.cursor()
            cur.execute(sql, params)
            out = []
            for i, row in enumerate(cur):
                if i >= row_cap:
                    return out, True
                out.append(row)
            return out, False
        finally:
            con.close()

    # ---------- Gates ----------

    def _safety_gate(self, sql: str) -> ValidationResult:
        if not _ALLOWED_LEAD.search(sql):
            return ValidationResult(False, sql, reason="only SELECT statements are allowed")
        if _PRAGMA.search(sql):
            return ValidationResult(False, sql, reason="PRAGMA not allowed")
        if _TEMP_TABLE.search(sql):
            return ValidationResult(False, sql, reason="TEMP tables not allowed")
        if _MULTISTMT.search(sql):
            return ValidationResult(False, sql, reason="multiple statements not allowed")
        if _ILLEGAL_TOKENS.search(sql):
            return ValidationResult(False, sql, reason="forbidden token")
        if _WRITE_HINTS.search(sql):
            return ValidationResult(False, sql, reason="transactional keywords not allowed")
        return ValidationResult(True, sql)

    def _append_limit(self, sql: str, n: int) -> str:
        # naive but safe: append if statement seems to end without LIMIT
        return sql.rstrip().rstrip(";") + f" LIMIT {n}"

    def _extract_identifiers(self, sql: str) -> Tuple[List[str], List[str]]:
        # Light-weight heuristic: capture table names after FROM/JOIN and column tokens after SELECT/WHERE/ORDER/GROUP.
        # We avoid quoting rules since our LLM prompt forbids quoted identifiers.
        tables = [t.lower() for t in re.findall(r"\bFROM\s+([a-zA-Z0-9_]+)", sql, re.IGNORECASE)]
        joins = [t.lower() for t in re.findall(r"\bJOIN\s+([a-zA-Z0-9_]+)", sql, re.IGNORECASE)]
        tables = list(dict.fromkeys(tables + joins))  # unique preserve order

        # Column heuristic: find tokens that look like t.c or bare c in select list/order/group/having/where
        col_like = re.findall(r"(?:(?<=\s)|^)([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", sql)
        cols = [f"{a.lower()}.{b.lower()}" for a, b in col_like]
        return tables, cols

    def _unknown_schema(self, tables: List[str], cols: List[str]) -> Optional[Dict[str, Any]]:
        missing = {}
        for t in tables:
            if t not in self.catalog:
                missing.setdefault("tables", []).append(t)
        for qc in cols:
            if "." in qc:
                t, c = qc.split(".", 1)
                if t in self.catalog and c not in self.catalog[t]:
                    missing.setdefault(f"{t}.columns", []).append(c)
        return missing or None

    def _explain(self, sql: str) -> ValidationResult:
        try:
            con = sqlite3.connect(self._ro_uri(), uri=True)
            try:
                cur = con.cursor()
                cur.execute("EXPLAIN QUERY PLAN " + sql)
                plan = cur.fetchall()
                return ValidationResult(True, sql, plan=plan)
            finally:
                con.close()
        except sqlite3.Error as e:
            return ValidationResult(False, sql, reason=f"sqlite error: {e}")

    def _is_heavy(self, plan: Optional[List[Tuple]], tables: List[str]) -> Optional[str]:
        if not plan:
            return None
        # Flag full scans on big tables and cartesian joins
        plan_text = " | ".join(str(p) for p in plan)
        if "CARTESIAN" in plan_text.upper():
            return "cartesian join detected"
        for t in tables:
            if t in self.big_tables and self.big_tables[t] >= 9:  # arbitrary 'very big'
                if re.search(rf"SCAN\s+TABLE\s+{re.escape(t)}\b", plan_text, re.IGNORECASE) or "SCAN " in plan_text.upper():
                    return f"full scan on big table {t}"
        return None

    # ---------- Utils ----------

    def _ro_uri(self) -> str:
        if not SQL_READONLY:
            return self.db_path  # unsafe, but allows tests
        if self.db_path.startswith("file:"):
            # respect existing query string; ensure mode=ro
            glue = "&" if "?" in self.db_path else "?"
            return f"{self.db_path}{glue}mode=ro"
        return f"file:{self.db_path}?mode=ro"
