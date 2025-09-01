# src/app/agent.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import random
import os

from src.services.db_executors import make_registry_from_env

# New: 2-stage routing helpers
from src.services.catalog_retriever import CatalogRetriever
from src.services.query_bank_runtime import QueryBankRuntime

# Types
from src.app.agent_types import (
    AgentReply,
    AgentStatus,
    AgentMeta,
    SlackContext,
    RouteDecision,
    ValidatorVerdict,
    SQLRunResult,
    Plan, PlanStep,
)

# High-level telemetry
from src import telemetry as T  # type: ignore
# Low-level for extended router details
from src.persistence.telemetry_store import log_routing_decision, log_qcache_event


class Agent:
    """Agent orchestrates a request lifecycle (Catalog → Bank → LLM)."""

    def __init__(
        self,
        *,
        settings: Any,
        query_bank: Any = None,        # will default to QueryBankRuntime
        table_selector: Any = None,    # kept for compatibility; not primary router now
        validator: Any,                # SQLGuard instance
        llm: Any | None = None,
        executors: dict | None = None,
    ) -> None:
        self.settings = settings
        self.query_bank = query_bank or QueryBankRuntime()
        self.table_selector = table_selector
        self.validator = validator
        self.llm = llm
        # Build or use provided registry
        self.executors = executors or make_registry_from_env(validator=validator)
        # New: catalog retriever for Stage-1
        self.catalog = CatalogRetriever()

    # --------------------------- Main entry ---------------------------

    def run(self, request_id: str, user_text: str, slack_ctx: SlackContext) -> AgentReply:
        plan_steps = 0
        total_retries = 0

        # PLAN TRACE
        plan = Plan(steps=[])

        def _step(name: str, **detail):
            nonlocal plan_steps
            s = PlanStep(name=name, detail=detail, status="pending")
            plan.steps.append(s)
            plan_steps += 1
            return s

        # 0) Log top-level request (partial)
        try:
            T.http_request(
                request_id=request_id,
                source="slack",
                path="/slack/sql",
                text=user_text,
                user_id=slack_ctx.user,
                channel_id=slack_ctx.channel,
                status="partial",
            )
        except Exception:
            pass

        # ---------------- Stage 1: Catalog shortlist (DBs) ----------------
        s_stage1 = _step("stage1_catalog_shortlist")
        shortlist_objs = self.catalog.shortlist_dbs(user_text)
        shortlist = [{"db": s.db_name, "score": float(s.score)} for s in shortlist_objs]
        db_filter = [s["db"] for s in shortlist] if shortlist else []
        s_stage1.status = "ok"
        s_stage1.detail.update({"shortlist": shortlist})

        # ---------------- Stage 2: Bank inside shortlist ----------------
        s_stage2 = _step("stage2_bank_search", db_filter=db_filter)
        bank_hit = None
        if db_filter:
            try:
                bank_hit = self.query_bank.bank_best_match(user_text, k=5, db_filter=db_filter)
            except Exception:
                bank_hit = None

        stage2_bank_json = None
        chosen_db = None
        chosen_reason = None

        if bank_hit:
            stage2_bank_json = {
                "db": bank_hit.db_name,
                "similarity": float(bank_hit.score),
                "threshold": float(os.getenv("BANK_MIN_SIMILARITY", "0.72")),
                "template_key": bank_hit.meta.get("key") if bank_hit.meta else None,
            }
            chosen_db = bank_hit.db_name
            chosen_reason = "bank_hit"
            s_stage2.status = "ok"
            s_stage2.detail.update(stage2_bank_json)
            # tiny bank event log
            try:
                log_qcache_event(request_id, "hit", qid=stage2_bank_json.get("template_key"), route="bank", question=user_text)
            except Exception:
                pass
        else:
            s_stage2.status = "miss"
            s_stage2.detail.update({"reason": "no_bank_match"})
            # If no bank hit, pick top DB (if any) for LLM path
            if shortlist:
                chosen_db = shortlist[0]["db"]
                chosen_reason = "catalog_llm"
                try:
                    log_qcache_event(request_id, "miss", qid=None, route="catalog_llm", question=user_text)
                except Exception:
                    pass

        # Router telemetry (stage1 + stage2)
        try:
            log_routing_decision(
                request_id=request_id,
                source="catalog_bank",
                k=5,
                candidates=shortlist,                     # same as stage1_shortlist for now
                selected=[chosen_db] if chosen_db else [],
                chosen_rank=0 if chosen_db else None,
                stage1_shortlist=shortlist,
                stage2_bank=stage2_bank_json or {},
                reason=chosen_reason or "no_route",
            )
        except Exception:
            pass

        # ---------------- Fast path: try to answer directly from bank ----------------
        if bank_hit:
            # Let the bank engine fill params and either execute (sqlite) or return SQL (oracle/etc.)
            s_bank = _step("bank_apply_template", db=chosen_db)
            bank_res = None
            try:
                bank_res = self.query_bank.try_answer_from_bank(user_text, db_filter=db_filter, allow_execute_sqlite=True)
            except Exception:
                bank_res = None

            if bank_res and bank_res.get("accepted"):
                # If it needs execution (non-sqlite), run via executor based on DB
                if bank_res.get("needs_execution"):
                    exec_key = self._executor_key_for_db(chosen_db)
                    executor = self.executors.get(exec_key)
                    if not executor:
                        # Can't execute here; fall back to LLM path
                        s_bank.status = "skip"
                    else:
                        try:
                            start = time.time()
                            rows = executor.execute(bank_res["sql"], row_cap=int(getattr(self.settings, "ROW_LIMIT", 200)))
                            duration_ms = int((time.time() - start) * 1000)
                            shaped = self._tuples_to_lists(rows)
                            s_bank.status = "ok"
                            s_bank.detail.update({"rows": len(shaped), "exec_key": exec_key})
                            # Emit SQL run telemetry (no text difference tracking here)
                            try:
                                T.sql_run(
                                    request_id=request_id,
                                    sql_before=bank_res["sql"],
                                    sql_after=bank_res["sql"],
                                    safety_flags=[],
                                    executed=1,
                                    duration_ms=duration_ms,
                                    rowcount=len(shaped),
                                    error=None,
                                )
                            except Exception:
                                pass

                            reply_text = self._format_rows_for_slack(
                                columns=[],  # unknown; renderer will synthesize headers if empty
                                rows=shaped,
                                source=f"bank/{exec_key}",
                                tables=[chosen_db],
                            )
                            route = RouteDecision(source="bank", candidates=[chosen_db], selected=[chosen_db])
                            meta = AgentMeta(request_id=request_id, route=route, retries=0, plan_steps=plan_steps, plan=plan)
                            try:
                                T.http_request(request_id=request_id, source="slack", path="/slack/sql", text=user_text, status="ok")
                            except Exception:
                                pass
                            return AgentReply(status=AgentStatus.OK, reply=reply_text, meta=meta)
                        except Exception as e:
                            # fall through to LLM path
                            s_bank.status = "error"
                            s_bank.detail.update({"error": type(e).__name__})

                else:
                    # Already executed (sqlite path)
                    s_bank.status = "ok"
                    rows = bank_res.get("rows") or []
                    cols = bank_res.get("columns") or []
                    reply_text = self._format_rows_for_slack(
                        columns=cols,
                        rows=rows,
                        source="bank/sqlite",
                        tables=[chosen_db],
                    )
                    route = RouteDecision(source="bank", candidates=[chosen_db], selected=[chosen_db])
                    meta = AgentMeta(request_id=request_id, route=route, retries=0, plan_steps=plan_steps, plan=plan)
                    try:
                        T.sql_run(
                            request_id=request_id, sql_before=None, sql_after=None,
                            safety_flags=[], executed=1, duration_ms=None,
                            rowcount=len(rows), error=None
                        )
                    except Exception:
                        pass
                    try:
                        T.http_request(request_id=request_id, source="slack", path="/slack/sql", text=user_text, status="ok")
                    except Exception:
                        pass
                    return AgentReply(status=AgentStatus.OK, reply=reply_text, meta=meta)
            # If bank couldn’t finalize, continue to LLM path using chosen_db.

        # ---------------- LLM path (with chosen DB from Stage-1) ----------------
        # If no chosen_db at all (no shortlist), fall back to legacy table_selector path
        if not chosen_db:
            route = self._route_retrieval_legacy(user_text)
        else:
            # Build DB-aware route: executor key + tables for that DB
            exec_key = self._executor_key_for_db(chosen_db)
            tables_for_db = self._tables_for_db(chosen_db, user_text)
            route = RouteDecision(source=exec_key, candidates=tables_for_db, selected=tables_for_db[: int(os.getenv("AGENT_LLMSCHEMA_MAX_TABLES", "25"))])

        s_route = _step("route_decision", strategy="catalog_then_bank_then_llm")
        s_route.status = "ok"
        s_route.detail.update({"source": route.source, "db": chosen_db, "selected": route.selected[:5]})

        try:
            T.retrieval(
                request_id=request_id,
                k=len(route.candidates or []),
                candidates=(route.candidates or [])[:10],
                chosen=route.selected[:1] if route.selected else [],
                chosen_rank=1,
                source=route.source,
            )
        except Exception:
            pass

        # 3) DRAFT via LLM
        sql_text, draft_meta = self._draft_sql(user_text, route)
        s_draft = _step("draft_sql", via="llm", db=chosen_db)
        s_draft.status = "ok"
        s_draft.detail.update({"has_sql": True})

        try:
            if draft_meta:
                T.llm_request(
                    request_id=request_id,
                    attempt=1,
                    provider=draft_meta.get("provider"),
                    model=draft_meta.get("model"),
                    tokens_in=draft_meta.get("tokens_in"),
                    tokens_out=draft_meta.get("tokens_out"),
                    latency_ms=draft_meta.get("latency_ms"),
                    extra_json={"db": chosen_db, "route_source": route.source},
                )
        except Exception:
            pass

        # 4) VALIDATE
        verdict = self._validate(sql_text)
        if verdict.fixed_sql:
            sql_text = verdict.fixed_sql
        s_val = _step("validate")
        s_val.status = "ok" if verdict.verdict == "ok" else ("retryable" if verdict.verdict == "retryable" else "error")
        s_val.detail.update({"verdict": verdict.verdict, "reasons": verdict.reasons})

        # 5) EXECUTE
        run_res = self._execute_with_retry(sql_text, route)
        total_retries += run_res.retry_count
        s_exec = _step("execute")
        s_exec.status = run_res.status
        s_exec.detail.update({"rowcount": len(run_res.rows or [])})

        if run_res.status == "success":
            # ✅ Cache template back into bank with db_name
            try:
                self.query_bank.maybe_cache_template_from_llm(user_text, sql_text, db_name=chosen_db or self._db_guess_from_exec(route.source))
            except Exception:
                pass

            reply_text = self._format_rows_for_slack(
                columns=[],
                rows=run_res.rows or [],
                source=f"llm/{route.source}",
                tables=[chosen_db] if chosen_db else (route.selected or []),
            )
            meta = AgentMeta(request_id=request_id, route=route, retries=total_retries, plan_steps=plan_steps, plan=plan)
            try:
                T.http_request(request_id=request_id, source="slack", path="/slack/sql", text=user_text, status="ok")
            except Exception:
                pass
            return AgentReply(status=AgentStatus.OK, reply=reply_text, meta=meta)

        # 5b) QUICK REROUTE to second DB (if available) on failure
        if shortlist and len(shortlist) >= 2:
            alt_db = shortlist[1]["db"]
            alt_exec = self._executor_key_for_db(alt_db)
            alt_tables = self._tables_for_db(alt_db, user_text)
            alt_route = RouteDecision(source=alt_exec, candidates=alt_tables, selected=alt_tables[: int(os.getenv("AGENT_LLMSCHEMA_MAX_TABLES", "25"))])

            s_reroute = _step("reroute_after_error", to_db=alt_db)
            # Draft/validate/exec on alt DB
            alt_sql, alt_meta = self._draft_sql(user_text, alt_route)
            alt_verdict = self._validate(alt_sql)
            if alt_verdict.fixed_sql:
                alt_sql = alt_verdict.fixed_sql
            alt_run = self._execute_with_retry(alt_sql, alt_route)
            total_retries += alt_run.retry_count
            s_reroute.status = alt_run.status
            s_reroute.detail.update({"rowcount": len(alt_run.rows or [])})

            if alt_run.status == "success":
                # ✅ Cache template for alt DB
                try:
                    self.query_bank.maybe_cache_template_from_llm(user_text, alt_sql, db_name=alt_db)
                except Exception:
                    pass

                reply_text = self._format_rows_for_slack(
                    columns=[],
                    rows=alt_run.rows or [],
                    source=f"llm/{alt_route.source}",
                    tables=[alt_db],
                )
                meta_alt = AgentMeta(request_id=request_id, route=alt_route, retries=total_retries, plan_steps=plan_steps, plan=plan)
                # Record router decision (explicit reroute)
                try:
                    log_routing_decision(
                        request_id=request_id,
                        source="catalog_bank",
                        k=5,
                        candidates=shortlist,
                        selected=[alt_db],
                        chosen_rank=1,
                        stage1_shortlist=shortlist,
                        stage2_bank=stage2_bank_json or {},
                        reason="reroute_after_error",
                    )
                except Exception:
                    pass
                return AgentReply(status=AgentStatus.OK, reply=reply_text, meta=meta_alt)

        # 6) REPHRASE (legacy)
        if verdict.verdict in ("retryable",) or run_res.status == "error":
            re_sql, re_meta = self._rephrase(sql_text, verdict)
            s_rephrase = _step("rephrase")
            s_rephrase.status = "ok"
            re_verdict = self._validate(re_sql)
            if re_verdict.fixed_sql:
                re_sql = re_verdict.fixed_sql
            re_run = self._execute_with_retry(re_sql, route)
            total_retries += re_run.retry_count
            s_exec2 = _step("execute_after_rephrase")
            s_exec2.status = re_run.status
            s_exec2.detail.update({"rowcount": len(re_run.rows or [])})

            if re_run.status == "success":
                try:
                    self.query_bank.maybe_cache_template_from_llm(user_text, re_sql, db_name=chosen_db or self._db_guess_from_exec(route.source))
                except Exception:
                    pass

                reply_text = self._format_rows_for_slack(
                    columns=[],
                    rows=re_run.rows or [],
                    source=f"llm/{route.source}",
                    tables=[chosen_db] if chosen_db else (route.selected or []),
                )
                meta = AgentMeta(request_id=request_id, route=route, retries=total_retries, plan_steps=plan_steps, plan=plan)
                return AgentReply(status=AgentStatus.OK, reply=reply_text, meta=meta)

        # 7) REPLAN (deep alt tables; kept for compatibility)
        for _ in range(int(getattr(self.settings, "ALT_TABLE_TRIES", 2))):
            alt_route = self._replan_legacy(user_text, prev_route=route, reason="retry_exhausted_or_fatal")
            s_replan = _step("replan", alt_selected=(alt_route.selected or [])[:5])
            s_replan.status = "ok"
            alt_sql, alt_meta = self._draft_sql(user_text, alt_route)
            alt_verdict = self._validate(alt_sql)
            if alt_verdict.fixed_sql:
                alt_sql = alt_verdict.fixed_sql
            alt_run = self._execute_with_retry(alt_sql, alt_route)
            total_retries += alt_run.retry_count
            s_exec_alt = _step("execute_alt")
            s_exec_alt.status = alt_run.status
            s_exec_alt.detail.update({"rowcount": len(alt_run.rows or [])})

            if alt_run.status == "success":
                try:
                    self.query_bank.maybe_cache_template_from_llm(user_text, alt_sql, db_name=self._db_guess_from_exec(alt_route.source))
                except Exception:
                    pass

                reply_text = self._format_rows_for_slack(
                    columns=[],
                    rows=alt_run.rows or [],
                    source=f"llm/{alt_route.source}",
                    tables=alt_route.selected or [],
                )
                meta_alt = AgentMeta(request_id=request_id, route=alt_route, retries=total_retries, plan_steps=plan_steps, plan=plan)
                return AgentReply(status=AgentStatus.OK, reply=reply_text, meta=meta_alt)

        # 8) FAIL
        msg = "I couldn't complete this after a few attempts. Please try rewording your question."
        meta = AgentMeta(request_id=request_id, route=route, retries=total_retries, plan_steps=plan_steps, plan=plan)
        return AgentReply(status=AgentStatus.ERROR, reply=msg, meta=meta)

    # --------------------------- Internals ---------------------------

    def _executor_key_for_db(self, db_name: Optional[str]) -> str:
        db = (db_name or "").lower()
        if db in {"informatica", "sqlite", "local"}:
            return "sqlite"
        # default other backends to Oracle unless configured otherwise
        return "oracle"

    def _db_guess_from_exec(self, exec_key: str) -> str:
        return "informatica" if exec_key == "sqlite" else "statements"

    def _tables_for_db(self, db_name: Optional[str], question: str) -> List[str]:
        """
        Heuristic table narrowing by DB using validator.catalog keys.
        This is a lightweight filter to keep LLM schema focused.
        """
        max_tables = int(os.getenv("AGENT_LLMSCHEMA_MAX_TABLES", "25"))
        if not hasattr(self.validator, "catalog") or not self.validator.catalog:
            return []

        keys = list(self.validator.catalog.keys())
        low = (db_name or "").lower()

        def pick(predicates: List[str]) -> List[str]:
            out = []
            for t in keys:
                tl = t.lower()
                if any(p in tl for p in predicates):
                    out.append(t)
            return out[:max_tables]

        if low in {"informatica", "sqlite", "local"}:
            # typical Informatica repo/workflow tables (adjust as your schema needs)
            return pick(["wf_", "rep_", "opb_", "pm", "session", "workflow"])
        if low == "statements":
            return pick(["ms_", "ms_pdf", "statement", "commission"])
        if low == "billing":
            return pick(["gl_", "ledger", "ar", "invoice", "charge", "billing"])
        # fallback: return a small generic slice
        return keys[:max_tables]

    # -------- Legacy (kept for compatibility paths) --------

    def _route_retrieval_legacy(self, user_text: str) -> RouteDecision:
        """
        Legacy table_selector path (used only if no DB could be chosen).
        """
        if self.table_selector and hasattr(self.table_selector, "choose_tables"):
            choice = self.table_selector.choose_tables(user_text)
            candidates = list(choice.get("tables", []))
            selected = candidates[:1]
        else:
            candidates = list(getattr(self.validator, "catalog", {}).keys())[:10]
            selected = candidates[:1]
        # assume sqlite for legacy path
        return RouteDecision(source="sqlite", candidates=candidates, selected=selected)

    def _replan_legacy(self, user_text: str, prev_route: RouteDecision, reason: str) -> RouteDecision:
        if self.table_selector and hasattr(self.table_selector, "choose_tables"):
            choice = self.table_selector.choose_tables(user_text)
            cands = [t for t in choice.get("tables", []) if t not in set(prev_route.selected or [])]
        else:
            all_tables = list(getattr(self.validator, "catalog", {}).keys())
            cands = [t for t in all_tables if t not in set(prev_route.selected or [])]
        selected = cands[:1]
        return RouteDecision(source=prev_route.source, candidates=cands, selected=selected)

    def _draft_sql(self, user_text: str, route: RouteDecision) -> Tuple[str, Dict[str, Any]]:
        if route.source == "bank":
            raise RuntimeError("_draft_sql called with bank route")
        if self.llm is None or not hasattr(self.llm, "draft_sql"):
            raise NotImplementedError("No LLM client injected")

        tables = list(route.selected or [])
        if not tables:
            catalog_keys = list(self.validator.catalog.keys()) if hasattr(self.validator, "catalog") else []
            max_tables = int(os.getenv("AGENT_LLMSCHEMA_MAX_TABLES", "25"))
            tables = catalog_keys[:max_tables]

        schema_context = {t: self.validator.catalog.get(t, []) for t in tables} if tables else None
        rules = "sqlite_strict" if route.source == "sqlite" else "oracle_strict"
        try:
            out = self.llm.draft_sql(user_text=user_text, tables=tables, rules=rules, schema_context=schema_context)
        except TypeError:
            out = self.llm.draft_sql(user_text=user_text, tables=tables, rules=rules)

        sql = out.get("sql") if isinstance(out, dict) else getattr(out, "sql", None)
        if not sql:
            raise RuntimeError("LLM draft returned no SQL")
        return sql, {
            "provider": out.get("provider") if isinstance(out, dict) else None,
            "model": out.get("model") if isinstance(out, dict) else None,
            "tokens_in": out.get("tokens_in") if isinstance(out, dict) else None,
            "tokens_out": out.get("tokens_out") if isinstance(out, dict) else None,
            "latency_ms": out.get("latency_ms") if isinstance(out, dict) else None,
        }

    def _rephrase(self, sql_text: str, verdict: ValidatorVerdict) -> Tuple[str, Dict[str, Any]]:
        if self.llm is None or not hasattr(self.llm, "rephrase_sql"):
            raise NotImplementedError("No LLM client injected")
        rules = "sqlite_strict" if verdict and "oracle" not in (verdict.reasons or []) else "oracle_strict"
        out = self.llm.rephrase_sql(sql_text=sql_text, reasons=verdict.reasons, rules=rules)
        sql = out.get("sql") if isinstance(out, dict) else getattr(out, "sql", None)
        if not sql:
            raise RuntimeError("LLM rephrase returned no SQL")
        return sql, {
            "provider": out.get("provider") if isinstance(out, dict) else None,
            "model": out.get("model") if isinstance(out, dict) else None,
            "tokens_in": out.get("tokens_in") if isinstance(out, dict) else None,
            "tokens_out": out.get("tokens_out") if isinstance(out, dict) else None,
            "latency_ms": out.get("latency_ms") if isinstance(out, dict) else None,
        }

    def _validate(self, sql_text: str) -> ValidatorVerdict:
        v = self.validator.validate(sql_text)
        verdict = "ok" if getattr(v, "ok", False) else ("retryable" if getattr(v, "reason", "").startswith("transient") else "fatal")
        fixed_sql = getattr(v, "sql", sql_text)
        reasons = [getattr(v, "reason", None)] if getattr(v, "reason", None) else []
        row_limit_applied = (fixed_sql != sql_text)
        return ValidatorVerdict(verdict=verdict, reasons=reasons, fixed_sql=fixed_sql, row_limit_applied=row_limit_applied)

    def _execute_with_retry(self, sql_text: str, route: RouteDecision) -> SQLRunResult:
        max_retries = int(getattr(self.settings, "RETRY_MAX", 3))
        base_ms = int(getattr(self.settings, "RETRY_BASE_MS", 150))
        jitter_ms = int(getattr(self.settings, "RETRY_JITTER_MS", 200))
        row_cap = int(getattr(self.settings, "ROW_LIMIT", 200))

        executor = self.executors.get(route.source)
        if not executor:
            return SQLRunResult(
                status="error",
                rows=None,
                error_class=f"NoExecutor({route.source})",
                duration_ms=0,
                retry_count=0,
            )

        attempt = 0
        last_err: Optional[str] = None
        started = time.time()

        while attempt <= max_retries:
            try:
                rows = executor.execute(sql_text, row_cap=row_cap)
                duration_ms = int((time.time() - started) * 1000)
                shaped = self._tuples_to_lists(rows)
                return SQLRunResult(
                    status="success",
                    rows=shaped,
                    error_class=None,
                    duration_ms=duration_ms,
                    retry_count=attempt,
                )
            except Exception as e:
                last_err = type(e).__name__
                if attempt == max_retries:
                    duration_ms = int((time.time() - started) * 1000)
                    return SQLRunResult(
                        status="error",
                        rows=None,
                        error_class=last_err,
                        duration_ms=duration_ms,
                        retry_count=attempt,
                    )
                sleep_ms = base_ms * (2 ** attempt) + random.randint(0, jitter_ms)
                time.sleep(sleep_ms / 1000.0)
                attempt += 1

        duration_ms = int((time.time() - started) * 1000)
        return SQLRunResult(
            status="error",
            rows=None,
            error_class=last_err or "UnknownError",
            duration_ms=duration_ms,
            retry_count=max_retries,
        )

    # ---------------- Utility ----------------

    @staticmethod
    def _tuples_to_lists(rows):
        out = []
        for r in rows or []:
            if isinstance(r, tuple):
                out.append(list(r))
            elif isinstance(r, dict):
                out.append(list(r.values()))
            elif isinstance(r, list):
                out.append(r)
            else:
                out.append([r])
        return out

    @staticmethod
    def _format_rows_for_slack(columns: List[str], rows: List[List[Any]], *, source: str, tables: List[str]) -> str:
        src = f"source={source} tables={','.join(tables or [])}"
        if not rows:
            return f"No rows returned ({src})."
        # Build header
        if columns:
            header = " | ".join(columns)
        else:
            header = " | ".join([f"c{i}" for i in range(len(rows[0]) or 0)])
        sep = "-+-".join(["-" * max(3, len(h)) for h in header.split(" | ")])
        lines = [header, sep]
        max_lines = 25
        for r in rows[:max_lines]:
            vals = [str(v) for v in (r.values() if isinstance(r, dict) else r)]
            lines.append(" | ".join(vals))
        body = "\n".join(lines)
        hint = "\n_(truncated)_" if len(rows) > max_lines else ""
        return f"```\n{body}\n```{hint}\n{src}"
