# src/app/agent_types.py

"""
Agent types (shared dataclasses/enums) for the DS_Agent orchestrator.

Overview
--------
This module defines the minimal, transport-friendly types used by the Agent
and its collaborators (router, validator, executor, telemetry). These types
are intentionally framework-agnostic and easy to serialize.

Key concepts
------------
- RouteSource: Where the SQL plan came from.
    - "bank"       → answered using a Query Bank template (including qcache hits)
    - "retrieval"  → generated via retrieval + LLM (BM25+Embeddings → LLM)
  Notes:
    * Keep RouteSource coarse (bank vs retrieval). Record LLM provider/model
      separately in telemetry (e.g., provider="local"/"cloudflare").
    * If you ever want finer granularity, you can extend with values like
      "bank_qcache" (a specialized bank path) without breaking callers.

- AgentStatus: High-level lifecycle outcome used by Slack/UI.
    * OK        → completed successfully
    * RETRYING  → (rarely surfaced) mid-flight status
    * REPLAN    → (rarely surfaced) mid-flight status
    * ERROR     → gave up after retries / re-plans

Type-by-type
------------
SlackContext
    Minimal Slack info used for telemetry/UX (channel/user/thread_ts).

RouteDecision
    Router's decision and the tables considered/selected.
    Fields:
      - source: RouteSource ("bank" | "retrieval")
      - bank_score: Optional[float]   # confidence/score on bank path
      - candidates: List[str]         # tables considered
      - selected: List[str]           # tables chosen for SQL drafting

ValidatorVerdict
    Result of guardrails validation.
    Fields:
      - verdict: "ok" | "retryable" | "fatal"
      - reasons: List[str]            # human-readable reasons/guards hit
      - fixed_sql: Optional[str]      # validator auto-fix (e.g., add LIMIT)
      - row_limit_applied: bool       # whether a LIMIT was inserted/enforced

SQLRunResult
    Outcome of executing SQL (with retry/backoff at the executor layer).
    Fields:
      - status: "success" | "error"
      - rows: Optional[List[Dict[str, Any]]]  # already shaped for formatting
      - error_class: Optional[str]            # taxonomy label (e.g., SQLITE_BUSY)
      - duration_ms: Optional[int]
      - retry_count: int

AgentMeta
    Diagnostic envelope that travels with every reply for observability.
    Fields:
      - request_id: str
      - route: RouteDecision
      - retries: int                   # total executor retries across attempts
      - plan_steps: int                # number of recipe steps taken
      - timings_ms: Dict[str, int]     # optional coarse timings
      - notes: Dict[str, Any]          # free-form debug (e.g., last_error)

AgentReply
    Normalized return value for Agent.run().
    Fields:
      - status: AgentStatus
      - reply: str                     # Slack-safe text
      - meta: AgentMeta

Compatibility notes
-------------------
- Telemetry:
    * Use RouteSource for the high-level route ("bank" vs "retrieval").
    * Log provider/model separately (e.g., telemetry.llm_request()).
- Backwards compatibility:
    * Keep "bank" and "retrieval" stable. Add new values only when
      downstream consumers (dashboards/queries) are ready.

Examples
--------
# A bank-route decision with a single selected table:
route = RouteDecision(source="bank", bank_score=0.78, candidates=["MS_SUMMARY"], selected=["MS_SUMMARY"])

# Successful agent reply:
meta = AgentMeta(request_id="req_123", route=route, retries=1, plan_steps=3)
reply = AgentReply(status=AgentStatus.OK, reply="Here are the top 10 rows…", meta=meta)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal

# ---- Simple literals used across the agent ----
RouteSource = Literal["bank", "retrieval"]

class AgentStatus(str, Enum):
    OK = "OK"
    RETRYING = "RETRYING"
    REPLAN = "REPLAN"
    ERROR = "ERROR"

@dataclass
class SlackContext:
    """Minimal Slack context we care about for telemetry/UX."""
    channel: str
    user: str
    thread_ts: Optional[str] = None

@dataclass
class RouteDecision:
    """What path we took and which tables we considered/chose."""
    source: RouteSource                     # bank | retrieval
    bank_score: Optional[float] = None      # set when source == bank
    candidates: List[str] = field(default_factory=list)  # table names considered
    selected: List[str] = field(default_factory=list)    # table names chosen

@dataclass
class ValidatorVerdict:
    """Mirror of validation outcome (compatible with SQLGuard)."""
    verdict: Literal["ok", "retryable", "fatal"]
    reasons: List[str] = field(default_factory=list)
    fixed_sql: Optional[str] = None
    row_limit_applied: bool = False

@dataclass
class SQLRunResult:
    """Execution result + basic timings for retries accounting."""
    status: Literal["success", "error"]
    rows: List[Dict[str, Any]] | None
    error_class: Optional[str] = None
    duration_ms: Optional[int] = None
    retry_count: int = 0

@dataclass
class AgentMeta:
    """Diagnostic metadata we can serialize to telemetry or logs."""
    request_id: str
    route: RouteDecision
    retries: int = 0
    plan_steps: int = 0
    timings_ms: Dict[str, int] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[Plan] = None 
    
    
@dataclass
class AgentReply:
    """Normalized return value of Agent.run()."""
    status: AgentStatus
    reply: str                               # Slack-safe message text
    meta: AgentMeta

@dataclass
class PlanStep:
    name: str                      # e.g., "route", "draft_sql", "validate", "execute"
    detail: Dict[str, Any]         # arbitrary metadata per step
    status: str = "pending"        # pending | ok | error

@dataclass
class Plan:
    steps: List[PlanStep]

__all__ = [
    "AgentStatus",
    "SlackContext",
    "RouteDecision",
    "ValidatorVerdict",
    "SQLRunResult",
    "AgentMeta",
    "AgentReply",
    "RouteSource",
]
