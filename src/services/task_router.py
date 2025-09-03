from __future__ import annotations
import os
from typing import List
from dataclasses import dataclass

from src.services.task_bank import TaskBank, ScoredTask

HIGH = float(os.getenv("TASK_ROUTER_THRESHOLD_HIGH", "0.65"))
MED  = float(os.getenv("TASK_ROUTER_THRESHOLD_MED",  "0.55"))

@dataclass
class TaskDecision:
    task_type: str
    confidence: float
    source: str               # "bank" | "llm" (llm later)
    candidates: List[ScoredTask]
    reasons: List[str]

class TaskRouter:
    def __init__(self, bank_path: str | None = None):
        self.bank = TaskBank(bank_path)

    def route(self, text: str) -> TaskDecision:
        cand = self.bank.search(text, k=3)
        top = cand[0] if cand else None
        if not top:
            return TaskDecision("sql_query", 0.0, "bank", [], ["no candidates"])

        if top.score >= HIGH:
            return TaskDecision(top.id, top.score, "bank", cand, top.reasons)
        if top.score >= MED:
            return TaskDecision(top.id, top.score, "bank", cand, top.reasons)

        # LLM fallback could go here later; for Phase 0 keep bank-only
        return TaskDecision(top.id, top.score, "bank", cand, top.reasons)
