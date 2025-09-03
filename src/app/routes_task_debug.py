from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, List, Dict

from src.services.task_router import TaskRouter

router = APIRouter(prefix="/debug/task", tags=["debug"])

class TaskRouteIn(BaseModel):
    text: str

@router.post("/route")
def debug_task_route(body: TaskRouteIn) -> Dict[str, Any]:
    tr = TaskRouter()
    d = tr.route(body.text)
    return {
        "task_type": d.task_type,
        "confidence": d.confidence,
        "source": d.source,
        "candidates": [{"id": c.id, "score": c.score, "reasons": c.reasons} for c in d.candidates],
        "reasons": d.reasons,
    }
