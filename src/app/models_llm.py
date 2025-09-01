# src/app/models_llm.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class BankSuccess(BaseModel):
    ok: bool = True
    source: str = "bank"
    template_id: str
    similarity: float
    params_used: Dict[str, Any]
    columns: List[str]
    rows: List[List[Any]]

class BankMissing(BaseModel):
    ok: bool = False
    source: str = "bank"
    reason: str = "missing_params"
    template_id: str
    missing: List[str]
    similarity: float
    message: str

class LLMSuccess(BaseModel):
    ok: bool = True
    source: str = "llm"
    sql: str
    columns: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
