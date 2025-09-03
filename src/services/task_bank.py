from __future__ import annotations
import json
import os
from typing import List
from dataclasses import dataclass

import numpy as np
from src.services.query_bank_runtime import QueryBankRuntime  # reuse embedder

@dataclass
class TaskCard:
    id: str
    description: str
    examples: List[str]
    synonyms: List[str]
    required_params: List[str]
    followups: List[str]

@dataclass
class ScoredTask:
    id: str
    score: float
    reasons: List[str]

def _to_vec(x) -> np.ndarray:
    """
    Normalize different embed output types to a float32 numpy vector.
    Supports: numpy arrays, lists, bytes/memoryview (float32).
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, (bytes, bytearray, memoryview)):
        # most embedders here return float32 bytes
        return np.frombuffer(x, dtype=np.float32)
    raise TypeError(f"Unsupported embedding type: {type(x)!r}")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

class TaskBank:
    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("TASK_BANK_PATH", "./data/task_bank.json")
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.cards: List[TaskCard] = [TaskCard(**c) for c in raw]
        self.qbr = QueryBankRuntime()  # for embeddings

    def search(self, text: str, k: int = 3) -> List[ScoredTask]:
        query_vec = _to_vec(self.qbr.embed(text))
        results: List[ScoredTask] = []

        text_l = text.lower()
        for card in self.cards:
            blob = " ".join([
                card.description,
                " ".join(card.examples),
                " ".join(card.synonyms),
            ])
            card_vec = _to_vec(self.qbr.embed(blob))
            score = _cosine(query_vec, card_vec)
            hits = sum(1 for syn in card.synonyms if syn.lower() in text_l)
            score += 0.03 * hits    # small boost for direct synonym matches
            reasons = []
            for syn in card.synonyms:
                if syn.lower() in text_l:
                    reasons.append(f"matched synonym:{syn}")

            results.append(ScoredTask(id=card.id, score=score, reasons=reasons))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
