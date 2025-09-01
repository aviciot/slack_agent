# src/config/providers.py
"""
Central place to read provider/model configuration for each task
(txt2sql, embeddings, summarization, etc.).

Usage:
    from src.config.providers import get_llm_config

    cfg = get_llm_config("txt2sql")
    print(cfg.provider, cfg.model)
"""

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str       # "local" | "cloudflare"
    model: str          # e.g. "qwen2.5-coder:7b-instruct" or "@cf/defog/sqlcoder-7b-2"
    dim: int | None = None  # for embeddings
    base_url: str | None = None
    api_token: str | None = None


def get_llm_config(task: str) -> LLMConfig:
    """
    Return an LLMConfig for the given task.
    task: "txt2sql", "embedding", "summary"
    """

    if task == "txt2sql":
        return LLMConfig(
            provider=os.getenv("TXT2SQL_PROVIDER", "local"),
            model=os.getenv("TXT2SQL_MODEL", "qwen2.5-coder:7b-instruct"),
            base_url=os.getenv("CF_BASE_URL"),
            api_token=os.getenv("CF_API_TOKEN"),
        )

    if task == "embedding":
        return LLMConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "local"),
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            dim=int(os.getenv("EMBED_DIM", "768")),
            base_url=os.getenv("CF_BASE_URL"),
            api_token=os.getenv("CF_API_TOKEN"),
        )

    if task == "summary":
        return LLMConfig(
            provider=os.getenv("SUMMARY_PROVIDER", "cloudflare"),
            model=os.getenv("SUMMARY_MODEL", "@cf/meta/llama-3-8b-instruct"),
            base_url=os.getenv("CF_BASE_URL"),
            api_token=os.getenv("CF_API_TOKEN"),
        )

    raise ValueError(f"Unknown task '{task}'")
