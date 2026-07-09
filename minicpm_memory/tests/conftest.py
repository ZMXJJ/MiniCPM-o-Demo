"""Shared test helpers: a fully-local PowerMem config (no model, no network)."""

from __future__ import annotations

import os


def local_powermem_config(db_path: str, dims: int = 64, *, llm_base_url: str | None = None) -> dict:
    """PowerMem config for offline tests.

    - SQLite file storage (no server)
    - mock embedder (constant vectors, zero deps; the built-in default needs pyseekdb)
    - openai LLM pointed at ``llm_base_url`` if given (a local shim), else a dead port
      that is never contacted (use with ``infer=False``).
    """
    os.environ.pop("OPENROUTER_API_KEY", None)  # documented base_url hijack guard
    return {
        "vector_store": {
            "provider": "sqlite",
            "config": {
                "database_path": db_path,
                "collection_name": "memories",
                "embedding_model_dims": dims,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "minicpm-o-mock",
                "api_key": "sk-local",
                "openai_base_url": llm_base_url or "http://127.0.0.1:59999/v1",
                "temperature": 0.1,
            },
        },
        "embedder": {"provider": "mock", "config": {"embedding_dims": dims}},
        "intelligent_memory": {"enabled": False},
    }
