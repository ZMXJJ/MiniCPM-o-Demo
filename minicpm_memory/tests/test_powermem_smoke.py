"""W2 smoke test: PowerMem add/search on the real library, fully local.

No model, no network:
- storage = SQLite file (temp per test)
- embedder = ``mock`` (constant vectors, zero deps — the built-in ``default`` embedder
  hard-errors without pyseekdb, which we deliberately don't install)
- LLM = pointed at a dead localhost port but NEVER called (we use ``infer=False``)

This pins down PowerMem 1.1.7's real API shape (method names, return keys) that the
rest of the project depends on, and guards against the README's phantom methods.
"""

from __future__ import annotations

import os

import pytest
from powermem import Memory


def _local_config(db_path: str, dims: int = 64) -> dict:
    # Guard against the documented OpenRouter base_url hijack.
    os.environ.pop("OPENROUTER_API_KEY", None)
    return {
        "vector_store": {
            "provider": "sqlite",
            "config": {
                "database_path": db_path,
                "collection_name": "memories",
                "embedding_model_dims": dims,
            },
        },
        # openai provider constructed but never called (infer=False path).
        "llm": {
            "provider": "openai",
            "config": {
                "model": "noop",
                "api_key": "sk-local",
                "openai_base_url": "http://127.0.0.1:59999/v1",
            },
        },
        "embedder": {"provider": "mock", "config": {"embedding_dims": dims}},
        # deterministic: turn off the Ebbinghaus/importance pipeline for the smoke test.
        "intelligent_memory": {"enabled": False},
    }


@pytest.fixture()
def mem(tmp_path):
    return Memory(config=_local_config(str(tmp_path / "smoke.db")))


@pytest.mark.smoke
def test_real_api_surface_matches_source():
    # Methods the project relies on must exist...
    for name in ("add", "search", "get_all", "delete", "reset", "distill_skills"):
        assert hasattr(Memory, name), f"expected Memory.{name} to exist"
    # ...and the README's phantom methods must NOT (guard against copy-pasting them).
    assert not hasattr(Memory, "distill_all"), "distill_all is README-only, should not exist"
    assert not hasattr(Memory, "add_experience"), "add_experience is README-only, should not exist"


@pytest.mark.smoke
def test_add_infer_false_returns_add_event(mem):
    r = mem.add("My name is Dennis and I work on MiniCPM.", user_id="u1", infer=False)
    assert isinstance(r, dict) and "results" in r
    events = [item.get("event") for item in r["results"]]
    assert "ADD" in events, f"expected an ADD event, got {r}"


@pytest.mark.smoke
def test_get_all_reflects_store(mem):
    mem.add("Dennis likes americano coffee.", user_id="u1", infer=False)
    all_ = mem.get_all(user_id="u1")
    assert "results" in all_
    texts = " ".join(item.get("memory", "") for item in all_["results"])
    assert "coffee" in texts.lower()


@pytest.mark.smoke
def test_search_returns_stored_memory(mem):
    mem.add("The project deadline is Friday July 19.", user_id="u1", infer=False)
    s = mem.search("deadline", user_id="u1", limit=5)
    assert "results" in s
    # search result items use key "memory" for text and "score" for relevance (per source map).
    joined = " ".join(item.get("memory", "") for item in s["results"])
    assert "deadline" in joined.lower() or len(s["results"]) >= 1


@pytest.mark.smoke
def test_reset_clears(mem):
    mem.add("ephemeral note", user_id="u1", infer=False)
    mem.reset()
    all_ = mem.get_all(user_id="u1")
    assert all_.get("results") == [] or len(all_.get("results", [])) == 0
