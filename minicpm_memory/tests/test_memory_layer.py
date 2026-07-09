"""W4 tests: MemoryLayer wrapper.

Mix of real-PowerMem integration (local, infer=False) and pure-logic unit tests
using a FakeMemory so token-budget / fail-safe / delegation behaviour is deterministic
and does not depend on search ranking or an LLM.
"""

from __future__ import annotations

import pytest
from conftest import local_powermem_config

from minicpm_memory.memory_layer import MemoryLayer, default_token_estimator


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeMemory:
    """Stand-in for powermem.Memory with recording + optional failure."""

    def __init__(self, search_results=None, fail=False):
        self.added = []
        self.distilled = None
        self._search_results = search_results or []
        self._fail = fail

    def add(self, content, user_id=None, infer=True, metadata=None):
        if self._fail:
            raise RuntimeError("boom")
        self.added.append({"content": content, "user_id": user_id, "infer": infer})
        return {"results": [{"id": "1", "memory": content, "event": "ADD"}]}

    def search(self, query, user_id=None, limit=30):
        if self._fail:
            raise RuntimeError("boom")
        return {"results": self._search_results[:limit]}

    def distill_skills(self, messages, today=None):
        self.distilled = {"messages": messages, "today": today}
        return [{"title": "t", "description": "d"}]


# --------------------------------------------------------------------------- #
# Integration (real PowerMem, offline)
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_store_and_retrieve_integration(tmp_path):
    cfg = local_powermem_config(str(tmp_path / "layer.db"))
    layer = MemoryLayer(config=cfg, store_infer=False)  # infer=False => no LLM
    layer.store("The product launch is on July 19 at WAIC.", session_id="s1", role="user")
    out = layer.retrieve("launch", session_id="s1")
    assert isinstance(out, str)
    assert "July 19" in out or "launch" in out.lower()


# --------------------------------------------------------------------------- #
# store()
# --------------------------------------------------------------------------- #
def test_store_passes_role_and_infer_flag():
    fake = FakeMemory()
    layer = MemoryLayer(fake, store_infer=True)
    layer.store("hello", session_id="s1", role="assistant")
    assert fake.added[0]["content"] == "[assistant] hello"
    assert fake.added[0]["user_id"] == "s1"
    assert fake.added[0]["infer"] is True
    # per-call override wins
    layer.store("raw", infer=False)
    assert fake.added[1]["infer"] is False


def test_store_skips_empty_text():
    fake = FakeMemory()
    layer = MemoryLayer(fake)
    assert layer.store("   ") is None
    assert fake.added == []


def test_store_is_failsafe():
    fake = FakeMemory(fail=True)
    layer = MemoryLayer(fake, raise_errors=False)
    assert layer.store("x", session_id="s1") is None  # swallowed
    with pytest.raises(RuntimeError):
        MemoryLayer(fake, raise_errors=True).store("x")


# --------------------------------------------------------------------------- #
# retrieve() + token budget
# --------------------------------------------------------------------------- #
def test_retrieve_joins_results():
    fake = FakeMemory(search_results=[{"memory": "a fact"}, {"memory": "b fact"}])
    layer = MemoryLayer(fake)
    assert layer.retrieve("q") == "a fact\nb fact"


def test_retrieve_respects_token_budget():
    # 5 items of 100 chars each; budget 120 tokens with 1-token-per-char estimator
    items = [{"memory": "x" * 100} for _ in range(5)]
    fake = FakeMemory(search_results=items)
    layer = MemoryLayer(
        fake, token_budget=120, token_estimator=len, retrieve_limit=10
    )
    out = layer.retrieve("q")
    # first item (100) fits; second would push to 200 > 120 => stop. Only 1 kept.
    assert out == "x" * 100
    assert len(out) == 100


def test_retrieve_keeps_at_least_one_even_if_over_budget():
    items = [{"memory": "y" * 5000}]
    fake = FakeMemory(search_results=items)
    layer = MemoryLayer(fake, token_budget=10, token_estimator=len)
    assert layer.retrieve("q") == "y" * 5000  # never drop the single best hit


def test_retrieve_failsafe_returns_empty():
    fake = FakeMemory(fail=True)
    layer = MemoryLayer(fake)
    assert layer.retrieve("q") == ""


def test_retrieve_as_system_message():
    fake = FakeMemory(search_results=[{"memory": "user likes tea"}])
    layer = MemoryLayer(fake)
    msg = layer.retrieve_as_system_message("q", header="[MEM]")
    assert msg == {"role": "system", "content": "[MEM]\nuser likes tea"}

    empty = MemoryLayer(FakeMemory(search_results=[]))
    assert empty.retrieve_as_system_message("q") is None


# --------------------------------------------------------------------------- #
# distill()
# --------------------------------------------------------------------------- #
def test_distill_delegates():
    fake = FakeMemory()
    layer = MemoryLayer(fake)
    skills = layer.distill([{"role": "user", "content": "how to deploy"}], today="2026-07-08")
    assert skills == [{"title": "t", "description": "d"}]
    assert fake.distilled["today"] == "2026-07-08"


# --------------------------------------------------------------------------- #
# token estimator
# --------------------------------------------------------------------------- #
def test_default_token_estimator_cjk_vs_ascii():
    assert default_token_estimator("你好世界") == 4  # 4 CJK => ~4 tokens
    assert default_token_estimator("abcd") <= 2  # 4 ascii => ~1 token
