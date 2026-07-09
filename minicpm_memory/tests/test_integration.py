"""W5 tests: integration facade (opt-in gate, injection, store, duplex augment).

Pure-logic, no model. Uses a FakeLayer and toggles the MINICPM_MEMORY_ENABLED gate.
The critical property tested here: when memory is DISABLED, every helper is a no-op and
does not touch model_msgs — i.e. the patched Demo behaves exactly like upstream.
"""

from __future__ import annotations

import pytest

from minicpm_memory import integration


class FakeLayer:
    def __init__(self, recall="user likes tea"):
        self.recall = recall
        self.stored = []

    def retrieve(self, query, session_id=None):
        return self.recall

    def retrieve_as_system_message(self, query, session_id=None, header="[历史记忆]"):
        if not self.recall:
            return None
        return {"role": "system", "content": f"{header}\n{self.recall}"}

    def store(self, text, session_id=None, role=None):
        self.stored.append({"text": text, "session_id": session_id, "role": role})


@pytest.fixture(autouse=True)
def _reset_integration():
    integration.set_memory_layer(None)
    yield
    integration.set_memory_layer(None)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("MINICPM_MEMORY_ENABLED", "1")
    layer = FakeLayer()
    integration.set_memory_layer(layer)
    return layer


# --------------------------------------------------------------------------- #
# opt-in gate: DISABLED by default => no-ops, model_msgs untouched
# --------------------------------------------------------------------------- #
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MINICPM_MEMORY_ENABLED", raising=False)
    assert integration.memory_enabled() is False
    assert integration.get_memory_layer() is None

    msgs = [{"role": "user", "content": "hello"}]
    assert integration.inject_memory(msgs, session_id="s1") is None
    assert msgs == [{"role": "user", "content": "hello"}]  # untouched

    integration.remember_turn("u", "a", session_id="s1")  # no error, no-op
    assert integration.augment_duplex_system_prompt("base", session_id="s1") == "base"


# --------------------------------------------------------------------------- #
# latest_user_text
# --------------------------------------------------------------------------- #
def test_latest_user_text_variants():
    assert integration.latest_user_text([{"role": "user", "content": "hi"}]) == "hi"
    # newest user wins
    assert (
        integration.latest_user_text(
            [{"role": "user", "content": "old"}, {"role": "assistant", "content": "x"},
             {"role": "user", "content": "new"}]
        )
        == "new"
    )
    # multimodal list -> text parts only
    mm = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "caption"}]}]
    assert integration.latest_user_text(mm) == "caption"
    assert integration.latest_user_text([{"role": "assistant", "content": "a"}]) == ""


# --------------------------------------------------------------------------- #
# enabled behaviour
# --------------------------------------------------------------------------- #
def test_inject_memory_prepends_system_and_returns_query(enabled):
    msgs = [{"role": "user", "content": "what did I say?"}]
    query = integration.inject_memory(msgs, session_id="s1")
    assert query == "what did I say?"
    assert msgs[0]["role"] == "system"
    assert "user likes tea" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_inject_memory_no_user_message(enabled):
    msgs = [{"role": "system", "content": "sys"}]
    assert integration.inject_memory(msgs, session_id="s1") is None
    assert len(msgs) == 1  # nothing inserted


def test_inject_memory_no_recall_does_not_insert(monkeypatch):
    monkeypatch.setenv("MINICPM_MEMORY_ENABLED", "1")
    integration.set_memory_layer(FakeLayer(recall=""))  # nothing to recall
    msgs = [{"role": "user", "content": "hi"}]
    q = integration.inject_memory(msgs, session_id="s1")
    assert q == "hi"  # query returned for later store
    assert len(msgs) == 1  # but no system message inserted


def test_remember_turn_stores_both(enabled):
    integration.remember_turn("my name is Dennis", "nice to meet you", session_id="s1")
    roles = [(s["role"], s["text"]) for s in enabled.stored]
    assert ("user", "my name is Dennis") in roles
    assert ("assistant", "nice to meet you") in roles


def test_augment_duplex_system_prompt(enabled):
    out = integration.augment_duplex_system_prompt("You are helpful.", session_id="s1")
    assert out.startswith("You are helpful.")
    assert "user likes tea" in out
