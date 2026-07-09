"""W6: end-to-end mock validation — the full memory chain without a real model.

Tier A (deterministic): drive the integration facade exactly as the Demo patches do —
  message in -> retrieve/inject -> store turn -> window eviction -> retrieve injects
  the remembered fact on the next turn. Real PowerMem (SQLite + mock embedder),
  infer=False, no network.

Tier B (HTTP round-trip): stand up the REAL shim (uvicorn on a real port) backed by a
  stateful mock LLM, point real PowerMem's openai provider at it, and exercise the
  infer=True path. Proves PowerMem talks ONLY to our local shim (zero external API) and
  that an extracted fact becomes retrievable.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
from conftest import local_powermem_config

from minicpm_memory import integration
from minicpm_memory.llm_shim import create_app
from minicpm_memory.memory_layer import MemoryLayer


# =========================================================================== #
# Tier A — deterministic chain through the integration facade
# =========================================================================== #
@pytest.fixture
def enabled_layer(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPM_MEMORY_ENABLED", "1")
    cfg = local_powermem_config(str(tmp_path / "e2e_a.db"))
    layer = MemoryLayer(config=cfg, store_infer=False)  # no LLM
    integration.set_memory_layer(layer)
    yield layer
    integration.set_memory_layer(None)


@pytest.mark.smoke
def test_full_chain_message_evict_store_retrieve(enabled_layer):
    sid = "sess-A"

    # --- turn 1: user speaks; nothing to recall yet ---
    turn1 = [{"role": "user", "content": "My name is Dennis and I work on MiniCPM-o."}]
    q1 = integration.inject_memory(turn1, session_id=sid)
    assert q1 == "My name is Dennis and I work on MiniCPM-o."
    assert all(m["role"] != "system" for m in turn1)  # no memory injected on first turn
    # store the completed turn (as the server.py patch does after inference)
    integration.remember_turn(q1, "Nice to meet you, Dennis!", session_id=sid)

    # --- window eviction promotes an older span to mid-term memory (patch 0002 path) ---
    integration.remember_evicted("[user] The launch deadline is Friday July 19.", session_id=sid)

    # --- turn 2: a new question should now recall the stored facts ---
    turn2 = [{"role": "user", "content": "what is my name and the deadline?"}]
    q2 = integration.inject_memory(turn2, session_id=sid)
    assert q2 == "what is my name and the deadline?"
    assert turn2[0]["role"] == "system", "expected a memory system message to be injected"
    injected = turn2[0]["content"]
    assert "Dennis" in injected or "deadline" in injected.lower()


@pytest.mark.smoke
def test_disabled_chain_is_noop(tmp_path, monkeypatch):
    # With the gate off, the same calls must not touch the messages.
    monkeypatch.delenv("MINICPM_MEMORY_ENABLED", raising=False)
    integration.set_memory_layer(None)
    msgs = [{"role": "user", "content": "hi"}]
    assert integration.inject_memory(msgs, session_id="x") is None
    integration.remember_turn("u", "a", session_id="x")
    integration.remember_evicted("evicted", session_id="x")
    assert msgs == [{"role": "user", "content": "hi"}]


# =========================================================================== #
# Tier B — real shim (uvicorn) + PowerMem infer=True over HTTP
# =========================================================================== #
class StatefulMockLLM:
    """Answers PowerMem's two JSON calls: fact-extraction then action-decision.

    Distinguishes by shape: fact-extraction has a system message; action-decision is a
    single user message. Carries the extracted facts from the first call to the second.
    """

    def __init__(self):
        self.calls = 0
        self._last_facts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, messages, max_new_tokens=512):
        with self._lock:
            self.calls += 1
            roles = [m.get("role") for m in messages]
            if "system" in roles:  # fact extraction
                user = next((m["content"] for m in messages if m.get("role") == "user"), "")
                convo = user.split("Input:", 1)[-1].strip()
                fact = (convo[:200] or "noted").strip()
                self._last_facts = [fact]
                return json.dumps({"facts": [fact]})
            # action decision
            mem = [{"text": f, "event": "ADD"} for f in self._last_facts]
            return json.dumps({"memory": mem})


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def shim_server():
    import uvicorn

    mock = StatefulMockLLM()
    port = _free_port()
    app = create_app(mock)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # wait for startup
    for _ in range(200):
        if getattr(server, "started", False):
            break
        time.sleep(0.02)
    assert getattr(server, "started", False), "shim server failed to start"
    try:
        yield {"url": f"http://127.0.0.1:{port}/v1", "mock": mock}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.mark.smoke
def test_infer_true_end_to_end_via_local_shim(tmp_path, shim_server, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = local_powermem_config(str(tmp_path / "e2e_b.db"), llm_base_url=shim_server["url"])
    layer = MemoryLayer(config=cfg, store_infer=True, raise_errors=True)

    # infer=True => PowerMem makes 2 LLM calls to our LOCAL shim (fact + action)
    layer.store("My name is Dennis and I work on MiniCPM-o.", session_id="s1", role="user")

    # Proves PowerMem routed to our local shim (zero external API).
    assert shim_server["mock"].calls >= 2, "expected fact-extraction + action-decision calls"

    # The extracted fact should now be retrievable.
    out = layer.retrieve("name", session_id="s1")
    assert "Dennis" in out
