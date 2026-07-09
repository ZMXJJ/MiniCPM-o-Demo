"""W3 tests: OpenAI-compatible LLM shim.

All local, no model, no network — the chat/embed callables are mocks.
Covers: protocol shape, token capping, embeddings on/off, and — most importantly —
that the shared inference lock actually serialises concurrent model calls.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest
from httpx import ASGITransport

from minicpm_memory.llm_shim import create_app


def _echo_chat(messages, max_new_tokens=512):
    # Return something that lets tests see what the shim forwarded.
    last = messages[-1]["content"] if messages else ""
    return f"echo(cap={max_new_tokens}):{last}"


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://shim")


@pytest.mark.asyncio
async def test_chat_completion_shape():
    app = create_app(_echo_chat)
    async with _client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"].endswith(":hi")
    assert body["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_max_tokens_capped():
    # Request an absurd max_tokens; shim must clamp to its ceiling (512).
    app = create_app(_echo_chat, max_new_tokens=512)
    async with _client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "q"}], "max_tokens": 100000},
        )
    assert "cap=512" in r.json()["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_models_endpoint():
    app = create_app(_echo_chat, model_name="minicpm-o-4.5")
    async with _client(app) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "minicpm-o-4.5"


@pytest.mark.asyncio
async def test_embeddings_disabled_by_default():
    app = create_app(_echo_chat)
    async with _client(app) as c:
        r = await c.post("/v1/embeddings", json={"input": "hello"})
    assert r.status_code == 501


@pytest.mark.asyncio
async def test_embeddings_when_configured():
    def fake_embed(texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    app = create_app(_echo_chat, embed_fn=fake_embed)
    async with _client(app) as c:
        r = await c.post("/v1/embeddings", json={"input": ["ab", "cde"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data[0]["embedding"][0] == 2.0
    assert data[1]["embedding"][0] == 3.0


@pytest.mark.asyncio
async def test_inference_lock_serialises_calls():
    """With a shared lock, concurrent requests must NOT run chat_fn in parallel."""
    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    def slow_chat(messages, max_new_tokens=512):
        with guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with guard:
            state["active"] -= 1
        return "ok"

    lock = threading.Lock()
    app = create_app(slow_chat, inference_lock=lock)
    async with _client(app) as c:
        import asyncio

        reqs = [
            c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": str(i)}]})
            for i in range(5)
        ]
        results = await asyncio.gather(*reqs)
    assert all(r.status_code == 200 for r in results)
    # The lock must have prevented any overlap.
    assert state["max_active"] == 1, f"lock failed to serialise: max_active={state['max_active']}"
