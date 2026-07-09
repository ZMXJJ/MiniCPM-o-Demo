"""OpenAI-compatible shim that lets PowerMem drive MiniCPM-o for memory work.

Design goals (see project AGENTS.md rules):
- ZERO external API: this shim is the ONLY LLM endpoint PowerMem talks to, and it
  forwards to a locally-loaded MiniCPM-o instance. Point PowerMem's
  ``OPENAI_LLM_BASE_URL`` at ``http://127.0.0.1:<port>/v1``.
- No second model: the chat callable is *injected*. In production it wraps the
  already-loaded MiniCPM-o ``model.chat``; in tests it is a mock. The shim itself
  never imports torch or a model.
- Serialised inference: HF Transformers cannot ``generate`` concurrently, so every
  model call is taken under a shared inference lock. The demo passes in the SAME
  lock it uses for the main dialog loop, so memory work never races real-time turns.

The shim intentionally implements the minimum OpenAI surface PowerMem needs:
``POST /v1/chat/completions``, optional ``POST /v1/embeddings`` and ``GET /v1/models``.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# ---- injected callable types -------------------------------------------------
# chat_fn(messages, max_new_tokens=...) -> assistant_text
ChatFn = Callable[..., str]
# embed_fn(texts) -> list of vectors
EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MODEL_NAME = "minicpm-o-4.5"


# ---- OpenAI-ish request/response schemas ------------------------------------
class _ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[_ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # PowerMem may send response_format / tools etc.; we ignore unknown fields.

    model_config = {"extra": "allow"}


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: object  # str | list[str]

    model_config = {"extra": "allow"}


def create_app(
    chat_fn: ChatFn,
    *,
    embed_fn: Optional[EmbedFn] = None,
    inference_lock: Optional[threading.Lock] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    model_name: str = DEFAULT_MODEL_NAME,
) -> FastAPI:
    """Build the shim app.

    Args:
        chat_fn: blocking callable ``(messages, max_new_tokens=int) -> str``. Wraps the
            real MiniCPM-o model in production; a mock in tests.
        embed_fn: optional blocking callable ``(texts) -> list[vector]``. Only needed if
            PowerMem is configured to fetch embeddings from this endpoint.
        inference_lock: shared lock serialising all model calls. If ``None`` a private
            lock is created (fine for standalone/testing; production MUST pass the demo's
            lock so memory work and the live dialog never generate concurrently).
        max_new_tokens: hard cap per memory generation, so a summary can never hog the GPU.
        model_name: label returned by ``/v1/models`` and echoed in responses.
    """
    lock = inference_lock if inference_lock is not None else threading.Lock()
    app = FastAPI(title="minicpm-o memory LLM shim")

    def _locked_chat(messages: list[dict], cap: int) -> str:
        # Runs in a threadpool worker; the lock guarantees single-flight inference.
        with lock:
            return chat_fn(messages, max_new_tokens=cap)

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "openbmb"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> dict:
        # Cap requested tokens at our ceiling so memory work stays bounded.
        requested = req.max_tokens or max_new_tokens
        cap = min(int(requested), max_new_tokens)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        text = await run_in_threadpool(_locked_chat, messages, cap)
        now = int(time.time())
        return {
            "id": f"chatcmpl-shim-{now}",
            "object": "chat.completion",
            "created": now,
            "model": req.model or model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest) -> dict:
        if embed_fn is None:
            raise HTTPException(
                status_code=501,
                detail="embeddings not configured on this shim (no embed_fn injected)",
            )
        raw = req.input
        texts = [raw] if isinstance(raw, str) else list(raw)
        vectors = await run_in_threadpool(_embed_locked, texts)
        return {
            "object": "list",
            "model": req.model or model_name,
            "data": [
                {"object": "embedding", "index": i, "embedding": list(v)}
                for i, v in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def _embed_locked(texts: list[str]):
        with lock:
            return embed_fn(texts)

    return app
