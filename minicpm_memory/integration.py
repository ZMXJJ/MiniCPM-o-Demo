"""Integration facade between MiniCPM-o-Demo and the MemoryLayer.

The Demo patches import ONLY from this module and call thin helpers that are
**no-ops unless memory is explicitly enabled** (env ``MINICPM_MEMORY_ENABLED=1``).
This guarantees the patched Demo behaves identically to upstream by default — the
memory layer is strictly opt-in.

Deployment note (Linux+GPU): set ``MINICPM_MEMORY_ENABLED=1`` and the PowerMem env
(``LLM_PROVIDER=openai``, ``OPENAI_LLM_BASE_URL=http://127.0.0.1:8003/v1`` → the shim,
``DATABASE_PROVIDER=seekdb`` or ``sqlite``, ``EMBEDDING_PROVIDER=...``). The shim's
``chat_fn`` must reuse the MiniCPM-o model with proper state isolation (see
INSERTION-POINTS.md §6). None of that runs on a Mac dev box; hence the opt-in gate.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("minicpm_memory.integration")

_lock = threading.Lock()
_layer = None  # lazily-built singleton MemoryLayer
_MISSING = object()


def memory_enabled() -> bool:
    return os.getenv("MINICPM_MEMORY_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


def set_memory_layer(layer) -> None:
    """Inject a ready MemoryLayer (used by tests and by custom bootstrap code)."""
    global _layer
    with _lock:
        _layer = layer


def get_memory_layer():
    """Return the shared MemoryLayer, or None if memory is disabled/unbuildable."""
    global _layer
    if not memory_enabled():
        return None
    if _layer is None:
        with _lock:
            if _layer is None:
                _layer = _build_from_env()
    return _layer


def _build_from_env():
    """Build a MemoryLayer from env. PowerMem reads LLM/embedder/storage from env itself."""
    try:
        from .memory_layer import MemoryLayer
        from powermem import create_memory

        # create_memory() reads .env / process env (auto_config) — provider, base_url, storage.
        memory = create_memory()
        budget = int(os.getenv("MINICPM_MEMORY_TOKEN_BUDGET", "1500"))
        infer = os.getenv("MINICPM_MEMORY_STORE_INFER", "1").strip().lower() in ("1", "true", "yes", "on")
        return MemoryLayer(memory=memory, token_budget=budget, store_infer=infer)
    except Exception:  # noqa: BLE001 — never let memory bootstrap crash the server
        logger.exception("failed to build MemoryLayer; memory disabled for this process")
        return None


# --------------------------------------------------------------------------- #
# Helpers the Demo patches call
# --------------------------------------------------------------------------- #
def latest_user_text(model_msgs: list) -> str:
    """Extract the newest user message's text from a list of {role, content} dicts.

    Content may be a plain string or a multimodal list; we keep only text parts.
    """
    for msg in reversed(model_msgs or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            if parts:
                return "\n".join(parts)
    return ""


def inject_memory(model_msgs: list, *, session_id: str) -> Optional[str]:
    """Retrieve memory for the latest user turn and prepend it as a system message.

    Mutates ``model_msgs`` in place. Returns the user text used (for a later
    ``remember_turn``), or None when memory is disabled / nothing to inject.
    No-op and safe when disabled.
    """
    layer = get_memory_layer()
    if layer is None:
        return None
    query = latest_user_text(model_msgs)
    if not query:
        return None
    sys_msg = layer.retrieve_as_system_message(query, session_id=session_id)
    if sys_msg is not None:
        model_msgs.insert(0, sys_msg)
    return query


def remember_turn(user_text: str, assistant_text: str, *, session_id: str) -> None:
    """Store a completed (user, assistant) turn into memory. No-op when disabled."""
    layer = get_memory_layer()
    if layer is None:
        return
    if user_text:
        layer.store(user_text, session_id=session_id, role="user")
    if assistant_text:
        layer.store(assistant_text, session_id=session_id, role="assistant")


def remember_evicted(text: str, *, session_id: str) -> None:
    """Store an evicted (sliding-window-dropped) span — the JoyAI-style mid-term trigger.

    Used by the optional model-file eviction patch (INSERTION-POINTS §C) as an
    alternative to per-turn ``remember_turn``. No-op when disabled.
    """
    layer = get_memory_layer()
    if layer is None:
        return
    if text:
        layer.store(text, session_id=session_id)


def augment_duplex_system_prompt(base_prompt: str, *, session_id: str, query: Optional[str] = None) -> str:
    """Return the duplex system prompt with long-term memory prepended. No-op when disabled."""
    layer = get_memory_layer()
    if layer is None:
        return base_prompt
    recall_query = query or base_prompt
    body = layer.retrieve(recall_query, session_id=session_id)
    if not body:
        return base_prompt
    return f"{base_prompt}\n\n[历史记忆]\n{body}"
