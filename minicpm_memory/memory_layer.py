"""W4: MemoryLayer — the single integration surface over PowerMem.

The MiniCPM-o-Demo patches only ever call this class; they never touch PowerMem
directly. Responsibilities:
- ``store()``  — write a turn / evicted span into PowerMem (mid-term Experience).
- ``retrieve()`` — semantic recall, joined and trimmed to a token budget, ready to
  inject as a system message before inference.
- ``distill()`` — promote recent messages into long-term Skills (LLM-only).

Design rules (see AGENTS.md):
- Zero external API: all LLM traffic goes through PowerMem's openai provider pointed
  at our local shim. This class does not talk to any network itself.
- Fail-safe: a memory error must NEVER crash a live conversation turn. ``store`` and
  ``retrieve`` swallow exceptions by default (``raise_errors=False``) and degrade to
  no-op / empty string.
- PowerMem API per the source-verified map in AGENTS.md: ``add``/``search`` return
  ``{"results": [...]}``; search items use key ``"memory"`` (text) and ``"score"``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("minicpm_memory.layer")

DEFAULT_TOKEN_BUDGET = 1500
DEFAULT_RETRIEVE_LIMIT = 5
DEFAULT_USER_ID = "default"


def default_token_estimator(text: str) -> int:
    """Cheap, dependency-free token estimate.

    CJK characters count ~1 token each; other characters ~1 token per 4 chars.
    Good enough to keep injected memory under a budget without pulling a tokenizer.
    """
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return int(cjk + other / 4 + 0.999)


class MemoryLayer:
    def __init__(
        self,
        memory: Any = None,
        *,
        config: Optional[dict] = None,
        default_user_id: str = DEFAULT_USER_ID,
        retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        token_estimator: Optional[Callable[[str], int]] = None,
        store_infer: bool = True,
        raise_errors: bool = False,
    ) -> None:
        """
        Args:
            memory: a ready PowerMem ``Memory`` (or compatible) instance. If ``None``,
                one is built from ``config`` (dict). Injecting a ready/mock instance is
                how tests avoid constructing real PowerMem.
            config: PowerMem config dict, used only when ``memory is None``.
            default_user_id: user/session scope used when a call omits ``session_id``.
            retrieve_limit: max memories fetched per ``retrieve``.
            token_budget: hard cap on the injected memory block (see ``token_estimator``).
            token_estimator: ``str -> int`` token counter; defaults to a CJK-aware heuristic.
            store_infer: default ``infer`` flag for ``store`` (True = PowerMem extracts
                facts via the LLM shim; False = raw store, no LLM).
            raise_errors: if False (default) store/retrieve never raise — a memory
                failure degrades gracefully and the live turn continues.
        """
        if memory is not None:
            self.memory = memory
        else:
            from powermem import Memory  # imported lazily so tests can skip it

            self.memory = Memory(config=config)
        self.default_user_id = default_user_id
        self.retrieve_limit = retrieve_limit
        self.token_budget = token_budget
        self.token_estimator = token_estimator or default_token_estimator
        self.store_infer = store_infer
        self.raise_errors = raise_errors

    # -- write -----------------------------------------------------------------
    def store(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        infer: Optional[bool] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Write one span into memory. Returns PowerMem's result dict, or None on failure."""
        text = (text or "").strip()
        if not text:
            return None
        user = session_id or self.default_user_id
        content = f"[{role}] {text}" if role else text
        use_infer = self.store_infer if infer is None else infer
        try:
            return self.memory.add(
                content, user_id=user, infer=use_infer, metadata=metadata
            )
        except Exception:  # noqa: BLE001 — memory must never break a live turn
            logger.exception("memory.store failed (user=%s)", user)
            if self.raise_errors:
                raise
            return None

    # -- read ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Recall relevant memories, joined newest-relevant-first and trimmed to budget.

        Returns a plain string (possibly empty). Never raises when raise_errors is False.
        """
        query = (query or "").strip()
        if not query:
            return ""
        user = session_id or self.default_user_id
        try:
            res = self.memory.search(query, user_id=user, limit=limit or self.retrieve_limit)
        except Exception:  # noqa: BLE001
            logger.exception("memory.retrieve failed (user=%s)", user)
            if self.raise_errors:
                raise
            return ""
        items = res.get("results", []) if isinstance(res, dict) else (res or [])
        return self._assemble(items)

    def retrieve_as_system_message(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        header: str = "[历史记忆]",
    ) -> Optional[dict]:
        """Convenience: retrieve and wrap as an OpenAI-style system message, or None if empty."""
        body = self.retrieve(query, session_id=session_id)
        if not body:
            return None
        return {"role": "system", "content": f"{header}\n{body}"}

    def _assemble(self, items: list) -> str:
        """Join memory texts (assumed score-desc) up to the token budget."""
        lines: list[str] = []
        used = 0
        for it in items:
            text = it.get("memory", "") if isinstance(it, dict) else str(it)
            text = (text or "").strip()
            if not text:
                continue
            cost = self.token_estimator(text)
            if lines and used + cost > self.token_budget:
                break
            lines.append(text)
            used += cost
        return "\n".join(lines)

    # -- long-term promotion ---------------------------------------------------
    def distill(self, messages: list, *, today: Optional[str] = None) -> list:
        """Distill recent messages into long-term Skills (LLM-only, no storage on SQLite).

        Returns the distilled skill list, or [] on failure.
        """
        try:
            return self.memory.distill_skills(messages, today=today)
        except Exception:  # noqa: BLE001
            logger.exception("memory.distill failed")
            if self.raise_errors:
                raise
            return []
