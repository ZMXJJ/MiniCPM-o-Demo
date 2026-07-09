# MiniCPM-o-Demo 记忆层插入点地图（W5 依据）

> 来源：W5 前置代码勘察（subagent，2026-07-08）。checkout：`vendor/MiniCPM-o-Demo/`（upstream OpenBMB）。
> 设计文档三处行号在本 checkout **全部精确命中**：server.py:284 / server.py:263 / modeling_minicpmo_unified.py:2717。

## 插入点 A — turn_based（Chat/Streaming）注入

`py_backend/server.py` → `_push_turn_based`（line 278）。消息转换在 `chat_util.py`（`parse_raw_messages` L91 / `convert_to_model_msgs` L121）。

```python
284  messages = parse_raw_messages(request.messages)      # schema Message 对象
285  model_msgs = convert_to_model_msgs(messages)         # <-- 最终 [{"role","content"}] 列表
286  # >>> 记忆检索注入点：model_msgs.insert(0, {"role":"system","content": retrieved})
287  await asyncio.to_thread(self.backend.chat_prefill,
288      session_id=self.session_id, msgs=model_msgs, ...)
```

- 注入变量：**`model_msgs`**（在 285↔287 之间 insert 一条 system dict）。
- session id：**`self.session_id`**。
- 用户 query：`request.messages` 最后一条 user（前端每轮回传全量历史）。
- 助手回复变量（供 store）：流式 **`full_text`**（`_stream_turn_based` L323/329，done 于 346-356）；非流式 **`text`**（`_non_stream_turn_based` L375，send 385-394）。
- **store 障碍**：助手文本只在两个 helper 内部生成 → store 要么加在两处（356 后 / 383 后），要么把 helper 改成返回 text。

## 插入点 B — Duplex init（trivial）

`py_backend/server.py` → `_init_duplex`（line 243），system prompt 在 `duplex_prepare` 调用：

```python
263  await asyncio.to_thread(self.backend.duplex_prepare,
265      system_prompt_text=_coalesce(params.get("system_prompt"),
267          params.get("instructions"), default="You are a helpful assistant."),
270      ref_audio_path=refs.llm_ref_audio_path, ...)
```

- 注入变量：kwarg **`system_prompt_text`**（L265）——把 retrieve 结果拼进这个字符串。一处编辑，最干净。

## 插入点 C — 滑动窗口驱逐（moderate，改到模型大文件）

`MiniCPMO45/modeling_minicpmo_unified.py`：

```python
2717  def _enforce_text_window(self) -> None:   # 驱逐入口
        ... while total_len > target: self._drop_next_round(cache) ...
```

真正带文本的驱逐在 `_drop_round`（2695-2714）：

```python
2706  for e in entries:
2707      logger.info("Dropped round=%s ... decoded=%s", ..., e.get("decoded"))
2714      self._omni_chunk_history.remove(e)   # <-- remove 之前 e["decoded"] 就是文本
```

- **关键好消息**：驱逐在 Python 层（prefill 期间，非 forward 深处），且 `_register_chunk`（2638）已把 `entry["decoded"] = _safe_decode(...)` 存好——**可拿到真实文本**，不只是 token id。
- 调用点：`non_streaming_prefill`（Chat）在 3189/3213；`streaming_prefill`（Half-Duplex）在 3566/3589。
- **caveats**：(1) `_safe_decode` 用 `skip_special_tokens=False`，字符串含 `<|im_start|>` 等标记，喂给记忆前要清洗；(2) 部分 chunk `decoded=None`（audio/vision 或 assistant 回退注册路径）；(3) 只覆盖 Chat + Half-Duplex，**Duplex 用另一套 decoder 级窗口**（DuplexWindowConfig，token-only，无 `_omni_chunk_history`），覆盖它会 invasive 且缺文本。

## 滑动窗口配置

`MiniCPMO45/utils.py`：
- `StreamingWindowConfig`（Chat+Half-Duplex）：`text_window_high_tokens=8000`（L898）、`text_window_low_tokens=6000`（L899）。
- `DuplexWindowConfig`：dataclass 默认 8000/6000，但 **modeling 运行时 kwargs 覆盖为 4000/3500**（L4335-4336）——Duplex 不要假设 8000。

## 模型句柄（供摘要复用）

`server.py` 全局 `_backend`（`PyTorchBackend`）→ `session.backend` → `_backend.processor.model`（唯一 `MiniCPMO` 实例，`unified.py:1358`）。
- **重要 caveat**：该对象重度有状态（live `llm_past_key_values` / `_omni_chunk_history` / mode / TTS cache）。中途拿它做摘要会污染 live KV/mode，**需要 state 存档-恢复**（模型有 `_save_speculative_snapshot`/restore）或走真正无状态的 `.chat()`。→ 这正是 shim 把 `chat_fn` 设计成**可注入**的原因：生产环境注入的 chat_fn 必须做状态隔离。

## 三种模式 → 插入点覆盖

| 模式 | server 路由 | 覆盖插入点 |
|---|---|---|
| Chat（turn_based） | `_push_turn_based`（278） | A（注入）、C（驱逐 store） |
| Half-Duplex | 仅在 `PyTorchBackend`，本 server.py 未路由 | C |
| Duplex（full_duplex） | `_init_duplex`（243）+ `_push_full_duplex`（396） | B（注入）；驱逐为独立 decoder 机制，不在 C 覆盖内 |

## 建议 patch 顺序

B（trivial）→ A-retrieve（trivial）→ A-store（moderate）→ C（moderate，模型文件）。Duplex 驱逐→记忆 默认 out of scope。
