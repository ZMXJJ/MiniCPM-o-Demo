# MiniCPM-o Demo · 三层长期记忆集成

为 MiniCPM-o Demo 增加**长期记忆能力**，突破 Demo 层 8K context 滑动窗口导致的"聊几分钟就忘"。借鉴 JoyAI-VL 的短期/中期/长期分层思路，用 [PowerMem](https://github.com/oceanbase/powermem) 托管中长期记忆。

> 灵感来自面壁智能 & OceanBase 生态合作（2026-07）。

## 三条设计铁律

1. **零外部 API**：绝对不调用任何云端 LLM。记忆的事实抽取/蒸馏只走本地 shim 复用 MiniCPM-o。
2. **不自建 RAG**：存储/提取/去重/蒸馏/衰减/检索全部由 PowerMem 内置能力完成。
3. **不部署第二个模型**：记忆摘要复用主模型，通过共享推理锁串行化，避免与实时对话争 GPU。

## 三层记忆

| 层级 | 载体 | 写入时机 | 实现 |
|---|---|---|---|
| 短期 | MiniCPM-o 8K context / KV Cache | 每轮实时 | Demo 原有机制，不改 |
| 中期 | PowerMem Experience 层 | 每轮完成后 / 滑动窗口驱逐前 | PowerMem `add(infer=True)` 调本地 shim → MiniCPM-o 抽取事实 |
| 长期 | PowerMem Skill 层 + 艾宾浩斯衰减 | 会话结束/定期蒸馏 | PowerMem `distill_skills`；衰减为纯本地算法 |

## 代码改动（本分支已应用）

| 文件 | 改动 | 说明 |
|---|---|---|
| `py_backend/server.py` | 插入点 A（`_push_turn_based`）+ B（`_init_duplex`） | 推理前检索注入 system 记忆、推理后存本轮；duplex system_prompt 增强 |
| `MiniCPMO45/modeling_minicpmo_unified.py` | 插入点 C（`_drop_round`） | 可选：滑动窗口驱逐旧轮次前，经实例 hook 上报被驱逐文本 |
| `minicpm_memory/` | 新增包 | shim / memory_layer / integration facade |

**默认零行为改动**：三处插入点全部 opt-in，只有设置 `MINICPM_MEMORY_ENABLED=1` 才生效，否则与上游逐字节一致。插入点行号依据见 `INSERTION-POINTS.md`。

## 组件

| 文件 | 作用 |
|---|---|
| `minicpm_memory/llm_shim.py` | OpenAI 兼容 shim：`chat_fn` 可注入 + 共享推理锁 + `max_tokens` 封顶 |
| `minicpm_memory/memory_layer.py` | PowerMem 封装：`store`/`retrieve`（token 预算裁剪）/`distill` + fail-safe |
| `minicpm_memory/integration.py` | Demo 唯一调用面，默认 no-op（`MINICPM_MEMORY_ENABLED` 门控） |

## 启用（Linux + GPU 真机）

```bash
pip install -r minicpm_memory/requirements.txt

export MINICPM_MEMORY_ENABLED=1
# PowerMem 的 LLM 指向本地 shim（复用 MiniCPM-o，需状态隔离，见 INSERTION-POINTS §6）
export LLM_PROVIDER=openai
export OPENAI_LLM_BASE_URL=http://127.0.0.1:8003/v1
export LLM_API_KEY=sk-local
export LLM_MODEL=minicpm-o-4.5
# 存储：Linux 用 seekdb，mac 开发用 sqlite
export DATABASE_PROVIDER=seekdb
# Embedder：本地服务（禁止云端）
export EMBEDDING_PROVIDER=openai
export OPENAI_EMBEDDING_BASE_URL=http://127.0.0.1:8003/v1
```

> ⚠️ shim 的 `chat_fn` 复用主模型时必须做 **KV/mode 状态隔离**（save/restore），否则会污染 live 对话——这是唯一需在真机验证的关键点。

## 测试（本地即可，无模型无网络）

```bash
pip install -r minicpm_memory/requirements.txt
pytest -c pytest_memory.ini        # 32 passed
```

覆盖：PowerMem API 契约冒烟、shim（含并发锁串行化）、memory_layer（token 预算 + fail-safe）、integration facade（禁用即 no-op）、端到端（真起 uvicorn shim + PowerMem `infer=True` HTTP 往返，证明只打本地 shim）。

本地测试用 SQLite + `embedder=mock`（PowerMem 默认 embedder 需 pyseekdb，本集成不装）。真机换成 SeekDB + 本地 embedding。
