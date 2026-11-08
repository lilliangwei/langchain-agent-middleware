# agent_02_Langchain_AI — 项目介绍

基于 **LangChain（LCEL）** 的多模型智能问答系统：多 LLM 轮询、多密钥降级、复杂度自动路由、ReAct 工具调用、token 计费、query 缓存、三级长期记忆、RAG 本地知识库，并提供深色发光风格的 Web 问答界面。

## 一、项目定位

一个**生产可用的多模型问答中台**：对上提供 CLI 与 Web 两种交互形态，对下屏蔽五家模型厂商（DeepSeek / 通义 / 智谱 / Moonshot / 本地 Ollama）的差异，用一套 LangChain 链式抽象完成路由、轮询、降级、计费与缓存。

## 项目亮点

- **成本敏感**：简单问题路由到免费/低价模型（glm-4-flash 等），重复提问缓存 0 成本，每次调用精确计费到分
- **高可用**：多模型 × 多密钥二维降级链，单厂商故障、单 key 限流均不影响服务
- **可观测**：每轮回答标注档位/模型/token/成本，计费 JSONL 落盘可审计
- **可扩展**：模型池由 .env 声明式构建，新增厂商只需加 key，无需改代码

## 二、技术栈与框架

| 层 | 选型 |
|----|------|
| LLM 编排框架 | **LangChain**（langchain-core / langchain-openai：ChatOpenAI、LCEL `prompt \| chain`、Runnable.with_fallbacks、bind_tools、流式 AIMessageChunk） |
| 文本切分 | langchain-text-splitters（RecursiveCharacterTextSplitter，中文分隔符定制） |
| 向量库 | ChromaDB（原生客户端，双 collection） |
| Embedding | 通义 text-embedding-v3（1024 维，默认）/ 本地 nomic-embed-text（768 维，可切换） |
| Web 层 | FastAPI + SSE 流式推送；前端为 Uiverse 发光输入框改造的深色主题单页 |
| 配置/存储 | python-dotenv / JSON 文件 / 可选 Redis 会话后端 |

> 说明：LLM 调用链 100% 经过 LangChain 抽象；向量库与 embedding 选择直连（精细控制 score 与双通道降级），属有意取舍而非遗漏。

## 三、系统架构

```
                        ┌─────────────────────────────┐
                        │        交互层               │
                        │  CLI: python main.py qa     │
                        │  Web: web_api.py (FastAPI)  │
                        └──────────────┬──────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
      QueryCache 精确缓存      ComplexityRouter 规则打分    实时关键词识别
      命中 → 0 成本返回        simple / complex 档          命中 → ReActAgent
                                       │                   （联网工具链）
                                       ▼                        │
              ┌──────────────────────────────────┐             │
              │   LLMPool（llm_pool.py）          │◄────────────┘
              │   档内多模型轮询 + 模型内多key轮询  │  bind_tools
              │   with_fallbacks 降级链：          │
              │   主key→同模型余key→同档模型→跨档  │
              └────────────────┬─────────────────┘
                               ▼
                  ChatOpenAI × N（模型×key）
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                 ▼
        UsageTracker      ReActAgent        RAGAgent
        计费 JSONL        calculator/       DocumentLoader
                          search/           （含多模态OCR）
                          knowledge_search  → Chroma 检索 → 来源标注
                               │
                    MemoryManager 三级记忆
              短期(内存/Redis) → 摘要(JSON+向量) → 长期召回
```

## 四、功能体系

### 1. 智能问答（核心）
- **复杂度自动路由**：规则打分（长度/推理关键词/多问题/代码/算式），零 LLM 成本。`你好`→simple 档便宜模型；`逐步推导快排复杂度`→complex 档强模型
- **多模型轮询**：每次请求轮转主模型，天然负载均衡与单点故障隔离
- **多密钥降级**：同厂商多 key 轮询首发；任意异常沿降级链自动切换，全程无感
- **流式输出**：CLI 与 Web 均为逐 token 渲染

### 2. ReAct 工具链（Agent）
| 工具 | 能力 |
|------|------|
| calculator | 数学表达式计算 |
| search | 联网搜索（火山引擎豆包搜索），天气/新闻/股价等实时信息 |
| knowledge_search | 检索本地 RAG 知识库 |

Web 端按实时关键词自动分流：天气/新闻类 → 工具链；普通问答 → 纯问答链。

### 3. 三级长期记忆
- **短期**：内存或 Redis（`MEMORY_BACKEND` 切换，Redis 故障自动降级内存）
- **摘要**：超 token 阈值（默认 2000）触发 LLM 摘要，短期压缩为「摘要 + 最近 4 条」
- **长期向量**：摘要入 Chroma，每轮对话语义召回 top-3 注入 system prompt

### 4. RAG 本地知识库
- 摄入 txt/md/pdf/docx/xlsx/csv；扫描版 PDF 可走多模态 OCR（`ENABLE_OCR`）
- LangChain 递归切分（中文分隔符定制）→ 向量化 → 检索 top-k → 强制 `[来源N]` 标注生成

### 5. 成本与性能
- **Token 计费**：usage × 单价表 → JSONL 落盘，按模型聚合成报表（CLI `/cost`，Web `GET /api/cost`）
- **Query 缓存**：归一化 + SHA256 + 7 天 TTL，重复提问 0 成本瞬时返回

## 五、快速开始

```bash
pip install -r requirements.txt   # langchain fastapi uvicorn chromadb 等
# 编辑 .env 填入各厂商 API key（多 key 逗号分隔）

python main.py qa       # CLI 问答（命令：/cost /cache /clear /quit）
python main.py agent    # ReAct 工具型 Agent 演示（多轮记忆 + 计算 + 联网）
python main.py rag      # RAG 文档问答（自动摄入 data/docs/）
python web_api.py       # Web 界面 → http://127.0.0.1:8000
```

## 六、目录结构

```
agent_02_Langchain_AI/
├── main.py                # CLI 唯一入口：argparse 子命令 qa / agent / rag
├── web_api.py             # Web 层：FastAPI + SSE 流式 + 工具分流
├── llm_pool.py            # 统一 LLM 层：模型池 + 复杂度路由器 + 视觉客户端
├── agent/
│   ├── react_agent.py     # ReAct 循环（bind_tools + tool_calls）
│   └── tools.py           # 全部工具 + 注册器
├── memory/
│   ├── document_loader.py # 文档加载 + LangChain 切分 + 多模态 OCR
│   ├── memory_manager.py  # 三级记忆编排
│   ├── rag_agent.py       # RAG 问答
│   ├── session_backend.py # 内存/Redis 会话后端
│   └── vector_store.py    # Chroma 双 collection + 双通道 embedding
├── utils/                 # logger / token 计数 / llm_extras（计费+缓存）
├── static/index.html      # Web 问答界面（深色发光主题）
├── data/                  # chroma_db / docs / 计费与缓存 JSON（已 gitignore）
└── documents/             # 项目文档
```

## 七、关键配置（.env）

| 配置 | 默认 | 说明 |
|------|------|------|
| `DEEPSEEK/QWEN/ZHIPU/MOONSHOT_API_KEY` | - | 多密钥逗号分隔 |
| `{ALIAS}_MODEL` / `{ALIAS}_TIER` | 见 llm_pool | 模型名 / 档位覆盖 |
| `ROUTER_COMPLEX_THRESHOLD` | 3 | 路由复杂度阈值 |
| `EMBEDDING_PROVIDER` | qwen | embedding 通道（qwen/ollama） |
| `MEMORY_BACKEND` | memory | 短期记忆后端 memory/redis |
| `MEMORY_SUMMARY_TOKEN_THRESHOLD` | 2000 | 摘要触发阈值（=0 退回按条数） |
| `QUERY_CACHE_TTL_SECONDS` | 604800 | 缓存 TTL |
| `VOLC_SEARCH_API_KEY` | - | 联网搜索工具密钥 |
| `ENABLE_OCR` | false | 扫描版 PDF OCR 开关 |

## 八、设计决策（为什么这么选）

1. **LLM 层统一走 LangChain，外围自研**：LangChain 负责模型抽象与链组合（它最擅长的），向量库/embedding/计费直连避免包装层版本耦合
2. **`max_retries=0` + with_fallbacks**：失败立刻切下一候选，不在坏 key 上指数退避空转
3. **路由用规则而非 LLM 分类**：零成本零延迟；误判代价低（跨档互为兜底）
4. **embedding 固定通道不参与轮询**：向量维度一致性是硬约束，切换必须清库重摄入
5. **工具链 token 暂不计费**：ReAct 中间轮 usage 不在流式末块，只记调用次数（后续可加各轮累加）

## 九、扩展路线

- 接 **LangSmith** 做链路观测（现有 Runnable 链无缝嵌入）
- 迁 **LangGraph** 编排多步 Agent（规划-执行-反思）
- 缓存升级语义近似匹配（向量相似度判重）
- RAG 高级检索：MultiQuery / 上下文压缩（届时再将向量库迁入 langchain-chroma 生态）

## 十、环境与依赖

- **Python ≥ 3.11**（代码使用 `str | List[str]` 等新语法）
- 核心依赖：`langchain` `langchain-openai` `langchain-text-splitters` `chromadb` `fastapi` `uvicorn` `openai` `python-dotenv` `requests` `tiktoken` `pypdf`
- 可选组件：
  - **Ollama**（本地模型 + 本地 embedding + 视觉 OCR，不装则自动全程走云端）
  - **Redis**（短期记忆持久化，不配则 `MEMORY_BACKEND=memory` 用内存）

## 十一、Web API 接口

启动 `python web_api.py` 后：

| 接口 | 说明 |
|------|------|
| `GET /` | 问答页面（static/index.html） |
| `POST /api/chat` | 流式问答，body：`{"question": "...", "session_id": "..."}`，返回 SSE |
| `POST /api/clear?session_id=` | 清空指定会话历史 |
| `GET /api/cost` | 计费汇总（按模型分组 + 总计 + 缓存命中数） |

**SSE 事件格式**（`\n\n` 分隔）：

```
event: meta   data: {"tier": "simple", "primary": "qwen"}       ← 路由结果（agent 链为 react-agent）
event: token  data: {"text": "..."}                              ← 逐块正文（多个）
event: done   data: {"model": "qwen-plus", "tokens": 48, "cost": 4.9e-05}
event: error  data: {"message": "..."}                           ← 全部降级耗尽时
```

## 十二、使用效果示例

```text
问：你好
→ simple 档 · qwen-plus · 48 tokens · ≈¥0.000049

问：请逐步推导为什么快速排序平均复杂度是 O(nlogn)
→ complex 档 · deepseek-chat（自动识别推理诉求）

问：青岛现在气温多少
→ react-agent（识别实时关键词）→ search 工具联网检索
→ “青岛当前气温 12℃，寒潮蓝色预警，今夜降温约10℃……”（真实数据）

再问：你好
→ 缓存命中 · 0 tokens · 0 成本（瞬时返回）
```

## 十三、常见问题（FAQ）

1. **所有模型都失败？** 检查 .env key 是否有效、余额是否充足；ollama 未启动属预期（会自动降级云端）
2. **RAG 检索报错/结果异常？** 是否切换过 `EMBEDDING_PROVIDER`——两通道向量维度不同（1024 vs 768），切换后必须删除 `data/chroma_db` 重新摄入
3. **记忆不召回？** 确认 `ENABLE_MEMORY=true`，且对话轮次/token 量已达到摘要触发阈值
4. **扣费与官网账单对不上？** 内置单价为参考价，以 `MODEL_PRICES_JSON` 环境变量覆盖为官网最新价
5. **密钥安全**：`.env` 与 `data/` 均已入 `.gitignore`，切勿提交；多 key 用逗号分隔写同一变量（重复写同名变量会被后者覆盖）
