"""
统一入口：LangChain 多模型智能问答系统。

子命令：
    python main.py qa     交互式智能问答（路由 + 轮询 + 降级 + 计费 + 缓存）
    python main.py agent  ReAct 工具型 Agent 演示（计算器/联网搜索/知识库检索 + 长期记忆）
    python main.py rag    RAG 本地文档问答演示（摄入 → 检索 → 标注来源生成）

不带子命令时打印本帮助。
"""
import argparse
import os


# ==================== qa：交互式智能问答 ====================
def cmd_qa():
    """
    LangChain 多模型轮询 + 多密钥降级 · 智能问答机器人。

    特性：
    - 复杂度自动路由：简单问题走 simple 档（便宜 flash），复杂推理走 complex 档（pro/reasoner）
    - 多模型轮询 + 多密钥降级：主 key 失败切同模型下一 key，全部失败跨模型降级
    - Token 计费统计：stream_usage 采集 usage × 单价表，JSONL 落盘，/cost 查看
    - Query 缓存：重复提问 0 成本直接返回，/cache 查看缓存状态
    - LCEL 链：prompt | model(with_fallbacks)，多轮记忆，流式输出
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    from llm_pool import build_default_pool, ComplexityRouter
    from utils.llm_extras import UsageTracker, QueryCache

    SYSTEM_PROMPT = "你是一个智能问答助手，回答准确、简洁、有条理。支持多轮对话，能记住上下文。"
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    def print_cost(tracker: UsageTracker):
        s = tracker.summary()
        print("--- Token 计费统计 ---")
        for model, b in s["by_model"].items():
            print(f"  {model:<18} 调用{b['calls']:>3}次  token={b['total_tokens']:>6}  成本≈¥{b['cost_cny']:.4f}")
        t = s["total"]
        print(f"  合计：{t['calls']} 次调用，{t['total_tokens']} tokens，≈¥{t['cost_cny']:.4f}，缓存命中 {t['cache_hits']} 次")

    pool = build_default_pool()
    router = ComplexityRouter()
    tracker = UsageTracker()
    cache = QueryCache()
    history: list = []

    print("=" * 56)
    print("LangChain 智能问答机器人（路由 + 轮询 + 降级 + 计费 + 缓存）")
    for d in pool.describe():
        print(f"  [{d['tier']:<7}] {d['alias']:<9} {d['model']}")
    print("命令：/cost 计费统计  /cache 缓存状态  /clear 清历史  /quit 退出")
    print("=" * 56)

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question == "/quit":
            print("再见！")
            break
        if question == "/clear":
            history.clear()
            print("[历史已清空]")
            continue
        if question == "/cost":
            print_cost(tracker)
            continue
        if question == "/cache":
            print(f"[缓存条数：{cache.size}]（/clearcache 清空）")
            continue
        if question == "/clearcache":
            cache.clear()
            print("[缓存已清空]")
            continue

        # ---- 1. 缓存命中：0 成本直接返回 ----
        cached = cache.get(question)
        if cached:
            print(f"机器人（缓存命中）：{cached}")
            tracker.record(model="cache", alias="cache", usage=None, cached=True)
            continue

        # ---- 2. 复杂度路由 → 档位内轮询 + 降级链 ----
        tier = router.classify(question)
        model_chain, primary = pool.get_chain(tier=tier)
        chain = prompt | model_chain

        print(f"机器人（{tier} 档 · 主模型 {primary}）：", end="", flush=True)
        collected = None
        try:
            for chunk in chain.stream({"question": question, "history": history}):
                # 聚合 AIMessageChunk：末块携带 usage_metadata（stream_usage=True）
                collected = chunk if collected is None else collected + chunk
                if chunk.content:
                    print(chunk.content, end="", flush=True)
        except Exception as e:
            # 所有模型 + 所有 key 均失败才会走到这里
            print(f"\n【全部模型调用失败】{e}")
            continue
        print()

        answer = collected.content if collected else ""
        # ---- 3. 计费统计（response_metadata 里有真实模型名）----
        usage = getattr(collected, "usage_metadata", None) if collected else None
        served_model = (
            (collected.response_metadata or {}).get("model_name", primary)
            if collected else primary
        )
        rec = tracker.record(model=served_model, alias=primary, usage=usage, tier=tier)
        if rec and rec["total_tokens"]:
            print(f"[tokens: {rec['input_tokens']}+{rec['output_tokens']}"
                  f"={rec['total_tokens']}，≈¥{rec['cost_cny']:.6f}]")

        # ---- 4. 写缓存 + 历史 ----
        cache.set(question, answer)
        history.extend([HumanMessage(content=question), AIMessage(content=answer)])


# ==================== agent：ReAct 工具型 Agent 演示 ====================
def cmd_agent():
    """多轮对话演示：验证长期记忆召回与工具调用能力"""
    from dotenv import load_dotenv

    from agent import ReActAgent
    from llm_pool import build_default_pool

    load_dotenv()
    pool = build_default_pool()

    # ==================== 长期记忆系统（可选，ENABLE_MEMORY 开关控制）====================
    ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
    memory_manager = None
    if ENABLE_MEMORY:
        try:
            from memory.memory_manager import MemoryManager
            memory_manager = MemoryManager(
                pool=pool,
                summary_tier="simple",
                summary_threshold=int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "8")),
                keep_recent=int(os.getenv("MEMORY_KEEP_RECENT", "4")),
            )
            print(f"[记忆系统] 已启用：session={os.getenv('AGENT_SESSION_ID', 'default')}")
        except Exception as e:
            # 记忆系统初始化失败不应阻塞 Agent 主流程，降级为无记忆模式
            print(f"[记忆系统] 初始化失败，降级为无记忆模式：{e}")
            memory_manager = None

    # 初始化 ReActAgent（注入记忆管理器）
    agent = ReActAgent(
        pool=pool,
        default_tier="complex",
        memory_manager=memory_manager,
        session_id=os.getenv("AGENT_SESSION_ID", "default"),
    )

    # 多轮对话测试：不 clear_history，验证跨轮长期记忆召回
    # 第1轮主动告知信息 → 第2轮验证短期记忆 → 第3轮验证工具能力
    test_cases = [
        "我叫李雷，目前在青岛做自媒体录屏工作，请记住我的信息。",
        "我刚才告诉你我叫什么名字？在哪个城市？做什么工作？",  # 验证短期/长期记忆召回
        "345 * 88 + 77等于多少？",  # 验证 calculator 工具仍正常
    ]

    for question in test_cases:
        print("=" * 50)
        print(f"用户提问：{question}")
        print("最终回答：", end="", flush=True)
        for chunk in agent.run_stream(question):
            print(chunk, end="", flush=True)
        print()
        # 注意：这里不再 clear_history，让多轮记忆生效
        # 如需切换会话，改 ReActAgent 的 session_id 或调 agent.clear_history()

    # 打印当前会话记忆状态
    if memory_manager:
        print("\n" + "=" * 50)
        print("记忆系统状态：")
        status = memory_manager.status(os.getenv("AGENT_SESSION_ID", "default"))
        for k, v in status.items():
            print(f"  {k}: {v}")


# ==================== rag：RAG 本地文档问答演示 ====================
def collect_docs():
    """
    收集 data/docs 下所有受支持的文档文件路径。
    若目录为空，则生成一份示例 txt 供演示摄入。
    支持格式：.txt .md .pdf .docx .xlsx .xls .csv
    """
    docs_dir = "./data/docs"
    os.makedirs(docs_dir, exist_ok=True)
    supported = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls", ".csv"}
    # 扫描目录下所有受支持的文件（不递归子目录）
    docs = []
    for name in os.listdir(docs_dir):
        if os.path.splitext(name)[1].lower() in supported:
            docs.append(os.path.join(docs_dir, name))
    # 目录为空时生成示例文档，方便首次体验
    if not docs:
        demo_path = os.path.join(docs_dir, "产品说明.txt")
        content = """智录屏 Pro 产品说明

一、产品概述
智录屏 Pro 是一款面向自媒体创作者的专业屏幕录制软件，支持 4K 超清录制、
实时画中画摄像头叠加、系统声音与麦克风混音录制。

二、核心功能
1. 4K/60fps 超清录制，支持 H.265 编码，文件体积小，画质高。
2. 画中画摄像头：可自由调整摄像头窗口位置与大小，支持美颜。
3. 音频混音：系统声音与麦克风独立轨道，后期可单独编辑。
4. 实时标注：录制过程中可随时在屏幕上画框、箭头、文字标注。
5. 一键导出：支持 MP4、MKV、GIF，可直传 B 站、抖音、YouTube。

三、系统要求
- Windows 10/11 或 macOS 12 及以上
- 至少 8GB 内存，推荐 16GB
- 独立显卡可获得更好编码性能

四、价格
- 个人版：199 元/年
- 专业版：499 元/年（含优先客服与未来版本免费升级）
"""
        with open(demo_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[演示] 目录为空，已生成示例文档：{demo_path}")
        docs.append(demo_path)
    return docs


def cmd_rag():
    """
    RAG 演示：
    1. 摄入本地文档（txt/md/pdf/docx/xlsx/csv 均可）→ 自动切分 → 向量化存入 Chroma
    2. 提问 → 先检索 top-k 文档块 → 流式生成回答（强制标注来源）
    3. 通用 ReActAgent 也能通过 knowledge_search 工具检索同一知识库
    """
    from dotenv import load_dotenv

    from agent import ReActAgent
    from llm_pool import build_default_pool
    from memory.rag_agent import RAGAgent
    from memory.vector_store import get_vector_store

    load_dotenv()
    pool = build_default_pool()
    vs = get_vector_store()
    rag = RAGAgent(pool=pool, tier="simple", vector_store=vs, top_k=4)

    # ========== 1. 摄入文档 ==========
    print("=" * 60)
    print("【第 1 步】摄入本地文档 → 切分 → 向量化存入 Chroma")
    print("=" * 60)
    docs = collect_docs()
    # 自动扫描 data/docs 下所有受支持文件（txt/md/pdf/docx/xlsx/csv）一并摄入
    for fp in docs:
        if os.path.exists(fp):
            n = rag.ingest_document(fp)
            print(f"  摄入 {os.path.basename(fp)}：{n} 个文档块")
    print(f"  当前知识库文档块总数：{vs.count_documents()}")

    # ========== 2. RAG 流式问答 ==========
    print("\n" + "=" * 60)
    print("【第 2 步】RAG 流式问答（先检索 top-k 再生成）")
    print("=" * 60)
    questions = [
        "智录屏 Pro 支持哪些导出格式？",
        "专业版的价格是多少？有什么权益？",
        "我的电脑只有 4GB 内存，能用这个软件吗？",
    ]
    for q in questions:
        print(f"\n用户提问：{q}")
        print("RAG 回答：", end="", flush=True)
        for chunk in rag.answer_stream(q):
            print(chunk, end="", flush=True)
        print()

    # ========== 3. 通用 Agent 也能用 knowledge_search 工具检索同一知识库 ==========
    print("\n" + "=" * 60)
    print("【第 3 步】通用 ReActAgent 通过 knowledge_search 工具检索知识库")
    print("=" * 60)
    agent = ReActAgent(pool=pool, default_tier="simple")
    q = "帮我查一下智录屏 Pro 的系统要求和录制功能"
    print(f"\n用户提问：{q}")
    print("Agent 回答：", end="", flush=True)
    for chunk in agent.run_stream(q):
        print(chunk, end="", flush=True)
    print()


# ==================== 入口 ====================
def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LangChain 多模型智能问答系统（统一入口）",
    )
    sub = parser.add_subparsers(dest="command", title="子命令")
    sub.add_parser("qa", help="交互式智能问答（路由+轮询+降级+计费+缓存）")
    sub.add_parser("agent", help="ReAct 工具型 Agent 演示（含长期记忆）")
    sub.add_parser("rag", help="RAG 本地文档问答演示")
    args = parser.parse_args()

    if args.command == "qa":
        cmd_qa()
    elif args.command == "agent":
        cmd_agent()
    elif args.command == "rag":
        cmd_rag()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
