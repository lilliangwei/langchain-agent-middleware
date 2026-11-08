"""
Web API 层：把 main.py qa 子命令的单轮逻辑 HTTP 化（SSE 流式输出）。

复用现有全套能力，零重写：
  QueryCache（重复 query 0 成本）→ ComplexityRouter（simple/complex 分档）
  → LLMPool.get_chain(tier)（档内轮询 + 多密钥/跨模型降级）→ SSE 逐块推送
  → UsageTracker 计费落盘

启动：uvicorn web_api:app --host 127.0.0.1 --port 8000
页面：http://127.0.0.1:8000 （static/index.html，同源免跨域）
会话：按 session_id 在内存维护历史（重启清空，与 CLI 行为一致）
"""
import json
import re
from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from agent import ReActAgent
from llm_pool import ComplexityRouter, build_default_pool
from utils.llm_extras import QueryCache, UsageTracker

SYSTEM_PROMPT = "你是一个智能问答助手，回答准确、简洁、有条理。支持多轮对话，能记住上下文。"

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

app = FastAPI(title="LangChain 多模型问答")

# 复用单例（进程级）
pool = build_default_pool()
router = ComplexityRouter()
tracker = UsageTracker()
cache = QueryCache()
sessions: Dict[str, List] = {}  # session_id → [HumanMessage/AIMessage, ...]
agents: Dict[str, ReActAgent] = {}  # session_id → ReActAgent（自维护会话历史）

# 实时类问题关键词：命中则走 ReAct Agent 工具链（search 联网查询），否则走纯问答链
TOOL_KEYWORDS = re.compile(
    r"天气|新闻|最新|实时|今天|今日|现在|当前|股价|汇率|价格|几点|日期|星期|热搜|搜索|查一下|查下"
)


def _get_agent(session_id: str, tier: str) -> ReActAgent:
    """每个会话一个 ReActAgent 实例（内部自维护消息历史），带完整工具集"""
    if session_id not in agents:
        agents[session_id] = ReActAgent(pool=pool, default_tier=tier, session_id=session_id)
    return agents[session_id]


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chat_stream(question: str, session_id: str):
    """生成 SSE 事件流：meta → token* → done"""
    history = sessions.setdefault(session_id, [])

    # 1. 缓存命中：0 成本，一次性推送
    cached = cache.get(question)
    if cached:
        tracker.record(model="cache", alias="cache", usage=None, cached=True)
        yield _sse("meta", {"tier": "cache", "primary": "cache"})
        yield _sse("token", {"text": cached})
        yield _sse("done", {"tokens": 0, "cost": 0.0})
        return

    # 2. 路由 → 档位轮询 + 降级链
    tier = router.classify(question)

    # 2a. 实时类问题 → ReAct Agent 工具链（可联网搜索天气/新闻/股价等）
    if TOOL_KEYWORDS.search(question):
        yield _sse("meta", {"tier": tier, "primary": "react-agent"})
        answer_parts: List[str] = []
        try:
            agent = _get_agent(session_id, tier)
            for chunk in agent.run_stream(question):
                answer_parts.append(chunk)
                yield _sse("token", {"text": chunk})
        except Exception as e:
            yield _sse("error", {"message": f"Agent 调用失败：{e}"})
            return
        answer = "".join(answer_parts)
        # ReAct 中间轮 tool_calls 的 usage 不在末块上，此处只记调用次数
        tracker.record(model="react-agent", alias="react", usage=None, cached=False)
        cache.set(question, answer)
        yield _sse("done", {"model": "react-agent", "tokens": 0, "cost": 0.0})
        return

    # 2b. 普通问题 → 纯问答链
    model_chain, primary = pool.get_chain(tier=tier)
    chain = prompt | model_chain
    yield _sse("meta", {"tier": tier, "primary": primary})

    # 3. 流式生成
    collected = None
    try:
        for chunk in chain.stream({"question": question, "history": history}):
            collected = chunk if collected is None else collected + chunk
            if chunk.content:
                yield _sse("token", {"text": chunk.content})
    except Exception as e:
        yield _sse("error", {"message": f"全部模型调用失败：{e}"})
        return

    # 4. 计费 + 缓存 + 历史
    answer = collected.content if collected else ""
    usage = getattr(collected, "usage_metadata", None) if collected else None
    served_model = (
        (collected.response_metadata or {}).get("model_name", primary)
        if collected else primary
    )
    rec = tracker.record(model=served_model, alias=primary, usage=usage, tier=tier)
    cache.set(question, answer)
    history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    yield _sse("done", {
        "model": served_model,
        "tokens": rec["total_tokens"] if rec else 0,
        "cost": rec["cost_cny"] if rec else 0.0,
    })


@app.post("/api/chat")
def chat(req: ChatRequest):
    return StreamingResponse(
        _chat_stream(req.question.strip(), req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/clear")
def clear(session_id: str = "default"):
    sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/api/cost")
def cost():
    return tracker.summary()


@app.get("/")
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
