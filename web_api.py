"""
Web API 层：把问答能力 HTTP 化（SSE 流式输出）。

所有问题统一走 ReAct Agent 工具链（不再用关键词硬判定）：
模型自主决定是否调用 search（联网查天气/新闻/股价）、calculator、knowledge_search，
简单闲聊则直接作答（一轮 LLM 调用，无额外开销）。

链路：QueryCache（重复 query 0 成本）→ ComplexityRouter（simple/complex 分档）
  → ReActAgent.run_stream（档内轮询 + 多密钥/跨模型降级 + 工具调用）→ SSE 逐块推送
  → UsageTracker 计费落盘

启动：uvicorn web_api:app --host 127.0.0.1 --port 8000
页面：http://127.0.0.1:8000 （static/index.html，同源免跨域）
会话：按 session_id 在内存维护 ReActAgent（重启清空，与 CLI 行为一致）
"""
import json
from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import ReActAgent
from llm_pool import ComplexityRouter, build_default_pool
from utils.llm_extras import QueryCache, UsageTracker

app = FastAPI(title="LangChain 多模型问答")

# 复用单例（进程级）
pool = build_default_pool()
router = ComplexityRouter()
tracker = UsageTracker()
cache = QueryCache()
agents: Dict[str, ReActAgent] = {}  # session_id → ReActAgent（自维护会话历史）


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
    # 1. 缓存命中：0 成本，一次性推送
    cached = cache.get(question)
    if cached:
        tracker.record(model="cache", alias="cache", usage=None, cached=True)
        yield _sse("meta", {"tier": "cache", "primary": "cache"})
        yield _sse("token", {"text": cached})
        yield _sse("done", {"tokens": 0, "cost": 0.0})
        return

    # 2. 复杂度路由决定模型档位；工具调用由模型自主判断
    tier = router.classify(question)
    yield _sse("meta", {"tier": tier, "primary": "react-agent"})

    # 3. 统一走 ReAct 工具链，逐块流式推送
    answer_parts: List[str] = []
    agent = _get_agent(session_id, tier)
    try:
        for chunk in agent.run_stream(question):
            answer_parts.append(chunk)
            yield _sse("token", {"text": chunk})
    except Exception as e:
        yield _sse("error", {"message": f"Agent 调用失败：{e}"})
        return

    # 4. 计费 + 缓存（usage 由 Agent 跨轮累计，含真实模型名）
    answer = "".join(answer_parts)
    rec = tracker.record(
        model=agent.last_model or "react-agent",
        alias="react-agent",
        usage=agent.last_usage,
        tier=tier,
    )
    cache.set(question, answer)
    yield _sse("done", {
        "model": agent.last_model or "react-agent",
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
    agents.pop(session_id, None)  # 丢弃实例即清空该会话历史
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
