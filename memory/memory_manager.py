"""
长期记忆管理器：编排三类记忆。

1. 短期记忆：当前会话上下文（由 SessionBackend 管理：内存 或 Redis 持久化）
2. 历史摘要：./data/memory_summaries.json，按 session_id 存摘要元数据
3. 长期向量记忆：Chroma memory_store collection，存摘要/关键事实，支持语义召回

工作流：
- 每轮对话把 user/assistant 写入短期记忆（load-modify-save 模式）
- 短期消息数超阈值时，maybe_summarize 用 LLM 摘要旧消息 → 写 JSON + 入向量库
  → 短期记忆压缩为「system 摘要 + 最近 keep_recent 条」
- get_context：召回相关长期记忆 + 取摘要 + 最近短期 → 组装可注入的 messages

后端说明：
- backend=InMemoryBackend（默认）：load 返回引用，行为与改造前完全一致。
- backend=RedisBackend：load 返回副本，重启进程后可从 Redis 恢复短期记忆（跨重启续接）。
  Redis 不可用时自动降级内存，不阻塞主流程。
"""
import os
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from llm_pool import LLMPool
from memory.session_backend import (
    SessionBackend,
    InMemoryBackend,
    RedisBackend,
    create_backend_from_env,
)
from memory.vector_store import ChromaVectorStore, get_vector_store
from utils.logger import logger
from utils.token_utils import count_message_tokens

load_dotenv()

SUMMARY_SYSTEM_PROMPT = "请把以下对话压缩为简洁摘要，保留关键事实、用户偏好、未决问题。"


class MemoryManager:
    def __init__(
        self,
        pool: LLMPool,
        summary_tier: str = "simple",
        vector_store: Optional[ChromaVectorStore] = None,
        summary_threshold: Optional[int] = None,
        keep_recent: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        summaries_path: Optional[str] = None,
        backend: Optional[SessionBackend] = None,
    ):
        self.pool = pool
        self.summary_tier = summary_tier
        self.vector_store = vector_store or get_vector_store()
        self.summary_threshold = summary_threshold or int(
            os.getenv("MEMORY_SUMMARY_THRESHOLD", "8")
        )
        # 摘要触发 token 阈值：>0 时优先按 token 触发（替代按条数），更精准控制上下文
        self.summary_token_threshold = int(
            os.getenv("MEMORY_SUMMARY_TOKEN_THRESHOLD", "2000")
        )
        self.keep_recent = keep_recent or int(os.getenv("MEMORY_KEEP_RECENT", "4"))
        self.max_context_tokens = max_context_tokens or int(
            os.getenv("MEMORY_MAX_CONTEXT_TOKENS", "6000")
        )
        self.summaries_path = summaries_path or os.getenv(
            "MEMORY_SUMMARIES_PATH", "./data/memory_summaries.json"
        )
        # 短期记忆后端：未显式注入时按 .env 的 MEMORY_BACKEND 决定
        self.backend: SessionBackend = backend or create_backend_from_env()
        # 摘要元数据：session_id → {summary, updated_at, message_count}
        self.summaries: Dict[str, Dict] = self._load_summaries()
        backend_type = self.backend.__class__.__name__
        logger.info(f"[记忆] 短期记忆后端：{backend_type}")

    # ==================== 会话 / 短期 ====================
    def get_or_create_session(self, session_id: str) -> List[Dict]:
        """
        读取某会话的短期消息列表。

        注意：返回值仅供读取，写操作必须用 add_user_message / add_assistant_message。
        （Redis 模式返回副本，原地 append 不会持久化。）
        """
        return self.backend.load(session_id)

    def add_user_message(self, session_id: str, content: str) -> None:
        msgs = self.backend.load(session_id)
        msgs.append({"role": "user", "content": content})
        self.backend.save(session_id, msgs)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        msgs = self.backend.load(session_id)
        msgs.append({"role": "assistant", "content": content})
        self.backend.save(session_id, msgs)

    def clear_session(self, session_id: str) -> None:
        """仅清短期记忆，不动摘要 / 向量长期记忆"""
        self.backend.clear(session_id)

    # ==================== 摘要 ====================
    def maybe_summarize(self, session_id: str) -> Optional[str]:
        """
        短期消息超阈值时触发：
          1. 用 LLM 摘要除最近 keep_recent 外的旧消息
          2. 摘要写 JSON + 入向量库 memory_store
          3. 短期压缩为 [system 摘要, *最近 keep_recent]
        触发条件：MEMORY_SUMMARY_TOKEN_THRESHOLD > 0 时按短期消息总 token 数判定
        （精准控制上下文体积），否则退回按消息条数（MEMORY_SUMMARY_THRESHOLD）。
        返回生成的摘要（未触发返回 None）。
        """
        msgs = self.backend.load(session_id)
        if self.summary_token_threshold > 0:
            triggered = (
                count_message_tokens(msgs) >= self.summary_token_threshold
            )
        else:
            triggered = len(msgs) >= self.summary_threshold
        # 消息太少时没有摘要价值（压缩后反而可能更长）
        if not triggered or len(msgs) <= self.keep_recent:
            return None

        to_summarize = msgs[: -self.keep_recent]
        recent = msgs[-self.keep_recent:]
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)

        logger.info(
            f"[记忆] 触发摘要 session={session_id}，"
            f"待摘要 {len(to_summarize)} 条，保留最近 {len(recent)} 条"
        )
        # 摘要调用走 LLMPool：档位内轮询 + 降级链（LangChain 消息格式）
        chain = self.pool.get_chain(tier=self.summary_tier)[0]
        summary = chain.invoke([
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=joined),
        ]).content

        # 写 JSON 摘要元数据
        self.summaries[session_id] = {
            "summary": summary,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message_count": len(msgs),
        }
        self._save_summaries()

        # 入向量库（语义可召回）
        try:
            self.vector_store.add_memory(
                text=f"[会话 {session_id} 摘要] {summary}",
                metadata={
                    "session_id": session_id,
                    "type": "summary",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        except Exception as e:
            logger.warning(f"[记忆] 摘要入向量库失败（不影响主流程）: {e}")

        # 压缩短期：用 system 摘要替换旧消息
        self.backend.save(
            session_id,
            [{"role": "system", "content": f"【历史摘要】{summary}"}, *recent],
        )
        logger.info(f"[记忆] 摘要完成并压缩短期记忆 session={session_id}")
        return summary

    # ==================== 上下文召回 ====================
    def get_context(
        self, session_id: str, user_query: str, top_k: int = 3
    ) -> List[Dict]:
        """
        组装可注入 ReActAgent.messages 的消息列表：
            [system(摘要+长期记忆), *最近短期, {user: user_query}]
        并按 max_context_tokens 从旧到新截断（保留 system 和最新消息）。

        返回独立组装的新 list，ReActAgent 本轮临时使用；工具中间消息不持久化，
        轮次结束仅把 user/assistant 最终结果写回 session（方案 A）。
        """
        # 1. 长期向量召回（异常不阻断主流程）
        recalled: List[Dict] = []
        try:
            recalled = self.vector_store.query_memory(user_query, top_k=top_k)
        except Exception as e:
            logger.warning(f"[记忆] 长期记忆召回失败: {e}")

        memory_block = (
            "\n".join(f"- {m['text']}" for m in recalled) if recalled else ""
        )

        # 2. 摘要
        summary_text = self.summaries.get(session_id, {}).get("summary", "")

        # 3. 拼系统提示
        sys_parts = ["你是一个具备长期记忆的智能助手，能跨轮次记住用户信息。"]
        if summary_text:
            sys_parts.append(f"【历史摘要】{summary_text}")
        if memory_block:
            sys_parts.append(f"【相关长期记忆】\n{memory_block}")
        system_msg = {"role": "system", "content": "\n\n".join(sys_parts)}

        # 4. 最近短期 + 当前提问（backend.load 已返回副本，再 list 一层保险）
        recent = list(self.backend.load(session_id))
        context = [system_msg, *recent, {"role": "user", "content": user_query}]

        # 5. token 截断：从旧到新删（保留 system 和最后的 user）
        while (
            count_message_tokens(context) > self.max_context_tokens
            and len(context) > 2
        ):
            context.pop(1)
        return context

    # ==================== 持久化 ====================
    def _load_summaries(self) -> Dict[str, Dict]:
        if os.path.exists(self.summaries_path):
            try:
                import json

                with open(self.summaries_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[记忆] 加载摘要 JSON 失败: {e}")
        return {}

    def _save_summaries(self) -> None:
        import json

        os.makedirs(os.path.dirname(self.summaries_path) or ".", exist_ok=True)
        with open(self.summaries_path, "w", encoding="utf-8") as f:
            json.dump(self.summaries, f, ensure_ascii=False, indent=2)

    def save(self) -> None:
        self._save_summaries()

    def load(self) -> None:
        self.summaries = self._load_summaries()

    # ==================== 调试 / 可观测 ====================
    def status(self, session_id: str) -> Dict:
        """返回某会话的记忆状态概览，便于调试"""
        return {
            "session_id": session_id,
            "short_term_count": len(self.backend.load(session_id)),
            "has_summary": session_id in self.summaries,
            "summary_updated_at": self.summaries.get(session_id, {}).get(
                "updated_at"
            ),
            "memory_vector_count": self.vector_store.count_memories(),
            "backend": self.backend.__class__.__name__,
        }
