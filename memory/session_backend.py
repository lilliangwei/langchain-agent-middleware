"""
会话短期记忆后端抽象层 + Redis 会话客户端（原 utils/redis_session.py 并入）。

- RedisSession：Redis 直连客户端（同步/异步），管理通用聊天历史（llm:session:）
  与短期记忆（llm:memory:session:）两个隔离命名空间；模块底部 redis_session 为全局单例。
- InMemoryBackend：纯内存 Dict，行为与改造前的 self.sessions 完全等价（默认）。
- RedisBackend：复用 redis_session 单例，Redis 不可用时自动降级内存。

设计要点：
- 两种实现共用同一接口，MemoryManager 无感知后端类型。
- InMemoryBackend.load 返回内部 list 的引用（保持现状"原地 append 即持久化"语义）。
- RedisBackend.load 返回副本（远程状态无法返回引用）；写操作必须走 save，原地 append 不持久化。
- RedisBackend 构造不连 Redis（redis.Redis 是惰性连接），首次读写才真正建链。
  Redis 不可用时置 _degraded=True，本进程生命周期内不再重试 Redis，操作全走 _fallback。
"""
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import redis
import redis.asyncio as async_redis
from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()


# ==================== Redis 会话客户端（原 utils/redis_session.py 并入） ====================
class RedisSession:
    def __init__(self):
        self.expire = int(os.getenv("SESSION_EXPIRE_SECONDS", 86400))
        self.sync_cli = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            db=int(os.getenv("REDIS_DB")),
            decode_responses=True
        )
        self._async_cli = None

    async def get_async_client(self):
        if not self._async_cli:
            self._async_cli = async_redis.Redis(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                db=int(os.getenv("REDIS_DB")),
                decode_responses=True
            )
        return self._async_cli

    # 同步接口
    def load_history(self, session_id: str):
        key = f"llm:session:{session_id}"
        data = self.sync_cli.get(key)
        return json.loads(data) if data else []

    def save_history(self, session_id: str, history: list):
        key = f"llm:session:{session_id}"
        self.sync_cli.setex(key, self.expire, json.dumps(history, ensure_ascii=False))

    # 异步接口
    async def aload_history(self, session_id: str):
        cli = await self.get_async_client()
        data = await cli.get(f"llm:session:{session_id}")
        return json.loads(data) if data else []

    async def asave_history(self, session_id: str, history: list):
        cli = await self.get_async_client()
        await cli.setex(
            f"llm:session:{session_id}",
            self.expire,
            json.dumps(history, ensure_ascii=False)
        )

    # ==================== 短期记忆专用（MemoryManager 使用）====================
    # 独立命名空间 llm:memory:session:{id}，与通用聊天历史 llm:session:{id} 隔离：
    # 短期记忆含 system 摘要压缩消息，schema 与纯 chat history 不同，混用易冲突。
    def load_memory(self, session_id: str, expire: int = None) -> list:
        """读取短期记忆列表，无则返回空 list。expire 仅用于 TTL 刷新时透传。"""
        key = f"llm:memory:session:{session_id}"
        data = self.sync_cli.get(key)
        if data is None:
            return []
        try:
            return json.loads(data)
        except (ValueError, TypeError):
            return []

    def save_memory(self, session_id: str, messages: list, expire: int = None) -> None:
        """整体写入短期记忆（覆盖），带 TTL 自动回收。"""
        key = f"llm:memory:session:{session_id}"
        ttl = expire if expire is not None else self.expire
        self.sync_cli.setex(key, ttl, json.dumps(messages, ensure_ascii=False))

    def clear_memory(self, session_id: str) -> None:
        """清空某会话短期记忆。"""
        self.sync_cli.delete(f"llm:memory:session:{session_id}")


redis_session = RedisSession()


# ==================== 短期记忆后端抽象层 ====================
class SessionBackend(ABC):
    """短期记忆后端统一接口。"""

    @abstractmethod
    def load(self, session_id: str) -> List[Dict]:
        """读取某会话的短期消息列表。"""

    @abstractmethod
    def save(self, session_id: str, messages: List[Dict]) -> None:
        """整体写入某会话的短期消息列表。"""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """清空某会话的短期消息（不动摘要 / 向量长期记忆）。"""

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """判断某会话是否存在短期消息。"""


class InMemoryBackend(SessionBackend):
    """纯内存后端，load 返回引用，与改造前行为完全一致。"""

    def __init__(self):
        self._store: Dict[str, List[Dict]] = {}

    def load(self, session_id: str) -> List[Dict]:
        if session_id not in self._store:
            self._store[session_id] = []
        return self._store[session_id]

    def save(self, session_id: str, messages: List[Dict]) -> None:
        self._store[session_id] = messages

    def clear(self, session_id: str) -> None:
        self._store[session_id] = []

    def exists(self, session_id: str) -> bool:
        return session_id in self._store and len(self._store[session_id]) > 0


class RedisBackend(SessionBackend):
    """
    Redis 持久化后端，复用本文件 redis_session 单例。

    - Redis 不可用时自动降级为内存模式（_degraded=True 后不再重试）。
    - load 返回副本（远程状态无法返回引用）。
    - key 命名空间：llm:memory:session:{session_id}（与原 llm:session: 通用历史隔离）。
    """

    def __init__(self, redis_session, expire: Optional[int] = None):
        self.redis_session = redis_session
        self.expire = expire  # 透传给 redis_session.save_memory
        self._fallback: Dict[str, List[Dict]] = {}
        self._degraded = False

    def _degrade(self, session_id: str, reason: str) -> List[Dict]:
        if not self._degraded:
            self._degraded = True
            logger.warning(
                f"[记忆] RedisBackend 降级为内存模式（session={session_id}）：{reason}"
            )

    def load(self, session_id: str) -> List[Dict]:
        if self._degraded:
            return self._fallback.get(session_id, [])
        try:
            msgs = self.redis_session.load_memory(session_id, expire=self.expire)
            # load_memory 已处理 None → []
            return list(msgs) if msgs else []
        except Exception as e:
            self._degrade(session_id, f"load 失败 {e}")
            return self._fallback.get(session_id, [])

    def save(self, session_id: str, messages: List[Dict]) -> None:
        if self._degraded:
            self._fallback[session_id] = messages
            return
        try:
            self.redis_session.save_memory(
                session_id, messages, expire=self.expire
            )
        except Exception as e:
            self._degrade(session_id, f"save 失败 {e}")
            self._fallback[session_id] = messages

    def clear(self, session_id: str) -> None:
        if self._degraded:
            self._fallback[session_id] = []
            return
        try:
            self.redis_session.clear_memory(session_id)
        except Exception as e:
            self._degrade(session_id, f"clear 失败 {e}")
            self._fallback[session_id] = []

    def exists(self, session_id: str) -> bool:
        if self._degraded:
            return session_id in self._fallback and len(self._fallback[session_id]) > 0
        try:
            msgs = self.redis_session.load_memory(session_id, expire=self.expire)
            return bool(msgs)
        except Exception as e:
            self._degrade(session_id, f"exists 失败 {e}")
            return session_id in self._fallback and len(self._fallback[session_id]) > 0


def create_backend_from_env() -> SessionBackend:
    """根据 .env 的 MEMORY_BACKEND 选择后端，构造失败回退内存。"""
    backend_type = os.getenv("MEMORY_BACKEND", "memory").lower()
    if backend_type == "redis":
        try:
            expire = int(os.getenv("SESSION_EXPIRE_SECONDS", "86400"))
            logger.info("[记忆] 启用 Redis 会话持久化后端")
            return RedisBackend(redis_session, expire=expire)
        except Exception as e:
            logger.warning(
                f"[记忆] RedisBackend 初始化失败，回退 InMemoryBackend：{e}"
            )
            return InMemoryBackend()
    # 默认 / 未知值 → 内存
    return InMemoryBackend()
