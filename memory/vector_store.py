"""
向量存储：基于 Chroma + Ollama 本地 Embedding 的统一向量库封装。

- OllamaEmbeddingFunction：把 Ollama 原生 /api/embeddings 接口包装为 Chroma EmbeddingFunction。
- ChromaVectorStore：PersistentClient 持久化，管理两个 collection：
    * rag_documents：RAG 文档块
    * memory_store ：会话记忆 / 摘要
- get_vector_store()：懒加载全局单例，避免 import 时即初始化 Chroma（拖慢 registry 加载）。
"""
import os
import uuid
from typing import Any, Dict, List, Optional

import chromadb
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Chroma 自定义 EmbeddingFunction，调用 Ollama 原生 embedding 接口。

    Chroma 1.5.9 的 EmbeddingFunction 是 Protocol[D]，继承 EmbeddingFunction[Documents]
    会触发 __init_subclass__ 自动包装 __call__，对返回值做 validate + normalize。
    因此 __call__ 必须返回非 None 的 List[List[float]]。
    实现 __init__/name/get_config/build_from_config 可避免 is_legacy() 为 True（消除告警）。
    """

    def __init__(self, base_url: str, model: str, timeout: int = 60):
        # base_url 形如 http://localhost:11434（不带 /v1）
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/api/embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        # Ollama 原生接口一次只接受单条 prompt，批量需循环
        embeddings: List[List[float]] = []
        for text in input:
            try:
                resp = requests.post(
                    self._endpoint,
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            except Exception as e:
                logger.error(f"Ollama embedding 调用失败: {e}")
                raise
        return embeddings

    @staticmethod
    def name() -> str:
        return "ollama_embedding"

    def get_config(self) -> Dict[str, Any]:
        return {"base_url": self.base_url, "model": self.model, "timeout": self.timeout}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(**config)


class QwenEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    通义 text-embedding-v3 EmbeddingFunction，走 DashScope OpenAI 兼容接口。

    与 OllamaEmbeddingFunction 的区别：
    - OpenAI 标准 /embeddings 接口，原生支持批量 input（一次请求多条）
    - 多密钥：QWEN_API_KEY 逗号分隔，每次调用随机选取（基础轮询）
    """

    def __init__(self, base_url: str, api_keys: str, model: str = "text-embedding-v3",
                 timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        if not self.api_keys:
            raise ValueError("QwenEmbeddingFunction 需要 QWEN_API_KEY")
        self.model = model
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        import random
        key = random.choice(self.api_keys)
        try:
            resp = requests.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": self.model, "input": list(input)},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # OpenAI 格式：按 index 排序取向量
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except Exception as e:
            logger.error(f"Qwen embedding 调用失败: {e}")
            raise

    @staticmethod
    def name() -> str:
        return "qwen_embedding"

    def get_config(self) -> Dict[str, Any]:
        return {"base_url": self.base_url, "model": self.model, "timeout": self.timeout}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "QwenEmbeddingFunction":
        # api_keys 不进 config（避免密钥落盘），从环境变量回补
        return QwenEmbeddingFunction(api_keys=os.getenv("QWEN_API_KEY", ""), **config)


def _build_embedding_function() -> EmbeddingFunction:
    """
    按 EMBEDDING_PROVIDER 构建 EmbeddingFunction（embedding 固定通道，不参与模型轮询）：
    - qwen（默认）：通义 text-embedding-v3（DeepSeek 官方无 embedding API）
    - ollama：本地 Ollama（免费离线）
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "qwen").lower()
    if provider == "ollama":
        return OllamaEmbeddingFunction(
            base_url=os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434"),
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        )
    return QwenEmbeddingFunction(
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_keys=os.getenv("QWEN_API_KEY", ""),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    )


class ChromaVectorStore:
    """基于 Chroma PersistentClient 的向量存储，统一管理 RAG 文档与会话记忆。"""

    RAG_COLLECTION = "rag_documents"
    MEMORY_COLLECTION = "memory_store"

    def __init__(self, persist_dir: str, embed_fn: Optional[EmbeddingFunction] = None):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self._embed_fn = embed_fn or _build_embedding_function()
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 两个 collection 均绑定同一个 EF；get_or_create 保证幂等
        # nomic-embed-text 适合 cosine 距离
        self.rag_collection = self.client.get_or_create_collection(
            name=self.RAG_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.memory_collection = self.client.get_or_create_collection(
            name=self.MEMORY_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaVectorStore 初始化完成: dir={persist_dir}, "
            f"rag={self.rag_collection.count()}条, memory={self.memory_collection.count()}条"
        )

    # ==================== RAG 文档（rag_documents）====================
    def add_documents(
        self, texts: List[str], metadatas: List[Dict], ids: List[str]
    ) -> None:
        """向量化并存入文档块（Chroma 会自动调用 self._embed_fn 生成向量）"""
        if not texts:
            return
        self.rag_collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def query_documents(self, text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """检索相关文档块，返回 [{text, source, score, id}, ...]"""
        res = self.rag_collection.query(query_texts=[text], n_results=top_k)
        return self._format_query_result(res)

    # ==================== 会话记忆（memory_store）====================
    def add_memory(self, text: str, metadata: Dict) -> str:
        """存入一条记忆/摘要，返回 mem_id"""
        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        self.memory_collection.add(
            documents=[text], metadatas=[metadata], ids=[mem_id]
        )
        return mem_id

    def query_memory(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """语义检索相关记忆，返回 [{text, metadata, score, id}, ...]"""
        res = self.memory_collection.query(query_texts=[text], n_results=top_k)
        return self._format_query_result(res)

    # ==================== 运维 ====================
    def count_documents(self) -> int:
        return self.rag_collection.count()

    def count_memories(self) -> int:
        return self.memory_collection.count()

    def delete_document_by_source(self, source: str) -> None:
        """按 source 元数据删除某文件的所有文档块"""
        self.rag_collection.delete(where={"source": source})

    def reset_rag_collection(self) -> None:
        """清空 RAG 文档库（谨慎调用）"""
        self.client.delete_collection(name=self.RAG_COLLECTION)
        self.rag_collection = self.client.get_or_create_collection(
            name=self.RAG_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ==================== 内部工具 ====================
    @staticmethod
    def _format_query_result(res: Dict) -> List[Dict[str, Any]]:
        """把 Chroma query 返回的多列结构整理为行列表"""
        if not res or not res.get("ids") or not res["ids"][0]:
            return []
        ids = res["ids"][0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        results: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids):
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 0.0
            results.append(
                {
                    "id": _id,
                    "text": doc,
                    "source": meta.get("source", ""),
                    "metadata": meta,
                    # cosine 距离越小越相似，转换为相似度分数
                    "score": round(1.0 - float(dist), 4),
                }
            )
        return results


# ==================== 懒加载全局单例 ====================
_vector_store: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """
    懒加载全局单例。

    延迟初始化可避免 agent.tool_registry 在 import 时即实例化所有工具时，
    强制初始化 Chroma（慢且可能在 Ollama 未启动时抛错）。
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        )
    return _vector_store
