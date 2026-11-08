"""
RAG 本地文档问答机器人（LangChain 版，基于 llm_pool）。

流程：文档 → DocumentLoader 加载 → TextChunker 切分 → ChromaVectorStore 向量化
      提问 → 检索 top-k 文档块 → 注入 system prompt → pool 生成（支持流式）

LLM 调用统一走 LLMPool.get_chain(tier, temperature)：档位内轮询 + 降级链，
低温生成忠实于检索资料并强制标注来源。
"""
import os
import uuid
from typing import Dict, Generator, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from llm_pool import LLMPool
from memory.document_loader import DocumentLoader, TextChunker
from memory.vector_store import ChromaVectorStore, get_vector_store
from utils.logger import logger

load_dotenv()

RAG_SYSTEM_PROMPT = """你是一个严谨的本地文档问答助手。请严格依据下方【参考资料】回答用户问题。

规则：
1. 答案必须能在参考资料中找到依据，禁止编造。
2. 若参考资料不足以回答，直接回答"参考资料中未涉及该问题"。
3. 回答末尾用 [来源N] 标注引用的资料编号。

【参考资料】
{context}
"""

RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.3"))


class RAGAgent:
    """独立 RAG 问答机器人：提问必先检索 rag_documents 再生成。"""

    def __init__(
        self,
        pool: LLMPool,
        tier: str = "simple",
        vector_store: Optional[ChromaVectorStore] = None,
        top_k: Optional[int] = None,
    ):
        self.pool = pool
        self.tier = tier
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k or int(os.getenv("RAG_TOP_K", "4"))
        # 注入 pool，使 _load_pdf 能经 pool.get_vision_client() 对扫描版 PDF 调多模态 OCR
        self.loader = DocumentLoader(pool=pool)
        self.chunker = TextChunker()

    # ==================== 摄入 ====================
    def ingest_document(self, file_path: str) -> int:
        """加载文件 → 切分 → 向量化存入 rag_documents，返回块数"""
        text = self.loader.load(file_path)
        return self._ingest_text(text, source=file_path)

    def ingest_text(self, text: str, source: str = "user_input") -> int:
        """直接摄入纯文本"""
        return self._ingest_text(text, source=source)

    def _ingest_text(self, text: str, source: str) -> int:
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        chunks = self.chunker.split(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            logger.warning(f"摄入 {source} 失败：切分后为空")
            return 0

        metadatas = [
            {"source": source, "chunk_index": i, "total": len(chunks)}
            for i in range(len(chunks))
        ]
        ids = [f"doc_{uuid.uuid4().hex[:16]}_{i}" for i in range(len(chunks))]
        self.vector_store.add_documents(chunks, metadatas, ids)
        logger.info(f"摄入 {source}：{len(chunks)} 个文档块已入库")
        return len(chunks)

    # ==================== 检索 ====================
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """检索 top-k 相关文档块"""
        hits = self.vector_store.query_documents(query, top_k=top_k or self.top_k)
        logger.info(f"RAG 检索 '{query}' → 命中 {len(hits)} 块")
        return hits

    def _build_context(self, hits: List[Dict]) -> str:
        """把检索结果拼成 system prompt 中的参考资料区"""
        if not hits:
            return "（无相关参考资料）"
        blocks = []
        for i, h in enumerate(hits, 1):
            src = h.get("source", "")
            score = h.get("score", 0)
            blocks.append(
                f"[来源{i}] (相似度={score:.3f}, 来源:{src})\n{h['text']}"
            )
        return "\n\n".join(blocks)

    # ==================== 生成 ====================
    def _build_messages(self, query: str, context: str) -> List:
        """组装 LangChain 消息（RAG 系统提示 + 用户问题）"""
        return [
            SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=query),
        ]

    def answer(self, query: str, top_k: Optional[int] = None) -> str:
        """非流式：检索 → 生成完整回答"""
        hits = self.retrieve(query, top_k)
        context = self._build_context(hits)
        chain = self.pool.get_chain(tier=self.tier, temperature=RAG_TEMPERATURE)[0]
        return chain.invoke(self._build_messages(query, context)).content

    def answer_stream(
        self, query: str, top_k: Optional[int] = None
    ) -> Generator[str, None, None]:
        """流式：检索 → 逐块 yield 生成文本"""
        hits = self.retrieve(query, top_k)
        context = self._build_context(hits)
        chain = self.pool.get_chain(tier=self.tier, temperature=RAG_TEMPERATURE)[0]
        for chunk in chain.stream(self._build_messages(query, context)):
            if chunk.content:
                yield chunk.content
