"""
统一 LLM 层：LangChain 版多模型轮询 + 多密钥降级模型池 + 复杂度路由器 + 视觉客户端。

设计：
- 每个 (模型, key) 组合 → 一个独立的 langchain_openai.ChatOpenAI 客户端
- get_chain() 每次调用返回一条「主模型 + 降级链」的 Runnable：
    * 多模型轮询：按请求序号轮转主模型（deepseek → qwen → zhipu → moonshot → ollama）
    * 多密钥轮询：同一模型内按 key 轮转首发 key
    * 降级链：主 key 失败 → 同模型剩余 key → 下一个模型的所有 key（由
      Runnable.with_fallbacks 实现，任意异常自动切换下一个候选）
- ComplexityRouter（原 router.py 并入）：规则打分把请求分流到 simple / complex 档
- get_vision_client()：扫描版 PDF OCR 用的多模态视觉客户端（Ollama 本地模型）
- 全项目唯一 LLM 入口，ReActAgent / RAGAgent / MemoryManager / OCR 均经此池调用
"""
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from utils.logger import logger

load_dotenv()

DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 60))


@dataclass
class ModelEntry:
    """一个模型厂商的配置（可含多个 API key）"""
    alias: str                      # 模型别名，如 deepseek
    model_name: str                 # 实际模型名，如 deepseek-chat
    base_url: str                   # OpenAI 兼容接口地址
    api_keys: List[str]             # 多密钥列表
    temperature: float = 0.7
    tier: str = "simple"            # 档位：simple（便宜快）/ complex（强推理），供 ComplexityRouter 分流
    # 运行时状态：key 轮询游标
    _key_cursor: int = field(default=0, repr=False)

    def rotated_keys(self) -> List[str]:
        """按游标轮转 key 顺序，实现多密钥轮询（每次调用游标 +1）"""
        keys = self.api_keys
        start = self._key_cursor % len(keys)
        self._key_cursor += 1
        return keys[start:] + keys[:start]


class LLMPool:
    """
    多模型轮询 + 多密钥降级池。

    用法：
        pool = build_default_pool()
        chain, alias = pool.get_chain()      # chain 可直接接入 LCEL：prompt | chain | parser
        answer = chain.invoke(messages)
    """

    def __init__(self, entries: List[ModelEntry], timeout: int = DEFAULT_TIMEOUT):
        if not entries:
            raise ValueError("LLMPool 至少需要一个 ModelEntry")
        self.entries = entries
        self.timeout = timeout
        self._model_cursor = 0
        self._lock = threading.Lock()

    # ---------- 客户端构建 ----------
    def _build_client(self, entry: ModelEntry, key: str) -> ChatOpenAI:
        """单 (模型, key) → ChatOpenAI。max_retries=0：失败立刻抛给降级链，不在同 key 上空转。"""
        return ChatOpenAI(
            model=entry.model_name,
            api_key=key,
            base_url=entry.base_url,
            temperature=entry.temperature,
            timeout=self.timeout,
            max_retries=0,
            streaming=True,
            stream_usage=True,  # 流式末块携带 usage，供计费统计
        )

    def _entry_chain(self, entry: ModelEntry) -> List[ChatOpenAI]:
        """一个模型的降级子链：轮转后的所有 key 各生成一个客户端"""
        return [self._build_client(entry, k) for k in entry.rotated_keys()]

    # ---------- 对外接口 ----------
    def get_chain(
        self,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[Runnable, str]:
        """
        返回 (带完整降级链的 Runnable, 本轮主模型别名)。

        tier 参数（配合 ComplexityRouter）：
        - 指定 "simple"/"complex" 时，仅在该档位内轮询选主模型；
          降级链 = 本档所有 (模型×key) → 另一档所有 (模型×key) 兜底
        - 不指定时退化为全池轮询（原行为）

        temperature 参数：不为 None 时对整条链 bind 覆盖温度
        （RunnableWithFallbacks 支持 bind，会同步作用到所有候选客户端）。

        降级顺序：主模型[key1..keyN] → 同档其他模型 → 跨档模型，直到候选耗尽。
        """
        if tier:
            tier_entries = [e for e in self.entries if e.tier == tier]
            other_entries = [e for e in self.entries if e.tier != tier]
            pool = tier_entries or self.entries  # 该档无模型时退化为全池
        else:
            tier_entries, other_entries = self.entries, []
            pool = self.entries

        with self._lock:
            start = self._model_cursor % len(pool)
            self._model_cursor += 1
        # 轮转模型顺序：本次主模型 = pool[start]，同档其余随后，跨档兜底
        ordered = pool[start:] + pool[:start] + other_entries

        candidates: List[Tuple[ChatOpenAI, str]] = []
        for entry in ordered:
            for client in self._entry_chain(entry):
                candidates.append((client, entry.alias))

        primary, primary_alias = candidates[0]
        fallbacks = [c for c, _ in candidates[1:]]
        chain = primary.with_fallbacks(fallbacks, exceptions_to_handle=(Exception,))
        if temperature is not None:
            chain = chain.bind(temperature=temperature)
        logger.info(
            f"[LLMPool] 本轮主模型={primary_alias}（tier={tier or 'all'}），降级链："
            + " → ".join(a for _, a in candidates)
        )
        return chain, primary_alias

    def get_vision_client(self) -> Optional[ChatOpenAI]:
        """
        构建多模态视觉客户端（扫描版 PDF OCR 用）。

        配置了 OLLAMA_VISION_MODEL 时返回 ChatOpenAI（本地 Ollama，dummy key 不校验）；
        未配置时返回 None，调用方据此降级为纯 pypdf 解析。
        """
        vision_model = os.getenv("OLLAMA_VISION_MODEL")
        if not vision_model:
            return None
        return ChatOpenAI(
            model=vision_model,
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key="dummy",
            temperature=0.1,
            max_retries=0,
        )

    def describe(self) -> List[Dict]:
        """池状态快照（调试/观测用）"""
        return [
            {
                "alias": e.alias,
                "model": e.model_name,
                "base_url": e.base_url,
                "key_count": len(e.api_keys),
                "tier": e.tier,
            }
            for e in self.entries
        ]


def _split_keys(raw: Optional[str]) -> List[str]:
    """逗号分隔的多密钥字符串 → 去空白后的列表"""
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def build_default_pool(timeout: int = DEFAULT_TIMEOUT) -> LLMPool:
    """
    从 .env 构建默认模型池。
    模型名可用 {ALIAS}_MODEL 环境变量覆盖（如 DEEPSEEK_MODEL=deepseek-reasoner）。
    未配置 key 的厂商自动跳过；ollama 使用 dummy key（本地服务不校验）。
    """
    specs = [
        # (别名, key环境变量, base_url环境变量, 默认模型名, 默认档位)
        ("deepseek", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "deepseek-chat", "complex"),
        ("qwen", "QWEN_API_KEY", "QWEN_BASE_URL", "qwen-plus", "simple"),
        ("zhipu", "ZHIPU_API_KEY", "ZHIPU_BASE_URL", "glm-4-flash", "simple"),
        ("moonshot", "MOONSHOT_API_KEY", "MOONSHOT_BASE_URL", "moonshot-v1-8k", "simple"),
        ("ollama", None, "OLLAMA_BASE_URL", "qwen2.5:3b", "simple"),
    ]
    entries: List[ModelEntry] = []
    for alias, key_env, url_env, default_model, default_tier in specs:
        keys = _split_keys(os.getenv(key_env)) if key_env else ["dummy"]
        base_url = os.getenv(url_env, "")
        if not keys or not base_url:
            continue  # 未配置则跳过
        entries.append(ModelEntry(
            alias=alias,
            model_name=os.getenv(f"{alias.upper()}_MODEL", default_model),
            base_url=base_url,
            api_keys=keys,
            # 档位可用 {ALIAS}_TIER 覆盖，如 ZHIPU_TIER=complex
            tier=os.getenv(f"{alias.upper()}_TIER", default_tier),
        ))
    return LLMPool(entries, timeout=timeout)


# ==================== 复杂度路由器（原 router.py 并入） ====================
# 推理/深度分析类关键词（命中加分），可用 ROUTER_COMPLEX_KEYWORDS 逗号分隔覆盖
DEFAULT_COMPLEX_KEYWORDS = [
    "为什么", "推理", "推导", "证明", "逐步", "step by step", "分析", "对比",
    "评估", "设计", "架构", "原理", "优缺点", "深入浅出", "详细解释",
    "数学", "计算", "方程", "概率", "复杂度", "算法", "证明题",
    "debug", "排查", "报错", "优化方案", "review", "重构",
]

# 默认判定阈值与长文本阈值
DEFAULT_THRESHOLD = int(os.getenv("ROUTER_COMPLEX_THRESHOLD", "3"))
LONG_TEXT_CHARS = int(os.getenv("ROUTER_LONG_TEXT_CHARS", "120"))


class ComplexityRouter:
    """
    规则打分式复杂度分类器：classify(question) -> "simple" | "complex"

    规则打分（无需额外 LLM 调用，零成本零延迟）：
    - 长度、推理类关键词、多步/多问题、代码与数学诉求等加权求分
    - 分数 >= threshold → complex（走 pro/reasoner 档），否则 simple（走便宜 flash 档）
    - 阈值与关键词可用环境变量覆盖

    与 LLMPool 配合：
        tier = router.classify(question)          # "simple" | "complex"
        chain, alias = pool.get_chain(tier=tier)  # 优先在该档内轮询+降级，跨档兜底
    """

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        raw = os.getenv("ROUTER_COMPLEX_KEYWORDS")
        self.keywords: List[str] = (
            [k.strip() for k in raw.split(",") if k.strip()]
            if raw else DEFAULT_COMPLEX_KEYWORDS
        )

    def score(self, question: str) -> int:
        """对问题打复杂度分（分数越高越复杂）"""
        q = question.strip()
        score = 0
        # 1. 长度：长文本通常承载复杂诉求
        if len(q) >= LONG_TEXT_CHARS:
            score += 2
        elif len(q) >= 60:
            score += 1
        # 2. 推理类关键词，每个命中 +1（封顶 +3，避免堆词刷分）
        hits = sum(1 for k in self.keywords if k.lower() in q.lower())
        score += min(hits, 3)
        # 3. 多问题 / 多步骤（多个问号或分号、编号列表）
        if len(re.findall(r"[?？]", q)) >= 2 or re.search(r"(\d[.、].*){2,}", q):
            score += 2
        # 4. 代码诉求
        if re.search(r"```|\b(python|java|sql|代码|函数|脚本)\b", q, re.I):
            score += 1
        # 5. 数学表达式（含运算符的算式）
        if re.search(r"\d+\s*[-+*/^]\s*\d+", q):
            score += 1
        return score

    def classify(self, question: str) -> str:
        return "complex" if self.score(question) >= self.threshold else "simple"
