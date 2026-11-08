"""
LLM 调用附加设施：Token 计费统计（UsageTracker）+ Query 缓存（QueryCache）。

UsageTracker（原 utils/billing.py 并入）：
- record()：追加一条调用记录到 JSONL（含时间戳、模型、tier、token 数、估算成本）
- summary()：聚合统计（按模型分组 + 总计），供 /cost 命令展示
- 单价表 PRICE_TABLE 单位为 元/百万 token，可用环境变量 MODEL_PRICES_JSON 覆盖，
  如 MODEL_PRICES_JSON={"deepseek-chat":{"input":2,"output":8}}
- usage 来源：LangChain stream_usage=True 时末块 AIMessageChunk.usage_metadata
  {"input_tokens":.., "output_tokens":.., "total_tokens":..}

QueryCache（原 utils/query_cache.py 并入）：
- key：query 归一化（去空白/全小写/去尾部标点）后的 SHA256
- 存储：JSON 文件持久化（跨重启生效），带 TTL（默认 7 天，QUERY_CACHE_TTL_SECONDS 覆盖）
- 命中判定：完全一致才命中（语义近似缓存需引入向量检索，暂未启用）
"""
import hashlib
import json
import os
import threading
import time
from typing import Dict, Optional

from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()

USAGE_LOG_PATH = os.getenv("TOKEN_USAGE_LOG", "./data/token_usage.jsonl")

# 参考单价（元/百万 token），以各厂商官网为准；未知模型按 0 计（仅统计 token 数）
DEFAULT_PRICE_TABLE = {
    "deepseek-chat":     {"input": 2.0,  "output": 8.0},
    "deepseek-reasoner": {"input": 4.0,  "output": 16.0},
    "qwen-plus":         {"input": 0.8,  "output": 2.0},
    "glm-4-flash":       {"input": 0.0,  "output": 0.0},   # 免费
    "moonshot-v1-8k":    {"input": 12.0, "output": 12.0},
    "qwen2.5:3b":        {"input": 0.0,  "output": 0.0},   # 本地
}


def _load_price_table() -> Dict[str, Dict[str, float]]:
    raw = os.getenv("MODEL_PRICES_JSON")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[计费] MODEL_PRICES_JSON 解析失败，使用内置单价表")
    return DEFAULT_PRICE_TABLE


class UsageTracker:
    """usage 采集 + 成本估算 + JSONL 落盘（线程安全）"""

    def __init__(self, log_path: str = USAGE_LOG_PATH):
        self.log_path = log_path
        self.price_table = _load_price_table()
        self._lock = threading.Lock()

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.price_table.get(model)
        if not price:
            return 0.0
        return (input_tokens * price.get("input", 0)
                + output_tokens * price.get("output", 0)) / 1_000_000

    def record(
        self,
        model: str,
        alias: str,
        usage: Optional[Dict],
        tier: str = "",
        cached: bool = False,
    ) -> Optional[Dict]:
        """
        记录一次调用。usage 为 None（部分厂商流式不返回 usage）时仅记次数。
        返回本条记录（含估算成本），便于调用方展示。
        """
        input_tokens = (usage or {}).get("input_tokens", 0)
        output_tokens = (usage or {}).get("output_tokens", 0)
        cost = 0.0 if cached else self._estimate_cost(model, input_tokens, output_tokens)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "alias": alias,
            "model": model,
            "tier": tier,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_cny": round(cost, 6),
            "cached": cached,
        }
        try:
            with self._lock:
                os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"[计费] 写 usage 日志失败: {e}")
        return record

    def summary(self) -> Dict:
        """聚合全部历史记录：按模型分组 + 总计"""
        by_model: Dict[str, Dict] = {}
        total = {"calls": 0, "total_tokens": 0, "cost_cny": 0.0, "cache_hits": 0}
        if not os.path.exists(self.log_path):
            return {"by_model": by_model, "total": total}
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bucket = by_model.setdefault(
                    r["model"],
                    {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                     "total_tokens": 0, "cost_cny": 0.0},
                )
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    bucket[k] += r.get(k, 0)
                bucket["calls"] += 1
                bucket["cost_cny"] = round(bucket["cost_cny"] + r.get("cost_cny", 0), 6)
                total["calls"] += 1
                total["total_tokens"] += r.get("total_tokens", 0)
                total["cost_cny"] = round(total["cost_cny"] + r.get("cost_cny", 0), 6)
                total["cache_hits"] += 1 if r.get("cached") else 0
        return {"by_model": by_model, "total": total}


# ==================== Query 缓存（原 utils/query_cache.py 并入） ====================
CACHE_PATH = os.getenv("QUERY_CACHE_PATH", "./data/query_cache.json")
CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL_SECONDS", str(7 * 86400)))


def _normalize(query: str) -> str:
    """归一化：压缩空白、转小写、去尾部标点，提升命中率"""
    q = " ".join(query.split()).lower()
    return q.rstrip("?？!！。. ")


class QueryCache:
    """精确匹配 query 缓存（JSON 持久化）：重复提问直接命中返回，0 token 成本"""

    def __init__(self, path: str = CACHE_PATH, ttl: int = CACHE_TTL):
        self.path = path
        self.ttl = ttl
        self._store = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"[缓存] 加载失败，重建空缓存: {e}")
        return {}

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"[缓存] 写入失败: {e}")

    def _key(self, query: str) -> str:
        return hashlib.sha256(_normalize(query).encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """命中返回答案，未命中或过期返回 None"""
        entry = self._store.get(self._key(query))
        if not entry:
            return None
        if time.time() - entry["ts"] > self.ttl:
            return None
        return entry["answer"]

    def set(self, query: str, answer: str) -> None:
        if not answer:
            return
        self._store[self._key(query)] = {"ts": time.time(), "answer": answer}
        self._save()

    def clear(self) -> None:
        self._store = {}
        self._save()

    @property
    def size(self) -> int:
        return len(self._store)
