"""
Agent 工具集（全部工具集中于本文件）。

- BaseTool：所有工具统一父类，输出 OpenAI Function Calling 标准 schema
  （dict 形式可直接传给 ChatOpenAI.bind_tools()）。
- CalcTool：安全数学表达式计算（AST 白名单求值）。
- SearchTool：火山引擎豆包搜索联网查询。
- KnowledgeSearchTool：本地知识库检索（延迟 import memory.vector_store，
  避免 import 时强制初始化 Chroma；Ollama 未启动也不影响其他工具）。
- ToolRegistry + registry 全局单例：注册内置工具、批量导出 schema、按名取工具。
"""
import ast
import operator
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


# ==================== 工具基类 ====================
class BaseTool(ABC):
    """所有工具统一父类"""
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    def run(self, **kwargs) -> str:
        """工具执行入口，返回字符串结果"""
        pass

    def get_function_schema(self) -> Dict[str, Any]:
        """输出Function Calling标准Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# ==================== 计算器工具 ====================
# 安全白名单，禁止执行危险代码
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg
}


class CalcTool(BaseTool):
    name = "calculator"
    description = "用于数学表达式计算，遇到算数问题必须调用，支持加减乘除、乘方、取模"
    parameters = {
        "type": "object",
        "required": ["expression"],
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如：345 * 88 + 77"
            }
        }
    }

    def _safe_eval(self, expr: str):
        """安全表达式计算，禁止执行系统代码"""
        tree = ast.parse(expr, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](left, right)
            raise ValueError("不支持的运算符")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](operand)
        raise ValueError("表达式非法")

    def run(self, expression: str, **kwargs) -> str:
        try:
            res = self._safe_eval(expression)
            return f"计算结果：{res}"
        except Exception as e:
            return f"计算失败：{str(e)}"


# ==================== 联网搜索工具 ====================
VOLC_SEARCH_API_KEY = os.getenv("VOLC_SEARCH_API_KEY", "")
VOLC_SEARCH_BASE_URL = os.getenv(
    "VOLC_SEARCH_BASE_URL",
    "https://open.feedcoopapi.com/search_api/web_search"
)


class SearchTool(BaseTool):
    name = "search"
    description = "获取实时、最新信息，日期、天气、实时价格、新闻等知识库不存在的数据必须调用"
    parameters = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询词"
            }
        }
    }

    def run(self, query: str, **kwargs) -> str:
        if not VOLC_SEARCH_API_KEY:
            return "搜索工具未配置VOLC_SEARCH_API_KEY，无法联网查询"

        headers = {
            "Authorization": f"Bearer {VOLC_SEARCH_API_KEY}",
            "Content-Type": "application/json",
        }
        # 火山引擎 豆包搜索 Custom版 请求体
        payload = {
            "Query": query,
            "SearchType": "web",
            "Count": 5,
            "QueryControl": {"QueryRewrite": False},
            "ContentFormats": "markdown",
        }
        try:
            resp = requests.post(VOLC_SEARCH_BASE_URL, headers=headers,
                                 json=payload, timeout=15)
            data = resp.json()
        except Exception as e:
            return f"搜索异常：{str(e)}"

        # 接口层错误（鉴权失败/限流等）
        meta = data.get("ResponseMetadata") or {}
        if meta.get("Error"):
            err = meta["Error"]
            msg = err.get("Message") or err.get("Code") or str(err)
            return f"搜索失败：{msg}"

        result = data.get("Result") or {}
        web_results = result.get("WebResults") or []
        if not web_results:
            return "未搜索到相关信息"

        # 摘要拼接：优先 Summary（500~1000字，适合大模型），其次 Snippet
        # 为节省上下文 token，Summary/Snippet 仅截取前 200 字符
        output = []
        for item in web_results[:5]:
            title = item.get("Title", "")
            snippet = item.get("Summary") or item.get("Snippet") or ""
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            url = item.get("Url", "")
            output.append(f"{title}\n{snippet}\n链接：{url}")
        return "\n\n".join(output)


# ==================== 本地知识库检索工具 ====================
class KnowledgeSearchTool(BaseTool):
    """
    本地知识库检索工具：让通用 ReActAgent 能按需检索已摄入 Chroma 的本地文档。
    异常时返回友好提示，不中断 Agent 循环。
    """
    name = "knowledge_search"
    description = (
        "检索本地知识库（RAG 文档）。当用户提问涉及已上传的文档/资料/手册/说明内容时调用，"
        "返回最相关的文档片段。若知识库为空会提示先摄入文档。"
    )
    parameters = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询词，描述想从文档中查找的内容",
            },
            "top_k": {
                "type": "integer",
                "description": "返回的文档片段数量，默认 4",
                "default": 4,
            },
        },
    }

    def run(self, query: str, top_k: int = 4, **kwargs) -> str:
        # 延迟导入 + 延迟初始化单例，避免 registry import 时即连 Chroma
        try:
            from memory.vector_store import get_vector_store

            vs = get_vector_store()
            hits = vs.query_documents(query, top_k=top_k)
        except Exception as e:
            return f"知识库检索失败：{e}"

        if not hits:
            return (
                "知识库为空或未检索到相关内容。"
                "请先通过 RAGAgent.ingest_document 摄入文档后再调用。"
            )

        blocks = []
        for i, h in enumerate(hits, 1):
            src = h.get("source", "")
            score = h.get("score", 0)
            blocks.append(
                f"[片段{i}] (相似度={score:.3f}, 来源:{src})\n{h['text']}"
            )
        return "\n\n".join(blocks)


# ==================== 工具注册中心 ====================
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        # 自动注册内置工具
        self.register(CalcTool())
        self.register(SearchTool())
        # 注册知识库检索工具（可选，注册失败不阻塞其他工具）
        try:
            self.register(KnowledgeSearchTool())
        except Exception:
            pass

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def get_all_schemas(self) -> List[dict]:
        """批量获取所有工具Function Calling Schema，直接传给LLM"""
        return [tool.get_function_schema() for tool in self.tools.values()]


# 全局单例注册器
registry = ToolRegistry()
