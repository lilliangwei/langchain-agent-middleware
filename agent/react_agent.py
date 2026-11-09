"""
ReAct 工具型 Agent（LangChain 版，基于 llm_pool）。

- LLM 调用统一走 LLMPool：档位内多模型轮询 + 多密钥降级链，
  工具绑定用 RunnableWithFallbacks.bind_tools（自动同步到所有候选客户端）。
- 工具 schema 直接复用 BaseTool.get_function_schema() 的 OpenAI 格式 dict，
  ChatOpenAI.bind_tools() 原生接受 dict。
- 长期记忆：可选注入 MemoryManager。启用后每轮用 get_context 组装上下文
  （长期记忆召回 + 摘要 + 最近短期），轮次结束仅写回 user/assistant 最终结果；
  不启用时自维护 messages 列表（LangChain 消息对象）。
"""
import os
from typing import Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.tools import registry
from llm_pool import LLMPool

# 长期记忆管理器：可选依赖，memory 包未就绪时 ReActAgent 仍可正常工作（向后兼容）
try:
    from memory.memory_manager import MemoryManager
except Exception:  # noqa: BLE001
    MemoryManager = None  # type: ignore

load_dotenv()
MAX_REACT_ROUND = int(os.getenv("AGENT_MAX_REACT_ROUND", 5))
ENABLE_REACT_LOG = os.getenv("AGENT_ENABLE_REACT_LOG", "True") == "True"


def log_react(text: str):
    """统一打印ReAct思考链路日志"""
    if ENABLE_REACT_LOG:
        print(f"\033[34m[ReAct] {text}\033[0m")


def _dict_to_lc_messages(messages: List[Dict]) -> List[BaseMessage]:
    """dict 消息（OpenAI 格式）转 LangChain 消息对象"""
    lc: List[BaseMessage] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            lc.append(SystemMessage(content=content))
        elif role == "user":
            lc.append(HumanMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
        elif role == "tool":
            lc.append(ToolMessage(
                content=content,
                tool_call_id=m.get("tool_call_id", ""),
            ))
    return lc


class ReActAgent:
    def __init__(
        self,
        pool: LLMPool,
        default_tier: Optional[str] = None,
        memory_manager: Optional["MemoryManager"] = None,
        session_id: str = "default"
    ):
        self.pool = pool
        self.default_tier = default_tier
        self.tool_schemas = registry.get_all_schemas()
        # 长期记忆：可选注入。启用后短期记忆由 MemoryManager 统一管理，
        # 并可在每轮召回长期记忆 + 触发摘要压缩；不启用时自维护消息列表。
        self.memory_manager = memory_manager
        self.session_id = session_id
        self.messages: List[BaseMessage] = []
        # 最近一次 run_stream 的计费信息（跨工具轮累计）
        self.last_usage: Optional[Dict[str, int]] = None
        self.last_model: Optional[str] = None

    def clear_history(self):
        """清空当前会话上下文（有 memory_manager 时仅清短期，不动摘要/向量长期记忆）"""
        if self.memory_manager:
            self.memory_manager.clear_session(self.session_id)
        self.messages = []

    def _init_messages(self, user_query: str) -> List[BaseMessage]:
        """组装本轮上下文：启用长期记忆时走 get_context（末尾已含当前问题），否则自维护消息列表"""
        if self.memory_manager:
            return _dict_to_lc_messages(
                self.memory_manager.get_context(self.session_id, user_query, top_k=3)
            )
        self.messages.append(HumanMessage(content=user_query))
        return self.messages

    def _execute_tool_calls(self, lc_messages: List[BaseMessage], ai_msg: AIMessage):
        """把 AIMessage 与各工具的执行结果（ToolMessage）追加进上下文"""
        lc_messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            log_react(f"Thought：准备调用工具 {t_name}")
            log_react(f"Action：{t_name} 参数={t_args}")

            tool = registry.get_tool(t_name)
            if not tool:
                obs = f"错误：不存在工具 {t_name}"
            else:
                try:
                    obs = tool.run(**t_args)
                except Exception as e:
                    obs = f"工具 {t_name} 执行异常：{e}"
            log_react(f"Observation：{obs}\n")
            lc_messages.append(ToolMessage(
                content=str(obs),
                tool_call_id=tool_call["id"],
            ))

    def _react_loop(self, user_query: str) -> Tuple[List[BaseMessage], Optional[AIMessage]]:
        """
        内部公共 ReAct 工具调用循环（流式、非流式共用）。

        返回 (完整消息列表, 最终 AIMessage)：
        - 最终 AIMessage 不为 None：模型已给出无工具调用的文本回答，可直接使用；
        - 为 None：达到最大工具调用轮次，调用方需再做一次最终生成。

        启用长期记忆时：召回相关长期记忆 + 摘要 + 最近短期，组装本轮上下文
        （get_context 返回的 list 末尾已含当前 user_query，无需再 append）；
        未启用时：自维护 self.messages，直接 append 当前 user_query。
        """
        lc_messages = self._init_messages(user_query)

        round_count = 0
        while round_count < MAX_REACT_ROUND:
            # 每轮取新链：档位内轮询选主模型 + 完整降级链，再绑定工具
            chain = self.pool.get_chain(tier=self.default_tier)[0].bind_tools(
                self.tool_schemas
            )
            ai_msg = chain.invoke(lc_messages)

            # 存在工具调用，执行工具
            if ai_msg.tool_calls:
                round_count += 1
                self._execute_tool_calls(lc_messages, ai_msg)
                continue

            # 无需工具，循环结束，本轮 ai_msg 即最终回答
            log_react("Thought：信息充足，准备生成最终回答")
            return lc_messages, ai_msg

        # 达到最大工具调用轮次，终止循环
        log_react(f"达到最大工具调用轮次{MAX_REACT_ROUND}，停止工具调用")
        return lc_messages, None

    def _final_generate(self, lc_messages: List[BaseMessage]) -> AIMessage:
        """达到最大轮次后的兜底最终生成（不绑工具，强制文本回答）"""
        chain = self.pool.get_chain(tier=self.default_tier)[0]
        return chain.invoke(lc_messages)

    def run(self, user_query: str) -> str:
        """非流式接口：一次性返回完整答案"""
        lc_messages, final_msg = self._react_loop(user_query)
        if final_msg is None:
            final_msg = self._final_generate(lc_messages)
        answer = final_msg.content
        if not self.memory_manager:
            self.messages.append(AIMessage(content=answer))
        self._post_turn(user_query, answer)
        return answer

    def run_stream(self, user_query: str) -> Generator[str, None, None]:
        """
        流式接口：每一轮都用 chain.stream 实时产出文本块。
        工具调用轮通常无正文内容，最终回答轮则逐 token 流式返回，前端无需等待整段生成。
        """
        lc_messages = self._init_messages(user_query)
        answer = ""
        round_count = 0
        self.last_usage = None
        self.last_model = None

        while round_count < MAX_REACT_ROUND:
            chain = self.pool.get_chain(tier=self.default_tier)[0].bind_tools(
                self.tool_schemas
            )
            collected = None
            for chunk in chain.stream(lc_messages):
                # 聚合完整 AIMessageChunk，以读取 tool_calls 与最终文本
                collected = chunk if collected is None else collected + chunk
                # 累计计费信息（usage 仅在末块出现，模型名取自 response_metadata）
                u = getattr(chunk, "usage_metadata", None)
                if u:
                    self.last_usage = self.last_usage or {}
                    for k in ("input_tokens", "output_tokens", "total_tokens"):
                        self.last_usage[k] = self.last_usage.get(k, 0) + (u.get(k) or 0)
                m = (chunk.response_metadata or {}).get("model_name")
                if m:
                    self.last_model = m
                if chunk.content:
                    answer += chunk.content
                    yield chunk.content
            if collected is None:
                break

            ai_msg = AIMessage(
                content=collected.content, tool_calls=collected.tool_calls or []
            )
            if not ai_msg.tool_calls:
                # 本轮即最终回答，已在上方流式输出完毕
                break
            round_count += 1
            self._execute_tool_calls(lc_messages, ai_msg)
        else:
            # 达到最大工具轮次：不绑工具做一次兜底生成，同样逐块流式
            log_react(f"达到最大工具调用轮次{MAX_REACT_ROUND}，停止工具调用")
            chain = self.pool.get_chain(tier=self.default_tier)[0]
            for chunk in chain.stream(lc_messages):
                u = getattr(chunk, "usage_metadata", None)
                if u:
                    self.last_usage = self.last_usage or {}
                    for k in ("input_tokens", "output_tokens", "total_tokens"):
                        self.last_usage[k] = self.last_usage.get(k, 0) + (u.get(k) or 0)
                m = (chunk.response_metadata or {}).get("model_name")
                if m:
                    self.last_model = m
                if chunk.content:
                    answer += chunk.content
                    yield chunk.content

        if not self.memory_manager:
            self.messages.append(AIMessage(content=answer))
        # 流式结束后统一写回记忆（在生成器耗尽后触发）
        self._post_turn(user_query, answer)

    def _post_turn(self, user_query: str, answer: str):
        """
        一轮对话结束后：写入短期记忆并按需触发摘要压缩。
        - 仅启用 memory_manager 时生效；否则空操作。
        - 工具循环的中间消息不持久化，仅记录用户提问与最终答案（语义清晰、摘要质量高）。
        """
        if not self.memory_manager or not answer:
            return
        try:
            self.memory_manager.add_user_message(self.session_id, user_query)
            self.memory_manager.add_assistant_message(self.session_id, answer)
            self.memory_manager.maybe_summarize(self.session_id)
            self.memory_manager.save()
        except Exception as e:
            # 记忆写入失败不应影响主流程
            print(f"[记忆] 写入失败：{e}")
