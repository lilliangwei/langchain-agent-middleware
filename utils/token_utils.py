from typing import List, Dict
import tiktoken
from utils.logger import logger

# 通用token编码器，兼容大部分openai兼容模型
def count_message_tokens(messages: List[Dict], model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = 0
    for msg in messages:
        tokens += 4
        for k, v in msg.items():
            tokens += len(enc.encode(str(v)))
    tokens += 2
    return tokens

def auto_truncate_messages(
    messages: List[Dict],
    max_tokens: int,
    reserve_ratio: float,
    enable_summary: bool,
    llm_call_func = None
) -> List[Dict]:
    """
    自动截断消息
    :param messages: 原始消息列表
    :param max_tokens: 模型最大上下文
    :param reserve_ratio: 触发阈值
    :param enable_summary: 是否启用摘要压缩
    :param llm_call_func: 摘要生成回调（需要传入模型对话函数）
    """
    msg_copy = messages.copy()
    threshold = int(max_tokens * reserve_ratio)

    current = count_message_tokens(msg_copy)
    if current <= threshold:
        return msg_copy

    logger.warning(f"消息上下文超限，触发截断，当前token:{current},阈值:{threshold}")
    # 方案1：摘要压缩
    if enable_summary and llm_call_func is not None:
        # 保留最近2轮对话，久远历史进行摘要
        keep_recent = 2
        if len(msg_copy) <= keep_recent + 1:
            while len(msg_copy) > 2 and count_message_tokens(msg_copy) > threshold:
                msg_copy.pop(1)
            return msg_copy
        history_part = msg_copy[1:-keep_recent]
        recent_part = msg_copy[-keep_recent:]
        summary_prompt = f"请精简总结下面对话内容，保留关键信息，尽量简短：\n{history_part}"
        summary = llm_call_func(summary_prompt, system_prompt="精简总结对话")
        new_messages = [
            msg_copy[0],
            {"role":"system", "content":f"【历史对话摘要】{summary}"},
            *recent_part
        ]
        return new_messages
    # 方案2：直接删除最早对话
    while len(msg_copy) > 2 and count_message_tokens(msg_copy) > threshold:
        msg_copy.pop(1)
    return msg_copy