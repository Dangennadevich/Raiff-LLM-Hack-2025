from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.llm import llm_with_tools, llm_final_answer
from src.agent.logging import get_node_logger
from src.agent.state import State
from src.rag import rag

import time

def judge_step(state: State) -> State:
    """
    LLM определяет, достаточно ли текущих данных RAG.
    НЕОБХОДИМО вызвать инструмент judge_rag_sufficiency.
    """
    start = time.perf_counter()
    logger = get_node_logger("judge_step")

    context_text = "\n\n".join(
        f"{c}"
        for c in state["rag_data"]
    )

    system_prompt = """
Ты — агент принятия решений RAG-системы.

Твоя задача:
- оценить, достаточно ли контекста, чтобы ответить на вопрос пользователя
- если недостаточно — сформулировать ОДИН точный поисковый запрос для БД

⚠️ **Ты ОБЯЗАН вызвать функцию judge_rag_sufficiency**!
⚠️ **Действуй согласно инструкции!**.
⚠️ **Не отвечай текстом!**
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Вопрос пользователя:
{state['user_question']}

Контекст:
{context_text}
""")
    ]

    # LLM выберет инструмент и заполнит его аргументы
    result = llm_with_tools.invoke(messages) 
        
    if not result.tool_calls:
        logger.warning("LLM returned no tool calls, fallback activated")
        state["sufficient"] = False
        state["followup_query"] = state["user_question"]
        state["confidence"] = 0.0
        return state

    tool_call = result.tool_calls[0]
    args = tool_call["args"]

    state["sufficient"] = args["sufficient"]
    state["followup_query"] = args.get("followup_query")
    state["confidence"] = args.get("confidence", 0.0)

    duration = time.perf_counter() - start
    logger.info(
        "LLM tool call result",
        extra={
            "session_id": state.get("session_id"),
            "tool_calls": result.tool_calls,
            "duration_ms": round(duration * 1000, 2),
        },
    )

    return state

def first_call_rag(state: State) -> State:
    """
    Retrieve new documents using followup_query
    """
    start = time.perf_counter()
    logger = get_node_logger("first_call_rag")

    logger.info("first_call_rag entered")

    query = state["user_question"]

    rag_answer = rag(user_query=query)

    state["rag_data"].append(rag_answer)
    state["iteration"] += 1

    duration = time.perf_counter() - start
    logger.info(
        "First RAG result appended",
        extra={
            "session_id": state.get("session_id"),
            "iteration": state["iteration"],
            "rag_chunk_len": len(state["rag_data"][0]),
            "duration_ms": round(duration * 1000, 2),
        },
    )

    return state

def rag_step(state: State) -> State:
    """
    Retrieve new documents using followup_query
    """
    start = time.perf_counter()
    logger = get_node_logger("rag_step")

    # query = state["followup_query"] or state["user_question"]
    if state['iteration'] > 0:
        query = state["followup_query"]
    else:
        query = state["user_question"]

    logger.info(
        "RAG query issued",
        extra={
            "session_id": state.get("session_id"),
            "iteration": state["iteration"],
            "query": query,
        },
    )

    rag_answer = rag(user_query=query)

    state["rag_data"].append({
        "text": rag_answer,
        "source": "rag_step",
        "query": query,
        "score": 1.0
    })
    state["iteration"] += 1

    duration = time.perf_counter() - start
    logger.info(
        "RAG result appended",
        extra={
            "session_id": state.get("session_id"),
            "iteration": state["iteration"],
            "rag_chunks": len(state["rag_data"]),
            "duration_ms": round(duration * 1000, 2),
        },
    )

    return state


def final_answer_step(state: State) -> State:
    """
    Final answering LLM
    """
    start = time.perf_counter()
    logger = get_node_logger("final_answer_step")

    context_text = "\n\n".join(
        f"{c}"
        for c in state["rag_data"]
    )

    prompt = [
        SystemMessage(content="""
Ты — помощник.
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленный контекст.
Не выдумывай факты.
"""),
        HumanMessage(content=f"""
Вопрос:
{state['user_question']}

Контекст:
{context_text}
""")
    ]

    answer = llm_final_answer.invoke(prompt)

    state["final_answer"] = answer.content

    duration = time.perf_counter() - start
    logger.info(
        "Final answer generated",
        extra={
            "session_id": state.get("session_id"),
            "rag_chunks": len(state["rag_data"]),
            "confidence": state["confidence"],
            "answer_length": len(state["final_answer"] or ""),
            "duration_ms": round(duration * 1000, 2),
        },
    )

    return state