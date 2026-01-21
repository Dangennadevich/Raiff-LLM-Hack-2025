from src.agent.state import State
from src.agent.llm import llm_with_tools, llm_final_answer
from src.rag import rag

from langchain_core.messages import SystemMessage, HumanMessage

def judge_step(state: State) -> State:
    """
    LLM определяет, достаточно ли текущих данных RAG.
    НЕОБХОДИМО вызвать инструмент judge_rag_sufficiency.
    """

    # context_text = "\n\n".join(
    #     f"[source={c['source']}] {c['text']}"
    #     for c in state["rag_data"]
    # )

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
    print('----', 'judge_step')
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
    print("result.tool_calls:", result.tool_calls)

    if not result.tool_calls:
        print('fallback: LLM не смог оценить')
        state["sufficient"] = False
        state["followup_query"] = state["user_question"]
        state["confidence"] = 0.0
        return state

    tool_call = result.tool_calls[0]
    args = tool_call["args"]

    state["sufficient"] = args["sufficient"]
    state["followup_query"] = args.get("followup_query")
    state["confidence"] = args.get("confidence", 0.0)

    return state

def first_call_rag(state: State) -> State:
    """
    Retrieve new documents using followup_query
    """
    print('----', 'first_call_rag')

    query = state["user_question"]

    rag_answer = rag(user_query=query)

    state["rag_data"].append(rag_answer)
    state["iteration"] += 1
    print(state)

    return state

def rag_step(state: State) -> State:
    """
    Retrieve new documents using followup_query
    """
    print('----', 'rag_step')

    if state['iteration'] > 0:
        query = state["followup_query"]
    else:
        query = state["user_question"]

    # 🔧 RAG 
    rag_answer = rag(user_query=query)

    cur_chunk = {
        "text": rag_answer,
        "source": "rag_step",
        "query": query,
        "score": 1.0
    }

    state["rag_data"].append(cur_chunk)
    state["iteration"] += 1

    return state


def final_answer_step(state: State) -> State:
    """
    Final answering LLM
    """
    print('----', 'final_answer_step')

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
    return state