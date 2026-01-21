
from langchain_core.tools import tool
from typing import Optional

@tool
def judge_rag_sufficiency(
    sufficient: bool,
    followup_query: Optional[str] = None,
    confidence: float = 0.0
):
    """
    Реши, является ли предоставленный контекст RAG достаточным для ответа на вопрос пользователя.

    Если контекста не хватает:
      - sufficient = False
      - followup_query должен быть конкретный поисковый запрос для базы данных

    Если контекста хватает:
      - sufficient = True
      - followup_query должно быть Null
    """
    return {
        "sufficient": sufficient,
        "followup_query": followup_query,
        "confidence": confidence,
    }