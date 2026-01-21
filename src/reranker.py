import requests
import os
import numpy as np
import pandas as pd

from typing import List
from src.const import KLLM

from dotenv import load_dotenv

load_dotenv() 

RERANKER_BEARER_KEY=os.getenv("RERANKER_BEARER_KEY")
RERANKER_URL=os.getenv("RERANKER_URL")

def rerank_docs(query, documents, bearer_key):
    # Формируем заголовок для запроса
    headers = {
        # Указываем тип получаемого контента
        "Content-Type": "application/json",
        # Указываем наш ключ, полученный ранее
        "Authorization": f"Bearer {bearer_key}"
    }

    response = requests.post(
        url = RERANKER_URL,
        headers=headers,
        json={
            "query" : query,
            "documents" : documents
        }
    )

    result_metrics = response.json()
    result_metrics = np.abs(result_metrics)

    return result_metrics


def reranker(user_query: str, message_for_rerank: List[str]):
    result_metrics = rerank_docs(
        query=user_query,
        documents=message_for_rerank,
        bearer_key=RERANKER_BEARER_KEY
    )

    final_sort = pd.DataFrame({
        'score' : result_metrics, 
        'text' : message_for_rerank
    }).sort_values('score', ascending=False)\
    .reset_index(drop=True)

    rag_message = '\n'.join([x for x in final_sort.loc[:KLLM-1, 'text']])

    rag_text_for_llm = f'\n**Для ответа используй следующие знания из RAG базы данных**:\n{rag_message}'

    return rag_text_for_llm