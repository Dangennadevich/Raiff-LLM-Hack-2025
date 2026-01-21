from openai import OpenAI
from dotenv import load_dotenv
from src.const import KLLM, TEXT_EMBEDDINGS_DIMENSIONS, ANNOTATION_EMBEDDINGS_DIMENSIONS
from src.utils import is_bad_answer, normalize_vector

import pandas as pd
import requests
import time
import os

load_dotenv() 

LLM_API_KEY = os.getenv('LLM_API_KEY')
EMBEDDER_API_KEY = os.getenv('EMBEDDER_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
EMBEDDER_MODEL_NAME = 'openai/text-embedding-3-small'


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

def get_embedding(text, client=client, batch_embeddings:bool = False):
    # Формируем запрос к клиенту
    response = client.embeddings.create(
        # Выбираем любую допступную модель из предоставленного списка
        model=EMBEDDER_MODEL_NAME,
        # Отправяем запрос
        input=text,
    )

    if batch_embeddings:
        return response
    else:
        return response.data[0].embedding


def get_question_embeddings(user_query:str):
    # Эмбединг вопроса для текста
    query_to_text = get_embedding(text=user_query, dimensions=TEXT_EMBEDDINGS_DIMENSIONS) 
    query_to_text = normalize_vector(query_to_text)

    # Эмбединг вопроса для тега + аннотации
    quert_to_annotations = get_embedding(text=user_query, dimensions=ANNOTATION_EMBEDDINGS_DIMENSIONS) 
    quert_to_annotations = normalize_vector(quert_to_annotations)

    return query_to_text, quert_to_annotations

def rerank_docs(query, documents, key):
    # Базовый url - сохранять без изменения
    url = "https://ai-for-finance-hack.up.railway.app/rerank"
    # Формируем заголовок для запроса
    headers = {
        # Указываем тип получаемого контента
        "Content-Type": "application/json",
        # Указываем наш ключ, полученный ранее
        "Authorization": f"Bearer {key}"
    }
    # Формируем сам запрос
    payload = {
        # Указываем модель
        "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
        # Формируем текст запроса
        "query": query,
        # Добавляем документы для ранжирования
        "documents": documents
    }
    # Отправляем запрос
    response = requests.post(url, headers=headers, json=payload)
    # Возвращаем результат запроса
    return response.json()

def reranker(user_query, message_for_rerank):
    answer = {}
    iter = 0

    # Костыль т.к. реранкер выкидывает иногда 502 (раз в 150 итераций)
    while 'results' not in answer:
        answer = rerank_docs(
            query=user_query,
            documents=message_for_rerank,
            key=EMBEDDER_API_KEY
        )
        if 'results' not in answer:
            iter += 1
            print(f'Error reranker! Iteration: {iter}, answer: {answer}')
            time.sleep(2**iter)

        if iter == 7:
            break

    result_metrics = [grade['relevance_score'] for grade in answer['results']]

    final_sort = pd.DataFrame({
        'score' : result_metrics, 
        'text' : message_for_rerank
    }).sort_values('score', ascending=False)\
    .reset_index(drop=True)

    rag_message = '\n'.join([x for x in final_sort.loc[:KLLM-1, 'text']])

    rag_text_for_llm = f'\n**Для ответа используй следующие знания из RAG базы данных**:\n{rag_message}'

    return rag_text_for_llm

def answer_generation(question):
    # Подключаемся к модели
    client = OpenAI(
        # Базовый url - сохранять без изменения
        base_url="https://ai-for-finance-hack.up.railway.app/",
        # Указываем наш ключ, полученный ранее
        api_key=LLM_API_KEY,
    )

    system_prompt = """Ты — RAG-ассистент, помогающий по банковским, финансовым и деловым вопросам.

Тебе будет передан:
1) Вопрос пользователя.
2) Набор фрагментов контекста (RAG база данных), полученных из проверенных источников.

Твоя задача — дать точный, полезный и понятный ответ, опираясь на предоставленный контекст.

Правила:
- Используй информацию, которая есть в контексте. 
- Пиши человеческим, деловым и уважительным тоном.
- Не повторяй вопрос пользователя.
- Не упоминай «контекст», «в тексте говорится», «в документе написано». Формулируй ответ так, будто ты владеешь знанием напрямую.
- Перед отправкой ответа быстро проверь, что все утверждения действительно подтверждаются контекстом (но не объясняй эту проверку вслух, не пиши проверку в ответе!).
- Отвечай в формате markdown для удобства чтения.

Формат ответа:
1) Короткое дружелюбное приветствие, отметь важность вопроса (1-2 предложение).
2) Прямой и понятный ответ по существу (4–10 предложений).
3) Если уместно — конкретный пример или разъяснение (1–3 предложения).
4) Завершение одним предложением, показывающим готовность помочь дальше.
"""
    # Параметры
    max_retries = 5
    delay_base = 1  # Начальная задержка в секундах

    # Формируем сообщение один раз, чтобы не дублировать
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": f"Ответь на вопрос: {question}"}]}
    ]

    response = None
    for attempt in range(max_retries):
        try:
            # Формируем запрос к клиенту
            response = client.chat.completions.create(
                model="openrouter/google/gemma-3-27b-it",
                messages=messages
            )
            model_responce = response.choices[0].message.content
            break
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < max_retries - 1:  # Не ждем после последней попытки
                # Экспоненциальное откладывание (exponential backoff) + jitter
                delay = delay_base * (2 ** attempt)
                print(f"Ждем {delay:.2f} секунд перед следующей попыткой...")
                time.sleep(delay)
            else:
                model_responce = ' '
                print("Все попытки исчерпаны. Запрос не выполнен.")

    # Формируем ответ на запрос и возвращаем его в результате работы функции
    return model_responce # response.choices[0].message.content

def get_answer_with_retries(question, retries=5):
    for i in range(retries):
        ans = answer_generation(question=question)

        if is_bad_answer(ans):
            wait = 2 ** i
            print(f"Bad answer detected. Retry {i+1}/{retries} after {wait}s...")
            time.sleep(wait)
            continue
        
        return ans  # успех

    return 'Прошу прощения, я на данный момент не могу ответить на ваш запрос ввиду технических проблем, повторите попытку позже, спасибо!'