from src.tokenizer import enc
from typing import Any

def chunkinizer(
    vanila_chunkinizer:int=0,
    question: Any = None,
    answer: Any = None, 
    text: Any = None,
    chunk_size: int = 512,
    overlap_part: float = 0.2,
    enc: Any = enc
):
    """
    Разбивает текст на чанки по количеству токенов, используя tiktoken.
    Для стартегии vanila_chunkinizer нет логики дополнительной обработки текста.
    Альтернативная логика - подставляем вопрос (выделяется из структурированных данных) в каждый чанк

    Args:
        vanila_chunkinizer: настройка стратегии чанкования
        text: полный текст для чанкования (vanila_chunkinizer=1)
        question: вопрос (vanila_chunkinizer=0)
        answer: ответ (vanila_chunkinizer=0)
        chunk_size: размер чанка в токенах.
        overlap_part: доля перекрытие в токенах между чанками.
        enc: токенизатор.

    Returns:
        Список чанков (строк).
    """
    assert vanila_chunkinizer in (0,1), \
            'не верное значение! 1 для vanila_chunkinizer 0 для чанкования с добавлением вопросом!'
    
    # Количество токенов перекрытия
    overlap_tokens = int(chunk_size * overlap_part)

    chunks = []
    start = 0

    if vanila_chunkinizer == 0:
        assert question is not None and answer  is not None, \
                'Отсутсвует question или answer для стратегии чанкования с добавлением вопроса'
        # Токены вопроса и ответа
        tokens_answer = enc.encode(answer)
        tokens_question = enc.encode(question)
        # Количество токенов вопроса
        len_tokens_question = len(tokens_question)
        # Избегаем ситуации с очень длинным вопросом
        if len_tokens_question > chunk_size/2:
            new_question_str_len = len(question)/2
            tokens_question = enc.encode(question[:new_question_str_len])
            len_tokens_question = len(tokens_question)
        # Размера чанка, который заполняется ответом
        answer_chunk_size = chunk_size - len_tokens_question
        
        while start < len(tokens_answer):
            # Поулчаем токены части ответа
            end = start + answer_chunk_size
            chunk_tokens = tokens_answer[start:end]

            # Текущий чанк Вопрос + Ответ
            concat_tokens = tokens_question + chunk_tokens
            chunk_text = enc.decode(concat_tokens)

            chunks.append(chunk_text)

            # Двигаем старт с учетом перекрытия
            start += answer_chunk_size - overlap_tokens

            if end >= len(tokens_answer):
                break
    
    else:
        assert text, 'Отсутсвует text для стратегии чанкования с добавлением вопроса'

        tokens = enc.encode(text)

        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += chunk_size - overlap_tokens
            if end >= len(tokens):
                break

    return chunks