import pandas as pd
from scipy.special import expit
import numpy as np
import math


def data_to_storage(id_series: pd.Series, data_series: pd.Series):
    '''Функция парсит данные из датасета, собирая в JSON БД
    
    Args:
        id_series - столбец датасета с идентификатором документа
        data_series - стобец датасета с чанками
    Output:
        JSON БД
    '''
    storage = {}
    key_id = 0

    for row_num in range(len(data_series)):
        id_doc = id_series[row_num]
        data_document = data_series[row_num]

        for document in data_document:
            # Проверка на дополнительную вложенность
            if isinstance(document, list):
                for chunk in document:

                    storage[key_id] = (id_doc, chunk)
                    key_id += 1
            else:
                storage[key_id] = (id_doc, document)
                key_id += 1

    print(f"Storage ready, key from 0 to {key_id-1}")

    return storage

def second_preprocess(text: str) -> str:
    '''Удаляет лишние проблемы и переносы строк'''
    text = text.replace(' \n', '\n')
    text = text.replace('\n ', '\n')
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ').replace('   ', ' ')
    return text.strip()

def parse_question(block:str) -> tuple:
    '''Находит вопрос, извлекает его, удаляет из исходного текста
    
    Args:
        row - один блок до обработки
    Returns:
        tuple, где на 0 позиции вопрос (или '' если вопроса не было), на позиции 1 ответ
    
    '''
    candidats = block.split('\n')
    candidats = [row for row in candidats if row.strip()]

    if '?' in candidats[0]:
        question = candidats[0]
        answer = block.replace(question, '')
    else:
        answer = block
        question = ''

    question = second_preprocess(question)
    answer = second_preprocess(answer)
    
    return question, answer

def split_text(row:str):
    '''Функция сплитует по вопросу'''
    row = row.replace('###','')
    chunks = row.split('##')
    chunks = [chunk for chunk in chunks if chunk != '']
    return chunks

def parse_text(row:list):
    '''Получает на вход документ.
    Обрабатывает каждый блок документа его при помощи parse_question.

    Args:
        row - один документ
    Returns:
        Список с обработанными блоками, где каждый блок это tuple с вопросом и ответом. Вопрос может быть пустым.
    '''
    return [parse_question(bloc) for bloc in row]

def preprocess(df):
    # annotation
    df['annotation'] = df['annotation'].apply(lambda x: '' if x is np.nan else x)

    # tags
    df['tags'] = [row[1:-1].replace("'", "") for row in df['tags']]

    # text
    df['text'] = df['text'].str.replace(r'Обновлено \d{2}\.\d{2}\.\d{4} в \d{2}:\d{2}', '', regex=True)

    df['text'] = df['text'].apply(lambda x: split_text(x))

    df['text'] = df['text'].apply(lambda x: parse_text(x))

    return df

def z_logistic(s, k=1.0):
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    return expit(k * z)


def is_bad_answer(ans):
    # пусто или None
    if ans is None:
        return True
    
    # np.nan или pd.NA
    if isinstance(ans, float) and math.isnan(ans):
        return True
    if pd.isna(ans):
        return True
    
    # не строка → считаем плохим
    if not isinstance(ans, str):
        return True
    
    # слишком короткий ответ → плохой
    if len(ans.strip()) < 10:
        return True
    
    return False


def normalize_vector(embedding: np.ndarray) -> np.ndarray:
    """
    Normalizes a given vector to have unit length.

    Args:
        embedding (np.ndarray): A NumPy array representing the vector to normalize.

    Returns:
        np.ndarray: A normalized vector with unit length.
    """

    norm = np.linalg.norm(embedding)
    if abs(norm) >= 1e-9: #защита от взрыва и погрешности
      embedding /= norm

    return embedding