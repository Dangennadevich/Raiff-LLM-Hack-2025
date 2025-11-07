from collections import Counter, defaultdict
from scipy.special import expit
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any
from tqdm import tqdm


import pandas as pd
import numpy as np
import tiktoken
import logging
import pickle
import faiss
import math
import re
import os

load_dotenv() 

EMBEDDER_API_KEY = os.getenv('EMBEDDER_API_KEY')
LLM_API_KEY = os.getenv('LLM_API_KEY')

TEXT_CHUNK_SIZE = 756
TAGS_ANNOTATIONS_CHUNK_SIZE = 256

TEXT_OVERLAP = 0.2
TAGS_ANNOTATIONS_OVERLAP = 0.75

BM25_KOEF = 0.2
EMBED_KA_KOEF = 0.15
EMBED_T_KOEF = 0.65

K = 10 # Топ K при поиске в faiss
BM25_TOPK = 10 # Top k для BM25
SK = 6 # Топ K документов для RAG

enc = tiktoken.get_encoding("cl100k_base")

logger = logging.getLogger(__name__)

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | Функции | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

logger.info("Init func")

# | -- | -- | -- | -- | -- | -- |  Подготовка данных | -- | -- | -- | -- | -- | -- |
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


# | -- | -- | -- | -- | -- | -- |  Чанкование | -- | -- | -- | -- | -- | -- |

def chunkinizer(
    question: str,
    answer: str, 
    chunk_size: int = 512,
    overlap_part: float = 0.2,
    enc: Any = enc
):
    """
    Разбивает текст на чанки по количеству токенов, используя tiktoken.

    Args:
        question: вопрос
        answer: ответ
        chunk_size: размер чанка в токенах.
        overlap_part: доля перекрытие в токенах между чанками.
        enc: токенизатор.

    Returns:
        Список чанков (строк).
    """
    # Количество токенов перекрытия
    overlap_tokens = int(chunk_size * overlap_part)
    # Токены вопроса и ответа
    tokens_answer = enc.encode(answer)
    tokens_question = enc.encode(question)
    # Количество токенов вопроса
    len_tokens_question = len(tokens_question)
    # Размера чанка, который заполняется ответом
    answer_chunk_size = chunk_size - len_tokens_question
        
    
    chunks = []

    start = 0

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

    return chunks


def vanila_chunkinizer(
    text: str,
    chunk_size: int = 512,
    overlap_part: float = 0.2,
    enc: Any = enc
):
    """
    Разбивает текст на чанки по количеству токенов, используя tiktoken.

    Args:
        text: исходный текст.
        chunk_size: размер чанка в токенах.
        overlap_part: доля перекрытие в токенах между чанками.
        enc: токенизатор.

    Returns:
        Список чанков (строк).
    """
    overlap_tokens = int(chunk_size * overlap_part)

    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

        start += chunk_size - overlap_tokens
        if end >= len(tokens):
            break

    return chunks

# | -- | -- | -- | -- | -- | -- |  Faiss | -- | -- | -- | -- | -- | -- |

# Function to build an HNSW index
def build_faiss_hnsw_index(dimension, ef_construction=200, M=32):
    """
    build_faiss_hnsw_index: Add a description here.

    Args:
        # List the arguments with types and descriptions.

    Returns:
        # Specify what the function returns.
    """
    """
    Builds a FAISS HNSW index for cosine similarity.

    Parameters:
        dimension (int): Dimensionality of the embeddings.
        ef_construction (int): Trade-off parameter between index construction speed and accuracy.
        M (int): Number of neighbors in the graph.

    Returns:
        index (faiss.IndexHNSWFlat): Initialized FAISS HNSW index.
    """
    index = faiss.IndexHNSWFlat(dimension, M)  # HNSW index
    index.hnsw.efConstruction = ef_construction  # Construction accuracy
    index.metric_type = faiss.METRIC_INNER_PRODUCT  # Cosine similarity via normalized vectors
    return index

# Function to populate the FAISS index
def populate_faiss_index(index: faiss.Index, documents: dict, batch_size: int=20):
    """
    populate_faiss_index: Add a description here.

    Args:
        # List the arguments with types and descriptions.

    Returns:
        # Specify what the function returns.
    """
    """
    Populates the FAISS HNSW index with normalized embeddings from the dataset.

    Parameters:
        index (faiss.Index): FAISS index to populate.
        documents (pd.Series): documents like List[list[str]]
        batch_size (int): Number of questions to process at a time.
    """
    buffer = []
    i = 0

    for _, embedding in documents.items():
        embedding = normalize_vector(embedding)
        buffer.append(embedding)
        i += 1

        # Add embeddings to the index in batches
        if len(buffer) >= batch_size:
            index.add(np.array(buffer, dtype=np.float32))
            buffer = []

    # Add remaining embeddings
    if buffer:
        index.add(np.array(buffer, dtype=np.float32))

# Function to perform a search query
def search_faiss_index(embeddings_storage, query, k=5):
    """
    search_faiss_index: Add a description here.

    Args:
        # List the arguments with types and descriptions.

    Returns:
        # Specify what the function returns.
    """
    """
    Searches the FAISS index for the closest matches to a query.

    Parameters:
        embeddings_storage (faiss.Index): FAISS index to search.
        query (str): Query string to search.
        k (int): Number of closest matches to retrieve.

    Returns:
        indices (np.ndarray): Indices of the top-k results.
        distances (np.ndarray): Distances of the top-k results.
    """
    # Preprocess and normalize the query embedding
    query_embedding = get_embedding(query)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    query_embedding = normalize_vector(query_embedding)
    # Search the embeddings_storage
    top_k_distances, top_k_indices = embeddings_storage.search(np.array([query_embedding], dtype=np.float32), k)

    # Match return format with that used in numpy storage search
    # Note that list manipulations will give an overhead
    top_k_indices_list = top_k_indices[0].tolist()
    top_k_distances_list = top_k_distances[0].tolist()

    return top_k_indices_list, top_k_distances_list

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

# | -- | -- | -- | -- | -- | -- |  BM25 | -- | -- | -- | -- | -- | -- |

# ---------------------------
# 1) Токенизация
# ---------------------------
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zа-я0-9]+", text)
    return tokens

# ---------------------------
# 2) Построение индекса
# docs — список строк (документов)
# ---------------------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in docs]
        self.N = len(self.docs)

        # длина документа
        self.doc_lengths = [len(doc) for doc in self.docs]
        self.avgDL = sum(self.doc_lengths) / self.N

        # TF-грамматика: list[Counter]
        self.tf = [Counter(doc) for doc in self.docs]

        # DF: количество документов, где встречается слово
        df = defaultdict(int)
        for doc in self.docs:
            for word in set(doc):
                df[word] += 1

        # IDF
        self.idf = {}
        for word, freq in df.items():
            # Okapi IDF с защитой от отрицательности
            self.idf[word] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    # ---------------------------
    # 3) Счёт BM25 для одного документа
    # ---------------------------
    def score(self, query, index):
        query_tokens = tokenize(query)
        score = 0.0
        doc_tf = self.tf[index]
        doc_len = self.doc_lengths[index]

        for term in query_tokens:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self.idf.get(term, 0)

            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgDL)
            score += idf * (tf * (self.k1 + 1)) / denom

        return score

    # ---------------------------
    # 4) Поиск top-k
    # ---------------------------
    def search(self, query, top_k=5):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    

def bm25_screach(query:str, bm25, BM25_TOPK:int = BM25_TOPK):
    bm25_results = bm25.search(query, top_k=BM25_TOPK)

    bm25_results = pd.DataFrame(
        bm25_results,
        columns=['top_k_indices_text', 'bm25_score']
        )
    
    return bm25_results

# | -- | -- | -- | -- | -- | -- |  Работа с моделями | -- | -- | -- | -- | -- | -- |

def get_embedding(text, dimensions=512):
    # Подключаемся к модели
    client = OpenAI(
        # Базовый url - сохранять без изменения
        base_url="https://ai-for-finance-hack.up.railway.app/",
        # Указываем наш ключ, полученный ранее
        api_key=EMBEDDER_API_KEY,
    )
    # Формируем запрос к клиенту
    response = client.embeddings.create(
        # Выбираем любую допступную модель из предоставленного списка
        model="text-embedding-3-small",
        # Отправяем запрос
        input=text, 
        # Определяем размерность эмбединга
        dimensions = dimensions
    )
    # Формируем ответ на запрос и возвращаем его в результате работы функции
    return response.data[0].embedding

def get_batch_embeddings(texts, batch_size=32, dimensions=512):
    """Батчевые запросы к embedding API

    Args:
        texts: список строк
        batch_size: количество текстов в одном запросе
        dimensions: размерность эмбединга
    Returns:
        Список эмбеддингов (list[list[float]]).
    """
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )

    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,            
            dimensions=dimensions
        )

        # каждая запись в response.data соответствует одному элементу из batch
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings

def z_logistic(s, k=1.0):
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    return expit(k * z)

def get_question_embeddings(user_query:str):

    # Эмбединг вопроса для текста
    example_1024 = get_embedding(text=user_query, dimensions=1024) 
    example_1024 = normalize_vector(example_1024)

    # Эмбединг вопроса для тега + аннотации
    example_512 = get_embedding(text=user_query, dimensions=512) 
    example_512 = normalize_vector(example_512)

    return example_1024, example_512

def get_similarities_bm25(storage):
    # Все наши документы (теги)
    docs = list()
    for _, val in storage.items():
        docs.append(val[1])

    bm25 = BM25(docs)

    bm25_results = bm25_screach(
        query = "В чем минусы и плюсы доверительного управления?",
        bm25=bm25
    )

    # Датасет в рамках которого будем крутить скоры
    bm25_results['doc_id'] = bm25_results['top_k_indices_text'].apply(lambda idx: storage_t[idx][0])
    # Номрализуем косинусную близость - чам больше - тем ближе, т.е. лучше
    bm25_results['bm25_score'] = z_logistic(bm25_results[['bm25_score']]) * BM25_KOEF

    return bm25_results


def get_similarities_datasets(query_512, query_1024, hnsw_index_t, hnsw_index_an_t, storage_t, storage_an_t, K=K):
    # Топ K вопросов по тексту
    top_k_similarities_text, top_k_indices_text = hnsw_index_t.search(np.array([query_1024], dtype=np.float32), K) # В БД Tags+annotation ищем пример

    # Ток K вопросам по аннотациям + теги 
    top_k_similarities_key_annot, top_k_indices_key_annot = hnsw_index_an_t.search(np.array([query_512], dtype=np.float32), K) # В БД Tags+annotation ищем пример

    # Датасет в рамках которого будем крутить скоры
    df_text_result = pd.DataFrame({
        'top_k_indices_text' : top_k_indices_text[0].T,
        'cos_scores' : top_k_similarities_text[0].T,
        'doc_id' : [storage_t[idx.item()][0] for idx in top_k_indices_text[0]]
        })
    # Номрализуем косинусную близость - чам больше - тем ближе, т.е. лучше
    df_text_result['cos_scores_m'] = z_logistic(df_text_result[['cos_scores']]) 
    # Взвешиваем скор
    df_text_result['cos_scores_m'] = df_text_result['cos_scores_m'] * EMBED_T_KOEF

    ## tags_annot
    df_tags_annot_result = pd.DataFrame({
        'top_k_indices_key_annot' : top_k_indices_key_annot[0].T,
        'cos_scores_ka' : top_k_similarities_key_annot[0].T,
        'doc_id' : [storage_an_t[idx.item()][0] for idx in top_k_indices_key_annot[0]]
        })
    # Могут быть дубли, на 1 документ 2 чанка - возьмем наиближайшего соседа
    df_tags_annot_result_agg = df_tags_annot_result.groupby('doc_id')['cos_scores_ka'].min().reset_index()
    # Номрализуем косинусную близость - чам больше - тем ближе, т.е. лучше
    df_tags_annot_result_agg['cos_scores_ka_m'] = z_logistic(df_tags_annot_result_agg['cos_scores_ka'])
    # Взвешиваем скор
    df_tags_annot_result_agg['cos_scores_ka_m'] = df_tags_annot_result_agg['cos_scores_ka_m'] * EMBED_KA_KOEF

    return df_text_result, df_tags_annot_result_agg

def prepare_rag_text(df_text_result, df_tags_annot_result_agg, bm25_results, data, SK=SK):
    # Холдер для аннотации
    annotation_doc_id = ''

    # Соберем единый датасет
    df_for_sort = df_text_result.merge(
        df_tags_annot_result_agg, 
        on='doc_id', 
        how='outer'
        ).merge(
        bm25_results, 
        on=['top_k_indices_text', 'doc_id'], 
        how='outer'        
        )

    # Заполним пропуски в скорах
    df_for_sort['cos_scores_m'] = df_for_sort['cos_scores_m'].fillna(0)
    df_for_sort['cos_scores_ka_m'] = df_for_sort['cos_scores_ka_m'].fillna(0)
    df_for_sort['bm25_score'] = df_for_sort['bm25_score'].fillna(0)

    # Единый скор и сортировка, оставляем топ SK чанков
    df_for_sort['result_score'] = df_for_sort['cos_scores_m'] + df_for_sort['cos_scores_ka_m'] + df_for_sort['bm25_score'] 
    df_for_sort = df_for_sort.sort_values('result_score', ascending=False)
    target_text_chunk = df_for_sort.reset_index(drop=True).loc[:SK-1, ['top_k_indices_text', 'doc_id']]

    # NEW - индексы для чанков по тексту
    target_text_chunk_notna = target_text_chunk[~target_text_chunk['top_k_indices_text'].isna()]

    # # Собираем чанки текста из storage_t
    rag_message = [storage_t[key][1] for key in target_text_chunk_notna['top_k_indices_text']]
    rag_message = '\n'.join(rag_message)

    # # Если есть doc_id без чанка, добавим анотацию doc_id 
    if (target_text_chunk['top_k_indices_text'].isna()).any(): 
        # Получаем doc_id у пропущенного значения 
        annotation_doc_id = target_text_chunk[
            target_text_chunk['top_k_indices_text'].isna()
            ].reset_index()\
            .loc[0, 'doc_id'] 
        
        annotation_missed_chunk = data[data['id']==annotation_doc_id]['annotation'].values[0] # Аннотация пропущенного id
        rag_message += f"\nАннотация: {annotation_missed_chunk}"

    # Добавим анатацию самого сонаправленного doc_id
    if annotation_doc_id != target_text_chunk.loc[0, 'doc_id']:
        annotation_doc_id = target_text_chunk.loc[0, 'doc_id']
        annotation_top1 = data[data['id']==annotation_doc_id]['annotation'].values[0] # Аннотация пропущенного id
        rag_message += f"\nАннотация: {annotation_top1}"

    rag_text_for_llm = f'\n**Для ответа используй следующие знания из RAG базы данных**:\n{rag_message}'

    return rag_text_for_llm

def rag_screach(user_query, hnsw_index_t, hnsw_index_an_t, storage_t, storage_an_t, data):

    query_1024, query_512 = get_question_embeddings(user_query)

    bm25_results = get_similarities_bm25(storage=storage_t)
    df_text_result, df_tags_annot_result_agg = get_similarities_datasets(
        query_512=query_512, 
        query_1024=query_1024, 
        hnsw_index_t=hnsw_index_t, 
        hnsw_index_an_t=hnsw_index_an_t, 
        storage_t=storage_t, 
        storage_an_t=storage_an_t
    )

    rag_text_for_llm = prepare_rag_text(
        df_text_result=df_text_result, 
        df_tags_annot_result_agg=df_tags_annot_result_agg, 
        bm25_results=bm25_results,
        data=data
        )

    question = user_query + rag_text_for_llm

    return question

def answer_generation(question):
    # Подключаемся к модели
    client = OpenAI(
        # Базовый url - сохранять без изменения
        base_url="https://ai-for-finance-hack.up.railway.app/",
        # Указываем наш ключ, полученный ранее
        api_key=LLM_API_KEY,
    )

    system_prompt = """Ты RAG-ассистент, вежливый помошник по банковским, финансовым и прочим вопросам. 
    Пользователь задает тебе вопрос с указанием ответить на него (Ответь на вопрос:). 
    Так же в сообщении пользователя будет указана информация, которую ты должен использовать для ответа (RAG базы данных).
    Отвечай открыто и интересно, по возможности приводи примеры, основанные на знаниях из RAG базы данных!
    Начинай с приветсвия пользователя!
    Не повторяй вопрос пользователя!"""

    # Формируем запрос к клиенту
    response = client.chat.completions.create(
        # Выбираем любую допступную модель из предоставленного списка
        model="openrouter/google/gemma-3-27b-it",
        # Формируем сообщение
        messages=[
            
                {"role": "system", "content": system_prompt},
                {"role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": f"Ответь на вопрос: {question}"
                    }
                ]}
        ]
    )
    # Формируем ответ на запрос и возвращаем его в результате работы функции
    return response.choices[0].message.content



# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 2. Подготовка данных | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

logger.info("[Prepare data] Start...")

# Загружаем не обработанный датасет
raw_data = pd.read_csv('train_data.csv')

# Подготовка данных
data = preprocess(raw_data)

logger.info("[Prepare data] End!")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 3. Чанки | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

logger.info("[Chunk] Start...")

# Собираем чанки для тегов и аннотации
chunks_ta = [
    vanila_chunkinizer(
            f"{tags}. {annototions}", 
            chunk_size=TAGS_ANNOTATIONS_CHUNK_SIZE, 
            overlap_part=TAGS_ANNOTATIONS_OVERLAP
            ) 
    for annototions, tags in zip(data['annotation'], data['tags'])
    ]

data['annotation_tags_chunk'] = chunks_ta

# Собираем чанки для текста
chunks_t = list()

for row in data['text']:
    doc_chunks = list()
    for doc in row:
        question, answer = doc[0], doc[1]
        doc_chunks.append(chunkinizer(
            question=question, 
            answer=answer, 
            chunk_size=TEXT_CHUNK_SIZE, 
            overlap_part=TEXT_OVERLAP)
            )

    chunks_t.append(doc_chunks)

data['text_chunk'] = chunks_t

logger.info("[Chunk] End")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 4. Подготовка json БД | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

logger.info("[JSON DB] Start...")

# Создаем JSON  DB в формате [Сквозной идентификатор : (doc_id, chunk)] для тегов и ннотаций
storage_an_t = data_to_storage(
        id_series = data['id'],
        data_series = data['annotation_tags_chunk']
    )

# Создаем JSON  DB в формате [Сквозной идентификатор : (doc_id, chunk)] для текстовых чанков
storage_t = data_to_storage(
        id_series = data['id'],
        data_series = data['text_chunk']
)

logger.info("[JSON DB] End")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 5. Получаем эмбединги | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

logger.info("[JSON DB] Get embeddings...")

logger.info("[JSON DB]     1. Annotations + tags")

### Для аннотаций
# Соберем текст для батчевого запроса
texts = list()
for _, val in storage_an_t.items():
    texts.append(val[1])

# Запрос к openrouter
embeddings = get_batch_embeddings(texts)

# Соберем временное хранилище векторов
embed_storage_an_t = dict()

for i in range(len(embeddings)):
    embed_storage_an_t[i] = np.array(embeddings[i], np.float32)

logger.info("[JSON DB]     2. Text")
### Для текста
# Соберем текст для батчевого запроса
texts = list()
for _, val in storage_t.items():
    texts.append(val[1])

# Запрос к openrouter
embeddings = get_batch_embeddings(texts, dimensions=1024)

# Соберем временное хранилище векторов
embed_storage_t = dict()

for i in range(len(embeddings)):
    embed_storage_t[i] = np.array(embeddings[i], np.float32)

# with open('data/embed_storage_t_1024.pickle', 'rb') as f:
#     embed_storage_t = pickle.load(f)

logger.info("[JSON DB] End")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 6. Faiss | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

logger.info("[Faiss] Start...")

# Define the dimensions of the embedding vectors
embedding_dimension = 512  # Depends on the FastText model
# Build the HNSW index
hnsw_index_an_t = build_faiss_hnsw_index(embedding_dimension)
# Populate the index from pd.Series
populate_faiss_index(index=hnsw_index_an_t, documents=embed_storage_an_t)

# Define the dimensions of the embedding vectors
embedding_dimension = 1024  # Depends on the FastText model
# Build the HNSW index
hnsw_index_t = build_faiss_hnsw_index(embedding_dimension)
# Populate the index from pd.Series
populate_faiss_index(index=hnsw_index_t, documents=embed_storage_t)

logger.info("[Faiss] End")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 6. Call LLM | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

if __name__ == "__main__":    
    logger.info("[__main__] Start...")
    # Считываем список вопросов
    questions = pd.read_csv('./questions.csv')
    # Выделяем список вопросов
    questions_list = questions['Вопрос'].tolist()
    # Создаем список для хранения ответов
    answer_list = []
    # Проходимся по списку вопросов
    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        logger.info(f"[__main__] current_question: {current_question}")
        prepared_question = rag_screach(
            user_query=current_question, 
            hnsw_index_t=hnsw_index_t, 
            hnsw_index_an_t=hnsw_index_an_t, 
            storage_t=storage_t, 
            storage_an_t=storage_an_t, 
            data=data
            )
        logger.info(f"[__main__] prepared_question: {prepared_question}")
        # Отправляем запрос на генерацию ответа
        answer = answer_generation(question=prepared_question)
        logger.info(f"[__main__] answer: {answer}")
        # Добавляем ответ в список
        answer_list.append(answer)
    # Добавляем в данные список ответов
    questions['Ответы на вопрос'] = answer_list
    # Сохраняем submission
    questions.to_csv('submission.csv', index=False)