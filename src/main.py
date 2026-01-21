

# from dotenv import load_dotenv
# from openai import OpenAI
from src.models import get_embedding, get_answer_with_retries

from tqdm import tqdm

from src.utils import data_to_storage, preprocess
from src.chunkinizer import chunkinizer
from src.faiss import build_faiss_hnsw_index, populate_faiss_index
from src.rag import rag_screach
from src.const import (
    TAGS_ANNOTATIONS_CHUNK_SIZE, 
    TAGS_ANNOTATIONS_OVERLAP, 
    TEXT_CHUNK_SIZE, 
    TEXT_OVERLAP, 
    ANNOTATION_EMBEDDINGS_DIMENSIONS, 
    TEXT_EMBEDDINGS_DIMENSIONS
    )

import pandas as pd
import numpy as np
import logging
import pickle
logging.basicConfig(
    level=logging.INFO,                 # или DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # вывод в консоль
)

logger = logging.getLogger(__name__)


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
    chunkinizer(
            vanila_chunkinizer=1,
            text = f"{tags}. {annototions}", 
            chunk_size=TAGS_ANNOTATIONS_CHUNK_SIZE, 
            overlap_part=TAGS_ANNOTATIONS_OVERLAP,
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
            vanila_chunkinizer=0,
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
# embeddings = get_batch_embeddings(texts, dimensions=ANNOTATION_EMBEDDINGS_DIMENSIONS)
# # Соберем временное хранилище векторов
# embed_storage_an_t = dict()
# for i in range(len(embeddings)):
#     embed_storage_an_t[i] = np.array(embeddings[i], np.float32)
with open('data/embed_storage_an_t_final.pickle', 'rb') as f:
    embed_storage_an_t = pickle.load(f)


logger.info("[JSON DB]     2. Text")
### Для текста
# Соберем текст для батчевого запроса
texts = list()
for _, val in storage_t.items():
    texts.append(val[1])

# # Запрос к openrouter
    
### OLD
# embeddings = get_batch_embeddings(texts, dimensions=TEXT_EMBEDDINGS_DIMENSIONS)
### OLD

### New
batch_size = 32
embeddings_t = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    response = get_embedding(text=texts, batch_embeddings=True) # # Запрос к openrouter

    batch_embeddings = [item.embedding for item in response.data]
    embeddings_t.extend(batch_embeddings)
### New
    
# Соберем временное хранилище векторов
embed_storage_t = dict()
for i in range(len(embeddings_t)):
    embed_storage_t[i] = np.array(embeddings_t[i], np.float32)

with open('data/embed_storage_t_final.pickle', 'rb') as f:
    embed_storage_t = pickle.load(f)

logger.info("[JSON DB] End")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 6. Faiss | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

logger.info("[Faiss] Start...")

# Define the dimensions of the embedding vectors
embedding_dimension = ANNOTATION_EMBEDDINGS_DIMENSIONS  # Depends on the FastText model ANNOTATION_EMBEDDINGS_DIMENSIONS = 512
# Build the HNSW index
hnsw_index_an_t = build_faiss_hnsw_index(embedding_dimension)
# Populate the index from pd.Series
populate_faiss_index(index=hnsw_index_an_t, documents=embed_storage_an_t)

# Define the dimensions of the embedding vectors
embedding_dimension = TEXT_EMBEDDINGS_DIMENSIONS  # Depends on the FastText model
# Build the HNSW index
hnsw_index_t = build_faiss_hnsw_index(embedding_dimension)
# Populate the index from pd.Series
populate_faiss_index(index=hnsw_index_t, documents=embed_storage_t)

logger.info("[Faiss] End")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 6. Call LLM | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat mode")
    args = parser.parse_args()

    logger.info("[__main__] Start...")

    if args.interactive:
        # === ИНТЕРАКТИВНЫЙ РЕЖИМ ===
        print("⚡ Запущен интерактивный режим. Введите вопрос.")
        print("Введите `exit` чтобы выйти.\n")

        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Выход.")
                break

            # Router if easy question: rag_screach (classic), else decomposition

            prepared_question = rag_screach(
                user_query=user_input,
                hnsw_index_t=hnsw_index_t,
                hnsw_index_an_t=hnsw_index_an_t,
                storage_t=storage_t,
                storage_an_t=storage_an_t,
                data=data
            )

            answer = get_answer_with_retries(question=prepared_question)
            print(f"\nОтвет:\n{answer}\n")

    else:
        # === ПАКЕТНАЯ ОБРАБОТКА CSV ===
        questions = pd.read_csv('./questions.csv')
        questions_list = questions['Вопрос'].tolist()
        answer_list = []

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

            answer = get_answer_with_retries(question=prepared_question)
            logger.info(f"[__main__] answer: {answer}")

            answer_list.append(answer)

        questions['Ответы на вопрос'] = answer_list
        questions.to_csv('submission_v3_np.csv', index=False)
        print("✅ Результат сохранён в submission_v3_np.csv")