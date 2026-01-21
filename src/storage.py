from src.faiss import build_faiss_hnsw_index, populate_faiss_index
from src.utils import data_to_storage, preprocess
from src.models import get_embedding
from src.chunkinizer import chunkinizer
from src.bm25 import BM25
from pathlib import Path
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


# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 1. Подготовка данных | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |
logger.info("[Prepare data] Start...")

# Загружаем не обработанный датасет
raw_data = pd.read_csv('train_data.csv')
# Подготовка данных
data = preprocess(raw_data)

logger.info("[Prepare data] End!")

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 2. Чанки | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |
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

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 3. Подготовка json БД | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |
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

# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 4. Получаем эмбединги | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |
logger.info("[JSON DB] Get embeddings...")
logger.info("[JSON DB]     1. Annotations + tags")

### Для аннотаций
# Соберем текст для батчевого запроса
texts = list()
for _, val in storage_an_t.items():
    texts.append(val[1])


path_storage = Path("data/embed_storage_an_t-embedding-3-small.pickle'")
path_embedings = Path("data/embeddings_an_t-embedding-3-small.pickle")

# Нам нужен embed_storage_an_t, который создается на базе embeddings_an_t
if path_storage.exists():
    logger.info("[JSON DB]        storage ready, load")
    with path_storage.open("rb") as f:
        embed_storage_an_t = pickle.load(f)

elif path_embedings.exists():
    logger.info("[JSON DB]        embeddings ready, create storage, load")
    with path_embedings.open("rb") as f:
        embeddings_an_t = pickle.load(f)

    embed_storage_an_t = dict()
    for i in range(len(embeddings_an_t)):
        embed_storage_an_t[i] = np.array(embeddings_an_t[i], np.float32)
    with path_storage.open("wb")as f:
        pickle.dump(embed_storage_an_t, f)
else:
    logger.info("[JSON DB]        create embeddings and storage, load")
    responce = get_embedding(text=texts, batch_embeddings=True)
    embeddings_an_t = [item.embedding for item in responce.data]
    with path_embedings.open("wb") as f:
        pickle.dump(embeddings_an_t, f)

    embed_storage_an_t = dict()
    for i in range(len(embeddings_an_t)):
        embed_storage_an_t[i] = np.array(embeddings_an_t[i], np.float32)
    with path_storage.open("wb")as f:
        pickle.dump(embed_storage_an_t, f)


logger.info("[JSON DB]     2. Text")
### Для текста
# Соберем текст для батчевого запроса
texts = list()
for _, val in storage_t.items():
    texts.append(val[1])

path_storage = Path("data/embed_storage_t-embedding-3-small.pickle")
path_embedings = Path("data/embeddings_t-embedding-3-small.pickle")

# Нам нужен embed_storage_t, который создается на базе embeddings_t
if path_storage.exists():
    logger.info("[JSON DB]        storage ready, load")
    with path_storage.open("rb") as f:
        embed_storage_t = pickle.load(f)

elif path_embedings.exists():
    logger.info("[JSON DB]        embeddings ready, create storage, load")
    with path_embedings.open("rb") as f:
        embeddings_t = pickle.load(f)

    embed_storage_t = dict()
    for i in range(len(embeddings_t)):
        embed_storage_t[i] = np.array(embeddings_t[i], np.float32)
    with path_storage.open("wb")as f:
        pickle.dump(embed_storage_t, f)
else:
    logger.info("[JSON DB]        create embeddings and storage, load")
    batch_size = 32
    embeddings_t = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = get_embedding(text=batch, batch_embeddings=True) # Запрос к openrouter
        batch_embeddings = [item.embedding for item in response.data]
        embeddings_t.extend(batch_embeddings)
    with path_embedings.open("wb") as f:
        pickle.dump(embeddings_t, f)

    embed_storage_t = dict()
    for i in range(len(embeddings_t)):
        embed_storage_t[i] = np.array(embeddings_t[i], np.float32)
    with path_storage.open("wb")as f:
        pickle.dump(embed_storage_t, f)


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


# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 7. BM25 | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

# Все наши документы (теги)
docs = list()
for _, val in storage_t.items():
    docs.append(val[1])

bm25 = BM25(docs)