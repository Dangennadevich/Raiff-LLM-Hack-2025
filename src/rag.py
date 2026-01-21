# from src.models import reranker, get_question_embeddings
from src.reranker import reranker
from src.models import get_embedding
from src.utils import z_logistic, normalize_vector
from src.const import (
    EMBED_T_KOEF, EMBED_KA_KOEF, FAISS_TEXT_K, FAISS_ANNOTATION_K, TOP_K, FAISS_TEXT_K, FAISS_ANNOTATION_K, BM25_TOPK, BM25_WEIGHT, FAISS_TEXT, FAISS_TAGS_ANNOT
    )
from src.storage import (
    hnsw_index_an_t, hnsw_index_t, bm25, storage_t, storage_an_t, data
    )

import pandas as pd
import numpy as np
from typing import Tuple

def rag_search(user_query:str, hnsw_index_an_t, hnsw_index_t, bm25) -> Tuple:
    '''RAG поиск документов в FAISS (Косинусная близость) и BM25 (Статистическое сходство)'''
    ####### FAISS (Косинусная близость)
    # Эмбединг вопроса
    embed_user_question = get_embedding(text=user_query) 
    embed_user_question = normalize_vector(embed_user_question)

    # Топ K чанков по тексту
    top_k_similarities_text, top_k_indices_text = hnsw_index_t.search(
        np.array([embed_user_question], dtype=np.float32), 
        FAISS_TEXT_K
        )

    text_sim = dict()
    for idx, sim in zip(top_k_indices_text[0], top_k_similarities_text[0]):
        text_sim[idx.item()] = sim.item()

    # Ток K чанков по аннотациям + теги 
    top_k_similarities_key_annot, top_k_indices_key_annot = hnsw_index_an_t.search(
        np.array([embed_user_question], dtype=np.float32), 
        FAISS_ANNOTATION_K
        )

    tags_annot_sim = dict()
    for idx, sim in zip(top_k_indices_key_annot[0], top_k_similarities_key_annot[0]):
        tags_annot_sim[idx.item()] = sim.item()

    ####### BM25 (Статистическое сходство)
    bm25_results = bm25.search(user_query, top_k=BM25_TOPK)

    bm25_results = pd.DataFrame(
        bm25_results,
        columns=['top_k_indices_text', 'bm25_score']
        )
    
    return bm25_results, top_k_similarities_text, top_k_indices_text, top_k_similarities_key_annot, top_k_indices_key_annot


def rag_prepare_txt2llm(
        bm25_results,
        top_k_similarities_text,
        top_k_indices_text,
        top_k_similarities_key_annot,
        top_k_indices_key_annot,
        storage_t,
        storage_an_t,
        data_chunks
    ):
    '''Формирование чанков текста из трех источников для последующего ранжирования'''
    ####### Подготавливаем идеентичные датасеты из трех источников

    #### 1. BM25
    # Датасет в рамках которого будем крутить скоры
    bm25_results['doc_id'] = bm25_results['top_k_indices_text'].apply(lambda idx: storage_t[idx][0])
    # Номрализуем косинусную близость - чам больше - тем ближе, т.е. лучше
    bm25_results['bm25_score'] = z_logistic(bm25_results[['bm25_score']]) * BM25_WEIGHT

    #### 2. Faiss Текст 
    # Собираем doc_id
    df_text_result = pd.DataFrame({
        'top_k_indices_text' : top_k_indices_text[0].T,
        'cos_scores' : top_k_similarities_text[0].T,
        'doc_id' : [storage_t[idx.item()][0] for idx in top_k_indices_text[0]]
        })
    # Номрализуем косинусную близость - чам больше - тем ближе, т.е. лучше
    df_text_result['cos_scores_m'] = z_logistic(df_text_result[['cos_scores']]) 
    # Взвешиваем скор
    df_text_result['cos_scores_m'] = df_text_result['cos_scores_m'] * FAISS_TEXT

    #### 3. Faiss Аннотация и теги 
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
    df_tags_annot_result_agg['cos_scores_ka_m'] = df_tags_annot_result_agg['cos_scores_ka_m'] * FAISS_TAGS_ANNOT

    ####### Подготовка чанков
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

    # Заполним пропуски в скорах, если чанк не оказался релевантным
    df_for_sort['cos_scores_m'] = df_for_sort['cos_scores_m'].fillna(0)
    df_for_sort['cos_scores_ka_m'] = df_for_sort['cos_scores_ka_m'].fillna(0)
    df_for_sort['bm25_score'] = df_for_sort['bm25_score'].fillna(0)

    # Единый скор и сортировка, оставляем топ SK чанков
    df_for_sort['result_score'] = df_for_sort['cos_scores_m'] + df_for_sort['cos_scores_ka_m'] + df_for_sort['bm25_score'] 
    df_for_sort = df_for_sort.sort_values('result_score', ascending=False)
    target_text_chunk = df_for_sort.reset_index(drop=True).loc[:TOP_K-1, ['top_k_indices_text', 'doc_id']]

    # Индексы для чанков по тексту
    target_text_chunk_notna = target_text_chunk[~target_text_chunk['top_k_indices_text'].isna()]

    # # Собираем чанки текста из storage_t
    message_for_rerank = [storage_t[key][1] for key in target_text_chunk_notna['top_k_indices_text']]

    # Если есть doc_id без чанка, добавим анотацию doc_id 
    if (target_text_chunk['top_k_indices_text'].isna()).any(): 
        # Получаем doc_id у пропущенного значения 
        annotation_doc_id = target_text_chunk[
            target_text_chunk['top_k_indices_text'].isna()
            ].reset_index()\
            .loc[0, 'doc_id'] 
        
        annotation_missed_chunk = data_chunks[data_chunks['id']==annotation_doc_id]['annotation'].values[0] # Аннотация пропущенного id
        message_for_rerank.append(f"\nАннотация: {annotation_missed_chunk}")

    # Добавим анатацию самого сонаправленного doc_id
    if annotation_doc_id != target_text_chunk.loc[0, 'doc_id']:
        annotation_doc_id = target_text_chunk.loc[0, 'doc_id']
        annotation_top1 = data_chunks[data_chunks['id']==annotation_doc_id]['annotation'].values[0] # Аннотация пропущенного id
        message_for_rerank.append(f"\nАннотация: {annotation_top1}")

    return message_for_rerank

def rag(user_query,
        hnsw_index_an_t=hnsw_index_an_t,
        hnsw_index_t=hnsw_index_t,
        bm25=bm25,
        storage_t=storage_t,
        storage_an_t=storage_an_t,
        data_chunks=data
        ):

        # Собираем голосование релевантности из 3х источников
        bm25_results, top_k_similarities_text, top_k_indices_text,\
                top_k_similarities_key_annot, top_k_indices_key_annot = rag_search(user_query, hnsw_index_an_t, hnsw_index_t, bm25)

        # Подготоавливаем чанки текста для рандирования
        message_for_rerank = rag_prepare_txt2llm(
                bm25_results=bm25_results, 
                top_k_similarities_text=top_k_similarities_text, 
                top_k_indices_text=top_k_indices_text, 
                top_k_similarities_key_annot=top_k_similarities_key_annot, 
                top_k_indices_key_annot=top_k_indices_key_annot,
                storage_t=storage_t,
                storage_an_t=storage_an_t,
                data_chunks=data_chunks
        )
        # Получаем RAG контекст (с топ релевантными чанками)
        rag_answer = reranker(user_query, message_for_rerank)

        return rag_answer