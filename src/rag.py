from src.models import reranker, get_question_embeddings
from src.bm25 import get_similarities_bm25
from src.utils import z_logistic
from src.const import EMBED_T_KOEF, EMBED_KA_KOEF, FAISS_TEXT_K, FAISS_ANNOTATION_K, TOP_K

import pandas as pd
import numpy as np


def get_similarities_datasets(
        query_to_annotations, query_to_text, hnsw_index_t, hnsw_index_an_t, storage_t, storage_an_t, FAISS_TEXT_K=FAISS_TEXT_K, FAISS_ANNOTATION_K=FAISS_ANNOTATION_K
        ):
    # Топ K вопросов по тексту
    top_k_similarities_text, top_k_indices_text = hnsw_index_t.search(np.array([query_to_text], dtype=np.float32), FAISS_TEXT_K) # В БД Tags+annotation ищем пример

    # Ток K вопросам по аннотациям + теги 
    top_k_similarities_key_annot, top_k_indices_key_annot = hnsw_index_an_t.search(np.array([query_to_annotations], dtype=np.float32), FAISS_ANNOTATION_K) # В БД Tags+annotation ищем пример

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

def prepare_rag_text(df_text_result, df_tags_annot_result_agg, bm25_results, data, storage_t, TOP_K=TOP_K):
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
    target_text_chunk = df_for_sort.reset_index(drop=True).loc[:TOP_K-1, ['top_k_indices_text', 'doc_id']]

    # NEW - индексы для чанков по тексту
    target_text_chunk_notna = target_text_chunk[~target_text_chunk['top_k_indices_text'].isna()]

    # # Собираем чанки текста из storage_t
    rag_message = [storage_t[key][1] for key in target_text_chunk_notna['top_k_indices_text']]

    # # Если есть doc_id без чанка, добавим анотацию doc_id 
    if (target_text_chunk['top_k_indices_text'].isna()).any(): 
        # Получаем doc_id у пропущенного значения 
        annotation_doc_id = target_text_chunk[
            target_text_chunk['top_k_indices_text'].isna()
            ].reset_index()\
            .loc[0, 'doc_id'] 
        
        annotation_missed_chunk = data[data['id']==annotation_doc_id]['annotation'].values[0] # Аннотация пропущенного id
        rag_message.append(f"\nАннотация: {annotation_missed_chunk}")

    # Добавим анатацию самого сонаправленного doc_id
    if annotation_doc_id != target_text_chunk.loc[0, 'doc_id']:
        annotation_doc_id = target_text_chunk.loc[0, 'doc_id']
        annotation_top1 = data[data['id']==annotation_doc_id]['annotation'].values[0] # Аннотация пропущенного id
        rag_message.append(f"\nАннотация: {annotation_top1}")

    return rag_message


def rag_screach(user_query, hnsw_index_t, hnsw_index_an_t, storage_t, storage_an_t, data):

    query_to_text, query_to_annotations = get_question_embeddings(user_query)

    bm25_results = get_similarities_bm25(storage=storage_t)
    df_text_result, df_tags_annot_result_agg = get_similarities_datasets(
        query_to_annotations=query_to_annotations, 
        query_to_text=query_to_text, 
        hnsw_index_t=hnsw_index_t, 
        hnsw_index_an_t=hnsw_index_an_t, 
        storage_t=storage_t, 
        storage_an_t=storage_an_t
    )

    message_for_rerank = prepare_rag_text(
        df_text_result=df_text_result, 
        df_tags_annot_result_agg=df_tags_annot_result_agg, 
        bm25_results=bm25_results,
        data=data, 
        storage_t=storage_t
        )
    
    rag_text_for_llm = reranker(user_query, message_for_rerank=message_for_rerank)

    question = user_query + rag_text_for_llm

    return question

