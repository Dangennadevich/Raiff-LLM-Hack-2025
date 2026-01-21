from collections import Counter, defaultdict
from src.const import BM25_TOPK, BM25_KOEF
from src.utils import z_logistic
import pandas as pd

import math
import re

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zа-я0-9]+", text)
    return tokens

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in docs] # список строк (документов)
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

    def score(self, query, index):
        'Счёт BM25 для одного документа'
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

    def search(self, query, top_k=5):
        '''Поиск top-k'''
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    

# def bm25_screach(query:str, bm25, bm25_topk:int = BM25_TOPK):
#     bm25_results = bm25.search(query, top_k=bm25_topk)

#     bm25_results = pd.DataFrame(
#         bm25_results,
#         columns=['top_k_indices_text', 'bm25_score']
#         )
    
#     return bm25_results



# def get_similarities_bm25(storage):
#     # Все наши документы (теги)
#     docs = list()
#     for _, val in storage.items():
#         docs.append(val[1])

#     bm25 = BM25(docs)

#     bm25_results = bm25_screach(
#         query = "В чем минусы и плюсы доверительного управления?",
#         bm25=bm25
#     )

#     # Датасет в рамках которого будем крутить скоры
#     bm25_results['doc_id'] = bm25_results['top_k_indices_text'].apply(lambda idx: storage[idx][0])
#     # Номрализуем косинусную близость - чам больше - тем ближе, т.е. лучше
#     bm25_results['bm25_score'] = z_logistic(bm25_results[['bm25_score']]) * BM25_KOEF

#     return bm25_results