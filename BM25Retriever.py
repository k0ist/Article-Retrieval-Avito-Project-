import numpy as np
from main import tokenize_stem
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, articles):
        self.article_ids = articles["article_id"].tolist()
        tokenized_corpus = [tokenize_stem(t) for t in articles["bm25_text"]]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k=50):
        scores = self.bm25.get_scores(tokenize_stem(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.article_ids[i], scores[i]) for i in top_idx]