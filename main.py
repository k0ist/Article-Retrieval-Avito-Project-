import pandas as pd
import numpy as np
import faiss
import nltk
import re
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.stem.snowball import SnowballStemmer


articles = pd.read_feather('articles.f')
calibration = pd.read_feather('calibration.f')
test = pd.read_feather('test.f')

nltk.download('punkt', quiet=True)
stemmer = SnowballStemmer("russian")


def preproccessing(html):
    if not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_corpus(articles):
    articles["clean_body"] = articles["body"].apply(preproccessing)
    articles["bm25_text"] = (articles["title"].fillna("") + " ") * 3 + articles["clean_body"]

    def truncate_for_nn(title, body):
        words = str(body).split()[:150]
        return f"{title} {' '.join(words)}"

    articles["nn_text"] = articles.apply(lambda row: truncate_for_nn(row["title"], row["clean_body"]), axis=1)
    return articles

def tokenize_stem(text):
    tokens = re.findall(r'\w+', text.lower())
    return [stemmer.stem(w) for w in tokens]

def rrf_fusion(rankings, k=60, top_k=10):
    fused_scores = {}
    for ranking in rankings:
        for rank, (article_id, _) in enumerate(ranking):
            fused_scores[article_id] = fused_scores.get(article_id, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [article_id for article_id, _ in ranked[:top_k]]


def eval_map_10(calibration, bm25_retriever, embedding_retriever, reranker, id_to_text):
    queries = calibration["query_text"].tolist()
    emb_results_batch = embedding_retriever.search_batch(queries, top_k=50)

    aps = []
    for idx, (i, row) in enumerate(calibration.iterrows()):
        q = row["query_text"]

        bm25_res = bm25_retriever.search(q, top_k=50)
        emb_res = emb_results_batch[idx]

        candidates_ids = rrf_fusion([bm25_res, emb_res], top_k=50)
        pairs = [[q, id_to_text[doc_id]] for doc_id in candidates_ids]
        scores = reranker.predict(pairs, show_progress_bar=False)

        scored_candidates = sorted(zip(candidates_ids, scores), key=lambda x: x[1], reverse=True)
        final_top_10 = [str(doc_id) for doc_id, _ in scored_candidates[:10]]

        relevant = set(row["ground_truth"].split())

        hits = 0
        score = 0.0
        for rank, p in enumerate(final_top_10, start=1):
            if p in relevant:
                hits += 1
                score += hits / rank

        aps.append(score / min(len(relevant), 10) if relevant else 0.0)

    return float(np.mean(aps))

