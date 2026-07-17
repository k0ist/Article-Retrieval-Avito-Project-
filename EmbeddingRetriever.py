import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


class EmbeddingRetriever:
    def __init__(self, articles, model_name="intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name)
        self.article_ids = np.array(articles["article_id"].tolist())
        passages = ["passage: " + t for t in articles["nn_text"]]
        embeddings = self.model.encode(passages, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype="float32"))

    def search_batch(self, queries, top_k=50):
        formatted_queries = ["query: " + q for q in queries]
        q_embs = self.model.encode(formatted_queries, batch_size=32, normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_embs, dtype="float32"), top_k)

        batch_results = []
        for i in range(len(queries)):
            res = [(self.article_ids[idx], scores[i][j]) for j, idx in enumerate(indices[i])]
            batch_results.append(res)
        return batch_results