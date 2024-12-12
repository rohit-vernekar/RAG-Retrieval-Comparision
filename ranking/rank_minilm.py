import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

model = SentenceTransformer("all-MiniLM-L6-v2")


def rank_passages_minilm_cosine(query: str, passages: List[str]) -> List[int]:
    """Rank passages by cosine similarity with the query using MiniLM embeddings."""
    # Generate embeddings
    query_embedding = model.encode([query])
    passage_embeddings = model.encode(passages)

    # Compute cosine similarity between query and all passages
    similarity_scores = cosine_similarity(query_embedding, passage_embeddings).flatten()

    # Rank passages by similarity score
    ranked_indices = np.argsort(similarity_scores)[::-1]
    return ranked_indices.tolist()


def rank_passages_minilm_knn(query: str, passages: List[str], k: int = 5) -> List[int]:
    """Rank passages by k-nearest neighbors similarity with the query using MiniLM embeddings."""
    # Generate embeddings
    query_embedding = model.encode([query])
    passage_embeddings = model.encode(passages)

    # Fit kNN model
    knn = NearestNeighbors(n_neighbors=min(k, len(passages)), metric="cosine")
    knn.fit(passage_embeddings)

    # Find k-nearest neighbors for the query
    distances, indices = knn.kneighbors(query_embedding)
    return indices.flatten().tolist()


def rank_passages_minilm_bm25(query: str, passages: List[str]) -> List[int]:
    """Rank passages by BM25 similarity with the query using tokenized MiniLM embeddings."""
    # Tokenize passages
    tokenized_passages = [word_tokenize(passage.lower()) for passage in passages]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_passages)

    # Rank passages by BM25 scores
    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices.tolist()
