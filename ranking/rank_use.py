import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import List
import tensorflow_hub as hub
import nltk

nltk.download("punkt")

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def rank_passages_use_cosine(query: str, passages: List[str]) -> List[int]:
    """Rank passages by cosine similarity using USE embeddings."""
    # Generate embeddings
    query_embedding = use_model([query]).numpy()
    passage_embeddings = use_model(passages).numpy()

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_embedding, passage_embeddings).flatten()

    # Rank passages by similarity score
    ranked_indices = np.argsort(similarity_scores)[::-1]
    return ranked_indices.tolist()


def rank_passages_use_dot(query: str, passages: List[str]) -> List[int]:
    """Rank passages by dot product similarity using USE embeddings."""
    # Generate embeddings
    query_embedding = use_model([query]).numpy()
    passage_embeddings = use_model(passages).numpy()

    # Compute dot product
    similarity_scores = np.dot(passage_embeddings, query_embedding.T).flatten()

    # Rank passages by similarity score
    ranked_indices = np.argsort(similarity_scores)[::-1]
    return ranked_indices.tolist()


def rank_passages_use_knn(query: str, passages: List[str], k: int = 5) -> List[int]:
    """Rank passages using KNN similarity with USE embeddings."""
    if len(passages) == 0:
        return []  # No passages to rank

    # Adjust k to be at most the number of passages
    adjusted_k = min(k, len(passages))

    # Generate embeddings
    query_embedding = use_model([query]).numpy()
    passage_embeddings = use_model(passages).numpy()

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=adjusted_k, metric="cosine").fit(
        passage_embeddings
    )

    # Find nearest neighbors for the query
    distances, indices = knn.kneighbors(query_embedding)

    # Return ranked indices (flatten since KNN returns a 2D array)
    return indices.flatten().tolist()


def rank_passages_use_bm25(query: str, passages: List[str]) -> List[int]:
    """Rank passages using BM25 with tokenized USE passages."""
    # Tokenize passages
    tokenized_passages = [word_tokenize(passage.lower()) for passage in passages]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_passages)

    # Tokenize query
    query_tokens = word_tokenize(query.lower())

    # Compute BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Rank passages by score
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices.tolist()
