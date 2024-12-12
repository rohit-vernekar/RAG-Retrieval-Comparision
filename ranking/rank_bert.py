import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Okapi
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)


def rank_passages_bert_cosine(query: str, passages: List[str]) -> List[int]:
    """
    Rank passages based on their similarity to a given query using SentenceTransformer embeddings.

    Args:
        query (str): The input query.
        passages (List[str]): A list of passages to rank.

    Returns:
        List[int]: Indices of passages ranked by similarity to the query (descending order).
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    # Encode the query
    query_embedding = model.encode(query)

    # Encode the passages
    passage_embeddings = model.encode(passages)

    # Compute cosine similarity between query and passages
    similarity_scores = cosine_similarity(
        [query_embedding], passage_embeddings
    ).flatten()

    # Rank indices by similarity scores in descending order
    ranked_indices = similarity_scores.argsort()[::-1]
    return ranked_indices.tolist()


def rank_passages_bert_mahalanobis(query: str, passages: List[str]) -> List[int]:
    """
    Rank passages based on their similarity to a given query using SentenceTransformer embeddings
    and Mahalanobis distance.

    Args:
        query (str): The input query.
        passages (List[str]): A list of passages to rank.

    Returns:
        List[int]: Indices of passages ranked by similarity to the query (ascending Mahalanobis distance).
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    # Encode query
    query_embedding = model.encode(query)

    # Encode passages
    passage_embeddings = model.encode(passages)

    # Calculate covariance matrix and its inverse
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(passage_embeddings)
    cov_matrix_inv = cov_estimator.precision_

    # Calculate Mahalanobis distances
    distances = [
        mahalanobis(query_embedding, passage_embedding, cov_matrix_inv)
        for passage_embedding in passage_embeddings
    ]

    # Rank indices based on distances
    ranked_indices = sorted(range(len(distances)), key=lambda i: distances[i])
    return ranked_indices


def rank_passages_bert_dot(query: str, passages: List[str]) -> List[int]:
    """
    Rank passages based on their similarity to a query using SentenceTransformer embeddings
    and dot product scoring.

    Args:
        query (str): The input query.
        passages (List[str]): A list of passages to rank.

    Returns:
        List[int]: Indices of passages ranked by similarity to the query (descending order).
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    # Encode the query
    query_embedding = model.encode(query)

    # Encode the passages
    passage_embeddings = model.encode(passages)

    # Compute dot product similarity scores
    similarity_scores = np.dot(passage_embeddings, query_embedding)

    # Rank indices by similarity scores in descending order
    ranked_indices = np.argsort(similarity_scores)[::-1].tolist()
    return ranked_indices


def rank_passages_bert_knn(query: str, passages: List[str], k: int = 5) -> List[int]:
    """
    Rank passages based on their similarity to a query using SentenceTransformer embeddings
    and K-Nearest Neighbors (KNN).

    Args:
        query (str): The input query.
        passages (List[str]): A list of passages to rank.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        List[int]: Indices of the top-k most similar passages ranked by similarity.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    # Encode the query and passages
    query_embedding = model.encode(query)
    passage_embeddings = model.encode(passages)

    # Use KNN to find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(passage_embeddings)
    distances, indices = knn.kneighbors([query_embedding])

    # Flatten indices (KNN returns a 2D array for batching)
    ranked_indices = indices[0].tolist()
    return ranked_indices


def rank_passages_bert_bm25(query: str, passages: List[str]) -> List[int]:
    tokenizer = BertTokenizer.from_pretrained(
        "transformersbook/bert-base-uncased-finetuned-clinc"
    )

    # Tokenize passages
    tokenized_passages = [tokenizer.tokenize(passage) for passage in passages]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_passages)

    # Tokenize query
    tokenized_query = tokenizer.tokenize(query)

    # Compute BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get ranked indices based on BM25 scores
    ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )
    return ranked_indices
