from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def rank_passages_tfidf(query: str, passages: List[str]) -> List[int]:
    """Rank passages by cosine similarity with the query using TF-IDF vectors."""
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    # Fit TF-IDF on passages and transform query
    passage_tfidf_matrix = tfidf_vectorizer.fit_transform(passages)
    query_tfidf = tfidf_vectorizer.transform([query])

    # Compute cosine similarity between query and all passages
    similarity_scores = cosine_similarity(query_tfidf, passage_tfidf_matrix).flatten()

    # Rank passages by similarity score
    ranked_indices = np.argsort(similarity_scores)[::-1]
    return ranked_indices.tolist()
