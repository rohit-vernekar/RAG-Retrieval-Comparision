import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OpenAIEmbeddings

open_ai_key = ""


def rank_passages_openai(query, passage_text):
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=open_ai_key
    )
    query_embedding = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    passage_embeddings = embedding_model.embed_documents(passage_text)

    similarity_scores = cosine_similarity(query_embedding, passage_embeddings).flatten()
    ranked_indices = np.argsort(similarity_scores)[::-1]
    return ranked_indices
