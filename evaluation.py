import numpy as np
import pandas as pd
from typing import List

def evaluate_metrics(df: pd.DataFrame, relevance_col: str, rank_col: str, k: int = 3):
    """
    Function to calculate 
        - MRR (Mean Reciprocal Rank) 
        - nDCG (Normalized Discounted Cumulative Gain)
        - Precision@k 
        - Recall@k
        - MAP (Mean Average Precision)
    """
    mrr_scores = []
    ndcg_scores = []
    precision_at_k = []
    recall_at_k = []
    average_precisions = []

    for _, row in df.iterrows():
        relevance = np.array(row[relevance_col])  # Convert relevance list to numpy array
        rank = np.array(row[rank_col])  # Ranked indices
        relevance_at_rank = relevance[rank]

        # MRR: Find the rank of the first relevant document
        first_relevant = np.where(relevance_at_rank == 1)[0]
        if len(first_relevant) > 0:
            mrr_scores.append(1 / (first_relevant[0] + 1))
        else:
            mrr_scores.append(0)

        # nDCG
        dcg = sum((2 ** relevance_at_rank[i] - 1) / np.log2(i + 2) for i in range(len(relevance_at_rank)))
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum((2 ** ideal_relevance[i] - 1) / np.log2(i + 2) for i in range(len(ideal_relevance)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

        # Top-k relevance
        relevance_at_k = relevance_at_rank[:k]

        # Precision@k
        precision = relevance_at_k.sum() / k
        precision_at_k.append(precision)

        # Recall@k
        total_relevant = relevance.sum()
        recall = relevance_at_k.sum() / total_relevant if total_relevant > 0 else 0
        recall_at_k.append(recall)

        # Average Precision (AP)
        num_relevant_retrieved = 0
        cumulative_precision = 0
        for i in range(len(relevance_at_rank)):
            if relevance_at_rank[i] == 1:
                num_relevant_retrieved += 1
                cumulative_precision += num_relevant_retrieved / (i + 1)
        ap = cumulative_precision / total_relevant if total_relevant > 0 else 0
        average_precisions.append(ap)

    # Aggregate results
    metrics = {
        "MRR": np.mean(mrr_scores),
        "nDCG": np.mean(ndcg_scores),
        f"Precision at {k}": np.mean(precision_at_k),
        f"Recall at {k}": np.mean(recall_at_k),
        "MAP": np.mean(average_precisions),
    }

    return metrics
