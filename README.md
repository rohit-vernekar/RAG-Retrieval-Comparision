# RAG Retrieval Evaluation

## Overview

This project evaluates various retrieval and ranking algorithms for Retrieval-Augmented Generation (RAG) systems in open-domain question answering. Models assessed include traditional approaches like TF-IDF and advanced models such as BERT, MiniLM, and DistilBERT. Fine-tuning of DistilBERT and MiniLM is explored, along with MPNet's performance, highlighting the challenges encountered. Evaluation metrics like Precision@3, Recall@3, Mean Average Precision (MAP), Mean Reciprocal Rank (MRR), and nDCG are used for performance comparisons.

## Dataset

This project uses the **MS MARCO dataset**, a large-scale corpus for machine reading comprehension and passage ranking tasks. It includes over 8.8 million passages paired with queries and relevance labels. The dataset, accessed via the Hugging Face library, provides a robust structure for evaluating retrieval models under real-world query scenarios.

## Key Features

### Baseline Model
- **TF-IDF Vectorizer**:
  - Implements Term Frequency-Inverse Document Frequency (TF-IDF) to transform text into numerical vectors.
  - Uses cosine similarity to rank passages based on their relevance to the query.

### Advanced Models
- **BERT**: A transformer-based model that generates contextual embeddings for better semantic understanding.
- **MiniLM**: A lightweight transformer model that balances efficiency and effectiveness for tasks like similarity scoring.
- **Universal Sentence Encoder (USE)**: A deep learning model that produces dense vector representations, capturing semantic relationships.
- **OpenAI Embeddings**: High-dimensional semantic embeddings (1536 dimensions) generated using OpenAI’s text embedding models.

### Fine-Tuned Models
- **DistilBERT**: Fine-tuned on the MS MARCO dataset to predict passage relevance efficiently.
- **MiniLM**: Optimized for relevance scoring through fine-tuning on MS MARCO.
- **MPNet**: Pretrained and fine-tuned for ranking passages, leveraging masked and permuted language modeling for superior performance.

### Similarity Scoring Methods
- **Cosine Similarity**: Measures the cosine of the angle between query and document embeddings.
- **Dot Product**: Calculates the alignment between two vectors to assess similarity.
- **K-Nearest Neighbors (KNN)**: Uses Euclidean distance to find the top k semantically similar embeddings.
- **Mahalanobis Distance**: Accounts for correlations between variables in high-dimensional embedding space to compute similarity.

## Evaluation Metrics

The following metrics are used to evaluate the performance of retrieval models:

- **Precision@k**: Measures the proportion of relevant documents retrieved in the top k results. A high Precision@k indicates the model effectively ranks relevant passages at the top.
- **Recall@k**: Calculates the fraction of all relevant documents retrieved within the top k results. This metric focuses on how comprehensively the model retrieves all relevant passages.
- **Mean Average Precision (MAP)**: Computes the average precision across all queries, considering the ranks at which relevant documents are retrieved. It provides an overall measure of precision and ranking effectiveness.
- **Mean Reciprocal Rank (MRR)**: Evaluates how quickly the first relevant document is retrieved for each query. A higher MRR indicates the most relevant results appear early in the ranking.
- **Normalized Discounted Cumulative Gain (nDCG)**: Assesses ranking quality by prioritizing highly relevant documents in higher positions. nDCG considers the order of results and assigns higher importance to top-ranked documents. It is normalized to ensure fair comparisons across queries of different lengths.

## References

1. [MS MARCO Dataset](https://huggingface.co/datasets/microsoft/ms_marco)
2. [RAG Research](https://arxiv.org/abs/2005.11401)
3. [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
4. [BERT](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens)
5. [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
6. [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
