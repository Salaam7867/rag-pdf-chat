"""Pure NumPy similarity calculations, independent of application services."""

from typing import Any

import numpy as np


def cosine_similarity_scores(query_vector: Any, matrix: Any) -> np.ndarray:
    """Calculate cosine similarity between one query vector and a matrix."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    vectors = np.asarray(matrix, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("The embedding matrix must be two-dimensional.")
    if vectors.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("Query and document embeddings have different dimensions.")

    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denominators = vector_norms * query_norm

    scores = np.zeros(vectors.shape[0], dtype=np.float32)
    valid = denominators > 0
    scores[valid] = (vectors[valid] @ query) / denominators[valid]
    return scores
