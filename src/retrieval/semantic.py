"""Existing NumPy cosine-similarity retrieval implementation."""

from typing import Any

import numpy as np

from src.embeddings.provider import load_embeddings
from src.retrieval.similarity import cosine_similarity_scores


def retrieve_relevant_chunks(
    chunks: list[Any],
    chunk_embeddings: Any,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve the top-k chunks using the project's existing dense similarity."""
    if not chunks or chunk_embeddings is None or top_k <= 0:
        return []

    matrix = np.asarray(chunk_embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Chunk count does not match the embedding matrix.")

    query_vector = load_embeddings().embed_query(query)
    scores = cosine_similarity_scores(query_vector, matrix)
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]

    return [
        {"chunk": chunks[int(index)], "score": float(scores[index])}
        for index in top_indices
    ]
