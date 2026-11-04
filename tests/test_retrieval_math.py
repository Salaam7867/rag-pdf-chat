"""Dependency-light tests for cosine similarity mathematics."""

import numpy as np
import pytest

from src.retrieval.similarity import cosine_similarity_scores


def test_cosine_similarity_for_identical_and_orthogonal_vectors():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    scores = cosine_similarity_scores(query, matrix)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_cosine_similarity_handles_zero_vectors():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.0, 0.0]], dtype=np.float32)

    scores = cosine_similarity_scores(query, matrix)

    assert scores.tolist() == [0.0]


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="different dimensions"):
        cosine_similarity_scores([1.0, 0.0], [[1.0, 0.0, 1.0]])
