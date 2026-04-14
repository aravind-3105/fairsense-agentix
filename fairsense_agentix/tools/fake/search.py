"""Fake embedding and search tool implementations for testing."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from fairsense_agentix.tools.interfaces import EmbedderTool


class FakeEmbedderTool:
    """Fake embedding tool for testing.

    Returns deterministic vectors based on text hash.

    Examples
    --------
    >>> embedder = FakeEmbedderTool(dimension=384)
    >>> vector = embedder.encode("Sample text")
    >>> len(vector)
    384
    >>> embedder.dimension
    384
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize fake embedder with specified dimension."""
        self._dimension = dimension
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def encode(self, text: str) -> np.ndarray:
        """Generate fake embedding vector."""
        self.call_count += 1
        self.last_call_args = {"text_len": len(text)}

        text_hash = hash(text)
        vector = [(text_hash + i) % 1000 / 1000.0 for i in range(self._dimension)]
        return np.array(vector, dtype="float32")

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class FakeFAISSIndexTool:
    """Fake FAISS index tool for testing.

    Returns deterministic search results. Optionally accepts custom results
    for specific test scenarios.

    Examples
    --------
    >>> index = FakeFAISSIndexTool(index_path=Path("data/risks.faiss"))
    >>> results = index.search([0.1, 0.2, ...], top_k=3)
    >>> len(results)
    3
    >>> results[0]["text"]
    'Risk 1: Bias in training data'
    """

    def __init__(
        self,
        index_path: Path,
        embedder: "EmbedderTool | None" = None,
        *,
        return_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize fake FAISS index.

        Parameters
        ----------
        index_path : Path
            Path to fake index file
        embedder : EmbedderTool | None, optional
            Embedder for text-based search, by default None
        return_results : list[dict[str, Any]] | None, optional
            Fixed results to return (if None, generate deterministic results),
            by default None
        """
        self._index_path = index_path
        self.embedder = embedder
        self.return_results = return_results
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform fake vector search.

        Phase 7.2: Updated to return appropriate data structure
        for risks vs RMF indexes.
        Detects index type from path and returns fields that pass risk evaluator checks.
        """
        self.call_count += 1
        self.last_call_args = {
            "query_vector_dim": len(query_vector),
            "top_k": top_k,
        }

        if self.return_results is not None:
            return self.return_results[:top_k]

        vector_sum = sum(query_vector)
        results = []

        is_rmf_index = "rmf" in str(self._index_path).lower()
        rmf_functions = ["Govern", "Map", "Measure", "Manage"]

        for i in range(top_k):
            if is_rmf_index:
                _rmf_line = (
                    f"RMF Recommendation {i + 1}: Implement controls "
                    f"(seed={vector_sum:.2f})"
                )
                results.append(
                    {
                        "id": f"RMF{i:03d}",
                        "rmf_id": f"RMF{i:03d}",
                        "text": _rmf_line,
                        "recommendation": _rmf_line,
                        "function": rmf_functions[i % 4],
                        "category": f"Category {(i % 3) + 1}",
                        "score": 0.95 - (i * 0.1),
                        "relevance_score": 0.95 - (i * 0.1),
                    },
                )
            else:
                _risk_line = (
                    f"Risk {i + 1}: Bias in training data (seed={vector_sum:.2f})"
                )
                results.append(
                    {
                        "id": f"RISK{i:03d}",
                        "risk_id": f"RISK{i:03d}",
                        "text": _risk_line,
                        "description": _risk_line,
                        "score": 0.95 - (i * 0.1),
                        "severity": "high" if i < 2 else "medium",
                        "category": "fairness",
                        "likelihood": "medium",
                    },
                )

        return results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform fake text-based search."""
        embedder = FakeEmbedderTool() if self.embedder is None else self.embedder
        query_vector = embedder.encode(query_text)
        return self.search(query_vector.tolist(), top_k)

    @property
    def index_path(self) -> Path:
        """Get index path."""
        return self._index_path
