"""Embedding and search tool protocol interfaces."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbedderTool(Protocol):
    """Protocol for text embedding tools.

    Converts text into dense vector representations using embedding models
    (sentence-transformers, OpenAI embeddings, etc.).

    Examples
    --------
    >>> embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    >>> vector = embedder.encode("Sample text")
    >>> print(len(vector))
    384
    >>> print(embedder.dimension)
    384
    """

    def encode(self, text: str) -> np.ndarray:
        """Convert text to dense vector embedding.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        np.ndarray
            Dense vector representation (1D array of shape (dimension,))

        Raises
        ------
        EmbeddingError
            If embedding fails (text too long, model error, etc.)
        """
        ...

    @property
    def dimension(self) -> int:
        """Get embedding dimension.

        Returns
        -------
        int
            Vector dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        ...


@runtime_checkable
class FAISSIndexTool(Protocol):
    """Protocol for FAISS vector similarity search tools.

    Performs efficient nearest-neighbor search over large vector collections.
    Used for retrieving relevant risks and RMF recommendations.

    Examples
    --------
    >>> index = FAISSIndex(index_path="data/indexes/risks.faiss")
    >>> results = index.search(query_vector, top_k=5)
    >>> for result in results:
    ...     print(result["text"], result["score"])
    """

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search index for nearest neighbors.

        Parameters
        ----------
        query_vector : list[float]
            Query vector to search for
        top_k : int, optional
            Number of results to return, by default 5

        Returns
        -------
        list[dict[str, Any]]
            List of result dictionaries with keys:
                - "id": int or str
                - "text": str
                - "score": float (similarity score)
                - Additional metadata fields

        Raises
        ------
        FAISSError
            If search fails (index not found, dimension mismatch, etc.)
        """
        ...

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search using text with automatic embedding.

        Requires the index to have an associated embedder.

        Parameters
        ----------
        query_text : str
            Text query to search for
        top_k : int, optional
            Number of results to return, by default 5

        Returns
        -------
        list[dict[str, Any]]
            List of result dictionaries (same format as search())

        Raises
        ------
        FAISSError
            If search fails
        """
        ...

    @property
    def index_path(self) -> Path:
        """Get path to FAISS index file.

        Returns
        -------
        Path
            Path to .faiss index file
        """
        ...
