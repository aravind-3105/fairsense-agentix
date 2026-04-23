"""Sentence-Transformers based embedder implementation.

This module provides a production-ready embedder using the sentence-transformers
library. It converts text into dense vector representations suitable for semantic
similarity search via FAISS.

Example
-------
    >>> embedder = SentenceTransformerEmbedder(
    ...     model_name="all-MiniLM-L6-v2", dimension=384
    ... )
    >>> vector = embedder.encode("AI bias detection")
    >>> vector.shape
    (384,)
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from fairsense_agentix.services.telemetry import TelemetryService
from fairsense_agentix.tools.exceptions import EmbeddingError


class SentenceTransformerEmbedder:
    """Embedder using Sentence-Transformers models.

    This class provides text embedding functionality using pre-trained
    sentence-transformers models. The model is cached in memory after first load
    to avoid repeated downloads and initialization.

    Parameters
    ----------
    model_name : str
        Name of the sentence-transformers model to use.
        Examples: "all-MiniLM-L6-v2" (384-dim), "all-mpnet-base-v2" (768-dim)
    dimension : int
        Expected embedding dimension. Must match the model's output dimension.
    normalize : bool, optional
        Whether to normalize embeddings to unit length (for cosine similarity).
        Default is True.

    Attributes
    ----------
    model : SentenceTransformer
        The loaded sentence-transformers model
    model_name : str
        Name of the model being used
    _dimension : int
        Embedding dimension

    Raises
    ------
    EmbeddingError
        If model loading fails or dimension mismatch detected

    Examples
    --------
    >>> embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", 384)
    >>> vector = embedder.encode("Sample text for embedding")
    >>> isinstance(vector, np.ndarray)
    True
    >>> vector.shape
    (384,)
    """

    def __init__(
        self,
        model_name: str,
        dimension: int,
        normalize: bool = True,
        telemetry: Optional[TelemetryService] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Initialize embedder with sentence-transformers model.

        Parameters
        ----------
        model_name : str
            Sentence-transformers model name
        dimension : int
            Expected embedding dimension
        normalize : bool
            Whether to normalize embeddings
        telemetry : TelemetryService, optional
            Telemetry service for logging model download events
        run_id : str, optional
            Run ID for tracking this initialization in distributed traces
        """
        self.model_name = model_name
        self._dimension = dimension
        self.normalize = normalize
        self._telemetry = telemetry
        self._run_id = run_id

        try:
            # Emit "downloading model" event (critical for UI feedback on first use)
            if self._telemetry and self._run_id:
                self._telemetry.log_info(
                    "model_download_start",
                    model_name=model_name,
                    model_type="embedder",
                    run_id=self._run_id,
                    message=(
                        f"Downloading embedding model '{model_name}' "
                        "(first use only, ~30-120s)..."
                    ),
                )

            # Load model (cached by sentence-transformers after first load)
            # NOTE: This call can take 30-120s on first use (downloads from HuggingFace)
            self.model = SentenceTransformer(model_name)

            # Verify dimension matches
            test_embedding = self.model.encode("test", normalize_embeddings=False)
            actual_dim = test_embedding.shape[0]

            if actual_dim != dimension:
                raise EmbeddingError(
                    f"Model dimension mismatch: expected {dimension}, got {actual_dim}",
                    context={
                        "model_name": model_name,
                        "expected_dimension": dimension,
                        "actual_dimension": actual_dim,
                    },
                )

            # Emit "model ready" event
            if self._telemetry and self._run_id:
                self._telemetry.log_info(
                    "model_download_complete",
                    model_name=model_name,
                    model_type="embedder",
                    dimension=actual_dim,
                    run_id=self._run_id,
                    message=f"Embedding model '{model_name}' ready",
                )

        except Exception as e:
            # Emit failure event
            if self._telemetry and self._run_id:
                self._telemetry.log_error(
                    "model_download_failed",
                    error=e,
                    model_name=model_name,
                    model_type="embedder",
                    run_id=self._run_id,
                )

            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(
                f"Failed to load sentence-transformers model: {e}",
                context={"model_name": model_name, "error_type": type(e).__name__},
            ) from e

    def encode(self, text: str) -> np.ndarray:
        """Encode text into embedding vector.

        Parameters
        ----------
        text : str
            Text to encode

        Returns
        -------
        np.ndarray
            Embedding vector of shape (dimension,)

        Raises
        ------
        EmbeddingError
            If encoding fails
        """
        if not text or not text.strip():
            raise EmbeddingError(
                "Cannot encode empty text",
                context={"text_length": len(text)},
            )

        try:
            # Encode with sentence-transformers (returns np.ndarray or Tensor)
            embedding_result = self.model.encode(
                text,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )

            # Ensure numpy array (convert from Tensor if needed)
            embedding_array: np.ndarray
            if not isinstance(embedding_result, np.ndarray):
                embedding_array = np.array(embedding_result)
            else:
                embedding_array = embedding_result

            # Verify shape
            if embedding_array.shape[0] != self._dimension:
                raise EmbeddingError(
                    f"Unexpected embedding shape: {embedding_array.shape}",
                    context={"expected_dimension": self._dimension},
                )

            return embedding_array

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(
                f"Failed to encode text: {e}",
                context={
                    "model_name": self.model_name,
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                },
            ) from e

    @property
    def dimension(self) -> int:
        """Return embedding dimension.

        Returns
        -------
        int
            Embedding vector dimension
        """
        return self._dimension
