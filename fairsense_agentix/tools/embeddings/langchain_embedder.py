"""LangChain-based embedder implementation.

This module provides a LangChain-compatible embedder that wraps HuggingFace embeddings
while satisfying the EmbedderTool protocol. This enables better integration with
LangChain's FAISS vector store and retriever patterns.

Design Choices
--------------
1. **Why LangChain wrapper**: Better integration with LangChain FAISS and retrievers
2. **Why maintain protocol**: Keep compatibility with existing graph code
3. **Why wrap HuggingFaceEmbeddings**: Leverage LangChain's caching and optimization
4. **Why similar API to custom**: Drop-in replacement for SentenceTransformerEmbedder

What This Enables
-----------------
- Seamless integration with LangChain FAISS vector store
- Access to LangChain's retriever patterns and chain composition
- Future-proof architecture for LangChain ecosystem updates
- Maintain backward compatibility with existing EmbedderTool protocol

Examples
--------
    >>> from langchain_huggingface import HuggingFaceEmbeddings
    >>> embedder = LangChainEmbedder(model_name="all-MiniLM-L6-v2", dimension=384)
    >>> vector = embedder.encode("AI bias detection")
    >>> vector.shape
    (384,)
"""

import logging
from typing import Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from fairsense_agentix.services.telemetry import TelemetryService
from fairsense_agentix.tools.exceptions import EmbeddingError


logger = logging.getLogger(__name__)


class LangChainEmbedder:
    """Embedder using LangChain's HuggingFaceEmbeddings.

    This class wraps LangChain's HuggingFaceEmbeddings to provide text embedding
    functionality while satisfying the EmbedderTool protocol. It serves as a bridge
    between LangChain's ecosystem and our custom tool interfaces.

    Design Philosophy:
    - **Protocol compliance**: Satisfies EmbedderTool for graph compatibility
    - **LangChain integration**: Enables retriever patterns and chain composition
    - **Drop-in replacement**: Same API as SentenceTransformerEmbedder
    - **Performance**: Leverages LangChain's optimizations and caching

    Parameters
    ----------
    model_name : str
        Name of the HuggingFace model to use for embeddings.
        Examples: "all-MiniLM-L6-v2" (384-dim), "all-mpnet-base-v2" (768-dim)
    dimension : int
        Expected embedding dimension. Must match the model's output dimension.
    normalize : bool, optional
        Whether to normalize embeddings to unit length (for cosine similarity).
        Default is True.
    model_kwargs : dict | None, optional
        Additional kwargs passed to HuggingFaceEmbeddings.
        Can include device="cuda" for GPU, trust_remote_code=True, etc.

    Attributes
    ----------
    embeddings : HuggingFaceEmbeddings
        The underlying LangChain embeddings object
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
    >>> # Basic usage
    >>> embedder = LangChainEmbedder("all-MiniLM-L6-v2", 384)
    >>> vector = embedder.encode("Sample text")
    >>>
    >>> # With GPU
    >>> embedder = LangChainEmbedder(
    ...     "all-MiniLM-L6-v2", 384, model_kwargs={"device": "cuda"}
    ... )
    >>>
    >>> # Access underlying LangChain embeddings for retriever usage
    >>> faiss_store = FAISS.from_documents(docs, embedder.embeddings)

    Notes
    -----
    The `.embeddings` attribute provides direct access to the LangChain
    HuggingFaceEmbeddings object, enabling use with LangChain's FAISS vector
    store and retriever patterns.
    """

    def __init__(
        self,
        model_name: str,
        dimension: int,
        normalize: bool = True,
        model_kwargs: dict | None = None,
        telemetry: Optional[TelemetryService] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Initialize LangChain embedder.

        Parameters
        ----------
        model_name : str
            HuggingFace model name
        dimension : int
            Expected embedding dimension
        normalize : bool
            Whether to normalize embeddings
        model_kwargs : dict | None
            Additional model kwargs (device, etc.)
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
                    model_type="embedder_langchain",
                    run_id=self._run_id,
                    message=f"Downloading LangChain embedding model '{model_name}' (first use only, ~30-120s)...",
                )

            # Prepare model kwargs
            final_model_kwargs = model_kwargs or {}

            # Add normalize_embeddings if specified
            encode_kwargs = {"normalize_embeddings": normalize}

            # Initialize LangChain HuggingFaceEmbeddings
            # NOTE: This call can take 30-120s on first use (downloads from HuggingFace)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=final_model_kwargs,
                encode_kwargs=encode_kwargs,
            )

            # Verify dimension matches by testing embedding
            logger.info(
                f"LangChainEmbedder initializing with model={model_name}, dimension={dimension}"
            )

            test_embedding = self.embeddings.embed_query("test")
            actual_dim = len(test_embedding)

            if actual_dim != dimension:
                raise EmbeddingError(
                    f"Model dimension mismatch: expected {dimension}, got {actual_dim}",
                    context={
                        "model_name": model_name,
                        "expected_dimension": dimension,
                        "actual_dimension": actual_dim,
                    },
                )

            logger.info(
                f"LangChainEmbedder initialized successfully (model={model_name}, dimension={actual_dim})"
            )

            # Emit "model ready" event
            if self._telemetry and self._run_id:
                self._telemetry.log_info(
                    "model_download_complete",
                    model_name=model_name,
                    model_type="embedder_langchain",
                    dimension=actual_dim,
                    run_id=self._run_id,
                    message=f"LangChain embedding model '{model_name}' ready",
                )

        except Exception as e:
            # Emit failure event
            if self._telemetry and self._run_id:
                self._telemetry.log_error(
                    "model_download_failed",
                    error=e,
                    model_name=model_name,
                    model_type="embedder_langchain",
                    run_id=self._run_id,
                )

            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(
                f"Failed to load LangChain HuggingFace embeddings: {e}",
                context={
                    "model_name": model_name,
                    "error_type": type(e).__name__,
                },
            ) from e

    def encode(self, text: str) -> np.ndarray:
        """Encode text into embedding vector.

        Uses LangChain's HuggingFaceEmbeddings to generate embeddings, then
        converts to numpy array for protocol compatibility.

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
            If encoding fails or text is empty

        Examples
        --------
        >>> embedder = LangChainEmbedder("all-MiniLM-L6-v2", 384)
        >>> vector = embedder.encode("AI fairness")
        >>> isinstance(vector, np.ndarray)
        True
        >>> vector.shape
        (384,)

        Notes
        -----
        LangChain's .embed_query() returns a List[float], which we convert
        to numpy array for consistency with the EmbedderTool protocol.
        """
        if not text or not text.strip():
            raise EmbeddingError(
                "Cannot encode empty text",
                context={"text_length": len(text)},
            )

        try:
            # Use LangChain's embed_query method (returns List[float])
            embedding_list = self.embeddings.embed_query(text)

            # Convert to numpy array
            embedding_array = np.array(embedding_list, dtype=np.float32)

            # Verify shape
            if embedding_array.shape[0] != self._dimension:
                raise EmbeddingError(
                    f"Unexpected embedding shape: {embedding_array.shape}",
                    context={
                        "expected_dimension": self._dimension,
                        "actual_shape": embedding_array.shape,
                    },
                )

            return embedding_array

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(
                f"Failed to encode text with LangChain embeddings: {e}",
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
