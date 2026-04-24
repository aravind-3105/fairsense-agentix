"""FAISS index tool for vector similarity search.

This module provides a production-ready FAISS index wrapper that loads pre-built
indexes and their metadata, performs similarity searches, and returns ranked results.

Example
-------
    >>> from fairsense_agentix.tools.embeddings import SentenceTransformerEmbedder
    >>> embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", 384)
    >>> index = FAISSIndexTool(
    ...     index_path=Path("data/indexes/risks.faiss"), embedder=embedder, top_k=5
    ... )
    >>> results = index.search("algorithmic bias in AI systems")
    >>> len(results)
    5
"""

import json
from pathlib import Path

import faiss
import numpy as np

from fairsense_agentix.tools.exceptions import FAISSError
from fairsense_agentix.tools.interfaces import EmbedderTool


class FAISSIndexTool:
    """FAISS-based vector similarity search tool.

    This class provides semantic search functionality using pre-built FAISS indexes.
    It loads both the FAISS index file and associated metadata JSON, and performs
    fast nearest-neighbor searches.

    Parameters
    ----------
    index_path : Path
        Path to the FAISS index file (e.g., "data/indexes/risks.faiss")
    embedder : EmbedderTool
        Embedder tool for converting text queries to vectors
    top_k : int, optional
        Default number of results to return, by default 5
    metadata_path : Path | None, optional
        Path to metadata JSON. If None, inferred as "{index_path}_meta.json"

    Attributes
    ----------
    index : faiss.Index
        Loaded FAISS index
    metadata : list[dict]
        Document metadata (IDs, text, etc.)
    embedder : EmbedderTool
        Embedder for query encoding
    index_path : Path
        Path to index file
    top_k : int
        Default number of results

    Raises
    ------
    FAISSError
        If index or metadata loading fails

    Examples
    --------
    >>> embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", 384)
    >>> index = FAISSIndexTool(Path("data/indexes/risks.faiss"), embedder)
    >>> results = index.search("data privacy violation", top_k=3)
    >>> results[0]["score"]  # Similarity score
    0.85
    """

    def __init__(
        self,
        index_path: Path,
        embedder: EmbedderTool,
        top_k: int = 5,
        metadata_path: Path | None = None,
    ) -> None:
        """Initialize FAISS index tool.

        Parameters
        ----------
        index_path : Path
            Path to FAISS index file
        embedder : EmbedderTool
            Embedder for query encoding
        top_k : int
            Default number of results
        metadata_path : Path | None
            Path to metadata JSON (auto-inferred if None)
        """
        self.index_path = index_path
        self.embedder = embedder
        self.top_k = top_k

        # Infer metadata path if not provided
        if metadata_path is None:
            # Convert risks.faiss -> risks_meta.json
            metadata_path = index_path.with_suffix("").with_name(
                f"{index_path.stem}_meta.json",
            )
        self.metadata_path = metadata_path

        # Load index and metadata
        self._load_index()
        self._load_metadata()

    def _load_index(self) -> None:
        """Load FAISS index from disk.

        Raises
        ------
        FAISSError
            If index file doesn't exist or loading fails
        """
        if not self.index_path.exists():
            raise FAISSError(
                f"FAISS index file not found: {self.index_path}",
                context={"index_path": str(self.index_path)},
            )

        try:
            self.index = faiss.read_index(str(self.index_path))
        except Exception as e:
            raise FAISSError(
                f"Failed to load FAISS index: {e}",
                context={
                    "index_path": str(self.index_path),
                    "error_type": type(e).__name__,
                },
            ) from e

    def _load_metadata(self) -> None:
        """Load metadata JSON from disk.

        Raises
        ------
        FAISSError
            If metadata file doesn't exist or loading fails
        """
        if not self.metadata_path.exists():
            raise FAISSError(
                f"Metadata file not found: {self.metadata_path}",
                context={"metadata_path": str(self.metadata_path)},
            )

        try:
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)

            # Validate metadata is a list
            if not isinstance(self.metadata, list):
                raise FAISSError(
                    "Metadata must be a list of dictionaries",
                    context={
                        "metadata_path": str(self.metadata_path),
                        "actual_type": type(self.metadata).__name__,
                    },
                )

            # Validate metadata length matches index
            if len(self.metadata) != self.index.ntotal:
                raise FAISSError(
                    f"Metadata count ({len(self.metadata)}) doesn't match "
                    f"index size ({self.index.ntotal})",
                    context={
                        "metadata_count": len(self.metadata),
                        "index_size": self.index.ntotal,
                    },
                )

        except FAISSError:
            raise
        except Exception as e:
            raise FAISSError(
                f"Failed to load metadata: {e}",
                context={
                    "metadata_path": str(self.metadata_path),
                    "error_type": type(e).__name__,
                },
            ) from e

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Search index for nearest neighbors using query vector.

        Parameters
        ----------
        query_vector : list[float]
            Query vector to search for
        top_k : int, optional
            Number of results to return, by default 5

        Returns
        -------
        list[dict]
            List of result dictionaries with keys:
            - "id": Document ID
            - "score": Similarity score (higher = more similar)
            - "rank": Result rank (0 = best match)
            - Additional metadata fields (e.g., "text", "severity")

        Raises
        ------
        FAISSError
            If search fails

        Examples
        --------
        >>> vector = [0.1, 0.2, ..., 0.5]  # 384-dim vector
        >>> results = index.search(vector, top_k=3)
        >>> results[0]["id"]
        'RISK001'
        """
        # Convert list to numpy array for internal processing
        query_np = np.array(query_vector, dtype="float32")
        return self._search_internal(query_np, top_k)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search using text with automatic embedding.

        Parameters
        ----------
        query_text : str
            Text query to search for
        top_k : int, optional
            Number of results to return, by default 5

        Returns
        -------
        list[dict]
            List of result dictionaries (same format as search())

        Raises
        ------
        FAISSError
            If search fails

        Examples
        --------
        >>> results = index.search_by_text("algorithmic bias", top_k=3)
        >>> results[0]["id"]
        'RISK001'
        """
        # Handle empty query
        if not query_text or not query_text.strip():
            raise FAISSError(
                "Query text cannot be empty",
                context={"query_length": len(query_text)},
            )

        # Embed the query text
        query_vector = self.embedder.encode(query_text)

        return self._search_internal(query_vector, top_k)

    def _search_internal(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        """Perform internal search operation (shared by search and search_by_text).

        Parameters
        ----------
        query_vector : np.ndarray
            Query vector (1D numpy array)
        top_k : int
            Number of results to return

        Returns
        -------
        list[dict]
            Search results

        Raises
        ------
        FAISSError
            If search fails
        """
        try:
            # Ensure query is 2D array (FAISS requirement: [1, dimension])
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            # Validate dimension
            expected_dim = self.index.d
            actual_dim = query_vector.shape[1]
            if actual_dim != expected_dim:
                raise FAISSError(
                    f"Query dimension mismatch: expected {expected_dim}, "
                    f"got {actual_dim}",
                    context={
                        "expected_dimension": expected_dim,
                        "actual_dimension": actual_dim,
                    },
                )

            # Perform FAISS search
            # Returns: distances (similarities), indices (document IDs)
            distances, indices = self.index.search(query_vector, top_k)

            # Build result list
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
                # Skip invalid indices (FAISS returns -1 for missing results)
                if idx == -1:
                    continue

                # Get metadata for this document
                doc_metadata = self.metadata[int(idx)].copy()

                # Add search metadata
                doc_metadata["score"] = float(score)
                doc_metadata["rank"] = rank

                results.append(doc_metadata)

            return results

        except FAISSError:
            raise
        except Exception as e:
            raise FAISSError(
                f"FAISS search failed: {e}",
                context={
                    "top_k": top_k,
                    "error_type": type(e).__name__,
                },
            ) from e
