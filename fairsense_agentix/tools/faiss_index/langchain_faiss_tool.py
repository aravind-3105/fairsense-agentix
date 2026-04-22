"""LangChain FAISS vector store wrapper.

This module provides a LangChain-compatible FAISS wrapper that satisfies the
FAISSIndexTool protocol while enabling access to LangChain's retriever patterns
and chain composition features.

Design Choices
--------------
1. **Why LangChain FAISS**: Cleaner API, built-in retrievers, better
   integration with LangGraph
2. **Why maintain protocol**: Keep compatibility with existing graph code
3. **Why expose vectorstore**: Enable retriever patterns (`.as_retriever()`)
4. **Why .from_documents()**: Simpler index building than manual ops

What This Enables
-----------------
- Retriever pattern for chain composition: `.as_retriever().get_relevant_documents()`
- MMR (Maximal Marginal Relevance) for diverse results
- Automatic persistence with `.save_local()` and `.load_local()`
- Better integration with LangGraph and LangChain chains
- Maintain backward compatibility with existing graphs

Examples
--------
    >>> from langchain_huggingface import HuggingFaceEmbeddings
    >>> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    >>> faiss_tool = LangChainFAISSTool.load_local(
    ...     folder_path=Path("data/indexes"), index_name="risks", embeddings=embeddings
    ... )
    >>> results = faiss_tool.search_by_text("algorithmic bias", top_k=5)
    >>>
    >>> # Use as retriever (LangChain pattern)
    >>> retriever = faiss_tool.as_retriever(search_kwargs={"k": 5})
    >>> docs = retriever.get_relevant_documents("bias detection")
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from fairsense_agentix.tools.exceptions import FAISSError


logger = logging.getLogger(__name__)


class LangChainFAISSTool:
    """FAISS vector store using LangChain's FAISS implementation.

    This class wraps LangChain's FAISS vector store to provide the same API as
    FAISSIndexTool while enabling access to LangChain's retriever patterns and
    chain composition features.

    Design Philosophy:
    - **Protocol compliance**: Satisfies FAISSIndexTool for graph compatibility
    - **LangChain integration**: Exposes vectorstore for retriever usage
    - **Drop-in replacement**: Same API as custom FAISSIndexTool
    - **Enhanced features**: MMR search, automatic persistence

    Parameters
    ----------
    vectorstore : FAISS
        LangChain FAISS vector store instance
    metadata : list[dict]
        Document metadata for each vector in the index
    top_k : int, optional
        Default number of results to return, by default 5

    Attributes
    ----------
    vectorstore : FAISS
        The underlying LangChain FAISS vector store
    metadata : list[dict]
        Document metadata
    top_k : int
        Default number of results

    Examples
    --------
    >>> # Load from disk
    >>> from langchain_huggingface import HuggingFaceEmbeddings
    >>> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    >>> faiss_tool = LangChainFAISSTool.load_local(
    ...     folder_path=Path("data/indexes"), index_name="risks", embeddings=embeddings
    ... )
    >>>
    >>> # Search with text
    >>> results = faiss_tool.search_by_text("data privacy", top_k=3)
    >>>
    >>> # Use as retriever (LangChain pattern)
    >>> retriever = faiss_tool.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    >>> docs = retriever.get_relevant_documents("algorithmic bias")

    Notes
    -----
    The `.vectorstore` attribute provides direct access to the LangChain FAISS
    object, enabling use with LangChain's retriever patterns and chain composition.
    """

    def __init__(
        self,
        vectorstore: FAISS,
        metadata: list[dict],
        top_k: int = 5,
        index_path: Path = Path("unknown.faiss"),  # For protocol compatibility
    ) -> None:
        """Initialize LangChain FAISS tool.

        Parameters
        ----------
        vectorstore : FAISS
            LangChain FAISS vector store
        metadata : list[dict]
            Document metadata
        top_k : int
            Default number of results
        index_path : Path
            Original index path (for protocol compatibility)
            Defaults to "unknown.faiss" for indexes built from documents
        """
        self.vectorstore = vectorstore
        self.metadata = metadata
        self.top_k = top_k
        self.index_path = index_path  # For FAISSIndexTool protocol

        logger.info(
            f"LangChainFAISSTool initialized (docs={len(metadata)}, top_k={top_k})",
        )

    @classmethod
    def load_local(
        cls,
        folder_path: Path,
        index_name: str,
        embeddings: Any,  # HuggingFaceEmbeddings or other LangChain embeddings
        top_k: int = 5,
    ) -> "LangChainFAISSTool":
        """Load FAISS index from disk using LangChain's persistence.

        This method loads a FAISS index and metadata that were previously saved
        using `.save_local()` or the build script.

        Parameters
        ----------
        folder_path : Path
            Directory containing the index files
        index_name : str
            Name of the index (e.g., "risks", "rmf")
        embeddings : Any
            LangChain embeddings object (HuggingFaceEmbeddings, etc.)
        top_k : int, optional
            Default number of results, by default 5

        Returns
        -------
        LangChainFAISSTool
            Loaded FAISS tool instance

        Raises
        ------
        FAISSError
            If loading fails (missing files, invalid format, etc.)

        Examples
        --------
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        >>> faiss_tool = LangChainFAISSTool.load_local(
        ...     folder_path=Path("data/indexes"),
        ...     index_name="risks",
        ...     embeddings=embeddings,
        ... )
        """
        try:
            # Load LangChain FAISS index
            # LangChain expects folder with index.faiss and index.pkl
            vectorstore = FAISS.load_local(
                folder_path=str(folder_path / index_name),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,  # Required for pickle
            )

            # Load metadata separately (our custom format)
            metadata_path = folder_path / f"{index_name}_meta.json"
            if not metadata_path.exists():
                raise FAISSError(
                    f"Metadata file not found: {metadata_path}",
                    context={"metadata_path": str(metadata_path)},
                )

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Validate metadata is a list
            if not isinstance(metadata, list):
                raise FAISSError(
                    "Metadata must be a list of dictionaries",
                    context={
                        "metadata_path": str(metadata_path),
                        "actual_type": type(metadata).__name__,
                    },
                )

            # Note: LangChain FAISS doesn't expose index.ntotal directly,
            # but we can check consistency later during search

            logger.info(
                f"Loaded LangChain FAISS index: {index_name} ({len(metadata)} docs)",
            )

            # Reconstruct index_path for protocol
            index_path_reconstructed = folder_path / f"{index_name}.faiss"

            return cls(
                vectorstore=vectorstore,
                metadata=metadata,
                top_k=top_k,
                index_path=index_path_reconstructed,
            )

        except FAISSError:
            raise
        except Exception as e:
            raise FAISSError(
                f"Failed to load LangChain FAISS index: {e}",
                context={
                    "folder_path": str(folder_path),
                    "index_name": index_name,
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
            - Additional metadata fields

        Raises
        ------
        FAISSError
            If search fails

        Examples
        --------
        >>> vector = [0.1, 0.2, ..., 0.5]  # 384-dim vector
        >>> results = faiss_tool.search(vector, top_k=3)
        >>> results[0]["score"]
        0.85
        """
        try:
            # Convert to numpy array for LangChain
            query_np = np.array(query_vector, dtype="float32")

            docs_with_scores = self.vectorstore.similarity_search_with_score_by_vector(
                embedding=query_np.tolist(),
                k=top_k,
            )

            results = []
            for rank, (doc, score) in enumerate(docs_with_scores):
                result_dict = doc.metadata.copy()
                # Convert L2 distance to cosine similarity for normalized vectors:
                # cos_sim = 1 - (l2_dist² / 2), clamped to [0, 1]
                cos_sim = max(0.0, 1.0 - (float(score) ** 2) / 2.0)
                result_dict["score"] = round(cos_sim, 3)
                result_dict["rank"] = rank
                if doc.page_content:
                    result_dict["text"] = doc.page_content
                results.append(result_dict)

            return results

        except Exception as e:
            raise FAISSError(
                f"FAISS search failed: {e}",
                context={
                    "top_k": top_k,
                    "error_type": type(e).__name__,
                },
            ) from e

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
        >>> results = faiss_tool.search_by_text("algorithmic bias", top_k=3)
        >>> results[0]["id"]
        'RISK001'
        """
        # Handle empty query
        if not query_text or not query_text.strip():
            raise FAISSError(
                "Query text cannot be empty",
                context={"query_length": len(query_text)},
            )

        try:
            # Use LangChain's similarity_search with scores
            results_with_scores = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query=query_text,
                    k=top_k,
                )
            )

            # Build result list matching our protocol
            results = []
            for rank, (doc, score) in enumerate(results_with_scores):
                # Extract metadata from document
                result_dict = doc.metadata.copy()

                # Add search metadata
                result_dict["score"] = float(score)
                result_dict["rank"] = rank

                # If text is available, add it
                if doc.page_content:
                    result_dict["text"] = doc.page_content

                results.append(result_dict)

            return results

        except Exception as e:
            raise FAISSError(
                f"FAISS text search failed: {e}",
                context={
                    "query_length": len(query_text),
                    "top_k": top_k,
                    "error_type": type(e).__name__,
                },
            ) from e

    def as_retriever(self, **kwargs: Any) -> Any:
        """Get LangChain retriever for chain composition.

        This method exposes the underlying FAISS vector store as a LangChain
        retriever, enabling use in chains and advanced patterns.

        Parameters
        ----------
        **kwargs : Any
            Arguments passed to FAISS.as_retriever()
            Common options:
            - search_type: "similarity" (default) or "mmr"
            - search_kwargs: {"k": 5, "fetch_k": 20} for MMR

        Returns
        -------
        VectorStoreRetriever
            LangChain retriever object

        Examples
        --------
        >>> # Standard similarity search
        >>> retriever = faiss_tool.as_retriever(search_kwargs={"k": 5})
        >>> docs = retriever.get_relevant_documents("bias detection")
        >>>
        >>> # MMR for diverse results
        >>> retriever = faiss_tool.as_retriever(
        ...     search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
        ... )
        >>> diverse_docs = retriever.get_relevant_documents("AI risks")

        Notes
        -----
        MMR (Maximal Marginal Relevance) retrieves diverse documents by
        balancing relevance with diversity. Use fetch_k > k to enable.
        """
        return self.vectorstore.as_retriever(**kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embeddings: Any,
        metadata: list[dict] | None = None,
        top_k: int = 5,
    ) -> "LangChainFAISSTool":
        """Create FAISS index from documents (for building new indexes).

        This method creates a new FAISS index from a list of LangChain documents.
        Useful for building indexes from scratch or rebuilding existing ones.

        Parameters
        ----------
        documents : list[Document]
            LangChain documents with page_content and metadata
        embeddings : Any
            LangChain embeddings object
        metadata : list[dict] | None, optional
            Custom metadata (if None, uses doc.metadata)
        top_k : int, optional
            Default number of results, by default 5

        Returns
        -------
        LangChainFAISSTool
            New FAISS tool instance

        Examples
        --------
        >>> from langchain_core.documents import Document
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>>
        >>> docs = [
        ...     Document(page_content="Risk 1 text", metadata={"id": "R1"}),
        ...     Document(page_content="Risk 2 text", metadata={"id": "R2"}),
        ... ]
        >>> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        >>> faiss_tool = LangChainFAISSTool.from_documents(docs, embeddings)
        """
        try:
            # Create FAISS index from documents
            vectorstore = FAISS.from_documents(documents, embeddings)

            # Extract metadata
            if metadata is None:
                metadata = [doc.metadata for doc in documents]

            logger.info(
                f"Created LangChain FAISS index from {len(documents)} documents",
            )

            return cls(
                vectorstore=vectorstore,
                metadata=metadata,
                top_k=top_k,
                index_path=Path("from_documents.faiss"),  # Placeholder path
            )

        except Exception as e:
            raise FAISSError(
                f"Failed to create FAISS index from documents: {e}",
                context={
                    "num_documents": len(documents),
                    "error_type": type(e).__name__,
                },
            ) from e

    def save_local(self, folder_path: Path, index_name: str) -> None:
        """Save FAISS index to disk using LangChain's persistence.

        Parameters
        ----------
        folder_path : Path
            Directory to save index files
        index_name : str
            Name of the index (e.g., "risks", "rmf")

        Raises
        ------
        FAISSError
            If saving fails

        Examples
        --------
        >>> faiss_tool.save_local(Path("data/indexes"), "risks")
        # Creates: data/indexes/risks/index.faiss and index.pkl
        #          data/indexes/risks_meta.json
        """
        try:
            # Create directory if needed
            index_dir = folder_path / index_name
            index_dir.mkdir(parents=True, exist_ok=True)

            # Save LangChain FAISS index
            self.vectorstore.save_local(str(index_dir))

            # Save metadata separately
            metadata_path = folder_path / f"{index_name}_meta.json"
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(
                (
                    f"Saved LangChain FAISS index: {index_name} "
                    f"({len(self.metadata)} docs)"
                ),
            )

        except Exception as e:
            raise FAISSError(
                f"Failed to save FAISS index: {e}",
                context={
                    "folder_path": str(folder_path),
                    "index_name": index_name,
                    "error_type": type(e).__name__,
                },
            ) from e
