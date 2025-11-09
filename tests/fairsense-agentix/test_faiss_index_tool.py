"""Tests for FAISSIndexTool."""

import json

import faiss
import numpy as np
import pytest

from fairsense_agentix.tools.embeddings import SentenceTransformerEmbedder
from fairsense_agentix.tools.exceptions import FAISSError
from fairsense_agentix.tools.faiss_index import FAISSIndexTool
from fairsense_agentix.tools.interfaces import (
    FAISSIndexTool as FAISSIndexToolProtocol,
)


@pytest.fixture
def embedder():
    """Create embedder for tests."""
    return SentenceTransformerEmbedder(
        model_name="all-MiniLM-L6-v2",
        dimension=384,
    )


@pytest.fixture
def tiny_faiss_index(tmp_path, embedder):
    """Create a tiny FAISS index for testing.

    Returns
    -------
    tuple[Path, Path, list[dict]]
        (index_path, metadata_path, metadata)
    """
    # Create sample documents
    documents = [
        {
            "id": "RISK001",
            "text": "Algorithmic bias in AI systems can lead to unfair outcomes",
            "severity": "HIGH",
        },
        {
            "id": "RISK002",
            "text": "Data privacy violations through unauthorized access",
            "severity": "CRITICAL",
        },
        {
            "id": "RISK003",
            "text": "Model robustness issues causing incorrect predictions",
            "severity": "MEDIUM",
        },
    ]

    # Generate embeddings
    vectors = np.array([embedder.encode(doc["text"]) for doc in documents])

    # Build FAISS index (IndexFlatIP for inner product / cosine similarity)
    dimension = embedder.dimension
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    # Save index
    index_path = tmp_path / "test_index.faiss"
    faiss.write_index(index, str(index_path))

    # Save metadata
    metadata_path = tmp_path / "test_index_meta.json"
    with open(metadata_path, "w") as f:
        json.dump(documents, f)

    return index_path, metadata_path, documents


class TestFAISSIndexToolConstruction:
    """Test FAISS index tool initialization."""

    def test_successful_initialization(self, tiny_faiss_index, embedder):
        """Test tool initializes with valid index."""
        index_path, metadata_path, _ = tiny_faiss_index

        tool = FAISSIndexTool(
            index_path=index_path,
            embedder=embedder,
            top_k=5,
            metadata_path=metadata_path,
        )

        assert tool.index is not None
        assert tool.metadata is not None
        assert len(tool.metadata) == 3

    def test_metadata_path_auto_inference(self, tmp_path, embedder):
        """Test metadata path is auto-inferred from index path."""
        # Create minimal index
        index = faiss.IndexFlatIP(384)
        index.add(np.random.rand(1, 384).astype("float32"))

        index_path = tmp_path / "myindex.faiss"
        faiss.write_index(index, str(index_path))

        # Create metadata with inferred name
        metadata_path = tmp_path / "myindex_meta.json"
        with open(metadata_path, "w") as f:
            json.dump([{"id": "DOC001", "text": "test"}], f)

        # Initialize without explicit metadata_path
        tool = FAISSIndexTool(index_path=index_path, embedder=embedder)

        assert tool.metadata_path == metadata_path
        assert len(tool.metadata) == 1

    def test_missing_index_file_raises_error(self, tmp_path, embedder):
        """Test initialization fails when index file missing."""
        index_path = tmp_path / "nonexistent.faiss"

        with pytest.raises(FAISSError) as exc_info:
            FAISSIndexTool(index_path=index_path, embedder=embedder)

        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent.faiss" in str(exc_info.value)

    def test_missing_metadata_file_raises_error(self, tmp_path, embedder):
        """Test initialization fails when metadata file missing."""
        # Create index but no metadata
        index = faiss.IndexFlatIP(384)
        index.add(np.random.rand(1, 384).astype("float32"))

        index_path = tmp_path / "test.faiss"
        faiss.write_index(index, str(index_path))

        with pytest.raises(FAISSError) as exc_info:
            FAISSIndexTool(index_path=index_path, embedder=embedder)

        assert "metadata" in str(exc_info.value).lower()
        assert "not found" in str(exc_info.value).lower()

    def test_metadata_size_mismatch_raises_error(self, tmp_path, embedder):
        """Test initialization fails when metadata count != index size."""
        # Create index with 2 vectors
        index = faiss.IndexFlatIP(384)
        index.add(np.random.rand(2, 384).astype("float32"))

        index_path = tmp_path / "test.faiss"
        faiss.write_index(index, str(index_path))

        # Create metadata with only 1 document (mismatch!)
        metadata_path = tmp_path / "test_meta.json"
        with open(metadata_path, "w") as f:
            json.dump([{"id": "DOC001"}], f)

        with pytest.raises(FAISSError) as exc_info:
            FAISSIndexTool(index_path=index_path, embedder=embedder)

        assert "doesn't match" in str(exc_info.value).lower()


class TestProtocolSatisfaction:
    """Test that tool satisfies FAISSIndexTool protocol."""

    def test_satisfies_faiss_tool_protocol(self, tiny_faiss_index, embedder):
        """Test tool is recognized as FAISSIndexTool."""
        index_path, metadata_path, _ = tiny_faiss_index

        tool = FAISSIndexTool(
            index_path=index_path,
            embedder=embedder,
            metadata_path=metadata_path,
        )

        assert isinstance(tool, FAISSIndexToolProtocol)

    def test_has_search_method(self, tiny_faiss_index, embedder):
        """Test tool has search method."""
        index_path, metadata_path, _ = tiny_faiss_index

        tool = FAISSIndexTool(
            index_path=index_path,
            embedder=embedder,
            metadata_path=metadata_path,
        )

        assert hasattr(tool, "search")
        assert callable(tool.search)


class TestSearch:
    """Test search functionality."""

    @pytest.fixture
    def tool(self, tiny_faiss_index, embedder):
        """Create tool for tests."""
        index_path, metadata_path, _ = tiny_faiss_index
        return FAISSIndexTool(
            index_path=index_path,
            embedder=embedder,
            top_k=3,
            metadata_path=metadata_path,
        )

    def test_search_with_text_query(self, tool):
        """Test searching with text query."""
        results = tool.search_by_text("algorithmic bias in machine learning")

        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 3  # top_k

    def test_search_with_vector_query(self, tool, embedder):
        """Test searching with pre-computed vector."""
        query_vector = embedder.encode("data privacy violation")
        results = tool.search(query_vector.tolist())

        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_returns_correct_structure(self, tool):
        """Test search results have expected structure."""
        results = tool.search_by_text("bias detection", top_k=2)

        assert len(results) <= 2

        for result in results:
            assert "id" in result
            assert "text" in result
            assert "severity" in result
            assert "score" in result  # Added by search
            assert "rank" in result  # Added by search

    def test_search_results_are_ranked(self, tool):
        """Test results are ranked by score."""
        results = tool.search_by_text("algorithmic bias", top_k=3)

        # Check ranks are sequential
        for i, result in enumerate(results):
            assert result["rank"] == i

        # Check scores are descending (higher = more similar)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]

    def test_search_top_k_override(self, tool):
        """Test top_k parameter override."""
        # Tool default is 3, but override with 1
        results = tool.search_by_text("privacy", top_k=1)

        assert len(results) == 1

    def test_most_relevant_result_first(self, tool):
        """Test most relevant document is ranked first."""
        # Query specifically about bias
        results = tool.search_by_text("algorithmic bias unfair outcomes")

        # First result should be RISK001 (bias-related)
        assert results[0]["id"] == "RISK001"
        assert "bias" in results[0]["text"].lower()

    def test_search_empty_query_raises_error(self, tool):
        """Test searching with empty query raises error."""
        with pytest.raises(FAISSError) as exc_info:
            tool.search_by_text("")

        assert "empty" in str(exc_info.value).lower()

    def test_search_dimension_mismatch_raises_error(self, tool):
        """Test searching with wrong dimension raises error."""
        # Create vector with wrong dimension
        wrong_vector = [0.1] * 512  # Should be 384

        with pytest.raises(FAISSError) as exc_info:
            tool.search(wrong_vector)

        assert "dimension mismatch" in str(exc_info.value).lower()


class TestSemanticSearch:
    """Test semantic search quality."""

    @pytest.fixture
    def tool(self, tiny_faiss_index, embedder):
        """Create tool for tests."""
        index_path, metadata_path, _ = tiny_faiss_index
        return FAISSIndexTool(
            index_path=index_path,
            embedder=embedder,
            top_k=3,
            metadata_path=metadata_path,
        )

    def test_semantic_similarity_retrieval(self, tool):
        """Test semantically similar queries retrieve relevant docs."""
        # Query about privacy
        results = tool.search_by_text("unauthorized data access and privacy concerns")

        # Top result should be RISK002 (privacy-related)
        assert results[0]["id"] == "RISK002"
        assert "privacy" in results[0]["text"].lower()

    def test_different_queries_return_different_top_results(self, tool):
        """Test different queries return different top results."""
        bias_results = tool.search_by_text("algorithmic fairness and bias", top_k=1)
        privacy_results = tool.search_by_text("data privacy violations", top_k=1)

        # Should retrieve different documents
        assert bias_results[0]["id"] != privacy_results[0]["id"]
