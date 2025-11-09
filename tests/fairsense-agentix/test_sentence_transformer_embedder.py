"""Tests for SentenceTransformerEmbedder."""

import numpy as np
import pytest

from fairsense_agentix.tools.embeddings import SentenceTransformerEmbedder
from fairsense_agentix.tools.exceptions import EmbeddingError
from fairsense_agentix.tools.interfaces import EmbedderTool


class TestSentenceTransformerEmbedderConstruction:
    """Test embedder initialization."""

    def test_successful_initialization(self):
        """Test embedder initializes with valid model."""
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
        )

        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.dimension == 384
        assert embedder.model is not None

    def test_dimension_mismatch_raises_error(self):
        """Test initialization fails when dimension doesn't match model."""
        with pytest.raises(EmbeddingError) as exc_info:
            SentenceTransformerEmbedder(
                model_name="all-MiniLM-L6-v2",
                dimension=768,  # Wrong! This model outputs 384-dim
            )

        assert "dimension mismatch" in str(exc_info.value).lower()
        assert "expected 768" in str(exc_info.value)
        assert "got 384" in str(exc_info.value)


class TestProtocolSatisfaction:
    """Test that embedder satisfies EmbedderTool protocol."""

    def test_satisfies_embedder_tool_protocol(self):
        """Test embedder is recognized as EmbedderTool."""
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
        )

        assert isinstance(embedder, EmbedderTool)

    def test_has_encode_method(self):
        """Test embedder has encode method."""
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
        )

        assert hasattr(embedder, "encode")
        assert callable(embedder.encode)

    def test_has_dimension_property(self):
        """Test embedder has dimension property."""
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
        )

        assert hasattr(embedder, "dimension")
        assert embedder.dimension == 384


class TestEncoding:
    """Test text encoding functionality."""

    @pytest.fixture
    def embedder(self):
        """Create embedder for tests."""
        return SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
        )

    def test_encode_returns_numpy_array(self, embedder):
        """Test encode returns numpy array."""
        vector = embedder.encode("sample text")

        assert isinstance(vector, np.ndarray)

    def test_encode_returns_correct_shape(self, embedder):
        """Test encoded vector has correct dimension."""
        vector = embedder.encode("sample text")

        assert vector.shape == (384,)

    def test_encode_consistency(self, embedder):
        """Test encoding same text twice produces same result."""
        text = "AI bias detection and fairness"

        vector1 = embedder.encode(text)
        vector2 = embedder.encode(text)

        np.testing.assert_array_almost_equal(vector1, vector2, decimal=6)

    def test_encode_different_texts_produce_different_vectors(self, embedder):
        """Test different texts produce different embeddings."""
        vector1 = embedder.encode("bias detection")
        vector2 = embedder.encode("privacy violation")

        # Vectors should not be identical
        assert not np.allclose(vector1, vector2)

        # But should be in same space
        assert vector1.shape == vector2.shape

    def test_encode_with_normalization(self):
        """Test normalized embeddings have unit length."""
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            normalize=True,
        )

        vector = embedder.encode("sample text")
        norm = np.linalg.norm(vector)

        # Should be approximately 1.0 (unit vector)
        assert abs(norm - 1.0) < 1e-5

    def test_encode_without_normalization(self):
        """Test unnormalized embeddings may not have unit length."""
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            normalize=False,
        )

        vector = embedder.encode("sample text")
        norm = np.linalg.norm(vector)

        # May not be exactly 1.0
        assert vector.shape == (384,)  # Still correct shape
        assert norm > 0  # But not zero


class TestErrorHandling:
    """Test error handling in embedder."""

    @pytest.fixture
    def embedder(self):
        """Create embedder for tests."""
        return SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
        )

    def test_encode_empty_string_raises_error(self, embedder):
        """Test encoding empty string raises error."""
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.encode("")

        assert "empty text" in str(exc_info.value).lower()

    def test_encode_whitespace_only_raises_error(self, embedder):
        """Test encoding whitespace-only string raises error."""
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.encode("   \n\t  ")

        assert "empty text" in str(exc_info.value).lower()


class TestSemanticSimilarity:
    """Test that embeddings capture semantic meaning."""

    @pytest.fixture
    def embedder(self):
        """Create embedder for tests."""
        return SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            normalize=True,
        )

    def test_similar_texts_have_high_cosine_similarity(self, embedder):
        """Test semantically similar texts have high cosine similarity."""
        text1 = "algorithmic bias in AI systems"
        text2 = "bias and fairness in artificial intelligence"

        vector1 = embedder.encode(text1)
        vector2 = embedder.encode(text2)

        # Cosine similarity (since normalized)
        similarity = np.dot(vector1, vector2)

        # Should be reasonably high (>0.5) for similar concepts
        assert similarity > 0.5

    def test_dissimilar_texts_have_low_cosine_similarity(self, embedder):
        """Test semantically different texts have lower similarity."""
        text1 = "algorithmic bias detection"
        text2 = "quantum computing hardware"

        vector1 = embedder.encode(text1)
        vector2 = embedder.encode(text2)

        similarity = np.dot(vector1, vector2)

        # Should be lower than similar texts (but still positive)
        assert 0 < similarity < 0.7
