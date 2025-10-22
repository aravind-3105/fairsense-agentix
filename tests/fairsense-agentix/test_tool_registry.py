"""Tests for tool registry and protocol satisfaction.

This test module focuses on:
    - Registry construction from settings
    - Protocol satisfaction (tools have correct interfaces)
    - Error handling for invalid configuration
    - System integration readiness

We do NOT extensively test fake tool internals - they're temporary stubs.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.tools import (
    CaptionTool,
    EmbedderTool,
    FAISSIndexTool,
    FormatterTool,
    LLMTool,
    OCRTool,
    PersistenceTool,
    SummarizerTool,
    ToolConfigurationError,
    ToolRegistry,
    create_tool_registry,
    get_tool_registry,
    reset_tool_registry,
)


# ============================================================================
# Registry Construction Tests
# ============================================================================


class TestToolRegistryConstruction:
    """Tests for creating tool registry from settings."""

    def test_create_registry_with_fake_tools(self):
        """Registry should construct successfully with all fake tools."""
        settings = Settings(
            ocr_tool="fake",
            caption_model="fake",
            llm_provider="fake",
        )

        registry = create_tool_registry(settings)

        assert isinstance(registry, ToolRegistry)
        assert registry.ocr is not None
        assert registry.caption is not None
        assert registry.llm is not None
        assert registry.summarizer is not None
        assert registry.embedder is not None
        assert registry.faiss_risks is not None
        assert registry.faiss_rmf is not None
        assert registry.formatter is not None
        assert registry.persistence is not None

    def test_create_registry_without_settings_uses_global(self):
        """Registry should use global settings if none provided."""
        # Should not raise - uses global settings singleton
        registry = create_tool_registry()

        assert isinstance(registry, ToolRegistry)

    def test_registry_raises_on_unknown_ocr_tool(self):
        """Settings should raise validation error for unknown OCR tool."""
        # Settings validates Literal types, so this raises before registry
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                ocr_tool="invalid_tool",  # type: ignore
                caption_model="fake",
                llm_provider="fake",
            )

        assert "ocr_tool" in str(exc_info.value).lower()

    def test_registry_raises_on_unknown_llm_provider(self):
        """Settings should raise validation error for unknown LLM provider."""
        # Settings validates Literal types, so this raises before registry
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                ocr_tool="fake",
                caption_model="fake",
                llm_provider="invalid_provider",  # type: ignore
            )

        assert "llm_provider" in str(exc_info.value).lower()

    def test_registry_raises_on_unimplemented_real_tool(self):
        """Registry should raise clear error for Phase 5 tools."""
        settings = Settings(
            ocr_tool="tesseract",  # Not yet implemented
            caption_model="fake",
            llm_provider="fake",
        )

        with pytest.raises(ToolConfigurationError) as exc_info:
            create_tool_registry(settings)

        error_msg = str(exc_info.value).lower()
        assert "tesseract" in error_msg or "phase 5" in error_msg


# ============================================================================
# Protocol Satisfaction Tests
# ============================================================================


class TestProtocolSatisfaction:
    """Tests that registry tools satisfy their protocols."""

    def test_all_tools_satisfy_protocols(self):
        """All tools from registry should satisfy their respective protocols."""
        registry = create_tool_registry()

        # Check that each tool satisfies its protocol
        assert isinstance(registry.ocr, OCRTool)
        assert isinstance(registry.caption, CaptionTool)
        assert isinstance(registry.llm, LLMTool)
        assert isinstance(registry.summarizer, SummarizerTool)
        assert isinstance(registry.embedder, EmbedderTool)
        assert isinstance(registry.faiss_risks, FAISSIndexTool)
        assert isinstance(registry.faiss_rmf, FAISSIndexTool)
        assert isinstance(registry.formatter, FormatterTool)
        assert isinstance(registry.persistence, PersistenceTool)

    def test_ocr_tool_has_extract_method(self):
        """OCR tool should have extract() method."""
        registry = create_tool_registry()

        # Should not raise AttributeError
        assert hasattr(registry.ocr, "extract")
        assert callable(registry.ocr.extract)

    def test_llm_tool_has_predict_method(self):
        """LLM tool should have predict() method."""
        registry = create_tool_registry()

        assert hasattr(registry.llm, "predict")
        assert callable(registry.llm.predict)

    def test_embedder_tool_has_encode_method(self):
        """Embedder tool should have encode() method."""
        registry = create_tool_registry()

        assert hasattr(registry.embedder, "encode")
        assert callable(registry.embedder.encode)

    def test_embedder_tool_has_dimension_property(self):
        """Embedder tool should have dimension property."""
        registry = create_tool_registry()

        assert hasattr(registry.embedder, "dimension")
        assert isinstance(registry.embedder.dimension, int)
        assert registry.embedder.dimension > 0


# ============================================================================
# Tool Functionality Smoke Tests
# ============================================================================


class TestToolFunctionalitySmokeTests:
    """Basic smoke tests that tools can be called without errors."""

    def test_ocr_tool_can_extract(self):
        """OCR tool extract() should return string."""
        registry = create_tool_registry()

        result = registry.ocr.extract(b"fake_image_data")

        assert isinstance(result, str)

    def test_caption_tool_can_caption(self):
        """Caption tool caption() should return string."""
        registry = create_tool_registry()

        result = registry.caption.caption(b"fake_image_data")

        assert isinstance(result, str)

    def test_llm_tool_can_predict(self):
        """LLM tool predict() should return string."""
        registry = create_tool_registry()

        result = registry.llm.predict("Test prompt")

        assert isinstance(result, str)

    def test_embedder_tool_can_encode(self):
        """Embedder tool encode() should return list of floats."""
        registry = create_tool_registry()

        result = registry.embedder.encode("Test text")

        assert isinstance(result, list)
        assert len(result) == registry.embedder.dimension
        assert all(isinstance(x, float) for x in result)

    def test_faiss_tool_can_search(self):
        """FAISS tool search() should return list of dicts."""
        registry = create_tool_registry()
        query_vector = [0.1] * registry.embedder.dimension

        results = registry.faiss_risks.search(query_vector, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(r, dict) for r in results)

    def test_formatter_tool_can_highlight(self):
        """Formatter tool highlight() should return HTML string."""
        registry = create_tool_registry()

        html = registry.formatter.highlight("text", [], {})

        assert isinstance(html, str)
        assert len(html) > 0

    def test_persistence_tool_can_save_csv(self):
        """Persistence tool save_csv() should return Path."""
        registry = create_tool_registry()

        path = registry.persistence.save_csv([{"a": 1}], "test.csv")

        assert isinstance(path, Path)


# ============================================================================
# Global Registry Singleton Tests
# ============================================================================


class TestGlobalRegistrySingleton:
    """Tests for global registry singleton pattern."""

    def test_get_tool_registry_returns_registry(self):
        """get_tool_registry() should return ToolRegistry instance."""
        reset_tool_registry()  # Clean slate

        registry = get_tool_registry()

        assert isinstance(registry, ToolRegistry)

    def test_get_tool_registry_returns_same_instance(self):
        """get_tool_registry() should return same instance on multiple calls."""
        reset_tool_registry()

        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        assert registry1 is registry2

    def test_reset_tool_registry_forces_recreation(self):
        """reset_tool_registry() should force new instance on next get."""
        registry1 = get_tool_registry()
        reset_tool_registry()
        registry2 = get_tool_registry()

        # Should be different instances
        assert registry1 is not registry2


# ============================================================================
# Registry with Different Configurations
# ============================================================================


class TestRegistryConfigurations:
    """Tests for registry with various configuration combinations."""

    @pytest.mark.parametrize(
        "ocr_tool,caption_model,llm_provider",
        [
            ("fake", "fake", "fake"),
            # Add real tool combinations in Phase 5:
            # ("tesseract", "blip2", "openai"),
            # ("paddleocr", "llava", "anthropic"),
        ],
    )
    def test_registry_with_valid_configurations(
        self, ocr_tool, caption_model, llm_provider
    ):
        """Registry should work with valid tool combinations."""
        settings = Settings(
            ocr_tool=ocr_tool,
            caption_model=caption_model,
            llm_provider=llm_provider,
        )

        registry = create_tool_registry(settings)

        assert registry is not None
        assert isinstance(registry.ocr, OCRTool)
        assert isinstance(registry.caption, CaptionTool)
        assert isinstance(registry.llm, LLMTool)


# ============================================================================
# Integration Readiness Tests
# ============================================================================


class TestIntegrationReadiness:
    """Tests that registry is ready for graph integration."""

    def test_registry_tools_have_no_required_positional_args(self):
        """Tools should not require positional args (for easy mocking)."""
        registry = create_tool_registry()

        # These should work with minimal required args
        registry.ocr.extract(b"image")
        registry.caption.caption(b"image")
        registry.llm.predict("prompt")
        registry.summarizer.summarize("text")
        registry.embedder.encode("text")
        registry.faiss_risks.search([0.1] * 384)
        registry.formatter.highlight("text", [], {})
        registry.persistence.save_csv([{"a": 1}], "file.csv")

    def test_registry_can_be_passed_to_functions(self):
        """Registry should be passable to graph nodes."""

        def mock_graph_node(registry: ToolRegistry) -> str:
            """Simulate a graph node using registry."""
            return registry.ocr.extract(b"image")  # text

        registry = create_tool_registry()
        result = mock_graph_node(registry)

        assert isinstance(result, str)

    def test_registry_tools_work_with_options_dict(self):
        """Tools should work with options passed as dict (graph pattern)."""
        registry = create_tool_registry()
        options = {"language": "eng", "confidence_threshold": 0.5}

        # Should not raise (options passed as kwargs)
        result = registry.ocr.extract(b"image", **options)

        assert isinstance(result, str)
