"""Smoke tests for Phase 5.2 LangChain LLM integration.

These tests verify the core integration without requiring real API keys.
They use fake tools to validate:
- Registry instantiation with LangChain models
- Protocol compliance (LLMTool interface)
- Adapter functionality
- Graph integration (zero-code-change verification)

Run with:
    uv run pytest tests/smoke_test_phase5_2.py -v
"""

import pytest

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.tools.interfaces import LLMTool
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput
from fairsense_agentix.tools.registry import (
    create_tool_registry,
    get_tool_registry,
    reset_tool_registry,
)


class TestPhase52Integration:
    """Smoke tests for Phase 5.2 LangChain integration."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_registry_instantiation_with_fake_provider(self):
        """Test 1: Verify registry can create LLM with fake provider.

        This validates:
        - Settings loading from environment
        - Registry dependency injection
        - Fake tool instantiation
        """
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        assert registry.llm is not None, "Registry should have LLM tool"
        assert hasattr(registry.llm, "predict"), "LLM should have predict method"
        assert hasattr(registry.llm, "get_token_count"), (
            "LLM should have get_token_count method"
        )

    def test_llm_tool_protocol_compliance(self):
        """Test 2: Verify adapter satisfies LLMTool protocol.

        This validates:
        - Protocol interface implementation
        - Structural subtyping (duck typing with type checking)
        - Graph compatibility
        """
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        # Protocol compliance check (runtime_checkable)
        assert isinstance(registry.llm, LLMTool), (
            "LLM tool must satisfy LLMTool protocol"
        )

    def test_fake_llm_prediction(self):
        """Test 3: Verify predict() method works with fake provider.

        This validates:
        - Full call path: graph → registry → adapter → fake LLM
        - Parameter passing (prompt, temperature, etc.)
        - Return value structure
        """
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        # Call predict with various parameters
        result = registry.llm.predict("Test prompt for bias analysis")

        # Fake LLM returns a string
        assert isinstance(result, str), "Fake LLM should return string"
        assert len(result) > 0, "Result should not be empty"

    def test_fake_llm_with_parameters(self):
        """Test predict() with custom parameters.

        This validates:
        - Parameter override mechanism
        - Default parameter resolution
        """
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        # Call with custom parameters
        result = registry.llm.predict(
            prompt="Analyze this text",
            temperature=0.5,
            max_tokens=500,
            timeout=30,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_token_counting(self):
        """Test get_token_count() method.

        This validates:
        - Token counting utility
        - Protocol method implementation
        """
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        token_count = registry.llm.get_token_count("Hello world, this is a test.")

        assert isinstance(token_count, int), "Token count should be integer"
        assert token_count > 0, "Token count should be positive"

    def test_graph_integration_bias_text(self):
        """Test 4: Verify bias_text_graph works unchanged (CRITICAL TEST).

        This is the ULTIMATE validation:
        - If this passes, our adapter is correct
        - Zero code changes in graph
        - Full integration working end-to-end
        """
        from fairsense_agentix.graphs.bias_text_graph import (  # noqa: PLC0415
            create_bias_text_graph,
        )

        # Create graph (should use registry internally)
        graph = create_bias_text_graph()

        # Invoke with sample input
        result = graph.invoke(
            {
                "text": "Sample job posting: Looking for a young, energetic salesman to join our team.",
                "options": {"temperature": 0.3},
            }
        )

        # Verify graph produced expected output structure
        assert "bias_analysis" in result, "Graph should produce bias_analysis"
        assert isinstance(result["bias_analysis"], BiasAnalysisOutput), (
            "bias_analysis should be structured BiasAnalysisOutput"
        )

    def test_graph_integration_bias_image(self):
        """Verify bias_image_graph works unchanged.

        This validates:
        - Image workflow integration
        - Multi-tool orchestration (OCR, Caption, LLM)
        """
        from fairsense_agentix.graphs.bias_image_graph import (  # noqa: PLC0415
            create_bias_image_graph,
        )

        graph = create_bias_image_graph()

        # Invoke with sample input (fake image bytes)
        result = graph.invoke(
            {
                "image_bytes": b"fake_image_data",
                "options": {"validate_image_bytes": False},
            }
        )

        # Verify output structure
        assert "bias_analysis" in result
        assert "ocr_text" in result or "caption_text" in result

    def test_graph_integration_risk_graph(self):
        """Verify risk_graph works unchanged.

        This validates:
        - Embedding + FAISS integration
        - Risk assessment workflow
        """
        from fairsense_agentix.graphs.risk_graph import (  # noqa: PLC0415
            create_risk_graph,
        )

        graph = create_risk_graph()

        # Invoke with sample input
        result = graph.invoke(
            {
                "scenario": "AI system making loan approval decisions",
                "options": {},
            }
        )

        # Verify output structure
        assert "risks" in result or "risk_table" in result

    def test_registry_singleton_behavior(self):
        """Verify get_tool_registry() singleton pattern.

        This validates:
        - get_tool_registry() returns same instance across calls
        - Tools are reused (no recreation)
        - Singleton persists across multiple calls
        """
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        # Same instance
        assert registry1 is registry2, "Registry should be singleton"
        assert registry1.llm is registry2.llm, "Tools should be reused"

        # Third call also returns same instance
        registry3 = get_tool_registry()
        assert registry1 is registry3, "Singleton should persist"

    def test_registry_creation_with_custom_settings(self):
        """Verify create_tool_registry() works with custom settings.

        This validates:
        - create_tool_registry() accepts settings parameter
        - Settings are properly applied to created tools
        - Each call creates a new instance (not singleton)
        """
        from fairsense_agentix.tools.fake import FakeLLMTool  # noqa: PLC0415

        settings = Settings(llm_provider="fake")

        # create_tool_registry() accepts settings and creates new instances
        registry1 = create_tool_registry(settings)
        registry2 = create_tool_registry(settings)

        # Different instances (create_tool_registry always creates new)
        assert registry1 is not registry2, (
            "create_tool_registry should create new instances"
        )

        # But both should have correct tool types
        assert isinstance(registry1.llm, FakeLLMTool), (
            "Registry 1 should have FakeLLMTool"
        )
        assert isinstance(registry2.llm, FakeLLMTool), (
            "Registry 2 should have FakeLLMTool"
        )

        # Both should work correctly
        result1 = registry1.llm.predict("Test prompt")
        result2 = registry2.llm.predict("Test prompt")
        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_multiple_predict_calls(self):
        """Test multiple predict calls work correctly.

        This validates:
        - Stateless adapter behavior
        - No state leakage between calls
        """
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        # Multiple calls
        result1 = registry.llm.predict("First prompt")
        result2 = registry.llm.predict("Second prompt")
        result3 = registry.llm.predict("Third prompt")

        # All should succeed
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str)

    def test_empty_prompt_handling(self):
        """Test error handling for invalid inputs.

        This validates:
        - Input validation
        - Clear error messages
        """
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        # Empty prompt should raise error
        with pytest.raises(Exception) as exc_info:
            registry.llm.predict("")

        # Error should mention prompt
        assert (
            "prompt" in str(exc_info.value).lower()
            or "empty" in str(exc_info.value).lower()
        )


class TestRegistryConfiguration:
    """Test registry configuration and provider selection."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_fake_provider_configuration(self):
        """Test fake provider configuration."""
        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        # Should get FakeLLMTool
        from fairsense_agentix.tools.fake import FakeLLMTool  # noqa: PLC0415

        assert isinstance(registry.llm, FakeLLMTool)

    def test_provider_validation(self):
        """Test invalid provider raises clear error from Pydantic validation.

        This validates:
        - Settings validates llm_provider against allowed Literal values
        - Error message is clear and mentions validation issue
        """
        # Pydantic should reject invalid provider at Settings creation
        with pytest.raises(Exception) as exc_info:
            Settings(llm_provider="invalid_provider")

        # Error should mention validation issue
        error_msg = str(exc_info.value).lower()
        assert (
            "validation" in error_msg
            or "literal" in error_msg
            or "invalid" in error_msg
        )


@pytest.mark.integration_test
class TestRealProviderConfiguration:
    """Integration tests for real LLM providers (requires API keys).

    These tests are marked as integration_test and skipped by default.
    Run with: pytest -m integration_test
    """

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_openai_provider_requires_api_key(self):
        """Test OpenAI provider validation."""
        # Without API key, should raise error
        with pytest.raises(Exception) as exc_info:
            Settings(llm_provider="openai")

        # Pydantic validation should catch missing API key
        error_msg = str(exc_info.value)
        assert "api_key" in error_msg.lower()

    def test_anthropic_provider_requires_api_key(self):
        """Test Anthropic provider validation."""
        # Without API key, should raise error
        with pytest.raises(Exception) as exc_info:
            Settings(llm_provider="anthropic")

        # Pydantic validation should catch missing API key
        error_msg = str(exc_info.value)
        assert "api_key" in error_msg.lower()
