"""End-to-end integration tests for graphs with tool registry.

This test module verifies that all graphs work end-to-end with the tool registry.
Tests focus on complete workflows, not implementation details.

These tests are marked with @pytest.mark.integration_test to run separately in CI.
"""

import pytest

from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph
from fairsense_agentix.graphs.risk_graph import create_risk_graph
from fairsense_agentix.tools import reset_tool_registry


# ============================================================================
# BiasTextGraph Integration Tests
# ============================================================================


@pytest.mark.integration_test
class TestBiasTextGraphIntegration:
    """End-to-end integration tests for BiasTextGraph."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_bias_text_graph_end_to_end(self):
        """BiasTextGraph should work end-to-end: analyze → highlight."""
        graph = create_bias_text_graph()

        result = graph.invoke(
            {
                "text": "Hiring software engineers",
                "options": {},
            }
        )

        # Verify complete pipeline execution
        assert "bias_analysis" in result
        assert isinstance(result["bias_analysis"], str)
        assert len(result["bias_analysis"]) > 0

        # Summary should be None for short text (conditional not triggered)
        assert result.get("summary") is None

        # Verify highlight output
        assert "highlighted_html" in result
        assert "<html>" in result["highlighted_html"]

    def test_bias_text_graph_conditional_summarization(self):
        """BiasTextGraph should conditionally run summarization."""
        graph = create_bias_text_graph()

        # Case 1: Long text (>500 chars) triggers summary automatically
        result_long = graph.invoke({"text": "a" * 600, "options": {}})
        assert result_long["summary"] is not None
        assert isinstance(result_long["summary"], str)

        # Case 2: Short text with explicit option triggers summary
        result_option = graph.invoke(
            {"text": "Short text", "options": {"enable_summary": True}}
        )
        assert result_option["summary"] is not None


# ============================================================================
# BiasImageGraph Integration Tests
# ============================================================================


@pytest.mark.integration_test
class TestBiasImageGraphIntegration:
    """End-to-end integration tests for BiasImageGraph."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_bias_image_graph_end_to_end(self):
        """BiasImageGraph end-to-end test.

        Tests: OCR ∥ caption → merge → analyze → summarize → highlight.
        """
        graph = create_bias_image_graph()

        result = graph.invoke(
            {
                "image_bytes": b"fake_image_data",
                "options": {},
            }
        )

        # Verify parallel extraction outputs (proves both branches executed)
        assert "ocr_text" in result
        assert isinstance(result["ocr_text"], str)
        assert "caption_text" in result
        assert isinstance(result["caption_text"], str)

        # Verify merge output
        assert "merged_text" in result
        assert "OCR Extracted Text" in result["merged_text"]
        assert "Image Caption" in result["merged_text"]

        # Verify downstream pipeline
        assert "bias_analysis" in result
        assert isinstance(result["bias_analysis"], str)
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert "highlighted_html" in result
        assert "<html>" in result["highlighted_html"]


# ============================================================================
# RiskGraph Integration Tests
# ============================================================================


@pytest.mark.integration_test
class TestRiskGraphIntegration:
    """End-to-end integration tests for RiskGraph."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_risk_graph_end_to_end(self):
        """RiskGraph end-to-end test.

        Tests: embed → search risks → search RMF → join → format → export.
        """
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "Deploying facial recognition in public spaces",
                "run_id": "test-run-123",
                "options": {},
            }
        )

        # Verify complete sequential pipeline
        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0

        assert "risks" in result
        assert isinstance(result["risks"], list)
        assert len(result["risks"]) > 0

        assert "rmf_recommendations" in result
        assert isinstance(result["rmf_recommendations"], dict)

        assert "joined_table" in result
        assert isinstance(result["joined_table"], list)
        assert len(result["joined_table"]) > 0

        assert "html_table" in result
        assert "<table>" in result["html_table"]

        assert "csv_path" in result
        assert isinstance(result["csv_path"], str)

    def test_risk_graph_with_top_k_option(self):
        """RiskGraph should respect top_k configuration option."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "AI deployment scenario",
                "run_id": "test-run-456",
                "options": {"top_k": 3},
            }
        )

        # Should return at most 3 risks
        assert len(result["risks"]) <= 3


# ============================================================================
# Cross-Graph Integration Tests
# ============================================================================


@pytest.mark.integration_test
class TestCrossGraphIntegration:
    """End-to-end integration tests across multiple graphs."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_graphs_are_stateless_and_independent(self):
        """All three graphs should execute independently without interference."""
        text_graph = create_bias_text_graph()
        image_graph = create_bias_image_graph()
        risk_graph = create_risk_graph()

        # Execute in random order to prove no shared state
        risk_result = risk_graph.invoke(
            {"scenario_text": "Test scenario", "run_id": "test-r1", "options": {}}
        )
        text_result = text_graph.invoke({"text": "Test text", "options": {}})
        image_result = image_graph.invoke({"image_bytes": b"test_image", "options": {}})

        # All should produce valid outputs
        assert "bias_analysis" in text_result
        assert "bias_analysis" in image_result
        assert "html_table" in risk_result
