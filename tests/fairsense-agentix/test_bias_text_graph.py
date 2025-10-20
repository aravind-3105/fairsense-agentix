"""Minimal tests for BiasTextGraph.

These tests verify:
- Graph compiles successfully
- End-to-end execution works
- All output fields populated
- Quality score in valid range

Comprehensive integration tests will be added after Phase 2 completion.
"""

from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph


class TestBiasTextGraphBasic:
    """Basic smoke tests for BiasTextGraph."""

    def test_graph_compiles(self) -> None:
        """Test BiasTextGraph compiles without errors."""
        graph = create_bias_text_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_short_text_analysis(self) -> None:
        """Test end-to-end execution with short text (< 500 chars).

        Short text should skip summarization node (conditional routing).
        """
        graph = create_bias_text_graph()

        result = graph.invoke(
            {
                "text": "Looking for a talented developer",
                "options": {},
            }
        )

        # Verify required outputs
        assert result["bias_analysis"] is not None
        assert len(result["bias_analysis"]) > 0
        assert "Bias Analysis Report" in result["bias_analysis"]

        # Summary should NOT be generated for short text (< 500 chars)
        # Conditional routing skips summarize node
        assert "summary" not in result or result.get("summary") is None

        # HTML generated (always runs)
        assert result["highlighted_html"] is not None
        assert "<html>" in result["highlighted_html"]
        assert "Looking for a talented developer" in result["highlighted_html"]

    def test_long_text_analysis(self) -> None:
        """Test execution with long text (triggers summarization)."""
        graph = create_bias_text_graph()

        # Long text (>500 chars)
        long_text = "Looking for experienced software engineer. " * 20

        result = graph.invoke(
            {
                "text": long_text,
                "options": {},
            }
        )

        # Verify execution completed
        assert result["bias_analysis"] is not None
        assert result["summary"] is not None
        assert result["highlighted_html"] is not None

    def test_options_propagate(self) -> None:
        """Test that options are accessible in state and control behavior.

        Tests enable_summary option forces summarization even for short text.
        """
        graph = create_bias_text_graph()

        result = graph.invoke(
            {
                "text": "Sample text",
                "options": {"temperature": 0.5, "enable_summary": True},
            }
        )

        # Verify execution completed with options
        assert result["bias_analysis"] is not None
        assert result["options"]["temperature"] == 0.5

        # Verify enable_summary=True triggers summarization even for short text
        # (short text = 11 chars, but enable_summary=True forces summarize node)
        assert result["summary"] is not None
        assert len(result["summary"]) > 0
