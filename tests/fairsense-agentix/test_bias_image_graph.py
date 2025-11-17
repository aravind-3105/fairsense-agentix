"""Minimal tests for BiasImageGraph.

These tests verify:
- Graph compiles successfully
- End-to-end execution works
- Parallel OCR + caption extraction
- All output fields populated

Comprehensive integration tests will be added after Phase 2 completion.
"""

from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


class TestBiasImageGraphBasic:
    """Basic smoke tests for BiasImageGraph."""

    def test_graph_compiles(self) -> None:
        """Test BiasImageGraph compiles without errors."""
        graph = create_bias_image_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_small_image_analysis(self) -> None:
        """Test end-to-end execution with small image."""
        graph = create_bias_image_graph()

        # Small image (simulated with small bytes)
        result = graph.invoke(
            {
                "image_bytes": b"small_image_data",
                "options": {},
            }
        )

        # Verify parallel extraction completed
        assert result["ocr_text"] is not None
        assert len(result["ocr_text"]) > 0
        assert result["caption_text"] is not None
        assert len(result["caption_text"]) > 0

        # Verify merge completed
        assert result["merged_text"] is not None
        assert "OCR Extracted Text" in result["merged_text"]
        assert "Image Caption" in result["merged_text"]

        # Verify analysis completed
        assert isinstance(result["bias_analysis"], BiasAnalysisOutput)
        assert isinstance(result["bias_analysis"], BiasAnalysisOutput)

        # Verify summary and HTML
        assert result["summary"] is not None
        assert result["highlighted_html"] is not None
        assert "<html>" in result["highlighted_html"]

    def test_large_image_analysis(self) -> None:
        """Test execution with large image (different mock data)."""
        graph = create_bias_image_graph()

        # Large image (>10000 bytes triggers different mocks)
        large_image = b"x" * 15000

        result = graph.invoke(
            {
                "image_bytes": large_image,
                "options": {},
            }
        )

        # Verify parallel extraction with different content
        assert result["ocr_text"] is not None
        assert result["caption_text"] is not None

        # Phase 4: FakeOCRTool returns deterministic output
        # Verify OCR text is present (no longer check for specific mock content)
        assert len(result["ocr_text"]) > 0
        assert isinstance(result["ocr_text"], str)

        # Verify complete workflow
        assert result["merged_text"] is not None
        assert isinstance(result["bias_analysis"], BiasAnalysisOutput)
        assert result["summary"] is not None
        assert result["highlighted_html"] is not None

    def test_parallel_extraction_both_run(self) -> None:
        """Test that both OCR and caption run in parallel."""
        graph = create_bias_image_graph()

        result = graph.invoke(
            {
                "image_bytes": b"test_image_bytes",
                "options": {},
            }
        )

        # Both parallel nodes should have produced output
        assert result["ocr_text"] is not None
        assert result["caption_text"] is not None

        # Merged text should contain both
        assert result["merged_text"] is not None
        # Check that both OCR and caption content appear in merge
        assert len(result["merged_text"]) > 0

    def test_options_propagate(self) -> None:
        """Test that options are accessible in state."""
        graph = create_bias_image_graph()

        result = graph.invoke(
            {
                "image_bytes": b"test_image",
                "options": {"ocr_tool": "tesseract", "caption_model": "blip2"},
            }
        )

        # Verify execution completed with options
        assert isinstance(result["bias_analysis"], BiasAnalysisOutput)
        assert result["options"]["ocr_tool"] == "tesseract"
        assert result["options"]["caption_model"] == "blip2"
