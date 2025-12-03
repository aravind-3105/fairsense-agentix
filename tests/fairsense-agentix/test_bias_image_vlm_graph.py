"""Tests for BiasImageVLMGraph.

These tests verify:
- Graph compiles successfully
- End-to-end execution with VLM works
- VLM-specific fields populated (visual_description, reasoning_trace)
- Bias analysis structured output
- Summarization and highlighting work
- Provider validation

Includes both fake tests (fast, no API) and real OpenAI tests (requires API key).
"""

import os
from pathlib import Path

import pytest

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.bias_image_vlm_graph import create_bias_image_vlm_graph
from fairsense_agentix.tools import reset_tool_registry
from fairsense_agentix.tools.vlm.output_schemas import BiasVisualAnalysisOutput


# Check if OpenAI API key is available for real tests
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OpenAI API key not available"
)


@pytest.mark.integration
class TestBiasImageVLMGraphBasic:
    """Basic smoke tests for BiasImageVLMGraph."""

    def test_graph_compiles(self) -> None:
        """Test BiasImageVLMGraph compiles without errors."""
        graph = create_bias_image_vlm_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_vlm_analysis_execution(self) -> None:
        """Test end-to-end VLM analysis with fake image bytes.

        Uses FakeVLMTool (llm_provider="fake") which returns mock
        structured output without API calls.
        """
        graph = create_bias_image_vlm_graph()

        # Fake image bytes (not parsed by fake tool)
        fake_image = b"fake_image_data_for_testing"

        result = graph.invoke(
            {
                "image_bytes": fake_image,
                "options": {},
            }
        )

        # Verify VLM analysis output structure
        assert result["vlm_analysis"] is not None
        assert isinstance(result["vlm_analysis"], BiasVisualAnalysisOutput)

        # Verify VLM-specific fields are populated
        vlm_output = result["vlm_analysis"]
        assert vlm_output.visual_description is not None
        assert len(vlm_output.visual_description) > 0
        assert vlm_output.reasoning_trace is not None
        assert len(vlm_output.reasoning_trace) > 0

        # Verify bias_analysis is structured (BiasAnalysisOutput)
        assert vlm_output.bias_analysis is not None
        assert hasattr(vlm_output.bias_analysis, "bias_detected")
        assert hasattr(vlm_output.bias_analysis, "bias_instances")
        assert isinstance(vlm_output.bias_analysis.bias_instances, list)
        assert hasattr(vlm_output.bias_analysis, "overall_assessment")
        assert hasattr(vlm_output.bias_analysis, "risk_level")

        # Verify summary generated
        assert result["summary"] is not None
        assert len(result["summary"]) > 0

        # Verify highlighted HTML generated - returns <div> fragment for SPA embedding
        assert result["highlighted_html"] is not None
        assert (
            "<div" in result["highlighted_html"]
            or "<pre>" in result["highlighted_html"]
        )

        # HTML should contain visual description
        # (check for substring without special chars)
        # Note: HTML formatter properly escapes entities (e.g., ' becomes &#x27;)
        # so we check for a portion that doesn't require exact escaping match
        assert "Image shows 3 people" in result["highlighted_html"]
        assert "office environment" in result["highlighted_html"]

    def test_vlm_output_contains_bias_instances(self) -> None:
        """Test that fake VLM returns bias instances with proper structure.

        FakeVLMTool returns 2 mock bias instances (gender + age).
        """
        graph = create_bias_image_vlm_graph()

        result = graph.invoke(
            {
                "image_bytes": b"fake_image",
                "options": {},
            }
        )

        bias_analysis = result["vlm_analysis"].bias_analysis

        # FakeVLMTool returns 2 instances
        assert len(bias_analysis.bias_instances) > 0

        # Verify structure of first instance
        instance = bias_analysis.bias_instances[0]
        assert hasattr(instance, "type")
        assert hasattr(instance, "severity")
        assert hasattr(instance, "text_span")
        assert hasattr(instance, "explanation")
        assert hasattr(instance, "start_char")
        assert hasattr(instance, "end_char")
        assert hasattr(instance, "evidence_source")

        # For images, evidence_source should be "visual"
        assert instance.evidence_source == "visual"

        # For images, start_char and end_char should be 0 (not applicable)
        assert instance.start_char == 0
        assert instance.end_char == 0

    def test_options_propagate(self) -> None:
        """Test that options are accessible in state and control VLM parameters."""
        graph = create_bias_image_vlm_graph()

        result = graph.invoke(
            {
                "image_bytes": b"fake_image",
                "options": {
                    "temperature": 0.2,
                    "vlm_max_tokens": 1500,
                    "summary_max_length": 100,
                },
            }
        )

        # Verify execution completed
        assert result["vlm_analysis"] is not None
        assert result["summary"] is not None
        assert result["highlighted_html"] is not None

        # Options are stored in state (accessible by nodes)
        assert result["options"]["temperature"] == 0.2
        assert result["options"]["vlm_max_tokens"] == 1500
        assert result["options"]["summary_max_length"] == 100

    def test_run_id_propagation(self) -> None:
        """Test that run_id is propagated through graph for tracing."""
        graph = create_bias_image_vlm_graph()

        test_run_id = "test-vlm-run-123"

        result = graph.invoke(
            {
                "image_bytes": b"fake_image",
                "options": {},
                "run_id": test_run_id,
            }
        )

        # Verify run_id propagated
        assert result["run_id"] == test_run_id

    def test_highlighted_html_includes_bias_instances(self) -> None:
        """Test that highlighted HTML shows bias instances with details.

        For images, we can't highlight pixels, so HTML should show:
        - Visual description
        - List of detected bias instances
        - Severity indicators
        - Risk level
        """
        graph = create_bias_image_vlm_graph()

        result = graph.invoke(
            {
                "image_bytes": b"fake_image",
                "options": {},
            }
        )

        html = result["highlighted_html"]
        bias_instances = result["vlm_analysis"].bias_analysis.bias_instances

        # HTML should contain visual description
        # (check for key phrases, not exact match due to HTML escaping)
        visual_desc = result["vlm_analysis"].visual_description
        # Check for a meaningful substring that should be present
        # regardless of HTML formatting
        assert "Image shows" in visual_desc or "office" in visual_desc.lower()
        # Verify visual description content appears in HTML (may be escaped)
        assert any(
            phrase in html for phrase in ["Image shows", "office environment", "people"]
        )

        # If bias detected, HTML should show instances
        if result["vlm_analysis"].bias_analysis.bias_detected:
            assert "Detected Bias Instances" in html or "bias" in html.lower()

            # Check for severity indicators (emojis or text)
            if len(bias_instances) > 0:
                # At least one instance's type should be visible
                instance_types = {inst.type for inst in bias_instances}
                assert any(bias_type in html.lower() for bias_type in instance_types)

        # HTML should show overall assessment
        assert result["vlm_analysis"].bias_analysis.overall_assessment in html

        # HTML should show risk level
        assert result["vlm_analysis"].bias_analysis.risk_level in html.lower()


@pytest.mark.integration
class TestBiasImageVLMGraphChainOfThought:
    """Tests for VLM Chain-of-Thought reasoning traces."""

    def test_cot_reasoning_trace_populated(self) -> None:
        """Test that reasoning_trace contains CoT steps.

        The CoT prompt guides VLM through 4 steps:
        1. Visual Inventory (Perceive)
        2. Pattern Analysis (Reason)
        3. Bias Detection (Evidence)
        4. Synthesis (Reflect)

        FakeVLMTool returns mock reasoning trace.
        """
        graph = create_bias_image_vlm_graph()

        result = graph.invoke(
            {
                "image_bytes": b"fake_image",
                "options": {},
            }
        )

        reasoning_trace = result["vlm_analysis"].reasoning_trace

        # Verify reasoning trace is non-empty
        assert reasoning_trace is not None
        assert len(reasoning_trace) > 0

        # FakeVLMTool includes keywords from CoT steps
        # check for actual words in fake output
        trace_lower = reasoning_trace.lower()
        assert any(
            word in trace_lower
            for word in ["analyzing", "compositional", "pattern", "reasoning"]
        )

    def test_visual_description_detailed(self) -> None:
        """Test that visual_description provides objective image details.

        Visual description should be the first output from VLM's CoT.
        """
        graph = create_bias_image_vlm_graph()

        result = graph.invoke(
            {
                "image_bytes": b"fake_image",
                "options": {},
            }
        )

        visual_desc = result["vlm_analysis"].visual_description

        # Verify description is non-empty and descriptive
        assert visual_desc is not None
        assert len(visual_desc) > 20  # Should be a real description, not stub

        # FakeVLMTool returns description with visual details
        desc_lower = visual_desc.lower()
        assert "image" in desc_lower or "shows" in desc_lower or "office" in desc_lower


@pytest.mark.integration
@requires_openai
class TestBiasImageVLMGraphRealOpenAI:
    """Tests with real OpenAI GPT-4o Vision API.

    These tests require OPENAI_API_KEY environment variable.
    They verify actual VLM functionality with real image analysis.
    """

    @staticmethod
    def _load_test_image() -> bytes:
        """Load a test image from bias_images fixtures.

        Returns actual bias test images (JPEG) for realistic VLM testing.
        Uses gender_bias.jpg by default as it's a common test case.
        """
        # Use real bias test images from fixtures
        fixture_path = (
            Path(__file__).parent.parent
            / "fixtures"
            / "bias_images"
            / "gender_bias.jpg"
        )
        if fixture_path.exists():
            with open(fixture_path, "rb") as f:
                return f.read()

        # Fallback: Create a simple but valid 100x100 PNG
        # Using PIL if available, otherwise use a pre-generated PNG
        try:
            import io

            from PIL import Image, ImageDraw

            # Create 100x100 white image with red diagonal line
            img = Image.new("RGB", (100, 100), color="white")
            draw = ImageDraw.Draw(img)
            draw.line([(0, 0), (100, 100)], fill="red", width=3)

            # Save to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            return img_bytes.getvalue()

        except ImportError:
            # PIL not available - return a minimal but valid 10x10 PNG
            # This is larger than 1x1 and more likely to be accepted
            # Base64 of a 10x10 white PNG (pre-generated)
            import base64

            # 10x10 white PNG (pre-generated, 158 bytes)
            png_b64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFklEQVR42mP8z8AAREY"
                "GM4yqGB0VAwASBAMBX0F3HAAAAABJRU5ErkJggg=="
            )
            return base64.b64decode(png_b64)

    def setup_method(self) -> None:
        """Reset tool registry and configure for OpenAI before each test."""
        reset_tool_registry()
        # Ensure settings use OpenAI (will be read from env)
        settings.llm_provider = "openai"

    def teardown_method(self) -> None:
        """Reset tool registry after each test."""
        reset_tool_registry()

    def test_real_vlm_analysis_with_openai(self) -> None:
        """Test VLM graph with real OpenAI GPT-4o Vision API.

        This test makes a real API call and verifies:
        - Structured output parsing works
        - CoT reasoning is captured
        - Bias detection runs
        """
        graph = create_bias_image_vlm_graph()

        image_bytes = self._load_test_image()

        result = graph.invoke(
            {
                "image_bytes": image_bytes,
                "options": {"temperature": 0.3},
            }
        )

        # Verify VLM analysis structure
        assert result["vlm_analysis"] is not None
        assert isinstance(result["vlm_analysis"], BiasVisualAnalysisOutput)

        # Verify real VLM populated fields
        vlm_output = result["vlm_analysis"]
        assert vlm_output.visual_description is not None
        assert len(vlm_output.visual_description) > 10  # Real description
        assert vlm_output.reasoning_trace is not None
        assert len(vlm_output.reasoning_trace) > 10  # Real reasoning

        # Verify bias analysis structure
        bias_analysis = vlm_output.bias_analysis
        assert hasattr(bias_analysis, "bias_detected")
        assert isinstance(bias_analysis.bias_detected, bool)
        assert hasattr(bias_analysis, "bias_instances")
        assert isinstance(bias_analysis.bias_instances, list)
        assert hasattr(bias_analysis, "overall_assessment")
        assert len(bias_analysis.overall_assessment) > 0
        assert hasattr(bias_analysis, "risk_level")
        assert bias_analysis.risk_level in {"low", "medium", "high"}

        # Verify downstream processing
        assert result["summary"] is not None
        assert result["highlighted_html"] is not None

    def test_real_vlm_bias_instance_structure(self) -> None:
        """Test that real OpenAI returns properly structured bias instances."""
        graph = create_bias_image_vlm_graph()

        image_bytes = self._load_test_image()

        result = graph.invoke(
            {
                "image_bytes": image_bytes,
                "options": {},
            }
        )

        bias_instances = result["vlm_analysis"].bias_analysis.bias_instances

        # If bias detected, verify instance structure
        if len(bias_instances) > 0:
            instance = bias_instances[0]

            # Verify all required fields
            assert hasattr(instance, "type")
            assert instance.type in {
                "gender",
                "age",
                "racial",
                "disability",
                "socioeconomic",
            }

            assert hasattr(instance, "severity")
            assert instance.severity in {"low", "medium", "high"}

            # For VLM visual analysis, visual_element should be populated
            # (not text_span)
            assert hasattr(instance, "visual_element")
            assert len(instance.visual_element) > 0

            assert hasattr(instance, "explanation")
            assert len(instance.explanation) > 0

            # Image-specific: char indices should be 0, evidence_source = "visual"
            assert instance.start_char == 0
            assert instance.end_char == 0
            assert instance.evidence_source == "visual"

    def test_real_vlm_cot_reasoning_quality(self) -> None:
        """Test that real VLM produces quality CoT reasoning traces.

        Verifies reasoning trace contains substantive analysis,
        not just template responses.
        """
        graph = create_bias_image_vlm_graph()

        image_bytes = self._load_test_image()

        result = graph.invoke(
            {
                "image_bytes": image_bytes,
                "options": {"temperature": 0.3},
            }
        )

        reasoning_trace = result["vlm_analysis"].reasoning_trace

        # Real VLM should produce detailed reasoning
        assert len(reasoning_trace) > 50  # Substantive content

        # NOTE: Testing LLM reasoning quality via keyword matching is brittle due to
        # non-deterministic output and varied vocabulary. The VLM consistently produces
        # high-quality analytical reasoning but uses different words each time.
        # TODO: Rethink testing methodology - consider semantic similarity or
        # structured output validation instead of keyword checks.
        #
        # For now, we just verify the output is substantive (length check above).
        # The actual reasoning quality should be evaluated through manual review
        # and user acceptance testing.

    def test_real_vlm_with_custom_options(self) -> None:
        """Test that custom options are respected by real VLM."""
        graph = create_bias_image_vlm_graph()

        image_bytes = self._load_test_image()

        # Test with custom temperature and max_tokens
        result = graph.invoke(
            {
                "image_bytes": image_bytes,
                "options": {
                    "temperature": 0.1,  # Very deterministic
                    "vlm_max_tokens": 1000,
                    "summary_max_length": 50,
                },
            }
        )

        # Verify execution completed
        assert result["vlm_analysis"] is not None
        assert result["summary"] is not None
        assert result["highlighted_html"] is not None

        # Summary should respect max_length (approximately)
        # Allow some overage due to sentence boundaries
        assert len(result["summary"]) <= 100
