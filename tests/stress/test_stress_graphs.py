"""Stress tests for FairSense-AgentiX graphs.

Phase 6.6: Comprehensive stress testing for extreme inputs, edge cases,
boundary conditions, and resource limits.

Run with:
    pytest tests/stress/test_stress_graphs.py -v
"""

import pytest

from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph
from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph
from fairsense_agentix.graphs.risk_graph import create_risk_graph


# ============================================================================
# BiasTextGraph Stress Tests
# ============================================================================


@pytest.mark.stress
class TestBiasTextGraphStress:
    """Stress tests for BiasTextGraph with extreme inputs."""

    def test_very_long_text(self) -> None:
        """Test with 10K+ character text input."""
        graph = create_bias_text_graph()

        # Generate 10K chars of realistic bias-laden text
        base_text = """We're seeking a young rockstar developer who's a cultural fit
        for our brotherhood. Must be a native English speaker with perfect vision
        and hearing. Recent graduates from elite universities preferred. This isn't
        a 9-5 job - we need someone energetic who can handle the demanding pace."""

        # Repeat to create 10K+ chars
        very_long_text = (base_text + " ") * 50  # ~10K characters

        result = graph.invoke({"text": very_long_text, "options": {}})

        # Should complete without errors
        assert result is not None
        assert "bias_analysis" in result
        assert "highlighted_html" in result
        assert len(result["highlighted_html"]) > 0

    def test_empty_text(self) -> None:
        """Test with empty string input."""
        graph = create_bias_text_graph()

        result = graph.invoke({"text": "", "options": {}})

        # Should handle gracefully
        assert result is not None
        assert "bias_analysis" in result
        assert "highlighted_html" in result

    def test_whitespace_only_text(self) -> None:
        """Test with whitespace-only input."""
        graph = create_bias_text_graph()

        result = graph.invoke({"text": "   \n\t\r   ", "options": {}})

        # Should handle gracefully
        assert result is not None
        assert "bias_analysis" in result

    def test_unicode_text(self) -> None:
        """Test with Unicode characters (emoji, special chars)."""
        graph = create_bias_text_graph()

        unicode_text = """Hiring 👨‍💼 Software Engineer 🚀

        Requirements:
        - Young 🎓 recent graduate
        - Native speaker 🗣️ (no accents ❌)
        - Able-bodied ✓ for open office
        - Cultural fit 🤝 with our team

        Join our élite équipe! 日本語 中文 العربية"""

        result = graph.invoke({"text": unicode_text, "options": {}})

        # Should handle Unicode without errors
        assert result is not None
        assert "bias_analysis" in result
        assert "highlighted_html" in result

    def test_special_html_characters(self) -> None:
        """Test with HTML-dangerous characters."""
        graph = create_bias_text_graph()

        dangerous_text = """Seeking <script>alert('xss')</script> developer

        Requirements & Benefits:
        - Salary > $100k < $150k
        - Age "25-35" preferred
        - Must be 'native' speaker
        - Team of "guys" & "bros"
        """

        result = graph.invoke({"text": dangerous_text, "options": {}})

        # Should escape HTML properly
        assert result is not None
        assert "bias_analysis" in result
        assert "<script>" not in result["highlighted_html"]  # Should be escaped

    def test_single_character_text(self) -> None:
        """Test with minimal single-character input."""
        graph = create_bias_text_graph()

        result = graph.invoke({"text": "a", "options": {}})

        # Should handle edge case
        assert result is not None
        assert "bias_analysis" in result

    def test_repeated_characters(self) -> None:
        """Test with pathological repeated characters."""
        graph = create_bias_text_graph()

        result = graph.invoke({"text": "a" * 5000, "options": {}})

        # Should complete without hanging
        assert result is not None
        assert "bias_analysis" in result

    def test_many_newlines(self) -> None:
        """Test with excessive newlines."""
        graph = create_bias_text_graph()

        text_with_newlines = "Hiring\n" * 1000 + "young developers"

        result = graph.invoke({"text": text_with_newlines, "options": {}})

        # Should handle formatting edge case
        assert result is not None
        assert "bias_analysis" in result


# ============================================================================
# BiasImageGraph Stress Tests
# ============================================================================


@pytest.mark.stress
class TestBiasImageGraphStress:
    """Stress tests for BiasImageGraph with extreme inputs."""

    INVALID_IMAGE_REGEX = (
        "Both OCR and caption extraction failed"
        + "|Invalid image data: unable to decode image bytes"
        + "|Image data is empty; provide a valid image file."
    )

    def test_very_large_image(self) -> None:
        """Test with 10MB+ image bytes (invalid format causes expected failure)."""
        graph = create_bias_image_graph()

        # Generate 10MB of fake image data (not a valid image format)
        large_image_bytes = b"fake_image_data" * 700000  # ~10MB

        # Invalid image bytes cause both OCR and caption to fail
        # This is expected behavior - the graph should raise ValueError
        with pytest.raises(ValueError, match=self.INVALID_IMAGE_REGEX):
            graph.invoke({"image_bytes": large_image_bytes, "options": {}})

    def test_empty_image(self) -> None:
        """Test with zero-length image bytes (causes expected failure)."""
        graph = create_bias_image_graph()

        # Empty image causes both OCR and caption to fail
        # This is expected behavior - fail fast on invalid input
        with pytest.raises(ValueError, match=self.INVALID_IMAGE_REGEX):
            graph.invoke({"image_bytes": b"", "options": {}})

    def test_single_byte_image(self) -> None:
        """Test with minimal 1-byte image (invalid format causes expected failure)."""
        graph = create_bias_image_graph()

        # Single byte is not a valid image - both extractors should fail
        # This validates proper error handling for corrupted images
        with pytest.raises(ValueError, match=self.INVALID_IMAGE_REGEX):
            graph.invoke({"image_bytes": b"x", "options": {}})

    def test_null_bytes_image(self) -> None:
        """Test with null bytes (invalid format causes expected failure)."""
        graph = create_bias_image_graph()

        null_image = b"\x00" * 1000

        # Null bytes are not valid image data - should fail gracefully
        # This validates error handling for corrupted binary data
        with pytest.raises(ValueError, match=self.INVALID_IMAGE_REGEX):
            graph.invoke({"image_bytes": null_image, "options": {}})

    def test_binary_data_extremes(self) -> None:
        """Test with extreme binary values (invalid format causes expected failure)."""
        graph = create_bias_image_graph()

        extreme_bytes = bytes(range(256)) * 100  # All possible byte values

        # Random binary data is not a valid image format
        # This validates the system correctly rejects malformed inputs
        with pytest.raises(ValueError, match=self.INVALID_IMAGE_REGEX):
            graph.invoke({"image_bytes": extreme_bytes, "options": {}})


# ============================================================================
# RiskGraph Stress Tests
# ============================================================================


@pytest.mark.stress
class TestRiskGraphStress:
    """Stress tests for RiskGraph with extreme inputs."""

    def test_very_long_scenario(self) -> None:
        """Test with 10K+ character scenario text."""
        graph = create_risk_graph()

        base_scenario = """Deploying a comprehensive AI system across our organization
        that will handle hiring, promotions, resource allocation, and performance
        reviews. The system uses historical data for predictions."""

        # Create 10K+ char scenario
        very_long_scenario = (base_scenario + " ") * 50

        result = graph.invoke({"scenario_text": very_long_scenario, "options": {}})

        # Should complete without errors
        assert result is not None
        assert "embedding" in result
        assert "risks" in result
        assert "html_table" in result
        assert "csv_path" in result

    def test_empty_scenario(self) -> None:
        """Test with empty scenario text (causes expected failure)."""
        graph = create_risk_graph()

        # Empty text cannot be embedded - the embedder correctly rejects it
        # This validates fail-fast behavior for invalid inputs
        # Using broad Exception catch as EmbeddingError is wrapped by LangGraph
        with pytest.raises(Exception):  # noqa: B017
            graph.invoke({"scenario_text": "", "options": {}})

    def test_extreme_top_k(self) -> None:
        """Test with very large top_k parameter."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "AI deployment in healthcare",
                "options": {"top_k": 1000},  # Extreme value
            },
        )

        # Should cap at reasonable limit
        assert result is not None
        assert "risks" in result
        assert len(result["risks"]) <= 100  # Should be capped

    def test_zero_top_k(self) -> None:
        """Test with invalid zero top_k."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "AI deployment",
                "options": {"top_k": 0},  # Invalid
            },
        )

        # Should use default
        assert result is not None
        assert "risks" in result
        assert len(result["risks"]) > 0  # Used default instead

    def test_negative_top_k(self) -> None:
        """Test with invalid negative top_k."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "AI deployment",
                "options": {"top_k": -5},  # Invalid
            },
        )

        # Should use default
        assert result is not None
        assert "risks" in result

    def test_unicode_scenario(self) -> None:
        """Test with Unicode characters in scenario."""
        graph = create_risk_graph()

        unicode_scenario = """Déploiement d'IA 🤖 dans système 医疗保健

        Requirements:
        - Automated decisions 🎯
        - Patient data 📊 analysis
        - Resource allocation 💰
        """

        result = graph.invoke({"scenario_text": unicode_scenario, "options": {}})

        # Should handle international text
        assert result is not None
        assert "embedding" in result
        assert "risks" in result

    def test_special_characters_scenario(self) -> None:
        """Test with special characters that might break CSV/HTML."""
        graph = create_risk_graph()

        special_scenario = """AI deployment with "quotes" and 'apostrophes'
        - Commas, semicolons; colons:
        - Pipes | and & ampersands
        - Brackets <> and [parentheses]
        - Newlines\n\tand\ttabs"""

        result = graph.invoke({"scenario_text": special_scenario, "options": {}})

        # Should escape properly for CSV/HTML
        assert result is not None
        assert "html_table" in result
        assert "csv_path" in result

    def test_single_word_scenario(self) -> None:
        """Test with minimal single-word scenario."""
        graph = create_risk_graph()

        result = graph.invoke({"scenario_text": "AI", "options": {}})

        # Should handle minimal input
        assert result is not None
        assert "embedding" in result
        assert len(result["embedding"]) > 0


# ============================================================================
# OrchestratorGraph Stress Tests
# ============================================================================


@pytest.mark.stress
class TestOrchestratorGraphStress:
    """Stress tests for OrchestratorGraph with extreme routing."""

    def test_ambiguous_input_type(self) -> None:
        """Test with content that could be text or risk."""
        graph = create_orchestrator_graph()

        ambiguous_content = "Bias risk analysis deployment"

        result = graph.invoke(
            {
                "input_type": "text",  # Explicit type
                "content": ambiguous_content,
                "options": {},
            },
        )

        # Should respect explicit type
        assert result is not None
        assert result["final_result"]["status"] == "success"

    def test_very_large_orchestrator_input(self) -> None:
        """Test orchestrator with 10K+ char text."""
        graph = create_orchestrator_graph()

        large_text = "Hiring young developers. " * 500  # ~10K chars

        result = graph.invoke(
            {
                "input_type": "text",
                "content": large_text,
                "options": {},
            },
        )

        # Should complete workflow
        assert result is not None
        assert result["final_result"]["status"] == "success"

    def test_orchestrator_with_large_image(self) -> None:
        """Test orchestrator routing large image (VLM is default as of Phase 6)."""
        graph = create_orchestrator_graph()

        large_image = b"fake_large_image" * 100000  # ~1.5MB

        result = graph.invoke(
            {
                "input_type": "image",
                "content": large_image,
                "options": {},
            },
        )

        # Should route to VLM image workflow (default mode)
        assert result is not None
        assert result["final_result"]["workflow_id"] == "bias_image_vlm"

    def test_orchestrator_with_unicode(self) -> None:
        """Test orchestrator with Unicode content."""
        graph = create_orchestrator_graph()

        unicode_content = "Embauche de développeurs 👨‍💻 jeunes 🎓"

        result = graph.invoke(
            {
                "input_type": "text",
                "content": unicode_content,
                "options": {},
            },
        )

        # Should handle Unicode through entire pipeline
        assert result is not None
        assert result["final_result"]["status"] == "success"


# ============================================================================
# Resource Limit Tests
# ============================================================================


@pytest.mark.stress
class TestResourceLimits:
    """Tests for resource limits and constraints."""

    def test_concurrent_bias_text_execution(self) -> None:
        """Test multiple concurrent text bias analyses (simulated)."""
        graph = create_bias_text_graph()

        texts = [
            "Hiring young developers",
            "Seeking native speakers only",
            "Perfect vision required",
            "Recent college graduates preferred",
            "Cultural fit with our brotherhood",
        ]

        results = []
        for text in texts:
            result = graph.invoke({"text": text, "options": {}})
            results.append(result)

        # All should complete
        assert len(results) == len(texts)
        assert all(r is not None for r in results)
        assert all("bias_analysis" in r for r in results)

    def test_repeated_risk_assessments(self) -> None:
        """Test repeated risk assessments for stability."""
        graph = create_risk_graph()

        scenario = "AI-powered hiring system deployment"

        # Run 10 times to check for state leaks
        for _ in range(10):
            result = graph.invoke({"scenario_text": scenario, "options": {}})
            assert result is not None
            assert "risks" in result
            assert len(result["risks"]) > 0

    def test_option_type_validation(self) -> None:
        """Test that invalid option types are handled."""
        graph = create_bias_text_graph()

        result = graph.invoke(
            {
                "text": "Sample text",
                "options": {
                    "temperature": "not_a_number",  # Invalid type
                    "llm_max_tokens": -1000,  # Invalid value
                    "summary_max_length": None,  # Invalid type
                },
            },
        )

        # Should use defaults and complete
        assert result is not None
        assert "bias_analysis" in result


@pytest.mark.skipif(
    True,  # Skip by default - only run manually for extreme stress
    reason=(
        "Extreme stress test - run manually with: "
        "pytest tests/stress/test_stress_graphs.py::test_extreme_memory_stress -v"
    ),
)
def test_extreme_memory_stress() -> None:
    """EXTREME: Test with 100MB+ input (manual run only)."""
    graph = create_bias_text_graph()

    # 100MB of text
    extreme_text = "Hiring young developers. " * 5000000

    result = graph.invoke({"text": extreme_text, "options": {}})

    assert result is not None
