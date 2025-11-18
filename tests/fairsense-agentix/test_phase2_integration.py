"""Comprehensive integration tests for Phase 2.

These tests verify end-to-end orchestrator → subgraph execution across
all three workflows (text, image, risk). They test the complete agentic
infrastructure scaffold with stub implementations.

These tests complement the minimal per-checkpoint tests (3-5 each) by
verifying the full integration:
- Orchestrator coordination
- Router selection
- Subgraph invocation
- State propagation
- Options flow
- Run ID tracing
"""

import pytest

from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


@pytest.mark.integration_test
class TestPhase2Integration:
    """End-to-end integration tests for Phase 2 infrastructure."""

    def test_text_bias_workflow_full_integration(self) -> None:
        """Test complete text bias analysis workflow end-to-end.

        Verifies:
        - Orchestrator routes text input to BiasTextGraph
        - Router creates correct SelectionPlan
        - BiasTextGraph executes all nodes (analyze → summarize → highlight)
        - Results propagate back to orchestrator
        - Final result packaged correctly
        - Options flow through entire stack
        - Run ID maintained for tracing
        """
        graph = create_orchestrator_graph()

        # Execute complete workflow with options
        result = graph.invoke(
            {
                "input_type": "text",
                "content": "We're looking for young, energetic professionals. Recent graduates preferred.",
                "options": {
                    "llm_model": "gpt-4-turbo",
                    "temperature": 0.5,
                    "enable_summary": True,
                },
            }
        )

        # Verify orchestrator coordination
        assert result["plan"] is not None
        assert result["plan"].workflow_id == "bias_text"
        assert result["plan"].reasoning is not None
        assert result["plan"].confidence == 1.0  # Deterministic routing

        # Verify router created correct plan
        assert result["plan"].tool_preferences["llm"] == "gpt-4-turbo"
        assert result["plan"].tool_preferences["temperature"] == 0.5
        assert result["plan"].node_params["bias_llm"]["model"] == "gpt-4-turbo"

        # Verify preflight evaluation ran
        assert result["preflight_eval"] is not None
        assert result["preflight_eval"].passed is True

        # Verify BiasTextGraph execution
        assert result["workflow_result"] is not None
        assert result["workflow_result"]["workflow_id"] == "bias_text"
        assert result["workflow_result"]["bias_analysis"] is not None
        assert isinstance(
            result["workflow_result"]["bias_analysis"], BiasAnalysisOutput
        )
        assert result["workflow_result"]["summary"] is not None
        assert result["workflow_result"]["highlighted_html"] is not None
        assert "<html>" in result["workflow_result"]["highlighted_html"]

        # Verify posthoc evaluation ran
        assert result["posthoc_eval"] is not None
        assert result["posthoc_eval"].passed is True
        assert result["posthoc_eval"].score > 0

        # Verify orchestrator decision
        assert result["decision"] == "accept"  # Refinement disabled in Phase 2

        # Verify final result packaging
        assert result["final_result"] is not None
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "bias_text"
        assert result["final_result"]["output"] is not None
        assert result["final_result"]["refinement_count"] == 0

        # Verify run_id maintained for tracing (in metadata)
        assert result["final_result"]["run_id"] is not None
        assert len(result["final_result"]["run_id"]) == 36  # UUID format

    def test_image_bias_workflow_full_integration(self) -> None:
        """Test complete image bias analysis workflow end-to-end.

        Verifies:
        - Orchestrator routes image input to BiasImageGraph
        - Router creates correct SelectionPlan with OCR + caption params
        - BiasImageGraph executes parallel extraction (OCR ∥ Caption)
        - Text merging happens correctly
        - Bias analysis runs on merged text
        - Results propagate back with image-specific fields
        - Fallback plan included in SelectionPlan
        """
        graph = create_orchestrator_graph()

        # Large image to trigger detailed mocks
        large_image = b"x" * 15000

        result = graph.invoke(
            {
                "input_type": "image",
                "content": large_image,
                "options": {
                    "ocr_tool": "tesseract",
                    "caption_model": "blip2",
                    "validate_image_bytes": False,
                },
            }
        )

        # Verify orchestrator coordination
        assert result["plan"] is not None
        assert result["plan"].workflow_id == "bias_image"
        assert result["plan"].tool_preferences["ocr"] == "tesseract"
        assert result["plan"].tool_preferences["caption"] == "blip2"
        assert result["plan"].fallbacks == ["bias_text"]  # Image can fallback to text

        # Verify BiasImageGraph parallel execution
        assert result["workflow_result"]["workflow_id"] == "bias_image"
        assert result["workflow_result"]["ocr_text"] is not None
        assert result["workflow_result"]["caption_text"] is not None
        assert len(result["workflow_result"]["ocr_text"]) > 0
        assert len(result["workflow_result"]["caption_text"]) > 0

        # Verify text merging happened
        assert result["workflow_result"]["merged_text"] is not None
        assert "OCR Extracted Text" in result["workflow_result"]["merged_text"]
        assert "Image Caption" in result["workflow_result"]["merged_text"]

        # Verify bias analysis on merged text
        assert result["workflow_result"]["bias_analysis"] is not None
        assert isinstance(
            result["workflow_result"]["bias_analysis"], BiasAnalysisOutput
        )
        assert result["workflow_result"]["summary"] is not None
        assert result["workflow_result"]["highlighted_html"] is not None

        # Verify complete orchestrator flow
        assert result["preflight_eval"].passed is True
        assert result["posthoc_eval"].passed is True
        assert result["decision"] == "accept"
        assert result["final_result"]["status"] == "success"

    def test_risk_assessment_workflow_full_integration(self) -> None:
        """Test complete AI risk assessment workflow end-to-end.

        Verifies:
        - Orchestrator routes CSV/scenario input to RiskGraph
        - Router creates correct SelectionPlan with FAISS params
        - RiskGraph executes sequential pipeline:
          * Embedding generation
          * FAISS risk search
          * Nested FAISS RMF search (per risk)
          * Data joining
          * HTML + CSV formatting
        - Results include all risk-specific fields
        - Top-k parameter honored
        """
        graph = create_orchestrator_graph()

        result = graph.invoke(
            {
                "input_type": "csv",
                "content": "Deploying facial recognition system in public spaces for real-time security monitoring and identification",
                "options": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "top_k": 5,
                },
            }
        )

        # Verify orchestrator coordination
        assert result["plan"] is not None
        assert result["plan"].workflow_id == "risk"
        assert result["plan"].tool_preferences["embedding"] == "all-MiniLM-L6-v2"
        assert result["plan"].tool_preferences["top_k"] == 5

        # Verify RiskGraph sequential execution
        assert result["workflow_result"]["workflow_id"] == "risk"

        # Verify embedding generation
        assert result["workflow_result"]["embedding"] is not None
        assert len(result["workflow_result"]["embedding"]) == 384

        # Verify FAISS risk search
        assert result["workflow_result"]["risks"] is not None
        assert len(result["workflow_result"]["risks"]) == 5  # top_k honored
        for risk in result["workflow_result"]["risks"]:
            # Phase 4: FakeFAISSIndexTool returns "id" key, not "risk_id"
            assert "risk_id" in risk or "id" in risk
            assert "category" in risk
            assert "description" in risk or "text" in risk
            assert "severity" in risk

        # Verify nested FAISS RMF search
        assert result["workflow_result"]["rmf_recommendations"] is not None
        assert len(result["workflow_result"]["rmf_recommendations"]) == 5
        for risk in result["workflow_result"]["risks"]:
            # Handle both key naming conventions
            risk_id = risk.get("risk_id") or risk.get("id")
            assert risk_id in result["workflow_result"]["rmf_recommendations"]
            rmf_recs = result["workflow_result"]["rmf_recommendations"][risk_id]
            assert len(rmf_recs) == 3  # 3 RMF per risk

        # Verify data joining
        assert result["workflow_result"]["joined_table"] is not None
        assert len(result["workflow_result"]["joined_table"]) == 15  # 5 risks × 3 RMF
        first_row = result["workflow_result"]["joined_table"][0]
        assert "risk_id" in first_row
        assert "rmf_id" in first_row
        assert "rmf_recommendation" in first_row

        # Verify HTML formatting
        assert result["workflow_result"]["html_table"] is not None
        assert "<table>" in result["workflow_result"]["html_table"]
        # Phase 4: FakeFormatterTool generates basic HTML table
        # (scenario text no longer embedded in table)

        # Verify CSV export
        assert result["workflow_result"]["csv_path"] is not None
        assert result["workflow_result"]["csv_path"].endswith(".csv")

        # Verify complete orchestrator flow
        assert result["preflight_eval"].passed is True
        assert result["posthoc_eval"].passed is True
        assert result["decision"] == "accept"
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "risk"
