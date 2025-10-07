"""Minimal tests for RiskGraph.

These tests verify:
- Graph compiles successfully
- End-to-end execution works
- FAISS search pipeline (embedding → risks → RMF)
- All output fields populated (HTML, CSV)

Comprehensive integration tests will be added after Phase 2 completion.
"""

from fairsense_agentix.graphs.risk_graph import create_risk_graph


class TestRiskGraphBasic:
    """Basic smoke tests for RiskGraph."""

    def test_graph_compiles(self) -> None:
        """Test RiskGraph compiles without errors."""
        graph = create_risk_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_risk_assessment_end_to_end(self) -> None:
        """Test complete execution flow for AI risk assessment."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "Deploying facial recognition system in public spaces for security monitoring",
                "options": {},
            }
        )

        # Verify embedding generated
        assert result["embedding"] is not None
        assert len(result["embedding"]) == 384  # Default dimension

        # Verify risks found
        assert result["risks"] is not None
        assert len(result["risks"]) == 5  # Default top_k
        assert all("risk_id" in r for r in result["risks"])
        assert all("description" in r for r in result["risks"])

        # Verify RMF recommendations retrieved
        assert result["rmf_recommendations"] is not None
        assert len(result["rmf_recommendations"]) > 0
        # Each risk should have RMF recommendations
        for risk in result["risks"]:
            risk_id = risk["risk_id"]
            assert risk_id in result["rmf_recommendations"]
            assert len(result["rmf_recommendations"][risk_id]) == 3  # 3 RMF per risk

        # Verify data joined
        assert result["joined_table"] is not None
        assert len(result["joined_table"]) > 0
        # Should have risk + RMF fields
        first_row = result["joined_table"][0]
        assert "risk_id" in first_row
        assert "rmf_id" in first_row
        assert "rmf_recommendation" in first_row

        # Verify HTML generated
        assert result["html_table"] is not None
        assert "<table>" in result["html_table"]
        assert "Deploying facial recognition" in result["html_table"]

        # Verify CSV path generated
        assert result["csv_path"] is not None
        assert result["csv_path"].endswith(".csv")

    def test_top_k_parameter(self) -> None:
        """Test that top_k option controls number of risks returned."""
        graph = create_risk_graph()

        # Request only top 3 risks
        result = graph.invoke(
            {
                "scenario_text": "Using AI for automated hiring decisions",
                "options": {"top_k": 3},
            }
        )

        # Verify only 3 risks returned
        assert len(result["risks"]) == 3

        # Verify RMF recommendations for all 3 risks
        assert len(result["rmf_recommendations"]) == 3

        # Verify joined table has 3 risks * 3 RMF = 9 rows
        assert len(result["joined_table"]) == 9

    def test_short_scenario(self) -> None:
        """Test execution with short scenario text."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "AI chatbot",
                "options": {},
            }
        )

        # Verify execution completed
        assert result["embedding"] is not None
        assert result["risks"] is not None
        assert result["html_table"] is not None

        # Short scenario should still get full risk analysis
        assert len(result["risks"]) == 5

    def test_options_propagate(self) -> None:
        """Test that options are accessible in state."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "Test scenario",
                "options": {"top_k": 2, "embedding_model": "custom-model"},
            }
        )

        # Verify execution completed with options
        assert result["risks"] is not None
        assert len(result["risks"]) == 2  # top_k applied
        assert result["options"]["embedding_model"] == "custom-model"
