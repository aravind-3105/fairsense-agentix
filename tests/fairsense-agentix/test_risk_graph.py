"""Minimal tests for RiskGraph.

These tests verify:
- Graph compiles successfully
- End-to-end execution works
- FAISS search pipeline (embedding → risks → RMF)
- All output fields populated (HTML, CSV)

Phase 6.3: Added deterministic fixture tests and golden HTML/CSV output tests
           to verify consistent output structure and format.

Comprehensive integration tests will be added after Phase 2 completion.
"""

import csv
from pathlib import Path

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
        # Phase 4: FakeFAISSIndexTool returns "id" key, not "risk_id"
        assert all(("risk_id" in r or "id" in r) for r in result["risks"])
        assert all(("description" in r or "text" in r) for r in result["risks"])

        # Verify RMF recommendations retrieved
        assert result["rmf_recommendations"] is not None
        assert len(result["rmf_recommendations"]) > 0
        # Each risk should have RMF recommendations
        for risk in result["risks"]:
            # Handle both key naming conventions
            risk_id = risk.get("risk_id") or risk.get("id")
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
        # Phase 4: FakeFormatterTool generates basic HTML table
        # (scenario text no longer embedded in table)

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


class TestRiskGraphDeterministicFixtures:
    """Deterministic fixture tests for RiskGraph.

    Phase 6.3: Tests RiskGraph with controlled scenario inputs to verify
               consistent behavior and output structure.
    """

    def test_facial_recognition_scenario(self) -> None:
        """Test risk assessment for facial recognition deployment.

        Verifies complete pipeline produces structured output.
        """
        graph = create_risk_graph()

        scenario = """Deploying facial recognition system in public spaces for
        real-time security monitoring and threat detection. System will capture
        and analyze faces of all individuals passing through monitored areas."""

        result = graph.invoke({"scenario_text": scenario, "options": {}})

        # Verify all pipeline stages completed
        assert result["embedding"] is not None
        assert len(result["embedding"]) == 384

        assert result["risks"] is not None
        assert len(result["risks"]) == 5

        assert result["rmf_recommendations"] is not None
        assert len(result["rmf_recommendations"]) == 5

        assert result["joined_table"] is not None
        assert len(result["joined_table"]) == 15  # 5 risks * 3 RMF each

        assert result["html_table"] is not None
        assert "<table>" in result["html_table"]

        assert result["csv_path"] is not None

    def test_hiring_ai_scenario(self) -> None:
        """Test risk assessment for AI-powered hiring system."""
        graph = create_risk_graph()

        scenario = """Implementing AI-based resume screening and candidate
        evaluation system to automate initial hiring decisions. System analyzes
        resumes, video interviews, and predicts job performance."""

        result = graph.invoke({"scenario_text": scenario, "options": {"top_k": 3}})

        # Verify top_k=3 produces correct structure
        assert len(result["risks"]) == 3
        assert len(result["rmf_recommendations"]) == 3
        assert len(result["joined_table"]) == 9  # 3 risks * 3 RMF

        # Verify joined table has all required fields
        for row in result["joined_table"]:
            assert "risk_id" in row
            assert "risk_category" in row
            assert "risk_description" in row
            assert "risk_severity" in row
            assert "rmf_id" in row
            assert "rmf_function" in row
            assert "rmf_recommendation" in row

    def test_healthcare_ai_scenario(self) -> None:
        """Test risk assessment for medical diagnosis AI."""
        graph = create_risk_graph()

        scenario = """Deploying AI system for automated medical image analysis
        and preliminary diagnosis recommendations. System processes X-rays, MRIs,
        and provides diagnostic suggestions to healthcare providers."""

        result = graph.invoke(
            {"scenario_text": scenario, "options": {"top_k": 4, "rmf_per_risk": 2}}
        )

        # Verify custom options applied
        assert len(result["risks"]) == 4
        assert len(result["joined_table"]) == 8  # 4 risks * 2 RMF

        # Verify each risk has exactly 2 RMF recommendations
        for risk in result["risks"]:
            risk_id = risk.get("risk_id") or risk.get("id")
            assert len(result["rmf_recommendations"][risk_id]) == 2

    def test_empty_scenario_handling(self) -> None:
        """Test graceful handling of minimal/empty scenario."""
        graph = create_risk_graph()

        result = graph.invoke({"scenario_text": "AI", "options": {}})

        # Even minimal input should produce valid output structure
        assert result["embedding"] is not None
        assert result["risks"] is not None
        assert result["html_table"] is not None
        assert result["csv_path"] is not None

    def test_long_scenario_text(self) -> None:
        """Test handling of long, detailed scenario description."""
        graph = create_risk_graph()

        long_scenario = " ".join(
            [
                "Implementing comprehensive AI-powered surveillance system",
                "with facial recognition, behavior prediction, anomaly detection,",
                "automated alerting, cross-database matching, real-time tracking,",
                "historical pattern analysis, and predictive threat modeling",
            ]
            * 5
        )  # Repeat to make it longer

        result = graph.invoke({"scenario_text": long_scenario, "options": {}})

        # Long scenarios should still produce valid output
        assert result["embedding"] is not None
        assert len(result["risks"]) == 5
        assert len(result["joined_table"]) == 15


class TestRiskGraphGoldenOutputs:
    """Golden output tests for RiskGraph HTML and CSV formats.

    Phase 6.3: Tests output format and structure consistency.
               Uses structural assertions (no exact matching).
    """

    def test_html_table_structure(self) -> None:
        """Test HTML table has correct structure and required elements."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "AI-powered credit scoring system",
                "options": {},
            }
        )

        html = result["html_table"]

        # Verify HTML structure
        assert "<table>" in html or "<table " in html
        assert "</table>" in html

        # Verify key column headers (flexible matching)
        # Fake formatter may use different casing/formatting
        html_lower = html.lower()
        assert "risk" in html_lower
        assert "rmf" in html_lower or "recommendation" in html_lower

        # Verify some content rows present (not just headers)
        # Look for data from joined_table
        assert len(result["joined_table"]) > 0
        # At least some field values should appear in HTML
        # (exact format depends on formatter implementation)

    def test_csv_export_structure(self) -> None:
        """Test CSV file is created with correct structure."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "Autonomous vehicle decision-making AI",
                "options": {},
            }
        )

        csv_path = Path(result["csv_path"])

        # Verify CSV file exists
        assert csv_path.exists()

        # Read and verify CSV structure
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Verify CSV has data
            assert len(rows) > 0

            # Verify CSV has required columns
            assert len(reader.fieldnames) > 0
            fieldnames = reader.fieldnames

            # Check for key risk/RMF columns
            assert any("risk" in col for col in fieldnames)
            assert any("rmf" in col or "recommendation" in col for col in fieldnames)

            # Verify each row has all columns
            for row in rows:
                assert len(row) == len(fieldnames)

        # Clean up test CSV file
        csv_path.unlink()

    def test_csv_data_consistency(self) -> None:
        """Test CSV data matches joined_table structure."""
        graph = create_risk_graph()

        result = graph.invoke(
            {
                "scenario_text": "AI content moderation system",
                "options": {"top_k": 2},
            }
        )

        csv_path = Path(result["csv_path"])

        # Read CSV
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        # Verify CSV row count matches joined_table
        assert len(csv_rows) == len(result["joined_table"])

        # Verify first row data consistency (field names may vary)
        if len(csv_rows) > 0 and len(result["joined_table"]) > 0:
            csv_first = csv_rows[0]
            # table_first = result["joined_table"][0]

            # Both should have similar number of fields
            assert len(csv_first.keys()) >= 5  # At least 5 columns

        # Clean up
        csv_path.unlink()

    def test_html_table_with_no_risks(self) -> None:
        """Test HTML generation when no risks found (edge case)."""
        graph = create_risk_graph()

        # This shouldn't happen with fake tools, but test the fallback
        # In real scenario, obscure text might return no good matches
        result = graph.invoke(
            {
                "scenario_text": "xyz abc test 123",
                "options": {},
            }
        )

        # Even with weird input, should produce valid HTML
        assert result["html_table"] is not None
        assert isinstance(result["html_table"], str)
        assert len(result["html_table"]) > 0
