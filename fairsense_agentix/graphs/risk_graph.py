"""RiskGraph subgraph for AI risk assessment.

This workflow embeds scenario text, searches for relevant AI risks via FAISS,
retrieves corresponding AI-RMF (Risk Management Framework) recommendations,
and outputs results as an HTML table and CSV file.

Workflow:
    START → embed_scenario → search_risks → search_rmf_per_risk → join_data
         → format_html → export_csv → END

Phase 2 (Checkpoint 2.6): Stub implementations with mock FAISS/embedding
Phase 5+: Real embedding models, FAISS integration, actual risk/RMF databases

Implementation lives under :mod:`fairsense_agentix.graphs.risk`.

Example:
    >>> from fairsense_agentix.graphs.risk_graph import create_risk_graph
    >>> graph = create_risk_graph()
    >>> result = graph.invoke(
    ...     {
    ...         "scenario_text": "Deploying facial recognition in public spaces",
    ...         "options": {},
    ...     }
    ... )
    >>> result["html_table"]
    '<table>...</table>'
"""

from fairsense_agentix.graphs.risk.build import create_risk_graph


__all__ = ["create_risk_graph"]
