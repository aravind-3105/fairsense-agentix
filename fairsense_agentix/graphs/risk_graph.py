"""RiskGraph subgraph for AI risk assessment.

This workflow embeds scenario text, searches for relevant AI risks via FAISS,
retrieves corresponding AI-RMF (Risk Management Framework) recommendations,
and outputs results as an HTML table and CSV file.

Workflow:
    START → embed_scenario → search_risks → search_rmf_per_risk → join_data
         → format_html → export_csv → END

Phase 2 (Checkpoint 2.6): Stub implementations with mock FAISS/embedding
Phase 5+: Real embedding models, FAISS integration, actual risk/RMF databases

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

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.graphs.state import RiskState


# ============================================================================
# Node Functions
# ============================================================================


def embed_scenario(state: RiskState) -> dict:
    """Embedding node: Generate vector embedding for scenario text.

    Uses sentence embedding models (sentence-transformers, OpenAI embeddings,
    etc.) to convert scenario text into a dense vector representation for
    semantic search.

    Phase 2: Stub returning mock embedding.
    Phase 5+: Real embedding using sentence-transformers or similar.

    Parameters
    ----------
    state : RiskState
        Current state with scenario_text

    Returns
    -------
    dict
        State update with embedding field

    Examples
    --------
    >>> state = RiskState(scenario_text="AI deployment scenario")
    >>> update = embed_scenario(state)
    >>> len(update["embedding"])
    384
    """
    # Phase 2: Stub implementation
    # Phase 5+: Real embedding using sentence-transformers (all-MiniLM-L6-v2, etc.)

    scenario_length = len(state.scenario_text)

    # Mock 384-dimensional embedding (default for all-MiniLM-L6-v2)
    # In reality, this would be: model.encode(state.scenario_text)
    mock_embedding = [0.1 * (i % 10) for i in range(384)]

    # Add some variation based on scenario length
    if scenario_length > 100:
        mock_embedding[0] = 0.9
        mock_embedding[1] = 0.8
    else:
        mock_embedding[0] = 0.3
        mock_embedding[1] = 0.4

    return {"embedding": mock_embedding}


def search_risks(state: RiskState) -> dict:
    """FAISS search node: Find top-k relevant AI risks.

    Performs semantic search in the AI risks database using FAISS (Facebook
    AI Similarity Search) to find the most relevant risks for the given
    scenario.

    Phase 2: Stub returning mock risks.
    Phase 5+: Real FAISS search against risks index.

    Parameters
    ----------
    state : RiskState
        Current state with embedding

    Returns
    -------
    dict
        State update with risks field

    Examples
    --------
    >>> state = RiskState(scenario_text="...", embedding=[...])
    >>> update = search_risks(state)
    >>> len(update["risks"])
    5
    """
    # Phase 2: Stub implementation
    # Phase 5+: Real FAISS search
    #   index = faiss.read_index(settings.faiss_risks_index_path)
    #   distances, indices = index.search(embedding, k=top_k)

    if state.embedding is None:
        return {"risks": []}

    # Get top_k from options (default 5)
    options = state.options or {}
    top_k = options.get("top_k", 5)

    # Mock risk data
    mock_risks = [
        {
            "risk_id": "RISK-001",
            "category": "Privacy",
            "description": "Unauthorized collection and use of personal biometric data",
            "severity": "High",
            "likelihood": "High",
            "score": 0.92,
        },
        {
            "risk_id": "RISK-002",
            "category": "Fairness",
            "description": "Algorithmic bias leading to discriminatory outcomes in identification",
            "severity": "High",
            "likelihood": "Medium",
            "score": 0.88,
        },
        {
            "risk_id": "RISK-003",
            "category": "Transparency",
            "description": "Lack of transparency in facial recognition system deployment",
            "severity": "Medium",
            "likelihood": "High",
            "score": 0.85,
        },
        {
            "risk_id": "RISK-004",
            "category": "Safety",
            "description": "Misidentification leading to wrongful detention or harm",
            "severity": "Critical",
            "likelihood": "Low",
            "score": 0.82,
        },
        {
            "risk_id": "RISK-005",
            "category": "Accountability",
            "description": "Unclear accountability for false positives and system failures",
            "severity": "Medium",
            "likelihood": "Medium",
            "score": 0.78,
        },
    ]

    # Return top_k risks
    return {"risks": mock_risks[:top_k]}


def search_rmf_per_risk(state: RiskState) -> dict:
    """RMF search node: Find AI-RMF recommendations for each risk.

    For each identified risk, performs a secondary FAISS search to find
    relevant recommendations from the NIST AI Risk Management Framework (AI-RMF).

    Phase 2: Stub returning mock RMF recommendations.
    Phase 5+: Real FAISS search against RMF index for each risk.

    Parameters
    ----------
    state : RiskState
        Current state with risks

    Returns
    -------
    dict
        State update with rmf_recommendations field (dict keyed by risk_id)

    Examples
    --------
    >>> state = RiskState(scenario_text="...", risks=[...])
    >>> update = search_rmf_per_risk(state)
    >>> "RISK-001" in update["rmf_recommendations"]
    True
    """
    # Phase 2: Stub implementation
    # Phase 5+: Real FAISS search for each risk
    #   For each risk:
    #     risk_embedding = embed(risk.description)
    #     rmf_results = rmf_index.search(risk_embedding, k=3)

    if not state.risks:
        return {"rmf_recommendations": {}}

    # Mock RMF recommendations keyed by risk_id
    rmf_recommendations = {}

    for risk in state.risks:
        risk_id = risk["risk_id"]

        # Mock 3 RMF recommendations per risk
        rmf_recommendations[risk_id] = [
            {
                "rmf_id": f"{risk_id}-RMF-1",
                "function": "GOVERN",
                "category": "Policies and Procedures",
                "recommendation": "Establish clear policies for biometric data collection, use, and retention",
                "relevance_score": 0.95,
            },
            {
                "rmf_id": f"{risk_id}-RMF-2",
                "function": "MAP",
                "category": "Impact Assessment",
                "recommendation": "Conduct comprehensive impact assessment covering privacy and civil liberties",
                "relevance_score": 0.89,
            },
            {
                "rmf_id": f"{risk_id}-RMF-3",
                "function": "MEASURE",
                "category": "Performance Monitoring",
                "recommendation": "Implement continuous monitoring for false positive and negative rates across demographics",
                "relevance_score": 0.84,
            },
        ]

    return {"rmf_recommendations": rmf_recommendations}


def join_data(state: RiskState) -> dict:
    """Join node: Combine risks and RMF recommendations into table rows.

    Merges risk data with their corresponding RMF recommendations into a
    flat table structure suitable for display and export.

    Parameters
    ----------
    state : RiskState
        Current state with risks and rmf_recommendations

    Returns
    -------
    dict
        State update with joined_table field

    Examples
    --------
    >>> state = RiskState(scenario_text="...", risks=[...], rmf_recommendations={...})
    >>> update = join_data(state)
    >>> len(update["joined_table"]) > 0
    True
    """
    if not state.risks or not state.rmf_recommendations:
        return {"joined_table": []}

    joined_table = []

    for risk in state.risks:
        risk_id = risk["risk_id"]
        rmf_recs = state.rmf_recommendations.get(risk_id, [])

        for rmf in rmf_recs:
            # Create joined row
            row = {
                # Risk fields
                "risk_id": risk["risk_id"],
                "risk_category": risk["category"],
                "risk_description": risk["description"],
                "risk_severity": risk["severity"],
                "risk_likelihood": risk["likelihood"],
                "risk_score": risk["score"],
                # RMF fields
                "rmf_id": rmf["rmf_id"],
                "rmf_function": rmf["function"],
                "rmf_category": rmf["category"],
                "rmf_recommendation": rmf["recommendation"],
                "rmf_relevance_score": rmf["relevance_score"],
            }
            joined_table.append(row)

    return {"joined_table": joined_table}


def format_html(state: RiskState) -> dict:
    """HTML formatting node: Generate HTML table from joined data.

    Creates a styled HTML table displaying risks and recommendations in a
    user-friendly format. Includes color-coding by severity and interactive
    elements.

    Phase 2: Basic HTML table.
    Phase 5+: Enhanced styling, sorting, filtering, collapsible sections.

    Parameters
    ----------
    state : RiskState
        Current state with joined_table

    Returns
    -------
    dict
        State update with html_table field

    Examples
    --------
    >>> state = RiskState(scenario_text="...", joined_table=[...])
    >>> update = format_html(state)
    >>> "<table>" in update["html_table"]
    True
    """
    if not state.joined_table:
        return {"html_table": "<p>No risks found for this scenario.</p>"}

    # Build HTML table
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Risk Assessment Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
        }
        .scenario {
            background-color: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background-color: #1976D2;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .severity-critical { background-color: #ffcdd2; }
        .severity-high { background-color: #ffe0b2; }
        .severity-medium { background-color: #fff9c4; }
        .severity-low { background-color: #c8e6c9; }
        .rmf-govern { color: #1976D2; font-weight: bold; }
        .rmf-map { color: #388E3C; font-weight: bold; }
        .rmf-measure { color: #F57C00; font-weight: bold; }
        .rmf-manage { color: #7B1FA2; font-weight: bold; }
    </style>
</head>
<body>
    <h1>AI Risk Assessment Report</h1>
    <div class="scenario">
        <strong>Scenario:</strong> """
        + state.scenario_text
        + """
    </div>
    <table>
        <thead>
            <tr>
                <th>Risk ID</th>
                <th>Category</th>
                <th>Description</th>
                <th>Severity</th>
                <th>RMF Function</th>
                <th>Recommendation</th>
            </tr>
        </thead>
        <tbody>"""
    ]

    for row in state.joined_table:
        severity_class = f"severity-{row['risk_severity'].lower()}"
        rmf_class = f"rmf-{row['rmf_function'].lower()}"

        html_parts.append(f"""
            <tr class="{severity_class}">
                <td>{row["risk_id"]}</td>
                <td>{row["risk_category"]}</td>
                <td>{row["risk_description"]}</td>
                <td><strong>{row["risk_severity"]}</strong></td>
                <td class="{rmf_class}">{row["rmf_function"]}</td>
                <td>{row["rmf_recommendation"]}</td>
            </tr>""")

    html_parts.append("""
        </tbody>
    </table>
    <p style="margin-top: 20px; font-size: 0.9em; color: #666;">
        <em>Report generated by FairSense-AgentiX AI Risk Assessment System</em>
    </p>
</body>
</html>""")

    return {"html_table": "".join(html_parts)}


def export_csv(state: RiskState) -> dict:
    """CSV export node: Generate CSV file from joined data.

    Exports the joined risk and RMF data to a CSV file for further analysis,
    reporting, or integration with other tools.

    Phase 2: Stub returning mock CSV path.
    Phase 5+: Real CSV file generation with configurable output directory.

    Parameters
    ----------
    state : RiskState
        Current state with joined_table

    Returns
    -------
    dict
        State update with csv_path field

    Examples
    --------
    >>> state = RiskState(scenario_text="...", joined_table=[...])
    >>> update = export_csv(state)
    >>> update["csv_path"].endswith(".csv")
    True
    """
    # Phase 2: Stub implementation
    # Phase 5+: Real CSV file writing
    #   import csv
    #   csv_path = Path(f"outputs/risk_assessment_{run_id}.csv")
    #   with open(csv_path, 'w') as f:
    #       writer = csv.DictWriter(f, fieldnames=...)
    #       writer.writeheader()
    #       writer.writerows(state.joined_table)

    if not state.joined_table:
        return {"csv_path": None}

    # Mock CSV path
    run_id = state.run_id
    mock_csv_path = f"/tmp/risk_assessment_{run_id[:8]}.csv"

    return {"csv_path": mock_csv_path}


# Note: Quality assessment removed from subgraph
# Orchestrator's posthoc_eval node owns all evaluation (Phase 7+)


# ============================================================================
# Graph Construction
# ============================================================================


def create_risk_graph() -> CompiledStateGraph:
    """Create and compile the RiskGraph subgraph.

    Builds the AI risk assessment workflow with FAISS-based retrieval.

    Graph Structure:
        START → embed_scenario → search_risks → search_rmf_per_risk
             → join_data → format_html → export_csv → END

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Returns
    -------
    StateGraph
        Compiled risk assessment graph ready for invocation

    Examples
    --------
    >>> graph = create_risk_graph()
    >>> result = graph.invoke(
    ...     {"scenario_text": "Deploying facial recognition", "options": {"top_k": 3}}
    ... )
    >>> result["html_table"]
    '<table>...</table>'
    """
    # Create graph with RiskState
    workflow = StateGraph(RiskState)

    # Add nodes
    workflow.add_node("embed_scenario", embed_scenario)
    workflow.add_node("search_risks", search_risks)
    workflow.add_node("search_rmf_per_risk", search_rmf_per_risk)
    workflow.add_node("join_data", join_data)
    workflow.add_node("format_html", format_html)
    workflow.add_node("export_csv", export_csv)

    # Add edges (sequential workflow)
    workflow.add_edge(START, "embed_scenario")
    workflow.add_edge("embed_scenario", "search_risks")
    workflow.add_edge("search_risks", "search_rmf_per_risk")
    workflow.add_edge("search_rmf_per_risk", "join_data")
    workflow.add_edge("join_data", "format_html")
    workflow.add_edge("format_html", "export_csv")
    workflow.add_edge("export_csv", END)

    # Compile graph
    return workflow.compile()
