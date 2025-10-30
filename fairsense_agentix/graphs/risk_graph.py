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
from fairsense_agentix.tools import get_tool_registry


# ============================================================================
# Node Functions
# ============================================================================


def embed_scenario(state: RiskState) -> dict:
    """Embedding node: Generate vector embedding for scenario text.

    Uses sentence embedding models (sentence-transformers, OpenAI embeddings,
    etc.) to convert scenario text into a dense vector representation for
    semantic search.

    Phase 4+: Uses tool registry to get embedder implementation (fake or real).
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
    # Get tools from registry
    registry = get_tool_registry()

    # Use embedder tool to generate embedding
    embedding = registry.embedder.encode(state.scenario_text)

    return {"embedding": embedding}


def search_risks(state: RiskState) -> dict:
    """FAISS search node: Find top-k relevant AI risks.

    Performs semantic search in the AI risks database using FAISS (Facebook
    AI Similarity Search) to find the most relevant risks for the given
    scenario.

    Phase 4+: Uses tool registry to get FAISS implementation (fake or real).
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
    # Get tools from registry
    registry = get_tool_registry()

    if state.embedding is None:
        return {"risks": []}

    # Get top_k from options (default 5)
    options = state.options or {}
    top_k = options.get("top_k", 5)

    # Validate top_k parameter
    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 5  # Use default for invalid values
    top_k = min(top_k, 100)  # Cap at reasonable maximum

    # Use FAISS risks index from registry
    risks = registry.faiss_risks.search(query_vector=state.embedding, top_k=top_k)

    return {"risks": risks}


def search_rmf_per_risk(state: RiskState) -> dict:
    """RMF search node: Find AI-RMF recommendations for each risk.

    For each identified risk, performs a secondary FAISS search to find
    relevant recommendations from the NIST AI Risk Management Framework (AI-RMF).

    Phase 4+: Uses tool registry to get embedder and FAISS RMF implementation.
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
    # Get tools from registry
    registry = get_tool_registry()

    if not state.risks:
        return {"rmf_recommendations": {}}

    # Get RMF recommendations per risk from options (default 3)
    options = state.options or {}
    rmf_per_risk = options.get("rmf_per_risk", 3)

    # Validate rmf_per_risk parameter
    if not isinstance(rmf_per_risk, int) or rmf_per_risk <= 0:
        rmf_per_risk = 3  # Use default for invalid values
    rmf_per_risk = min(rmf_per_risk, 20)  # Cap at reasonable maximum

    # RMF recommendations keyed by risk_id
    rmf_recommendations = {}

    for risk in state.risks:
        risk_id = risk["id"] if "id" in risk else risk.get("risk_id", "")
        risk_description = risk.get("description", risk.get("text", ""))

        # Embed the risk description
        risk_embedding = registry.embedder.encode(risk_description)

        # Search RMF index for recommendations
        rmf_results = registry.faiss_rmf.search(
            query_vector=risk_embedding, top_k=rmf_per_risk
        )

        rmf_recommendations[risk_id] = rmf_results

    return {"rmf_recommendations": rmf_recommendations}


def join_data(state: RiskState) -> dict:
    """Join node: Combine risks and RMF recommendations into table rows.

    Merges risk data with their corresponding RMF recommendations into a
    flat table structure suitable for display and export.

    Phase 4+: Handles both fake FAISS and future real FAISS key structures.

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
        # Handle both key naming conventions (fake FAISS vs original mock)
        risk_id = risk.get("risk_id") or risk.get("id", "")
        rmf_recs = state.rmf_recommendations.get(risk_id, [])

        for rmf in rmf_recs:
            # Create joined row - handle flexible key names
            row = {
                # Risk fields (handle both "risk_id" and "id")
                "risk_id": risk.get("risk_id") or risk.get("id", ""),
                "risk_category": risk.get("category", ""),
                "risk_description": risk.get("description") or risk.get("text", ""),
                "risk_severity": risk.get("severity", ""),
                "risk_likelihood": risk.get("likelihood", "N/A"),
                "risk_score": risk.get("score", 0.0),
                # RMF fields (handle both naming conventions)
                "rmf_id": rmf.get("rmf_id") or rmf.get("id", ""),
                "rmf_function": rmf.get("function", ""),
                "rmf_category": rmf.get("category", ""),
                "rmf_recommendation": rmf.get("recommendation") or rmf.get("text", ""),
                "rmf_relevance_score": rmf.get("relevance_score")
                or rmf.get("score", 0.0),
            }
            joined_table.append(row)

    return {"joined_table": joined_table}


def format_html(state: RiskState) -> dict:
    """HTML formatting node: Generate HTML table from joined data.

    Creates a styled HTML table displaying risks and recommendations in a
    user-friendly format. Includes color-coding by severity and interactive
    elements.

    Phase 4+: Uses formatter tool from registry.
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
    # Get tools from registry
    registry = get_tool_registry()

    if not state.joined_table:
        return {"html_table": "<p>No risks found for this scenario.</p>"}

    # Define table headers
    headers = [
        "risk_id",
        "risk_category",
        "risk_description",
        "risk_severity",
        "rmf_function",
        "rmf_recommendation",
    ]

    # Use formatter tool to generate HTML table with error handling
    try:
        html_table = registry.formatter.table(data=state.joined_table, headers=headers)
    except Exception:
        # TODO Phase 8: Log formatting failure with telemetry
        # Fallback: Plain HTML message
        # Formatting is presentation layer - if it fails, provide error message
        # rather than killing the entire workflow
        html_table = "<p>Error generating risk assessment table</p>"

    return {"html_table": html_table}


def export_csv(state: RiskState) -> dict:
    """CSV export node: Generate CSV file from joined data.

    Exports the joined risk and RMF data to a CSV file for further analysis,
    reporting, or integration with other tools.

    Phase 4+: Uses persistence tool from registry.
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
    # Get tools from registry
    registry = get_tool_registry()

    if not state.joined_table:
        return {"csv_path": None}

    # Generate filename using run_id with validation
    run_id = state.run_id if state.run_id else "unknown"
    # Sanitize run_id to prevent path traversal
    run_id_safe = "".join(c for c in run_id[:8] if c.isalnum() or c == "-")
    if not run_id_safe:
        run_id_safe = "unknown"
    filename = f"risk_assessment_{run_id_safe}.csv"

    # Use persistence tool to save CSV with error handling
    try:
        csv_path = registry.persistence.save_csv(
            data=state.joined_table, filename=filename
        )
    except Exception:
        # TODO Phase 8: Log persistence failure with telemetry
        # Fallback: None (CSV export failed)
        # If persistence fails, workflow continues but without CSV output
        # Orchestrator can handle this gracefully
        csv_path = None

    return {"csv_path": str(csv_path) if csv_path else None}


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
