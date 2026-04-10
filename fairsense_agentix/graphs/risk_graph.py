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
from fairsense_agentix.services.telemetry import telemetry
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
    Phase 6.4: Added telemetry for observability and performance tracking.

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
    with telemetry.timer("risk.embed_scenario", text_length=len(state.scenario_text)):
        telemetry.log_info(
            "risk_embed_start",
            text_length=len(state.scenario_text),
            run_id=state.run_id or "unknown",
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            # Use embedder tool to generate embedding
            with telemetry.timer("risk.embedder_tool"):
                embedding = registry.embedder.encode(state.scenario_text)

            telemetry.log_info(
                "risk_embed_complete",
                embedding_dim=len(embedding) if hasattr(embedding, "__len__") else 0,
            )
            return {"embedding": embedding}

        except Exception as e:
            telemetry.log_error("risk_embed_scenario_failed", error=e)
            raise


def search_risks(state: RiskState) -> dict:
    """FAISS search node: Find top-k relevant AI risks.

    Performs semantic search in the AI risks database using FAISS (Facebook
    AI Similarity Search) to find the most relevant risks for the given
    scenario.

    Phase 4+: Uses tool registry to get FAISS implementation (fake or real).
    Phase 5+: Real FAISS search against risks index.
    Phase 6.4: Added telemetry for observability and performance tracking.

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
    with telemetry.timer("risk.search_risks"):
        telemetry.log_info("risk_search_start", run_id=state.run_id or "unknown")

        try:
            # Get tools from registry
            registry = get_tool_registry()

            if state.embedding is None:
                telemetry.log_warning("search_risks_skipped", reason="no_embedding")
                return {"risks": []}

            # Get top_k from options (default 5)
            options = state.options or {}
            top_k = options.get("top_k", 5)

            # Validate top_k parameter
            if not isinstance(top_k, int) or top_k <= 0:
                telemetry.log_warning(
                    "invalid_top_k",
                    top_k=top_k,
                    using_default=5,
                )
                top_k = 5  # Use default for invalid values
            top_k = min(top_k, 100)  # Cap at reasonable maximum

            # Use FAISS risks index from registry
            with telemetry.timer("risk.faiss_risks_search", top_k=top_k):
                risks = registry.faiss_risks.search(
                    query_vector=state.embedding,
                    top_k=top_k,
                )

            telemetry.log_info("risk_search_complete", risks_found=len(risks))
            return {"risks": risks}

        except Exception as e:
            telemetry.log_error("risk_search_risks_failed", error=e)
            raise


def search_rmf_per_risk(state: RiskState) -> dict:
    """RMF search node: Find AI-RMF recommendations for each risk.

    For each identified risk, performs a secondary FAISS search to find
    relevant recommendations from the NIST AI Risk Management Framework (AI-RMF).

    Phase 4+: Uses tool registry to get embedder and FAISS RMF implementation.
    Phase 5+: Real FAISS search against RMF index for each risk.
    Phase 6.4: Added telemetry for observability and performance tracking.

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
    with telemetry.timer("risk.search_rmf_per_risk"):
        telemetry.log_info(
            "risk_rmf_search_start",
            run_id=state.run_id or "unknown",
            risk_count=len(state.risks) if state.risks else 0,
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            if not state.risks:
                telemetry.log_warning("search_rmf_skipped", reason="no_risks")
                return {"rmf_recommendations": {}}

            # Get RMF recommendations per risk from options (default 3)
            options = state.options or {}
            rmf_per_risk = options.get("rmf_per_risk", 3)

            # Validate rmf_per_risk parameter
            if not isinstance(rmf_per_risk, int) or rmf_per_risk <= 0:
                telemetry.log_warning(
                    "invalid_rmf_per_risk",
                    rmf_per_risk=rmf_per_risk,
                    using_default=3,
                )
                rmf_per_risk = 3  # Use default for invalid values
            rmf_per_risk = min(rmf_per_risk, 20)  # Cap at reasonable maximum

            # RMF recommendations keyed by risk_id
            rmf_recommendations = {}

            for risk in state.risks:
                risk_id = risk["id"] if "id" in risk else risk.get("risk_id", "")
                risk_description = risk.get("description", risk.get("text", ""))

                # Embed the risk description
                with telemetry.timer("risk.embed_risk_description"):
                    risk_embedding = registry.embedder.encode(risk_description)

                # Search RMF index for recommendations
                # (convert ndarray to list for protocol compatibility)
                with telemetry.timer(
                    "risk.faiss_rmf_search",
                    rmf_per_risk=rmf_per_risk,
                ):
                    rmf_results = registry.faiss_rmf.search(
                        query_vector=risk_embedding.tolist(),
                        top_k=rmf_per_risk,
                    )

                rmf_recommendations[risk_id] = rmf_results

            telemetry.log_info(
                "risk_rmf_search_complete",
                total_rmf_recs=sum(len(recs) for recs in rmf_recommendations.values()),
            )
            return {"rmf_recommendations": rmf_recommendations}

        except Exception as e:
            telemetry.log_error("risk_search_rmf_per_risk_failed", error=e)
            raise


def join_data(state: RiskState) -> dict:
    """Join node: Combine risks and RMF recommendations into table rows.

    Merges risk data with their corresponding RMF recommendations into a
    flat table structure suitable for display and export.

    Phase 4+: Handles both fake FAISS and future real FAISS key structures.
    Phase 6.4: Added telemetry for observability and performance tracking.

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
    with telemetry.timer("risk.join_data"):
        telemetry.log_info(
            "risk_join_start",
            run_id=state.run_id or "unknown",
            risk_count=len(state.risks) if state.risks else 0,
        )

        try:
            if not state.risks or not state.rmf_recommendations:
                telemetry.log_warning(
                    "join_data_skipped",
                    has_risks=bool(state.risks),
                    has_rmf=bool(state.rmf_recommendations),
                )
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
                        "risk_description": risk.get("description")
                        or risk.get("text", ""),
                        "risk_severity": risk.get("severity", ""),
                        "risk_likelihood": risk.get("likelihood", "N/A"),
                        "risk_score": risk.get("score", 0.0),
                        # RMF fields (handle both naming conventions)
                        "rmf_id": rmf.get("rmf_id") or rmf.get("id", ""),
                        "rmf_function": rmf.get("function", ""),
                        "rmf_category": rmf.get("category", ""),
                        "rmf_recommendation": rmf.get("recommendation")
                        or rmf.get("text", ""),
                        "rmf_relevance_score": rmf.get("relevance_score")
                        or rmf.get("score", 0.0),
                    }
                    joined_table.append(row)

            telemetry.log_info("risk_join_complete", table_rows=len(joined_table))
            return {"joined_table": joined_table}

        except Exception as e:
            telemetry.log_error("risk_join_data_failed", error=e)
            raise


def format_html(state: RiskState) -> dict:
    """HTML formatting node: Generate HTML table from joined data.

    Creates a styled HTML table displaying risks and recommendations in a
    user-friendly format. Includes color-coding by severity and interactive
    elements.

    Phase 4+: Uses formatter tool from registry.
    Phase 5+: Enhanced styling, sorting, filtering, collapsible sections.
    Phase 6.4: Added telemetry for observability and performance tracking.

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
    with telemetry.timer("risk.format_html"):
        telemetry.log_info(
            "risk_format_start",
            run_id=state.run_id or "unknown",
            table_rows=len(state.joined_table) if state.joined_table else 0,
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            if not state.joined_table:
                telemetry.log_warning("format_html_skipped", reason="no_joined_table")
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
                with telemetry.timer("risk.formatter_tool"):
                    html_table = registry.formatter.table(
                        data=state.joined_table,
                        headers=headers,
                    )
                telemetry.log_info("risk_format_complete", html_length=len(html_table))
            except Exception as e:
                telemetry.log_error("html_formatting_failed", error=e)
                # Fallback: Plain HTML message
                # Formatting is presentation layer - if it fails, provide error message
                # rather than killing the entire workflow
                html_table = "<p>Error generating risk assessment table</p>"

            return {"html_table": html_table}

        except Exception as e:
            telemetry.log_error("risk_format_html_failed", error=e)
            raise


def export_csv(state: RiskState) -> dict:
    """CSV export node: Generate CSV file from joined data.

    Exports the joined risk and RMF data to a CSV file for further analysis,
    reporting, or integration with other tools.

    Phase 4+: Uses persistence tool from registry.
    Phase 5+: Real CSV file generation with configurable output directory.
    Phase 6.4: Added telemetry for observability and performance tracking.

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
    with telemetry.timer("risk.export_csv"):
        telemetry.log_info(
            "risk_export_start",
            run_id=state.run_id or "unknown",
            table_rows=len(state.joined_table) if state.joined_table else 0,
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            if not state.joined_table:
                telemetry.log_warning("export_csv_skipped", reason="no_joined_table")
                return {"csv_path": None}

            # Generate filename using run_id with validation
            run_id = state.run_id if state.run_id else "unknown"
            # Sanitize run_id to prevent path traversal
            run_id_safe = "".join(c for c in run_id[:8] if c.isalnum() or c == "-")
            if not run_id_safe:
                telemetry.log_warning(
                    "invalid_run_id",
                    run_id=run_id,
                    using_default="unknown",
                )
                run_id_safe = "unknown"
            filename = f"risk_assessment_{run_id_safe}.csv"

            # Use persistence tool to save CSV with error handling
            try:
                with telemetry.timer("risk.persistence_tool"):
                    csv_path = registry.persistence.save_csv(
                        data=state.joined_table,
                        filename=filename,
                    )
                telemetry.log_info(
                    "risk_export_complete",
                    csv_path=str(csv_path) if csv_path else None,
                )
            except Exception as e:
                telemetry.log_error("csv_export_failed", error=e)
                # Fallback: None (CSV export failed)
                # If persistence fails, workflow continues but without CSV output
                # Orchestrator can handle this gracefully
                csv_path = None

            return {"csv_path": str(csv_path) if csv_path else None}

        except Exception as e:
            telemetry.log_error("risk_export_csv_failed", error=e)
            raise


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
