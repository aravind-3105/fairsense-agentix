"""Risk graph nodes: join rows, HTML table, CSV export."""

from fairsense_agentix.graphs.state import RiskState
from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools import get_tool_registry


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
