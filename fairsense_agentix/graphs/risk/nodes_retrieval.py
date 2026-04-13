"""Risk graph nodes: embedding and FAISS retrieval."""

from fairsense_agentix.graphs.state import RiskState
from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools import get_tool_registry


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
