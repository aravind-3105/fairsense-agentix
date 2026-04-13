"""Orchestrator node: delegate to workflow subgraphs."""

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
from fairsense_agentix.graphs.bias_image_vlm_graph import create_bias_image_vlm_graph
from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph
from fairsense_agentix.graphs.risk_graph import create_risk_graph
from fairsense_agentix.graphs.state import OrchestratorState
from fairsense_agentix.services import telemetry


def execute_workflow(state: OrchestratorState) -> dict:
    """Executor node: Delegate to appropriate workflow subgraph.

    Routes to bias_text, bias_image, or risk subgraphs based on plan.workflow_id.

    Checkpoint 2.4: bias_text uses real BiasTextGraph
    Checkpoint 2.5: bias_image will use real BiasImageGraph
    Checkpoint 2.6: risk will use real RiskGraph
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Current state with plan

    Returns
    -------
    dict
        State update with workflow_result

    Examples
    --------
    >>> state = OrchestratorState(input_type="text", content="test", plan=...)
    >>> update = execute_workflow(state)
    >>> update["workflow_result"]["workflow_id"]
    'bias_text'
    """
    with telemetry.timer(
        "orchestrator.execute_workflow",
        workflow_id=state.plan.workflow_id if state.plan else "none",
    ):
        telemetry.log_info(
            "orchestrator_execute_start",
            run_id=state.run_id or "unknown",
            workflow_id=state.plan.workflow_id if state.plan else None,
        )

        if state.plan is None:
            telemetry.log_error("execute_workflow_failed", reason="no_plan")
            return {
                "workflow_result": None,
                "errors": state.errors + ["Cannot execute: no plan available"],
            }

        workflow_id = state.plan.workflow_id

        try:
            if workflow_id == "bias_text":
                # Checkpoint 2.4: Real BiasTextGraph
                with telemetry.timer("orchestrator.bias_text_subgraph"):
                    text_graph = create_bias_text_graph()

                    # Extract text content
                    text = (
                        state.content
                        if isinstance(state.content, str)
                        else state.content.decode("utf-8", errors="ignore")
                    )

                    # Invoke subgraph with options from plan
                    subgraph_result = text_graph.invoke(
                        {
                            "text": text,
                            "options": state.options,
                            "run_id": state.run_id,  # Propagate run_id for tracing
                        },
                    )

                # Package result
                # Note: summary is optional (conditional node in BiasTextGraph)
                result = {
                    "workflow_id": "bias_text",
                    "bias_analysis": subgraph_result["bias_analysis"],
                    "summary": subgraph_result.get("summary"),  # Optional field
                    "highlighted_html": subgraph_result["highlighted_html"],
                }

            elif workflow_id == "bias_image":
                # Extract image bytes
                image_bytes = (
                    state.content
                    if isinstance(state.content, bytes)
                    else state.content.encode("utf-8")
                )

                # Route based on image_analysis_mode setting
                if settings.image_analysis_mode == "vlm":
                    # VLM-based analysis (fast, requires OpenAI or Anthropic)
                    # Validate provider supports VLM
                    if settings.llm_provider not in {"openai", "anthropic", "fake"}:
                        telemetry.log_error(
                            "vlm_mode_invalid_provider",
                            provider=settings.llm_provider,
                        )
                        raise ValueError(
                            f"VLM mode requires llm_provider to be "
                            f"'openai', 'anthropic', or 'fake'. "
                            f"Current provider: {settings.llm_provider}. "
                            f"Either change llm_provider or set "
                            f"image_analysis_mode='traditional'.",
                        )

                    with telemetry.timer("orchestrator.bias_image_vlm_subgraph"):
                        vlm_graph = create_bias_image_vlm_graph()

                        # Invoke VLM subgraph
                        subgraph_result = vlm_graph.invoke(
                            {
                                "image_bytes": image_bytes,
                                "options": state.options,
                                "run_id": state.run_id,
                            },
                        )

                    # Package result (VLM format)
                    result = {
                        "workflow_id": "bias_image_vlm",
                        "vlm_analysis": subgraph_result["vlm_analysis"],
                        "visual_description": subgraph_result[
                            "vlm_analysis"
                        ].visual_description,
                        "reasoning_trace": subgraph_result[
                            "vlm_analysis"
                        ].reasoning_trace,
                        "bias_analysis": subgraph_result["vlm_analysis"].bias_analysis,
                        "summary": subgraph_result.get("summary"),
                        "highlighted_html": subgraph_result["highlighted_html"],
                        "image_base64": subgraph_result.get("image_base64"),
                    }

                else:
                    # Traditional OCR + Caption analysis (slower, backward compatible)
                    with telemetry.timer("orchestrator.bias_image_subgraph"):
                        image_graph = create_bias_image_graph()

                        # Invoke traditional subgraph
                        subgraph_result = image_graph.invoke(
                            {
                                "image_bytes": image_bytes,
                                "options": state.options,
                                "run_id": state.run_id,
                            },
                        )

                    # Package result (traditional format)
                    result = {
                        "workflow_id": "bias_image",
                        "ocr_text": subgraph_result["ocr_text"],
                        "caption_text": subgraph_result["caption_text"],
                        "merged_text": subgraph_result["merged_text"],
                        "bias_analysis": subgraph_result["bias_analysis"],
                        "summary": subgraph_result.get("summary"),
                        "highlighted_html": subgraph_result["highlighted_html"],
                    }

            elif workflow_id == "bias_image_vlm":
                # VLM-based image bias analysis (direct route from router)
                # Extract image bytes
                image_bytes = (
                    state.content
                    if isinstance(state.content, bytes)
                    else state.content.encode("utf-8")
                )

                # Validate provider supports VLM
                if settings.llm_provider not in {"openai", "anthropic", "fake"}:
                    telemetry.log_error(
                        "vlm_mode_invalid_provider",
                        provider=settings.llm_provider,
                    )
                    raise ValueError(
                        f"VLM mode requires llm_provider to be "
                        f"'openai', 'anthropic', or 'fake'. "
                        f"Current provider: {settings.llm_provider}. "
                        f"Either change llm_provider or set "
                        f"image_analysis_mode='traditional'.",
                    )

                with telemetry.timer("orchestrator.bias_image_vlm_subgraph"):
                    vlm_graph = create_bias_image_vlm_graph()

                    # Invoke VLM subgraph
                    subgraph_result = vlm_graph.invoke(
                        {
                            "image_bytes": image_bytes,
                            "options": state.options,
                            "run_id": state.run_id,
                        },
                    )

                # Package result (VLM format)
                result = {
                    "workflow_id": "bias_image_vlm",
                    "vlm_analysis": subgraph_result["vlm_analysis"],
                    "visual_description": subgraph_result[
                        "vlm_analysis"
                    ].visual_description,
                    "reasoning_trace": subgraph_result["vlm_analysis"].reasoning_trace,
                    "bias_analysis": subgraph_result["vlm_analysis"].bias_analysis,
                    "summary": subgraph_result.get("summary"),
                    "highlighted_html": subgraph_result["highlighted_html"],
                    "image_base64": subgraph_result.get("image_base64"),
                }

            elif workflow_id == "risk":
                # Checkpoint 2.6: Real RiskGraph
                with telemetry.timer("orchestrator.risk_subgraph"):
                    risk_graph = create_risk_graph()

                    # Extract scenario text
                    scenario_text = (
                        state.content
                        if isinstance(state.content, str)
                        else state.content.decode("utf-8", errors="ignore")
                    )

                    # Invoke subgraph with options from plan
                    subgraph_result = risk_graph.invoke(
                        {
                            "scenario_text": scenario_text,
                            "options": state.options,
                            "run_id": state.run_id,  # Propagate run_id for tracing
                        },
                    )

                # Package result
                result = {
                    "workflow_id": "risk",
                    "embedding": subgraph_result["embedding"],
                    "risks": subgraph_result["risks"],
                    "rmf_recommendations": subgraph_result["rmf_recommendations"],
                    "joined_table": subgraph_result["joined_table"],
                    "html_table": subgraph_result["html_table"],
                    "csv_path": subgraph_result["csv_path"],
                }
            else:
                telemetry.log_error("unknown_workflow_id", workflow_id=workflow_id)
                raise ValueError(f"Unknown workflow_id: {workflow_id}")

            telemetry.log_info("orchestrator_execute_complete", workflow_id=workflow_id)
            return {"workflow_result": result}

        except Exception as e:
            telemetry.log_error(
                "orchestrator_execute_workflow_failed",
                workflow_id=workflow_id,
                error=e,
            )
            return {
                "workflow_result": None,
                "errors": state.errors + [f"Workflow execution failed: {e!s}"],
            }
