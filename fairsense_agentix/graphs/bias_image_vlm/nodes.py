"""LangGraph node functions for the VLM image bias workflow.

Nodes
-----
visual_analyze
    Single VLM call with Chain-of-Thought prompt; populates ``vlm_analysis``.
summarize
    Condenses the overall assessment via the summarizer tool.
highlight
    Builds highlighted HTML from the visual description and bias instances.
"""

import base64
import logging
from pathlib import Path

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import BiasImageVLMState
from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools import get_tool_registry
from fairsense_agentix.tools.vlm.output_schemas import BiasVisualAnalysisOutput


logger = logging.getLogger(__name__)


def visual_analyze(state: BiasImageVLMState) -> dict:
    """VLM analysis node with Chain-of-Thought reasoning.

    Single API call to GPT-4o Vision or Claude Sonnet Vision (depending on
    llm_provider) with 4-step CoT prompt for systematic bias detection:
        1. Visual Inventory (Perceive)
        2. Pattern Analysis (Reason)
        3. Bias Detection (Evidence)
        4. Synthesis (Reflect)

    Parameters
    ----------
    state : BiasImageVLMState
        Current workflow state with image_bytes and options

    Returns
    -------
    dict
        Updated state with vlm_analysis populated

    Notes
    -----
    Provider routing is automatic based on llm_provider setting:
        - llm_provider="openai" → GPT-4o Vision
        - llm_provider="anthropic" → Claude Sonnet Vision
        - llm_provider="fake" → FakeVLMTool (testing)
    """
    with telemetry.timer(
        "bias_image_vlm.visual_analyze",
        image_size=len(state.image_bytes),
    ):
        telemetry.log_info(
            "bias_image_vlm_analyze_start",
            image_size=len(state.image_bytes),
            run_id=state.run_id or "unknown",
            provider=settings.llm_provider,
            model=settings.llm_model_name,
        )

        try:
            registry = get_tool_registry()

            prompt_path = (
                Path(__file__).parent.parent.parent
                / "prompts"
                / "templates"
                / "bias_visual_analysis_v1.txt"
            )

            with open(prompt_path, encoding="utf-8") as f:
                prompt_template = f.read()

            temperature = state.options.get("temperature", settings.llm_temperature)
            max_tokens = state.options.get("vlm_max_tokens", settings.llm_max_tokens)

            logger.info(
                f"VLM analyzing image: {len(state.image_bytes)} bytes "
                f"via {settings.llm_provider} ({settings.llm_model_name})",
            )

            with telemetry.timer("bias_image_vlm.vlm_api_call"):
                vlm_result = registry.vlm.analyze_image(
                    image_bytes=state.image_bytes,
                    prompt=prompt_template,
                    response_model=BiasVisualAnalysisOutput,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            assert isinstance(vlm_result, BiasVisualAnalysisOutput)

            logger.info(
                f"VLM analysis complete: "
                f"bias_detected={vlm_result.bias_analysis.bias_detected}, "
                f"instance_count={len(vlm_result.bias_analysis.bias_instances)}",
            )

            telemetry.log_info(
                "bias_image_vlm_analyze_complete",
                bias_detected=vlm_result.bias_analysis.bias_detected,
                instance_count=len(vlm_result.bias_analysis.bias_instances),
                visual_description_length=len(vlm_result.visual_description),
                reasoning_trace_length=len(vlm_result.reasoning_trace),
            )

            return {"vlm_analysis": vlm_result}

        except Exception as e:
            logger.error(f"VLM analysis failed: {e}", exc_info=True)
            telemetry.log_error(
                "bias_image_vlm_analyze_failed",
                error=e,
                error_type=type(e).__name__,
            )
            raise


def summarize(state: BiasImageVLMState) -> dict:
    """Summarization node (REUSES existing summarizer tool).

    Extracts bias_analysis from VLM output and generates condensed summary.

    Parameters
    ----------
    state : BiasImageVLMState
        Current workflow state with vlm_analysis

    Returns
    -------
    dict
        Updated state with summary populated

    Notes
    -----
    This node reuses the existing LLM-based summarizer tool, demonstrating
    code reuse across graphs. The summarizer uses the same llm_provider as VLM.
    """
    with telemetry.timer("bias_image_vlm.summarize"):
        telemetry.log_info(
            "bias_image_vlm_summarize_start",
            run_id=state.run_id or "unknown",
        )

        try:
            if state.vlm_analysis is None:
                logger.warning("Summarize skipped: no vlm_analysis available")
                telemetry.log_warning("summarize_skipped", reason="no_vlm_analysis")
                return {"summary": None}

            registry = get_tool_registry()

            bias_analysis = state.vlm_analysis.bias_analysis

            max_length = state.options.get(
                "summary_max_length",
                settings.summarizer_max_length,
            )

            try:
                summary = registry.summarizer.summarize(
                    text=bias_analysis.overall_assessment,
                    max_length=max_length,
                )

                logger.info(
                    f"Summary generated: {len(summary) if summary else 0} characters",
                )

                telemetry.log_info(
                    "bias_image_vlm_summarize_complete",
                    summary_length=len(summary) if summary else 0,
                )

            except Exception as e:
                logger.error(f"Summarization failed: {e}", exc_info=True)
                telemetry.log_error("summarization_failed", error=e)
                summary = None

            return {"summary": summary}

        except Exception as e:
            logger.error(f"Summarize node failed: {e}", exc_info=True)
            telemetry.log_error("bias_image_vlm_summarize_failed", error=e)
            raise


def highlight(state: BiasImageVLMState) -> dict:  # noqa: PLR0915
    """Highlighting node (ADAPTS existing formatter tool).

    For images, we create an HTML representation showing the visual description
    with bias type indicators and a detailed bias instances summary.

    Since we can't highlight pixels directly, we show:
        1. Visual description from VLM
        2. Legend of detected bias types
        3. Detailed list of bias instances with explanations

    Parameters
    ----------
    state : BiasImageVLMState
        Current workflow state with vlm_analysis

    Returns
    -------
    dict
        Updated state with highlighted_html populated

    Notes
    -----
    This node adapts the existing HTML formatter tool to work with visual bias.
    Unlike text highlighting (character-level spans), image highlighting shows
    structured bias findings with color-coded types.
    """
    with telemetry.timer("bias_image_vlm.highlight"):
        telemetry.log_info(
            "bias_image_vlm_highlight_start",
            run_id=state.run_id or "unknown",
        )

        try:
            registry = get_tool_registry()

            if state.vlm_analysis is None:
                logger.warning("Highlight skipped: no vlm_analysis available")
                telemetry.log_warning("highlight_skipped", reason="no_vlm_analysis")
                return {"highlighted_html": "<p>No analysis available</p>"}

            text_to_display = state.vlm_analysis.visual_description
            bias_analysis = state.vlm_analysis.bias_analysis
            bias_types = settings.get_bias_type_colors()

            # No character-level spans for images — legend only
            spans: list[tuple[int, int, str]] = []

            try:
                highlighted_html = registry.formatter.highlight_fragment(
                    text=text_to_display,
                    spans=spans,
                    bias_types=bias_types,
                )

                if bias_analysis.bias_detected and bias_analysis.bias_instances:
                    instances_html = (
                        "<div style='margin-top: 1.5rem; padding: 1rem; "
                        "background: rgba(255,255,255,0.05); "
                        "border-radius: 0.5rem; border-left: 3px solid #00b2b2;'>"
                    )
                    instances_html += (
                        "<h4 style='margin: 0 0 1rem 0; color: #f8fafc; "
                        "font-size: 1rem;'>🔍 Detected Bias Instances:</h4>"
                    )
                    instances_html += (
                        "<ul style='margin: 0; padding-left: 1.5rem; "
                        "list-style-type: none;'>"
                    )

                    for _i, instance in enumerate(bias_analysis.bias_instances, 1):
                        color = bias_types.get(instance.type, "#888888")
                        severity_emoji = {
                            "low": "⚠️",
                            "medium": "⚠️⚠️",
                            "high": "🚨",
                        }.get(instance.severity, "⚠️")

                        instances_html += (
                            "<li style='margin: 0.75rem 0; "
                            "padding: 0.75rem; background: rgba(0,0,0,0.2); "
                            "border-radius: 0.375rem;'>"
                        )
                        instances_html += (
                            "<div style='display: flex; align-items: center; "
                            "margin-bottom: 0.5rem;'>"
                        )
                        instances_html += (
                            f"<span style='color: {color}; font-weight: bold; "
                            f"font-size: 0.875rem; text-transform: uppercase; "
                            f"margin-right: 0.5rem;'>{instance.type}</span>"
                        )
                        instances_html += (
                            "<span style='color: #cbd5e1; font-size: 0.75rem; "
                            "margin-right: 0.5rem;'>|</span>"
                        )
                        instances_html += (
                            f"<span style='color: #94a3b8; font-size: 0.75rem; "
                            f"text-transform: capitalize;'>"
                            f"{severity_emoji} {instance.severity} severity</span>"
                        )
                        instances_html += "</div>"

                        evidence = instance.visual_element or instance.text_span
                        instances_html += (
                            f"<div style='color: #e2e8f0; font-size: 0.875rem; "
                            f"line-height: 1.5; margin-bottom: 0.5rem;'>"
                            f"<strong>Evidence:</strong> {evidence}</div>"
                        )

                        instances_html += (
                            f"<div style='color: #cbd5e1; font-size: 0.8125rem; "
                            f"line-height: 1.5; font-style: italic;'>"
                            f"{instance.explanation}</div>"
                        )

                        instances_html += "</li>"

                    instances_html += "</ul>"

                    instances_html += (
                        "<div style='margin-top: 1rem; padding-top: 1rem; "
                        "border-top: 1px solid rgba(255,255,255,0.1);'>"
                    )
                    instances_html += (
                        "<div style='color: #94a3b8; font-size: 0.75rem; "
                        "text-transform: uppercase; letter-spacing: 0.05em; "
                        "margin-bottom: 0.5rem;'>Overall Assessment</div>"
                    )
                    instances_html += (
                        f"<div style='color: #f1f5f9; font-size: 0.875rem; "
                        f"line-height: 1.6;'>{bias_analysis.overall_assessment}</div>"
                    )

                    risk_colors = {
                        "low": "#22c55e",
                        "medium": "#f59e0b",
                        "high": "#ef4444",
                    }
                    risk_color = risk_colors.get(bias_analysis.risk_level, "#6b7280")
                    instances_html += (
                        f"<div style='margin-top: 0.75rem;'>"
                        f"<span style='display: inline-block; "
                        f"padding: 0.25rem 0.75rem; "
                        f"background: {risk_color}; color: white; "
                        f"border-radius: 9999px; "
                        f"font-size: 0.75rem; font-weight: 600; "
                        f"text-transform: uppercase;'>"
                        f"Risk: {bias_analysis.risk_level}</span>"
                        f"</div>"
                    )

                    instances_html += "</div>"
                    instances_html += "</div>"

                    highlighted_html += instances_html

                logger.info(
                    f"Highlighted HTML generated: {len(highlighted_html)} characters",
                )

                telemetry.log_info(
                    "bias_image_vlm_highlight_complete",
                    html_length=len(highlighted_html),
                    instances_shown=len(bias_analysis.bias_instances),
                )

            except Exception as e:
                logger.error(f"Highlighting failed: {e}", exc_info=True)
                telemetry.log_error("highlighting_failed", error=e)
                highlighted_html = f"<pre>{text_to_display}</pre>"

            # Encode image for UI display
            try:
                from fairsense_agentix.tools.vlm.unified_vlm_tool import (
                    _detect_image_format,
                )

                mime_type = _detect_image_format(state.image_bytes)
                image_b64 = base64.b64encode(state.image_bytes).decode("utf-8")
                image_base64 = f"data:{mime_type};base64,{image_b64}"

                logger.info(
                    f"Image encoded for UI: {len(state.image_bytes)} bytes "
                    f"({mime_type})",
                )

            except Exception as e:
                logger.error(f"Image encoding failed: {e}", exc_info=True)
                telemetry.log_error("image_encoding_failed", error=e)
                image_base64 = None

            return {"highlighted_html": highlighted_html, "image_base64": image_base64}

        except Exception as e:
            logger.error(f"Highlight node failed: {e}", exc_info=True)
            telemetry.log_error("bias_image_vlm_highlight_failed", error=e)
            raise
