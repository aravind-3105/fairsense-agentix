"""Bias text graph nodes: LLM analysis, summary, HTML highlight."""

from pathlib import Path

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.bias_text.spans import _extract_spans_from_analysis
from fairsense_agentix.graphs.state import BiasTextState
from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools import get_tool_registry
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def analyze_bias(state: BiasTextState) -> dict:
    """LLM node: Analyze text for various forms of bias.

    Uses LLM to identify and explain bias in text across multiple dimensions:
    - Gender bias
    - Age bias
    - Racial/ethnic bias
    - Disability bias
    - Socioeconomic bias

    Phase 4+: Uses tool registry to get LLM implementation (fake or real).
    Phase 5+: Real LLM calls with structured prompts when real tools configured.
    Phase 6.1: Uses bias_analysis_v1.txt prompt template with structured output
               for precise character-level span extraction.
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasTextState
        Current state with text to analyze

    Returns
    -------
    dict
        State update with bias_analysis field

    Examples
    --------
    >>> state = BiasTextState(text="Job posting")
    >>> update = analyze_bias(state)
    >>> "bias_analysis" in update
    True
    """
    with telemetry.timer("bias_text.analyze_bias", text_length=len(state.text)):
        telemetry.log_info(
            "bias_text_analyze_start",
            text_length=len(state.text),
            run_id=state.run_id or "unknown",
        )

        try:
            # Get tools from registry (configuration-driven: fake or real)
            registry = get_tool_registry()

            # Extract options
            text = state.text
            temperature = state.options.get("temperature", 0.3)
            max_tokens = state.options.get("llm_max_tokens", 2000)

            # Validate parameters
            if not isinstance(temperature, (int, float)) or not (
                0.0 <= temperature <= 1.0
            ):
                telemetry.log_warning(
                    "invalid_temperature",
                    temperature=temperature,
                    using_default=0.3,
                )
                temperature = 0.3  # Use default for invalid values
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                telemetry.log_warning(
                    "invalid_max_tokens",
                    max_tokens=max_tokens,
                    using_default=2000,
                )
                max_tokens = 2000  # Use default for invalid values

            # Phase 6.1: Load prompt template for bias analysis
            # Template instructs LLM to return precise character positions
            prompt_template_path = (
                _PKG_ROOT / "prompts" / "templates" / "bias_analysis_v1.txt"
            )

            try:
                with open(prompt_template_path) as f:
                    prompt_template = f.read()
            except FileNotFoundError:
                telemetry.log_warning(
                    "prompt_template_not_found",
                    path=str(prompt_template_path),
                    using_fallback=True,
                )
                # Fallback to hardcoded prompt if template not found
                prompt_template = (
                    "You are a bias detection specialist analyzing text for "
                    "various forms of bias.\n\n"
                    "Your task is to carefully analyze the provided text and "
                    "identify ANY instances of bias across these categories:\n"
                    "- **Gender bias**: Gendered language, stereotypes, "
                    "exclusionary terms\n"
                    "- **Age bias**: Age discrimination, ageist assumptions, "
                    "generational stereotypes\n"
                    "- **Racial/ethnic bias**: Racial stereotypes, cultural "
                    "assumptions, exclusionary language\n"
                    "- **Disability bias**: Ableist language, accessibility "
                    "assumptions, capability stereotypes\n"
                    "- **Socioeconomic bias**: Class-based assumptions, "
                    "privilege bias, economic stereotypes\n\n"
                    "**TEXT TO ANALYZE:**\n"
                    "{text}\n\n"
                    "Return valid JSON with structure: "
                    '{{"bias_detected": bool, "bias_instances": [...], '
                    '"overall_assessment": str, "risk_level": str}}'
                )

            # Build prompt with text
            prompt = prompt_template.format(text=text)

            critique_feedback = state.options.get("bias_prompt_feedback")
            if isinstance(critique_feedback, list) and critique_feedback:
                feedback_section = "\n".join(f"- {item}" for item in critique_feedback)
                prompt += (
                    "\nEvaluator feedback to address:\n"
                    f"{feedback_section}\n"
                    "Ensure the refreshed analysis explicitly incorporates "
                    "the feedback above.\n"
                )

            # Call LLM via registry with structured output
            # Real LLMs use .with_structured_output() and return BiasAnalysisOutput
            # Fake LLM now returns JSON string that we parse into BiasAnalysisOutput
            with telemetry.timer("bias_text.llm_call", temperature=temperature):
                llm_output = registry.llm.predict(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Parse LLM output into BiasAnalysisOutput
            # Real LLM: already BiasAnalysisOutput object
            # Fake LLM: JSON string that needs parsing
            if isinstance(llm_output, BiasAnalysisOutput):
                # Already structured (real LLM with .with_structured_output())
                bias_analysis = llm_output
                telemetry.log_info("bias_analysis_structured", format="pydantic_object")
            elif isinstance(llm_output, str):
                # JSON string (fake LLM) - parse into structured object
                try:
                    bias_analysis = BiasAnalysisOutput.model_validate_json(llm_output)
                    telemetry.log_info(
                        "bias_analysis_parsed",
                        format="json_to_pydantic",
                    )
                except Exception as e:
                    telemetry.log_error(
                        "bias_analysis_parse_failed",
                        error=e,
                        output_preview=llm_output[:100],
                    )
                    raise ValueError(
                        f"Failed to parse LLM output into BiasAnalysisOutput: {e}",
                    ) from e
            else:
                # Unexpected type
                raise TypeError(
                    f"Unexpected LLM output type: {type(llm_output)}. "
                    "Expected BiasAnalysisOutput or JSON string.",
                )

            telemetry.log_info(
                "bias_text_analyze_complete",
                bias_detected=bias_analysis.bias_detected,
                instance_count=len(bias_analysis.bias_instances),
            )
            return {"bias_analysis": bias_analysis}

        except Exception as e:
            telemetry.log_error("bias_text_analyze_failed", error=e)
            raise


def summarize(state: BiasTextState) -> dict:
    """Summarization node: Generate concise summary of bias findings.

    Creates a short executive summary of the bias analysis, highlighting
    key findings and recommendations. This is useful for long texts or
    when users need quick insights.

    Conditional execution: Only runs if text is long (>500 chars) or if
    explicitly requested in options.

    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasTextState
        Current state with bias_analysis

    Returns
    -------
    dict
        State update with summary field

    Examples
    --------
    >>> state = BiasTextState(text="Long text...", bias_analysis="...")
    >>> update = summarize(state)
    >>> "summary" in update
    True
    """
    with telemetry.timer("bias_text.summarize"):
        telemetry.log_info(
            "bias_text_summarize_start",
            run_id=state.run_id or "unknown",
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            if state.bias_analysis is None:
                telemetry.log_warning("summarize_skipped", reason="no_bias_analysis")
                return {"summary": None}

            # Use summarizer tool to generate summary with error handling
            max_length = state.options.get("summary_max_length", 200)

            # Validate max_length parameter
            if not isinstance(max_length, int) or max_length <= 0:
                telemetry.log_warning(
                    "invalid_max_length",
                    max_length=max_length,
                    using_default=200,
                )
                max_length = 200  # Use default for invalid values

            try:
                # Convert BiasAnalysisOutput to string for summarization
                if isinstance(state.bias_analysis, BiasAnalysisOutput):
                    # Use overall_assessment as summary source
                    text_to_summarize = state.bias_analysis.overall_assessment
                else:
                    # Fallback (shouldn't happen)
                    text_to_summarize = str(state.bias_analysis)

                with telemetry.timer("bias_text.summarizer_tool"):
                    summary = registry.summarizer.summarize(
                        text=text_to_summarize,
                        max_length=max_length,
                    )
                telemetry.log_info(
                    "bias_text_summarize_complete",
                    summary_length=len(summary) if summary else 0,
                )
            except Exception as e:
                telemetry.log_error("summarization_failed", error=e)
                # Fallback: None (summary is already optional in the workflow)
                # If summarization fails, workflow continues without summary
                summary = None

            return {"summary": summary}

        except Exception as e:
            telemetry.log_error("bias_text_summarize_failed", error=e)
            raise


def highlight(state: BiasTextState) -> dict:
    """Highlighting node: Generate HTML with bias spans highlighted.

    Creates an HTML representation of the input text with problematic
    spans highlighted in different colors based on bias type:
    - Gender bias: Pink
    - Age bias: Orange
    - Racial bias: Yellow
    - Disability bias: Blue
    - Socioeconomic bias: Purple

    Phase 6.1: Extracts precise character positions from bias_analysis
               for accurate span highlighting.
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasTextState
        Current state with text and bias_analysis

    Returns
    -------
    dict
        State update with highlighted_html field

    Examples
    --------
    >>> state = BiasTextState(text="Sample text", bias_analysis="...")
    >>> update = highlight(state)
    >>> "<html>" in update["highlighted_html"]
    True
    """
    with telemetry.timer("bias_text.highlight"):
        telemetry.log_info(
            "bias_text_highlight_start",
            run_id=state.run_id or "unknown",
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            # Phase 6.1: Extract spans from bias analysis with character positions
            spans: list[tuple[int, int, str]] = []

            if state.bias_analysis:
                # Extract (start_char, end_char, bias_type) tuples from analysis
                with telemetry.timer("bias_text.extract_spans"):
                    spans = _extract_spans_from_analysis(
                        bias_analysis=state.bias_analysis,
                        original_text=state.text,
                    )
                telemetry.log_info("spans_extracted", span_count=len(spans))
            else:
                telemetry.log_warning(
                    "highlight_no_analysis",
                    reason="no_bias_analysis",
                )

            # Get bias type colors from configuration
            bias_types = settings.get_bias_type_colors()

            # Use formatter tool to generate highlighted HTML fragment
            # (dark-mode) with error handling
            try:
                with telemetry.timer("bias_text.formatter_tool"):
                    highlighted_html = registry.formatter.highlight_fragment(
                        text=state.text,
                        spans=spans,
                        bias_types=bias_types,
                    )
                telemetry.log_info(
                    "bias_text_highlight_complete",
                    html_length=len(highlighted_html),
                )
            except Exception as e:
                telemetry.log_error("highlighting_failed", error=e)
                # Fallback: Plain HTML with unformatted text
                # Formatting is presentation layer - if it fails, provide plain text
                # rather than killing the entire workflow
                highlighted_html = f"<pre>{state.text}</pre>"

            return {"highlighted_html": highlighted_html}

        except Exception as e:
            telemetry.log_error("bias_text_highlight_failed", error=e)
            raise
